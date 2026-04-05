"""
Training Loop for Conditional Flow Matching (Variant 1).
- Extract x1 and Condition from Dataloader
- Generate x0 (Noise)
- OT Flow Matcher extracts t, xt, ut
- DiT model learns MSE Loss between predicted velocity (vt_pred) and target vector field (ut).
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import csv

# Integrate original torchcfm library
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root / "conditional-flow-matching"))
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher

# Custom imports
from src.data.datamodule import SubjectDataModule
from src.models import MultimodalFusionEncoder
from src.models.dit import DiTConditional


class FlowMatchingTrainer:
    def __init__(self, subject: str, config_path: str = "src/configs/configs.yml", 
                 fast_dev_run: bool = False):
        self.subject = subject
        self.fast_dev_run = fast_dev_run
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_data()
        self._setup_models()
        self._setup_flow_matcher()
        self._setup_optimizer()
        
        # Create save dir in advance
        self.save_dir = Path(self.config["training"]["save_dir"]) / self.subject
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.save_dir / "history.csv"
    def _setup_data(self):
        logger.info(f"Setting up dataset for {self.subject}...")
        self.datamodule = SubjectDataModule(
            subject=self.subject, 
            config_path="src/configs/configs.yml"
        )
        self.datamodule.setup()
        self.train_dl = self.datamodule.train_dataloader()
        self.val_dl = self.datamodule.val_dataloader()
        
    def _setup_models(self):
        logger.info("Initializing Fusion Encoder and DiT Models...")
        enc_cfg = self.config["fusion_encoder"]
        self.fusion_encoder = MultimodalFusionEncoder(
            input_dim=22144,
            d_model=enc_cfg["d_model"],
            num_latents=enc_cfg["num_latents"],
            num_layers=enc_cfg["num_layers"],
            nhead=enc_cfg["nhead"],
            dropout=enc_cfg["dropout"]
        ).to(self.device)
        
        dit_cfg = self.config["velocity_net"]
        self.dit = DiTConditional(
            in_features=1000,
            patch_size=dit_cfg["patch_size"],
            hidden_size=dit_cfg["d_model"],
            depth=dit_cfg["depth"],
            num_heads=dit_cfg["nhead"],
            context_dim=dit_cfg["context_dim"],
            dropout=dit_cfg["dropout"]
        ).to(self.device)
        
    def _setup_flow_matcher(self):
        sigma = self.config["flow_matching"].get("sigma", 0.0)
        self.flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        self.criterion = nn.MSELoss()
        
    def _setup_optimizer(self):
        train_cfg = self.config["training"]
        self.epochs = train_cfg["epochs"]
        
        # Optimize both models simultaneously
        params = list(self.fusion_encoder.parameters()) + list(self.dit.parameters())
        self.optimizer = AdamW(
            params, 
            lr=float(train_cfg["learning_rate"]), 
            weight_decay=float(train_cfg["weight_decay"])
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)

    def train_step(self, batch):
        self.optimizer.zero_grad()
        
        # 1. Load Data
        x1 = batch["x1"].to(self.device)                   # (B, 1000)
        condition = batch["condition"].to(self.device)     # (B, 31, 22144)
        
        # 2. Noise Generation
        x0 = torch.randn_like(x1) # (B, 1000)
        
        # 3. Sample from OT Flow Matching to get t, xt, and target vector field ut
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x1)
        
        # 4. Predict Flow
        # Forward condition through Fusion Encoder to get `C` tensor (B, 8, 1024)
        C = self.fusion_encoder(condition)
        
        # Pass to Velocity network to predict ut (vt_pred)
        vt_pred = self.dit(xt, t, C)
        
        # 5. MSE Loss Backpropagation
        loss = self.criterion(vt_pred, ut)
        loss.backward()
        
        # Gradient Clipping to prevent exploding gradients.
        torch.nn.utils.clip_grad_norm_(self.fusion_encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.dit.parameters(), 1.0)
        
        self.optimizer.step()
        
        return loss.item()

    def train(self):
        log_freq = self.config["training"].get("log_freq", 50)
        
        # Init CSV
        if not self.fast_dev_run and not self.history_file.exists():
            with open(self.history_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss", "val_pcc", "lr"])
        
        log_freq = self.config["training"].get("log_freq", 50)
        
        logger.info(f"Starting training on device: {self.device}")
        
        for epoch in range(1, self.epochs + 1):
            self.fusion_encoder.train()
            self.dit.train()
            
            pbar = tqdm(self.train_dl, desc=f"Epoch {epoch}/{self.epochs}")
            epoch_loss = 0.0
            
            for step, batch in enumerate(pbar):
                loss = self.train_step(batch)
                epoch_loss += loss
                
                # Log Progress
                if step % log_freq == 0:
                    pbar.set_postfix({"Loss": f"{loss:.4f}"})
                
                if self.fast_dev_run and step >= 5: # Fast run a few steps for debugging
                    break
                    
            avg_train_loss = epoch_loss / len(self.train_dl)
            
            # Validation every 5 epochs
            val_every = self.config["training"].get("val_every", 5)
            if epoch % val_every == 0 or epoch == self.epochs:
                val_loss, val_pcc = self.validate()
                
                logger.info(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PCC: {val_pcc:.4f}")
                
                # Log to CSV only after validation
                if not self.fast_dev_run:
                    with open(self.history_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch, avg_train_loss, val_loss, val_pcc, self.scheduler.get_last_lr()[0]])
            else:
                logger.info(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val (Skipped)")
            
            self.scheduler.step()
            
            # Checkpointing
            if not self.fast_dev_run and epoch % 10 == 0:
                torch.save({
                    "epoch": epoch,
                    "fusion_encoder": self.fusion_encoder.state_dict(),
                    "dit": self.dit.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "val_loss": val_loss
                }, self.save_dir / f"checkpoint_epoch_{epoch}.pt")
                
            if self.fast_dev_run:
                break
                
        logger.info("Training Complete!")

    @torch.no_grad()
    def validate(self):
        self.fusion_encoder.eval()
        self.dit.eval()
        
        val_loss = 0.0
        num_batches = 0
        
        all_x1 = []
        all_pred = []
        
        for step, batch in enumerate(self.val_dl):
            x1 = batch["x1"].to(self.device)
            condition = batch["condition"].to(self.device)
            x0 = torch.randn_like(x1)
            
            # 1. Validation Loss on flow matching
            t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x1)
            C = self.fusion_encoder(condition)
            vt_pred = self.dit(xt, t, C)
            
            loss = self.criterion(vt_pred, ut)
            val_loss += loss.item()
            
            # 2. ODE Solver (Euler method) for Generation & PCC computation
            # We predict x1_pred from x0 noise by integrating vt_pred
            x_pred = x0.clone()
            steps = 10  # 10 steps is usually enough for CFM validation
            dt = 1.0 / steps
            for st in range(steps):
                t_val = torch.full((x0.size(0),), st * dt, device=self.device)
                v_pred = self.dit(x_pred, t_val, C)
                x_pred = x_pred + v_pred * dt
                
            all_x1.append(x1.cpu())
            all_pred.append(x_pred.cpu())
            
            num_batches += 1
            if self.fast_dev_run and step >= 2:
                break
                
        # 3. Calculate Temporal PCC (per voxel) matching baseline logic
        all_x1 = torch.cat(all_x1, dim=0)     # Shape: (Total_TRs, 1000)
        all_pred = torch.cat(all_pred, dim=0) # Shape: (Total_TRs, 1000)
        
        x_mean = all_pred.mean(dim=0, keepdim=True)
        y_mean = all_x1.mean(dim=0, keepdim=True)
        x_c = all_pred - x_mean
        y_c = all_x1 - y_mean
        
        cov = (x_c * y_c).sum(dim=0)
        var_x = (x_c**2).sum(dim=0)
        var_y = (y_c**2).sum(dim=0)
        pcc = cov / torch.sqrt(var_x * var_y + 1e-8)
        val_pcc = pcc.mean().item()
        
        return val_loss / num_batches, val_pcc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CFM Model for specific Subject")
    parser.add_argument("--subject", type=str, default="sub-01", help="Subject ID (e.g. sub-01)")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run 5 steps for debugging")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B Logging")
    args = parser.parse_args()
    
    trainer = FlowMatchingTrainer(
        subject=args.subject, 
        fast_dev_run=args.fast_dev_run
    )
    trainer.train()
