"""
Training Loop for Conditional Flow Matching (v3 — U-ViT).
Changes vs v2:
  - U-ViT 1D replaces DiT as velocity network (long skip connections)
  - HierarchicalFusionEncoder replaces flat FusionEncoder (multi-level Q-Former)
  - EMA + CFG retained from v2
  - python -m src.train --subject sub-01
"""

import gc
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import csv

# Integrate original torchcfm library
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root / "conditional-flow-matching"))
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher

# Custom imports
from src.data.datamodule import SubjectDataModule
from src.models import HierarchicalFusionEncoder
from src.models.uvit1d import UViT1D
from src.utils.ema import EMAModel


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
        self._setup_ema()
        
        # CFG config
        cfg_config = self.config.get("cfg", {})
        self.cfg_enabled = cfg_config.get("enabled", False)
        self.cond_drop_prob = cfg_config.get("cond_drop_prob", 0.15)
        self.guidance_scale = cfg_config.get("guidance_scale", 2.0)
        
        if self.cfg_enabled:
            logger.info(f"CFG enabled: drop_prob={self.cond_drop_prob}, guidance_scale={self.guidance_scale}")
        
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
        logger.info("Initializing Hierarchical Fusion Encoder and U-ViT 1D...")
        enc_cfg = self.config["fusion_encoder"]
        self.fusion_encoder = HierarchicalFusionEncoder(
            input_dim=22144,
            d_model=enc_cfg["d_model"],
            num_levels=enc_cfg["num_levels"],
            queries_per_level=enc_cfg["queries_per_level"],
            bottleneck_queries=enc_cfg["bottleneck_queries"],
            num_temporal_layers=enc_cfg["num_temporal_layers"],
            num_qformer_layers=enc_cfg["num_qformer_layers"],
            nhead=enc_cfg["nhead"],
            dropout=enc_cfg["dropout"],
        ).to(self.device)
        
        # Compute total condition tokens for U-ViT
        num_cond_tokens = (
            enc_cfg["num_levels"] * enc_cfg["queries_per_level"]
            + enc_cfg["bottleneck_queries"]
        )
        
        uvit_cfg = self.config["uvit"]
        self.uvit = UViT1D(
            in_features=1000,
            patch_size=uvit_cfg["patch_size"],
            embed_dim=uvit_cfg["embed_dim"],
            depth=uvit_cfg["depth"],
            num_heads=uvit_cfg["num_heads"],
            mlp_ratio=uvit_cfg["mlp_ratio"],
            qkv_bias=uvit_cfg.get("qkv_bias", True),
            drop_rate=uvit_cfg["drop_rate"],
            num_cond_tokens=num_cond_tokens,
            skip=uvit_cfg.get("skip", True),
            use_checkpoint=uvit_cfg.get("use_checkpoint", False),
        ).to(self.device)

        # Log parameter counts
        enc_params = sum(p.numel() for p in self.fusion_encoder.parameters())
        uvit_params = sum(p.numel() for p in self.uvit.parameters())
        logger.info(f"  Fusion Encoder: {enc_params/1e6:.1f}M params")
        logger.info(f"  U-ViT 1D:      {uvit_params/1e6:.1f}M params")
        logger.info(f"  Total:          {(enc_params + uvit_params)/1e6:.1f}M params")
        logger.info(f"  Condition tokens: {num_cond_tokens}")
        
    def _setup_flow_matcher(self):
        sigma = self.config["flow_matching"].get("sigma", 0.0)
        self.flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        self.criterion = nn.MSELoss()
        
    def _setup_optimizer(self):
        train_cfg = self.config["training"]
        self.epochs = train_cfg["epochs"]
        
        # Optimize both models simultaneously
        params = list(self.fusion_encoder.parameters()) + list(self.uvit.parameters())
        self.optimizer = AdamW(
            params, 
            lr=float(train_cfg["learning_rate"]), 
            weight_decay=float(train_cfg["weight_decay"])
        )
        # Warmup for first 5 epochs then cosine decay
        warmup_epochs = 5
        self.warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        self.cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.epochs - warmup_epochs
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[self.warmup_scheduler, self.cosine_scheduler],
            milestones=[warmup_epochs]
        )

    def _setup_ema(self):
        """Initialize EMA if enabled in config."""
        ema_cfg = self.config.get("ema", {})
        self.ema_enabled = ema_cfg.get("enabled", False)
        
        if self.ema_enabled:
            self.ema = EMAModel(
                models=[self.fusion_encoder, self.uvit],
                decay=ema_cfg.get("decay", 0.999),
                warmup_steps=ema_cfg.get("warmup_steps", 1000),
            )
            logger.info(f"EMA enabled: decay={ema_cfg.get('decay', 0.999)}, warmup={ema_cfg.get('warmup_steps', 1000)} steps")
        else:
            self.ema = None

    def _apply_cfg_dropout(self, C: torch.Tensor) -> torch.Tensor:
        """Randomly drop condition tokens for CFG training.
        
        With probability `cond_drop_prob`, replace the entire condition
        tensor with zeros for a sample in the batch. This teaches the model
        to generate unconditionally (null condition = zeros).
        
        Args:
            C: Condition tensor from Fusion Encoder, shape (B, num_cond_tokens, d_model)
            
        Returns:
            C with some samples zeroed out
        """
        if not self.cfg_enabled or not self.fusion_encoder.training:
            return C
        
        B = C.size(0)
        # Create mask: 1 = keep condition, 0 = drop condition
        drop_mask = torch.rand(B, device=C.device) > self.cond_drop_prob  # (B,)
        drop_mask = drop_mask.float().unsqueeze(1).unsqueeze(2)  # (B, 1, 1) for broadcasting
        
        return C * drop_mask

    def train_step(self, batch):
        self.optimizer.zero_grad()
        
        # 1. Load Data
        x1 = batch["x1"].to(self.device)                   # (B, 1000)
        condition = batch["condition"].to(self.device)     # (B, 31, 22144)
        
        # 2. Noise Generation
        x0 = torch.randn_like(x1) # (B, 1000)
        
        # 3. Predict Condition (hierarchical multi-level tokens)
        C = self.fusion_encoder(condition)  # (B, 18, 768)
        
        # 4. CFG: Randomly drop condition for some samples
        C = self._apply_cfg_dropout(C)

        # 5. Sample from OT Flow Matching to get t, xt, and target vector field ut
        # C gets shuffled together with x1 according to OT plan
        t, xt, ut, _, C_shuffled = self.flow_matcher.guided_sample_location_and_conditional_flow(
            x0, x1, y1=C
        )
        
        # 6. Predict velocity using U-ViT
        vt_pred = self.uvit(xt, t, C_shuffled)
        
        # 7. MSE Loss Backpropagation
        loss = self.criterion(vt_pred, ut)
        loss.backward()
        
        # Gradient Clipping to prevent exploding gradients.
        enc_grad_norm = torch.nn.utils.clip_grad_norm_(self.fusion_encoder.parameters(), 1.0)
        uvit_grad_norm = torch.nn.utils.clip_grad_norm_(self.uvit.parameters(), 1.0)
        
        self.optimizer.step()
        
        # 8. EMA update (after optimizer step)
        if self.ema is not None:
            self.ema.update()
        
        return loss.item(), enc_grad_norm.item(), uvit_grad_norm.item()

    def train(self):
        log_freq = self.config["training"].get("log_freq", 50)
        
        # Init CSV (always overwrite to ensure correct header)
        if not self.fast_dev_run:
            with open(self.history_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss", "val_pcc", "lr"])
        
        logger.info(f"Starting training on device: {self.device}")
        
        best_pcc = -1.0
        
        for epoch in range(1, self.epochs + 1):
            self.fusion_encoder.train()
            self.uvit.train()
            
            pbar = tqdm(self.train_dl, desc=f"Epoch {epoch}/{self.epochs}")
            epoch_loss = 0.0
            
            for step, batch in enumerate(pbar):
                loss, enc_gn, uvit_gn = self.train_step(batch)
                epoch_loss += loss
                
                # Log Progress with gradient norms
                if step % log_freq == 0:
                    pbar.set_postfix({
                        "Loss": f"{loss:.4f}",
                        "EncGN": f"{enc_gn:.2f}",
                        "UViTGN": f"{uvit_gn:.2f}",
                        "LR": f"{self.scheduler.get_last_lr()[0]:.2e}"
                    })
                
                if self.fast_dev_run and step >= 5: # Fast run a few steps for debugging
                    break
                    
            avg_train_loss = epoch_loss / len(self.train_dl)
            
            # Validation every N epochs
            val_every = self.config["training"].get("val_every", 5)
            if epoch % val_every == 0 or epoch == self.epochs:
                val_loss, val_pcc = self.validate()
                
                # Free validation memory immediately
                gc.collect()
                torch.cuda.empty_cache()
                
                logger.info(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PCC: {val_pcc:.4f}")
                
                # Log to CSV only after validation
                if not self.fast_dev_run:
                    with open(self.history_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch, avg_train_loss, val_loss, val_pcc, self.scheduler.get_last_lr()[0]])
                
                # Save best model (only after validation)
                if not self.fast_dev_run and val_pcc > best_pcc:
                    best_pcc = val_pcc
                    self._save_checkpoint(epoch, val_loss, val_pcc, tag="best")
                    logger.info(f"  ★ New best PCC: {val_pcc:.4f}")
            else:
                logger.info(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val (Skipped)")
            
            self.scheduler.step()
            
            # Save latest checkpoint every 10 epochs (reduces I/O and memory pressure)
            if not self.fast_dev_run and epoch % 10 == 0:
                self._save_checkpoint(epoch, avg_train_loss, best_pcc, tag="latest")
                
            if self.fast_dev_run:
                break
                
        logger.info("Training Complete!")

    def _save_checkpoint(self, epoch, val_loss, val_pcc, tag="latest"):
        """Save checkpoint including EMA state."""
        ckpt = {
            "epoch": epoch,
            "fusion_encoder": self.fusion_encoder.state_dict(),
            "uvit": self.uvit.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "val_pcc": val_pcc,
        }
        if self.ema is not None:
            ckpt["ema"] = self.ema.state_dict()
        
        torch.save(ckpt, self.save_dir / f"checkpoint_{tag}.pt")

    @torch.no_grad()
    def validate(self):
        """Validate using EMA weights (if enabled) and CFG inference."""
        self.fusion_encoder.eval()
        self.uvit.eval()
        
        # Use EMA weights for validation if available
        ema_ctx = self.ema.apply() if self.ema is not None else _nullcontext()
        
        with ema_ctx:
            val_loss, val_pcc = self._run_validation()
        
        return val_loss, val_pcc
    
    def _run_validation(self):
        """Core validation logic, separated for EMA context usage."""
        val_loss = 0.0
        num_batches = 0
        
        all_x1 = []
        all_pred = []
        
        for step, batch in enumerate(self.val_dl):
            x1 = batch["x1"].to(self.device)
            condition = batch["condition"].to(self.device)
            x0 = torch.randn_like(x1)
            
            # 1. Validation Loss on flow matching (no CFG dropout)
            C = self.fusion_encoder(condition)
            t, xt, ut, _, C_shuffled = self.flow_matcher.guided_sample_location_and_conditional_flow(
                x0, x1, y1=C
            )
            vt_pred = self.uvit(xt, t, C_shuffled)
            
            loss = self.criterion(vt_pred, ut)
            val_loss += loss.item()
            
            # 2. ODE Solver with CFG for Generation & PCC computation
            x_pred = x0.clone()
            steps = 10
            dt = 1.0 / steps
            
            for st in range(steps):
                t_val = torch.full((x0.size(0),), st * dt, device=self.device)
                
                if self.cfg_enabled:
                    # CFG inference: v_guided = v_uncond + w * (v_cond - v_uncond)
                    # Sequential passes to avoid doubling batch size (OOM-safe)
                    null_C = torch.zeros_like(C)
                    v_uncond = self.uvit(x_pred, t_val, null_C)
                    v_cond = self.uvit(x_pred, t_val, C)
                    
                    v_guided = v_uncond + self.guidance_scale * (v_cond - v_uncond)
                    x_pred = x_pred + v_guided * dt
                else:
                    v_pred = self.uvit(x_pred, t_val, C)
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


class _nullcontext:
    """Minimal null context manager for Python < 3.10 compatibility."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


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
