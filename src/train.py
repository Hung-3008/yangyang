"""
Training Loop for Conditional Flow Matching (v6 — SimpleFusion + DiT-style U-ViT).
Changes vs v5:
  - SimpleFusionEncoder replaces Q-Former (3-layer Transformer, outputs dual signals)
  - U-ViT now uses AdaLN-Zero + Cross-Attention (no more token concatenation)
  - CFG dropout zeros both context_tokens AND global_cond
  - Bug fixes: seed, scheduler train mode, avg_train_loss
  - python -m src.train --subject sub-01
"""

import gc
import os
import sys
import argparse
import yaml
import time
import random
import numpy as np
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
from src.models import SimpleFusionEncoder, SchedulerNetwork
from src.models.uvit1d import UViT1D
from src.utils.ema import EMAModel


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FlowMatchingTrainer:
    def __init__(self, subject: str, config_path: str = "src/configs/configs.yml", 
                 fast_dev_run: bool = False):
        self.subject = subject
        self.fast_dev_run = fast_dev_run
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        # Set seed for reproducibility
        seed = self.config["training"].get("seed", 42)
        set_seed(seed)
        logger.info(f"Random seed set to {seed}")
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # CFG config
        cfg_config = self.config.get("cfg", {})
        self.cfg_enabled = cfg_config.get("enabled", False)
        self.cond_drop_prob = cfg_config.get("cond_drop_prob", 0.15)
        self.guidance_scale = cfg_config.get("guidance_scale", 2.0)
        
        if self.cfg_enabled:
            logger.info(f"CFG enabled: drop_prob={self.cond_drop_prob}, guidance_scale={self.guidance_scale}")
        
        self._setup_data()
        self._setup_models()
        self._setup_flow_matcher()
        self._setup_optimizer()
        self._setup_ema()
        
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
        logger.info("Initializing SimpleFusionEncoder and U-ViT 1D v2...")
        enc_cfg = self.config["fusion_encoder"]
        
        # Extract per-modality dimensions from config
        modality_names = list(self.config["modalities"].keys())
        modality_dims = [cfg["dim"] for cfg in self.config["modalities"].values()]
        total_input_dim = sum(modality_dims)
        logger.info(f"  {len(modality_dims)} modalities: {modality_names}")
        logger.info(f"  Dims: {modality_dims} (total={total_input_dim})")
        
        self.fusion_encoder = SimpleFusionEncoder(
            modality_dims=modality_dims,
            d_model=enc_cfg["d_model"],
            num_layers=enc_cfg["num_layers"],
            num_heads=enc_cfg["num_heads"],
            dropout=enc_cfg["dropout"],
            modality_drop_prob=enc_cfg.get("modality_drop_prob", 0.1),
        ).to(self.device)
        
        uvit_cfg = self.config["uvit"]
        self.uvit = UViT1D(
            in_features=self.config["data"]["fmri_dim"],
            patch_size=uvit_cfg["patch_size"],
            embed_dim=uvit_cfg["embed_dim"],
            depth=uvit_cfg["depth"],
            num_heads=uvit_cfg["num_heads"],
            cross_attn_heads=uvit_cfg.get("cross_attn_heads", 8),
            context_dim=enc_cfg["d_model"],  # Must match encoder d_model
            mlp_ratio=uvit_cfg["mlp_ratio"],
            qkv_bias=uvit_cfg.get("qkv_bias", True),
            drop_rate=uvit_cfg["drop_rate"],
            skip=uvit_cfg.get("skip", True),
            use_checkpoint=uvit_cfg.get("use_checkpoint", False),
        ).to(self.device)

        # Dynamic Interpolant Scheduler
        sched_cfg = self.config.get("scheduler", {})
        self.scheduler_enabled = sched_cfg.get("enabled", False)
        if self.scheduler_enabled:
            self.interp_scheduler = SchedulerNetwork(
                d_cond=enc_cfg["d_model"],
                d_hidden=sched_cfg.get("d_hidden", 256),
            ).to(self.device)
            logger.info("  Dynamic Interpolant Scheduler: enabled")
        else:
            self.interp_scheduler = None

        # Log parameter counts
        enc_params = sum(p.numel() for p in self.fusion_encoder.parameters())
        uvit_params = sum(p.numel() for p in self.uvit.parameters())
        sched_params = sum(p.numel() for p in self.interp_scheduler.parameters()) if self.interp_scheduler else 0
        logger.info(f"  SimpleFusionEncoder: {enc_params/1e6:.1f}M params")
        logger.info(f"  U-ViT 1D v2:        {uvit_params/1e6:.1f}M params")
        if sched_params > 0:
            logger.info(f"  Scheduler:          {sched_params/1e6:.1f}M params")
        logger.info(f"  Total:              {(enc_params + uvit_params + sched_params)/1e6:.1f}M params")
        
    def _setup_flow_matcher(self):
        sigma = self.config["flow_matching"].get("sigma", 0.0)
        self.flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        self.criterion = nn.MSELoss()
        
    def _setup_optimizer(self):
        train_cfg = self.config["training"]
        self.epochs = train_cfg["epochs"]
        
        # Optimize all models simultaneously
        params = list(self.fusion_encoder.parameters()) + list(self.uvit.parameters())
        if self.interp_scheduler is not None:
            params += list(self.interp_scheduler.parameters())
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
        self.lr_scheduler = SequentialLR(
            self.optimizer,
            schedulers=[self.warmup_scheduler, self.cosine_scheduler],
            milestones=[warmup_epochs]
        )

    def _setup_ema(self):
        """Initialize EMA if enabled in config."""
        ema_cfg = self.config.get("ema", {})
        self.ema_enabled = ema_cfg.get("enabled", False)
        
        if self.ema_enabled:
            ema_models = [self.fusion_encoder, self.uvit]
            if self.interp_scheduler is not None:
                ema_models.append(self.interp_scheduler)
            self.ema = EMAModel(
                models=ema_models,
                decay=ema_cfg.get("decay", 0.999),
                warmup_steps=ema_cfg.get("warmup_steps", 1000),
            )
            logger.info(f"EMA enabled: decay={ema_cfg.get('decay', 0.999)}, warmup={ema_cfg.get('warmup_steps', 1000)} steps")
        else:
            self.ema = None

    def _apply_cfg_dropout(
        self, context_tokens: torch.Tensor, global_cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomly drop condition for CFG training.
        
        With probability `cond_drop_prob`, zero out BOTH context_tokens
        and global_cond for a sample in the batch. This teaches the model
        to generate unconditionally (null condition = zeros).
        
        Args:
            context_tokens: (B, T, d_model) from Fusion Encoder
            global_cond:    (B, d_model) from Fusion Encoder
            
        Returns:
            Masked context_tokens and global_cond
        """
        if not self.cfg_enabled or not self.fusion_encoder.training:
            return context_tokens, global_cond
        
        B = context_tokens.size(0)
        # Create mask: 1 = keep condition, 0 = drop condition
        drop_mask = torch.rand(B, device=context_tokens.device) > self.cond_drop_prob
        
        # Mask for context_tokens: (B, 1, 1) for broadcasting over (B, T, D)
        ctx_mask = drop_mask.float().unsqueeze(1).unsqueeze(2)
        # Mask for global_cond: (B, 1) for broadcasting over (B, D)
        glob_mask = drop_mask.float().unsqueeze(1)
        
        return context_tokens * ctx_mask, global_cond * glob_mask

    def train_step(self, batch):
        self.optimizer.zero_grad()
        
        # 1. Load Data
        x1 = batch["x1"].to(self.device)                   # (B, 1000)
        condition = batch["condition"].to(self.device)     # (B, 31, 28032)
        
        # 2. Noise Generation
        x0 = torch.randn_like(x1) # (B, 1000)
        
        # 3. Encode condition → context_tokens (B, 31, 768) + global_cond (B, 768)
        context_tokens, global_cond = self.fusion_encoder(condition)
        
        # 4. CFG: Randomly drop condition for some samples
        context_tokens, global_cond = self._apply_cfg_dropout(context_tokens, global_cond)

        # 5. OT coupling (pair x0,x1 optimally) + shuffle condition
        #    We need to shuffle BOTH context_tokens and global_cond
        #    Use context_tokens as the label (it carries the batch ordering)
        x0_ot, x1_ot, _, ctx_shuffled = self.flow_matcher.ot_sampler.sample_plan_with_labels(
            x0, x1, y1=context_tokens
        )
        # Apply same OT permutation to global_cond
        # OT sampler returns permuted labels — infer permutation from context_tokens
        # Since OT only permutes batch dim, we can detect permutation by comparing
        # But simpler: re-derive global_cond from shuffled context_tokens
        glob_shuffled = ctx_shuffled.mean(dim=1)  # (B, d_model)
        
        # 6. Interpolation: dynamic (scheduler) or linear (fallback)
        B = x0_ot.size(0)
        t = torch.rand(B, device=self.device)
        
        if self.interp_scheduler is not None:
            alpha_t, sigma_t, alpha_dot, sigma_dot = self.interp_scheduler(glob_shuffled, t)
            xt = alpha_t * x1_ot + sigma_t * x0_ot
            ut = alpha_dot * x1_ot + sigma_dot * x0_ot
        else:
            t_ = t.unsqueeze(-1)  # (B, 1)
            xt = t_ * x1_ot + (1.0 - t_) * x0_ot
            ut = x1_ot - x0_ot
        
        # 7. Predict velocity using U-ViT v2
        vt_pred = self.uvit(xt, t, ctx_shuffled, glob_shuffled)
        
        # 8. Loss
        loss = self.criterion(vt_pred, ut)
        
        loss.backward()
        
        # Gradient Clipping to prevent exploding gradients.
        enc_grad_norm = torch.nn.utils.clip_grad_norm_(self.fusion_encoder.parameters(), 1.0)
        uvit_grad_norm = torch.nn.utils.clip_grad_norm_(self.uvit.parameters(), 1.0)
        if self.interp_scheduler is not None:
            torch.nn.utils.clip_grad_norm_(self.interp_scheduler.parameters(), 1.0)
        
        self.optimizer.step()
        
        # 9. EMA update (after optimizer step)
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
            if self.interp_scheduler is not None:
                self.interp_scheduler.train()
            
            pbar = tqdm(self.train_dl, desc=f"Epoch {epoch}/{self.epochs}")
            epoch_loss = 0.0
            num_steps = 0
            
            for step, batch in enumerate(pbar):
                loss, enc_gn, uvit_gn = self.train_step(batch)
                epoch_loss += loss
                num_steps += 1
                
                # Log Progress with gradient norms
                if step % log_freq == 0:
                    pbar.set_postfix({
                        "Loss": f"{loss:.4f}",
                        "EncGN": f"{enc_gn:.2f}",
                        "UViTGN": f"{uvit_gn:.2f}",
                        "LR": f"{self.lr_scheduler.get_last_lr()[0]:.2e}"
                    })
                
                if self.fast_dev_run and step >= 5: # Fast run a few steps for debugging
                    break
                    
            avg_train_loss = epoch_loss / num_steps
            
            # Validation every N epochs
            val_every = self.config["training"].get("val_every", 5)
            if epoch % val_every == 0 or epoch == self.epochs:
                val_loss, val_pcc = self.validate()
                
                # Free validation memory immediately
                gc.collect()
                torch.cuda.empty_cache()
                
                logger.info(
                    f"[Epoch {epoch}] Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val PCC: {val_pcc:.4f}"
                )
                
                # Log to CSV only after validation
                if not self.fast_dev_run:
                    with open(self.history_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch, avg_train_loss, val_loss, val_pcc, self.lr_scheduler.get_last_lr()[0]])
                
                # Save best model (only after validation)
                if not self.fast_dev_run and val_pcc > best_pcc:
                    best_pcc = val_pcc
                    self._save_checkpoint(epoch, val_loss, val_pcc, tag="best")
                    logger.info(f"  ★ New best PCC: {val_pcc:.4f}")
            else:
                logger.info(f"[Epoch {epoch}] Loss: {avg_train_loss:.4f} | Val (Skipped)")
            
            self.lr_scheduler.step()
            
            # Save latest checkpoint every 10 epochs (reduces I/O and memory pressure)
            if not self.fast_dev_run and epoch % 10 == 0:
                self._save_checkpoint(epoch, avg_train_loss, best_pcc, tag="latest")
                
            if self.fast_dev_run:
                break
                
        logger.info("Training Complete!")

    def _save_checkpoint(self, epoch, val_loss, val_pcc, tag="latest"):
        """Save checkpoint including EMA and scheduler state."""
        ckpt = {
            "epoch": epoch,
            "fusion_encoder": self.fusion_encoder.state_dict(),
            "uvit": self.uvit.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "val_loss": val_loss,
            "val_pcc": val_pcc,
        }
        if self.interp_scheduler is not None:
            ckpt["interp_scheduler"] = self.interp_scheduler.state_dict()
        if self.ema is not None:
            ckpt["ema"] = self.ema.state_dict()
        
        torch.save(ckpt, self.save_dir / f"checkpoint_{tag}.pt")

    @torch.no_grad()
    def validate(self):
        """Validate using EMA weights (if enabled) and CFG inference."""
        self.fusion_encoder.eval()
        self.uvit.eval()
        if self.interp_scheduler is not None:
            self.interp_scheduler.eval()
        
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
        
        # Fix validation noise for stable PCC across epochs
        rng_state = torch.cuda.get_rng_state() if self.device.type == 'cuda' else None
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(42)
        
        for step, batch in enumerate(self.val_dl):
            x1 = batch["x1"].to(self.device)
            condition = batch["condition"].to(self.device)
            x0 = torch.randn_like(x1)
            
            # 1. Encode condition
            context_tokens, global_cond = self.fusion_encoder(condition)
            
            # 2. Validation Loss on flow matching (with dynamic interpolation)
            x0_ot, x1_ot, _, ctx_shuffled = self.flow_matcher.ot_sampler.sample_plan_with_labels(
                x0, x1, y1=context_tokens
            )
            glob_shuffled = ctx_shuffled.mean(dim=1)
            
            B = x0_ot.size(0)
            t = torch.rand(B, device=self.device)
            
            if self.interp_scheduler is not None:
                alpha_t, sigma_t, alpha_dot, sigma_dot = self.interp_scheduler(glob_shuffled, t)
                xt = alpha_t * x1_ot + sigma_t * x0_ot
                ut = alpha_dot * x1_ot + sigma_dot * x0_ot
            else:
                t_ = t.unsqueeze(-1)
                xt = t_ * x1_ot + (1.0 - t_) * x0_ot
                ut = x1_ot - x0_ot
            
            vt_pred = self.uvit(xt, t, ctx_shuffled, glob_shuffled)
            
            loss = self.criterion(vt_pred, ut)
            val_loss += loss.item()
            
            # 3. ODE Solver with CFG for Generation & PCC computation
            x_pred = x0.clone()
            steps = 50  # More steps for accurate ODE integration
            dt = 1.0 / steps
            
            for st in range(steps):
                t_val = torch.full((x0.size(0),), st * dt, device=self.device)
                
                if self.cfg_enabled:
                    # CFG inference: v_guided = v_uncond + w * (v_cond - v_uncond)
                    null_ctx = torch.zeros_like(context_tokens)
                    null_glob = torch.zeros_like(global_cond)
                    v_uncond = self.uvit(x_pred, t_val, null_ctx, null_glob)
                    v_cond = self.uvit(x_pred, t_val, context_tokens, global_cond)
                    
                    v_guided = v_uncond + self.guidance_scale * (v_cond - v_uncond)
                    x_pred = x_pred + v_guided * dt
                else:
                    v_pred = self.uvit(x_pred, t_val, context_tokens, global_cond)
                    x_pred = x_pred + v_pred * dt
                
            all_x1.append(x1.cpu())
            all_pred.append(x_pred.cpu())
            
            num_batches += 1
            if self.fast_dev_run and step >= 2:
                break
        
        # Restore RNG state
        if rng_state is not None:
            torch.cuda.set_rng_state(rng_state)
                
        # 4. Calculate Temporal PCC (per voxel) matching baseline logic
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
