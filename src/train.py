"""
Conditional Flow Matching: Video → fMRI Generation

Training:
    x_t = t·x_1 + (1-t)·x_0          (linear interpolation)
    u_t = x_1 - x_0                    (target velocity)
    L   = E[‖v_θ(x_t, t, C) - u_t‖²]  (MSE loss)

Inference (Euler ODE):
    x_{t+Δt} = x_t + Δt · v_θ(x_t, t, C)

Run: python -m src.train --subject sub-01
"""

import argparse
import csv
import gc
import logging
import yaml
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.datamodule import SubjectDataModule
from src.models import FusionEncoder, UViT1D

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, subject: str, config_path: str = "src/configs/configs.yml",
                 fast_dev_run: bool = False):
        self.subject = subject
        self.fast_dev_run = fast_dev_run

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.config["training"].get("seed", 42))

        self._setup_data()
        self._setup_models()
        self._setup_optimizer()

        self.save_dir = Path(self.config["training"]["save_dir"]) / subject
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.save_dir / "history.csv"

    # ── Data ──────────────────────────────────────────────────────────
    def _setup_data(self):
        self.datamodule = SubjectDataModule(
            subject=self.subject, config_path="src/configs/configs.yml"
        )
        self.datamodule.setup()
        self.train_dl = self.datamodule.train_dataloader()
        self.val_dl = self.datamodule.val_dataloader()

    # ── Models ────────────────────────────────────────────────────────
    def _setup_models(self):
        enc_cfg = self.config["fusion_encoder"]
        modality_dims = [cfg["dim"] for cfg in self.config["modalities"].values()]

        self.encoder = FusionEncoder(
            modality_dims=modality_dims,
            d_model=enc_cfg["d_model"],
            num_queries=enc_cfg["num_queries"],
            dropout=enc_cfg["dropout"],
        ).to(self.device)

        uvit_cfg = self.config["uvit"]
        self.velocity_net = UViT1D(
            in_features=self.config["data"]["fmri_dim"],
            patch_size=uvit_cfg["patch_size"],
            embed_dim=uvit_cfg["embed_dim"],
            depth=uvit_cfg["depth"],
            num_heads=uvit_cfg["num_heads"],
            mlp_ratio=uvit_cfg["mlp_ratio"],
            drop_rate=uvit_cfg["drop_rate"],
        ).to(self.device)

        enc_p = sum(p.numel() for p in self.encoder.parameters())
        vel_p = sum(p.numel() for p in self.velocity_net.parameters())
        logger.info(f"Encoder: {enc_p/1e6:.1f}M  Velocity: {vel_p/1e6:.1f}M  Total: {(enc_p+vel_p)/1e6:.1f}M")

    # ── Optimizer ─────────────────────────────────────────────────────
    def _setup_optimizer(self):
        cfg = self.config["training"]
        self.epochs = cfg["epochs"]
        self.all_params = list(self.encoder.parameters()) + list(self.velocity_net.parameters())
        self.optimizer = AdamW(
            self.all_params,
            lr=float(cfg["learning_rate"]),
            weight_decay=float(cfg["weight_decay"]),
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)

    # ── Train step ────────────────────────────────────────────────────
    def train_step(self, batch) -> float:
        self.optimizer.zero_grad()

        x1 = batch["x1"].to(self.device)                      # (B, fmri_dim)
        condition = batch["condition"].to(self.device)         # (B, T, feat_dim)

        C = self.encoder(condition)                            # (B, N_cond, d_model)

        B = x1.size(0)
        x0 = torch.randn_like(x1)                             # (B, fmri_dim)
        t = torch.rand(B, device=self.device)                  # (B,)

        # x_t = t·x1 + (1-t)·x0
        t_ = t.unsqueeze(1)
        x_t = t_ * x1 + (1.0 - t_) * x0

        # u_t = x1 - x0
        u_t = x1 - x0

        # Predict velocity
        v_t = self.velocity_net(x_t, t, C)

        loss = F.mse_loss(v_t, u_t)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.all_params, max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    # ── Validation ────────────────────────────────────────────────────
    @torch.no_grad()
    def validate(self):
        self.encoder.eval()
        self.velocity_net.eval()

        ode_steps = self.config["training"].get("ode_steps", 50)
        dt = 1.0 / ode_steps

        val_loss_sum = 0.0
        all_x1, all_pred = [], []

        for batch in self.val_dl:
            x1 = batch["x1"].to(self.device)
            condition = batch["condition"].to(self.device)
            C = self.encoder(condition)
            B = x1.size(0)

            # ── Validation loss (same CFM objective) ──
            x0 = torch.randn_like(x1)
            t = torch.rand(B, device=self.device)
            t_ = t.unsqueeze(1)
            x_t = t_ * x1 + (1.0 - t_) * x0
            u_t = x1 - x0
            v_t = self.velocity_net(x_t, t, C)
            val_loss_sum += F.mse_loss(v_t, u_t).item()

            # ── Generate fMRI via Euler ODE solver ──
            x = torch.randn(B, x1.size(1), device=self.device)
            for i in range(ode_steps):
                t_i = torch.full((B,), i * dt, device=self.device)
                x = x + self.velocity_net(x, t_i, C) * dt

            all_x1.append(x1.cpu())
            all_pred.append(x.cpu())

        val_loss = val_loss_sum / len(self.val_dl)
        val_pcc = self._pearson_corrcoef(torch.cat(all_x1), torch.cat(all_pred))

        self.encoder.train()
        self.velocity_net.train()
        gc.collect()
        torch.cuda.empty_cache()

        return val_loss, val_pcc

    @staticmethod
    def _pearson_corrcoef(x: torch.Tensor, y: torch.Tensor) -> float:
        """Mean per-sample Pearson correlation."""
        vx = x - x.mean(dim=1, keepdim=True)
        vy = y - y.mean(dim=1, keepdim=True)
        r = (vx * vy).sum(dim=1) / (vx.norm(dim=1) * vy.norm(dim=1) + 1e-8)
        return r.mean().item()

    # ── Main loop ─────────────────────────────────────────────────────
    def train(self):
        log_freq = self.config["training"].get("log_freq", 50)
        val_every = self.config["training"].get("val_every", 5)

        if not self.fast_dev_run:
            with open(self.history_file, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_pcc", "lr"])

        logger.info(f"Training on {self.device}")

        for epoch in range(1, self.epochs + 1):
            self.encoder.train()
            self.velocity_net.train()

            pbar = tqdm(self.train_dl, desc=f"Epoch {epoch}/{self.epochs}")
            epoch_loss = 0.0

            for step, batch in enumerate(pbar):
                loss = self.train_step(batch)
                epoch_loss += loss

                if (step + 1) % log_freq == 0:
                    pbar.set_postfix(Loss=f"{loss:.4f}", LR=f"{self.scheduler.get_last_lr()[0]:.2e}")

                if self.fast_dev_run and step >= 4:
                    break

            avg_loss = epoch_loss / max(step + 1, 1)

            # Validate periodically
            if epoch % val_every == 0 or epoch == self.epochs:
                val_loss, val_pcc = self.validate()
                logger.info(f"[Epoch {epoch}] Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val PCC: {val_pcc:.4f}")

                if not self.fast_dev_run:
                    lr = self.scheduler.get_last_lr()[0]
                    with open(self.history_file, "a", newline="") as f:
                        csv.writer(f).writerow([epoch, avg_loss, val_loss, val_pcc, lr])

                    torch.save({
                        "epoch": epoch,
                        "encoder": self.encoder.state_dict(),
                        "velocity_net": self.velocity_net.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "val_pcc": val_pcc,
                    }, self.save_dir / "checkpoint.pt")
            else:
                logger.info(f"[Epoch {epoch}] Loss: {avg_loss:.4f}")

            self.scheduler.step()

            if self.fast_dev_run:
                break

        logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CFM: Video → fMRI")
    parser.add_argument("--subject", type=str, default="sub-01")
    parser.add_argument("--fast_dev_run", action="store_true")
    args = parser.parse_args()

    Trainer(subject=args.subject, fast_dev_run=args.fast_dev_run).train()
