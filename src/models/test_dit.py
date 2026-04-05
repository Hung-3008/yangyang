"""
Test Script cho DiTConditional
Chạy lệnh: python -m src.models.test_dit
"""

import logging
import torch
import sys
import yaml
import time
from src.models.dit import DiTConditional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def test_dit():
    with open("src/configs/configs.yml", "r") as f:
        config = yaml.safe_load(f)
    
    cfg = config["velocity_net"]
        
    logger.info("=" * 60)
    logger.info("TEST: DiT Velocity Network")
    logger.info("=" * 60)
    
    model = DiTConditional(
        in_features=1000,
        patch_size=cfg["patch_size"],
        hidden_size=cfg["d_model"],
        depth=cfg["depth"],
        num_heads=cfg["nhead"],
        context_dim=cfg["context_dim"],
        dropout=cfg["dropout"]
    )
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Parameters: {total_params / 1e6:.2f} M")

    # Create dummy input: Batch = 32
    B = 32
    x = torch.randn(B, 1000)                      # Noisy fMRI (x_t)
    t = torch.rand(B)                             # Timesteps (t)
    context = torch.randn(B, 8, cfg["context_dim"]) # Condition từ Fusion Encoder
    
    logger.info(f"\nDummy Input:")
    logger.info(f"  x: {x.shape}")
    logger.info(f"  t: {t.shape}")
    logger.info(f"  context (C): {context.shape}")
    
    t0 = time.time()
    out = model(x, t, context)
    t1 = time.time()
    
    logger.info(f"\nOutput shape: {out.shape}")
    logger.info(f"Forward time (B=32): {(t1-t0)*1000:.1f} ms")
    
    assert out.shape == torch.Size([B, 1000]), f"Sai Output Shape! Kì vọng {[B, 1000]} nhưng nhận được {out.shape}"
    logger.info("✅ Shape verified!")
    
    # Kiem tra gradients
    logger.info("\nChecking gradient flow...")
    x.requires_grad_(True)
    t.requires_grad_(True)
    context.requires_grad_(True)
    
    out = model(x, t, context)
    loss = out.mean()
    loss.backward()
    
    assert model.blocks[0].adaLN_modulation[1].weight.grad is not None, "AdaLN gradient None!"
    assert model.blocks[0].attn2.q_proj_weight.grad is not None, "Cross-Attention gradient None!"
    logger.info("✅ Gradients check passed! Mạng hỗ trợ Backpropagation thành công.")

if __name__ == "__main__":
    try:
        test_dit()
        logger.info("\n" + "=" * 60)
        logger.info("ALL DiT TESTS PASSED ✅")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}", exc_info=True)
        sys.exit(1)
