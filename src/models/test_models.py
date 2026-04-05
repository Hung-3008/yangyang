"""
Test script for models.
Run: python -m src.models.test_models
"""

import logging
import torch
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def test_fusion_encoder():
    from src.models import MultimodalFusionEncoder
    import yaml

    with open("src/configs/configs.yml", "r") as f:
        config = yaml.safe_load(f)

    enc_cfg = config["fusion_encoder"]
    
    logger.info("=" * 60)
    logger.info("TEST: Multimodal Fusion Encoder")
    logger.info("=" * 60)
    
    model = MultimodalFusionEncoder(
        input_dim=22144,
        d_model=enc_cfg["d_model"],
        num_latents=enc_cfg["num_latents"],
        num_layers=enc_cfg["num_layers"],
        nhead=enc_cfg["nhead"],
        dropout=enc_cfg["dropout"]
    )
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Parameters: {total_params / 1e6:.2f} M")

    # Create dummy input: Batch = 8, Time = 31, Dim = 22144
    B, T, D = 8, 31, 22144
    logger.info(f"\nCreating dummy input: B={B}, T={T}, D={D}")
    x = torch.randn(B, T, D)
    
    # Enable eval to ignore dropout stochasticity
    model.eval()
    
    t0 = time.time()
    with torch.no_grad():
        out = model(x)
    t1 = time.time()

    logger.info(f"Output shape: {out.shape}")
    logger.info(f"Forward time: {(t1-t0)*1000:.1f}ms")
    
    expected_shape = (B, enc_cfg["num_latents"], enc_cfg["d_model"])
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    logger.info("✅ Shape verified!")
    
    # Verify backward capability
    logger.info("\nChecking gradient flow...")
    model.train()
    x = torch.randn(B, T, D, requires_grad=True)
    out = model(x)
    
    loss = out.mean()
    loss.backward()
    
    # Verify some grads
    assert model.latent_queries.grad is not None, "latent_queries gradient is None!"
    assert model.proj.weight.grad is not None, "Projector gradient is None!"
    logger.info("✅ Gradients check passed!")

if __name__ == "__main__":
    try:
        test_fusion_encoder()
        logger.info("\n" + "=" * 60)
        logger.info("ALL MODEL TESTS PASSED ✅")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}", exc_info=True)
        sys.exit(1)
