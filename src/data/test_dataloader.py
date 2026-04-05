"""
Test script để verify data loader hoạt động chính xác.
Chạy: python -m src.data.test_dataloader
"""

import logging
import time
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def test_dataset_basic():
    """Test basic dataset creation and sample loading."""
    from src.data.dataset import FlowMatchingDataset
    import yaml

    with open("src/configs/configs.yml", "r") as f:
        config = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("TEST 1: Basic Dataset Creation (friends s01 only)")
    logger.info("=" * 60)

    ds = FlowMatchingDataset(
        subject="sub-01",
        split="friends",
        modality_configs=config["modalities"],
        data_cfg=config["data"],
        seasons=["s01"],  # Just s01 for quick test
        cache_in_memory=False,
    )

    logger.info(f"Dataset: {ds}")
    logger.info(f"Total samples: {len(ds)}")
    logger.info(f"Total clips: {len(ds.clips)}")
    logger.info(f"Feature dim: {ds.total_feat_dim}")

    # Load first sample
    logger.info("\nLoading first sample...")
    t0 = time.time()
    sample = ds[0]
    t1 = time.time()

    logger.info(f"  x1 shape: {sample['x1'].shape}, dtype: {sample['x1'].dtype}")
    logger.info(f"  condition shape: {sample['condition'].shape}, dtype: {sample['condition'].dtype}")
    logger.info(f"  Load time: {(t1-t0)*1000:.1f}ms")

    # Check values
    x1 = sample["x1"]
    cond = sample["condition"]
    logger.info(f"  x1 range: [{x1.min():.4f}, {x1.max():.4f}], mean={x1.mean():.4f}")
    logger.info(f"  condition range: [{cond.min():.4f}, {cond.max():.4f}], mean={cond.mean():.4f}")

    # Check for NaN/Inf
    assert not x1.isnan().any(), "NaN in fMRI!"
    assert not x1.isinf().any(), "Inf in fMRI!"
    assert not cond.isnan().any(), "NaN in condition!"
    assert not cond.isinf().any(), "Inf in condition!"
    logger.info("  ✅ No NaN/Inf detected")

    # Check dimensions
    assert x1.shape == (config["data"]["fmri_dim"],), f"Expected fMRI dim {config['data']['fmri_dim']}, got {x1.shape}"
    expected_cond_shape = (config["data"].get("window_size", 31), ds.total_feat_dim)
    assert cond.shape == expected_cond_shape, f"Expected feat dim {expected_cond_shape}, got {cond.shape}"
    logger.info("  ✅ Shapes correct")

    return ds


def test_movie10():
    """Test movie10 split (with naming convention handling)."""
    from src.data.dataset import FlowMatchingDataset
    import yaml

    with open("src/configs/configs.yml", "r") as f:
        config = yaml.safe_load(f)

    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Movie10 Dataset")
    logger.info("=" * 60)

    ds = FlowMatchingDataset(
        subject="sub-01",
        split="movie10",
        modality_configs=config["modalities"],
        data_cfg=config["data"],
        seasons=None,
        cache_in_memory=False,
    )

    logger.info(f"Dataset: {ds}")
    logger.info(f"Total clips: {len(ds.clips)}")

    # Print which clips were found
    for clip in ds.clips[:5]:
        avail = sum(1 for p in clip["feat_paths"].values() if p is not None)
        total = len(clip["feat_paths"])
        runs = len(clip["fmri_keys"])
        logger.info(
            f"  clip={clip['task_id']}: "
            f"modalities={avail}/{total}, "
            f"fmri_runs={runs}, "
            f"trs={clip['n_trs']}"
        )

    # Load a sample
    if len(ds) > 0:
        sample = ds[0]
        logger.info(f"\n  Sample 0: x1={sample['x1'].shape}, cond={sample['condition'].shape}")
        logger.info("  ✅ Movie10 loading works")

    return ds


def test_datamodule():
    """Test full SubjectDataModule."""
    from src.data.datamodule import SubjectDataModule

    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: SubjectDataModule (sub-01)")
    logger.info("=" * 60)

    dm = SubjectDataModule(
        subject="sub-01",
        config_path="src/configs/configs.yml",
        batch_size=32,
        num_workers=0,  # 0 for testing
    )
    dm.setup()

    logger.info(f"\nDataModule: {dm}")

    # Test train dataloader
    logger.info("\nTesting train dataloader...")
    train_dl = dm.train_dataloader()
    batch = next(iter(train_dl))
    logger.info(f"  Batch x1: {batch['x1'].shape}")
    logger.info(f"  Batch condition: {batch['condition'].shape}")
    logger.info("  ✅ Train dataloader works")

    # Test val dataloader
    logger.info("\nTesting val dataloader...")
    val_dl = dm.val_dataloader()
    batch = next(iter(val_dl))
    logger.info(f"  Batch x1: {batch['x1'].shape}")
    logger.info(f"  Batch condition: {batch['condition'].shape}")
    logger.info("  ✅ Val dataloader works")

    # Benchmark loading speed
    logger.info("\nBenchmarking train loading speed (1 epoch)...")
    t0 = time.time()
    n_batches = 0
    n_samples = 0
    for batch in train_dl:
        n_batches += 1
        n_samples += batch["x1"].shape[0]
        if n_batches >= 50:  # Just test 50 batches
            break
    t1 = time.time()
    logger.info(
        f"  {n_batches} batches, {n_samples} samples in {t1-t0:.2f}s "
        f"({n_samples/(t1-t0):.0f} samples/s)"
    )


if __name__ == "__main__":
    try:
        ds = test_dataset_basic()
        test_movie10()
        test_datamodule()
        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS PASSED ✅")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}", exc_info=True)
        sys.exit(1)
