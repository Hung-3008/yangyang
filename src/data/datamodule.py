"""
SubjectDataModule: Creates train and val DataLoaders for a single subject.

Train = movie10 (all) + friends (s01-s05)
Val   = friends (s06)
"""

import logging
from pathlib import Path

import yaml
from torch.utils.data import DataLoader, ConcatDataset

from .dataset import FlowMatchingDataset

logger = logging.getLogger(__name__)


class SubjectDataModule:
    """Manages train/val datasets and dataloaders for one subject."""

    def __init__(
        self,
        subject: str,
        config_path: str = "src/configs/configs.yml",
        batch_size: int = None,
        num_workers: int = None,
        cache_in_memory: bool = False,
    ):
        """
        Parameters
        ----------
        subject : str
            Subject ID, e.g. "sub-01"
        config_path : str
            Path to configs.yml
        batch_size : int | None
            Override batch_size from config
        num_workers : int | None
            Override num_workers from config
        cache_in_memory : bool
            Preload all data into RAM
        """
        self.subject = subject
        self.cache_in_memory = cache_in_memory

        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.data_cfg = self.config["data"]
        self.modality_configs = self.config["modalities"]
        self.training_cfg = self.config["training"]

        self.cache_in_memory = cache_in_memory if cache_in_memory is not False else self.data_cfg.get("cache_in_memory", False)

        self.batch_size = batch_size or self.training_cfg["batch_size"]
        self.num_workers = num_workers or self.training_cfg["num_workers"]

        self._train_dataset = None
        self._val_dataset = None

    def setup(self):
        """Create train and val datasets."""
        logger.info(f"Setting up datasets for {self.subject}...")

        train_datasets = []

        # 1. Friends (train seasons: s01-s05)
        train_seasons = self.data_cfg["train_seasons"]
        friends_train = FlowMatchingDataset(
            subject=self.subject,
            split="friends",
            modality_configs=self.modality_configs,
            data_cfg=self.data_cfg,
            seasons=train_seasons,
            cache_in_memory=self.cache_in_memory,
        )
        train_datasets.append(friends_train)
        logger.info(f"  Friends train (seasons {train_seasons}): {len(friends_train)} TRs")

        # 2. Movie10 (all)
        movie10_train = None
        if self.data_cfg.get("train_include_movie10", True):
            movie10_train = FlowMatchingDataset(
                subject=self.subject,
                split="movie10",
                modality_configs=self.modality_configs,
                data_cfg=self.data_cfg,
                seasons=None,  # include all
                cache_in_memory=self.cache_in_memory,
            )
            train_datasets.append(movie10_train)
            logger.info(f"  Movie10 train: {len(movie10_train)} TRs")

        # --- Unify normalization stats across ALL training data ---
        # Merge stats from friends + movie10 using combined Welford merge
        if self.data_cfg.get("normalize_fmri", True) and hasattr(friends_train, "_fmri_mean"):
            if movie10_train is not None and hasattr(movie10_train, "_fmri_mean"):
                # Combine two sets of Welford stats
                import numpy as np
                n_f = friends_train._fmri_n if hasattr(friends_train, "_fmri_n") else len(friends_train)
                n_m = movie10_train._fmri_n if hasattr(movie10_train, "_fmri_n") else len(movie10_train)
                n_total = n_f + n_m
                mean_f = friends_train._fmri_mean.astype(np.float64)
                mean_m = movie10_train._fmri_mean.astype(np.float64)
                std_f = friends_train._fmri_std.astype(np.float64)
                std_m = movie10_train._fmri_std.astype(np.float64)

                # Combined mean
                combined_mean = (n_f * mean_f + n_m * mean_m) / n_total
                # Combined variance (parallel Welford)
                combined_var = (
                    (n_f * (std_f**2 + (mean_f - combined_mean)**2) +
                     n_m * (std_m**2 + (mean_m - combined_mean)**2)) / n_total
                )
                combined_std = np.sqrt(combined_var).astype(np.float32)
                combined_mean = combined_mean.astype(np.float32)

                # Share unified stats to ALL datasets
                friends_train._fmri_mean = combined_mean
                friends_train._fmri_std = combined_std
                movie10_train._fmri_mean = combined_mean
                movie10_train._fmri_std = combined_std
                logger.info(f"  Unified fMRI stats across friends+movie10 (n={n_total})")

        # Combine
        if len(train_datasets) > 1:
            self._train_dataset = ConcatDataset(train_datasets)
        else:
            self._train_dataset = train_datasets[0]

        # 3. Validation: friends s06
        val_seasons = self.data_cfg["val_seasons"]
        self._val_dataset = FlowMatchingDataset(
            subject=self.subject,
            split="friends",
            modality_configs=self.modality_configs,
            data_cfg=self.data_cfg,
            seasons=val_seasons,
            cache_in_memory=self.cache_in_memory,
        )
        logger.info(f"  Friends val (seasons {val_seasons}): {len(self._val_dataset)} TRs")

        total_train = len(self._train_dataset)
        total_val = len(self._val_dataset)
        logger.info(
            f"  Total: {total_train} train TRs, {total_val} val TRs "
            f"({total_val / (total_train + total_val) * 100:.1f}% val)"
        )

        # Share normalization stats from train → val
        if hasattr(friends_train, "_fmri_mean"):
            self._val_dataset._fmri_mean = friends_train._fmri_mean
            self._val_dataset._fmri_std = friends_train._fmri_std
            logger.info("  Shared fMRI normalization stats from train → val")

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            self.setup()
        return self._train_dataset

    @property
    def val_dataset(self):
        if self._val_dataset is None:
            self.setup()
        return self._val_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def __repr__(self):
        train_n = len(self.train_dataset) if self._train_dataset else "?"
        val_n = len(self.val_dataset) if self._val_dataset else "?"
        return (
            f"SubjectDataModule(subject={self.subject}, "
            f"train={train_n}, val={val_n}, bs={self.batch_size})"
        )
