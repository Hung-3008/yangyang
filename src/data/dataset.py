"""
FlowMatchingDataset: Per-TR sampling dataset for Video-to-fMRI Flow Matching.

Each sample = 1 TR (time repetition):
  - x1:        fMRI response at that TR, shape (1000,)
  - condition:  concatenated multimodal features at that TR, shape (D_total,)

Features and fMRI are aligned by task_id (clip name).
"""

import logging
from pathlib import Path
from collections import defaultdict
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FlowMatchingDataset(Dataset):
    """Dataset sampling individual TRs for conditional flow matching.

    Handles:
    - Multiple modalities with different naming conventions
    - fMRI run-1/run-2 duplicates (average by default)
    - Feature shape differences: (T, D) vs (T, 1, D)
    - Time alignment mismatches (±2 TRs between features and fMRI)
    - Missing modality features (zero-filled)
    """

    def __init__(
        self,
        subject: str,
        split: str,
        modality_configs: dict,
        data_cfg: dict,
        seasons: Optional[list[str]] = None,
        cache_in_memory: bool = False,
    ):
        """
        Parameters
        ----------
        subject : str
            Subject ID, e.g. "sub-01"
        split : str
            "friends" or "movie10"
        modality_configs : dict
            From configs.yml modalities section
        data_cfg : dict
            From configs.yml data section
        seasons : list[str] | None
            For friends split, which seasons to include, e.g. ["s01", "s02"].
            None = include all seasons.
        cache_in_memory : bool
            If True, preload all data into RAM during __init__.
        """
        super().__init__()
        self.subject = subject
        self.split = split
        self.modality_configs = modality_configs
        self.data_cfg = data_cfg
        self.seasons = seasons
        self.cache_in_memory = cache_in_memory

        self.features_dir = Path(data_cfg["features_dir"])
        self.fmri_dir = Path(data_cfg["fmri_dir"])
        self.fmri_dim = data_cfg["fmri_dim"]
        self.window_size = data_cfg.get("window_size", 31)
        self.run_strategy = data_cfg.get("fmri_run_strategy", "average")
        self.normalize_fmri = data_cfg.get("normalize_fmri", True)

        # Compute total feature dimension
        self.total_feat_dim = sum(cfg["dim"] for cfg in modality_configs.values())

        # Build the index
        self._build_index()

        # Compute normalization stats
        if self.normalize_fmri:
            self._compute_fmri_stats()

        # Optionally cache raw modalities and fMRI in memory
        self._feat_cache = {}
        self._fmri_cache = {}
        if cache_in_memory:
            self._preload_cache()

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------
    def _build_index(self):
        """Build flat index of (clip_info, tr_idx) for all valid TRs."""
        self.clips = []      # List of clip info dicts
        self.index = []       # List of (clip_idx, tr_idx)

        # Load fMRI file
        fmri_filename = self.data_cfg["fmri_files"][self.split].format(
            subject=self.subject
        )
        fmri_path = self.fmri_dir / self.subject / "func" / fmri_filename
        if not fmri_path.exists():
            raise FileNotFoundError(f"fMRI file not found: {fmri_path}")

        # Parse fMRI sessions
        with h5py.File(fmri_path, "r") as fh:
            session_keys = sorted(fh.keys())

        # Group by base task_id (handling run-1/run-2)
        task_sessions = defaultdict(list)
        for key in session_keys:
            # key format: "ses-XXX_task-{task_id}" or "ses-XXX_task-{task_id}_run-{N}"
            parts = key.split("_task-")
            if len(parts) != 2:
                logger.warning(f"Skipping unexpected fMRI key format: {key}")
                continue

            task_part = parts[1]
            # Split off _run-N if present
            if "_run-" in task_part:
                base_task = task_part.split("_run-")[0]
            else:
                base_task = task_part

            task_sessions[base_task].append(key)

        # Filter by season if specified
        if self.split == "friends" and self.seasons:
            filtered = {}
            for task_id, keys in task_sessions.items():
                # task_id like "s01e01a" → season = "s01"
                season = task_id[:3]
                if season in self.seasons:
                    filtered[task_id] = keys
            task_sessions = filtered

        # Build clips
        n_skipped = 0
        for task_id, fmri_keys in sorted(task_sessions.items()):
            # Find feature files for this task
            feat_paths = self._find_feature_paths(task_id)

            # Check if at least ONE modality has features
            available_mods = {m for m, p in feat_paths.items() if p is not None}
            if not available_mods:
                logger.warning(f"No features found for task {task_id}, skipping")
                n_skipped += 1
                continue

            # Determine valid time range (min across features and fMRI)
            n_trs = self._get_valid_trs(fmri_path, fmri_keys, feat_paths)
            if n_trs <= 0:
                logger.warning(f"No valid TRs for task {task_id}, skipping")
                n_skipped += 1
                continue

            clip_idx = len(self.clips)
            self.clips.append({
                "task_id": task_id,
                "fmri_path": str(fmri_path),
                "fmri_keys": fmri_keys,
                "feat_paths": feat_paths,
                "n_trs": n_trs,
            })

            for tr in range(n_trs):
                self.index.append((clip_idx, tr))

        if n_skipped > 0:
            logger.info(f"Skipped {n_skipped} clips (no matching features)")

        logger.info(
            f"[{self.subject}/{self.split}] "
            f"Built index: {len(self.clips)} clips, "
            f"{len(self.index)} TRs total"
        )

    def _find_feature_paths(self, task_id: str) -> dict:
        """Find feature file paths for a given task_id across all modalities.

        Returns dict: {modality_name: Path or None}
        """
        feat_paths = {}

        for mod_name, mod_cfg in self.modality_configs.items():
            path = self._resolve_feature_path(mod_name, mod_cfg, task_id)
            feat_paths[mod_name] = path

        return feat_paths

    def _resolve_feature_path(
        self, mod_name: str, mod_cfg: dict, task_id: str
    ) -> Optional[Path]:
        """Resolve the feature file path for a specific modality and task.

        Handles naming convention differences:
        - friends: {mod}/friends/s{N}/friends_{task_id}.h5
        - movie10: {mod}/movie10/{category}/{[prefix]}{task_id}.h5
          where prefix = "movie10_" for Llama models, "" for others
        """
        mod_dir = self.features_dir / mod_name / self.split

        if self.split == "friends":
            # task_id = "s01e01a" → season folder = "s1"
            season_num = int(task_id[1:3])  # "01" → 1
            season_dir = mod_dir / f"s{season_num}"
            filename = f"friends_{task_id}.npy"
            path = season_dir / filename

        elif self.split == "movie10":
            # Need to find the category subfolder containing this task
            prefix = mod_cfg.get("movie10_prefix", "")
            filename = f"{prefix}{task_id}.npy"

            # Search in category subdirs
            path = None
            if mod_dir.exists():
                for cat_dir in mod_dir.iterdir():
                    if cat_dir.is_dir():
                        candidate = cat_dir / filename
                        if candidate.exists():
                            path = candidate
                            break
        else:
            path = None

        if path is not None and path.exists():
            return path

        return None

    def _get_valid_trs(
        self,
        fmri_path: Path,
        fmri_keys: list[str],
        feat_paths: dict,
    ) -> int:
        """Determine number of valid TRs.

        Uses fMRI T as base, verified against ONE feature file to catch
        large mismatches. This avoids opening all 8 modality files per clip
        during index building (which is very slow on HDD).
        """
        # Get fMRI time dimension (fMRI file is shared, fast to read shape)
        with h5py.File(fmri_path, "r") as fh:
            fmri_trs = [fh[key].shape[0] for key in fmri_keys]
            fmri_T = min(fmri_trs)

        # Check ONE feature file to verify alignment
        for mod_name, path in feat_paths.items():
            if path is None:
                continue
            try:
                # Pooled numpy files are directly accessible
                feat_T = np.load(path, mmap_mode='r').shape[0]
                # Use min to handle ±2 TR mismatches
                return min(fmri_T, feat_T)
            except Exception as e:
                logger.warning(f"Error reading {path}: {e}")
                continue

        # No features readable — use fMRI T
        return fmri_T

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        clip_idx, tr_idx = self.index[idx]
        clip = self.clips[clip_idx]

        # Load fMRI
        fmri = self._load_fmri(clip, tr_idx)

        # Load and concat features
        condition = self._load_features(clip, tr_idx)

        sample = {
            "x1": torch.from_numpy(fmri).float(),
            "condition": torch.from_numpy(condition).float(),
        }

        return sample

    def _get_h5_handle(self, path):
        if not hasattr(self, '_h5_handles'):
            self._h5_handles = {}
        path_str = str(path)
        if path_str not in self._h5_handles:
            self._h5_handles[path_str] = h5py.File(path_str, "r")
        return self._h5_handles[path_str]

    def __del__(self):
        if hasattr(self, '_h5_handles'):
            for fh in self._h5_handles.values():
                try:
                    fh.close()
                except:
                    pass

    def _load_fmri(self, clip: dict, tr_idx: int) -> np.ndarray:
        """Load fMRI for a specific TR, handling run averaging."""
        fmri_path = clip["fmri_path"]
        fmri_keys = clip["fmri_keys"]

        def get_arr(key):
            if self.cache_in_memory and fmri_path in self._fmri_cache:
                return self._fmri_cache[fmri_path][key]
            return self._get_h5_handle(fmri_path)[key]

        if self.run_strategy == "average" and len(fmri_keys) > 1:
            # Average across runs
            runs = [get_arr(k)[tr_idx].astype(np.float32) for k in fmri_keys]
            fmri = np.mean(runs, axis=0)
        elif self.run_strategy == "first":
            fmri = get_arr(fmri_keys[0])[tr_idx].astype(np.float32)
        else:
            # "all" or single run
            fmri = get_arr(fmri_keys[0])[tr_idx].astype(np.float32)

        # Normalize
        if self.normalize_fmri and hasattr(self, "_fmri_mean"):
            fmri = (fmri - self._fmri_mean) / (self._fmri_std + 1e-8)

        return fmri

    def _load_features(self, clip: dict, tr_idx: int) -> np.ndarray:
        """Load and concatenate features across a temporal window."""
        feat_parts = []
        
        start_idx = tr_idx - self.window_size + 1
        valid_start = max(0, start_idx)
        
        for mod_name, mod_cfg in self.modality_configs.items():
            path = clip["feat_paths"].get(mod_name)
            dim = mod_cfg["dim"]
            
            if path is None:
                feat_parts.append(np.zeros((self.window_size, dim), dtype=np.float32))
                continue
                
            if self.cache_in_memory and str(path) in self._feat_cache:
                feat_arr = self._feat_cache[str(path)]
            else:
                feat_arr = np.load(path, mmap_mode='r')
            feat_T = feat_arr.shape[0]
            
            valid_end = min(tr_idx + 1, feat_T)
            actual_valid_start = min(valid_start, feat_T)
            
            if actual_valid_start >= valid_end:
                feat_parts.append(np.zeros((self.window_size, dim), dtype=np.float32))
                continue
            
            feat_slice = feat_arr[actual_valid_start:valid_end].astype(np.float32)
            
            # Xử lý trường hợp mảng numpy có dạng (1, T, D) hoặc (T, 1, D) thay vì (T, D)
            if mod_cfg.get("needs_squeeze", False) and feat_slice.ndim > 2:
                # Nếu pooling script chưa squeeze được hoàn toàn thì squeeze
                feat_slice = feat_slice.reshape(-1, dim)
            
            # Xử lý avg pool cho modality có spatial tokens (e.g. dinov2_giant: T, 4, 1536)
            if mod_cfg.get("needs_avg_pool", False) and feat_slice.ndim > 2:
                feat_slice = feat_slice.mean(axis=1)  # (T, S, D) → (T, D)
                
            valid_L = valid_end - actual_valid_start
            feat = feat_slice.reshape(valid_L, dim)
            
            # We need exact window_size output
            pad_start = actual_valid_start - start_idx
            
            out_feat = np.zeros((self.window_size, dim), dtype=np.float32)
            out_feat[pad_start:pad_start + valid_L] = feat
            
            feat_parts.append(out_feat)
                
        # Concatenate along feature dimension
        return np.concatenate(feat_parts, axis=-1)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    def _compute_fmri_stats(self):
        """Compute per-parcel mean and std across all TRs in this dataset."""
        logger.info(f"Computing fMRI normalization stats for {self.subject}/{self.split}...")

        # Accumulate online (Welford's algorithm) to avoid loading everything
        n = 0
        mean = np.zeros(self.fmri_dim, dtype=np.float64)
        M2 = np.zeros(self.fmri_dim, dtype=np.float64)

        for clip in self.clips:
            with h5py.File(clip["fmri_path"], "r") as fh:
                for key in clip["fmri_keys"]:
                    data = fh[key][: clip["n_trs"]].astype(np.float64)
                    for row in data:
                        n += 1
                        delta = row - mean
                        mean += delta / n
                        delta2 = row - mean
                        M2 += delta * delta2

        if n > 1:
            variance = M2 / (n - 1)
            std = np.sqrt(variance)
        else:
            std = np.ones(self.fmri_dim, dtype=np.float64)

        self._fmri_mean = mean.astype(np.float32)
        self._fmri_std = std.astype(np.float32)
        self._fmri_n = n  # Store count for potential stat merging

        logger.info(
            f"  fMRI stats: mean=[{mean.min():.3f}, {mean.max():.3f}], "
            f"std=[{std.min():.3f}, {std.max():.3f}], n_trs={n}"
        )

    def get_fmri_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) arrays for external use (e.g., denormalization)."""
        return self._fmri_mean.copy(), self._fmri_std.copy()

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------
    def _preload_cache(self):
        """Preload all base numpy arrays and fMRI vectors into memory to prevent File I/O overhead and redundant sliding window copies."""
        logger.info(f"[{self.subject}/{self.split}] Preloading raw features and fMRI into RAM (File-level Caching)...")
        
        for clip in self.clips:
            # 1. Preload features
            for mod_name, path in clip["feat_paths"].items():
                if path is not None:
                    path_str = str(path)
                    if path_str not in self._feat_cache:
                        # Load whole numpy array to RAM
                        self._feat_cache[path_str] = np.load(path)
                        
            # 2. Preload fMRI (check per-key, not per-file)
            fmri_path = clip["fmri_path"]
            if fmri_path not in self._fmri_cache:
                self._fmri_cache[fmri_path] = {}
            missing_keys = [k for k in clip["fmri_keys"] if k not in self._fmri_cache[fmri_path]]
            if missing_keys:
                with h5py.File(fmri_path, "r") as fh:
                    for key in missing_keys:
                        self._fmri_cache[fmri_path][key] = fh[key][:]

        logger.info(f"[{self.subject}/{self.split}] Preloading complete (Unique Features: {len(self._feat_cache)} files).")

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
    def __repr__(self):
        return (
            f"FlowMatchingDataset("
            f"subject={self.subject}, split={self.split}, "
            f"clips={len(self.clips)}, trs={len(self.index)}, "
            f"feat_dim={self.total_feat_dim}, fmri_dim={self.fmri_dim})"
        )
