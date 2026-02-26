"""Covertype tabular data loader for AlphaScale.

Forest Cover Type dataset from UCI — 581,012 samples, 54 features, 7 classes.
Features: 10 continuous (elevation, slope, etc.) + 44 binary (soil/wilderness).
Labels: 1-7 shifted to 0-6 for cross-entropy compatibility.

Named load_tabular for compatibility with scaling_runner.py.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# Module-level cache — built once, reused across all scale/fraction experiments
_cache: Dict = {}

VAL_SIZE   = 10_000
TEST_SIZE  = 50_000


class CovertypeDataset(Dataset):
    """Tabular dataset returning (feature_tensor, label) pairs."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.X[idx], int(self.y[idx])


class SliceDataset(Dataset):
    """Slice pre-built tensors by index list — zero copy overhead."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor, indices: List[int]) -> None:
        self.X = X
        self.y = y
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        i = self.indices[idx]
        return self.X[i], int(self.y[i])


def _build_cache(data_path: Path, seed: int) -> None:
    """Read CSV, scale features, split, store tensors once."""

    print(f"  [Covertype] Reading covtype.data.gz...", flush=True)
    df = pd.read_csv(
        data_path / "covtype.data.gz",
        header=None,
        compression="gzip",
    )
    print(f"  [Covertype] {len(df):,} samples, {df.shape[1]-1} features, 7 classes", flush=True)

    X = df.iloc[:, :-1].values.astype(np.float32)   # 54 features
    y = df.iloc[:, -1].values.astype(np.int64) - 1  # shift 1-7 → 0-6

    # ── Train / val / test split ──────────────────────────────────────────
    # Hold out fixed test set first, then carve val from remaining train pool
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    X_pool, X_val, y_pool, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=VAL_SIZE, random_state=seed, stratify=y_trainval
    )

    # ── Standardize continuous features (first 10 columns) ───────────────
    print(f"  [Covertype] Fitting scaler on {len(X_pool):,} train pool samples...", flush=True)
    scaler = StandardScaler()
    X_pool[:, :10]  = scaler.fit_transform(X_pool[:, :10])
    X_val[:, :10]   = scaler.transform(X_val[:, :10])
    X_test[:, :10]  = scaler.transform(X_test[:, :10])

    # ── Convert to tensors ────────────────────────────────────────────────
    _cache["pool_X"]   = torch.tensor(X_pool,  dtype=torch.float32)
    _cache["pool_y"]   = torch.tensor(y_pool,  dtype=torch.long)
    _cache["pool_size"] = len(y_pool)
    _cache["val_X"]    = torch.tensor(X_val,   dtype=torch.float32)
    _cache["val_y"]    = torch.tensor(y_val,   dtype=torch.long)
    _cache["test_X"]   = torch.tensor(X_test,  dtype=torch.float32)
    _cache["test_y"]   = torch.tensor(y_test,  dtype=torch.long)
    _cache["n_features"] = X_pool.shape[1]

    print(f"  [Covertype] Cache built: {len(y_pool):,} train pool / "
          f"{len(y_val):,} val / {len(y_test):,} test", flush=True)
    print(f"  [Covertype] Subsequent calls reuse cached tensors.", flush=True)


def load_tabular(
    data_path: str,
    dataset_fraction: float = 1.0,
    batch_size: int = 256,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load Covertype and return train/val/test DataLoaders.

    First call: reads CSV + scales features (~10 seconds).
    Subsequent calls: slices pre-built tensors (milliseconds).

    Args:
        data_path: Directory containing covtype.data.gz.
        dataset_fraction: Fraction of train pool to use.
        batch_size: DataLoader batch size.
        seed: Random seed (used only on first call for splits).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    data_path = Path(data_path)

    if not _cache:
        _build_cache(data_path, seed)
    else:
        print(f"  [Covertype] Using cached tensors.", flush=True)

    pool_size = _cache["pool_size"]
    n_train   = max(1, int(dataset_fraction * pool_size))
    train_idx = list(range(n_train))
    val_idx   = list(range(len(_cache["val_y"])))
    test_idx  = list(range(len(_cache["test_y"])))

    print(f"  [Covertype] Fraction={dataset_fraction} → "
          f"{n_train:,} train / {len(val_idx):,} val / {len(test_idx):,} test", flush=True)

    train_dataset = SliceDataset(_cache["pool_X"], _cache["pool_y"], train_idx)
    val_dataset   = CovertypeDataset(_cache["val_X"],  _cache["val_y"])
    test_dataset  = CovertypeDataset(_cache["test_X"], _cache["test_y"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, _cache["n_features"]


def get_input_dim() -> int:
    """Return number of input features (54 for Covertype).
    
    Call after load_tabular has been invoked at least once.
    """
    return _cache.get("n_features", 54)
