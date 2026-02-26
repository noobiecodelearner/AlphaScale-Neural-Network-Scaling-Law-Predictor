"""UCI Adult data loader for AlphaScale — loads from local files."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset


COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

CATEGORICAL_COLS = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country"
]

NUMERICAL_COLS = [
    "age", "fnlwgt", "education_num", "capital_gain",
    "capital_loss", "hours_per_week"
]


class AdultDataset(Dataset):
    """UCI Adult dataset as a PyTorch Dataset.

    Args:
        features: Float32 tensor of shape (N, input_dim).
        labels: Long tensor of shape (N,).
    """

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def _read_adult_file(filepath: str, skip_header: bool = False) -> pd.DataFrame:
    """Read an adult.data or adult.test file into a DataFrame.

    Args:
        filepath: Path to the data file.
        skip_header: Whether to skip the first row (adult.test has a header line).

    Returns:
        Cleaned DataFrame.
    """
    df = pd.read_csv(
        filepath,
        names=COLUMN_NAMES,
        skipinitialspace=True,
        na_values="?",
        skiprows=1 if skip_header else 0,
    )
    return df


def _preprocess(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess Adult data: encode, impute, normalize.

    Args:
        train_df: Raw training DataFrame.
        test_df: Raw test DataFrame.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test) as numpy float32 arrays.
    """
    # Clean income labels (test set has trailing periods)
    train_df["income"] = train_df["income"].str.strip().str.replace(".", "", regex=False)
    test_df["income"] = test_df["income"].str.strip().str.replace(".", "", regex=False)

    # Drop rows with NaN
    train_df = train_df.dropna().reset_index(drop=True)
    test_df = test_df.dropna().reset_index(drop=True)

    # Encode labels
    y_train = (train_df["income"] == ">50K").astype(np.int64).values
    y_test = (test_df["income"] == ">50K").astype(np.int64).values

    # Drop label column
    train_df = train_df.drop(columns=["income"])
    test_df = test_df.drop(columns=["income"])

    # One-hot encode categoricals — fit on train, apply to both
    combined = pd.concat([train_df, test_df], axis=0)
    combined = pd.get_dummies(combined, columns=CATEGORICAL_COLS)
    n_train = len(train_df)

    train_enc = combined.iloc[:n_train].values.astype(np.float32)
    test_enc = combined.iloc[n_train:].values.astype(np.float32)

    # Normalize numerical columns by train stats
    num_col_indices = [
        list(combined.columns).index(c) for c in NUMERICAL_COLS
        if c in combined.columns
    ]
    means = train_enc[:, num_col_indices].mean(axis=0)
    stds = train_enc[:, num_col_indices].std(axis=0) + 1e-8
    train_enc[:, num_col_indices] = (train_enc[:, num_col_indices] - means) / stds
    test_enc[:, num_col_indices] = (test_enc[:, num_col_indices] - means) / stds

    return train_enc, y_train, test_enc, y_test


def load_adult(
    data_path: str,
    dataset_fraction: float = 1.0,
    batch_size: int = 256,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Load UCI Adult dataset from local files.

    Args:
        data_path: Directory containing adult.data and adult.test.
        dataset_fraction: Fraction of training data to use.
        batch_size: DataLoader batch size.
        seed: Random seed for subset sampling.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, input_dim).
    """
    data_path = Path(data_path)
    train_df = _read_adult_file(str(data_path / "adult.data"), skip_header=False)
    test_df = _read_adult_file(str(data_path / "adult.test"), skip_header=True)

    X_train, y_train, X_test, y_test = _preprocess(train_df, test_df)
    input_dim = X_train.shape[1]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    full_train = AdultDataset(X_train_t, y_train_t)
    full_test = AdultDataset(X_test_t, y_test_t)

    # Val split: 10%
    rng = np.random.RandomState(seed)
    n_total = len(full_train)
    all_idx = rng.permutation(n_total)
    n_val = max(1, int(0.1 * n_total))
    val_idx = all_idx[:n_val].tolist()
    train_pool_idx = all_idx[n_val:]

    # Apply fraction
    n_train = max(1, int(dataset_fraction * len(train_pool_idx)))
    selected_train_idx = train_pool_idx[:n_train].tolist()

    train_subset = Subset(full_train, selected_train_idx)
    val_subset = Subset(full_train, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(full_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, input_dim