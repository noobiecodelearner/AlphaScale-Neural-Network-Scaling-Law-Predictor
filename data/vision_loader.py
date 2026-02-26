"""CIFAR-10 data loader for AlphaScale — loads from local disk."""

import os
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms


class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset loaded from local pickle files.

    Args:
        data: Numpy array of images (N, 3, 32, 32) as float32.
        labels: Numpy array of integer labels.
        transform: Optional torchvision transform pipeline.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        transform=None,
    ) -> None:
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.data[idx]  # (3, 32, 32) float32
        # Convert to HWC uint8 for PIL-compatible transforms
        img_hwc = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
        label = int(self.labels[idx])

        if self.transform:
            from PIL import Image
            img_pil = Image.fromarray(img_hwc)
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = torch.tensor(img, dtype=torch.float32)

        return img_tensor, label


def _load_batch(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single CIFAR-10 batch pickle file.

    Args:
        filepath: Path to the pickle file.

    Returns:
        Tuple of (images, labels) as numpy arrays.
    """
    with open(filepath, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    images = batch[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.array(batch[b"labels"])
    return images, labels


def load_cifar10(
    data_path: str,
    dataset_fraction: float = 1.0,
    batch_size: int = 128,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load CIFAR-10 from local disk and return train/val/test DataLoaders.

    Args:
        data_path: Path to the cifar-10-batches-py directory.
        dataset_fraction: Fraction of training data to use (0 < f <= 1.0).
        batch_size: DataLoader batch size.
        seed: Random seed for subset sampling.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    data_path = Path(data_path)

    train_images_list, train_labels_list = [], []
    for i in range(1, 6):
        batch_path = data_path / f"data_batch_{i}"
        imgs, lbls = _load_batch(str(batch_path))
        train_images_list.append(imgs)
        train_labels_list.append(lbls)

    train_images = np.concatenate(train_images_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)

    test_images, test_labels = _load_batch(str(data_path / "test_batch"))

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    full_train = CIFAR10Dataset(train_images, train_labels, transform=train_transform)
    full_test = CIFAR10Dataset(test_images, test_labels, transform=eval_transform)

    # Val split: 10% of full training set (before fraction sampling)
    rng = np.random.RandomState(seed)
    n_total = len(full_train)
    all_idx = rng.permutation(n_total)
    n_val = max(1, int(0.1 * n_total))
    val_idx = all_idx[:n_val]
    train_pool_idx = all_idx[n_val:]

    # Apply dataset fraction to training pool
    n_train = max(1, int(dataset_fraction * len(train_pool_idx)))
    selected_train_idx = train_pool_idx[:n_train]

    train_subset = Subset(full_train, selected_train_idx.tolist())
    val_subset = Subset(
        CIFAR10Dataset(train_images, train_labels, transform=eval_transform),
        val_idx.tolist(),
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(full_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader
