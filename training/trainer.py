"""Domain-agnostic trainer for AlphaScale experiments."""

import time
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.metrics import compute_epoch_metrics


class Trainer:
    """Trains a PyTorch model for a fixed number of epochs.

    Supports both standard (image/tabular) and transformer-style (dict-input)
    forward passes via a configurable input extractor.

    Args:
        model: PyTorch model to train.
        optimizer: Configured optimizer.
        criterion: Loss function (reduction='mean').
        device: Torch device for training.
        domain: One of 'vision', 'nlp', 'tabular'.
        epochs: Number of training epochs.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        domain: str,
        epochs: int,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.domain = domain
        self.epochs = epochs

    def _move_batch(self, batch: Any) -> Tuple[Any, torch.Tensor]:
        """Move a batch to the training device.

        Handles both tensor batches (vision/tabular) and dict batches (nlp).

        Args:
            batch: Raw batch from DataLoader.

        Returns:
            Tuple of (inputs, targets) on device.
        """
        if self.domain == "nlp":
            inputs, targets = batch
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            targets = targets.to(self.device)
        else:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
        return inputs, targets

    def _forward(self, inputs: Any) -> torch.Tensor:
        """Run a forward pass appropriate for the domain.

        Args:
            inputs: Batch inputs (tensor or dict).

        Returns:
            Logits tensor.
        """
        if self.domain == "nlp":
            return self.model(inputs)
        return self.model(inputs)

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run one training epoch.

        Args:
            loader: Training DataLoader.

        Returns:
            Dict with 'loss' and 'accuracy'.
        """
        self.model.train()
        logits_list, targets_list, loss_list = [], [], []

        for batch in loader:
            inputs, targets = self._move_batch(batch)
            self.optimizer.zero_grad()
            logits = self._forward(inputs)
            loss = self.criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            logits_list.append(logits.detach().cpu())
            targets_list.append(targets.cpu())
            loss_list.append(loss.item())

        return compute_epoch_metrics(logits_list, targets_list, loss_list)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a DataLoader.

        Args:
            loader: Validation or test DataLoader.

        Returns:
            Dict with 'loss' and 'accuracy'.
        """
        self.model.eval()
        logits_list, targets_list, loss_list = [], [], []

        for batch in loader:
            inputs, targets = self._move_batch(batch)
            logits = self._forward(inputs)
            loss = self.criterion(logits, targets)

            logits_list.append(logits.cpu())
            targets_list.append(targets.cpu())
            loss_list.append(loss.item())

        return compute_epoch_metrics(logits_list, targets_list, loss_list)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train for the configured number of epochs.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            verbose: Print per-epoch metrics.

        Returns:
            Training summary dict with keys:
            - 'train_time': total seconds
            - 'best_val_accuracy': best validation accuracy across epochs
            - 'final_train_accuracy': training accuracy at last epoch
            - 'epoch_logs': list of per-epoch metric dicts
        """
        best_val_acc = 0.0
        epoch_logs = []
        t0 = time.time()

        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            log = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
            epoch_logs.append(log)

            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]

            if verbose:
                print(
                    f"  Epoch {epoch:02d}/{self.epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Train Acc: {train_metrics['accuracy']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )

        train_time = time.time() - t0
        final_train_acc = epoch_logs[-1]["train_accuracy"]

        return {
            "train_time": train_time,
            "best_val_accuracy": best_val_acc,
            "final_train_accuracy": final_train_acc,
            "epoch_logs": epoch_logs,
        }


def build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Construct an optimizer for the model.

    Args:
        model: PyTorch model whose parameters to optimize.
        optimizer_name: 'adam' or 'sgd'.
        learning_rate: Learning rate.
        weight_decay: L2 regularization coefficient.

    Returns:
        Configured optimizer instance.

    Raises:
        ValueError: If optimizer_name is not supported.
    """
    if optimizer_name.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
