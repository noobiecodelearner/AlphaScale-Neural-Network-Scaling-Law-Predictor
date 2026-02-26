"""Evaluation metrics for AlphaScale training."""

from typing import Dict

import torch


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy.

    Args:
        logits: Model output tensor of shape (N, C).
        targets: Ground truth labels of shape (N,).

    Returns:
        Accuracy as a float in [0, 1].
    """
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).sum().item()
    return correct / len(targets)


def compute_epoch_metrics(
    logits_list: list,
    targets_list: list,
    loss_list: list,
) -> Dict[str, float]:
    """Aggregate per-batch logits, targets, and losses into epoch metrics.

    Args:
        logits_list: List of logit tensors per batch.
        targets_list: List of target tensors per batch.
        loss_list: List of scalar loss values per batch.

    Returns:
        Dictionary with keys 'loss' and 'accuracy'.
    """
    all_logits = torch.cat(logits_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)
    avg_loss = sum(loss_list) / len(loss_list)
    acc = compute_accuracy(all_logits, all_targets)
    return {"loss": avg_loss, "accuracy": acc}
