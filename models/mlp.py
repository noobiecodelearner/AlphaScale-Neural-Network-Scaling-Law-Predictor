"""Scalable MLP for tabular classification in AlphaScale."""

from typing import List, Tuple

import torch
import torch.nn as nn


class ScalableMLP(nn.Module):
    """Depth-and-width scalable MLP for tabular classification.

    The network has a fixed 3-layer architecture where each hidden layer has
    the same hidden_size. Scaling is achieved by varying hidden_size.

    Args:
        input_dim: Number of input features.
        num_classes: Number of output classes.
        hidden_size: Number of units in each hidden layer.
        num_hidden_layers: Number of hidden layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_size: int = 128,
        num_hidden_layers: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        layers: List[nn.Module] = []

        # Input → first hidden
        layers += [
            nn.Linear(input_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ]

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]

        # Output
        layers.append(nn.Linear(hidden_size, num_classes))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_flops(self) -> int:
        """Estimate FLOPs for a single forward pass (MACs * 2).

        Returns:
            Approximate FLOPs count.
        """
        flops = 2 * self.input_dim * self.hidden_size  # input layer
        # Each hidden-to-hidden layer
        for _ in range(2):
            flops += 2 * self.hidden_size * self.hidden_size
        flops += 2 * self.hidden_size * self.num_classes  # output layer
        return flops
