"""Scalable CNN for CIFAR-10 in AlphaScale."""

import math
from typing import Tuple

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolution → BatchNorm → ReLU block.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ScalableCNN(nn.Module):
    """Width-scalable CNN for CIFAR-10 classification.

    Architecture: Three conv stages, each with two conv blocks and pooling,
    followed by an adaptive average pool and a linear classifier.

    The number of base channels is 64. A width multiplier scales all channel
    counts proportionally.

    Args:
        num_classes: Number of output classes.
        width_multiplier: Float multiplier applied to base channel widths.
        in_channels: Number of input image channels.
    """

    BASE_CHANNELS = [64, 128, 256]

    def __init__(
        self,
        num_classes: int = 10,
        width_multiplier: float = 1.0,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.width_multiplier = width_multiplier

        ch = [max(1, int(c * width_multiplier)) for c in self.BASE_CHANNELS]

        self.stage1 = nn.Sequential(
            ConvBlock(in_channels, ch[0]),
            ConvBlock(ch[0], ch[0]),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(ch[0], ch[1]),
            ConvBlock(ch[1], ch[1]),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(ch[1], ch[2]),
            ConvBlock(ch[2], ch[2]),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch[2], max(1, ch[2] // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(max(1, ch[2] // 2), num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return self.classifier(x)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_flops(self, input_size: Tuple[int, int, int] = (3, 32, 32)) -> int:
        """Estimate FLOPs for a single forward pass via a dummy forward hook.

        Args:
            input_size: (C, H, W) input dimensions.

        Returns:
            Approximate FLOPs count.
        """
        flops = [0]

        def conv_hook(module, inp, out):
            if isinstance(module, nn.Conv2d):
                _, c_in, h, w = inp[0].shape
                _, c_out, h_out, w_out = out.shape
                flops[0] += 2 * c_in * module.kernel_size[0] * module.kernel_size[1] * c_out * h_out * w_out

        def linear_hook(module, inp, out):
            if isinstance(module, nn.Linear):
                flops[0] += 2 * module.in_features * module.out_features

        hooks = []
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                hooks.append(m.register_forward_hook(conv_hook))
            elif isinstance(m, nn.Linear):
                hooks.append(m.register_forward_hook(linear_hook))

        device = next(self.parameters()).device
        dummy = torch.zeros(1, *input_size, device=device)
        with torch.no_grad():
            self(dummy)

        for h in hooks:
            h.remove()

        return flops[0]
