"""Scalable Transformer for AG News classification in AlphaScale."""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ScalableTransformer(nn.Module):
    """Scalable Transformer encoder for text classification.

    The architecture consists of a token embedding, positional encoding,
    a stack of Transformer encoder layers, mean pooling, and a classifier head.

    Args:
        vocab_size: Vocabulary size.
        num_classes: Number of output classes.
        num_layers: Number of Transformer encoder layers.
        d_model: Embedding and model dimension.
        nhead: Number of attention heads (auto-derived from d_model).
        dim_feedforward: Feedforward layer dimension (defaults to 4 * d_model).
        max_seq_len: Maximum token sequence length.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        num_classes: int = 4,
        num_layers: int = 4,
        d_model: int = 256,
        nhead: int = None,
        dim_feedforward: int = None,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Auto-derive nhead: largest power-of-2 divisor of d_model, ≤ 8
        if nhead is None:
            nhead = 1
            for h in [8, 4, 2, 1]:
                if d_model % h == 0:
                    nhead = h
                    break

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=self.d_model ** -0.5)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            batch: Dict with keys 'input_ids' and 'attention_mask'.

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # TransformerEncoder expects key_padding_mask where True = ignore
        key_padding_mask = attention_mask == 0

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # Mean pooling over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return self.classifier(x)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_flops(self, seq_len: int = 128) -> int:
        """Estimate FLOPs for a single forward pass.

        Approximation: 2 * seq_len * d_model * total_params_in_encoder.

        Args:
            seq_len: Sequence length.

        Returns:
            Approximate FLOPs count.
        """
        # Attention per layer:
        #   QKV projections: 6 * seq_len * d^2
        #   Attention scores + weighted sum: 4 * seq_len^2 * d
        # FFN per layer:
        #   Two linear layers (d → 4d → d): 2 * seq_len * d * 4d * 2
        d = self.d_model
        attn_flops = (6 * seq_len * d * d) + (4 * seq_len * seq_len * d)
        ffn_flops = 2 * seq_len * d * 4 * d * 2
        flops_per_layer = attn_flops + ffn_flops
        return flops_per_layer * self.num_layers
