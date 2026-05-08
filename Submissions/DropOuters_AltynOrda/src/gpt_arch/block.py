from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        ff_hidden_size: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.1,
        qkv_bias: bool = False,
        max_seq_length: int = 2048,
    ):
        super().__init__()
        self.ln1 = nn.RMSNorm(embed_size)
        self.mha = MultiHeadAttention(
            embed_size,
            num_heads,
            n_kv_heads=n_kv_heads,
            qkv_bias=qkv_bias,
            max_seq_length=max_seq_length,
        )

        self.ln2 = nn.RMSNorm(embed_size)
        self.ff = FeedForward(embed_size, ff_hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        causal: bool = True,
    ) -> torch.Tensor:
        x = x + self.dropout(self.mha(self.ln1(x), padding_mask=padding_mask, causal=causal))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x
