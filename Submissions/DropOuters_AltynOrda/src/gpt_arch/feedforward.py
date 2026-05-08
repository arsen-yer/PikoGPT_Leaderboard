from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """SwiGLU FFN: x -> (W1(x) * SiLU(W3(x))) -> W2
    Three bias-free projections; hidden size should be set to (2/3 * 4 * embed_size)
    rounded to a multiple of 64 to maintain parameter parity with a standard 4x GELU MLP.
    """

    def __init__(self, embed_size: int, ff_hidden_size: int):
        super().__init__()
        self.w1 = nn.Linear(embed_size, ff_hidden_size, bias=False)
        self.w2 = nn.Linear(ff_hidden_size, embed_size, bias=False)
        self.w3 = nn.Linear(embed_size, ff_hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))