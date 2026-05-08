from __future__ import annotations

import torch
import torch.nn as nn


class RotaryPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoPE) — Su et al. 2021 (RoFormer).

    Precomputes cos/sin frequency tables up to max_seq_length.
    Applied inside attention to Q and K vectors, not to the token embeddings.
    No learnable parameters.
    """

    def __init__(self, head_dim: int, max_seq_length: int = 2048, base: int = 10000):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        # Frequency bands: one per pair of head_dim dimensions
        # theta_i = 1 / (base ^ (2i / head_dim))
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin tables for all positions up to max_seq_length
        self._build_cache(max_seq_length)

    def _build_cache(self, seq_len: int) -> None:
        positions = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(positions, self.inv_freq)       # (seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)             # (seq_len, head_dim)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the last dimension into the first half."""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to a tensor of shape (B, n_heads, S, head_dim)."""
        seq_len = x.size(2)
        if seq_len > self.cos_cache.size(0):
            self._build_cache(seq_len)
        cos = self.cos_cache[:seq_len].to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:seq_len].to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        return x * cos + self._rotate_half(x) * sin
