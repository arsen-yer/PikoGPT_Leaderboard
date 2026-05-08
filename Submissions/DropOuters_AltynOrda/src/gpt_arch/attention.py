from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional import RotaryPositionalEncoding


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        n_kv_heads: int | None = None,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        max_seq_length: int = 2048,
    ):
        super().__init__()
        if embed_size % num_heads != 0:
            raise ValueError("embed_size must be divisible by num_heads")

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else num_heads
        if self.num_heads % self.n_kv_heads != 0:
            raise ValueError("num_heads must be divisible by n_kv_heads for GQA")
        self.n_rep = self.num_heads // self.n_kv_heads
        self.head_dim = embed_size // num_heads
        self.attn_dropout = attn_dropout

        self.query = nn.Linear(embed_size, embed_size, bias=qkv_bias)
        kv_dim = self.n_kv_heads * self.head_dim
        self.key = nn.Linear(embed_size, kv_dim, bias=qkv_bias)
        self.value = nn.Linear(embed_size, kv_dim, bias=qkv_bias)
        self.out = nn.Linear(embed_size, embed_size)

        self.rope = RotaryPositionalEncoding(self.head_dim, max_seq_length=max_seq_length)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        causal: bool = True,
    ) -> torch.Tensor:
        # x: (B, S, D)
        B, S, D = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # (B, S, D) -> (B, n_heads, S, head_dim) for Q and (B, n_kv_heads, S, head_dim) for K/V
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q = self.rope(q)
        k = self.rope(k)

        dropout_p = self.attn_dropout if self.training else 0.0

        # Fast path for Flash Attention kernels via SDPA:
        # avoid explicit attn_mask and use is_causal flag directly.
        can_use_flash_path = (
            padding_mask is None
            and x.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
        )
        if can_use_flash_path:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=causal,
                enable_gqa=(self.n_kv_heads != self.num_heads),
            )
        else:
            # Build attention mask: True = keep, False = mask
            attn_mask = None
            if padding_mask is not None or causal:
                keep = torch.ones((B, S, S), dtype=torch.bool, device=x.device)
                if causal:
                    causal_mask = torch.tril(
                        torch.ones((S, S), dtype=torch.bool, device=x.device)
                    )
                    keep = keep & causal_mask.unsqueeze(0)
                if padding_mask is not None:
                    keep = keep & padding_mask.unsqueeze(1)
                attn_mask = keep.unsqueeze(1)  # (B, 1, S, S)

            if self.n_kv_heads != self.num_heads:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.repeat_interleave(self.n_rep, dim=1)

            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )

        # Prevent padded query tokens from producing/propagating activations.
        if padding_mask is not None:
            y = y * padding_mask[:, None, :, None].to(dtype=y.dtype)

        y = y.transpose(1, 2).contiguous().view(B, S, D)
        y = self.out(y)
        if padding_mask is not None:
            y = y * padding_mask[:, :, None].to(dtype=y.dtype)
        return y
