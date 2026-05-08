from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .block import TransformerBlock
from .embedding import EmbeddingLayer


class GPTTiny(nn.Module):
    """A small, general GPT-style language model.

    Flow:
      token ids -> token embeddings -> dropout -> transformer blocks (RoPE inside attn) -> final LN -> vocab logits
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        emb_dim: int,
        mlp_ratio: int,
        n_heads: int,
        n_decoders: int,
        drop_rate: float,
        qkv_bias: bool,
        n_kv_heads: int | None = None,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if emb_dim % n_heads != 0:
            raise ValueError("emb_dim must be divisible by n_heads")

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.embedding = EmbeddingLayer(vocab_size, emb_dim)
        self.dropout = nn.Dropout(drop_rate)

        # SwiGLU uses 3 projections vs 2 for GELU. To keep total FFN params equal:
        # 2 * (emb_dim * 4*emb_dim) = 3 * (emb_dim * ff_hidden_size)  →  ff = 8/3 * emb_dim
        # Round to nearest multiple of 64 for memory alignment.
        ff_hidden_size = round(mlp_ratio * emb_dim * 2 / 3 / 64) * 64
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_size=emb_dim,
                num_heads=n_heads,
                ff_hidden_size=ff_hidden_size,
                n_kv_heads=n_kv_heads,
                dropout=drop_rate,
                qkv_bias=qkv_bias,
                max_seq_length=context_length,
            )
            for _ in range(n_decoders)
        ])

        self.ln_f = nn.RMSNorm(emb_dim)
        self.fc_out = nn.Linear(emb_dim, vocab_size, bias=False)
        self.fc_out.weight = self.embedding.embedding.weight  # weight tying

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        causal: bool = True,
        use_cache: bool = False,
        past_key_values=None,   # accepted but ignored — no KV cache implemented
    ) -> torch.Tensor | tuple:
        x = self.embedding(input_ids)
        x = self.dropout(x)

        for block in self.transformer_blocks:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(block, x, padding_mask, causal, use_reentrant=False)
            else:
                x = block(x, padding_mask=padding_mask, causal=causal)

        x = self.ln_f(x)
        logits = self.fc_out(x)
        # Always return tuple when use_cache=True: (logits, past_key_values)
        # past_key_values is None since we don't implement KV cache — generate()
        # will keep passing the full sequence each step which is correct but slower.
        return (logits, None) if use_cache else logits
