"""Autoregressive text generation with greedy / sampling decoding."""
from __future__ import annotations

import torch
import torch.nn.functional as F

CONTEXT_WINDOW = 1024


def _apply_rep_penalty(logits: torch.Tensor, seq: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0 or seq.numel() == 0:
        return logits
    for tok in set(seq[0].tolist()):
        if logits[0, tok] < 0:
            logits[0, tok] *= penalty
        else:
            logits[0, tok] /= penalty
    return logits


def _blocked_by_ngram(seq: torch.Tensor, n: int) -> set[int]:
    if n <= 0 or seq.size(1) < n - 1:
        return set()
    tokens = seq[0].tolist()
    prefix = tuple(tokens[-(n - 1):])
    blocked: set[int] = set()
    for i in range(len(tokens) - n + 1):
        if tuple(tokens[i:i + n - 1]) == prefix:
            blocked.add(tokens[i + n - 1])
    return blocked


@torch.no_grad()
def complete(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: str,
    rep_penalty: float = 1.0,
    no_repeat_ngram: int = 0,
) -> str:
    """Generate a completion for *prompt*, return only the new text."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()
    kv_cache  = None

    for _ in range(max_new_tokens):
        feed = generated[:, -1:] if kv_cache is not None else generated[:, -CONTEXT_WINDOW:]
        logits, kv_cache = model(feed, causal=True, past_key_values=kv_cache, use_cache=True)

        next_logits = logits[:, -1, :].float()
        next_logits = _apply_rep_penalty(next_logits, generated, rep_penalty)

        blocked = _blocked_by_ngram(generated, no_repeat_ngram)
        if blocked:
            next_logits[:, list(blocked)] = float("-inf")

        if temperature <= 0:
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            probs   = F.softmax(next_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_id], dim=1)
        if next_id.item() == tokenizer.eos_token_id:
            break

    new_ids = generated[0][input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()
