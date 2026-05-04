"""Shared inference utilities: forward pass, tokenization, and text generation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def forward_logits(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    except TypeError:
        try:
            outputs = model(
                input_ids=input_ids,
                padding_mask=attention_mask.bool(),
                causal=True,
            )
        except TypeError:
            outputs = model(input_ids, padding_mask=attention_mask.bool(), causal=True)
    return outputs.logits if hasattr(outputs, "logits") else outputs


def encode_text(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return list(input_ids)


def decode_ids(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(token_ids)


@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer: Any,
    prompt_ids: list[int],
    device: torch.device,
    max_context: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> list[int]:
    """Generate completion tokens given prompt token ids.

    Returns only the newly generated token ids (not the prompt).
    Caller is responsible for setting model.eval() / model.train().
    """
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    generated = list(prompt_ids)

    for _ in range(max_new_tokens):
        window = generated[-max_context:]
        input_ids = torch.tensor([window], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        next_token_logits = forward_logits(model, input_ids, attention_mask)[0, -1, :].clone()

        if repetition_penalty != 1.0:
            for token_id in set(generated):
                if next_token_logits[token_id] < 0:
                    next_token_logits[token_id] *= repetition_penalty
                else:
                    next_token_logits[token_id] /= repetition_penalty

        if temperature <= 0:
            next_id = int(next_token_logits.argmax().item())
        else:
            next_token_logits = next_token_logits / temperature
            if top_k > 0:
                k = min(top_k, next_token_logits.size(-1))
                values, _ = torch.topk(next_token_logits, k)
                next_token_logits[next_token_logits < values[-1]] = -float("inf")
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p
                next_token_logits[sorted_indices[sorted_indices_to_remove]] = -float("inf")
            probs = torch.softmax(next_token_logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

        generated.append(next_id)
        if eos_token_id is not None and next_id == int(eos_token_id):
            break

    return generated[len(prompt_ids):]