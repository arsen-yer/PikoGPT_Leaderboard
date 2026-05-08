"""
MC answer selection via calibrated likelihood scoring.

Four ranking strategies, applied in priority order:

  blank_fill    — WinoGrande cloze: insert each option into the _ slot and
                  rank the resulting full-sentence log-probability.
  swapped_pair  — Binary debiasing: score each choice in both slot positions
                  (A then B, B then A) to remove presentation-order bias;
                  subtract a length-normalised unconditional prior.
  continuation  — HellaSwag: rank each ending as a natural continuation of
                  the context stem, debiased against a neutral "The …" prefix.
  answer_text   — OpenBookQA / ARC / MMLU / SciQ: rank each answer phrase
                  after "Question: … Answer:", debiased against the standalone
                  "Answer:" prior.
  letter_logit  — Fallback for any other format: calibrated next-token log P
                  for each letter label, subtracting the unconditional prior.
"""
from __future__ import annotations

import re

import torch
import torch.nn.functional as F

CONTEXT_WINDOW = 1024


# ── helpers ───────────────────────────────────────────────────────────────────

def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _parse_options(prompt: str) -> dict[str, str]:
    pattern = re.compile(r"\n([A-E])\)\s*(.*?)(?=\n[A-E]\)|\nAnswer:|$)", re.DOTALL)
    return {m.group(1): m.group(2).strip() for m in pattern.finditer(prompt)}


def _seq_score(
    model,
    tokenizer,
    prefix: str,
    continuation: str,
    device: str,
    *,
    normalise: bool,
) -> float:
    """Sum (or mean) of log P for each continuation token given prefix."""
    pfx_ids  = _encode(tokenizer, prefix)
    full_ids = _encode(tokenizer, prefix + " " + continuation.strip())
    n_cont   = len(full_ids) - len(pfx_ids)
    if n_cont <= 0:
        return float("-inf")

    if len(full_ids) > CONTEXT_WINDOW:
        full_ids = full_ids[-CONTEXT_WINDOW:]
        pfx_len  = max(0, CONTEXT_WINDOW - n_cont)
    else:
        pfx_len = len(pfx_ids)

    t      = torch.tensor([full_ids], dtype=torch.long, device=device)
    out, _ = model(t, causal=True, use_cache=True)
    lp     = F.log_softmax(out[0].float(), dim=-1)

    total, n = 0.0, 0
    for i in range(max(0, pfx_len - 1), len(full_ids) - 1):
        total += lp[i, full_ids[i + 1]].item()
        n += 1

    if n == 0:
        return float("-inf")
    return total / n if normalise else total


# ── ranking strategies ────────────────────────────────────────────────────────

@torch.no_grad()
def rank_by_letter_logit(model, tokenizer, prompt: str, device: str) -> str:
    """Calibrated single-token scoring: log P(L|ctx) − log P(L|'Answer:')."""
    ctx_ids   = tokenizer.encode(prompt,    return_tensors="pt").to(device)[:, -CONTEXT_WINDOW:]
    prior_ids = tokenizer.encode("Answer:", return_tensors="pt").to(device)

    ctx_out,   _ = model(ctx_ids,   causal=True, use_cache=True)
    prior_out, _ = model(prior_ids, causal=True, use_cache=True)

    ctx_lp   = F.log_softmax(ctx_out[0, -1].float(),   dim=-1)
    prior_lp = F.log_softmax(prior_out[0, -1].float(), dim=-1)

    options = _parse_options(prompt)
    labels  = list(options.keys()) if options else (
        list("ABCD") if re.search(r"\n[CD]\)", prompt) else list("AB")
    )

    best, top = labels[0], float("-inf")
    for lbl in labels:
        for variant in (f" {lbl}", lbl):
            tids = _encode(tokenizer, variant)
            if len(tids) == 1:
                score = (ctx_lp[tids[0]] - prior_lp[tids[0]]).item()
                if score > top:
                    top, best = score, lbl
    return best


@torch.no_grad()
def rank_by_blank_fill(model, tokenizer, prompt: str, device: str) -> str | None:
    """WinoGrande: fill _ with each option, rank full-sentence likelihood."""
    options = _parse_options(prompt)
    if set(options) != {"A", "B"}:
        return None

    boundary = re.search(r"\n[AB]\)", prompt)
    if boundary is None:
        return None

    stem = prompt[:boundary.start()]
    if stem.startswith("Context:"):
        stem = stem[len("Context:"):].strip()
    if "_" not in stem:
        return None

    scores: dict[str, float] = {}
    for lbl, text in options.items():
        filled = stem.replace("_", text.strip(), 1)
        scores[lbl] = _seq_score(model, tokenizer, "", filled, device, normalise=True)

    return max(scores, key=scores.get) if scores else None


@torch.no_grad()
def rank_by_swapped_pair(model, tokenizer, prompt: str, device: str) -> str | None:
    """Binary debiasing: test each choice in both A and B positions."""
    options = _parse_options(prompt)
    if set(options) != {"A", "B"}:
        return None

    boundary = re.search(r"\n[AB]\)", prompt)
    answer   = re.search(r"\nAnswer:\s*$", prompt)
    if boundary is None or answer is None:
        return None

    stem    = prompt[:boundary.start()]
    swapped = stem + f"\nA) {options['B']}\nB) {options['A']}"
    W       = 3.0

    def _slot(ctx, slot, txt):
        return _seq_score(model, tokenizer, ctx + f"\n{slot})", txt, device, normalise=False)

    sa = _slot(stem, "A", options["A"]) + _slot(swapped, "B", options["A"])
    sb = _slot(stem, "B", options["B"]) + _slot(swapped, "A", options["B"])

    pa = _seq_score(model, tokenizer, "Answer:", options["A"], device, normalise=False)
    pb = _seq_score(model, tokenizer, "Answer:", options["B"], device, normalise=False)
    if pa != float("-inf"):
        sa -= W * pa
    if pb != float("-inf"):
        sb -= W * pb

    return "A" if sa >= sb else "B"


@torch.no_grad()
def rank_by_continuation(model, tokenizer, prompt: str, device: str) -> str | None:
    """HellaSwag: rank choice texts as continuations, debiased vs 'The …'."""
    options = _parse_options(prompt)
    if len(options) != 4 or not prompt.lstrip().startswith("Context:"):
        return None

    boundary = re.search(r"\n[A-D]\)", prompt)
    if boundary is None:
        return None

    stem = prompt[:boundary.start()]
    if stem.startswith("Context:"):
        stem = stem[len("Context:"):].strip()

    BIAS_W = 0.5
    scores: dict[str, float] = {}
    for lbl, text in options.items():
        text = text.strip()
        if not text:
            continue
        s = _seq_score(model, tokenizer, stem,  text, device, normalise=True)
        b = _seq_score(model, tokenizer, "The", text, device, normalise=True)
        scores[lbl] = s - (BIAS_W * b if b != float("-inf") else 0.0)

    return max(scores, key=scores.get) if scores else None


@torch.no_grad()
def rank_by_answer_text(model, tokenizer, prompt: str, device: str) -> str | None:
    """OpenBookQA / ARC / MMLU: rank answer phrases after question + 'Answer:'."""
    options = _parse_options(prompt)
    if len(options) not in (4, 5) or not prompt.lstrip().startswith("Question:"):
        return None

    boundary = re.search(r"\n[A-E]\)", prompt)
    if boundary is None:
        return None

    question = prompt[:boundary.start()]
    BIAS_W   = 1.5

    scores: dict[str, float] = {}
    for lbl, text in options.items():
        text = text.strip()
        if not text:
            continue
        s = _seq_score(model, tokenizer, question + "\nAnswer:", text, device, normalise=False)
        b = _seq_score(model, tokenizer, "Answer:",              text, device, normalise=False)
        scores[lbl] = s - (BIAS_W * b if b != float("-inf") else 0.0)

    return max(scores, key=scores.get) if scores else None


# ── main dispatch ─────────────────────────────────────────────────────────────

def select_answer(model, tokenizer, prompt: str, device: str) -> str:
    """Choose the best scoring strategy for this prompt and return a letter."""
    options = _parse_options(prompt)
    n = len(options)

    if n == 2:
        result = rank_by_blank_fill(model, tokenizer, prompt, device)
        if result is not None:
            return result
        result = rank_by_swapped_pair(model, tokenizer, prompt, device)
        if result is not None:
            return result

    if n == 4:
        result = rank_by_continuation(model, tokenizer, prompt, device)
        if result is not None:
            return result
        result = rank_by_answer_text(model, tokenizer, prompt, device)
        if result is not None:
            return result

    if n == 5:
        result = rank_by_answer_text(model, tokenizer, prompt, device)
        if result is not None:
            return result

    return rank_by_letter_logit(model, tokenizer, prompt, device)
