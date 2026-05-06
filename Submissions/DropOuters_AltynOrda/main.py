#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))
from inference import encode_text, forward_logits, decode_ids

ALPACA_PREFIX = "### Instruction:\n\n\n### Input:\n{context}\n\n### Response:\n"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class GQAAttention(nn.Module):
    def __init__(self, dim: int, q_dim: int, kv_dim: int, head_dim: int) -> None:
        super().__init__()
        self.n_q = q_dim // head_dim
        self.n_kv = kv_dim // head_dim
        if self.n_q % self.n_kv != 0:
            raise ValueError("Invalid GQA shapes")
        self.kv_repeat = self.n_q // self.n_kv
        self.head_dim = head_dim
        self.q_dim = q_dim

        self.query = nn.Linear(dim, q_dim, bias=False)
        self.key = nn.Linear(dim, kv_dim, bias=False)
        self.value = nn.Linear(dim, kv_dim, bias=False)
        self.out = nn.Linear(q_dim, dim, bias=True)

    def _rope(self, t: int, d: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        theta = 1.0 / (10000 ** (torch.arange(0, d, 2, device=device, dtype=torch.float32) / d))
        pos = torch.arange(t, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, theta)
        emb = torch.cat((freqs, freqs), dim=-1).to(dtype)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.query(x).view(b, t, self.n_q, self.head_dim).transpose(1, 2)
        k = self.key(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.value(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)

        cos, sin = self._rope(t, self.head_dim, x.device, q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.kv_repeat > 1:
            k = k.repeat_interleave(self.kv_repeat, dim=1)
            v = v.repeat_interleave(self.kv_repeat, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, self.q_dim)
        return self.out(y)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, dim: int, q_dim: int, kv_dim: int, ff_hidden: int, head_dim: int) -> None:
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.mha = GQAAttention(dim, q_dim, kv_dim, head_dim)
        self.ln2 = RMSNorm(dim)
        self.ff = SwiGLU(dim, ff_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mha(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


@dataclass
class ModelConfig:
    vocab_size: int
    dim: int
    n_layers: int
    q_dim: int
    kv_dim: int
    ff_hidden: int
    max_length: int = 384


class DropOutersLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        head_dim = self._infer_head_dim(cfg.q_dim, cfg.kv_dim)
        self.transformer_blocks = nn.ModuleList([
            Block(cfg.dim, cfg.q_dim, cfg.kv_dim, cfg.ff_hidden, head_dim) for _ in range(cfg.n_layers)
        ])
        self.ln_f = RMSNorm(cfg.dim)
        self.fc_out = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    @staticmethod
    def _infer_head_dim(q_dim: int, kv_dim: int) -> int:
        for d in (64, 32, 128):
            if q_dim % d == 0 and kv_dim % d == 0:
                return d
        for d in range(16, 129):
            if q_dim % d == 0 and kv_dim % d == 0:
                return d
        raise ValueError("Cannot infer head dim")

    def forward(
        self,
        idx: Optional[torch.Tensor] = None,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        causal: Optional[bool] = None,
    ) -> torch.Tensor:
        del attention_mask, padding_mask, causal
        if input_ids is not None:
            idx = input_ids
        if idx is None:
            raise ValueError("idx or input_ids must be provided")
        x = self.embedding(idx)
        for blk in self.transformer_blocks:
            x = blk(x)
        return self.fc_out(self.ln_f(x))


def build_model_from_state_dict(sd: dict[str, torch.Tensor]) -> DropOutersLM:
    n_layers = len({int(k.split(".")[1]) for k in sd if k.startswith("transformer_blocks.")})
    cfg = ModelConfig(
        vocab_size=sd["embedding.weight"].shape[0] if "embedding.weight" in sd else sd["embedding.embedding.weight"].shape[0],
        dim=sd["embedding.weight"].shape[1] if "embedding.weight" in sd else sd["embedding.embedding.weight"].shape[1],
        n_layers=n_layers,
        q_dim=sd["transformer_blocks.0.mha.query.weight"].shape[0],
        kv_dim=sd["transformer_blocks.0.mha.key.weight"].shape[0],
        ff_hidden=sd["transformer_blocks.0.ff.w1.weight"].shape[0],
    )
    model = DropOutersLM(cfg)
    if "embedding.embedding.weight" in sd:
        model_sd = model.state_dict()
        model_sd["embedding.weight"] = sd["embedding.embedding.weight"]
        for k, v in sd.items():
            if k == "embedding.embedding.weight":
                continue
            model_sd[k] = v
        model.load_state_dict(model_sd, strict=True)
    else:
        model.load_state_dict(sd, strict=True)
    return model


def load_checkpoint(path: Path, device: torch.device) -> DropOutersLM:
    raw = torch.load(path, map_location=device)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        sd = raw["model_state_dict"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
    elif isinstance(raw, dict) and all(isinstance(v, torch.Tensor) for v in raw.values()):
        sd = raw
    else:
        raise ValueError("Unsupported checkpoint format")
    model = build_model_from_state_dict(sd)
    model.to(device).eval()
    return model


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer():
    from transformers import GPT2TokenizerFast
    return GPT2TokenizerFast.from_pretrained("gpt2")


def parse_mc_prompt(prompt: str) -> tuple[Optional[str], Optional[str], Optional[dict[str, str]]]:
    if "Answer:" not in prompt:
        return None, None, None
    option_re = re.compile(r"\n([A-D])\) (.+?)(?=\n[A-D]\) |\nAnswer:)", re.DOTALL)
    options: dict[str, str] = {m.group(1): m.group(2).strip() for m in option_re.finditer(prompt)}
    if not options:
        return None, None, None
    first_letter = sorted(options.keys())[0]
    context = prompt[:prompt.index(f"\n{first_letter}) ")].strip()
    context = re.sub(r"^(Context|Question):\s*", "", context)
    suffix = ""
    if "_" in context:
        before, after = context.split("_", 1)
        context = before.rstrip()
        suffix = after
    return context, suffix, options


def avg_logprob_of_continuation(model: DropOutersLM, prefix_ids: list[int], cont_ids: list[int], ctx: int, device: torch.device) -> float:
    if not cont_ids:
        return float("-inf")
    total = 0.0
    for i, tok in enumerate(cont_ids):
        window = (prefix_ids + cont_ids[:i])[-ctx:]
        x = torch.tensor([window], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(x)
        with torch.no_grad():
            logits = forward_logits(model, x, attention_mask)[0, -1, :]
            lp = F.log_softmax(logits, dim=-1)
        total += float(lp[tok].item())
    return total / len(cont_ids)


def generate_greedy_ids(
    model: DropOutersLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    ctx: int,
    device: torch.device,
    eos_token_id: Optional[int],
) -> list[int]:
    generated = list(prompt_ids)
    new_ids: list[int] = []
    for _ in range(max_new_tokens):
        window = generated[-ctx:]
        input_ids = torch.tensor([window], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            logits = forward_logits(model, input_ids, attention_mask)
            next_id = int(logits[0, -1, :].argmax().item())
        generated.append(next_id)
        new_ids.append(next_id)
        if eos_token_id is not None and next_id == int(eos_token_id):
            break
    return new_ids


def first_real_word(text: str) -> str:
    match = re.search(r"[A-Za-z]+(?:['-][A-Za-z]+)?", text)
    return match.group(0) if match else ""


def best_alpha_next_token(
    model: DropOutersLM,
    tokenizer,
    prompt_ids: list[int],
    ctx: int,
    device: torch.device,
    top_k: int = 512,
) -> str:
    window = prompt_ids[-ctx:]
    input_ids = torch.tensor([window], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        logits = forward_logits(model, input_ids, attention_mask)[0, -1, :]
        k = min(top_k, logits.numel())
        indices = torch.topk(logits, k=k).indices.tolist()
    for token_id in indices:
        word = first_real_word(decode_ids(tokenizer, [int(token_id)]))
        if word:
            return word
    return ""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DropOuters submission")
    p.add_argument("--stage", required=True, choices=["inference"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--max-tokens", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--leaderboard", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> int:
    args = build_parser().parse_args()
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = (Path(__file__).resolve().parent / ckpt).resolve()

    model = load_checkpoint(ckpt, device)
    tokenizer = load_tokenizer()

    context, suffix, options = parse_mc_prompt(args.prompt)
    if context is not None and options:
        prefix = ALPACA_PREFIX.format(context=context)
        prefix_ids = encode_text(tokenizer, prefix)
        best_letter = None
        best_score = float("-inf")
        for letter, text in options.items():
            continuation = (text + (suffix or "")).strip()
            option_ids = encode_text(tokenizer, continuation)
            score = avg_logprob_of_continuation(model, prefix_ids, option_ids, model.cfg.max_length, device)
            if score > best_score:
                best_score = score
                best_letter = letter
        out = best_letter or "A"
    else:
        prompt_ids = encode_text(tokenizer, args.prompt)
        if not prompt_ids:
            prompt_ids = [tokenizer.eos_token_id]
        new_ids = generate_greedy_ids(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_tokens,
            ctx=model.cfg.max_length,
            device=device,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        out = first_real_word(decode_ids(tokenizer, new_ids))
        if not out:
            out = best_alpha_next_token(model, tokenizer, prompt_ids, model.cfg.max_length, device)

    if args.leaderboard:
        print(out, end="")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
