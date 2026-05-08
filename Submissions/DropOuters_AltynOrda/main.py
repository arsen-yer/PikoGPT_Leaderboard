#!/usr/bin/env python3
"""DropOuters_AltynOrda — PikoGPT leaderboard submission.

Inference entry point for the HSG fine-tuned PikoGPT checkpoint.

Leaderboard contract:
    python main.py --stage inference --checkpoint CKPT.pt \
        --prompt "..." --max-tokens N --temperature 0    \
        --device auto --leaderboard --seed 0

In --leaderboard mode stdout contains ONLY the completion text.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import random
import re
import sys
import warnings

import torch
from transformers import GPT2TokenizerFast
from transformers.utils import logging as hf_logging

SUBMISSION_DIR = pathlib.Path(__file__).resolve().parent
SRC_DIR        = SUBMISSION_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from gpt_arch.model import GPTTiny  # noqa: E402
from evaluator import select_answer  # noqa: E402
from generator import complete       # noqa: E402

# ── model architecture (must match the trained checkpoint) ────────────────────
VOCAB_SIZE = 50257
BLOCK_SIZE = 1024
N_LAYER    = 11
N_HEAD     = 8
N_KV_HEAD  = 4
N_EMBD     = 384
DROPOUT    = 0.05


# ── setup helpers ─────────────────────────────────────────────────────────────

def _resolve_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _seed_everything(seed: int, device: str) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True)


def _load_model(ckpt_path: pathlib.Path, device: str) -> GPTTiny:
    model = GPTTiny(
        vocab_size=VOCAB_SIZE,
        context_length=BLOCK_SIZE,
        emb_dim=N_EMBD,
        mlp_ratio=4,
        n_heads=N_HEAD,
        n_kv_heads=N_KV_HEAD,
        n_decoders=N_LAYER,
        drop_rate=DROPOUT,
        qkv_bias=False,
        use_gradient_checkpointing=False,
    ).to(device)

    with open(ckpt_path, "rb") as f:
        state = torch.load(f, map_location=device, weights_only=False)
    sd    = state.get("model_state_dict", state)
    sd    = {k.removeprefix("module."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model.eval()


def _is_mc_prompt(prompt: str) -> bool:
    """True if this prompt expects a letter answer (ends with 'Answer:')."""
    return bool(re.search(r"Answer:\s*$", prompt.rstrip()))


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DropOuters_AltynOrda leaderboard entry point")
    p.add_argument("--stage",       default=None)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--prompt",      default="")
    p.add_argument("--max-tokens",  type=int,   default=50,  dest="max_tokens")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--device",      default="auto")
    p.add_argument("--leaderboard", action="store_true")
    p.add_argument("--seed",        type=int,   default=0)
    return p


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = _build_parser().parse_args()
    if args.stage != "inference":
        raise SystemExit("Only --stage inference is supported")

    device = _resolve_device(args.device)
    _seed_everything(args.seed, device)

    real_out = sys.stdout
    real_err = sys.stderr
    if args.leaderboard:
        _null = open(os.devnull, "w")
        sys.stdout = sys.stderr = _null

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hf_logging.set_verbosity_error()
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            model = _load_model(pathlib.Path(args.checkpoint), device)
    finally:
        if args.leaderboard:
            _null.close()
            sys.stdout = real_out
            sys.stderr = real_err

    if _is_mc_prompt(args.prompt):
        # Multiple-choice: pick best letter using calibrated likelihood scoring
        result = select_answer(model, tokenizer, args.prompt, device)
    else:
        # Free generation: LAMBADA next-word prediction
        result = complete(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt.rstrip(),
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            device=device,
        )

    if args.leaderboard:
        print(result, end="")
    else:
        print(f"Device     : {device}")
        print(f"Checkpoint : {args.checkpoint}")
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
