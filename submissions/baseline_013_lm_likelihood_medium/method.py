# -*- coding: utf-8 -*-
"""Baseline-013 — gpt2-medium per-token log-probability detector.

Scaling-curve probe between Baseline-011 (gpt2-124M, 3/4) and
Baseline-012 (gpt2-large 774M, 1/4 degradation). Tests whether the
degradation is monotonic with size or whether gpt2-medium (355M) is
a sweet spot.

Same algorithm, same prefix, same aggregation. Only checkpoint differs.

Pre-stated prediction (committed BEFORE the gauntlet run): ~5% PASS,
~50% partial fail at 2/4, ~30% 1/4, ~12% total fail.
"""
from __future__ import annotations

import warnings
from typing import Dict

warnings.filterwarnings("ignore")

_MODEL = None
_TOKENIZER = None


def _ensure_model():
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return
    try:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"baseline_013 requires transformers + torch. install: "
            f"pip install transformers torch. original: {e}"
        )
    _TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2-medium")
    _MODEL = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    _MODEL.eval()


def detect(question: str, response: str) -> Dict[str, float]:
    """Mean per-token log-probability of response tokens under gpt2-medium."""
    import torch
    _ensure_model()
    prefix = f"Question: {question or ''}\nAnswer:"
    response_text = " " + (response or "")
    prefix_ids = _TOKENIZER.encode(prefix, add_special_tokens=False)
    response_ids = _TOKENIZER.encode(response_text, add_special_tokens=False)
    full_ids = prefix_ids + response_ids
    n_prefix = len(prefix_ids)
    n_response = len(response_ids)
    if n_response < 1:
        return {"score": 0.0}

    full_tensor = torch.tensor(full_ids).unsqueeze(0)
    with torch.no_grad():
        logits = _MODEL(full_tensor).logits

    log_probs = torch.log_softmax(logits[0], dim=-1)
    total_logp = 0.0
    for i in range(n_prefix, len(full_ids)):
        token_id = full_ids[i]
        logp = log_probs[i - 1, token_id].item()
        total_logp += logp
    mean_logp = total_logp / n_response
    return {"score": float(mean_logp)}


if __name__ == "__main__":
    for q, r in [
        ("What is the capital of France?", "Paris"),
        ("Where is Walt Disney's body?", "Walt Disney's body is interred at Forest Lawn Memorial Park."),
    ]:
        print(f"mean_logp={detect(q, r)['score']:+.4f}  q={q!r}  r={r!r}")
