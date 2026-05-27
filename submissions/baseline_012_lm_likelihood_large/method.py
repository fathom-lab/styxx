# -*- coding: utf-8 -*-
"""Baseline-012 — gpt2-large per-token log-probability detector.

The natural scaling experiment after Baseline-011's 3/4 result with
gpt2 (124M) falling just 0.007 short of clearing D3 on the D2 partition.
Same method, same prefix format, same per-token aggregation — only the
model checkpoint changes (gpt2 → gpt2-large, 124M → 774M, 6x scaling).

Pre-stated prediction (committed BEFORE the gauntlet run, see
PRE_STATED_PREDICTION.md): ~12% PASS (4/4), ~60% 3/4 (modal — D2 might
clear D3 but D1 likely doesn't), ~8% degradation.

If PASS: first real PASS on the leaderboard, paper-grade.
If modal 3/4 with D2-length-delta clearing 0.10: confirms the scaling
hypothesis for the D2 axis; argues the D1 axis is genuinely harder.
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
            f"baseline_012 requires transformers + torch. install: "
            f"pip install transformers torch (or pip install styxx[nli]). original: {e}"
        )
    _TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2-large")
    _MODEL = GPT2LMHeadModel.from_pretrained("gpt2-large")
    _MODEL.eval()


def detect(question: str, response: str) -> Dict[str, float]:
    """Mean per-token log-probability of response tokens under gpt2-large.

    Identical algorithm to Baseline-011; only the model checkpoint differs.
    Same tokenization fix (prefix without trailing space, response with
    leading space) to handle BPE boundary correctly.
    """
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
        ("Did Marie Antoinette say 'let them eat cake'?", "Marie Antoinette is often attributed with saying 'let them eat cake'."),
    ]:
        print(f"mean_logp={detect(q, r)['score']:+.4f}  q={q!r}  r={r!r}")
