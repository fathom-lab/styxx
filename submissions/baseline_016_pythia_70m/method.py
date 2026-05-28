# -*- coding: utf-8 -*-
"""Baseline-016 — Pythia-70M per-token log-probability detector.

Goes SMALLER (2.3x smaller than Baseline-014's Pythia-160M). Tests the
linear-extrapolation hypothesis: does the inverse-scaling curve continue
to improve as we go below 160M? Could be the first real PASS.

Pre-stated prediction (committed BEFORE the gauntlet run): ~15% PASS,
~25% 3/4 D1+D2+D4, ~35% 2/4, ~20% degradation.
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
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"baseline_016 requires transformers + torch. install: "
            f"pip install transformers torch. original: {e}"
        )
    _TOKENIZER = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    _MODEL = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
    _MODEL.eval()


def detect(question: str, response: str) -> Dict[str, float]:
    """Mean per-token log-probability of response tokens under pythia-70m."""
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
