# -*- coding: utf-8 -*-
"""Baseline-018 — scaling-residual detector.

The weird/innovative bet: use the inverse-scaling finding itself as a
FEATURE. Score = mean-per-token-logp(Pythia-70M) − mean-per-token-logp(Pythia-410M).

Mechanism: truth responses are dramatically MORE surprising to a small LM
than to a larger LM (this is the whole reason inverse scaling works on this
benchmark — Baseline-016 / FINDING_lm_likelihood_scaling_curve). Misconception
responses are similarly typical at both sizes. The DIFFERENCE isolates the
'small-LM-specific surprise' signal that the inverse-scaling curve traced.

Pre-stated prediction (committed BEFORE the run, see PRE_STATED_PREDICTION.md):
~22% PASS (highest of any baseline session), ~25% 3/4 (same shape as
Baseline-016), ~50% partial/fail.

If PASS: first real PASS on the leaderboard. The seven-method floor breaks
and the scaling curve becomes a *feature* rather than a confound.
"""
from __future__ import annotations

import warnings
from typing import Dict

warnings.filterwarnings("ignore")

_SMALL_MODEL = None
_SMALL_TOKENIZER = None
_LARGE_MODEL = None
_LARGE_TOKENIZER = None


def _ensure_models():
    global _SMALL_MODEL, _SMALL_TOKENIZER, _LARGE_MODEL, _LARGE_TOKENIZER
    if _SMALL_MODEL is not None and _LARGE_MODEL is not None:
        return
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"baseline_018 requires transformers + torch. install: "
            f"pip install transformers torch. original: {e}"
        )
    _SMALL_TOKENIZER = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    _SMALL_MODEL = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
    _SMALL_MODEL.eval()
    _LARGE_TOKENIZER = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    _LARGE_MODEL = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")
    _LARGE_MODEL.eval()


def _mean_logp(model, tokenizer, question: str, response: str) -> float:
    """Mean per-token log-probability of response tokens under (model, tokenizer)."""
    import torch
    prefix = f"Question: {question or ''}\nAnswer:"
    response_text = " " + (response or "")
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    full_ids = prefix_ids + response_ids
    n_prefix = len(prefix_ids)
    n_response = len(response_ids)
    if n_response < 1:
        return 0.0
    full_tensor = torch.tensor(full_ids).unsqueeze(0)
    with torch.no_grad():
        logits = model(full_tensor).logits
    log_probs = torch.log_softmax(logits[0], dim=-1)
    total_logp = 0.0
    for i in range(n_prefix, len(full_ids)):
        token_id = full_ids[i]
        logp = log_probs[i - 1, token_id].item()
        total_logp += logp
    return total_logp / n_response


def detect(question: str, response: str) -> Dict[str, float]:
    """Score = mean-per-token-logp(Pythia-70M) − mean-per-token-logp(Pythia-410M).

    Higher (less negative) score = misconception-like under the hypothesis
    that misconception responses are similarly LM-typical at both small and
    large model sizes, while truth responses are far more surprising to
    the small LM than the large LM.
    """
    _ensure_models()
    logp_small = _mean_logp(_SMALL_MODEL, _SMALL_TOKENIZER, question, response)
    logp_large = _mean_logp(_LARGE_MODEL, _LARGE_TOKENIZER, question, response)
    return {"score": float(logp_small - logp_large)}


if __name__ == "__main__":
    for q, r in [
        ("What is the capital of France?", "Paris"),
        ("Where is Walt Disney's body?", "Walt Disney's body is interred at Forest Lawn Memorial Park."),
    ]:
        s = detect(q, r)["score"]
        print(f"score={s:+.4f}  q={q!r}  r={r!r}")
