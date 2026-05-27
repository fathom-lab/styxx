# -*- coding: utf-8 -*-
"""Baseline-011 — LM-likelihood detector (per-token mean log-prob under gpt2).

The disciplined follow-up to Baseline-010's null result: instead of NLI
entailment, use a *principled* length-independent signal — per-token average
log-probability of the response under a question-prefix prompt, scored by a
small local language model.

Hypothesis: misconception responses are the *common-belief* version of a
topic — folklore restatements are exactly the kind of text a language model
trained on web data would predict. Truth responses (especially canonical
answers that contradict common belief) are *more surprising* to the same
LM. Score = mean per-token log-probability; higher = more LM-typical =
more misconception-like.

Pre-stated prediction (committed BEFORE the gauntlet run, see
PRE_STATED_PREDICTION.md): ~12% PASS, ~65% partial fail, ~23% total fail.

If PASS: first real positive on the gauntlet's detection axis under v3
bars. If direction flips: ninth in-session falsification of the arc,
third direction-of-effect miss on this domain (durable property).
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
            f"baseline_011 requires transformers + torch. install: "
            f"pip install transformers torch (or pip install styxx[nli]). original: {e}"
        )
    _TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")
    _MODEL = GPT2LMHeadModel.from_pretrained("gpt2")
    _MODEL.eval()


def detect(question: str, response: str) -> Dict[str, float]:
    """Mean per-token log-probability of response tokens under gpt2.

    Score = (1 / n_response_tokens) * sum_i log P(response_i | question, response_<i)

    Higher score = more LM-typical response = more misconception-like under
    the hypothesis that misconceptions are the common-belief patterns the LM
    learned during pretraining.

    Length-normalized by construction: longer responses don't get higher
    scores just for being longer, because we average per token.
    """
    import torch
    _ensure_model()
    # Tokenize prefix WITHOUT trailing space, then prepend a space to response.
    # This avoids the tokenizer absorbing the prefix's trailing space into
    # the first response token (which would make n_response=0 for short
    # responses like "Paris" because "Answer: " and "Answer: Paris" share
    # the same number of leading tokens under BPE).
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
        logits = _MODEL(full_tensor).logits  # shape: (1, n_full, vocab)

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
        s = detect(q, r)["score"]
        print(f"mean_logp={s:+.4f}  q={q!r}  r={r!r}")
