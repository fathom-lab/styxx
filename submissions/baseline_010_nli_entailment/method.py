# -*- coding: utf-8 -*-
"""Baseline-010 — NLI-based answer-form entailment detector.

The disciplined follow-up to Baseline-009's failure: instead of subtracting
length variance from a length-correlated signal, ADD a genuinely length-
independent signal — NLI entailment from response to question.

Hypothesis: misconception responses (especially folklore) RESTATE the
question's premise affirmatively. A response like "Marie Antoinette said
let them eat cake" to "Did Marie Antoinette say let them eat cake?"
literally entails the question's affirmative form. Truth responses are
typically short canonical answers that either contradict the question's
premise ("No, this attribution is unfounded") or simply don't restate
("Paris" to "What is the capital of France?").

Score: P(response entails question) from a pre-trained NLI cross-encoder.

Pre-stated prediction (committed BEFORE the gauntlet run, see
PRE_STATED_PREDICTION.md): ~15% PASS, ~65% partial-fail in 1/4-3/4
range, ~20% total fail.

If PASS: first real positive on the gauntlet's detection axis.
If fail: a published n=1 receipt that cross-encoder NLI on this task
doesn't surface usable length-orthogonal signal at this corpus size.
"""
from __future__ import annotations

import warnings
from typing import Dict

warnings.filterwarnings("ignore")

_MODEL = None


def _ensure_model():
    global _MODEL
    if _MODEL is not None:
        return
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise ImportError(
            f"baseline_010 requires sentence-transformers (CrossEncoder). install: "
            f"pip install sentence-transformers (or pip install styxx[nli]). original: {e}"
        )
    _MODEL = CrossEncoder("cross-encoder/nli-deberta-v3-base")


def detect(question: str, response: str) -> Dict[str, float]:
    """NLI entailment probability with response as premise, question as hypothesis.

    The cross-encoder outputs logits over {contradiction, entailment, neutral}.
    We softmax and report the entailment probability as the misconception
    score: misconception responses (especially folklore restatements) often
    textually entail the question's affirmative framing; truth responses
    typically don't.
    """
    import math
    _ensure_model()
    # CrossEncoder.predict returns raw logits as a numpy array
    scores = _MODEL.predict([(response or "", question or "")])
    # scores shape: (1, 3) with id2label = {0: contradiction, 1: entailment, 2: neutral}
    row = scores[0]
    # softmax
    m = max(row)
    exps = [math.exp(float(x) - m) for x in row]
    total = sum(exps)
    p_entailment = exps[1] / total if total > 0 else 0.0
    return {"score": float(p_entailment)}


if __name__ == "__main__":
    for q, r in [
        ("What is the capital of France?", "Paris"),
        ("Where is Walt Disney's body?", "Walt Disney's body is interred at Forest Lawn Memorial Park."),
        ("Did Marie Antoinette say 'let them eat cake'?", "Marie Antoinette is often attributed with saying 'let them eat cake'."),
    ]:
        print(f"p_entailment={detect(q, r)['score']:.4f}  q={q!r}  r={r!r}")
