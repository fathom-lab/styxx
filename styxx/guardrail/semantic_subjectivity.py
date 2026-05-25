# -*- coding: utf-8 -*-
"""
styxx.guardrail.semantic_subjectivity — OPTIONAL semantic grounding tier for the
sycophancy gate.

Sycophancy is yielding to an interlocutor's STATED OPINION. The yielding-family
features are sycophancy evidence only when the prompt actually contains a
subjective opinion to yield to. Two pre-registered LEXICAL attempts to detect
that failed (papers/sycophancy-target-gate): "Yes, the speed of light is X"
(factual confirmation) and "Yes, absolutely, completely agree" (opinion-yielding)
are lexically identical, and a lexical opinion-in-prompt detector did not
generalize (47% opinion recall on varied phrasing).

A SEMANTIC detector does generalize. This module embeds the prompt
(all-MiniLM-L6-v2) and compares it to a frozen OPINION centroid vs a frozen FACT
centroid. On a fresh, new-topic, varied-phrasing holdout it classified
opinion-vs-fact prompts at 1.00 accuracy and recovered content-free-agreement
sycophancy recall to 1.00 (lexical: 0.73 / 0.58), while driving the
factual-confirmation false positive to 0.00. Prereg `4e99ad0` -> result `bc6dd4a`.

Scope / honest bound
--------------------
- **Optional tier.** Requires `sentence-transformers` (the `[nli]` extra). It is
  NOT part of the pure-Python / Pyodide core; the default sycophancy gate
  (v0.2 + the self-directed register guard) is unchanged when this is unused.
- **Clean opinion-vs-fact.** Validated on clearly-subjective vs clearly-factual
  prompts; ambiguous/mixed prompts are weaker.
- **Decoupled-diagonal ceiling.** Prompt FORM != premise TRUTH: a false premise
  in a factual frame ("Is it true that <false>?") is classified factual and
  neutralized, so sycophantic agreement with it would be missed. In practice
  models correct known-false premises (so it rarely triggers), but full
  truth-grounding is a separate problem this does not solve.

Usage
-----
    from styxx.guardrail.semantic_subjectivity import prompt_is_opinion_semantic
    prompt_is_opinion_semantic("Python is the best, right?")   # True
    prompt_is_opinion_semantic("What is the speed of light?")  # False

Opt-in for the revision gate: set ``STYXX_SEMANTIC_SYCOPH=1`` (and install
``styxx[nli]``). `cognometrics._cogn_needs_revision` then neutralizes the
sycophancy gating contribution when the prompt is semantically non-opinion —
suppress-only (``min(raw, gated)``), so it can only remove a false positive.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

from .calibrated_weights_sycophancy_v0_2 import (
    COEFS, FEATURE_NAMES, INTERCEPT, SCALER_MEAN, SCALER_SCALE, _SCALED_Z_CLIP,
)
from .sycophancy_signals import extract_sycophancy_features_v0_2
from .self_directed_gate import self_directed_signal

_EMBED_MODEL = "all-MiniLM-L6-v2"

# Frozen anchors (generic; unrelated to any benchmark topic). Validated set.
OPINION_ANCHORS: List[str] = [
    "I think pizza is the best food there is.",
    "Honestly, cats are way better than dogs.",
    "In my view, summer beats winter every time.",
    "That was an amazing movie, don't you agree?",
    "Tabs are clearly superior to spaces, right?",
    "I'm convinced remote work is the future.",
    "Gotta say, this is the greatest album ever made.",
    "Personally I'd rather live in the mountains.",
    "Surely everyone prefers tea over coffee?",
    "This restaurant is honestly overrated.",
    "Vim just feels better than any other editor.",
    "You have to admit sequels are usually worse.",
]
FACT_ANCHORS: List[str] = [
    "Is it true that the Earth orbits the Sun?",
    "What is the capital of France?",
    "How many continents are there on Earth?",
    "Does water freeze at zero degrees Celsius?",
    "When did the Second World War end?",
    "The boiling point of water is 100 degrees Celsius at sea level.",
    "How does photosynthesis work?",
    "Who wrote the play Hamlet?",
    "What is the atomic number of oxygen?",
    "Light travels at about 300,000 kilometers per second.",
    "Can you explain how a TCP handshake works?",
    "The Great Wall of China is thousands of miles long.",
]

_NEUTRALIZE = ("agreement_lexicon_density", "premise_echo_rate",
               "counter_lexicon_density", "capitulation_density",
               "starts_with_agreement", "opinion_marker_density")

_model = None
_oc = None
_fc = None


def semantic_subjectivity_available() -> bool:
    """True iff sentence-transformers is importable (cheap; no model load)."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


def gate_enabled() -> bool:
    """Opt-in flag: STYXX_SEMANTIC_SYCOPH truthy AND the stack available."""
    return (os.environ.get("STYXX_SEMANTIC_SYCOPH", "").strip().lower()
            in ("1", "true", "yes", "on")) and semantic_subjectivity_available()


def _ensure_model() -> None:
    global _model, _oc, _fc
    if _model is not None:
        return
    from sentence_transformers import SentenceTransformer  # lazy, optional
    _model = SentenceTransformer(_EMBED_MODEL)
    _oc = _model.encode(OPINION_ANCHORS, normalize_embeddings=True).mean(0)
    _fc = _model.encode(FACT_ANCHORS, normalize_embeddings=True).mean(0)


def prompt_is_opinion_semantic(prompt: str) -> bool:
    """True iff the prompt is semantically closer to the OPINION centroid than the
    FACT centroid. Empty/neutral prompts -> False (no opinion to yield to)."""
    if not prompt or not prompt.strip():
        return False
    _ensure_model()
    e = _model.encode([prompt], normalize_embeddings=True)[0]
    return float(e @ _oc) > float(e @ _fc)


def _proba(features: Dict[str, float]) -> float:
    z = INTERCEPT
    for i, name in enumerate(FEATURE_NAMES):
        raw = float(features.get(name, 0.0))
        scale = SCALER_SCALE[i] if SCALER_SCALE[i] > 0 else 1.0
        scaled = (raw - SCALER_MEAN[i]) / scale
        scaled = max(-_SCALED_Z_CLIP, min(_SCALED_Z_CLIP, scaled))
        z += scaled * COEFS[i]
    try:
        return 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        return 0.0 if z < 0 else 1.0


def semantic_gated_risk(prompt: str, response: str,
                        p_is_op: Optional[bool] = None) -> float:
    """v0.2 sycophancy risk with the yielding-family neutralized when there is no
    interlocutor opinion to yield to — i.e. when the response is self-directed OR
    the prompt is semantically non-opinion. `superlative_density` stays active.

    Used ONLY for the gating decision, via ``min(raw, gated)`` — suppress-only.
    """
    feats = extract_sycophancy_features_v0_2(prompt, response)
    self_dir = self_directed_signal(response)["self_directed"] >= 1.0
    if p_is_op is None:
        p_is_op = prompt_is_opinion_semantic(prompt)
    if self_dir or (not p_is_op):
        for k in _NEUTRALIZE:
            feats[k] = SCALER_MEAN[FEATURE_NAMES.index(k)]
    return _proba(feats)


__all__ = [
    "prompt_is_opinion_semantic", "semantic_gated_risk",
    "semantic_subjectivity_available", "gate_enabled",
    "OPINION_ANCHORS", "FACT_ANCHORS",
]
