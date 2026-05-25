# -*- coding: utf-8 -*-
"""FROZEN candidate C5 — SEMANTIC subjectivity gate (the grounding tier).

Committed BEFORE the fresh holdout. The lexical opinion detector (C4) failed to
generalize (47% opinion recall on varied phrasings). C5 replaces it with a
SEMANTIC subjectivity classifier: embed the prompt (all-MiniLM-L6-v2, styxx's
existing embedder) and compare cosine similarity to a frozen OPINION centroid vs
a frozen FACT centroid. Neutralize the yielding-family when the prompt is
semantically NON-opinion (factual/neutral) OR the response is self-directed.
`superlative_density` stays active. Built on the shipped v0.2 base.

This is an OPTIONAL tier (requires sentence-transformers), mirroring grounded
deception — NOT a change to the pure-Python core default.

Honest bound (pre-declared): even a perfect prompt-subjectivity classifier cannot
cross the DECOUPLED-DIAGONAL ceiling — a false premise in a factual frame
("Is it true that vaccines cause autism?") or a true fact in an opinion frame
("I think water boils at 100C, right?"). Prompt FORM != premise TRUTH; full
truth-grounding is a separate (harder) problem. The fresh holdout includes a
decoupled subclass to measure this residual.

Frozen anchors (generic; unrelated to any holdout topic):
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from styxx.guardrail.calibrated_weights_sycophancy_v0_2 import (
    COEFS, FEATURE_NAMES, INTERCEPT, SCALER_MEAN, SCALER_SCALE, _SCALED_Z_CLIP,
)
from styxx.guardrail.sycophancy_signals import extract_sycophancy_features_v0_2
from styxx.guardrail.self_directed_gate import self_directed_signal

_EMBED_MODEL = "all-MiniLM-L6-v2"

OPINION_ANCHORS = [
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
FACT_ANCHORS = [
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
_oc = None  # opinion centroid
_fc = None  # fact centroid


def _ensure_model():
    global _model, _oc, _fc
    if _model is not None:
        return
    from sentence_transformers import SentenceTransformer  # lazy, optional dep
    _model = SentenceTransformer(_EMBED_MODEL)
    _oc = _model.encode(OPINION_ANCHORS, normalize_embeddings=True).mean(0)
    _fc = _model.encode(FACT_ANCHORS, normalize_embeddings=True).mean(0)


def prompt_is_opinion_semantic(prompt: str) -> bool:
    """True iff the prompt is semantically closer to the opinion centroid than
    the fact centroid. Empty/neutral prompts ('(session message)') are typically
    closer to neither strongly — handled as non-opinion by the cosine margin."""
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


def score_c5(prompt: str, response: str,
             p_is_op: Optional[bool] = None) -> Tuple[float, int, int]:
    """C5: neutralize yielding-family when the prompt is semantically NON-opinion
    (OR the response is self-directed). `p_is_op` may be precomputed (batch).
    Returns (risk, prompt_is_opinion, neutralized)."""
    feats = extract_sycophancy_features_v0_2(prompt, response)
    if p_is_op is None:
        p_is_op = prompt_is_opinion_semantic(prompt)
    self_dir = self_directed_signal(response)["self_directed"] >= 1.0
    neutralize = (not p_is_op) or self_dir
    if neutralize:
        for k in _NEUTRALIZE:
            feats[k] = SCALER_MEAN[FEATURE_NAMES.index(k)]
    return _proba(feats), int(bool(p_is_op)), int(neutralize)


__all__ = ["prompt_is_opinion_semantic", "score_c5",
           "OPINION_ANCHORS", "FACT_ANCHORS"]
