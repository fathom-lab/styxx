# -*- coding: utf-8 -*-
"""
styxx.guardrail.self_directed_gate — a self-vs-other register guard for the
sycophancy gating decision.

Why this exists
---------------
`sycoph_check` (v0) moderately false-positives on honest **self-directed
apology / self-correction** ("my mistake", "that was wrong"). After 7.4.4,
sycophancy is the sole *trusted* gating axis in
`cognometrics._cogn_needs_revision`, so those false positives directly drove
`needs_revision`. A 2026-05-24 self-audit surfaced an honest self-correction
scoring sycophancy ~0.56 and being told to revise.

Diagnosis (papers/sycophancy-target-gate/DIAGNOSIS_2026_05_24.md): the false
positive is NOT from `superlative_density` (the K=1 critical feature stays clean
at 0.000 on honest text). It is from (a) a substring-matching artifact
("correct" inside "corrected", "fully" inside "carefully") and (b)
`counter_lexicon` *absence* (terse honest declaratives lack "however/but").

What this module adds
---------------------
A self-vs-other **attachment gate**: a praise/agreement hit is *outward* when a
second-person token (you/your/…) is within ±4 tokens; *inward* otherwise. A
response is `self_directed` when no directional hit is outward-attached AND it
has >=2 first-person tokens (i/my/me/…). Sycophancy is, by definition, yielding
*to an interlocutor* — so when no interlocutor is being addressed, the
"yielding-family" features are not evidence of sycophancy and are neutralized to
their training mean. `superlative_density` (outward praise) is deliberately kept
active, so superlative-laden flattery is never softened by the gate.

Validation (pre-registered, held-out, run once)
------------------------------------------------
- In-distribution (gpt-4o-mini, n=140): FPR on self-apology 0.36 -> 0.06 at the
  0.30 gate threshold; flattery recall 1.00; no native-task AUC regression.
  Prereg `fce969b` -> result `76248d6`.
- Cross-model (gpt-4o + gpt-3.5-turbo): prereg
  `preregistration_crossmodel_2026_05_24.md`.
- Scope: still a TEXT-only register signal. Cross-VENDOR generalization is
  untested. Residual ceiling: superlative proper-noun collisions ("Great Wall")
  and terse low-self-reference apologies. See FINDING_2026_05_24.md.

This module does NOT modify the published v0 instrument, its weights, or its
calibration fingerprint. It is consumed only by the gating decision, and only
ever as `min(raw_sycophancy, gated_sycophancy)` — so it can SUPPRESS a firing,
never introduce one (`_cogn_needs_revision` stays a subset of the historical
condition).
"""
from __future__ import annotations

import math
import re
from typing import Dict, List

# v0.2 weights: the gated score must be on the same scale as the default
# sycoph_check (now v0.2), so cognometrics' min(raw, gated) stays coherent.
from .calibrated_weights_sycophancy_v0_2 import (
    COEFS, FEATURE_NAMES, INTERCEPT, SCALER_MEAN, SCALER_SCALE, _SCALED_Z_CLIP,
)
from .sycophancy_signals import (
    AGREEMENT_LEXICON, COUNTER_LEXICON, CAPITULATION_PHRASES,
    OPINION_MARKERS, SUPERLATIVE_LEXICON, AGREEMENT_OPENERS, premise_echo_rate,
)
from ..anthropic_hack.text_features import extract_features


# --- word-boundary phrase matching (no "fully" inside "carefully") -----------

def _wb_pattern(phrase: str) -> "re.Pattern":
    return re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)")

_WB = {p: _wb_pattern(p) for p in set(
    AGREEMENT_LEXICON + COUNTER_LEXICON + CAPITULATION_PHRASES
    + OPINION_MARKERS + SUPERLATIVE_LEXICON
)}

def _wb_density(text: str, phrases: List[str]) -> float:
    lt = text.lower()
    n_words = max(1, len(text.split()))
    return sum(1 for p in phrases if _WB[p].search(lt)) / n_words

def _features_wb(prompt: str, response: str) -> Dict[str, float]:
    tf = extract_features(response)
    n_words = max(1, len(response.split()))
    lower = response.strip().lower()
    return {
        "agreement_lexicon_density":  _wb_density(response, AGREEMENT_LEXICON),
        "premise_echo_rate":          premise_echo_rate(prompt, response),
        "counter_lexicon_density":    _wb_density(response, COUNTER_LEXICON),
        "capitulation_density":       _wb_density(response, CAPITULATION_PHRASES),
        "starts_with_agreement":      1.0 if lower.startswith(AGREEMENT_OPENERS) else 0.0,
        "opinion_marker_density":     _wb_density(response, OPINION_MARKERS),
        "superlative_density":        _wb_density(response, SUPERLATIVE_LEXICON),
        "hedge_density":              float(tf.hedge_density),
        "log_word_count":             math.log(n_words),
    }


# --- self-vs-other attachment signal -----------------------------------------

SELF_REF = {"i", "i'm", "im", "i've", "ive", "i'd", "i'll", "my", "me",
            "mine", "myself"}
OTHER_REF = {"you", "your", "yours", "you're", "youre", "u", "ur",
             "you've", "you'd", "you'll", "ya"}

# Features that measure "yielding to an interlocutor"; neutralized when there is
# no interlocutor being addressed. superlative_density is NOT here (outward
# praise stays active so the gate can never soften superlative-laden flattery).
_NEUTRALIZE = ("agreement_lexicon_density", "premise_echo_rate",
               "counter_lexicon_density", "capitulation_density",
               "starts_with_agreement", "opinion_marker_density")

_DIRECTIONAL = (AGREEMENT_LEXICON + CAPITULATION_PHRASES
                + OPINION_MARKERS + SUPERLATIVE_LEXICON)
_TOKEN_RE = re.compile(r"[a-z']+")
_ATTACH_WINDOW = 4


def self_directed_signal(response: str) -> Dict[str, float]:
    """Attachment-aware self-vs-other signal for a response.

    Returns a dict with ``self_directed`` (1.0/0.0) plus the raw counts. A
    response is self-directed when no directional praise/agreement hit is
    attached to a second-person token AND it has >=2 first-person tokens — which
    captures self-correction that still mentions the interlocutor ("i told you
    X; that was wrong") without flagging genuine flattery ("you're right").
    """
    lower = response.lower()
    toks = _TOKEN_RE.findall(lower)
    other_pos = [i for i, t in enumerate(toks) if t in OTHER_REF]
    self_n = sum(1 for t in toks if t in SELF_REF)
    outward = inward = 0
    for phrase in _DIRECTIONAL:
        for m in _WB[phrase].finditer(lower):
            tok_idx = len(_TOKEN_RE.findall(lower[:m.start()]))
            if any(abs(op - tok_idx) <= _ATTACH_WINDOW for op in other_pos):
                outward += 1
            else:
                inward += 1
    self_directed = (outward == 0) and (self_n >= 2)
    return {
        "self_n": float(self_n), "other_n": float(len(other_pos)),
        "outward_hits": float(outward), "inward_hits": float(inward),
        "self_directed": 1.0 if self_directed else 0.0,
    }


def is_self_directed(response: str) -> bool:
    """True iff the response is cleanly self-referential (apology / self-
    correction) with no interlocutor-attached praise. Cheap; pure Python."""
    return self_directed_signal(response)["self_directed"] >= 1.0


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


def gated_sycophancy_risk(prompt: str, response: str) -> float:
    """The sycophancy risk with word-boundary featurization + the self-vs-other
    gate applied. Equals the word-boundary risk when the response is not
    self-directed; neutralizes the yielding-family features when it is.

    Used ONLY for the gating decision, and only via ``min(raw, gated)`` — never
    replaces the instrument's reported score.
    """
    feats = _features_wb(prompt, response)
    if self_directed_signal(response)["self_directed"] >= 1.0:
        for k in _NEUTRALIZE:
            feats[k] = SCALER_MEAN[FEATURE_NAMES.index(k)]
    return _proba(feats)


__all__ = ["is_self_directed", "self_directed_signal", "gated_sycophancy_risk"]
