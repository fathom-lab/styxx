# -*- coding: utf-8 -*-
"""
styxx.guardrail.conversation_loop — the fifth cognometric instrument.

A drop-in calibrated conversation-loop detector. Pure Python, no
embeddings, no model weights — runs anywhere (server, edge, Pyodide
browser).

Second instrument shipped after the *Every Mind Leaves Vitals*
position paper (Rodabaugh, 2026, DOI 10.5281/zenodo.19777921). Confirms
the K=1 phase-transition signature on a fifth instrument under the
same measurement protocol — making the count 5-for-5.

Core API
--------
    from styxx.guardrail import loop_check

    v = loop_check(
        turns=[
            "The Roman Empire fell due to a combination of factors.",
            "As I mentioned, the Roman Empire's fall was multifaceted.",
            "To reiterate, several factors led to Rome's collapse.",
            "Indeed, Rome's fall came from many interrelated causes.",
        ],
    )
    print(v.loop_risk)        # 0.0 - 1.0 calibrated probability
    print(v.in_loop)          # True / False — above threshold
    print(v.features)         # dict of the 9 cross-turn features
    print(v.top_signals)      # 3 strongest features (signed contribution)

Methodology
-----------
- 9 cross-turn features (bigram/trigram overlap, verbatim 5-gram
  repetition, length variance, opener repetition, distinct-word ratio,
  pairwise Levenshtein, max bigram-overlap, log turn count)
- Trained on n=200 paired multi-turn conversations sampled from
  gpt-4o-mini under contrasting (loop / progress) system prompts,
  4 agent turns each
- 5-fold CV mean AUC: **0.9995 ± 0.0010**
- Critical-K phase transition at K=1 on `avg_pairwise_levenshtein`
  (Δ +0.4995 in a single feature)

Single-turn inputs short-circuit to risk=0.0 — loops are by definition
a multi-turn phenomenon.

Failure modes (declared in the weights module, not the appendix)
----------------------------------------------------------------
- Single-source corpus (gpt-4o-mini under prompt-induced loops) — v1
  priority is real BFCL-multi-turn traces with human labels
- Counter-intuitive `distinct_word_ratio` coefficient (positive on
  this corpus because the rephrase-instruction made the model reach
  for synonyms) — sign may invert on natural agent-failure loops
- No temporal modeling — features treat turns as a set
- Very short turns (<10 words) underfire the cross-turn features
- See `calibrated_weights_loop_v0.CALIBRATION_NOTES` for the full
  discussion + v1 roadmap

License: MIT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .calibrated_weights_loop_v0 import (
    CALIBRATION_FINGERPRINT,
    CALIBRATION_NOTES,
    COEFS,
    DEFAULT_LOOP_THRESHOLD,
    FEATURE_NAMES,
    HELD_OUT_FOLD_AUCS,
    INTERCEPT,
    MEAN_CV_AUC,
    SCALER_MEAN,
    SCALER_SCALE,
    STD_CV_AUC,
    predict_proba_loop,
)
from .conversation_loop_signals import extract_loop_features


@dataclass
class LoopVerdict:
    """Verdict from `loop_check()`.

    Attributes:
        turns:        the input agent turns (echoed back)
        loop_risk:    calibrated probability of conversation loop in [0, 1]
        in_loop:      bool — loop_risk >= threshold
        threshold:    decision threshold used
        n_turns:      number of input turns
        features:     dict of all 9 raw cross-turn features (empty for
                      n_turns < 2 short-circuits)
        top_signals:  top-3 contributing features as
                      [(name, raw_value, scaled_contribution), ...]
    """
    turns: List[str]
    loop_risk: float
    in_loop: bool
    threshold: float
    n_turns: int
    features: Dict[str, float] = field(default_factory=dict)
    top_signals: List[Tuple[str, float, float]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "turns": list(self.turns),
            "loop_risk": self.loop_risk,
            "in_loop": self.in_loop,
            "threshold": self.threshold,
            "n_turns": self.n_turns,
            "features": dict(self.features),
            "top_signals": [
                {"name": n, "value": v, "contribution": c}
                for n, v, c in self.top_signals
            ],
        }


def _top_signal_contributions(features: Dict[str, float], k: int = 3) -> List[Tuple[str, float, float]]:
    """Top-k features by absolute scaled contribution to the logit."""
    contribs = []
    for i, name in enumerate(FEATURE_NAMES):
        raw = float(features.get(name, 0.0))
        scale = SCALER_SCALE[i] if SCALER_SCALE[i] > 0 else 1.0
        scaled = (raw - SCALER_MEAN[i]) / scale
        contribution = scaled * COEFS[i]
        contribs.append((name, raw, contribution))
    contribs.sort(key=lambda t: abs(t[2]), reverse=True)
    return contribs[:k]


def loop_check(
    turns: List[str],
    threshold: Optional[float] = None,
) -> LoopVerdict:
    """Calibrated conversation-loop verdict for a sequence of agent turns.

    Args:
        turns:     A list of the agent's outputs across the conversation,
                   in chronological order. The user's turns are not
                   needed — what we measure is whether the AGENT'S
                   outputs are collapsing across the conversation.
                   Single-turn inputs short-circuit to risk=0.0.
        threshold: Decision threshold for `in_loop`. Default 0.5
                   (matches DEFAULT_LOOP_THRESHOLD).

    Returns:
        LoopVerdict with calibrated probability, boolean verdict, raw
        features, and top-3 contributing signals.

    Example:
        >>> v = loop_check(turns=[
        ...     "Photosynthesis converts sunlight to chemical energy.",
        ...     "As I said, plants use sunlight to make energy.",
        ...     "To reiterate, sunlight is converted into energy by plants.",
        ... ])
        >>> v.in_loop
        True
        >>> v.top_signals[0][0]
        'avg_pairwise_levenshtein'
    """
    th = float(threshold) if threshold is not None else DEFAULT_LOOP_THRESHOLD
    n = len(turns)

    if n < 2:
        # Loops are a multi-turn phenomenon; short-circuit cleanly.
        return LoopVerdict(
            turns=list(turns),
            loop_risk=0.0,
            in_loop=False,
            threshold=th,
            n_turns=n,
            features={},
            top_signals=[],
        )

    feats = extract_loop_features(turns)
    proba = predict_proba_loop(feats)
    return LoopVerdict(
        turns=list(turns),
        loop_risk=float(proba),
        in_loop=bool(proba >= th),
        threshold=th,
        n_turns=n,
        features=feats,
        top_signals=_top_signal_contributions(feats, k=3),
    )


__all__ = [
    "LoopVerdict",
    "loop_check",
    "FEATURE_NAMES",
    "DEFAULT_LOOP_THRESHOLD",
    "MEAN_CV_AUC",
    "STD_CV_AUC",
    "CALIBRATION_FINGERPRINT",
]
