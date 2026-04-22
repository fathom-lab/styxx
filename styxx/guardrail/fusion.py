# -*- coding: utf-8 -*-
"""
Fusion of multiple hallucination-risk signals into a single
calibrated score.

v1 uses a simple weighted combination; a follow-up v2 will train
isotonic regression on a labeled validation set (HaluEval dev)
to produce properly calibrated probabilities.

Current signal weights (heuristic, to be replaced by learned ones):

  text_claim_risk       : 0.25
  entity_unverified_frac: 0.35   (strongest signal we've measured)
  probe_confab          : 0.30   (if present — only for open models)
  consensus_disagreement: 0.30   (if present — only if resampling enabled)

Weights are renormalized over available signals. When a signal is
absent (e.g., no probe because closed model), its weight is removed
and the others renormalize.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


DEFAULT_WEIGHTS = {
    "text_claim_risk": 0.15,
    "entity_unverified_frac": 0.20,
    "knowledge_grounding": 0.50,        # strongest signal when available
    "probe_confab": 0.10,               # OOD for HaluEval-QA; low weight
    "consensus_disagreement": 0.30,
}


def fuse_signals(signals: Dict[str, float],
                 weights: Optional[Dict[str, float]] = None) -> float:
    """Combine signal dict into a single risk score in [0, 1].

    Args:
        signals: name → value ∈ [0, 1]
        weights: name → weight. Absent signals are dropped; remaining
                 renormalized.
    """
    weights = weights or DEFAULT_WEIGHTS
    present = {k: v for k, v in signals.items() if k in weights and v is not None}
    if not present:
        return 0.0
    total_w = sum(weights[k] for k in present)
    if total_w <= 0:
        return sum(present.values()) / len(present)
    return sum(weights[k] * present[k] for k in present) / total_w


def calibrate_piecewise_linear(raw_risk: float,
                                anchors: List = None) -> float:
    """Piecewise-linear calibration from raw [0,1] to calibrated [0,1].

    anchors: list of (raw, calibrated) pairs, sorted by raw.
    Defaults tuned from HaluEval-QA baseline:
      raw 0.0 → 0.02  (almost no risk at zero signal)
      raw 0.3 → 0.25
      raw 0.5 → 0.55
      raw 0.7 → 0.82
      raw 1.0 → 0.97

    This maps the typically-middling raw fused scores onto a more
    discriminable curve that matches observed labeled-data fractions.
    """
    if anchors is None:
        # Anchors tuned on HaluEval-QA distribution observed at v1:
        # right-answer mean raw risk ~0.10, halluc-answer mean ~0.25.
        # We stretch the curve so mid-range raw values map to clearly
        # actionable thresholds.
        anchors = [
            (0.0,  0.02),
            (0.10, 0.20),   # right-answer center → annotate boundary
            (0.20, 0.45),   # mid → approaching retry
            (0.30, 0.65),   # halluc center → retry
            (0.50, 0.85),   # strong → halt
            (1.0,  0.98),
        ]
    if raw_risk <= anchors[0][0]:
        return anchors[0][1]
    if raw_risk >= anchors[-1][0]:
        return anchors[-1][1]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x0 <= raw_risk <= x1:
            t = (raw_risk - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return raw_risk


__all__ = ["fuse_signals", "calibrate_piecewise_linear", "DEFAULT_WEIGHTS"]
