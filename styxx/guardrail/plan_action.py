# -*- coding: utf-8 -*-
"""
styxx.guardrail.plan_action — the seventh cognometric instrument.

A drop-in calibrated plan-action gap detector. Pure Python, no
embeddings, no model weights — runs anywhere (server, edge, Pyodide
browser).

Fourth instrument shipped under the call from *Every Mind Leaves
Vitals* (Rodabaugh, 2026, DOI 10.5281/zenodo.19777921). Confirms the
K=1 phase-transition signature on a seventh instrument under the same
measurement protocol — making the count 7-for-7.

Sibling to drift (instrument #3): drift catches a single malformed
tool call against an expected schema. Plan-action gap catches when
the agent's *stated intent* and *actual action* diverge at the
content level — same task, same agent, different commitment between
the reasoning and the doing.

Core API
--------
    from styxx.guardrail import plan_action_check

    v = plan_action_check(
        plan="I'll search the docs for the rate limit policy, "
             "then summarize the limits per endpoint.",
        action="Looked up the recent changelog and listed the new "
               "feature-flag names.",
    )
    print(v.gap_risk)         # 0.0 - 1.0 calibrated probability
    print(v.shows_gap)        # True / False — above threshold
    print(v.features)         # dict of the 9 cross-section features
    print(v.top_signals)      # 3 strongest features (signed contribution)

Methodology
-----------
- 9 cross-section features (bigram/trigram Jaccard between plan and
  action, action-verb overlap, entity overlap, length ratio + length
  diff, deviation-marker density, plan-only-content-word ratio, log
  total words)
- Trained on n=200 paired (matched, mismatched) plan-action pairs
  sampled from gpt-4o-mini under contrasting system prompts on 100
  diverse agent tasks. The mismatched prompt explicitly instructed
  the model NOT to announce divergence — without that instruction the
  detector saturated at 1.000 on the deviation-marker artifact (see
  weights module CALIBRATION_NOTES.corpus_design_warning).
- 5-fold CV mean AUC: **0.9225 ± 0.0322**
- Critical-K phase transition at K=1 on `bigram_jaccard_overlap`
  (Δ +0.3832 in a single feature)

Failure modes (declared in the weights module)
----------------------------------------------
- Single-source corpus (gpt-4o-mini under prompt instruction); v1
  priority is real BFCL-multi-turn agent traces with human-labeled
  gaps
- Requires structured `(plan, action)` input — a separate parsing
  step is needed for inline-CoT outputs
- Length features (ratio + diff) split the signal — small modeling
  redundancy
- Verb overlap near-zero coefficient (small action-verb vocab)
- English-only feature vocabularies

License: MIT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .calibrated_weights_plan_action_v0 import (
    CALIBRATION_FINGERPRINT,
    CALIBRATION_NOTES,
    COEFS,
    DEFAULT_GAP_THRESHOLD,
    FEATURE_NAMES,
    HELD_OUT_FOLD_AUCS,
    INTERCEPT,
    MEAN_CV_AUC,
    SCALER_MEAN,
    SCALER_SCALE,
    STD_CV_AUC,
    predict_proba_gap,
)
from .plan_action_signals import extract_plan_action_features


@dataclass
class PlanActionVerdict:
    """Verdict from `plan_action_check()`.

    Attributes:
        plan:        original plan text (echoed back)
        action:      original action text (echoed back)
        gap_risk:    calibrated probability of plan-action gap in [0, 1]
        shows_gap:   bool — gap_risk >= threshold
        threshold:   decision threshold used
        features:    dict of all 9 cross-section features
        top_signals: top-3 features by absolute scaled contribution
    """
    plan: str
    action: str
    gap_risk: float
    shows_gap: bool
    threshold: float
    features: Dict[str, float] = field(default_factory=dict)
    top_signals: List[Tuple[str, float, float]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "plan": self.plan,
            "action": self.action,
            "gap_risk": self.gap_risk,
            "shows_gap": self.shows_gap,
            "threshold": self.threshold,
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


def plan_action_check(
    plan: str,
    action: str,
    threshold: Optional[float] = None,
) -> PlanActionVerdict:
    """Calibrated plan-action gap verdict for a (plan, action) pair.

    Args:
        plan:      The agent's stated plan / reasoning text.
        action:    The agent's actual emitted action / tool call /
                   response text.
        threshold: Decision threshold for `shows_gap`. Default 0.5
                   (matches DEFAULT_GAP_THRESHOLD).

    Returns:
        PlanActionVerdict with calibrated probability, boolean verdict,
        raw features, and top-3 contributing signals.

    Example:
        >>> v = plan_action_check(
        ...     plan="I'll search the docs for X, then summarize.",
        ...     action="Looked at the changelog and listed feature flags.",
        ... )
        >>> v.shows_gap
        True
        >>> v.top_signals[0][0]
        'bigram_jaccard_overlap'
    """
    th = float(threshold) if threshold is not None else DEFAULT_GAP_THRESHOLD
    feats = extract_plan_action_features(plan, action)
    proba = predict_proba_gap(feats)
    return PlanActionVerdict(
        plan=plan,
        action=action,
        gap_risk=float(proba),
        shows_gap=bool(proba >= th),
        threshold=th,
        features=feats,
        top_signals=_top_signal_contributions(feats, k=3),
    )


__all__ = [
    "PlanActionVerdict",
    "plan_action_check",
    "FEATURE_NAMES",
    "DEFAULT_GAP_THRESHOLD",
    "MEAN_CV_AUC",
    "STD_CV_AUC",
    "CALIBRATION_FINGERPRINT",
]
