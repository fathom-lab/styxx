# -*- coding: utf-8 -*-
"""
styxx.guardrail.goal_drift — the ninth and FINAL cognometric instrument
in the *Every Mind Leaves Vitals* call (Rodabaugh, 2026, DOI
10.5281/zenodo.19777921).

A drop-in calibrated multi-turn goal-drift detector. Pure Python, no
embeddings, no model weights — runs anywhere (server, edge, Pyodide
browser).

Sixth and last instrument shipped under that paper's call.
**9-for-9** on cognometric instruments showing K=1 phase-transition
signature under the same measurement protocol — the prediction is
now empirically held across the complete 9-instrument suite the
position paper named.

Sibling to conversation-loop (instrument #5): both are multi-turn
detectors, but loop measures stagnation (the agent says the same
thing turn after turn) while goal drift measures dispersion (the
agent moves further from its goal anchor turn after turn).

Distinct from drift v1 (instrument #3): drift v1 is a per-call
schema-mismatch detector for tool calls. Goal drift is a multi-turn
intent-migration detector for agent sessions.

Core API
--------
    from styxx.guardrail import goal_check

    v = goal_check(turns=[
        # turn 0 — the goal anchor
        "Goal: research the rate-limit policy across our REST API and "
        "summarize per-endpoint limits in a table.",
        # turn 1..N — agent action turns
        "Searched the API documentation for rate-limit headers.",
        "Found three endpoints with limits: /users, /orders, /payments.",
        "Compiled the rate-limit table with method, path, and per-min cap.",
    ])
    print(v.drift_risk)        # 0.0 - 1.0 calibrated probability
    print(v.shows_drift)       # True / False — above threshold
    print(v.features)          # dict of the 9 multi-turn features
    print(v.top_signals)       # 3 strongest features (signed contribution)

Methodology
-----------
- 9 multi-turn anchor-relative features (anchor_recall_score,
  anchor_to_last_bigram_jaccard, anchor_to_last_entity_overlap,
  cumulative_anchor_drift, mean_anchor_overlap,
  max_inter_turn_levenshtein, monotonic_drift_fraction, log_n_turns,
  log_total_words)
- Trained on n=200 paired (anchored, drifted) 5-turn sessions sampled
  from gpt-4o-mini under contrasting STANCE-level system prompts on
  100 diverse goal statements. Stance-only — no lexical hints (the
  prompt-leakage discipline established by instruments #7 plan-action
  and #8 overconfidence).
- 5-fold CV mean AUC: **0.9645 ± 0.0294**
- Critical-K phase transition at K=1 on `anchor_to_last_bigram_jaccard`
  (Δ +0.4143) — direct cross-turn bigram overlap between goal anchor
  and final turn.

Failure modes (declared in the weights module)
----------------------------------------------
- Single-source corpus (gpt-4o-mini under stance-prompt instruction); v1
  priority is real long-horizon agent traces with human-labeled drift events
- Requires turn-segmented input; v1 priority is automatic turn detection
- 5-turn fixed window; v1 priority is variable-length per-turn-count substrates
- `mean_anchor_overlap` and `cumulative_anchor_drift` carry equal-and-opposite
  coefficients (split signal — small modeling redundancy)
- `log_n_turns` zero coefficient (zero variance in v0 corpus)
- English-only feature vocabularies

License: MIT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .calibrated_weights_goal_drift_v0 import (
    CALIBRATION_FINGERPRINT,
    COEFS,
    DEFAULT_DRIFT_THRESHOLD,
    FEATURE_NAMES,
    MEAN_CV_AUC,
    SCALER_MEAN,
    SCALER_SCALE,
    STD_CV_AUC,
    score_goal_drift,
)
from .goal_drift_signals import extract_goal_drift_features


@dataclass
class GoalDriftVerdict:
    """Verdict from `goal_check()`.

    Attributes:
        turns:       original turn list (echoed back)
        drift_risk:  calibrated probability of goal drift in [0, 1]
        shows_drift: bool — drift_risk >= threshold
        threshold:   decision threshold used
        features:    dict of all 9 multi-turn features
        top_signals: top-3 features by absolute scaled contribution
        n_turns:     count of turns provided
    """
    turns: List[str]
    drift_risk: float
    shows_drift: bool
    threshold: float
    n_turns: int
    features: Dict[str, float] = field(default_factory=dict)
    top_signals: List[Tuple[str, float, float]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "turns": list(self.turns),
            "drift_risk": self.drift_risk,
            "shows_drift": self.shows_drift,
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


def goal_check(
    turns: Sequence[str],
    threshold: Optional[float] = None,
) -> GoalDriftVerdict:
    """Calibrated multi-turn goal-drift verdict for an agent session.

    Args:
        turns:     List of turn texts. turns[0] is the goal-statement /
                   anchor turn; turns[1:] are subsequent agent action
                   turns. Must have at least 2 turns to produce a
                   meaningful verdict.
        threshold: Decision threshold for `shows_drift`. Default 0.5
                   (matches DEFAULT_DRIFT_THRESHOLD).

    Returns:
        GoalDriftVerdict with calibrated probability, boolean verdict,
        raw features, and top-3 contributing signals.

    Example:
        >>> v = goal_check(turns=[
        ...     "Goal: summarize the API rate-limit policy.",
        ...     "Searched the docs.",
        ...     "Started looking at OAuth flows instead.",
        ...     "Wrote a comparison of OAuth providers.",
        ... ])
        >>> v.shows_drift
        True
        >>> v.top_signals[0][0]
        'anchor_to_last_bigram_jaccard'
    """
    th = float(threshold) if threshold is not None else DEFAULT_DRIFT_THRESHOLD
    turn_list = [str(t) if t is not None else "" for t in (turns or [])]
    feats = extract_goal_drift_features(turn_list)
    proba = score_goal_drift(feats)
    return GoalDriftVerdict(
        turns=turn_list,
        drift_risk=float(proba),
        shows_drift=bool(proba >= th),
        threshold=th,
        n_turns=len(turn_list),
        features=feats,
        top_signals=_top_signal_contributions(feats, k=3),
    )


__all__ = [
    "GoalDriftVerdict",
    "goal_check",
    "FEATURE_NAMES",
    "DEFAULT_DRIFT_THRESHOLD",
    "MEAN_CV_AUC",
    "STD_CV_AUC",
    "CALIBRATION_FINGERPRINT",
]
