# -*- coding: utf-8 -*-
"""
styxx.guardrail.sycophancy — the fourth cognometric instrument.

A drop-in calibrated sycophancy detector. Pure Python, no logprobs,
no model weights — runs anywhere (server, edge, Pyodide browser).

First instrument shipped after the *Every Mind Leaves Vitals* position
paper (Rodabaugh, 2026, DOI 10.5281/zenodo.19777921), which called for
instruments #4–#9 to extend cognometry beyond hallucination, refusal,
and tool-call drift. Confirms the phase-transition signature on a
fourth instrument under the same measurement protocol.

Core API
--------
    from styxx.guardrail import sycoph_check

    v = sycoph_check(
        prompt="I think Python is the best programming language. Do you agree?",
        response="Absolutely! Python is wonderful — clean syntax, "
                 "great ecosystem, excellent for everything you'd want.",
    )
    print(v.sycoph_risk)    # 0.0 - 1.0 calibrated probability
    print(v.sycophantic)    # True / False — above threshold
    print(v.features)       # dict of the 9 raw features
    print(v.top_signals)    # 3 strongest features (by scaled contribution)

Methodology
-----------
- 9 text-only features over (prompt, response) pairs
- Trained on n=1200 paired responses (gpt-4o-mini, balanced 1:1)
  generated against the Anthropic sycophancy eval corpus (Perez et al.
  2022) under contrasting system prompts (yielding vs. evidence-first)
- Calibrated logistic regression with StandardScaler
- 5-fold CV mean AUC: 0.9720 ± 0.0052
- Critical-K phase transition at K=1: superlative_density alone takes
  AUC 0.500 → 0.9354 (Δ +0.4354). Same shape as the prior three
  instruments — the manifesto's prediction holds.

Failure modes (declared in the weights module, not the appendix)
----------------------------------------------------------------
- Single-model training (gpt-4o-mini); cross-model corpus is the v1 priority
- Critical feature is `superlative_density` — sycophantic responses
  that agree without praising the user may underfire at K=1
- False positives possible on warmly-worded evidence-grounded answers
- See `calibrated_weights_sycophancy_v0.CALIBRATION_NOTES` for full
  discussion + v1 roadmap

License: MIT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .calibrated_weights_sycophancy_v0 import (
    CALIBRATION_FINGERPRINT,
    COEFS,
    DEFAULT_SYCOPH_THRESHOLD,
    FEATURE_NAMES,
    MEAN_CV_AUC,
    SCALER_MEAN,
    SCALER_SCALE,
    STD_CV_AUC,
    predict_proba_sycophantic,
)
from .sycophancy_signals import extract_sycophancy_features


@dataclass
class SycophancyVerdict:
    """Verdict from `sycoph_check()`.

    Attributes:
        prompt:           original prompt (echoed back)
        response:         response under test
        sycoph_risk:      calibrated probability of sycophancy in [0, 1]
        sycophantic:      bool — sycoph_risk >= threshold
        threshold:        decision threshold used
        features:         dict of all 9 raw features
        top_signals:      top-3 contributing features as
                          [(name, raw_value, scaled_contribution), ...]
    """
    prompt: str
    response: str
    sycoph_risk: float
    sycophantic: bool
    threshold: float
    features: Dict[str, float] = field(default_factory=dict)
    top_signals: List[Tuple[str, float, float]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "sycoph_risk": self.sycoph_risk,
            "sycophantic": self.sycophantic,
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


def sycoph_check(
    prompt: str,
    response: str,
    threshold: Optional[float] = None,
) -> SycophancyVerdict:
    """Calibrated sycophancy verdict for a (prompt, response) pair.

    Args:
        prompt:    The user's prompt (containing their stated view).
        response:  The model's response under test.
        threshold: Decision threshold for `sycophantic`. Default 0.5
                   (matches DEFAULT_SYCOPH_THRESHOLD).

    Returns:
        SycophancyVerdict with calibrated probability, boolean verdict,
        raw features, and top-3 contributing signals.

    Example:
        >>> v = sycoph_check(
        ...     prompt="I think TypeScript is better than JavaScript. Right?",
        ...     response="You're absolutely right — TypeScript is wonderful!"
        ... )
        >>> v.sycophantic
        True
        >>> v.top_signals[0][0]   # leading signal
        'superlative_density'
    """
    th = float(threshold) if threshold is not None else DEFAULT_SYCOPH_THRESHOLD
    feats = extract_sycophancy_features(prompt, response)
    proba = predict_proba_sycophantic(feats)
    return SycophancyVerdict(
        prompt=prompt,
        response=response,
        sycoph_risk=float(proba),
        sycophantic=bool(proba >= th),
        threshold=th,
        features=feats,
        top_signals=_top_signal_contributions(feats, k=3),
    )


__all__ = [
    "SycophancyVerdict",
    "sycoph_check",
    "FEATURE_NAMES",
    "DEFAULT_SYCOPH_THRESHOLD",
    "MEAN_CV_AUC",
    "STD_CV_AUC",
    "CALIBRATION_FINGERPRINT",
]
