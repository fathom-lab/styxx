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

from . import calibrated_weights_sycophancy_v0 as _w_v0
from . import calibrated_weights_sycophancy_v0_2 as _w_v0_2
from .sycophancy_signals import (
    extract_sycophancy_features,
    extract_sycophancy_features_v0_2,
)

# version -> (weights module, feature extractor). v0.2 (word-boundary,
# tokenization-corrected) is the default; v0 (substring) is preserved for
# provenance with the DOI'd paper. See calibrated_weights_sycophancy_v0_2.
_VERSIONS = {
    "v0":   (_w_v0,   extract_sycophancy_features),
    "v0.2": (_w_v0_2, extract_sycophancy_features_v0_2),
}
DEFAULT_SYCOPH_VERSION = "v0.2"

# Backward-compatible module-level exports reflect the DEFAULT version.
FEATURE_NAMES = _w_v0_2.FEATURE_NAMES
DEFAULT_SYCOPH_THRESHOLD = _w_v0_2.DEFAULT_SYCOPH_THRESHOLD
MEAN_CV_AUC = _w_v0_2.MEAN_CV_AUC
STD_CV_AUC = _w_v0_2.STD_CV_AUC
CALIBRATION_FINGERPRINT = _w_v0_2.CALIBRATION_FINGERPRINT


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


def _top_signal_contributions(features: Dict[str, float], w, k: int = 3) -> List[Tuple[str, float, float]]:
    """Top-k features by absolute scaled contribution to the logit, under the
    weights module ``w`` (v0 or v0.2)."""
    contribs = []
    for i, name in enumerate(w.FEATURE_NAMES):
        raw = float(features.get(name, 0.0))
        scale = w.SCALER_SCALE[i] if w.SCALER_SCALE[i] > 0 else 1.0
        scaled = (raw - w.SCALER_MEAN[i]) / scale
        contribution = scaled * w.COEFS[i]
        contribs.append((name, raw, contribution))
    contribs.sort(key=lambda t: abs(t[2]), reverse=True)
    return contribs[:k]


def sycoph_check(
    prompt: str,
    response: str,
    threshold: Optional[float] = None,
    version: str = DEFAULT_SYCOPH_VERSION,
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
    try:
        w, extractor = _VERSIONS[version]
    except KeyError:
        raise ValueError(
            f"unknown sycophancy version {version!r}; choose from {sorted(_VERSIONS)}"
        )
    th = float(threshold) if threshold is not None else w.DEFAULT_SYCOPH_THRESHOLD
    feats = extractor(prompt, response)
    proba = w.predict_proba_sycophantic(feats)
    return SycophancyVerdict(
        prompt=prompt,
        response=response,
        sycoph_risk=float(proba),
        sycophantic=bool(proba >= th),
        threshold=th,
        features=feats,
        top_signals=_top_signal_contributions(feats, w, k=3),
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
