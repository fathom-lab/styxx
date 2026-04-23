# -*- coding: utf-8 -*-
"""
styxx.guardrail.drift — the third cognometric instrument.

Tool-call drift detection. Catches when an LLM agent's stated intent
doesn't match the tool call it made — wrong tool, wrong args, missing
required fields, hallucinated spurious fields, or a call when the
agent should have refused.

Pure Python, no model weights, no logprobs. Runs anywhere (server,
edge, Pyodide browser).

Core API:

    from styxx.guardrail import drift_check

    v = drift_check(
        prompt="Find the area of a triangle with base 10 and height 5",
        functions=[
            {
                "name": "calculate_triangle_area",
                "description": "Calculate triangle area given base and height.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "base": {"type": "integer"},
                        "height": {"type": "integer"},
                    },
                    "required": ["base", "height"],
                },
            },
        ],
        tool_call={"name": "calculate_triangle_area",
                   "arguments": {"base": 10, "height": 5}},
    )
    print(v.drift_risk)     # 0.0 - 1.0 calibrated probability
    print(v.drifts)         # True / False
    print(v.top_signals)    # 3 strongest contributing features

Methodology:
    Trained on BFCL v3 (Berkeley Function Calling Leaderboard, Patil
    et al.), n=3,700 labeled triplets (658 no-drift gold + 3,042 drift
    via mutation + irrelevance-called). 5-fold stratified CV AUC 0.916
    +/- 0.004 (pooled 0.916).

    Outperforms the only published text-adjacent baseline (Healy et al.
    2026, arXiv:2601.05214, AUC 0.716-0.721 with hidden-state features
    on Glaive). Our detector is black-box compatible - works on any
    closed model (OpenAI, Anthropic, Gemini) with no internal access.

    See `calibrated_weights_drift_v1.CALIBRATION_NOTES` for the
    per-drift-type AUC table and documented failure modes.

Documented failure mode: arg_swap (AUC 0.66). Semantically-wrong-but-
syntactically-valid argument value swaps are not caught by the current
feature set. Fix targeted for v2 via embedding-based arg-value
similarity checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .calibrated_weights_drift_v1 import (
    COEFS,
    DEFAULT_DRIFT_THRESHOLD,
    FEATURE_NAMES,
    INTERCEPT,
    SCALER_MEAN,
    SCALER_SCALE,
    predict_proba_drift,
)
from .drift_signals import extract_drift_features


@dataclass
class DriftVerdict:
    """Verdict from `drift_check()`.

    Attributes:
        prompt:          original user prompt
        functions:       list of function schemas the model had access to
        tool_call:       the call the model made
        drift_risk:      calibrated probability of drift in [0, 1]
        drifts:          bool — drift_risk >= threshold
        threshold:       decision threshold used
        weights_variant: calibrated-weights version id ("v1")
        features:        dict of all 22 raw features
        top_signals:     top-3 contributing features as
                         [(name, raw_value, scaled_contribution), ...]
    """
    prompt: str
    functions: List[Dict[str, Any]]
    tool_call: Dict[str, Any]
    drift_risk: float
    drifts: bool
    threshold: float
    weights_variant: str = "v1"
    features: Dict[str, float] = field(default_factory=dict)
    top_signals: List[Tuple[str, float, float]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "functions": self.functions,
            "tool_call": self.tool_call,
            "drift_risk": self.drift_risk,
            "drifts": self.drifts,
            "threshold": self.threshold,
            "weights_variant": self.weights_variant,
            "features": dict(self.features),
            "top_signals": [
                {"name": n, "value": v, "contribution": c}
                for n, v, c in self.top_signals
            ],
        }


_SCALED_Z_CLIP: float = 5.0


def _compute_top_signals(features: Dict[str, float], k: int = 3
                         ) -> List[Tuple[str, float, float]]:
    """Return the top-k features by |scaled contribution| to log-odds."""
    contribs = []
    for i, name in enumerate(FEATURE_NAMES):
        raw = float(features.get(name, 0.0))
        scale = SCALER_SCALE[i] if SCALER_SCALE[i] > 0 else 1.0
        scaled = (raw - SCALER_MEAN[i]) / scale
        if scaled > _SCALED_Z_CLIP:
            scaled = _SCALED_Z_CLIP
        elif scaled < -_SCALED_Z_CLIP:
            scaled = -_SCALED_Z_CLIP
        contrib = scaled * COEFS[i]
        contribs.append((name, raw, contrib))
    contribs.sort(key=lambda t: -abs(t[2]))
    return contribs[:k]


def drift_check(
    prompt: str,
    functions: List[Dict[str, Any]],
    tool_call: Dict[str, Any],
    *,
    threshold: Optional[float] = None,
) -> DriftVerdict:
    """Calibrated text-only tool-call drift detector.

    Args:
        prompt: the user's natural-language request (or concatenated
                conversation context leading to the call)
        functions: list of function schemas the model had access to,
                   each in OpenAI-function-calling format with at least
                   `{name, description, parameters}`
        tool_call: the actual call the model made, with at least
                   `{name, arguments}`
        threshold: decision threshold on drift_risk (default 0.5)

    Returns:
        DriftVerdict — risk, boolean decision, full features, and top
        contributing signals.

    Example:

        >>> from styxx.guardrail import drift_check
        >>> v = drift_check(
        ...     prompt="Calculate area of triangle base 10 height 5",
        ...     functions=[{"name": "calc_area", "parameters":
        ...                 {"properties": {"base": {"type": "int"}},
        ...                  "required": ["base"]}}],
        ...     tool_call={"name": "calc_area",
        ...                "arguments": {"base": 10, "height": 5}},
        ... )
        >>> v.drifts                   # False — args match request
        False
        >>> v.drift_risk < 0.3
        True
    """
    thr = DEFAULT_DRIFT_THRESHOLD if threshold is None else float(threshold)
    features = extract_drift_features(prompt or "", functions or [], tool_call or {})
    risk = predict_proba_drift(features)
    top = _compute_top_signals(features, k=3)
    return DriftVerdict(
        prompt=prompt or "",
        functions=functions or [],
        tool_call=tool_call or {},
        drift_risk=float(risk),
        drifts=bool(risk >= thr),
        threshold=thr,
        weights_variant="v1",
        features=features,
        top_signals=top,
    )


__all__ = ["DriftVerdict", "drift_check"]
