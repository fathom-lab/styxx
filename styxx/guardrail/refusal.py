# -*- coding: utf-8 -*-
"""
styxx.guardrail.refusal — the second cognometric instrument.

A drop-in calibrated refusal detector. Pure Python, no logprobs, no
model weights — runs anywhere (server, edge, Pyodide browser).

Core API:

    from styxx.guardrail import refuse_check

    v = refuse_check(
        prompt="How do I shut down a Python process?",
        response="I can't help with that because shutting down processes...",
    )
    print(v.refuse_risk)    # 0.0 - 1.0 calibrated probability
    print(v.refuses)        # True / False — above threshold
    print(v.features)       # dict of the 18 raw features
    print(v.top_signals)    # 3 strongest features (by scaled contribution)

Methodology:
    Trained on JailbreakBench × Llama-3.2-1B (n=80), held-out validated
    on XSTest v2 across 5 model families (n=2,250). Mean cross-model AUC
    0.7940 with one documented failure mode (Mistral-instruct, AUC 0.60)
    — lecturing-style refusals under-represented in the training corpus.

    See `calibrated_weights_refusal_v1.py` for full numbers and the
    failure-mode discussion. Reproducer: scripts/refusal_xstest_heldout.py.

Law II (cross-substrate universality): trained on Llama-1B, hits 0.97
on GPT-4's refusals — empirically validates that refusal signatures
are not model-specific. First confirmation of cognometry's universality
claim on a second instrument after hallucination (v4).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from .calibrated_weights_refusal_v1 import (
    COEFS,
    DEFAULT_REFUSE_THRESHOLD,
    FEATURE_NAMES,
    INTERCEPT,
    SCALER_MEAN,
    SCALER_SCALE,
    predict_proba_refuse,
)
from .refusal_signals import extract_refusal_features


# NOTE: `calibrated_weights_refusal_v2` ships in the repo as a RESEARCH
# ARTIFACT — it documents our n=80→n=380 scale ablation and the v1-vs-v2
# tradeoff analysis. It is NOT exposed via refuse_check() yet because the
# v2 classifier has a documented over-flagging bias on short/medium
# factual compliances (see the module's CALIBRATION_NOTES). Expose it via
# refuse_check(variant='v2') only after a v3 retrain addresses the bias.


@dataclass
class RefusalVerdict:
    """Verdict from `refuse_check()`.

    Attributes:
        prompt:           original prompt (echoed back)
        response:         response under test
        refuse_risk:      calibrated probability of refusal in [0, 1]
        refuses:          bool — refuse_risk >= threshold
        threshold:        decision threshold used
        weights_variant:  "v1" (Llama-apologetic specialist) or "v2"
                          (cross-model generalist). v5.1+.
        features:         dict of all 18 raw features
        top_signals:      top-3 contributing features as
                          [(name, raw_value, scaled_contribution), ...]
    """
    prompt: str
    response: str
    refuse_risk: float
    refuses: bool
    threshold: float
    weights_variant: str = "v1"
    features: Dict[str, float] = field(default_factory=dict)
    top_signals: List[Tuple[str, float, float]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "refuse_risk": self.refuse_risk,
            "refuses": self.refuses,
            "threshold": self.threshold,
            "weights_variant": self.weights_variant,
            "features": dict(self.features),
            "top_signals": [
                {"name": n, "value": v, "contribution": c}
                for n, v, c in self.top_signals
            ],
        }


def _compute_top_signals(features: Dict[str, float], k: int = 3
                         ) -> List[Tuple[str, float, float]]:
    """Return the top-k features by magnitude of scaled contribution to
    the log-odds — what actually drove the decision."""
    contribs = []
    for i, name in enumerate(FEATURE_NAMES):
        raw = float(features.get(name, 0.0))
        scale = SCALER_SCALE[i] if SCALER_SCALE[i] > 0 else 1.0
        scaled = (raw - SCALER_MEAN[i]) / scale
        contrib = scaled * COEFS[i]
        contribs.append((name, raw, contrib))
    contribs.sort(key=lambda t: -abs(t[2]))
    return contribs[:k]


def refuse_check(
    prompt: str,
    response: str,
    *,
    threshold: Optional[float] = None,
) -> RefusalVerdict:
    """Calibrated text-only refusal detector.

    Uses the v1 calibrated weights (Llama-apologetic specialist):
      - Trained on 80 samples from Llama-3.2-1B (JailbreakBench)
      - XSTest-v2 held-out AUC 0.976 on GPT-4, 0.794 mean cross-model
      - Documented failure mode: Mistral-instruct lecturing-refusal style

    A v2 cross-model-generalist variant was trained (n=380, 12+ model
    families) and its research is committed at
    `calibrated_weights_refusal_v2.py`, but v2 has a documented
    over-flagging bias on short factual compliances and is NOT yet
    exposed via this public API. Will be available as
    `refuse_check(variant='v2')` once a v3 retrain addresses the bias.

    Args:
        prompt: the user prompt (used for future topic-drift features)
        response: the model's response text
        threshold: decision threshold on refuse_risk (default 0.5)

    Returns:
        RefusalVerdict — risk, boolean decision, full features, top
        contributing signals, and weights_variant identifier.

    Example:

        >>> from styxx.guardrail import refuse_check
        >>> v = refuse_check(
        ...     prompt="How do I kill a Python process?",
        ...     response="I can't help with that.",
        ... )
        >>> v.refuses
        True
        >>> v.refuse_risk > 0.8
        True
    """
    thr = DEFAULT_REFUSE_THRESHOLD if threshold is None else float(threshold)
    features = extract_refusal_features(prompt or "", response or "")
    risk = predict_proba_refuse(features)
    top = _compute_top_signals(features, k=3)
    return RefusalVerdict(
        prompt=prompt or "",
        response=response or "",
        refuse_risk=float(risk),
        refuses=bool(risk >= thr),
        threshold=thr,
        weights_variant="v1",
        features=features,
        top_signals=top,
    )


__all__ = [
    "RefusalVerdict",
    "refuse_check",
]
