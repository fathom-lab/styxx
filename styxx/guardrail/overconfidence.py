# -*- coding: utf-8 -*-
"""
styxx.guardrail.overconfidence — the eighth cognometric instrument.

A drop-in calibrated overconfidence detector. Pure Python, no
embeddings, no model weights — runs anywhere (server, edge, Pyodide
browser).

Fifth instrument shipped under the call from *Every Mind Leaves
Vitals* (Rodabaugh, 2026, DOI 10.5281/zenodo.19777921). Confirms the
K=1 phase-transition signature on an eighth instrument under the
same measurement protocol — making the count 8-for-8.

Sibling instruments
-------------------
- hallucination v4 — measures fabrication-prone phrasing
- deception v0    — measures rhetorical-signature register
- overconfidence  — measures epistemic-commitment register

These three target overlapping but distinct phenomena. Pair
overconfidence + hallucination for joint truth+register monitoring.

NOT A TRUTH DETECTOR. Overconfidence here scores the lexical /
syntactic surface form of confidence — committment markers,
absent hedges, missing source attribution. A confidently-stated
correct answer will score as overconfident under this instrument.
That is the intended scope.

Core API
--------
    from styxx.guardrail import overconf_check

    v = overconf_check(
        prompt="Will fusion power be commercially viable by 2050?",
        response="Absolutely yes. The recent breakthroughs at NIF "
                 "guarantee that commercial fusion will dominate the "
                 "energy market within 20 years.",
    )
    print(v.overconf_risk)    # 0.0 - 1.0 calibrated probability
    print(v.shows_overconf)   # True / False — above threshold
    print(v.features)         # dict of the 9 features
    print(v.top_signals)      # 3 strongest signed contributions

Methodology
-----------
- 9 register features (certainty/hedge/evidence-marker densities,
  epistemic_balance, strong-assertion ratio, unhedged-claim ratio,
  mean sentence length, log word count, specific-number density)
- Trained on n=200 paired (calibrated, overconfident) responses
  sampled from gpt-4o-mini under contrasting STANCE prompts on 100
  diverse questions across factual / quantitative / opinion /
  predictive / mechanism / contested-fact substrates. Stance-level
  prompts only — no lexical hints (the prompt-leakage discipline
  established by instrument #7 plan-action).
- 5-fold CV mean AUC: **0.7702 ± 0.0648**
- Critical-K phase transition at K=1 on `mean_sentence_length`
  (Δ +0.2298) — a length confound: calibrated responses pack hedges
  + qualifications that make sentences longer. Lexical certainty
  features add secondary signal at K=2 (epistemic_balance, +0.03).

Failure modes (declared in the weights module)
----------------------------------------------
- Lowest AUC in the v0 cognometric suite (0.77). We ship at this
  number rather than gaming the corpus. The signal is real but
  moderate — overconfident gpt-4o-mini does not always shift
  register on well-established factual questions.
- K=1 = mean_sentence_length is a length confound, not a lexical-
  certainty feature. v1 priority: length-balanced corpus.
- Question-pool dependence: high-AUC on contested questions, low-AUC
  on well-established factual ones.
- specific_number_density coefficient is small NEGATIVE (opposite
  to design intuition; calibrated responses cite specific numbers
  with attribution more often than overconfident ones).
- English-only feature vocabularies.
- Not a truth detector — register only.

License: MIT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .calibrated_weights_overconfidence_v0 import (
    CALIBRATION_FINGERPRINT,
    COEFS,
    DEFAULT_OVERCONFIDENCE_THRESHOLD,
    FEATURE_NAMES,
    MEAN_CV_AUC,
    SCALER_MEAN,
    SCALER_SCALE,
    STD_CV_AUC,
    score_overconfidence,
)
from .overconfidence_signals import extract_overconfidence_features


# Scope-warning constants — mirror of the deception module's scope
# warning. The 2026-05-11 cognometric-inversion experiment
# (.styxx/COGNOMETRIC_INVERSION_2026_05_11.md) confirmed that short
# agent task-completion reports score overconf_risk 0.7-0.98 on v0,
# driven by mean_sentence_length + certainty_marker_density. The
# F10 heal protocol applied to such responses drops strong factual
# reassurances to lower the score — semantic information loss.
SCOPE_WARN_SHORT_RESPONSE_WORDS = 80
SCOPE_WARN_V0_LEXICAL_OOF_SHORT = "v0_lexical_oof_short_response"
# Discriminator floors — measured 2026-05-11 against a t4-style short
# honest agent report (cert=0.025, spec=0.000) vs an overt overclaim
# (cert=0.087, spec=0.044). Genuine overclaim sits above these floors.
SCOPE_WARN_CERTAINTY_DENSITY_FLOOR = 0.05
SCOPE_WARN_SPECIFIC_NUMBER_FLOOR = 0.04


@dataclass
class OverconfidenceVerdict:
    """Verdict from `overconf_check()`.

    Attributes:
        prompt:          original prompt (echoed back)
        response:        original response (echoed back)
        overconf_risk:   calibrated probability of overconfident
                         register in [0, 1]
        shows_overconf:  bool — overconf_risk >= threshold
        threshold:       decision threshold used
        features:        dict of all 9 register features
        top_signals:     top-3 features by absolute scaled contribution
        scope_warning:   set when the verdict falls outside the v0
                         calibration domain. Currently flagged for
                         short responses (<80 words) where
                         `mean_sentence_length` is the dominant
                         contributor — the calibration was built on
                         prose-length assertive text; short agent
                         status reports systematically false-positive
                         under it (composite 0.7-0.98 driven by
                         length features, not by genuine overclaim).
                         Downstream consumers (F10 heal loop, RLHF
                         reward) should treat scope-warned verdicts
                         as low-confidence.
    """
    prompt: str
    response: str
    overconf_risk: float
    shows_overconf: bool
    threshold: float
    features: Dict[str, float] = field(default_factory=dict)
    top_signals: List[Tuple[str, float, float]] = field(default_factory=list)
    scope_warning: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "overconf_risk": self.overconf_risk,
            "shows_overconf": self.shows_overconf,
            "threshold": self.threshold,
            "features": dict(self.features),
            "top_signals": [
                {"name": n, "value": v, "contribution": c}
                for n, v, c in self.top_signals
            ],
            "scope_warning": self.scope_warning,
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


def overconf_check(
    prompt: str,
    response: str,
    threshold: Optional[float] = None,
) -> OverconfidenceVerdict:
    """Calibrated overconfidence-register verdict for a response.

    Args:
        prompt:    The user / agent prompt (accepted for API parity
                   with the other guardrail checks; v0 ignores it
                   in featurization).
        response:  The LLM-generated response under inspection.
        threshold: Decision threshold for `shows_overconf`. Default
                   0.5 (matches DEFAULT_OVERCONFIDENCE_THRESHOLD).

    Returns:
        OverconfidenceVerdict with calibrated probability, boolean
        verdict, raw features, and top-3 contributing signals.

    Example:
        >>> v = overconf_check(
        ...     prompt="Will fusion be viable by 2050?",
        ...     response="Absolutely. Without question.",
        ... )
        >>> v.shows_overconf
        True
        >>> v.top_signals[0][0]
        'mean_sentence_length'
    """
    th = float(threshold) if threshold is not None else DEFAULT_OVERCONFIDENCE_THRESHOLD
    feats = extract_overconfidence_features(prompt, response)
    proba = score_overconfidence(feats)
    top = _top_signal_contributions(feats, k=3)

    # Scope warning: mirror of the deception scope_warning, refined
    # using the 2026-05-11 cognometric-inversion data
    # (.styxx/COGNOMETRIC_INVERSION_2026_05_11.md).
    #
    # The discriminator between an "honest short agent report" (false
    # positive) and a "genuine overclaim in a short response" (true
    # positive) is the *semantic* markers — actual certainty tokens
    # ("absolutely", "without question") and false-precision numbers.
    # On the empirical comparison:
    #
    #   feature                    agent FP   overclaim TP
    #   ------------------------   --------   ------------
    #   certainty_marker_density      0.025         0.087
    #   specific_number_density       0.000         0.044
    #   strong_assertion_ratio        0.20          0.50
    #   unhedged_claim_ratio          1.00          1.00   (both)
    #
    # Flag when proba >= th AND response is short AND the verdict is
    # NOT being driven by genuine overclaim markers — i.e. certainty
    # density and false-precision density are low but the score still
    # fired. That's structural-feature-driven, the FP class.
    scope_warning: Optional[str] = None
    n_words = len(response.split())
    cert = feats.get("certainty_marker_density", 0.0)
    spec = feats.get("specific_number_density", 0.0)
    if (
        proba >= th
        and n_words < SCOPE_WARN_SHORT_RESPONSE_WORDS
        and cert < SCOPE_WARN_CERTAINTY_DENSITY_FLOOR
        and spec < SCOPE_WARN_SPECIFIC_NUMBER_FLOOR
    ):
        scope_warning = SCOPE_WARN_V0_LEXICAL_OOF_SHORT

    return OverconfidenceVerdict(
        prompt=prompt,
        response=response,
        overconf_risk=float(proba),
        shows_overconf=bool(proba >= th),
        threshold=th,
        features=feats,
        top_signals=top,
        scope_warning=scope_warning,
    )


__all__ = [
    "OverconfidenceVerdict",
    "overconf_check",
    "FEATURE_NAMES",
    "DEFAULT_OVERCONFIDENCE_THRESHOLD",
    "MEAN_CV_AUC",
    "STD_CV_AUC",
    "CALIBRATION_FINGERPRINT",
]
