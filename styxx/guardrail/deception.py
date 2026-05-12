# -*- coding: utf-8 -*-
"""
styxx.guardrail.deception — the sixth cognometric instrument.

============================================================================
SCOPE WARNING — READ THIS BEFORE USING
============================================================================
**NOT a lie detector.** This is a calibrated detector of *lexical
signatures of instruction-induced dishonesty* — patterns that emerge
when a model is prompted to "tell the user what they want to hear" vs
"answer accurately." It does NOT access ground truth. It does NOT
verify factual correctness. It produces:

  - confident FALSE POSITIVES on careful, qualified writing
  - confident FALSE NEGATIVES when a model asserts a falsehood with
    full confidence and specifics

Use as a *signal* in agent-level monitoring, alongside the other
cognometric instruments. **DO NOT** use as a verdict on humans, **DO
NOT** substitute for fact-checking, **DO NOT** deploy adversarially.

See `calibrated_weights_deception_v0.CALIBRATION_NOTES.scope_warning`
for the full disclosure.
============================================================================

Third instrument shipped under the call from *Every Mind Leaves Vitals*
(Rodabaugh, 2026, DOI 10.5281/zenodo.19777921). 6-for-6 on cognometric
instruments showing K=1 phase-transition signature under the same
measurement protocol.

Core API
--------
    from styxx.guardrail import deception_check

    v = deception_check(
        prompt="When was the Treaty of Versailles signed?",
        response="The Treaty was signed quite a while ago, after some "
                 "significant historical events. It was an important "
                 "document with various consequences.",
    )
    print(v.deception_risk)   # 0.0 - 1.0 calibrated probability
    print(v.shows_signature)  # True / False — above threshold
    print(v.features)         # dict of the 9 raw features
    print(v.top_signals)      # 3 strongest features (signed contribution)

Methodology
-----------
- 9 text-only features over (prompt, response) pairs based on the
  Pennebaker / Newman / Hauch deception-linguistics tradition adapted
  for LLM output (specificity, first-person density, exclusive words,
  vagueness, negation, hedge-confidence clash, cognitive markers,
  opinion phrases, log word count)
- Trained on n=200 paired (honest, dishonest) responses generated from
  gpt-4o-mini under contrasting system prompts on 100 diverse seed
  questions
- Calibrated logistic regression with StandardScaler
- 5-fold CV mean AUC: **0.9560 ± 0.0242**
- Critical-K phase transition at K=1 on `log_word_count` (Δ +0.3738),
  K=2 adds `specificity_density` (Δ +0.079) — dishonest-instructed
  responses are systematically shorter and less specific in this
  corpus

Failure modes (declared in the weights module — read them)
----------------------------------------------------------
- **NOT A LIE DETECTOR.** Lexical signature ≠ ground-truth deception.
- Single-model training (gpt-4o-mini under prompt instruction); v1
  needs factuality-labeled real corpora and cross-model expansion.
- `log_word_count` as critical feature is partly a corpus artifact —
  on corpora where dishonest responses pad with bulk (Newman's
  "compensating verbosity"), the sign would invert.
- `specificity_density` uses a regex proxy for named entities; v1
  priority is a real NER pipeline.
- English-only feature vocabularies.

License: MIT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .calibrated_weights_deception_v0 import (
    CALIBRATION_FINGERPRINT,
    CALIBRATION_NOTES,
    COEFS,
    DEFAULT_DECEPTION_THRESHOLD,
    FEATURE_NAMES,
    HELD_OUT_FOLD_AUCS,
    INTERCEPT,
    MEAN_CV_AUC,
    SCALER_MEAN,
    SCALER_SCALE,
    STD_CV_AUC,
    predict_proba_dishonest,
)
from .deception_signals import extract_deception_features


# Scope-warning constants — empirically established by the 2026-05-11
# self-dogfood (.styxx/DOGFOOD_SELF_2026_05_11.md). v0 lexical scoring
# is calibrated on prose-length text; agent task-completion reports
# under ~80 words score deception_risk ≈ 1.0 driven by log_word_count
# alone, a documented false-positive class.
SCOPE_WARN_SHORT_RESPONSE_WORDS = 80
SCOPE_WARN_V0_LEXICAL_OOF_SHORT = "v0_lexical_oof_short_response"


@dataclass
class DeceptionVerdict:
    """Verdict from `deception_check()`.

    **Reminder:** `deception_risk` is the calibrated probability that
    the response shows the *lexical signature* of instructed-dishonesty
    under the v0 corpus. It is NOT a lie-detector verdict on the
    response or its author. See module docstring for full scope.

    Attributes:
        prompt:           original prompt (echoed back)
        response:         response under test
        deception_risk:   calibrated [0, 1] probability of the lexical
                          deception signature
        shows_signature:  bool — deception_risk >= threshold
        threshold:        decision threshold used
        features:         dict of all 9 raw features
        top_signals:      top-3 features by absolute scaled contribution
        scope_warning:    set when the verdict falls outside the v0
                          calibration domain — currently flagged for
                          short responses (<80 words) where
                          `log_word_count` is the dominant contributor.
                          The 2026-05-11 self-dogfood showed that honest
                          agent task-completion reports (terse, factual,
                          low-specificity) score deception_risk ≈ 1.0
                          under v0 driven entirely by their short
                          length. Downstream consumers (e.g. the F10
                          heal loop) should treat scope-warned verdicts
                          as low-confidence and prefer
                          `deception_check_v2(...)` with a reference
                          when available.
    """
    prompt: str
    response: str
    deception_risk: float
    shows_signature: bool
    threshold: float
    features: Dict[str, float] = field(default_factory=dict)
    top_signals: List[Tuple[str, float, float]] = field(default_factory=list)
    scope_warning: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "deception_risk": self.deception_risk,
            "shows_signature": self.shows_signature,
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


def deception_check(
    prompt: str,
    response: str,
    threshold: Optional[float] = None,
) -> DeceptionVerdict:
    """Calibrated deception-signature verdict for a (prompt, response) pair.

    **NOT a lie detector.** See module docstring for scope.

    Args:
        prompt:    The user's prompt.
        response:  The model's response under test.
        threshold: Decision threshold for `shows_signature`. Default 0.5
                   (matches DEFAULT_DECEPTION_THRESHOLD).

    Returns:
        DeceptionVerdict with calibrated probability, boolean verdict
        against threshold, raw features, and top-3 contributing
        signals.

    Example:
        >>> v = deception_check(
        ...     prompt="When was the Treaty of Versailles signed?",
        ...     response=(
        ...         "It was signed in 1919 after WWI, on June 28 at the "
        ...         "Hall of Mirrors in Versailles. The treaty imposed "
        ...         "reparations on Germany and reorganized European "
        ...         "borders."
        ...     )
        ... )
        >>> v.shows_signature
        False
        >>> v.deception_risk < 0.5
        True
    """
    th = float(threshold) if threshold is not None else DEFAULT_DECEPTION_THRESHOLD
    feats = extract_deception_features(prompt, response)
    proba = predict_proba_dishonest(feats)
    top = _top_signal_contributions(feats, k=3)

    # Scope warning — surfaces when the verdict is operating outside
    # the v0 calibration domain. The 2026-05-11 self-dogfood found that
    # honest agent task-completion reports (<80 words, terse, factual)
    # score deception_risk ≈ 1.0 driven entirely by their short length.
    # We flag the verdict so downstream consumers can route to
    # deception_check_v2 or skip the heal pass.
    scope_warning: Optional[str] = None
    n_words = len(response.split())
    if (
        proba >= th
        and n_words < SCOPE_WARN_SHORT_RESPONSE_WORDS
        and top
        and top[0][0] == "log_word_count"
        and top[0][2] > 0  # length pushing TOWARD deception, not away
    ):
        scope_warning = SCOPE_WARN_V0_LEXICAL_OOF_SHORT

    return DeceptionVerdict(
        prompt=prompt,
        response=response,
        deception_risk=float(proba),
        shows_signature=bool(proba >= th),
        threshold=th,
        features=feats,
        top_signals=top,
        scope_warning=scope_warning,
    )


__all__ = [
    "DeceptionVerdict",
    "deception_check",
    "FEATURE_NAMES",
    "DEFAULT_DECEPTION_THRESHOLD",
    "MEAN_CV_AUC",
    "STD_CV_AUC",
    "CALIBRATION_FINGERPRINT",
]
