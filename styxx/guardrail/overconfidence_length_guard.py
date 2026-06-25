# -*- coding: utf-8 -*-
"""Length-aware deployment guard for the overconfidence_v0 instrument (OPT-IN).

The shipped overconfidence_v0 score carries a length bias: it learned SHORT = overconfident, because the v0
training corpus's calibrated class was longer (hedging costs words). On an orthogonal register x length 2x2
(FINDING_overconfidence_adversarial_lenxreg_2026_06_25) this produced a MEASURED 4% -> 46% false-positive /
false-negative swing by length at a fixed threshold, even though within-stratum discrimination is robust
(AUC 0.83-0.86). This guard corrects the SCORE for length so a deployment threshold is length-fair, WITHOUT
retraining the instrument — the register signal (hedge/certainty words) is left untouched. (Force-ablating the
length features was a pre-registered HONEST NULL: length is partly register-intrinsic, so removing it deletes
real signal. The fix is a deployment-side score correction, not a retrain.)

Correction (operating-point preserving):

    adjusted = raw_score - SLOPE * (log1p(n_words) - REF_LOG_WORDS)

- ``SLOPE`` is fit on the register-BALANCED 2x2 (register is orthogonal to length there, so the slope estimates
  the PURE length effect on the score, free of the register-length confound).
- ``REF_LOG_WORDS`` is the mean log-word-count of the v0 training corpus (~69 words), so text of TYPICAL length
  is left unchanged and only DEVIATIONS from typical length are corrected (operating point preserved).

Validation (see the finding): collapses the 2x2 false-positive length-disparity (-0.45 -> +0.07; the mitigation
holds 5-fold OUT-OF-SAMPLE, AUC 0.807 -> 0.843); preserves register discrimination on length-matched data; and
on the confounded v0 corpus moves AUC from the confound-INFLATED 0.796 toward its true length-robust value
(~0.725 from CEM) — it removes inflation, it does not degrade the real signal.

SCOPE / why OPT-IN: coefficients come from a single frontier generator (Gemini 2.5-flash), n=200, single seed.
On length-CONFOUNDED inputs the guard trades some confound-inflated headline AUC for length-fairness — the right
trade for a guardrail (a careful terse answer must not be auto-flagged), but a DELIBERATE one, so it is not a
default. Use it when length-fairness matters; a production deployment may refit SLOPE/REF on its own
register-balanced reference corpus.
"""
from __future__ import annotations
import math

# Frozen 2026-06-25 from the orthogonal register x length 2x2 (Gemini 2.5-flash, n=200).
SLOPE = -2.1558            # overconfidence score per log-word (pure length effect, register held orthogonal)
REF_LOG_WORDS = 4.2454     # mean log1p(word_count) of the v0 training corpus (~69 words)
SOURCE = "FINDING_overconfidence_adversarial_lenxreg_2026_06_25"


def length_adjust(raw_score: float, n_words: int) -> float:
    """Length-correct an overconfidence_v0 score (operating-point preserving).

    Parameters
    ----------
    raw_score : the overconfidence_v0 logit/score for a response.
    n_words   : that response's word count (>= 0).

    Returns the length-adjusted score. At ~REF length the score is unchanged; SHORT text is pulled DOWN
    (correcting the short=overconfident bias) and LONG text nudged up. See the module docstring for validation
    and scope. This does NOT change the instrument's feature weights — it is a deployment-side correction.
    """
    if n_words < 0:
        raise ValueError("n_words must be >= 0")
    return float(raw_score - SLOPE * (math.log1p(n_words) - REF_LOG_WORDS))
