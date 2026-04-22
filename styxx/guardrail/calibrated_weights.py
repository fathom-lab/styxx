# -*- coding: utf-8 -*-
"""
Calibrated logistic-regression weights for the 4-signal guardrail
fusion, fit on HaluEval-QA dev (n=300) and evaluated on held-out
test (n=230) with question-level deduplication against dev.

Reported metrics on held-out test:
  AUC = 0.9012
  threshold=0.5: precision 0.873, recall 0.839, F1 0.856
  threshold=0.7: precision 1.000, recall 0.578 (zero false positives)

Signal order (must match entry.py signal extraction):
  0. text_claim_risk
  1. entity_unverified_frac
  2. knowledge_grounding
  3. probe_confab

Usage:

    from .calibrated_weights import LR_COEFS, LR_INTERCEPT, predict_proba
    p = predict_proba({
        "text_claim_risk": 0.35,
        "entity_unverified_frac": 0.10,
        "knowledge_grounding": 0.25,
        "probe_confab": 0.55,
    })
    # p ∈ [0, 1] — calibrated probability of hallucination
"""
from __future__ import annotations

import math
from typing import Dict

# Fit on HaluEval-QA n_dev=300 seed 11, evaluated on n_test=230 seed 17
# (question-dedupled against dev). Logistic regression L2 C=1.0.
LR_COEFS = {
    "text_claim_risk":        1.4887,
    "entity_unverified_frac": 1.4331,
    "knowledge_grounding":    8.2097,
    "probe_confab":           1.3469,
}
LR_INTERCEPT = -3.4586

# Benchmark results
FIT_METADATA = {
    "benchmark": "HaluEval-QA",
    "n_dev": 300,
    "n_test": 230,
    "seed_dev": 11,
    "seed_test": 17,
    "dedup_by_question": True,
    "dev_auc": 0.9411,
    "test_auc": 0.9012,
    "model_for_probe": "meta-llama/Llama-3.2-1B-Instruct",
    "probe_task": "halueval",
}


def predict_proba(signals: Dict[str, float],
                   default_if_missing: float = 0.5) -> float:
    """Return calibrated P(hallucination) from signal dict.

    Any signal missing from the input dict defaults to
    `default_if_missing` (0.5 = neutral).
    """
    logit = LR_INTERCEPT
    for name, coef in LR_COEFS.items():
        val = signals.get(name, default_if_missing)
        if val is None:
            val = default_if_missing
        logit += coef * val
    # sigmoid
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    ex = math.exp(logit)
    return ex / (1.0 + ex)


__all__ = ["LR_COEFS", "LR_INTERCEPT", "FIT_METADATA", "predict_proba"]
