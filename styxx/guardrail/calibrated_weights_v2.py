# -*- coding: utf-8 -*-
"""
Calibrated LR weights v2 — pooled multi-dataset fit.

v1 (calibrated_weights.py) fit LR on HaluEval-QA alone (n=300) with
4 signals. Test AUC 0.9012 on HaluEval-QA, but cross-dataset
validation on 2026-04-22 showed the weights overfit to that
distribution (AUC 0.56-0.63 on dialog, summarization, TruthfulQA).

v2 fits on POOLED HaluEval-QA + HaluEval-Dialog +
HaluEval-Summarization + TruthfulQA (n=200 pairs each, n=800 total)
with 8 signals including four new response-novelty features that
capture "what the response added that the reference doesn't
support":

  - text_claim_risk
  - entity_unverified_frac
  - knowledge_grounding (claim-level, existing)
  - content_novelty     (response-level, new)
  - entity_novelty      (response-level, new)
  - number_novelty      (response-level, new)
  - bigram_novelty      (response-level, new — strong signal)
  - trigram_novelty     (response-level, new — strongest signal)

Held-out per-dataset test AUC (n=100 pairs per dataset, seed 31):

  HaluEval-QA              AUC 1.0000
  HaluEval-Dialog          AUC 0.6014
  HaluEval-Summarization   AUC 0.5954
  TruthfulQA               AUC 0.9767
  ─────────────────────────────────────
  mean                     AUC 0.7934
  min                      AUC 0.5954

Compared to v1:
    dataset                  v1 AUC      v2 AUC
    HaluEval-QA              0.9049  →   1.0000  (+0.095)
    HaluEval-Dialog          0.5984  →   0.6014  (+0.003)
    HaluEval-Summarization   0.5897  →   0.5954  (+0.006)
    TruthfulQA               0.6261  →   0.9767  (+0.351)
    mean                     0.6548  →   0.7934  (+0.139)

Honest interpretation: v2 dramatically improves reference-grounded
QA (QA, TruthfulQA) — the most common LLM use case (RAG). Dialog
and summarization remain harder because they need NLI-style
contradiction detection, not just response-novelty.

For the simpler 4-signal path (when response_novelty is not
computed), use calibrated_weights.py (v1).
"""
from __future__ import annotations

import math
from typing import Dict


# Pooled multi-dataset LR fit — 8 features, L2=0.05, n=800 train
LR_COEFS_V2 = {
    "text_claim_risk":        0.2156,
    "entity_unverified_frac": 0.0000,
    "knowledge_grounding":    0.1346,
    "content_novelty":        0.3781,
    "entity_novelty":         0.1645,
    "number_novelty":        -0.0308,
    "bigram_novelty":         0.4330,
    "trigram_novelty":        0.7970,
}

LR_INTERCEPT_V2 = -0.9992

# Per-dataset test-set AUC at training time (for audit/debug)
PER_DATASET_AUC = {
    "halueval_qa":             1.0000,
    "halueval_dialogue":       0.6014,
    "halueval_summarization":  0.5954,
    "truthfulqa":              0.9767,
}

CALIBRATION_NOTES = {
    "dataset": (
        "pooled HaluEval-QA + HaluEval-Dialog + "
        "HaluEval-Summarization + TruthfulQA"
    ),
    "n_train": 800,
    "n_test_per_dataset": 100,
    "seed": 31,
    "train_frac": 0.75,
    "l2": 0.05,
    "epochs": 800,
    "use_entity_verify": False,
}


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def predict_proba_v2(signals: Dict[str, float]) -> float:
    """Return calibrated hallucination probability in [0, 1].

    Expects all 8 signals in ``signals``; missing ones default to 0
    (fail-open, reduces to a weaker classifier).
    """
    z = LR_INTERCEPT_V2
    for name, coef in LR_COEFS_V2.items():
        z += coef * float(signals.get(name, 0.0))
    return _sigmoid(z)


__all__ = [
    "LR_COEFS_V2",
    "LR_INTERCEPT_V2",
    "PER_DATASET_AUC",
    "CALIBRATION_NOTES",
    "predict_proba_v2",
]
