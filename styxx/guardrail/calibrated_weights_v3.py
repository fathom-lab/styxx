# -*- coding: utf-8 -*-
"""
Calibrated LR weights v3 — 9-signal fit with NLI contradiction.

v2 (`calibrated_weights_v2.py`) was the 8-signal pooled LR trained on
HaluEval-QA + HaluEval-Dialog + HaluEval-Summarization + TruthfulQA.
It solved reference-grounded QA (AUC 1.000 / 0.977) but dialog and
summarization remained near chance (AUC 0.60) because
response-novelty signals cannot separate faithful dialog additions
from contradictions.

v3 adds a ninth feature: NLI contradiction probability. A small
entailment model (``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli``,
~184M params) scores (reference → response) pairs and returns the
probability that the response is contradicted by the reference.

Held-out per-dataset test AUC at v4.0.0rc1 calibration (n=200
pairs/dataset, train/test 0.75/0.25, L2=0.05, 3-seed averaged
over seeds [31, 47, 83]):

  HaluEval-QA              AUC 0.9963 ± 0.002   (v2: 1.000, -0.004)
  HaluEval-Dialog          AUC 0.7288 ± 0.042   (v2: 0.605, +0.123)
  HaluEval-Summarization   AUC 0.6646 ± 0.029   (v2: 0.636, +0.028)
  TruthfulQA               AUC 0.9952 ± 0.004   (v2: 0.977, +0.018)
  ────────────────────────────────────────────────
  mean                     AUC 0.8462           (v2: 0.8047, +0.041)

The NLI coefficient (``nli_contradict``: 0.878 avg) is now the
strongest single signal, exceeding ``trigram_novelty`` (0.773).
The two are complementary, not redundant: n-gram novelty catches
"the response added content not in the reference"; NLI catches "the
response asserts something the reference denies." Dialog and
summarization hallucinations are dominated by the latter, which is
why adding NLI produces the biggest gain there (+0.123 on dialog).

Honest interpretation: v3 lifts dialog/summarization from near-chance
to real signal territory, while preserving the ~1.0 AUC on
reference-grounded QA. Dialog/summ have not reached the 0.80+ ceiling
we want for v4.0 final — but they are unambiguously above chance on a
proper held-out test, which v2 was not.

v3 is a **preview** calibration. Multi-seed averaging and extension to
8 benchmark datasets (FEVER, FactCC, XSum-Faithful, PHD-A) land in
v4.0 final.

v3 is only used when both ``response_novelty`` AND an NLI
contradiction score are available. When NLI is absent (no torch,
no reference, or load failure), the pipeline falls back to v2.
"""
from __future__ import annotations

import math
from typing import Dict


# 9-feature pooled LR fit (3-seed averaged, preview calibration).
LR_COEFS_V3 = {
    "text_claim_risk":        0.1751,
    "entity_unverified_frac": 0.0000,
    "knowledge_grounding":    0.1231,
    "content_novelty":        0.3368,
    "entity_novelty":         0.1353,
    "number_novelty":         0.0333,
    "bigram_novelty":         0.4104,
    "trigram_novelty":        0.7727,
    "nli_contradict":         0.8784,
}

LR_INTERCEPT_V3 = -1.1257

# Per-dataset test-set AUC (3-seed mean + std) for audit/debug.
PER_DATASET_AUC = {
    "halueval_qa":             0.9963,
    "halueval_dialogue":       0.7288,
    "halueval_summarization":  0.6646,
    "truthfulqa":              0.9952,
}

PER_DATASET_AUC_STD = {
    "halueval_qa":             0.0022,
    "halueval_dialogue":       0.0418,
    "halueval_summarization":  0.0292,
    "truthfulqa":              0.0041,
}

CALIBRATION_NOTES = {
    "dataset": (
        "pooled HaluEval-QA + HaluEval-Dialog + "
        "HaluEval-Summarization + TruthfulQA"
    ),
    "nli_model": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "n_train": 600,     # per seed
    "n_test_per_dataset": 50,   # per seed
    "seeds": [31, 47, 83],
    "averaged": True,
    "train_frac": 0.75,
    "l2": 0.05,
    "epochs": 800,
    "use_entity_verify": False,
    "preview": True,
    "planned_final": "v4.0.0 (8-dataset cross-validation)",
}


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def predict_proba_v3(signals: Dict[str, float]) -> float:
    """Return calibrated hallucination probability in [0, 1].

    Expects all 9 signals in ``signals``; missing ones default to 0
    (fail-open, reduces to a weaker classifier — but v3 is only
    preferred when ``nli_contradict`` is present; see
    ``guardrail.entry.check``).
    """
    z = LR_INTERCEPT_V3
    for name, coef in LR_COEFS_V3.items():
        z += coef * float(signals.get(name, 0.0))
    return _sigmoid(z)


__all__ = [
    "LR_COEFS_V3",
    "LR_INTERCEPT_V3",
    "PER_DATASET_AUC",
    "PER_DATASET_AUC_STD",
    "CALIBRATION_NOTES",
    "predict_proba_v3",
]
