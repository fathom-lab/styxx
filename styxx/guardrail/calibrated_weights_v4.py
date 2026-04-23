# -*- coding: utf-8 -*-
"""
Calibrated LR weights v4 — 8-benchmark cross-validated fit.

v3 (``calibrated_weights_v3.py``) was the 9-signal pooled LR trained
on 4 benchmarks (HaluEval-QA, HaluEval-Dialog, HaluEval-Summ,
TruthfulQA). v4 extends calibration to **8 benchmarks** by adding
4 new domains sourced from PatronusAI's public HaluBench:

  - DROP          (reading comprehension QA, n=300 per seed)
  - PubMedQA      (biomedical QA, n=300)
  - FinanceBench  (financial document QA, n=300)
  - RAGTruth      (RAG-style retrieval faithfulness, n=300)

Held-out per-dataset test AUC at v4.0.0 calibration (3-seed mean ± std,
seeds [31, 47, 83], L2=0.05, train/test 0.75/0.25, n=150/dataset
target):

  HaluEval-QA              AUC 0.998 ± 0.001
  TruthfulQA               AUC 0.994 ± 0.006
  HaluBench-RAGTruth       AUC 0.807 ± 0.043   ← new
  HaluBench-PubMedQA       AUC 0.719 ± 0.051   ← new
  HaluEval-Dialog          AUC 0.676 ± 0.037
  HaluEval-Summarization   AUC 0.643 ± 0.060
  HaluBench-FinanceBench   AUC 0.492 ± 0.026   ← below chance
  HaluBench-DROP           AUC 0.424 ± 0.080   ← below chance
  ────────────────────────────────────────────
  overall mean             AUC 0.719

**Honest interpretation.** 5/8 benchmarks above AUC 0.65. 3 new
domains add real signal (RAGTruth 0.81, PubMedQA 0.72, Dialog holds).
Two new domains — reading comprehension (DROP) and financial
document QA (FinanceBench) — fall BELOW chance. These are published
failure modes, not bugs:

  - **DROP.** Answers are extractive spans from the passage. An
    extractive hallucination (wrong span) is entailed by its parent
    passage, so NLI scores it as non-contradictory. Novelty signals
    fail for the same reason. Future work: span-level faithfulness
    models.

  - **FinanceBench.** Hallucinations are usually calculation/aggregation
    errors on numbers copied verbatim from the passage. Novelty and
    NLI are semantically blind to arithmetic correctness. Future work:
    number-symbolic verification signals.

v4 is strictly broader-domain than v3 but marginally weaker on dialog
(-0.05) and summarization (-0.04) because the pooled fit now averages
across domains where the existing signal mix fundamentally does not
apply. When the caller knows their traffic is reference-grounded QA
or dialog, ``predict_proba_v3`` remains the more peaked classifier.
``predict_proba_v4`` is the more honest cross-domain default.

This is the **v4.0.0 final** calibration. The matching paper is
"Cognometry v0: 8-benchmark cross-validated hallucination detection"
on Zenodo.
"""
from __future__ import annotations

import math
from typing import Dict


# 9-feature pooled LR fit (3-seed averaged across 8 benchmarks).
LR_COEFS_V4 = {
    "text_claim_risk":        0.1733,
    "entity_unverified_frac": 0.0000,
    "knowledge_grounding":    0.0792,
    "content_novelty":        0.2551,
    "entity_novelty":         0.1315,
    "number_novelty":         0.1271,
    "bigram_novelty":         0.1867,
    "trigram_novelty":        0.4943,
    "nli_contradict":         0.5570,
}

LR_INTERCEPT_V4 = -0.7518

# Per-dataset test-set AUC (3-seed mean + std) for audit/debug.
PER_DATASET_AUC = {
    "halueval_qa":             0.9984,
    "truthfulqa":              0.9938,
    "halubench_ragtruth":      0.8066,
    "halubench_pubmed":        0.7193,
    "halueval_dialogue":       0.6763,
    "halueval_summarization":  0.6427,
    "halubench_finance":       0.4917,
    "halubench_drop":          0.4238,
}

PER_DATASET_AUC_STD = {
    "halueval_qa":             0.0007,
    "truthfulqa":              0.0063,
    "halubench_ragtruth":      0.0433,
    "halubench_pubmed":        0.0507,
    "halueval_dialogue":       0.0367,
    "halueval_summarization":  0.0599,
    "halubench_finance":       0.0259,
    "halubench_drop":          0.0796,
}

CALIBRATION_NOTES = {
    "dataset": (
        "pooled HaluEval-QA + HaluEval-Dialog + "
        "HaluEval-Summarization + TruthfulQA + "
        "HaluBench(DROP+pubmedQA+FinanceBench+RAGTruth)"
    ),
    "n_benchmarks": 8,
    "nli_model": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "n_train_pooled": 1800,     # per seed
    "n_test_pooled": 600,       # per seed
    "seeds": [31, 47, 83],
    "averaged": True,
    "train_frac": 0.75,
    "l2": 0.05,
    "epochs": 800,
    "use_entity_verify": False,
    "overall_mean_auc": 0.7191,
    "datasets_above_0_65": 5,
    "documented_failure_modes": ["halubench_drop", "halubench_finance"],
}


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def predict_proba_v4(signals: Dict[str, float]) -> float:
    """Return calibrated hallucination probability in [0, 1].

    The v4 pipeline is cross-validated on 8 benchmarks. Use
    ``predict_proba_v3`` if your traffic is dominated by HaluEval-style
    dialog/summarization and you want the more peaked classifier.
    """
    z = LR_INTERCEPT_V4
    for name, coef in LR_COEFS_V4.items():
        z += coef * float(signals.get(name, 0.0))
    return _sigmoid(z)


__all__ = [
    "LR_COEFS_V4",
    "LR_INTERCEPT_V4",
    "PER_DATASET_AUC",
    "PER_DATASET_AUC_STD",
    "CALIBRATION_NOTES",
    "predict_proba_v4",
]
