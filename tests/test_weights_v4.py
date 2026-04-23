# -*- coding: utf-8 -*-
"""Tests for styxx.guardrail.calibrated_weights_v4.

v4 is the 8-benchmark cross-validated calibration (3-seed averaged).
It is NOT the default path in ``guardrail.check`` — v3 remains the
default because it peaks better on the common HaluEval-style traffic.
v4 is for callers who want broader cross-domain generalization.
"""
from __future__ import annotations

from styxx.guardrail.calibrated_weights_v4 import (
    LR_COEFS_V4,
    LR_INTERCEPT_V4,
    PER_DATASET_AUC,
    PER_DATASET_AUC_STD,
    CALIBRATION_NOTES,
    predict_proba_v4,
)


def test_v4_has_nine_coefs():
    assert len(LR_COEFS_V4) == 9
    expected = {
        "text_claim_risk", "entity_unverified_frac",
        "knowledge_grounding",
        "content_novelty", "entity_novelty", "number_novelty",
        "bigram_novelty", "trigram_novelty", "nli_contradict",
    }
    assert set(LR_COEFS_V4.keys()) == expected


def test_v4_covers_eight_benchmarks():
    assert len(PER_DATASET_AUC) == 8
    # Must include all 4 HaluEval/TruthfulQA + 4 HaluBench subsets
    expected = {
        "halueval_qa", "halueval_dialogue",
        "halueval_summarization", "truthfulqa",
        "halubench_drop", "halubench_pubmed",
        "halubench_finance", "halubench_ragtruth",
    }
    assert set(PER_DATASET_AUC.keys()) == expected
    assert CALIBRATION_NOTES["n_benchmarks"] == 8


def test_v4_at_least_five_above_065():
    above = sum(1 for a in PER_DATASET_AUC.values() if a >= 0.65)
    assert above >= 5, (
        f"Expected >=5 datasets above AUC 0.65, got {above}. "
        f"Per-dataset: {PER_DATASET_AUC}"
    )


def test_v4_failure_modes_declared():
    """DROP and FinanceBench are known failure modes; must be labeled."""
    declared = set(CALIBRATION_NOTES["documented_failure_modes"])
    assert "halubench_drop" in declared
    assert "halubench_finance" in declared
    # And their AUCs should in fact be below 0.55
    for ds in declared:
        assert PER_DATASET_AUC[ds] < 0.55, (
            f"{ds} is declared a failure mode but AUC is "
            f"{PER_DATASET_AUC[ds]}"
        )


def test_v4_std_matches_auc_keys():
    assert set(PER_DATASET_AUC_STD.keys()) == set(PER_DATASET_AUC.keys())
    for ds, std in PER_DATASET_AUC_STD.items():
        assert std >= 0
        assert std < 0.1, f"{ds} std {std} is suspiciously large"


def test_v4_predict_monotone_in_nli():
    base = {k: 0.2 for k in LR_COEFS_V4}
    base["nli_contradict"] = 0.0
    p_low = predict_proba_v4(base)
    base["nli_contradict"] = 0.9
    p_high = predict_proba_v4(base)
    assert p_high > p_low


def test_v4_zero_signals_near_baseline():
    p = predict_proba_v4({k: 0.0 for k in LR_COEFS_V4})
    assert 0.0 <= p < 0.5  # intercept -0.75 → sigmoid ≈ 0.32
