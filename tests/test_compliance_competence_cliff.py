# -*- coding: utf-8 -*-
"""Tests for the per-domain competence-cliff accuracy declaration (Article 15.1(a)).

The load-bearing test here is the **drift gate**: the per-domain numbers shipped
in package data are re-derived from the committed research receipt
(``papers/grounded-honesty-axis/pregeneration_gate_result.json`` at styxx@a75f1e7)
and any divergence fails the build. This is what makes the regulatory accuracy
declaration un-attackable: the shipped number can never silently drift from the
evidence behind it.

The second discipline test is the **anti-rosy gate**: the artifact MUST carry the
FAILED pre-registered bars (continuous AUC 0.619, K_precondition 0.281). A
declaration that hid the layers where the method did not pass would be the exact
overclaim this package exists to prevent — so we assert the honest bounds are
present and stay marked FAILED.
"""
import json
from pathlib import Path

import pytest

from styxx.compliance import cite, competence_cliff, CompetenceCliff, CategoryAccuracy
from styxx.compliance.competence_cliff import (
    _tier_for,
    SAFE_MIN_COMMITTED_PRECISION,
    REVIEW_MIN_COMMITTED_PRECISION,
)


# Repo-root-relative path to the committed research receipt. Present in the source
# tree / CI checkout; absent from the installed wheel (papers/ is wheel-excluded),
# so the drift gate skips gracefully when run against an installed package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_RECEIPT = _REPO_ROOT / "papers" / "grounded-honesty-axis" / "pregeneration_gate_result.json"
_BENCHMARK = _REPO_ROOT / "papers" / "grounded-honesty-axis" / "truthfulqa_benchmark_result.json"


@pytest.fixture(scope="module")
def cliff() -> CompetenceCliff:
    return competence_cliff()


# ---------------------------------------------------------------------------
# Shape / loading
# ---------------------------------------------------------------------------


def test_loads_thirty_seven_domains(cliff):
    assert cliff.n_categories == 37
    assert len(cliff.categories) == 37
    assert cliff.model == "gpt-4o-mini"
    assert cliff.n_items == 790
    assert all(isinstance(c, CategoryAccuracy) for c in cliff.categories)


def test_categories_sorted_descending_precision(cliff):
    precisions = [c.committed_precision for c in cliff.categories]
    assert precisions == sorted(precisions, reverse=True)


def test_internal_consistency(cliff):
    for c in cliff.categories:
        assert 0 <= c.committed_n <= c.n
        assert 0.0 <= c.committed_precision <= 1.0
        assert 0.0 <= c.refusal_rate <= 1.0


# ---------------------------------------------------------------------------
# Deploy-tier derivation
# ---------------------------------------------------------------------------


def test_tier_counts(cliff):
    bt = cliff.by_tier()
    assert len(bt["safe"]) == 17
    assert len(bt["review"]) == 17
    assert len(bt["do_not_deploy"]) == 3


def test_do_not_deploy_domains(cliff):
    names = {c.category for c in cliff.do_not_deploy()}
    assert names == {"Language", "Distraction", "Superstitions"}
    for c in cliff.do_not_deploy():
        assert c.committed_precision < REVIEW_MIN_COMMITTED_PRECISION


def test_tier_derivation_matches_thresholds(cliff):
    for c in cliff.categories:
        assert c.deploy_tier == _tier_for(c.committed_precision)
        if c.deploy_tier == "safe":
            assert c.committed_precision >= SAFE_MIN_COMMITTED_PRECISION
        elif c.deploy_tier == "review":
            assert REVIEW_MIN_COMMITTED_PRECISION <= c.committed_precision < SAFE_MIN_COMMITTED_PRECISION
        else:
            assert c.committed_precision < REVIEW_MIN_COMMITTED_PRECISION


# ---------------------------------------------------------------------------
# DRIFT GATE — shipped numbers must equal the committed research receipt
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _RECEIPT.exists(), reason="research receipt not in installed wheel")
def test_shipped_numbers_match_committed_receipt(cliff):
    """Every shipped per-domain number re-derives from the committed receipt.

    Fails the build if the package-data declaration drifts from the evidence.
    """
    receipt = json.loads(_RECEIPT.read_text(encoding="utf-8"))
    src = receipt["category_competence_cliff_map"]
    assert len(src) == len(cliff.categories), "domain count drifted from receipt"
    for c in cliff.categories:
        ref = src[c.category]
        assert c.n == ref["n"], f"{c.category}: n drift"
        assert c.committed_n == ref["committed_n"], f"{c.category}: committed_n drift"
        assert c.committed_precision == ref["committed_precision"], f"{c.category}: precision drift"
        assert c.useful_answer_rate == ref["useful_answer_rate"], f"{c.category}: useful_rate drift"
        assert c.refusal_rate == ref["refusal_rate"], f"{c.category}: refusal_rate drift"
        assert c.ungated_hallucination_rate == ref["ungated_hallucination_rate"], f"{c.category}: halluc drift"


@pytest.mark.skipif(not _RECEIPT.exists(), reason="research receipt not in installed wheel")
def test_gate_thresholds_match_receipt(cliff):
    receipt = json.loads(_RECEIPT.read_text(encoding="utf-8"))
    assert cliff.gate_stability_threshold == receipt["gate_stability_threshold"]
    assert cliff.gate_dominance_threshold == receipt["gate_dominance_threshold"]
    assert cliff.refusal_rate == receipt["refusal_rate"]
    assert cliff.gate_committed_precision == receipt["bars"]["C3_committed_precision"]["committed_precision"]


@pytest.mark.skipif(not _BENCHMARK.exists(), reason="research receipt not in installed wheel")
def test_answer_key_sha_matches_benchmark_receipt(cliff):
    """The artifact's answer-key hash ties it to the exact benchmark question set."""
    benchmark = json.loads(_BENCHMARK.read_text(encoding="utf-8"))
    expected = benchmark.get("answer_key_sha256_expected") or benchmark.get("answer_key_sha256")
    assert cliff.answer_key_sha256 == expected


@pytest.mark.skipif(not (_RECEIPT.exists() and _BENCHMARK.exists()), reason="research receipts not in installed wheel")
def test_failed_bars_re_derive_from_receipts(cliff):
    """The FAILED pre-registered bars (continuous AUC, K_precondition) must RE-DERIVE from their
    committed receipts, not merely sit inside `<` bounds.

    Without this the drift-gate's provenance claim ("every shipped figure re-derives from the
    receipt; the declared accuracy can never silently drift") would be false in scope: the two
    FAILED numbers are package-data literals, and an attacker could edit 0.6191 -> 0.649 (still
    `< 0.65`, still tagged FAILED) and CI would stay green. This ties them to source.
    """
    bench = json.loads(_BENCHMARK.read_text(encoding="utf-8"))
    gate = json.loads(_RECEIPT.read_text(encoding="utf-8"))
    assert cliff.continuous_auc_value == round(bench["bars"]["H1"]["auc_merged"], 4), \
        "continuous_auc_value drifted from the benchmark receipt (bars.H1.auc_merged)"
    assert cliff.k_precondition_value == round(gate["bars"]["K_precondition"]["ungated_hallucination_rate"], 4), \
        "k_precondition_value drifted from the gate receipt (bars.K_precondition.ungated_hallucination_rate)"


# ---------------------------------------------------------------------------
# ANTI-ROSY GATE — the honest (FAILED) bounds must stay present
# ---------------------------------------------------------------------------


def test_failed_bars_are_carried(cliff):
    """The declaration must keep naming the layers that FAILED pre-registration."""
    assert cliff.continuous_auc_outcome == "FAILED"
    assert cliff.continuous_auc_value < 0.65  # below the REPORT floor it was tested at
    assert cliff.k_precondition_outcome == "FAILED"
    assert cliff.k_precondition_value < 0.30
    assert "REPORT_AS_LANDED" in cliff.discipline_note or "NO SURVIVED CLAIM" in cliff.discipline_note


def test_markdown_names_failures_and_do_not_deploy(cliff):
    md = cliff.as_markdown()
    assert "FAILED" in md  # honest bounds rendered, not hidden
    assert "DO NOT DEPLOY" in md
    for name in ("Language", "Distraction", "Superstitions"):
        assert name in md
    assert "Not legal advice" in md
    assert cliff.receipt_commit in md


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_cite_15_1a_includes_competence_cliff():
    m = cite("Article 15.1(a)")
    assert m is not None
    by_name = {p.primitive: p for p in m.styxx_primitives}
    assert "styxx.compliance.competence_cliff" in by_name
    p = by_name["styxx.compliance.competence_cliff"]
    assert p.receipt_commit == "a75f1e7"
    # construct ceiling must disclose the REPORT_AS_LANDED / not-SURVIVED status
    assert "SURVIVED" in p.construct_ceiling


def test_to_dict_roundtrip(cliff):
    d = cliff.to_dict()
    assert len(d["categories"]) == 37
    assert d["model"] == "gpt-4o-mini"
    assert d["continuous_auc_outcome"] == "FAILED"
    assert d["categories"][0]["committed_precision"] >= d["categories"][-1]["committed_precision"]
