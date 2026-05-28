# -*- coding: utf-8 -*-
"""Tests for the v0.1 NIST AI RMF Measure-function compliance bridge.

Mirrors the structural-integrity tests in test_compliance_eu_ai_act.py.
Same kill-gates A1, A2, A3 apply; the citation format differs (MS-2.N
instead of Article N.N(x)).
"""
import re

import pytest

from styxx.compliance import nist_ai_rmf
from styxx.compliance.nist_ai_rmf import (
    cite,
    coverage_table,
    uncovered_requirements,
    MEASURE_REGISTRY,
)
from styxx.compliance import ComplianceMap, PrimitiveCoverage


# ---------------------------------------------------------------------------
# Structural integrity
# ---------------------------------------------------------------------------


def test_registry_is_non_empty_dict():
    assert isinstance(MEASURE_REGISTRY, dict)
    assert len(MEASURE_REGISTRY) >= 5


def test_registry_keys_cite_specific_measure_subcategories():
    """Kill-gate A1 (NIST flavor): clauses cite specific MS-N.N subcategories."""
    pattern = re.compile(r"^MS-\d+\.\d+$")
    for clause in MEASURE_REGISTRY:
        assert pattern.match(clause), f"non-specific clause: {clause!r}"


def test_cite_returns_compliance_map():
    m = cite("MS-2.5")
    assert isinstance(m, ComplianceMap)
    assert m.clause == "MS-2.5"
    assert "valid and reliable" in m.requirement_text


def test_cite_ms_2_13_recursive_subcategory():
    """MS-2.13 is the meta-subcategory: TEVV effectiveness. styxx.agent_audit
    is the recursive instrument."""
    m = cite("MS-2.13")
    assert m is not None
    assert any("agent_audit" in p.primitive for p in m.styxx_primitives)


def test_cite_missing_returns_none():
    assert cite("MS-99.42") is None


def test_coverage_table_returns_tuple():
    table = coverage_table()
    assert isinstance(table, tuple)
    assert all(isinstance(m, ComplianceMap) for m in table)
    assert len(table) == len(MEASURE_REGISTRY)


# ---------------------------------------------------------------------------
# Kill-gate compliance
# ---------------------------------------------------------------------------


def test_killgate_a3_uncovered_at_least_as_long_as_covered():
    """A3: uncovered list >= covered list. v0.1 ships 6 uncovered vs 5 covered."""
    assert len(uncovered_requirements()) >= len(coverage_table()), (
        "uncovered_requirements list must be at least as long as the covered "
        "list (kill-gate A3 from papers/EU_AI_ACT_COMPLIANCE_2026.md)"
    )


def test_every_primitive_has_construct_ceiling():
    """A2: each primitive's failure mode is disclosed (>= 50 chars)."""
    for m in coverage_table():
        for p in m.styxx_primitives:
            assert isinstance(p, PrimitiveCoverage)
            assert p.construct_ceiling, f"missing construct_ceiling on {p.primitive}"
            assert len(p.construct_ceiling) > 50, (
                f"construct_ceiling for {p.primitive} is too short"
            )


def test_every_primitive_has_commit_receipt():
    """Each primitive cites a styxx commit hash receipt."""
    sha_re = re.compile(r"^[0-9a-f]{7,40}$")
    for m in coverage_table():
        for p in m.styxx_primitives:
            assert sha_re.match(p.receipt_commit), (
                f"non-SHA receipt_commit on {p.primitive}: {p.receipt_commit!r}"
            )


def test_every_primitive_has_calibrated_metric():
    for m in coverage_table():
        for p in m.styxx_primitives:
            assert p.calibrated_metric
            assert len(p.calibrated_metric) > 20


def test_uncovered_clauses_cite_specific_subcategories():
    pattern = re.compile(r"^MS-\d+\.\d+")
    for u in uncovered_requirements():
        assert pattern.match(u.clause), f"non-specific uncovered clause: {u.clause!r}"


def test_uncovered_requirements_have_alternatives():
    for u in uncovered_requirements():
        assert u.clause
        assert u.reason
        assert u.alternative
        assert len(u.alternative) > 20


def test_no_legal_advice_in_notes():
    """Every ComplianceMap.notes must include 'not legal advice' / 'legal review' disclaimer."""
    for m in coverage_table():
        text = m.notes.lower()
        assert "not legal advice" in text or "independent" in text or "legal review" in text, (
            f"ComplianceMap for {m.clause} missing 'not legal advice' disclaimer"
        )


def test_module_accessible_via_top_level_compliance():
    """styxx.compliance.nist_ai_rmf is importable + the same module object as nist_ai_rmf."""
    from styxx.compliance import nist_ai_rmf as via_compliance
    assert via_compliance is nist_ai_rmf


def test_shared_dataclasses_between_bridges():
    """ComplianceMap from EU AI Act and NIST bridges are the SAME class."""
    from styxx.compliance.eu_ai_act import ComplianceMap as eu_cm
    from styxx.compliance.nist_ai_rmf import ComplianceMap as nist_cm
    from styxx.compliance import ComplianceMap as top_cm
    assert eu_cm is nist_cm is top_cm


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
