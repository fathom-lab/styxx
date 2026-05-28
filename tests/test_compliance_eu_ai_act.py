# -*- coding: utf-8 -*-
"""Tests for the v0.1 EU AI Act Article 15 / Annex III compliance bridge.

These tests verify structural integrity of the mapping (real citations,
honest construct-ceiling disclosures, kill-gate A3 honored) and
backward compatibility with the legacy `styxx.compliance` v1.3.0 API.
"""
import re

import pytest

import styxx
from styxx.compliance import (
    cite,
    coverage_table,
    uncovered_requirements,
    ComplianceMap,
    PrimitiveCoverage,
    ARTICLE_15_REGISTRY,
)


# ---------------------------------------------------------------------------
# Legacy v1.3.0 API preserved
# ---------------------------------------------------------------------------


def test_legacy_compliance_report_still_importable():
    """The pre-v0.1 styxx.compliance public API must remain importable."""
    from styxx.compliance import AnomalyEvent, ComplianceReport, compliance_report
    assert AnomalyEvent is not None
    assert ComplianceReport is not None
    assert callable(compliance_report)


def test_legacy_top_level_compliance_report():
    """styxx.compliance_report top-level attr (pre-v0.1) is preserved."""
    assert hasattr(styxx, "compliance_report")


# ---------------------------------------------------------------------------
# v0.1 EU AI Act bridge — structural integrity
# ---------------------------------------------------------------------------


def test_registry_is_non_empty_dict():
    assert isinstance(ARTICLE_15_REGISTRY, dict)
    assert len(ARTICLE_15_REGISTRY) >= 4


def test_registry_keys_cite_specific_subparagraphs():
    """Kill-gate A1: each clause must cite a specific Article subparagraph."""
    pattern = re.compile(r"^Article \d+(\.\d+)?(\([a-z]\))?$")
    for clause in ARTICLE_15_REGISTRY:
        assert pattern.match(clause), f"non-specific clause: {clause!r}"


def test_cite_returns_compliance_map():
    m = cite("Article 15.1(a)")
    assert isinstance(m, ComplianceMap)
    assert m.clause == "Article 15.1(a)"
    assert m.requirement_text.startswith("Levels of accuracy")


def test_cite_missing_returns_none():
    assert cite("Article 99.42") is None


def test_coverage_table_returns_tuple():
    table = coverage_table()
    assert isinstance(table, tuple)
    assert all(isinstance(m, ComplianceMap) for m in table)
    assert len(table) == len(ARTICLE_15_REGISTRY)


# ---------------------------------------------------------------------------
# v0.1 kill-gate compliance
# ---------------------------------------------------------------------------


def test_killgate_a3_uncovered_at_least_as_long_as_covered():
    """Kill-gate A3: 'what styxx does NOT cover' >= 'covers'.

    The companion paper pre-states this as a non-negotiable scoping
    discipline. v0.1 ships 7 uncovered vs 4 covered.
    """
    assert len(uncovered_requirements()) >= len(coverage_table()), (
        "uncovered_requirements list must be at least as long as the covered "
        "list (kill-gate A3 from papers/EU_AI_ACT_COMPLIANCE_2026.md)"
    )


def test_every_primitive_has_construct_ceiling():
    """Kill-gate A2: each primitive's failure mode must be disclosed."""
    for m in coverage_table():
        for p in m.styxx_primitives:
            assert isinstance(p, PrimitiveCoverage)
            assert p.construct_ceiling, f"missing construct_ceiling on {p.primitive}"
            assert len(p.construct_ceiling) > 50, (
                f"construct_ceiling for {p.primitive} is suspiciously short — "
                "honest failure modes need substance"
            )


def test_every_primitive_has_commit_receipt():
    """Each primitive must cite a styxx commit hash receipt."""
    sha_re = re.compile(r"^[0-9a-f]{7,40}$")
    for m in coverage_table():
        for p in m.styxx_primitives:
            assert sha_re.match(p.receipt_commit), (
                f"non-SHA receipt_commit on {p.primitive}: {p.receipt_commit!r}"
            )


def test_every_primitive_has_calibrated_metric():
    """No empty calibrated_metric strings."""
    for m in coverage_table():
        for p in m.styxx_primitives:
            assert p.calibrated_metric, f"missing calibrated_metric on {p.primitive}"
            assert len(p.calibrated_metric) > 20


def test_article_15_4_honestly_empty():
    """Article 15.4 (bias/feedback) has NO covered primitives in v0.1.

    The honest empty-coverage is itself a discipline statement — the
    notes field explains why and points to UNCOVERED_REQUIREMENTS.
    """
    m = cite("Article 15.4")
    assert m is not None
    assert m.styxx_primitives == ()
    assert "no styxx-side coverage" in m.notes.lower()


def test_uncovered_requirements_have_alternatives():
    """Every uncovered requirement must point at an alternative tool/methodology."""
    for u in uncovered_requirements():
        assert u.clause
        assert u.reason
        assert u.alternative
        assert len(u.alternative) > 20, (
            f"alternative for {u.clause} is too short to be actionable"
        )


def test_uncovered_clauses_are_specific():
    """Uncovered clauses must cite specific Articles too."""
    pattern = re.compile(r"^Article \d+")
    for u in uncovered_requirements():
        assert pattern.match(u.clause), f"non-specific uncovered clause: {u.clause!r}"


def test_no_legal_advice_in_notes():
    """Compliance maps must disclaim legal advice; no hedging language permitted to be absent."""
    for m in coverage_table():
        assert "not legal advice" in m.notes.lower() or "legal review" in m.notes.lower(), (
            f"ComplianceMap for {m.clause} missing 'not legal advice' / 'legal review' disclaimer"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
