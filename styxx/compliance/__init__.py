# -*- coding: utf-8 -*-
"""styxx.compliance — regulatory-grade methodology bridges.

v0.1 ships the EU AI Act Article 15 / Annex III compliance map under
`styxx.compliance.eu_ai_act`. Each citation maps a regulatory clause
to a set of styxx primitives, calibrated metrics, construct ceilings
(honest failure modes), and commit-level reproducibility receipts.

This module IS NOT legal advice. It is a measurement-methodology
bridge: the kind of artifact the EU Commission explicitly invited
stakeholders to develop under EU AI Act Article 15 paragraph 2. Use
in any actual conformity assessment requires independent legal review.

See papers/EU_AI_ACT_COMPLIANCE_2026.md for full methodology, scope,
and limitations.
"""
from __future__ import annotations

# 7.7.10: EU AI Act Article 15 / Annex III compliance map (new in v0.1)
from .eu_ai_act import (
    ComplianceMap,
    PrimitiveCoverage,
    cite,
    coverage_table,
    uncovered_requirements,
    ARTICLE_15_REGISTRY,
)

# Preserve legacy v1.3.0 public surface: AnomalyEvent, ComplianceReport,
# compliance_report. These were a single styxx/compliance.py file before
# v0.1 of the EU AI Act bridge converted compliance into a package.
from ._legacy import (  # noqa: F401
    AnomalyEvent,
    ComplianceReport,
    compliance_report,
)

__all__ = [
    # EU AI Act bridge (v0.1)
    "ComplianceMap",
    "PrimitiveCoverage",
    "cite",
    "coverage_table",
    "uncovered_requirements",
    "ARTICLE_15_REGISTRY",
    # legacy compliance API (preserved)
    "AnomalyEvent",
    "ComplianceReport",
    "compliance_report",
]
