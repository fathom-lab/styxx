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

# 7.7.10: framework-agnostic dataclasses (shared across jurisdictional bridges)
from ._common import ComplianceMap, PrimitiveCoverage, UncoveredRequirement

# 7.7.10: EU AI Act Article 15 / Annex III compliance map (new in v0.1).
# The default top-level cite/coverage_table/uncovered_requirements bind to
# the EU AI Act bridge for back-compat with the v0.1 release announcement;
# call `styxx.compliance.eu_ai_act.cite()` or `styxx.compliance.nist_ai_rmf.cite()`
# explicitly for jurisdictional clarity.
from . import eu_ai_act  # noqa: F401
from .eu_ai_act import (
    cite,
    coverage_table,
    uncovered_requirements,
    ARTICLE_15_REGISTRY,
)

# 7.7.10: NIST AI RMF 1.0 Measure-function compliance map (new in v0.1)
from . import nist_ai_rmf  # noqa: F401
from .nist_ai_rmf import (
    MEASURE_REGISTRY,
)

# Per-domain competence-cliff accuracy declaration — the literal Article 15.1(a)
# artifact (accuracy levels stratified by domain). Loaded from package data,
# drift-gated in CI against the committed research receipt (styxx@a75f1e7).
from . import competence_cliff as competence_cliff_module  # noqa: F401
from .competence_cliff import (
    competence_cliff,
    CompetenceCliff,
    CategoryAccuracy,
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
    # shared dataclasses
    "ComplianceMap",
    "PrimitiveCoverage",
    "UncoveredRequirement",
    # EU AI Act bridge (default top-level binding)
    "eu_ai_act",
    "cite",
    "coverage_table",
    "uncovered_requirements",
    "ARTICLE_15_REGISTRY",
    # NIST AI RMF bridge
    "nist_ai_rmf",
    "MEASURE_REGISTRY",
    # per-domain competence-cliff accuracy declaration (Article 15.1(a))
    "competence_cliff",
    "CompetenceCliff",
    "CategoryAccuracy",
    # legacy compliance API (preserved)
    "AnomalyEvent",
    "ComplianceReport",
    "compliance_report",
]
