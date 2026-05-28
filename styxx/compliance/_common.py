# -*- coding: utf-8 -*-
"""Framework-agnostic dataclasses used by all `styxx.compliance.*` bridges.

The same three structures map regulatory clauses to styxx primitives for any
jurisdictional framework (EU AI Act, NIST AI RMF, future additions). They are
deliberately regulation-agnostic — the `clause` field is a free-form citation
string that each bridge module fills with its framework's native citation
format (e.g., "Article 15.1(a)" for EU AI Act, "MS-2.5" for NIST AI RMF).
"""
from __future__ import annotations

from dataclasses import dataclass


__all__ = ["PrimitiveCoverage", "ComplianceMap", "UncoveredRequirement"]


@dataclass(frozen=True)
class PrimitiveCoverage:
    """One styxx primitive's coverage of one regulatory clause.

    Attributes:
        primitive: the styxx public API symbol (e.g., 'styxx.critique_detector')
        calibrated_metric: a published AUC/accuracy number with context
        construct_ceiling: the honest failure mode (e.g., 'restrained-tech FPR 0.30')
        receipt_commit: the styxx git commit that produced the metric (short SHA)
        receipt_doc: path within the styxx repo with the validating FINDING
    """
    primitive: str
    calibrated_metric: str
    construct_ceiling: str
    receipt_commit: str
    receipt_doc: str


@dataclass(frozen=True)
class ComplianceMap:
    """Coverage of one regulatory clause by styxx primitives.

    Attributes:
        clause: regulatory citation in the framework's native format
            (e.g., 'Article 15.1(a)' for EU AI Act, 'MS-2.5' for NIST AI RMF)
        requirement_text: short description of the requirement (the
            framework's own wording, paraphrased lightly for length)
        styxx_primitives: list of PrimitiveCoverage entries
        notes: methodology notes for this clause (always includes
            "not legal advice" reminder)
    """
    clause: str
    requirement_text: str
    styxx_primitives: tuple[PrimitiveCoverage, ...]
    notes: str = "v0.1 mapping. Not legal advice. Independent conformity review required."


@dataclass(frozen=True)
class UncoveredRequirement:
    """A regulatory requirement styxx does NOT cover, honestly enumerated.

    Attributes:
        clause: regulatory citation in the framework's native format
        reason: why styxx primitives do not produce relevant evidence
        alternative: tool or methodology operators should consult instead
    """
    clause: str
    reason: str
    alternative: str
