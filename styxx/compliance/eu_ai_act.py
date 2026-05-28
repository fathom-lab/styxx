# -*- coding: utf-8 -*-
"""EU AI Act Article 15 / Annex III compliance map for styxx primitives.

This is the v0.1 measurement-methodology bridge described in
papers/EU_AI_ACT_COMPLIANCE_2026.md.

USAGE:

    from styxx.compliance import cite, coverage_table, uncovered_requirements

    # Look up which styxx primitives produce evidence for a clause:
    m = cite("Article 15.1(a)")
    print(m.requirement_text)
    print(m.styxx_primitives)
    print(m.construct_ceilings)
    print(m.reproducibility_receipts)

    # Full coverage table (all mapped clauses):
    for row in coverage_table():
        print(row.clause, row.styxx_primitives)

    # Honest list of EU AI Act requirements styxx does NOT cover:
    for u in uncovered_requirements():
        print(u.clause, u.reason)

DESIGN PRINCIPLES (also pre-stated in the companion paper):
  - Each map entry cites a SPECIFIC Article paragraph or sub-paragraph,
    not generic compliance language.
  - Each entry lists construct ceilings (known failure modes) explicitly,
    not hidden in caveats.
  - The uncovered_requirements() list MUST be at least as long as the
    covered list (kill-gate A3 from the companion paper): the boundary
    of applicability is explicit.
  - Reproducibility receipts cite styxx commit hashes that produced the
    calibrated metrics. The receipts are checkable via git.

NOT legal advice. NOT a substitute for an organization's own AI Act
conformity assessment. Independent legal review required for any
production deployment.
"""
from __future__ import annotations

from typing import Optional

from ._common import ComplianceMap, PrimitiveCoverage, UncoveredRequirement


__all__ = [
    "ComplianceMap",
    "PrimitiveCoverage",
    "cite",
    "coverage_table",
    "uncovered_requirements",
    "ARTICLE_15_REGISTRY",
    "UNCOVERED_REQUIREMENTS",
]


# ---------------------------------------------------------------------------
# Coverage registry (Article 15 — Accuracy, Robustness and Cybersecurity)
# ---------------------------------------------------------------------------

# All commit hashes below are at fathom-lab/styxx@main as of this module's
# commit. Future-style usage: re-run reproducibility scripts to verify.

_COGNOMETRIC_CARD = PrimitiveCoverage(
    primitive="styxx.cognometric_card",
    calibrated_metric=(
        "HaluEval-QA AUC 0.998 (mean over 150 items, seeds 31/47/83); "
        "XSTest refusal AUC 0.976; BFCL v3 tool-drift AUC 0.943"
    ),
    construct_ceiling=(
        "Text-only register space has construct ceilings: instruments measure "
        "register, not always content. Sycophancy false-positive 0.30 on "
        "restrained-tech responses (gpt-3.5: 0.60). Logprob-validity works "
        "for refusal but fails for hallucination (confident confabulation). "
        "See feedback_register_pareto_frontier and "
        "project_grounded_arc_bet0_engine for closed-negative boundary maps."
    ),
    receipt_commit="cf14c83",
    receipt_doc="submissions/_results/leaderboard.json",
)

_CRITIQUE_DETECTOR = PrimitiveCoverage(
    primitive="styxx.critique_detector(model='gpt-4o-mini')",
    calibrated_metric=(
        "Baseline-019 PASSes the gauntlet's v3 detection bars 4/4 at AUC 0.95 "
        "on the dark-core consensus-hallucination benchmark (n=108 records, "
        "34 folklore subset). Pre-stated 28% probability landed cleanly."
    ),
    construct_ceiling=(
        "Mechanism is 'out-of-context critique' (RLHF discrimination on labeled "
        "candidate text), NOT within-model generation-vs-critique asymmetry. "
        "The true within-model asymmetry rate is 5.88% on dark-core / 17.00% "
        "on TruthfulQA (v3 measurement, single-character T/F/U NLI). "
        "In-council bias caveat: the default backend gpt-4o-mini was IN the "
        "original 3-vendor council that generated expected_consensus. For "
        "cross-vendor robustness use a different-vendor judge model."
    ),
    receipt_commit="1ab0e22",
    receipt_doc="papers/agent-self-audit/FINDING_asymmetry_v3_measurement_2026_05_27.md",
)

_AGENT_AUDIT = PrimitiveCoverage(
    primitive="styxx.agent_audit",
    calibrated_metric=(
        "Substrate-grounded session-output verifier. Pre-registered against "
        "13 logical claims at commit 1b82e27 BEFORE the instrument existed "
        "(commit 3c24b5e). Result: 13/13 PASS modal pre-stated outcome. L7 "
        "uncurated extension (prereg b18ce93, run cf14c83) caught real "
        "off-by-one count drift in the recursive-discipline paper (2 FAILs "
        "as pre-disclosed). 9 registered checkers covering commits, files, "
        "JSON paths, Python attributes, PDF pages, directory counts."
    ),
    construct_ceiling=(
        "First-occurrence-only by default — caught initial drift but missed "
        "systematic propagation to 4 additional places where the same "
        "off-by-one appeared. Richer multi-occurrence checker is methodology "
        "future work, not v0.1. Limited to substrate-checkable claims; "
        "does not score interpretive, causal, or generalization claims."
    ),
    receipt_commit="3c24b5e",
    receipt_doc="papers/agent-self-audit/FINDING_agent_claim_audit_2026_05_28.md",
)

_RECOVER_POSTURE = PrimitiveCoverage(
    primitive="styxx.recover_posture",
    calibrated_metric=(
        "Cognitive-integrity persistence primitive for agents crossing "
        "context-compaction boundaries. v1 reads what chart.jsonl persists; "
        "PostureSummary dataclass + MCP tool cogn_recover_posture."
    ),
    construct_ceiling=(
        "Honest v1 scope: does not include cogn_audit scores. Falsifiable "
        "claim attached pending bet-2 outcome study. Not validated on "
        "third-party agent platforms."
    ),
    receipt_commit="ee6e49d",
    receipt_doc="project_recover_posture_shipped (memory reference)",
)

_GAUNTLET_PREFLIGHT = PrimitiveCoverage(
    primitive="styxx.gauntlet + styxx.preflight",
    calibrated_metric=(
        "v3 detection bars (D1+D2+D3+D4) with regression-tested oracles. "
        "Eighteen pre-registered baselines tested (Baseline-002 through "
        "Baseline-019); seventeen FAILed bars before Baseline-019 PASSed. "
        "Confound audit primitive scans 8 default oracle features."
    ),
    construct_ceiling=(
        "Bars catch confounds the project pre-stated; do not catch unknown-"
        "unknown confound classes. Each bar revision documents one prior "
        "missed confound. The discipline is the recursion (bars catch "
        "themselves on session timescale), not a terminal guarantee."
    ),
    receipt_commit="d8f4843",
    receipt_doc="papers/agent-self-audit/FINDING_confound_audit_2026_05_27.md",
)


# Article 15 registry: each entry maps a sub-paragraph to styxx primitives
# that produce relevant measurement evidence. Citations follow the
# numbering at https://artificialintelligenceact.eu/article/15/.
ARTICLE_15_REGISTRY: dict[str, ComplianceMap] = {
    "Article 15.1": ComplianceMap(
        clause="Article 15.1",
        requirement_text=(
            "High-risk AI systems shall be designed and developed in such a "
            "way that they achieve an appropriate level of accuracy, "
            "robustness and cybersecurity, and that they perform "
            "consistently in those respects throughout their lifecycle."
        ),
        styxx_primitives=(_COGNOMETRIC_CARD, _GAUNTLET_PREFLIGHT, _AGENT_AUDIT),
    ),
    "Article 15.1(a)": ComplianceMap(
        clause="Article 15.1(a)",
        requirement_text=(
            "Levels of accuracy and relevant accuracy metrics shall be "
            "declared in the accompanying instructions of use."
        ),
        styxx_primitives=(_COGNOMETRIC_CARD, _CRITIQUE_DETECTOR, _GAUNTLET_PREFLIGHT),
    ),
    "Article 15.3": ComplianceMap(
        clause="Article 15.3",
        requirement_text=(
            "Robustness against errors, faults, or inconsistencies, "
            "including via technical-redundancy solutions such as "
            "backup or fail-safe plans."
        ),
        styxx_primitives=(_RECOVER_POSTURE, _AGENT_AUDIT),
    ),
    "Article 15.4": ComplianceMap(
        clause="Article 15.4",
        requirement_text=(
            "Bias and feedback-loop mitigation for high-risk AI systems "
            "that continue to learn after being placed on the market or "
            "put into service."
        ),
        # styxx does NOT have first-class bias-amplification or
        # feedback-loop primitives. The honest map is empty here; see
        # UNCOVERED_REQUIREMENTS below.
        styxx_primitives=(),
        notes=(
            "v0.1 has NO styxx-side coverage for bias-amplification or "
            "feedback-loop mitigation. Operators should consult separate "
            "tools (e.g., fairness libraries, drift monitors); "
            "styxx.cognometric_card provides per-step register signals "
            "that MAY surface drift indirectly but is not a substitute. "
            "See UNCOVERED_REQUIREMENTS for the boundary statement. "
            "Not legal advice. Independent conformity review required."
        ),
    ),
}


# Article 15 requirements + Annex III concerns styxx does NOT cover.
# This list MUST be at least as long as the covered list (kill-gate A3
# from the companion paper). Below it has 7 entries vs 4 covered.
UNCOVERED_REQUIREMENTS: tuple[UncoveredRequirement, ...] = (
    UncoveredRequirement(
        clause="Article 15.4 (bias amplification)",
        reason=(
            "styxx primitives operate on per-step agent cognition signals; "
            "they do not measure population-level disparate impact, "
            "demographic outcome equalization, or training-data bias "
            "amplification across protected classes."
        ),
        alternative=(
            "Fairness audit libraries (e.g., Fairlearn, AIF360) plus "
            "domain-specific impact assessment; legal review."
        ),
    ),
    UncoveredRequirement(
        clause="Article 15 (cybersecurity)",
        reason=(
            "styxx instruments observe agent cognition, not network, host, "
            "or supply-chain security. Prompt-injection robustness is "
            "ADJACENT (refusal-related instruments may flag some injection "
            "attempts) but not the primary scope."
        ),
        alternative=(
            "Dedicated LLM security tools (e.g., Lakera Guard, Prompt-Armor, "
            "OWASP LLM Top 10 mitigations); standard application-security "
            "audit; penetration testing."
        ),
    ),
    UncoveredRequirement(
        clause="Article 9 (risk management)",
        reason=(
            "styxx provides MEASUREMENT evidence; it does not implement "
            "the risk-management lifecycle (identification, estimation, "
            "evaluation, mitigation, residual-risk acceptance) prescribed "
            "by Article 9."
        ),
        alternative=(
            "ISO/IEC 23894:2023 or NIST AI RMF 1.0 frameworks; QMS-style "
            "organizational processes."
        ),
    ),
    UncoveredRequirement(
        clause="Article 10 (data governance)",
        reason=(
            "styxx does not score training-data quality, provenance, "
            "labeling consistency, or representativeness. Cognometric "
            "instruments operate post-training on agent outputs."
        ),
        alternative=(
            "Data documentation frameworks (Datasheets for Datasets, "
            "Data Statements); training-data audit tooling."
        ),
    ),
    UncoveredRequirement(
        clause="Article 12 (record-keeping)",
        reason=(
            "styxx writes per-step cognometric vitals but does not, by "
            "itself, satisfy Article 12's logging-traceability "
            "requirements end-to-end (event capture, retention, integrity)."
        ),
        alternative=(
            "Production observability platforms (OpenTelemetry, Langfuse, "
            "Arize, Datadog LLM Observability) for trace-level logging "
            "alongside styxx for cognition-level scoring."
        ),
    ),
    UncoveredRequirement(
        clause="Article 13 (transparency and information to deployers)",
        reason=(
            "styxx instruments produce evidence; they do not generate the "
            "deployer-facing instructions of use, capability statements, "
            "or output-explanation interfaces that Article 13 mandates."
        ),
        alternative=(
            "Model card frameworks; capability-statement templates; "
            "deployer documentation processes."
        ),
    ),
    UncoveredRequirement(
        clause="Article 14 (human oversight)",
        reason=(
            "styxx is a measurement layer; it does not implement the "
            "human-in-the-loop interfaces, stop-button affordances, or "
            "operator-control surfaces Article 14 requires."
        ),
        alternative=(
            "Agent-platform-level oversight UI; interrupt and rollback "
            "primitives in the agent runtime; operator training programs."
        ),
    ),
)


def cite(article: str) -> Optional[ComplianceMap]:
    """Return the ComplianceMap for a given regulatory clause citation.

    Args:
        article: regulatory citation, e.g., 'Article 15.1(a)' or 'Article 15.4'.
            Matching is by exact key lookup against ARTICLE_15_REGISTRY.

    Returns:
        A ComplianceMap, or None if the citation is not in the registry.
        (None is not "not applicable" — it means v0.1 does not address
        this clause. See UNCOVERED_REQUIREMENTS for honest boundary.)
    """
    return ARTICLE_15_REGISTRY.get(article)


def coverage_table() -> tuple[ComplianceMap, ...]:
    """Return every ComplianceMap in the v0.1 registry."""
    return tuple(ARTICLE_15_REGISTRY.values())


def uncovered_requirements() -> tuple[UncoveredRequirement, ...]:
    """Return every EU AI Act requirement styxx v0.1 does NOT cover.

    Per kill-gate A3 of the companion paper, this list MUST be at least as
    long as the covered list. v0.1 ships 7 uncovered vs 4 covered.
    """
    return UNCOVERED_REQUIREMENTS
