# -*- coding: utf-8 -*-
"""NIST AI RMF 1.0 Measure-function compliance map for styxx primitives.

NIST AI 100-1 (the AI Risk Management Framework 1.0, published Jan 2023)
defines four core functions: Govern, Map, Measure, Manage. The Measure
function is the analytical engine — 22 subcategories across four
categories that evaluate AI system properties along seven trustworthy
characteristics (valid/reliable, safe, secure/resilient, accountable/
transparent, explainable/interpretable, privacy-enhanced, fair/bias-
mitigated).

This module ships v0.1 of styxx's mapping to the Measure 2.x subcategories
(MS-2.3, MS-2.4, MS-2.5, MS-2.6, MS-2.13), with explicit boundaries
enumerating Measure subcategories styxx does NOT cover (MS-2.7, MS-2.8,
MS-2.9 partial, MS-2.10, MS-2.11, MS-2.12).

USAGE (parallel to styxx.compliance.eu_ai_act):

    from styxx.compliance.nist_ai_rmf import cite, coverage_table, uncovered_requirements
    m = cite("MS-2.5")
    print(m.requirement_text)
    print(m.styxx_primitives)

DESIGN PRINCIPLES (same as the EU AI Act bridge):
  - Each map entry cites a SPECIFIC Measure subcategory (e.g., MS-2.5),
    not the entire Measure function.
  - Each entry lists construct ceilings (known failure modes) explicitly.
  - The uncovered_requirements() list MUST be at least as long as the
    covered list (kill-gate A3 from the companion paper). v0.1 ships
    6 uncovered vs 5 covered (6 >= 5).
  - Reproducibility receipts cite styxx commit hashes that produced the
    calibrated metrics. The receipts are checkable via git.

Source: https://airc.nist.gov/airmf-resources/airmf/5-sec-core/ (NIST AIRC).

NOT legal advice. NOT a substitute for an organization's own NIST AI RMF
implementation. NIST AI RMF is voluntary in the US federal context but
referenced by procurement, contracting, and state-level legislation.
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
    "MEASURE_REGISTRY",
    "UNCOVERED_MEASURE",
]


# ---------------------------------------------------------------------------
# Primitive coverage entries (shared definitions across NIST + EU AI Act)
# ---------------------------------------------------------------------------

# Note: the same styxx primitive can appear in multiple registry entries.
# Each PrimitiveCoverage instance is defined ONCE here and referenced by
# multiple MEASURE_REGISTRY entries — this keeps construct ceilings and
# reproducibility receipts in a single source of truth.

_COGNOMETRIC_CARD = PrimitiveCoverage(
    primitive="styxx.cognometric_card",
    calibrated_metric=(
        "HaluEval-QA AUC 0.998 (mean over 150 items, seeds 31/47/83); "
        "XSTest refusal AUC 0.976; BFCL v3 tool-drift AUC 0.943. "
        "Per-step cognometric readout: drift, confabulation, refusal, "
        "sycophancy, phase transition, low trust, incoherence."
    ),
    construct_ceiling=(
        "Text-only register space has construct ceilings: instruments measure "
        "register, not always content. Sycophancy false-positive 0.30 on "
        "restrained-tech responses (gpt-3.5: 0.60). Logprob-validity works "
        "for refusal but fails for hallucination (confident confabulation). "
        "Validity-cross-substrate confirmed (apologetic-refusal training "
        "transfers to GPT-4 out-of-family); cross-vendor generalization is "
        "partial. See feedback_register_pareto_frontier for boundary map."
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
        "Mechanism is 'out-of-context critique' (RLHF discrimination on "
        "labeled candidate text), NOT within-model generation-vs-critique "
        "asymmetry. The true within-model asymmetry rate is 5.88% on "
        "dark-core / 17.00% on TruthfulQA (v3 measurement, single-character "
        "T/F/U NLI). In-council bias caveat: the default backend gpt-4o-mini "
        "was IN the original 3-vendor council that generated expected_consensus."
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
        "as pre-disclosed). 12 registered checkers as of 7.7.10."
    ),
    construct_ceiling=(
        "First-occurrence-only by default — caught initial drift but missed "
        "systematic propagation to 4 additional places where the same "
        "off-by-one appeared. Richer multi-occurrence checker is methodology "
        "future work. Limited to substrate-checkable claims; does not score "
        "interpretive, causal, or generalization claims."
    ),
    receipt_commit="3c24b5e",
    receipt_doc="papers/agent-self-audit/FINDING_agent_claim_audit_2026_05_28.md",
)

_RECOVER_POSTURE = PrimitiveCoverage(
    primitive="styxx.recover_posture",
    calibrated_metric=(
        "Cognitive-integrity persistence primitive for agents crossing "
        "context-compaction boundaries. v1 reads what chart.jsonl persists; "
        "PostureSummary dataclass + MCP tool cogn_recover_posture. "
        "Enables fail-safe handoffs across context windows."
    ),
    construct_ceiling=(
        "Honest v1 scope: does not include cogn_audit scores. Falsifiable "
        "claim attached pending bet-2 outcome study. Not validated on "
        "third-party agent platforms. Not a substitute for runtime "
        "fail-safe controls (watchdogs, interrupts, rollback)."
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


# ---------------------------------------------------------------------------
# NIST AI RMF Measure subcategory registry
# ---------------------------------------------------------------------------

# Trustworthy AI characteristic each Measure subcategory primarily addresses
# (per NIST AI 100-1 Section 5.3):
#   MS-2.3 → "valid and reliable" (performance demonstration)
#   MS-2.4 → "valid and reliable" (production monitoring)
#   MS-2.5 → "valid and reliable"
#   MS-2.6 → "safe"
#   MS-2.7 → "secure and resilient"
#   MS-2.8 → "accountable and transparent"
#   MS-2.9 → "explainable and interpretable"
#   MS-2.10 → "privacy-enhanced"
#   MS-2.11 → "fair with harmful bias managed"
#   MS-2.12 → (environmental impact — broader sustainability)
#   MS-2.13 → "TEVV process effectiveness" (recursive evaluation)

MEASURE_REGISTRY: dict[str, ComplianceMap] = {
    "MS-2.3": ComplianceMap(
        clause="MS-2.3",
        requirement_text=(
            "AI system performance or assurance criteria are measured "
            "qualitatively or quantitatively and demonstrated for conditions "
            "similar to deployment setting(s). Measures are documented."
        ),
        styxx_primitives=(_COGNOMETRIC_CARD, _CRITIQUE_DETECTOR, _GAUNTLET_PREFLIGHT),
    ),
    "MS-2.4": ComplianceMap(
        clause="MS-2.4",
        requirement_text=(
            "The functionality and behavior of the AI system and its "
            "components — as identified in the map function — are monitored "
            "when in production."
        ),
        styxx_primitives=(_COGNOMETRIC_CARD, _AGENT_AUDIT),
    ),
    "MS-2.5": ComplianceMap(
        clause="MS-2.5",
        requirement_text=(
            "The AI system to be deployed is demonstrated to be valid and "
            "reliable. Limitations of the generalizability beyond the "
            "conditions under which the technology was developed are "
            "documented."
        ),
        styxx_primitives=(_COGNOMETRIC_CARD, _CRITIQUE_DETECTOR, _GAUNTLET_PREFLIGHT),
        notes=(
            "MS-2.5 explicitly requires documentation of generalizability "
            "limitations. styxx's construct-ceiling discipline (every "
            "primitive ships with a documented failure mode) directly "
            "addresses this. The 'Limitations of generalizability' phrase "
            "in MS-2.5 is the exact regulatory hook for construct ceilings. "
            "Not legal advice. Independent conformity review required."
        ),
    ),
    "MS-2.6": ComplianceMap(
        clause="MS-2.6",
        requirement_text=(
            "The AI system is evaluated regularly for safety risks. The AI "
            "system to be deployed is demonstrated to be safe, its residual "
            "negative risk does not exceed the risk tolerance, and it can "
            "fail safely, particularly if made to operate beyond its "
            "knowledge limits. Safety metrics reflect system reliability "
            "and robustness, real-time monitoring, and response times for "
            "AI system failures."
        ),
        styxx_primitives=(_COGNOMETRIC_CARD, _RECOVER_POSTURE, _AGENT_AUDIT),
        notes=(
            "Partial coverage: styxx provides per-step cognitive monitoring "
            "(cognometric_card), context-compaction-boundary recovery "
            "(recover_posture), and substrate-grounded claim verification "
            "(agent_audit). Operators must add runtime fail-safe controls "
            "(watchdogs, interrupts, rollback) to satisfy 'can fail safely' "
            "end-to-end. Not legal advice. Independent conformity review required."
        ),
    ),
    "MS-2.13": ComplianceMap(
        clause="MS-2.13",
        requirement_text=(
            "Effectiveness of the employed TEVV (Test, Evaluation, "
            "Verification, Validation) metrics and processes in the measure "
            "function are evaluated and documented."
        ),
        styxx_primitives=(_AGENT_AUDIT,),
        notes=(
            "MS-2.13 is the recursive subcategory: it asks whether the "
            "measurement methodology itself is effective. styxx's "
            "recursive-discipline paper (PAPER_recursive_discipline_2026_05_27.md) "
            "documents six layers of self-falsification of the project's own "
            "measurement methodology, with pre-registered kill-gates and "
            "public commit-level receipts. styxx.agent_audit is the "
            "specific instrument that operationalizes this for substrate-"
            "checkable claims. Not legal advice. Independent review required."
        ),
    ),
}


# Measure subcategories explicitly NOT covered by v0.1.
# Kill-gate A3: this list MUST be at least as long as the covered list.
# v0.1 ships 6 uncovered vs 5 covered (6 >= 5).
UNCOVERED_MEASURE: tuple[UncoveredRequirement, ...] = (
    UncoveredRequirement(
        clause="MS-2.7 (secure and resilient)",
        reason=(
            "styxx instruments observe agent cognition, not system "
            "security or resilience under adversarial conditions. Prompt-"
            "injection robustness is ADJACENT (refusal-related instruments "
            "may flag some attempts) but security is not the primary scope."
        ),
        alternative=(
            "Dedicated LLM security tools (Lakera Guard, Prompt-Armor, "
            "OWASP LLM Top 10 mitigations); standard application-security "
            "audit; penetration testing; red-teaming."
        ),
    ),
    UncoveredRequirement(
        clause="MS-2.8 (accountable and transparent)",
        reason=(
            "styxx produces measurement evidence; it does not generate "
            "deployer-facing accountability documentation, decision-rationale "
            "interfaces, or governance artifacts that MS-2.8 implies."
        ),
        alternative=(
            "Model card frameworks; capability-statement templates; "
            "deployer documentation processes; AI ethics review boards."
        ),
    ),
    UncoveredRequirement(
        clause="MS-2.9 (explainable and interpretable)",
        reason=(
            "PARTIAL gap: styxx's construct-ceiling discipline contributes "
            "to interpretability by documenting WHEN each primitive fails, "
            "but does not generate per-decision feature attributions, "
            "counterfactual explanations, or per-input rationales that "
            "modern XAI methods (SHAP, LIME, attention-based) provide."
        ),
        alternative=(
            "Dedicated XAI libraries (SHAP, LIME, Captum, Inseq); "
            "mechanistic interpretability research methods; user studies."
        ),
    ),
    UncoveredRequirement(
        clause="MS-2.10 (privacy-enhanced)",
        reason=(
            "styxx does not measure differential privacy guarantees, "
            "membership inference resistance, training-data memorization, "
            "or PII leakage detection. Cognometric instruments operate "
            "on outputs but not on privacy properties of those outputs."
        ),
        alternative=(
            "Differential privacy auditing tools (Google's DP libraries); "
            "PII leakage scanners (Presidio, scrubadub); membership "
            "inference test suites."
        ),
    ),
    UncoveredRequirement(
        clause="MS-2.11 (fair with harmful bias managed)",
        reason=(
            "styxx primitives operate on per-step agent cognition signals; "
            "they do not measure population-level disparate impact, "
            "demographic outcome equalization, or training-data bias "
            "amplification across protected classes."
        ),
        alternative=(
            "Fairness audit libraries (Fairlearn, AIF360); domain-specific "
            "impact assessment; demographic parity / equalized odds metrics."
        ),
    ),
    UncoveredRequirement(
        clause="MS-2.12 (environmental impact and sustainability)",
        reason=(
            "styxx instruments operate at inference time on agent outputs; "
            "they do not measure training-energy consumption, inference-"
            "carbon footprint, or supply-chain environmental impact."
        ),
        alternative=(
            "ML CO2 calculators (CodeCarbon, MLCO2); cloud-provider "
            "sustainability reports; supply-chain LCA tooling."
        ),
    ),
)


def cite(subcategory: str) -> Optional[ComplianceMap]:
    """Return the ComplianceMap for a NIST AI RMF Measure subcategory citation.

    Args:
        subcategory: subcategory citation, e.g., 'MS-2.5' or 'MS-2.13'.
            Matching is by exact key lookup against MEASURE_REGISTRY.

    Returns:
        A ComplianceMap, or None if the subcategory is not in the v0.1
        registry. (None is not 'not applicable' — it means v0.1 does not
        address this subcategory. See UNCOVERED_MEASURE for honest boundary.)
    """
    return MEASURE_REGISTRY.get(subcategory)


def coverage_table() -> tuple[ComplianceMap, ...]:
    """Return every ComplianceMap in the v0.1 Measure subcategory registry."""
    return tuple(MEASURE_REGISTRY.values())


def uncovered_requirements() -> tuple[UncoveredRequirement, ...]:
    """Return every Measure subcategory v0.1 does NOT cover.

    Per kill-gate A3 of the companion paper, this list MUST be at least as
    long as the covered list. v0.1 ships 6 uncovered vs 5 covered.
    """
    return UNCOVERED_MEASURE
