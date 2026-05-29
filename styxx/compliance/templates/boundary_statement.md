# BOUNDARY STATEMENT — What styxx does NOT cover under the EU AI Act

> **Template — paste verbatim or customize the alternative-tool pointers.** Not legal advice. Independent legal review required.
> **styxx version:** 7.7.13 candidate · companion paper: `papers/EU_AI_ACT_COMPLIANCE_2026.md`

---

## Why this statement exists

Kill-gate A3 of the styxx EU AI Act compliance bridge (Section 7 of `papers/EU_AI_ACT_COMPLIANCE_2026.md`) requires that the list of EU AI Act requirements styxx does **NOT** cover be **at least as long as the list of requirements styxx does cover**. The boundary statement is the load-bearing legal-defensibility frame for any conformity declaration using styxx measurement primitives.

The seven uncovered EU AI Act provisions below MUST appear at least as prominently as the coverage statement in the conformity declaration. They are not appendix material. They are equal first-class requirements an operator's conformity story must address through alternative tooling.

## v0.2 coverage summary

| status | count | clauses |
|---|---|---|
| Covered (with calibrated AUC + construct ceiling) | 4 | Article 15.1, 15.1(a), 15.3, 15.4 (Article 15.4 is empty by design — see below) |
| NOT covered (alternative tooling required) | 7 | Article 9, 10, 12, 13, 14, 15 cybersecurity, 15.4 bias |

The styxx coverage list is reproducible programmatically via `styxx.compliance.eu_ai_act.coverage_table()`. The boundary statement is reproducible via `styxx.compliance.eu_ai_act.uncovered_requirements()`. Both fields are mandatory-tested by `tests/test_compliance_eu_ai_act.py`.

## The seven uncovered EU AI Act provisions

### 1. Article 15.4 — bias and feedback-loop mitigation

**Reason styxx does not cover:** styxx primitives operate on per-step agent cognition signals; they do not measure population-level disparate impact, demographic outcome equalization, or training-data bias amplification across protected classes.

**Alternative tooling:** Fairness audit libraries (e.g., Fairlearn, AIF360) plus domain-specific impact assessment; legal review of disparate-impact outcomes.

**Operator territory:** the deployment's own demographic-outcome monitoring + remediation processes.

### 2. Article 15 — cybersecurity

**Reason styxx does not cover:** styxx instruments observe agent cognition, not network, host, or supply-chain security. Prompt-injection robustness is *adjacent* — refusal-related instruments may flag some injection attempts, and `detect_context_injection` provides item-level injection suspicion on factual self-claims — but cybersecurity is NOT the primary scope.

**Alternative tooling:** Dedicated LLM security tools (e.g., Lakera Guard, Prompt-Armor, OWASP LLM Top 10 mitigations); standard application-security audit; penetration testing.

**Operator territory:** the deployment's full cybersecurity posture, including supply-chain integrity for both the model artifact and the agent runtime.

### 3. Article 9 — risk management

**Reason styxx does not cover:** styxx provides MEASUREMENT evidence; it does not implement the risk-management lifecycle (identification, estimation, evaluation, mitigation, residual-risk acceptance) prescribed by Article 9.

**Alternative tooling:** ISO/IEC 23894:2023 or NIST AI RMF 1.0 frameworks. `styxx.compliance.nist_ai_rmf` provides a parallel measurement-methodology bridge for NIST AI RMF's Measure function ONLY, not for the full RMF Govern + Map + Manage + Measure lifecycle. QMS-style organisational processes.

**Operator territory:** the organisation's own risk-management process and documentation.

### 4. Article 10 — data governance

**Reason styxx does not cover:** styxx does not score training-data quality, provenance, labeling consistency, or representativeness. Cognometric instruments operate post-training on agent outputs.

**Alternative tooling:** Data documentation frameworks (Datasheets for Datasets, Data Statements); training-data audit tooling; provenance-tracking systems.

**Operator territory:** training-data lineage, labeling QA, and representativeness audits.

### 5. Article 12 — record-keeping

**Reason styxx does not cover:** styxx writes per-step cognometric vitals but does not, by itself, satisfy Article 12's logging-traceability requirements end-to-end (event capture, retention, integrity).

**Adjacent tooling NOTE:** styxx 7.7.11 added cryptographic attestation chains (`styxx.attest_chain`) which provide tamper-evident integrity over the cognometric vitals stream. This is an *adjacent* primitive for the integrity-preservation portion of Article 12 but does NOT satisfy the end-to-end logging mandate (event capture, retention policy, deployment-wide log infrastructure).

**Alternative tooling:** Production observability platforms (OpenTelemetry, Langfuse, Arize, Datadog LLM Observability) for trace-level logging alongside styxx for cognition-level scoring.

**Operator territory:** the deployment's logging architecture and retention policy.

### 6. Article 13 — transparency and information to deployers

**Reason styxx does not cover:** styxx instruments produce evidence; they do not generate the deployer-facing instructions of use, capability statements, or output-explanation interfaces that Article 13 mandates.

**Alternative tooling:** Model card frameworks; capability-statement templates; deployer documentation processes; XAI / SHAP / LIME for output explanation where required.

**Operator territory:** the instructions-of-use artifact itself (which may *cite* this declaration as one component but is broader in scope than this declaration).

### 7. Article 14 — human oversight

**Reason styxx does not cover:** styxx is a measurement layer; it does not implement the human-in-the-loop interfaces, stop-button affordances, or operator-control surfaces Article 14 requires.

**Alternative tooling:** Agent-platform-level oversight UI; interrupt and rollback primitives in the agent runtime; operator training programs.

**Operator territory:** the human-oversight UX and operator-training program for the deployed system.

## Cross-cutting observations

### What "alternative tooling required" means

For each uncovered requirement, the operator's conformity story must include evidence from the named alternative tool **or** an equivalent methodology. A blank entry is not acceptable; an explicit pointer is required by kill-gate A3. The pointers above are reference-level — they are NOT endorsements of specific products and are NOT exhaustive of available tooling.

### What this boundary does NOT preclude

styxx's measurement primitives may *interact* with uncovered domains. For example: `detect_context_injection` is calibrated under one cybersecurity-adjacent threat model (system_lie injection of factual claims). This does NOT make styxx a cybersecurity tool — the broader Article 15 cybersecurity requirements (network, host, supply-chain) remain uncovered. Likewise, `cognometric_card`'s per-step register signals may indirectly surface drift that interacts with Article 15.4 bias amplification — but this does NOT make styxx a bias-amplification tool. The boundary stays.

### Reassessment trigger

Per kill-gate A5, if no independent party cites the `papers/EU_AI_ACT_COMPLIANCE_2026.md` methodology by 2027-02-01, the methodology is reassessed and possibly deprecated. The boundary statement above may also need updating if (a) a new styxx primitive bridges into a currently-uncovered domain, in which case the count in the v0.2 coverage summary changes; or (b) a regulatory clarification narrows or expands an Article 15 sub-paragraph's scope.

---

**Not legal advice. Independent legal review required for any production declaration.**

**Reproducibility receipt:** the seven uncovered requirements above are reproducible verbatim from `styxx.compliance.eu_ai_act.UNCOVERED_REQUIREMENTS` at the deployed styxx version. `len(uncovered_requirements()) >= len(coverage_table())` is enforced by `tests/test_compliance_eu_ai_act.py::test_killgate_a3_uncovered_at_least_as_long_as_covered`.

**Methodology citation:** Rodabaugh, A. (Fathom Lab), *"A Pre-Registration-Disciplined Measurement Methodology for EU AI Act Article 15 Accuracy and Robustness Requirements on AI Agent Cognitive Observability"*, 2026-05-29 v0.2, CC-BY 4.0.
