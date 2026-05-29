# `styxx.compliance.templates` — conformity declaration templates for EU AI Act Article 15

**Status:** v0.1 (initial set) — 2026-05-29 — styxx 7.7.13 candidate

These are **paste-and-customize markdown templates** for the accuracy / robustness / boundary sections of an EU AI Act Article 15 conformity declaration. Each template is pre-filled with styxx output examples and an operator-substitutable `<...>` placeholder surface.

**Not legal advice.** Independent legal review is required for any production declaration. The templates encode the *measurement-methodology shape* the recursive-discipline thesis argues should be present in any defensible declaration; they do not encode the legal correctness, jurisdictional sufficiency, or notified-body acceptability of any particular declaration's substance.

## Template index

| template | clause | purpose |
|---|---|---|
| `accuracy_declaration.md` | Article 15.1(a) | "levels of accuracy and the relevant accuracy metrics declared in the accompanying instructions of use" — paste-able section with AUC numbers, construct-ceiling disclosures, and reproducibility receipts |
| `robustness_statement.md` | Article 15.3 | "technical redundancy solutions such as backup or fail-safe plans" — paste-able section covering the cognometric-vitals / stateless-resample / cross-context-divergence fail-safe + redundancy stack |
| `boundary_statement.md` | (governance) | the "what styxx does NOT cover" enumeration: seven EU AI Act requirements outside scope (Article 9, 10, 12, 13, 14, 15 cybersecurity, 15.4 bias) with alternative-tool pointers — must appear at least as prominently as the coverage statement (kill-gate A3) |
| `sycophancy_disclosure.md` | Article 15.1(a) construct ceiling | the restrained-tech FPR 0.30 published failure mode (gpt-3.5: 0.60), framed for inclusion in instructions of use as required honest-disclosure under the recursive-discipline thesis |
| `injection_resistance_disclosure.md` | Article 15.3 + SECURITY MODEL | the load-bearing operational requirement: the scoring harness MUST sample statelessly for `grounded_honesty`. Quantified: AUC 0.944 under attack vs 0.011 in-session inverted. Paste-able with the architectural-defense statement |

## How to use

1. **Read the companion paper** (`papers/EU_AI_ACT_COMPLIANCE_2026.md`) first to understand what each template covers and what it does NOT cover. The paper's Section 4 boundary statement is the load-bearing legal-defensibility frame; the templates cite it.
2. **Reproduce the calibration numbers** against the deployed styxx version. Numbers in these templates are accurate to styxx 7.7.13 candidate; future versions may calibrate differently. Run `pytest tests/test_compliance_eu_ai_act.py` to verify the registry is intact, and run the FINDING reproduction scripts at the cited commit hashes (`papers/grounded-honesty-axis/FINDING_*.md`) to re-derive the per-primitive AUC numbers.
3. **Customize the `<...>` placeholders** with deployment-specific values (system name, version, intended Annex III use, organisation, link to legal review).
4. **Run the template through your legal-review process.** The template is the *measurement-methodology shape*; the legal substance is your organisation's, with notified-body review where Article 43 requires it.
5. **Preserve the construct-ceiling disclosures.** The published failure modes (sycophancy FPR 0.30; logprob-validity refusal-specific; grounded_honesty belief-not-truth; detect_context_injection single-attack calibration; SECURITY MODEL stateless requirement) are part of the methodology's *honesty contract* — removing them produces an unhonest declaration the methodology cannot back.

## What changes between styxx versions

The templates ship at the styxx version they were committed at (currently 7.7.13 candidate). When styxx is bumped:

- **Patch bumps** (7.7.13 → 7.7.14): primitive calibrations may be re-run; AUC numbers may shift by ≤ 0.01. Re-derive against the deployed version.
- **Minor bumps** (7.7 → 7.8): new primitives may add coverage rows; the boundary statement may shrink (a previously uncovered requirement may gain coverage). Diff the templates against the new release notes.
- **Major bumps**: full re-read recommended.

`pytest tests/test_compliance_eu_ai_act.py -v` is the smoke check the registry didn't drift in a way that breaks A1–A3 kill-gates.

## Templates DO NOT cover

- **Article 9 (risk management):** consult ISO/IEC 23894:2023 or NIST AI RMF 1.0; `styxx.compliance.nist_ai_rmf` provides a parallel measurement-methodology bridge for the Measure function only.
- **Article 10 (data governance):** Datasheets for Datasets, Data Statements, training-data audit tooling.
- **Article 12 (record-keeping):** OpenTelemetry, Langfuse, Arize, Datadog LLM Observability. styxx 7.7.11 attestation chains provide *integrity* over the cognometric stream but not end-to-end logging.
- **Article 13 (transparency to deployers):** model card frameworks; capability-statement templates.
- **Article 14 (human oversight):** agent-platform-level oversight UI; interrupt and rollback primitives.
- **Article 15.4 (bias amplification):** Fairlearn, AIF360, demographic outcome equalization tooling.
- **Article 15 cybersecurity:** Lakera Guard, Prompt-Armor, OWASP LLM Top 10; application security audit; pen testing.

Conformity declarations addressing the above must cite the alternative tooling, not these templates.
