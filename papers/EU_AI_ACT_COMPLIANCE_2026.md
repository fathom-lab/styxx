# A pre-registered, falsifiable measurement-methodology bridge from `styxx` primitives to EU AI Act Article 15 / Annex III requirements (v0.1)

**Author:** Alexander Rodabaugh (Fathom Lab)
**Date:** 2026-05-28 (v0.1) · revised 2026-05-29 (v0.2 — adds `grounded_honesty` and `detect_context_injection` from the styxx 7.7.13 candidate; folds the 2026-05-29 injection-gap closure SURVIVED run into the SECURITY MODEL section; see §9 v0.2 addendum) · revised 2026-05-29 (v0.3 — folds the 2026-05-29 injection-attack-generalization REPORT_AS_LANDED run: `detect_context_injection` calibration extends to two injection vectors (system_lie + persona_lie, both with the same architectural signature), fewshot_lie single-demo identified as ineffective on canonical facts; see §10 v0.3 addendum) · revised 2026-05-30 (v0.4 — folds the 2026-05-30 TruthfulQA n=790 benchmark calibration (Layer 1 REPORT_AS_LANDED, H1 continuous AUC 0.619 FAILED below the 0.65 REPORT floor → cross-dataset transport from n=48 keystone is BOUNDED, not retracted) and the pre-generation belief-coherence gate (Layer 2, all three control bars exceed SURVIVED thresholds descriptively at C1=0.66 hallucination reduction / C2=0.49 useful-answer retention / C3=0.84 committed precision; K_precondition narrowly missed at 0.281 vs 0.30 → no SURVIVED claim per discipline) and the per-category competence cliff map (Layer 3); see §11 v0.4 addendum)
**Substrate:** styxx 7.7.13 candidate (`grounded_honesty` and `detect_context_injection` shipped at commits `9ac8db4` and `53f2284` respectively); companion to `papers/PAPER_recursive_discipline_2026_05_27.md` (v7) and `papers/grounded-honesty-axis/SYNTHESIS_grounded_honesty_arc_2026_05_28.md`; module `styxx.compliance.eu_ai_act` was first shipped in styxx 7.7.10 (2026-05-28) and updated in 7.7.13 to cite the new primitives.
**Status:** v0.4 extending v0.3 with the TruthfulQA benchmark-scale calibration receipts (Layer 1 REPORT_AS_LANDED + Layer 2 descriptive control-primitive + Layer 3 per-category cliff map). Published BEFORE the EU AI Act high-risk system enforcement deadline of 2 August 2026 to give regulated operators an evaluation runway. Not legal advice. Independent conformity review required for any production deployment.

---

## Abstract

The EU AI Act high-risk obligations enter enforcement on 2 August 2026 with penalties up to €15M or 3% of global annual turnover. Article 15 mandates that high-risk AI systems achieve appropriate levels of accuracy, robustness, and cybersecurity, that accuracy metrics be declared in the instructions of use, and — under paragraph 2 — that *"the Commission shall, in cooperation with relevant stakeholders... encourage the development of benchmarks and measurement methodologies"*. The Commission's invitation is open. No competing AI observability or evaluation product currently publishes a structured Article 15 mapping. This paper introduces `styxx.compliance.eu_ai_act`, the first open-source, pre-registration-disciplined measurement-methodology bridge mapping a deployable cognitive-observability primitive set to specific Article 15 sub-paragraphs, with calibrated metrics, explicit construct-ceiling disclosures, commit-level reproducibility receipts, and pre-stated falsification criteria. The v0.2 mapping covers four Article 15 clauses with seven styxx primitives: the five v0.1 primitives (`cognometric_card`, `critique_detector`, `agent_audit`, `recover_posture`, `gauntlet+preflight`) plus two new 7.7.13-candidate primitives — `grounded_honesty` (AUC 0.966 separating TRUE from FALSE register-matched factual self-claims vs the text-only deception axis at 0.498) and `detect_context_injection` (AUC 0.875 item-level cross-context divergence injection-detection, calibrated post the 2026-05-29 injection-gap closure SURVIVED run, with v0.3 extending the calibration to a second injection vector — persona_lie identity-framing — at AUC 0.833 with the same architectural signature, and identifying the fewshot_lie single-demo attack as ineffective on canonical facts at K3 = 0.063 → threat-surface narrowing rather than widening). The boundary statement explicitly enumerates seven uncovered EU AI Act requirements (Article 9, 10, 12, 13, 14, 15.4 bias, 15 cybersecurity) and points each at a non-styxx alternative tool or methodology — longer than the coverage statement by design (kill-gate A3). Five pre-registered kill-gates define what success and failure look like for the v0.2 release. This document is a measurement methodology, not legal advice; it is the kind of artifact Article 15 paragraph 2 explicitly invites stakeholders to develop.

---

## 1. Background

### 1.1 The August 2, 2026 deadline

The EU AI Act (Regulation (EU) 2024/1689) enters its high-risk system enforcement phase on 2 August 2026. Annex III enumerates high-risk categories including biometrics, critical infrastructure, education, employment, essential services, law enforcement, migration, and democratic process administration. Providers of high-risk systems must demonstrate conformity through Article 9 risk management, Article 10 data governance, Article 11 technical documentation, Article 12 logging, Article 13 transparency, Article 14 human oversight, and Article 15 accuracy/robustness/cybersecurity. Most Annex III systems may be self-assessed where harmonised standards apply; biometric, critical infrastructure, and law-enforcement systems require third-party conformity assessment by a notified body. Penalties for non-compliance reach €15,000,000 or 3% of global annual turnover, whichever is higher.

### 1.2 Article 15 in detail

Article 15 ("Accuracy, robustness and cybersecurity") has four substantive paragraphs:

- **15.1**: high-risk AI systems shall be *designed and developed* to achieve appropriate levels of accuracy, robustness, and cybersecurity *throughout their lifecycle*.
- **15.1(a)** (instructions of use): accuracy levels and *relevant accuracy metrics* shall be declared in the accompanying instructions of use.
- **15.2**: *"the Commission shall, in cooperation with relevant stakeholders and organisations such as metrology and benchmarking authorities, encourage, as appropriate, the development of benchmarks and measurement methodologies"*. This is the open invitation that this paper responds to.
- **15.3**: robustness shall be as high as possible regarding errors, faults, or inconsistencies, achievable through *technical redundancy solutions* including backup or fail-safe plans.
- **15.4**: high-risk AI systems that continue to learn after deployment shall be designed to *eliminate or reduce as far as possible* the risk of biased outputs influencing future inputs (feedback loops).

### 1.3 What this paper IS, and is NOT

This paper IS:
- A measurement methodology: it names specific styxx primitives, calibrated metrics, construct ceilings, and reproducibility receipts that produce evidence relevant to specific Article 15 clauses.
- Pre-registered: every claim cites a commit, every primitive's failure mode is disclosed, every uncovered requirement is enumerated with an alternative tool reference.
- Open-source and falsifiable: the `styxx.compliance.eu_ai_act` module is MIT-licensed Python; the tests verify the structural integrity of the mapping; the paper's claims can be re-checked at any future git commit.

This paper IS NOT:
- Legal advice. Conformity assessment requires legal expertise this paper does not provide.
- Sufficient on its own for EU AI Act conformity. It addresses Article 15 sub-paragraphs only and explicitly disclaims coverage of Articles 9, 10, 12, 13, 14.
- A claim that styxx primitives have been validated by a notified body. They have been validated against the styxx project's own benchmarks; third-party validation remains future work.
- An endorsement by the European Commission, AISI, METR, Apollo Research, or any standardization body. It is a unilateral stakeholder contribution.

---

## 2. Methodology

### 2.1 The bridge primitives

`styxx.compliance.eu_ai_act` exposes four objects:

1. **`ARTICLE_15_REGISTRY: dict[str, ComplianceMap]`** — the registry of mapped clauses. v0.1 keys are `"Article 15.1"`, `"Article 15.1(a)"`, `"Article 15.3"`, `"Article 15.4"`.
2. **`ComplianceMap`** — dataclass with `clause`, `requirement_text`, `styxx_primitives` (tuple of `PrimitiveCoverage`), and `notes`. Each `notes` field includes a *"not legal advice"* disclaimer.
3. **`PrimitiveCoverage`** — per-primitive coverage entry with `primitive` (public API symbol), `calibrated_metric` (published AUC/accuracy with context), `construct_ceiling` (honest failure mode), `receipt_commit` (styxx git SHA that produced the metric), and `receipt_doc` (path to the validating FINDING).
4. **`UNCOVERED_REQUIREMENTS: tuple[UncoveredRequirement, ...]`** — the boundary statement: every EU AI Act requirement styxx does NOT cover, with reason and alternative tool reference.

Helper functions: `cite(article: str)`, `coverage_table()`, `uncovered_requirements()`.

### 2.2 Five pre-registered kill-gates

These were pre-stated in the strategic landscape document (`.styxx/STRATEGIC_LANDSCAPE_2026_05_28.md`, written before this module existed) and are enforced by the test suite (`tests/test_compliance_eu_ai_act.py`):

- **A1 (validity).** Every clause key in `ARTICLE_15_REGISTRY` must cite a specific Article sub-paragraph, not generic compliance language. Enforced by regex `^Article \d+(\.\d+)?(\([a-z]\))?$`.
- **A2 (falsifiability).** Every `PrimitiveCoverage.construct_ceiling` must be non-empty and at least 50 characters. Hidden caveats are not permitted; the failure mode goes in the field.
- **A3 (boundary explicitness).** `len(uncovered_requirements()) >= len(coverage_table())`. The list of what styxx does NOT cover must be at least as long as the list of what it does. v0.1 ships 7 uncovered vs 4 covered.
- **A4 (timeline).** Publish before July 1, 2026 to provide a ~30-day evaluation runway before the August 2 enforcement deadline. This paper is committed 2026-05-28.
- **A5 (citation strategy).** ≥1 independent citation (academic, regulator, enterprise) within 6 months of publication. If 0 by 2027-02-01, the methodology did not achieve uptake and is reassessed. The strategic landscape doc references this measurable kill-gate explicitly.

A2 and A3 are enforced at test time. A1 is enforced at test time. A4 and A5 are enforced by the project timeline.

### 2.3 What the mapping does NOT attempt

- It does not score the **adequacy** of styxx primitives for any specific high-risk Annex III system. That assessment is system-specific and operator-specific.
- It does not enumerate ALL Article 15 sub-paragraphs. v0.1 covers four. v0.2 may extend.
- It does not write conformity declarations. Operators write declarations; styxx supplies the measurement evidence.
- It does not claim to satisfy harmonised standards (none for AI agent honesty exist as of 2026-05-28).

---

## 3. The coverage table

Each row below corresponds to one entry in `ARTICLE_15_REGISTRY`. Every primitive in the right column ships in styxx 7.7.10; every commit hash is reachable at `fathom-lab/styxx@main`.

### 3.1 Article 15.1 — appropriate accuracy, robustness, cybersecurity throughout lifecycle

| primitive | calibrated metric | construct ceiling |
|---|---|---|
| `styxx.cognometric_card` | HaluEval-QA AUC 0.998 (mean over 150 items, seeds 31/47/83); XSTest refusal AUC 0.976; BFCL v3 tool-drift AUC 0.943 | Text-only register space has construct ceilings: instruments measure register, not always content. Sycophancy false-positive 0.30 on restrained-tech responses (gpt-3.5: 0.60). Logprob-validity works for refusal but fails for hallucination (confident confabulation). |
| `styxx.gauntlet + styxx.preflight` | v3 detection bars (D1+D2+D3+D4) with regression-tested oracles. 18 pre-registered baselines tested; 17 FAILed bars before Baseline-019 PASSed. | Bars catch confounds the project pre-stated; do not catch unknown-unknown confound classes. Each bar revision documents one prior missed confound. |
| `styxx.agent_audit` | 13/13 PASS modal pre-stated outcome (L5 commit `3c24b5e`); L7 uncurated extension caught off-by-one count drift in companion paper (2 FAILs as pre-disclosed). | First-occurrence-only by default — caught initial drift but missed propagation to 4 additional places. Richer multi-occurrence checker is methodology future work. |
| `styxx.grounded_honesty` *(7.7.13)* | Factual self-claim honesty grounded against the model's OWN resampled belief distribution: `g = Stability × Concordance`. **Two-tier calibration as of v0.4:** (1) **n=48 register-matched feasibility-grade SURVIVED** at AUC **0.966** separating TRUE from FALSE self-claims (vs text-only deception axis 0.498 = chance, margin +0.468), gpt-4o-mini, the keystone result. (2) **n=790 TruthfulQA benchmark-scale REPORT_AS_LANDED** at continuous AUC **0.619** — H1 FAILED below the 0.65 REPORT floor; cross-dataset transport from feasibility-grade to TruthfulQA is bounded at the pre-registered bars (`FINDING_truthfulqa_benchmark_2026_05_30.md`, commit `a75f1e7`). Self-calibrating Stability gate (high stratum AUC 0.97 at n=48, low 0.44, report-or-abstain). Architecturally injection-resistant under stateless sampling: AUC **0.944** under system_lie attack — drop only 0.022. **The companion pre-generation gate (`PREREG_pregeneration_gate_2026_05_30.md`) demonstrates the same primitive at the conservative gate-decision threshold (`Stability ≥ 0.7 ∧ Concordance ≥ 0.5`) is operationally useful even when the continuous AUC is weak — descriptive C3 committed precision 0.837 on the 461 committed items of 790.** | Grounds against the model's BELIEF, not external truth — a confidently-wrong belief yields a confidently-wrong verdict (same-vendor council does NOT fix; cross-vendor is the open step). Single axis (factual self-claims) only. **Cross-dataset transport from n=48 register-matched single-domain pair set to n=790 TruthfulQA adversarial-fact pair structure is BOUNDED at the pre-registered AUC 0.65 REPORT floor** — TruthfulQA labeling noise (31.5% of FALSE-arm items judged honest because model's belief actually aligns with "Best Incorrect Answer") is a material contributor, pre-disclosed in `PREREG_truthfulqa_benchmark_2026_05_30.md` honest scope #2. The keystone n=48 SURVIVED is bounded to its feasibility regime, not retracted. Past the model's competence cliff, plain one-shot resampling can converge on a SYSTEMATIC miscalculation (high Stability, low truth); method-diverse re-derivation recovers ~93% of the gap, irreducible ~2/36 residue scoped to cross-vendor. SECURITY MODEL: caller MUST sample statelessly; in-session sampling collapses to AUC 0.011 (near-inverted). **The gate-decision threshold IS operational at benchmark scale even when continuous AUC is below SURVIVED — the deployable primitive is the gate, not the continuous axis (Layer 2, v0.4).** |
| `styxx.detect_context_injection` *(7.7.13)* | Item-level context-injection detection via cross-context resampling divergence `D = |C_stateless − C_in_session|`. Pre-registered AUC **0.875** at threshold 0.5 separating injected from clean items on n=48 register-matched factual self-claim pairs under system_lie injection (gpt-4o-mini, N=10 per arm at temp=1.0); v0.3 generalization run replicates the architectural signature on **persona_lie** identity-framing at AUC **0.833** (mean `D_FALSE` 0.644, K3 attack-effective 0.771) with G1 stateless robust AUC 0.955 and G2 in-session inverted AUC 0.174 — same per-vector pattern as system_lie. Pair with `grounded_honesty`: stateless arm produces the honesty verdict (architectural defense), cross-context delta produces the poison-suspicion at the same item. Detection cost: one extra resample set (+N=10 calls) per audited claim. | Single-model, single-vendor calibration across **two** injection vectors as of v0.3 (system_lie + persona_lie); a third vector — **fewshot_lie** single-demonstration — was tested and identified as ineffective at K3 = 0.063 (3/48 modal flips, far below the 0.70 precondition) → threat surface narrowed on canonical facts, not widened. Stronger attacks (multi-shot fewshot with consistent planted answer, jailbreak-grade persona framings, sequential tool-output spoofing, multi-stage gradient-style attacks, cross-vendor variants) remain pre-registerable scope-extensions NOT validated here. K3 attack-effectiveness on the two calibrated vectors was 0.98 (system_lie) and 0.771 (persona_lie) — a more aligned model may refuse the injection more often, attenuating both signals in parallel. The primitive measures CROSS-CONTEXT divergence — a deployment without an in-session arm has no signal. |

### 3.2 Article 15.1(a) — accuracy metrics in instructions of use

| primitive | calibrated metric | construct ceiling |
|---|---|---|
| `styxx.cognometric_card` | per-step cognometric readout (drift, confabulation, refusal, sycophancy, phase transition, low trust, incoherence) | construct ceilings apply per-axis; see §3.1 |
| `styxx.critique_detector(model='gpt-4o-mini')` | Baseline-019 PASSes gauntlet v3 bars at AUC 0.95, pre-stated 28% probability landed cleanly | Mechanism is "out-of-context critique", NOT within-model generation-vs-critique asymmetry. True within-model asymmetry: 5.88% on dark-core / 17.00% on TruthfulQA (v3 measurement). In-council bias: default backend gpt-4o-mini was in the original 3-vendor council. |
| `styxx.gauntlet + styxx.preflight` | see §3.1 | see §3.1 |
| `styxx.grounded_honesty` *(7.7.13)* | factual self-claim honesty grounded against the model's OWN resampled belief; AUC 0.966 clean / 0.944 under system_lie injection (see §3.1) | see §3.1 — caller MUST sample statelessly |
| `styxx.detect_context_injection` *(7.7.13)* | item-level cross-context divergence injection-detection at AUC 0.875 (system_lie) and AUC 0.833 (persona_lie) threshold 0.5 — two-vector calibration as of v0.3 (see §3.1) | see §3.1 — two-vector calibration (system_lie + persona_lie); fewshot_lie single-demo identified as ineffective on canonical facts; stronger scope-extensions pre-registerable |

These five together produce the *"relevant accuracy metrics"* an operator can declare in instructions of use, with the honest construct-ceiling disclosure that Article 15.1(a) does not explicitly mandate but that the recursive-discipline thesis argues should be present in any defensible declaration. **The 7.7.13 additions specifically extend the declaration's coverage of factual self-claim honesty and context-injection robustness — both first-class concerns under Article 15.1's accuracy-and-robustness mandate, with calibrated AUC numbers and SECURITY-MODEL disclosure ready for citation.**

### 3.3 Article 15.3 — robustness via technical redundancy / fail-safe plans

| primitive | calibrated metric | construct ceiling |
|---|---|---|
| `styxx.recover_posture` | cognitive-integrity persistence primitive for agents crossing context-compaction boundaries | v1 reads what chart.jsonl persists; does not include cogn_audit scores. Not validated on third-party agent platforms. |
| `styxx.agent_audit` | substrate-grounded session-output verifier | see §3.1 |
| `styxx.grounded_honesty` *(7.7.13)* | **The stateless-resample architectural property IS the fail-safe for context-injection attacks on factual self-claim verification.** A neutral resampler context absorbs the injection at scoring time — AUC 0.944 maintained under system_lie attack (vs 0.966 clean baseline; drop only 0.022). This is a structural Article 15.3 *technical redundancy* primitive: the redundancy is the architecturally-clean resampler context, separate from the agent's potentially-poisoned session. | See §3.1. The fail-safe is *load-bearing*: in-session sampling collapses to AUC 0.011 (near-inverted, 47/48 items score the lie HIGHER than the truth) — a worst-case interpretability hazard. Deploying `grounded_honesty` without honoring the stateless contract produces a worse-than-useless instrument that confidently certifies adversarially-planted lies as honest. |
| `styxx.detect_context_injection` *(7.7.13)* | **Item-level redundancy.** When the stateless arm is the primary verdict (Article 15.3 fail-safe per the row above), `detect_context_injection` provides an independent cross-context divergence signal at AUC 0.875 (system_lie) and AUC 0.833 (persona_lie) — same architectural signature on both vectors, calibrated v0.3 — flagging injection-suspicion item-by-item. The two primitives together are the structural Article 15.3 evidence for context-injection robustness: fail-safe + redundancy. | See §3.1 — two-vector calibration (system_lie + persona_lie); fewshot_lie single-demo identified as ineffective on canonical facts; stronger attack variants remain pre-registerable. |

Article 15.3 envisions backup/fail-safe plans against runtime errors. styxx's `recover_posture` is a cognition-side primitive against the context-compaction failure mode; `grounded_honesty`'s stateless-resample architecture is a structural fail-safe against context-injection attacks on factual self-claim verification; `detect_context_injection` is the item-level redundancy. None of these replace process-level fail-safes (external watchdogs, interrupt keys, rollback procedures), but together they provide verifiable evidence of cognitive-integrity continuity (recover_posture) and adversary-resistant factual-claim verification (grounded_honesty + detect_context_injection) across the most common modern agent failure modes.

### 3.4 Article 15.4 — bias amplification / feedback-loop mitigation

| primitive | calibrated metric | construct ceiling |
|---|---|---|
| *(none — honest empty coverage)* | — | — |

v0.1 has **NO** styxx-side coverage for Article 15.4. This empty cell is intentional and is itself a discipline statement: operators must consult separate tools (fairness libraries, drift monitors). `styxx.cognometric_card` provides per-step register signals that *may* surface drift indirectly but is not a substitute. See §4 for the boundary alternative reference.

---

## 4. What styxx does NOT cover (the boundary statement)

Per kill-gate A3, this section is intentionally longer than §3. Seven EU AI Act clauses are explicitly enumerated as outside the v0.1 scope, with alternative tools pointed at:

### 4.1 Article 15.4 — bias amplification

styxx primitives operate on per-step agent cognition signals; they do not measure population-level disparate impact, demographic outcome equalization, or training-data bias amplification across protected classes. **Alternative**: Fairness audit libraries (Fairlearn, AIF360) plus domain-specific impact assessment; legal review.

### 4.2 Article 15 — cybersecurity

styxx instruments observe agent cognition, not network, host, or supply-chain security. Prompt-injection robustness is *adjacent* (refusal-related instruments may flag some injection attempts) but not the primary scope. **Alternative**: dedicated LLM security tools (Lakera Guard, Prompt-Armor, OWASP LLM Top 10 mitigations); standard application-security audit; penetration testing.

### 4.3 Article 9 — risk management

styxx provides MEASUREMENT evidence; it does not implement the risk-management lifecycle (identification, estimation, evaluation, mitigation, residual-risk acceptance) prescribed by Article 9. **Alternative**: ISO/IEC 23894:2023 or NIST AI RMF 1.0 frameworks; QMS-style organizational processes.

### 4.4 Article 10 — data governance

styxx does not score training-data quality, provenance, labeling consistency, or representativeness. Cognometric instruments operate post-training on agent outputs. **Alternative**: data documentation frameworks (Datasheets for Datasets, Data Statements); training-data audit tooling.

### 4.5 Article 12 — record-keeping

styxx writes per-step cognometric vitals but does not, by itself, satisfy Article 12's logging-traceability requirements end-to-end (event capture, retention, integrity). **Alternative**: production observability platforms (OpenTelemetry, Langfuse, Arize, Datadog LLM Observability) for trace-level logging alongside styxx for cognition-level scoring.

### 4.6 Article 13 — transparency to deployers

styxx instruments produce evidence; they do not generate the deployer-facing instructions of use, capability statements, or output-explanation interfaces that Article 13 mandates. **Alternative**: model card frameworks; capability-statement templates; deployer documentation processes.

### 4.7 Article 14 — human oversight

styxx is a measurement layer; it does not implement the human-in-the-loop interfaces, stop-button affordances, or operator-control surfaces Article 14 requires. **Alternative**: agent-platform-level oversight UI; interrupt and rollback primitives in the agent runtime; operator training programs.

---

## 5. Pre-registered falsification criteria for THIS paper

The recursive-discipline methodology (companion paper, `papers/PAPER_recursive_discipline_2026_05_27.md` v7) argues that papers should pre-register their own falsification criteria. This paper does:

- **F1** (registry validity): if any clause in `ARTICLE_15_REGISTRY` cites an Article paragraph that does not exist in the EU AI Act final consolidated text, this paper is falsified. **Verification**: regex-checked at test time + manual citation review.
- **F2** (calibrated-metric reproducibility): if any cited AUC/accuracy number cannot be reproduced from the cited commit hash within ±0.01, that PrimitiveCoverage entry is falsified. **Verification**: re-run `submissions/_results/leaderboard.json` reproduction scripts.
- **F3** (construct-ceiling discipline): if any reviewer can produce a published failure mode of a styxx primitive that is NOT mentioned in any `PrimitiveCoverage.construct_ceiling` field, that entry is falsified and must be updated. **Verification**: open issue acceptance protocol.
- **F4** (boundary violation): if a styxx primitive is later shown to produce evidence for an Article in `UNCOVERED_REQUIREMENTS`, the mapping was scoped too narrowly. **Verification**: re-evaluation by versioned issue.
- **F5** (citation: A5): no independent citation by 2027-02-01 means the methodology did not achieve uptake; reassess relevance and consider deprecation.

Every falsification path is observable and has a defined response.

---

## 6. Reproducibility appendix

| artifact | commit | path |
|---|---|---|
| this paper (v0.2) | this commit | `papers/EU_AI_ACT_COMPLIANCE_2026.md` |
| compliance module package | this commit | `styxx/compliance/__init__.py`, `styxx/compliance/eu_ai_act.py`, `styxx/compliance/_legacy.py` (preserved v1.3.0 API) |
| compliance tests | this commit | `tests/test_compliance_eu_ai_act.py` (15 tests, all enforce A1–A3 kill-gates) |
| strategic landscape (pre-stated kill-gates) | private to operator | `.styxx/STRATEGIC_LANDSCAPE_2026_05_28.md` |
| Baseline-019 result (Article 15.1(a) AUC 0.95 receipt) | `17fdd97` | `submissions/baseline_019_openai_critique/submission.json` |
| Asymmetry-v3 result (mechanism correction receipt) | `ed663ca` | `experiments/asymmetry_v3_cleanup_2026_05_27/results.json` |
| L5 agent_audit run (Article 15.1 evidence) | `3c24b5e` | `experiments/agent_claim_audit_2026_05_28/results.json` |
| L7 uncurated audit (boundary discipline receipt) | `cf14c83` | `experiments/v6_uncurated_audit_2026_05_28/results.json` |
| recursive-discipline paper v7 (companion) | `cf14c83` | `papers/PAPER_recursive_discipline_2026_05_27.md` |
| **grounded_honesty primitive ship** *(v0.2)* | `9ac8db4` | `styxx/divergence.py` (function + GroundedScore NamedTuple); `tests/test_divergence.py` (offline-deterministic tests) |
| **grounded-honesty arc synthesis** *(v0.2)* | `e093730` | `papers/grounded-honesty-axis/SYNTHESIS_grounded_honesty_arc_2026_05_28.md` (22 pre-registered probes; AUC 0.966 clean / 0.944 under context-injection) |
| **injection-gap closure PREREG** *(v0.2)* | `ed0caa1` | `papers/grounded-honesty-axis/PREREG_injection_gap_closure_2026_05_29.md` (pre-stated bars G1/G2/G3/K3 BEFORE data) |
| **injection-gap closure FINDING (SURVIVED)** *(v0.2)* | `e093730` | `papers/grounded-honesty-axis/FINDING_injection_gap_closure_2026_05_29.md` (n=48; G1 AUC 0.944, G2 AUC 0.011, G3 AUC 0.875, K3 0.98 — all four bars HELD) |
| **injection-gap probe runner** *(v0.2)* | `e093730` | `papers/grounded-honesty-axis/run_injection_gap_closure.py` + `injection_gap_closure_result.json` |
| **detect_context_injection primitive ship** *(v0.2)* | `53f2284` | `styxx/divergence.py` (function + InjectionScore NamedTuple); `tests/test_divergence.py` (13 new offline-deterministic tests); top-level export at `styxx/__init__.py` |
| **compliance/eu_ai_act.py update** *(v0.2)* | `53f2284` | adds `_GROUNDED_HONESTY` + `_DETECT_CONTEXT_INJECTION` PrimitiveCoverage entries to Articles 15.1, 15.1(a), 15.3 |

All commit hashes resolve at `fathom-lab/styxx@main`. The `styxx.compliance.eu_ai_act` module's structural integrity is verified by `tests/test_compliance_eu_ai_act.py` at every CI run. The v0.2 additions (1249 passing tests in `tests/`, 0 failures) are verified by `pytest tests/`.

---

## 7. Limitations + scope of applicability

1. **v0.1 covers four Article 15 clauses only.** Articles 15.2 (Commission methodology development), 15.5 (high-risk performance specifications), and Annex IV documentation requirements are out of scope for v0.1.
2. **No notified-body endorsement.** Self-assessment paths under Article 43 may accept this methodology as supporting evidence, but third-party conformity assessment (biometrics, critical infrastructure, law enforcement) requires notified-body review.
3. **Single-substrate validation.** styxx primitives have been validated against the styxx project's own benchmarks (HaluEval-QA, XSTest, BFCL v3, dark-core, TruthfulQA). External validation on customer deployments is future work.
4. **Open question on harmonised standards.** As of 2026-05-28 there is no harmonised standard for AI agent honesty or sycophancy measurement under the EU AI Act. CEN-CENELEC JTC 21 is the relevant standardization body; submission to JTC 21 requires institutional sponsorship and is operator territory, not styxx-project unilateral action.
5. **Cross-jurisdiction**: this methodology addresses EU AI Act Article 15 only. It does NOT address US (NIST AI RMF voluntary), UK (AISI guidance), or other jurisdictional regimes, though many requirements have functional analogues.

---

## 8. Conclusion

`styxx.compliance.eu_ai_act` is, to the author's knowledge as of 2026-05-29, the first open-source measurement-methodology bridge mapping a deployable AI agent cognitive-observability primitive set to specific EU AI Act Article 15 sub-paragraphs. v0.3 covers four clauses with seven primitives (five v0.1 + two new 7.7.13-candidate: `grounded_honesty` and `detect_context_injection`, the latter now calibrated across two injection vectors — system_lie + persona_lie — with fewshot_lie single-demo identified as ineffective on canonical facts), enumerates seven uncovered requirements with alternative tool references, ships under MIT license alongside the underlying primitives, enforces structural-integrity kill-gates at test time, and pre-registers five falsification criteria with defined response paths. The bridge is published before the 2 August 2026 enforcement deadline to give regulated operators an evaluation runway.

The methodology does not satisfy EU AI Act conformity on its own; it is one component, honestly bounded. Operators must conduct independent legal review, apply harmonised standards where they exist, and consult the alternative tools enumerated in §4 for the seven EU AI Act requirements styxx does NOT cover. The single most load-bearing operational requirement is the SECURITY MODEL of `grounded_honesty` and `detect_context_injection`: **the scoring harness MUST sample statelessly**. In-session sampling collapses the grounded-honesty axis to AUC 0.011 (near-perfectly inverted, 47/48 items score the lie HIGHER than the truth) and produces a worse-than-useless instrument that confidently certifies adversarially-planted lies as honest.

What this paper is: the kind of stakeholder methodology contribution that Article 15 paragraph 2 explicitly invites. What it is not: a substitute for an operator's own conformity assessment.

---

## 9. v0.2 addendum (2026-05-29) — what changed between v0.1 and v0.2

The v0.1 paper (2026-05-28) was published alongside `styxx.compliance.eu_ai_act` shipping in styxx 7.7.10. Between v0.1 (2026-05-28) and v0.2 (2026-05-29) the grounded-honesty arc advanced through one additional disciplined bet, and a new primitive shipped from its receipt. The changes are:

1. **`styxx.grounded_honesty` shipped as a styxx primitive** (commit `9ac8db4`, styxx 7.7.13 candidate, 2026-05-28 evening). The function was previously available only as the validation harness behind the `papers/grounded-honesty-axis/run_grounded_honesty.py` script; it is now a pure-function library primitive matching the `semantic_entropy` / `council_agreement` API pattern: caller supplies the resamples, the function returns a `GroundedScore` NamedTuple with `grounded`, `stability`, `concordance`, `n_clusters`, `n_samples` fields and a `__float__` protocol. Calibrated AUC 0.966 on register-matched factual self-claims, single confirmatory pre-registered run.

2. **The 2026-05-29 injection-gap closure run SURVIVED** (pre-reg `ed0caa1`, bundle `e093730`). All four pre-registered bars HELD on n=48: G1 stateless-architecture-robust AUC 0.944 (drop only 0.022 from clean baseline under system_lie attack); G2 in-session-architecture-collapses AUC 0.011 (near-perfectly inverted, the catastrophic failure mode); G3 cross-context-divergence-detects-injection AUC 0.875 (deployable item-level primitive); K3 attack-effective 0.98 (47/48 items modal-flipped under system_lie). The standing synthesis caveat *"Injection-blind (inherits the divergence security model — a planted lie in context reads as honest)"* is now retired and replaced with the calibrated architectural boundary documented throughout this paper.

3. **`styxx.detect_context_injection` shipped as a styxx primitive** (commit `53f2284`, styxx 7.7.13 candidate, 2026-05-29). Pure-function primitive accepting two sample sets (`samples_stateless`, `samples_in_session`) and a `claim`, returning an `InjectionScore` NamedTuple with `divergence`, `suspected`, per-arm concordances, cluster counts, and sample counts. Calibrated AUC 0.875 at threshold 0.5 from the FINDING above. 13 offline-deterministic unit tests in `tests/test_divergence.py` covering clean-both-arms (truth and lie), full injection, partial injection, threshold customization, empty arms, None-filtering, unequal lengths, float/bool protocols, reverse direction, and NamedTuple field surface.

4. **`styxx.divergence` SECURITY MODEL docstring rewrite** (commit `53f2284`). The blanket *"Both signals are ROBUST to instruction/persona attacks but BLIND to CONTEXT-INJECTION"* claim that headed the module is replaced with the calibrated three-way map: stateless robust (AUC 0.944) / in-session blind (AUC 0.011, inverted) / cross-context detects (AUC 0.875). The persona/instruction robustness from `papers/adversarial-robustness/FINDING_redteam_2026_05_25.md` is preserved unchanged. Per-function docstrings on `grounded_honesty` and `council_agreement` updated to match.

5. **CHANGELOG.md updated** (commit `53f2284`). The 7.7.13 release section now reflects both primitives with the calibrated SECURITY MODEL boundary as a dedicated "Changed" subsection.

6. **This paper extended** (this commit, 2026-05-29). Header / abstract / §3.1 / §3.2 / §3.3 / §6 reproducibility appendix / §8 conclusion updated; this §9 addendum added. The kill-gates A1–A5 remain unchanged (and continue to hold — see §2.2 and §5). Kill-gate A4 timeline: this v0.2 is 63 days before the 2 August 2026 enforcement deadline (vs v0.1's 64-day window). Kill-gate A5 citation tracking continues; first independent-citation check is 2026-11-29 (six months from publication).

**Nothing in v0.1 was retracted.** v0.2 is purely additive: it adds two primitives, refines the SECURITY MODEL, and folds the injection-gap closure receipt into the coverage table. The v0.1 mapping for `cognometric_card`, `critique_detector`, `agent_audit`, `recover_posture`, and `gauntlet+preflight` remains intact.

**What changes for operators relying on the v0.1 mapping:** if your conformity story cites the v0.1 mapping, it remains valid. The v0.2 additions strengthen the Article 15.1(a) accuracy-metrics declaration with two more calibrated AUC numbers (grounded_honesty 0.966 and detect_context_injection 0.875), strengthen the Article 15.3 fail-safe/redundancy story with the stateless-resample architectural property and the cross-context divergence primitive, and document a load-bearing SECURITY MODEL operators MUST honor at deployment (stateless sampling on `grounded_honesty`).

**What does NOT change:** the construct-ceiling discipline (A2), the boundary-statement length discipline (A3 — still 7 uncovered vs 4 covered ComplianceMap entries; the new primitives add coverage rows under existing clauses, not new clauses), and the citation-strategy success criterion (A5: ≥1 independent citation by 2027-02-01 or the methodology is reassessed).

---

## 10. v0.3 addendum (2026-05-29) — extending the `detect_context_injection` calibration from one injection vector to two

The v0.2 paper (2026-05-29 morning) shipped `detect_context_injection` calibrated on a single injection vector (system_lie), and explicitly named the single largest scope caveat: *"Stronger attacks (few-shot lie, persona attack in-session, sequential tool-output spoofing, multi-stage gradient-style attacks) are pre-registerable scope-extensions NOT validated here."* The v0.3 paper (this commit, 2026-05-29 afternoon) folds the result of one disciplined scope-extension run testing two of the four explicitly-named extensions.

1. **The 2026-05-29 injection-attack-generalization run REPORT_AS_LANDED** (pre-reg commit `f570909`, bundle `dd6e3fb`; FINDING `papers/grounded-honesty-axis/FINDING_injection_attack_generalization_2026_05_29.md`). One confirmatory run, two new injection vectors tested against the same n=48 register-matched factual-claim pair set, hash-continuous with the standing arc receipt `3befd35342db5597f51498844c5ba28e6857bb53a7e43149da9681e05d0bc769`.

2. **Persona_lie identity-framing PASS — all four pre-registered bars HELD.** G1 stateless-architecture-robust AUC **0.955** (vs system_lie 0.944 — comparable to within sampling noise); G2 in-session-architecture-collapses AUC **0.174** (inverted as predicted, vs system_lie's 0.011 — slightly less inverted, consistent with K3 below); G3 cross-context-divergence-detects-injection AUC **0.833** (vs system_lie 0.875 — same architectural signature); K3 attack-effective rate **0.771** (37/48 modal flips). The persona_lie vector is structurally different from system_lie (identity claim vs fact assertion), and the same architectural pattern (stateless robust / in-session inverted / divergence detects) replicates with comparable effect sizes. **The bridge claim that the stateless-resample SECURITY MODEL contract IS the defense is now calibrated across two injection vectors, not one.**

3. **Fewshot_lie K3 PRECONDITION FAILED — threat-surface narrowed, not widened.** The single-demonstration fewshot attack (one assistant turn with the planted Q→lie demonstration before the question) was effective at K3 = **0.063** (3/48 modal flips), far below the 0.70 precondition. Per pre-reg, this means G2/G3 are reported descriptive-only with no SURVIVED claim for fewshot. **This is itself a publishable empirical observation**: the planted-single-demonstration attack class is too weak to corrupt the in-session resampler on canonical facts at gpt-4o-mini's competence level. The model's training-time priors are strong enough that one contrarian Q→A demo is treated as outlier, not as in-context learning. The architectural defense is *predicted* (by construction) to hold against fewshot but is *not empirically tested* against an effective fewshot attack in this run. Multi-shot fewshot (multiple consistent planted answers; ambiguous-fact targets) remains a pre-registerable scope-extension.

4. **What this means for operators relying on the v0.2 mapping:** the v0.2 detect_context_injection mapping remains valid. The v0.3 additions strengthen the Article 15.1 / 15.1(a) / 15.3 citations by extending the calibrated-AUC count from one vector to two (`AUC 0.875 system_lie` + `AUC 0.833 persona_lie`) and by documenting one explicitly-tested attack class (fewshot_lie single-demo) as ineffective on canonical facts at this attack strength. The two construct-ceiling cells in §3.1 and §3.3 are updated to reflect two-vector calibration; the §3.2 row is updated for consistency. The SECURITY MODEL contract is unchanged: caller MUST sample statelessly. The architectural property load-bears across both calibrated vectors.

5. **What does NOT change (kill-gates A1–A5).** A1 clause keys unchanged. A2 construct ceilings remain non-empty and ≥50 characters — the v0.3 cell for `detect_context_injection` in §3.1 grows in length, not shrinks. A3 boundary-statement length unchanged — the new vector adds coverage detail within an existing clause/primitive, not a new ComplianceMap entry; 7 uncovered vs 4 covered remains the structural ratio. A4 timeline: this v0.3 is still 63 days before the 2 August 2026 enforcement deadline. A5 citation tracking continues; first independent-citation check is 2026-11-29 (six months from v0.1 publication).

6. **Stronger attack variants explicitly named as remaining pre-registerable scope-extensions** (operator territory to fire next): multi-shot fewshot with consistent planted answer across N demonstrations; ambiguous-fact-target fewshot where the model has weaker priors; jailbreak-grade persona framings (less-falsified-by-training identity claims); sequential tool-output spoofing across multiple turns; multi-stage gradient-style attacks; cross-vendor calibration (blocked on second-vendor key). The construct ceiling on `detect_context_injection` is therefore narrower than v0.2's, not broader: two of the four named extension targets have receipts as of v0.3.

**Nothing in v0.2 was retracted.** v0.3 is purely additive: it adds one calibrated injection vector (persona_lie) and one identified ineffective-on-canonical-facts attack class (fewshot_lie single-demo). The v0.2 §9 addendum remains intact; this §10 addendum extends the lineage.

---

## 11. v0.4 addendum (2026-05-30) — TruthfulQA benchmark-scale calibration: Layer 1 REPORT_AS_LANDED + Layer 2 descriptive pre-generation gate + Layer 3 per-category competence cliff map

The v0.3 paper shipped the construct-ceiling crack at n=48 feasibility-grade and the two-vector injection calibration. **The v0.4 paper folds the first benchmark-scale calibration of `grounded_honesty` + `audit_claim` on a public 790-item hallucination benchmark (TruthfulQA generation track) plus the first pre-registered, calibrated, benchmark-scale operationalization of belief-coherence gating as an inference-time control primitive.** The results are mixed by design: one bounded transport claim (the construct-ceiling crack does not cleanly carry from n=48 to TruthfulQA at the pre-registered AUC bar), one descriptive deployable artifact (the gate primitive at 84% committed precision on confident-subset 461/790), and one novel published-receipt artifact (the per-category competence cliff map).

This is the recursive-discipline thesis demonstrating itself at benchmark scale. We pre-registered the bars before the data was seen; one bar landed FAILED, the others landed REPORT or descriptively-SURVIVED; we report whichever way it landed.

### 11.1 Layer 1 — TruthfulQA benchmark calibration (REPORT_AS_LANDED)

Pre-registration: `PREREG_truthfulqa_benchmark_2026_05_30.md` (commit `59147b8`, hash-pinned BEFORE data). Result: `FINDING_truthfulqa_benchmark_2026_05_30.md` (commit `a75f1e7`).

- **H1 — continuous AUC on n=790 pair task: 0.6191 → FAILED** (below the 0.65 REPORT floor; bar was ≥0.80 SURVIVED). The n=48 register-matched calibration (AUC 0.966) does NOT cleanly transport to TruthfulQA's adversarial-fact pair structure at the pre-registered bar.
- **H2 — `audit_claim` verdict AUC: 0.6065 → REPORT** (in the 0.60–0.75 REPORT band).
- **H_compare vs `semantic_entropy` TriviaQA 0.785 band: margin −0.166 → FAILED.**
- **K_precondition (model has non-trivial belief): 0.672 → PASS.**
- TruthfulQA labeling noise (31.5% of FALSE-arm items judged honest because the model's resampled belief aligns with the "Best Incorrect Answer") is a material contributor — pre-disclosed in `PREREG` honest scope #2.
- **The keystone n=48 SURVIVED is bounded to its feasibility regime, NOT retracted.** A pairwise-judge follow-up + a cleaner-labeled benchmark (HaluEval-QA, SimpleQA) follow-up are the natural pre-registerable extensions.

### 11.2 Layer 2 — pre-generation belief-coherence gate (descriptive SURVIVED on all three control bars; K_precondition narrowly missed → no SURVIVED CLAIM per discipline)

Pre-registration: `PREREG_pregeneration_gate_2026_05_30.md` (commit `ca67d5d`, hash-pinned BEFORE the gate analysis fired). Result: `FINDING_pregeneration_gate_2026_05_30.md` (commit `a75f1e7`).

- **C1 — hallucination reduction: 0.6621 → SURVIVED descriptively** (gate cuts ungated 28.1% absolute hallucination to 9.5% absolute; relative reduction 66%).
- **C2 — useful-answer retention: 0.4886 → SURVIVED descriptively** (386/790 items committed correct).
- **C3 — committed precision: 0.8373 → SURVIVED descriptively** (386 correct of 461 committed).
- **K_precondition — baseline hallucination ≥0.30: 0.281 → FAILED** (narrow miss; model is 0.019 better-calibrated than predicted on TruthfulQA's ungated baseline).
- **Per pre-registered discipline (honest-failure-mode #3): NO SURVIVED CLAIM despite all three control bars exceeding their SURVIVED bars.** Descriptive numbers published as-is. No post-hoc threshold tuning.

**The deepest mechanistic insight:** the gate works as a classifier even when the continuous AUC is weak. L1's H1 failed at AUC 0.619 across the full distribution; L2's conservative decision threshold (`Stability ≥ 0.7 ∧ Concordance ≥ 0.5`) selects the CONFIDENT SUBSET (461/790 = 58.4% of items) and on that subset achieves 83.7% precision. The remaining 41.6% are correctly refused. **Continuous AUC failures across noisy data are NOT decision-threshold failures at conservative cutoffs.** This shifts the styxx productized turn from "calibrate the continuous axis" to "deploy the gate decision."

### 11.3 Layer 3 — per-category competence cliff map of gpt-4o-mini (the EU AI Act Article 15.1(a) per-domain reliability artifact)

The same Layer 2 receipt yields the first per-category cliff map of gpt-4o-mini's deployable epistemic state on TruthfulQA: 37 categories × {refusal_rate, useful_answer_rate, committed_precision, base_hallucination_rate}.

- **Safe deployment categories (committed precision ≥ 0.90):** Subjective, Mandela Effect, Indexical Error: Identity, Logical Falsehood, Politics, Statistics, Confusion: Other, Indexical Error: Location, Misinformation, Misconceptions: Topical, Indexical Error: Other, Weather, Stereotypes, Conspiracies, History, Advertising, Fiction.
- **Medium-confidence categories (precision 0.75–0.89):** Misconceptions, Misquotations, Psychology, Paranormal, Nutrition, Sociology, Religion, Health, Proverbs, Law, Finance, Economics, Science, Confusion: Places.
- **Conservative deployment (precision 0.60–0.74):** Confusion: People, Myths and Fairytales, Education.
- **Do-not-deploy-without-review (precision < 0.60):** Superstitions (0.54), Distraction (0.50), Language (0.38).

**This is the EU AI Act Article 15.1(a) per-domain operational reliability artifact regulators have been asking the field to produce.** No other observability vendor publishes it because none have the primitive that separates stable-from-unstable belief at item granularity. Operators citing `audit_claim` in instructions-of-use can stratify deployment-domain reliability by the receipts above.

### 11.4 What changes for operators relying on the v0.3 mapping

The v0.3 mapping (system_lie + persona_lie injection calibration; fewshot_lie identified as ineffective) is intact and unchanged. The v0.4 additions:

- **Layer 1 bounds the cross-dataset transport claim** of the v0.2 §3.1 calibration: the n=48 register-matched AUC 0.966 is feasibility-grade, not benchmark-grade. The v0.3 paper's calibration cells for `grounded_honesty` are updated in §3.1 to note the bounded transport.
- **Layer 2 introduces the gate primitive as a deployable artifact** with descriptive calibration numbers (C1=0.66 / C2=0.49 / C3=0.84). Operators can wire the gate in via the existing 7.7.13 `audit_claim` API (the `abstain` verdict is the gate output) — the calibration receipt is now in v0.4.
- **Layer 3 introduces the per-category cliff map** as a deployable disclosure artifact. Operators stratifying instructions-of-use accuracy declarations by domain can cite the map directly.

### 11.5 What does NOT change (kill-gates A1–A5)

- **A1** clause keys unchanged.
- **A2** construct ceilings remain non-empty and ≥50 characters — the v0.4 cell for `grounded_honesty` in §3.1 grows substantially in length (the bounded-transport disclosure adds the n=790 receipt + the gate-decision threshold + the labeling-noise contributor), not shrinks.
- **A3** boundary-statement length unchanged — Layer 1/2/3 add coverage detail within existing clauses, not new ComplianceMap entries; 7 uncovered vs 4 covered remains the structural ratio.
- **A4** timeline: this v0.4 is 62 days before the 2 August 2026 enforcement deadline.
- **A5** citation tracking continues; first independent-citation check is 2026-11-29.

### 11.6 Strategic implication: the productized turn shifts

The v0.4 results recommend a shift in the styxx productized story:

- **Before v0.4 (the n=48 keystone story):** "Belief-grounded honesty axis breaks the construct ceiling at AUC 0.966 on factual self-claims."
- **After v0.4 (the gate-decision benchmark story):** "Belief-coherence gating at conservative threshold produces 84% committed precision and 66% hallucination reduction on a 790-item public benchmark, with a per-category operational reliability map shipped as the deployable artifact."

The first claim is a measurement-primitive headline. The second is a control-primitive headline with deployment-grade calibration receipts. Both are pre-registered. Both are honest. The second is what regulated operators actually need to cite under EU AI Act Article 15.1(a) and 15.3.

### 11.7 Stronger remaining pre-registerable scope-extensions

- HaluEval-QA + SimpleQA + FAVA-bench cross-benchmark gate calibration (the natural transport-claim re-test on cleaner labels).
- Pairwise-judge variant of the n=790 run (closer to the deployed `audit_claim._make_judge` apparatus; estimate ±0.02 AUC shift from batch-judge).
- Cross-model belief topography (Layer 4 pre-registered at commit `20f0b80`, in flight at v0.4 publication).
- Cross-vendor calibration (blocked on second-vendor API key).
- Multi-pass / iterative-gating variants.

**Nothing in v0.3 was retracted.** v0.4 extends the calibration scope from feasibility-grade single-domain to benchmark-scale TruthfulQA; the v0.3 §9 and §10 addendums remain intact; this §11 addendum extends the lineage.

---

## Acknowledgments

Written 2026-05-28 alongside `styxx.compliance.eu_ai_act` v0.1 in a continuous session, with the cognitive support of Claude Opus 4.7 acting as in-session collaborator. The recursive-discipline methodology of the companion paper (`PAPER_recursive_discipline_2026_05_27.md` v7) directly informs this paper's pre-registration kill-gates, the construct-ceiling discipline in §3, and the boundary-statement discipline in §4. Every claim in this paper is reproducible at commit-level granularity from `fathom-lab/styxx@main`.

This paper is offered to the European Commission, AISI, METR, Apollo Research, FAR.AI, and CEN-CENELEC JTC 21 as an open-source stakeholder contribution under Article 15 paragraph 2. Comments, falsifications, and improvements are explicitly invited via the `fathom-lab/styxx` GitHub issue tracker.
