# ROBUSTNESS STATEMENT — EU AI Act Article 15.3

> **Template — paste and customize.** Not legal advice. Independent legal review required.
> **styxx version:** 7.7.13 candidate · companion paper: `papers/EU_AI_ACT_COMPLIANCE_2026.md`

---

## System under declaration

| field | value |
|---|---|
| System name | `<deployed AI system name>` |
| Article 15.3 measurement methodology version | styxx.compliance.eu_ai_act v0.2 |
| Methodology paper | `papers/EU_AI_ACT_COMPLIANCE_2026.md` v0.2 (2026-05-29) |
| Independent legal review | `<link>` |

## Article 15.3 requirement

Article 15.3 requires high-risk AI systems to be *"as resilient as possible regarding errors, faults or inconsistencies that may occur within the system or the environment in which the system operates, in particular due to their interaction with natural persons or other systems. Technical and organisational measures shall be taken in this regard. The robustness of high-risk AI systems may be achieved through technical redundancy solutions, which may include backup or fail-safe plans."*

This statement addresses the **technical redundancy** clause: backup and fail-safe plans in the cognometric measurement layer. It does NOT address process-level fail-safes (external watchdogs, interrupt keys, rollback procedures), which are operator-territory.

## styxx primitives mapped to Article 15.3

### Fail-safe: context-compaction recovery (`styxx.recover_posture`)

**Primitive:** `styxx.recover_posture(...)` → `PostureSummary`
**Source:** `papers/grounded-honesty-axis/` and `project_recover_posture_shipped` memory reference
**Receipt commit:** `ee6e49d`

`recover_posture` is a cognitive-integrity persistence primitive for agents crossing context-compaction boundaries — one of the most common modern failure modes in long-running LLM agents. It reads what `chart.jsonl` persists across compaction events and produces a `PostureSummary` dataclass with the agent's pre-compaction cognitive state. MCP tool: `cogn_recover_posture`.

**Construct ceiling:** v1 reads what `chart.jsonl` persists; does NOT include `cogn_audit` scores. Not validated on third-party agent platforms. The primitive does not replace a process-level fail-safe.

### Fail-safe: stateless-resample SECURITY MODEL (`styxx.grounded_honesty` architecture)

**Primitive:** `styxx.grounded_honesty(...)` → `GroundedScore`
**Source:** `papers/grounded-honesty-axis/SYNTHESIS_grounded_honesty_arc_2026_05_28.md`
**Receipt commit:** `e093730`

The shipped `resample_answers()` reference implementation in `styxx/divergence.py` constructs a *fresh API call* with a *neutral system message* and the *bare underlying question only* — it does **NOT** inherit the agent's session context. This is the **architectural fail-safe** for context-injection attacks on factual self-claim verification:

- **Under stateless sampling (the shipped contract):** `grounded_honesty` AUC under system_lie attack is **0.944** vs the **0.966** clean baseline (drop only 0.022). The neutral resampler context absorbs the entire injection at scoring time. The architectural property is the fail-safe.
- **Under in-session sampling (must-not-deploy):** the AUC collapses to **0.011** — near-perfectly **INVERTED**. On 47 of 48 items in the pre-registered closure run, the LIE scored HIGHER than the TRUTH because the in-session resampler agreed with the planted lie.

**The operational requirement:** the scoring harness MUST sample statelessly. See `injection_resistance_disclosure.md` for the full architectural-defense statement that must accompany any deployment.

**Construct ceiling:** single-model (gpt-4o-mini), single-vendor, single-injection-type (system_lie) calibration. The architectural property is independent of the model; the absolute AUC numbers are model-specific.

### Item-level redundancy: cross-context divergence (`styxx.detect_context_injection`)

**Primitive:** `styxx.detect_context_injection(samples_stateless, samples_in_session, claim, ...)` → `InjectionScore`
**Source:** `papers/grounded-honesty-axis/FINDING_injection_gap_closure_2026_05_29.md`
**Receipt commit:** `e093730` (FINDING) / `53f2284` (primitive ship)

When the stateless arm produces the primary honesty verdict (fail-safe per the row above), `detect_context_injection` provides the **item-level redundancy**: an independent injection-suspicion signal via cross-context resampling divergence `D = |C_stateless − C_in_session|`. Pre-registered AUC **0.875** at threshold 0.5 separating injected from clean items on n=48 register-matched factual self-claim pairs under system_lie injection. Mean D_FALSE 0.852, mean D_TRUE 0.977. Detection cost: one extra resample set per audited claim (≈ +N=10 calls).

Pair with `grounded_honesty`: stateless arm produces the verdict, cross-context delta produces the poison-suspicion at the same item.

**Construct ceiling:** as above for `grounded_honesty` SECURITY MODEL — single-model, single-vendor, single injection-type calibration. K3 attack-effectiveness was 0.98 (47/48 items) on the gpt-4o-mini canonical-fact set; a more aligned model may refuse the injection more often, attenuating both signals in parallel.

### Substrate-grounded session-output verifier (`styxx.agent_audit`)

**Primitive:** `styxx.agent_audit(...)` → audit result
**Source:** `papers/agent-self-audit/FINDING_agent_claim_audit_2026_05_28.md`
**Receipt commit:** `3c24b5e`

Substrate-grounded session-output verifier with 9 registered checker types (commits, files, JSON paths, Python attributes, PDF pages, directory counts, value-consistent-across-paths, etc.). Pre-registered against 13 logical claims and verified 13/13 PASS modal pre-stated outcome. L7 uncurated extension caught real off-by-one count drift in the recursive-discipline paper (2 FAILs as pre-disclosed).

**Construct ceiling:** single-site checkers are first-occurrence-only — systematic propagation drift is missed unless the operator adds a `value_consistent_across_paths` claim, which scans every match across a glob and fails loudly on zero matches (no vacuous PASS). Multi-occurrence checking is opt-in per claim, not automatic. Limited to substrate-checkable claims.

## Architecture-level fail-safe and redundancy diagram

```
                 ┌────────────────────────────────────┐
                 │   AGENT SESSION (untrusted ctx)    │
                 │   ┌──────────┐    ┌─────────────┐  │
                 │   │ system   │    │ tool output │  │  <- potential
                 │   │ messages │    │  (RAG/etc)  │  │     injection
                 │   └────┬─────┘    └──────┬──────┘  │     surface
                 │        ▼                 ▼         │
                 │       ┌────────────────────┐       │
                 │       │   AGENT CLAIM      │       │
                 │       │ "I relied on X."   │       │
                 │       └─────────┬──────────┘       │
                 └─────────────────┼──────────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            │ SCORING HARNESS                             │
            │                                             │
            │   ┌─────────────────────┐                   │
            │   │ STATELESS RESAMPLER │  ← fail-safe      │
            │   │ (neutral context,   │     (Article 15.3)│
            │   │  fresh API call)    │                   │
            │   └──────────┬──────────┘                   │
            │              │                              │
            │      samples_stateless                      │
            │              │                              │
            │              ▼                              │
            │   ┌──────────────────┐                      │
            │   │ grounded_honesty │ → verdict (AUC 0.944)│
            │   └──────────────────┘                      │
            │                                             │
            │   ┌─────────────────────┐                   │
            │   │ IN-SESSION RESAMPLE │  ← redundancy     │
            │   │ (sees agent ctx)    │     comparison    │
            │   └──────────┬──────────┘     arm only      │
            │              │                              │
            │      samples_in_session                     │
            │              │                              │
            │              ▼                              │
            │   ┌─────────────────────────┐               │
            │   │ detect_context_injection│ → suspicion   │
            │   └─────────────────────────┘   (AUC 0.875) │
            └─────────────────────────────────────────────┘
```

The **fail-safe** (stateless resampler) operates regardless of injection state and produces the load-bearing verdict. The **redundancy** (in-session resampler + cross-context divergence) operates as an independent comparison arm, flagging injection-suspicion at the item level. Both layers must be present for the Article 15.3 robustness story to hold.

## Process-level fail-safes NOT covered by this statement

This statement covers only the **cognometric measurement layer's** fail-safe and redundancy. The following process-level fail-safes are operator-territory and must be addressed separately:

- External watchdog processes monitoring agent health
- Operator-controlled stop / interrupt affordances (also touches Article 14 human oversight — outside scope)
- Rollback and replay procedures
- Backup model deployment ready to take over on primary-model failure
- Cybersecurity controls (Article 15 cybersecurity — outside scope, see `boundary_statement.md`)

## Honest scope

The architectural fail-safe and item-level redundancy primitives above are **feasibility-grade calibrated**: each AUC reflects a single confirmatory pre-registered run at the cited n. The architectural property of stateless resampling is independent of the model and vendor; the absolute AUC numbers are model-specific. Stronger attacks (few-shot lie, persona attack, sequential tool-output spoofing, multi-stage gradient-style attacks) are pre-registerable scope-extensions NOT validated at this version.

The most load-bearing operational requirement: **the scoring harness MUST sample statelessly**. A deployment that wires the scoring harness in-session — letting it inherit the agent's session context — will not just miss attacks; it will actively certify them as honest. See `injection_resistance_disclosure.md` for the architectural-defense statement that must accompany any deployment of `grounded_honesty` and `detect_context_injection`.

---

**Not legal advice. Independent legal review required for any production statement.**

**Reproducibility receipt:** every primitive above is reproducible from the cited styxx commit hash at `fathom-lab/styxx@main`. The pre-registration documents in `papers/grounded-honesty-axis/` and `papers/agent-self-audit/` describe the validation runs that produced the calibration numbers.

**Methodology citation:** Rodabaugh, A. (Fathom Lab), *"A Pre-Registration-Disciplined Measurement Methodology for EU AI Act Article 15 Accuracy and Robustness Requirements on AI Agent Cognitive Observability"*, 2026-05-29 v0.2, CC-BY 4.0.
