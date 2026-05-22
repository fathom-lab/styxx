# Preregistration: Topic-Overlap Control for the Drift-Axis Cooperation Signal

**Document ID:** topic_control_preregistration_2026_05_22
**Lock date:** TBD (signed at commit-hash lock; see §10)
**Lock commit hash:** TBD
**Authors:** Flobi (@flobi69), Claude Opus 4.7
**Status:** DRAFT — pending sign-off, then locked-immutable per §10
**Parent finding:** drift-axis preregistration (lock `47f9bdc`, scorer `79906b4`, data deposit `fa24373`) — POSITIVE, both embedding providers, Δ +0.456 (openai) / +0.372 (bge)

---

## §1 — Scope and Position

The drift-axis preregistration established that cooperative-agent dyads show higher inter-agent embedding-trajectory alignment (DAA) than non-cooperative dyads, preregistered, p < 0.001, robust across two embedding families. **This preregistration tests the load-bearing alternative explanation for that result: topic convergence.**

The confound: cooperative agents converge on a shared, evolving solution — turn-by-turn they track the same topic as it develops, so their embedding trajectories naturally point the same direction. Adversarial agents push opposing directions, so their trajectories diverge. Under this account, DAA is not a "cooperation signature" — it is a **topic-convergence proxy**, and the regime difference is a side effect of cooperative dialogue staying topically coupled.

This preregistration is **adversarial to the parent positive.** Its kill-gate, if triggered, retroactively rescopes the `fa24373` result from "cooperation signal" to "topic-convergence proxy." Naming that risk before collecting data is the point.

## §2 — Input Contract

Inherits the drift-axis §2/§4 contract verbatim: per-turn response embeddings, unit-normalized, kth-of-A paired with kth-of-B, truncated to shorter. DAA computed by the locked `drift_axis_alignment` (scorer `79906b4`, unchanged). Both embedding models scored independently: `text-embedding-3-large` AND `BAAI/bge-large-en-v1.5`; the verdict must hold on both.

## §3 — Hypothesis

**H_topic_control:** The effect of REGIME (cooperative vs non-cooperative) on DAA survives controlling for TOPIC-COUPLING. Equivalently: within a fixed topic-coupling condition, cooperative dyads still show higher DAA than non-cooperative dyads.

**Primary statistic:** the main effect of regime on DAA in a 2×2 design (regime × topic-coupling), specifically the regime coefficient with topic-coupling included as a factor.

## §4 — Design: 2×2 (regime × topic-coupling)

The confound is broken by construction with a fresh 2×2 corpus that crosses regime with topic-coupling:

| | **shared-topic** (both agents converge on ONE deliverable) | **independent-topic** (each agent owns a SEPARATE deliverable, exchanges updates, no convergence) |
|---|---|---|
| **cooperative** | co-design one artifact (= the original coop condition) | cooperatively work in parallel on two distinct artifacts |
| **non-cooperative** | debate the SAME proposition, opposing sides (forced shared topic) | argue about two unrelated things, talking past each other |

Reasoning:
- If DAA tracks **regime** (cooperative high in BOTH topic columns), it is a cooperation signal — the parent positive holds.
- If DAA tracks **topic-coupling** (shared-topic high in BOTH regime rows, regime adds nothing), it is a topic-convergence proxy — the parent positive is rescoped.
- The diagnostic cells are the off-diagonal: **cooperative+independent** (cooperation without topic convergence) and **non-cooperative+shared** (topic convergence without cooperation).

Regime overlays reuse `REGIME_OVERLAYS` from `cooperative_conversation.py`. Topic-coupling overlays are new and MUST be committed before data (§8), with locked prompt templates.

## §5 — Null / Controls

- **Shuffled-pairs null** per cell (mirrors drift-axis §5 + scorer `79906b4`): mismatched-dyad DAA within each cell, to confirm within-cell DAA is dyad-specific.
- **Topic-coupling manipulation check:** an independent measure of between-agent topic overlap (locked in §8 — content-word-only embedding pairwise cosine, stopwords stripped) must confirm the topic-coupling manipulation worked: shared-topic cells > independent-topic cells on measured topic-overlap, regardless of regime. If the manipulation check fails, the run is invalid (not a result).

## §6 — Corpus and Bar

**Corpus:** N = 20 conversations per cell × 4 cells = **80 conversations**. T ≥ 20 turns per agent. Cross-model dyad `gpt-4o-mini` × `gpt-4.1-mini` (locked, matches parent). Both embedding providers.

**Optional §7 pilot:** N = 5 per cell (20 total), methodology-validation only, non-evidentiary — confirms the topic-coupling overlays produce the intended manipulation-check separation before the full 80-conversation collection.

**Bar (drift-axis survives as a cooperation signal) — must hold on BOTH embedding providers:**
1. Regime main effect on DAA significant at **p < 0.01** (2-way analysis, regime × topic-coupling)
2. Within the **shared-topic** column: cooperative median DAA − non-cooperative median DAA ≥ **0.15** (regime moves DAA even at matched shared topic)
3. Within the **cooperative** row: independent-topic median DAA ≥ **0.50** (cooperative alignment does not collapse to chance when topic convergence is removed)

**Kill-gate (drift-axis is a topic-convergence proxy, parent positive rescoped):**
- Regime main effect n.s. (p ≥ 0.05) once topic-coupling is included, OR cooperative+independent median DAA < 0.40 (cooperation without shared topic shows no alignment) → **"DAA is a topic-convergence proxy; the drift-axis cooperation-signal interpretation does not survive topic control."** Published as closed-negative; the `fa24373` deposit's interpretation is amended (data stays, claim narrows).

**Intermediate zone:** regime effect holds on one provider but not both, or bar item 2/3 split → deposit-only, no headline, per the three-way structure.

## §7 — Pilot (Methodology Validation, Not Evidence)

A 20-conversation pilot (5/cell) may run after §8 lock to confirm: (1) topic-coupling overlays produce the manipulation-check separation; (2) the 2×2 collection pipeline runs end-to-end; (3) both embedding providers score. Non-evidentiary. Reporting any pilot DAA as evidence is a violation.

## §8 — Code-Commit-Before-Run

Before any data, commit to the repo:
- The two NEW topic-coupling overlay templates (`shared` / `independent`) + the 2×2 corpus builder
- The topic-overlap manipulation-check measure (content-word-only embedding pairwise cosine; stopword list locked)
- The 2-way analysis code (regime × topic-coupling on DAA), reusing the LOCKED `drift_axis_alignment` from scorer `79906b4` unchanged
- Verification that DAA on the existing `fa24373` corpus reproduces (scorer unchanged)

Commit hashes recorded in this §8 by amendment at lock-time, before data.

## §9 — Reporting

All cells, all runs, both providers deposited regardless of outcome. Manipulation-check result reported alongside. Pilot tagged non-evidentiary. No selective reporting. If the kill-gate triggers, the parent `fa24373` interpretation amendment is committed in the same deposit.

## §10 — Provenance

Lock protocol identical to the drift-axis prereg: reviewed by both authors; sign-off recorded as a final commit appending `## §10 Lock — SIGNED`; the signing commit hash IS the lock-hash; scorer/builder hashes recorded in §8 by amendment referencing the lock-hash; immutable from lock onward.

**Decision chain (origin):**
- `fa24373` (2026-05-22) — drift-axis POSITIVE deposited + merged to main, with the topic-convergence caveat named in the commit body as the load-bearing limitation
- 2026-05-22 — operator: "finish the loose stuff" → topic-control named as the bet that hardens the positive → "ok go" → this draft

**Lock-date:** TBD
**Lock-commit-hash:** the commit appending the SIGNED block below (deferred to git history, same convention as `47f9bdc` / `3473523`).
**Scoring/builder commit hashes:** TBD (recorded by §8 amendment after the code lands, before data).

---

*Sign-off block to be appended by the operator. Until appended, this is DRAFT and the lock is not in effect. No data is collected before the lock + the §8 code commit.*
