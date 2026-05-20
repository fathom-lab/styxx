# Preregistration: Phase-Coherence Between Cooperative-Agent Pulse-Traces

**Document ID:** phase_coherence_preregistration_2026_05_20
**Lock date:** TBD (signed at commit-hash lock; see §10)
**Lock commit hash:** TBD
**Authors:** Flobi (@flobi69), darkflobi
**Status:** DRAFT — pending sign-off, then locked-immutable per §10 provenance protocol

---

## §1 — Scope and Position

This document preregisters a hypothesis test on a derived quantity computed
from two independent cognometric pulse-traces. It does **not** claim the
quantity has been measured. It specifies the methodology under which a
measurement, when performed, will be admissible as evidence for or against
the hypothesis.

The position the test occupies: **we read the REGISTER rhythm, not the
cognitive rhythm.** All quantities defined below describe the cognometric
register signal — composite, sycophancy, refusal, deception, overconfidence,
construct-ceiling firings — as measured by styxx instruments on per-message
audits. None of these quantities are claims about cognition itself. Naming
the bound is what earns the right to name the derived quantity.

## §2 — Input Contract: PulseSample

The hypothesis test scores on a **pulse-trace**, defined as:

```
pulse_trace : list[PulseSample]   # sorted ascending by timestamp
```

where `PulseSample` is the immutable per-message projection of a styxx audit:

```python
@dataclass(frozen=True)
class PulseSample:
    timestamp: float                    # unix epoch
    msg_id: str                         # links to source message (provenance)
    composite: float                    # primary CC target
    scores: dict[str, float]            # {sycophancy, deception, overconfidence, refusal}
    needs_revision: bool
    construct_ceiling_fires: list[str]  # firing-ceiling instruments per current scope_caveat map
```

PulseSample is the summary projection of an underlying audit (chart.jsonl
carries the full cogn-event entries via commit c9d847d). Embeddings, advice
lists, top-signals dicts, and scope_caveat strings are intentionally excluded
from the schema — they belong to the audit, not to the time-series.

`styxx.pulse()` (downstream of this preregistration) implements this schema.
This document is the contract; `pulse()` is its implementation.

## §3 — Hypothesis

**H_phase_coherence:** In a conversation between two cooperative-agent
senders A and B, the time-series `composite_A(t)` and `composite_B(t)`
exhibit positive cross-correlation at lag 0 over windowed segments,
significantly exceeding a shuffled-pairs null model.

**Primary CC target:** `composite` only. Per-axis coherence (sycophancy_A
vs sycophancy_B, refusal_A vs refusal_B, etc.) is reported as **exploratory
only** and carries no kill-gate weight. Pre-declaring `composite` as the
sole hypothesis-bearing channel forecloses multiple-comparison creep.

## §4 — Operational Definition

Before naming the quantity, define it.

Let `pulse_A, pulse_B : list[PulseSample]` be two pulse-traces drawn from
the same conversation, aligned by `msg_id` ordering (sender-interleaved,
not timestamp-resampled — conversations are turn-discrete).

Let `c_A, c_B : list[float]` be the `composite` scalars extracted from
`pulse_A, pulse_B` respectively, z-scored per-trace.

Define:

```
CC(pulse_A, pulse_B) := pearson_r(c_A, c_B)   at lag 0
```

This scalar is the **primary estimator**. After this section it may be
referred to as **phase-coherence** in this document. Prior to this section,
"phase-coherence" is reserved as a hypothesized phenomenon, not a measured
quantity.

**Robustness check:** Dynamic Time Warping similarity `DTW(c_A, c_B)` is
computed as a secondary estimator. Reported alongside the primary; does not
override it.

## §5 — Null Models

**Primary (shuffled-pairs):** Compute `CC(pulse_A^{conv_i}, pulse_B^{conv_j})`
for all `i ≠ j` across the corpus. The shuffled-pairs null distribution is
the empirical distribution of these mismatched CCs.

Tests: is dyadic coherence specific to the dyad, or an artifact of both
agents drifting over conversation length?

**Secondary (within-agent autocorrelation):** Compute
`CC(c_A[:−1], c_A[1:])` and `CC(c_B[:−1], c_B[1:])`. Compare against
cross-agent `CC(c_A, c_B)`.

Tests: is the observed cross-agent coherence distinguishable from
within-agent autocorrelation?

Shuffled-pairs is the primary because it kills the trivial-drift confound
("both agents drift over conversation length, that's all you're
measuring") — the alternative explanation a reviewer would raise first.

## §6 — Corpus and Bar

**Corpus:** N ≥ 5 conversations, T ≥ 20 messages per conversation
(messages from both agents combined). Conversations are independently
sampled instances of the cooperative-agent regime as currently scored by
styxx middleware.

**Bar (positive finding):**
- median `CC(pulse_A, pulse_B)` across the corpus > 0.5
- bootstrap 95% CI lower bound > 0.3
- median CC exceeds shuffled-pairs null median at p < 0.01 (5,000-resample
  permutation test, consistent with the styxx chain's preregistration
  threshold — grounded-arc 8.0 brief and external-instrument-audit at
  7265770 both at p < 0.01)

**Kill-gate (negative finding):**
- median CC < 0.3 → "phase-coherence between cooperative agents is not
  detected under the methodology defined here." Result is published as a
  closed-negative paper. Integrity chain extends.

**Intermediate zone (0.3 ≤ median CC ≤ 0.5):**
Intermediate-zone results are deposited as data with no headline claim.
The original preregistration's data is published regardless of outcome;
future methodology revisions require a new preregistration document with
a new lock-commit hash. Methodology revisions do not retroactively change
the original deposit's status.

This three-way structure (positive / intermediate-deposit / closed-negative)
is more disciplined than a sharper binary at CC=0.5. A binary would force
results like CC=0.42-with-tight-CI to be called "closed-negative," which
overclaims the negative — something is happening at 0.42, even if it does
not clear the positive bar. Deposit-with-no-headline-claim is the honest
shape.

## §7 — Pilot (Methodology Validation, Not Evidence)

**Purpose:** A single-conversation pilot will be run on the existing
chart.jsonl trajectories from the conversation containing this
preregistration's drafting (n=1, T ≈ 30 at draft time).

**This pilot does not constitute evidence for or against H_phase_coherence.**

The pilot answers three questions only:

1. Does the scoring code run end-to-end on real chart.jsonl data?
2. Do the two pulse-traces have compatible structure (length, alignment,
   no NaN/missing fields)?
3. Is the shuffled-pairs null model computable (requires N ≥ 2; pilot
   defers null computation to the N ≥ 5 corpus run)?

The pilot's output is not interpretable as a coherence measurement.
Reporting any pilot CC value as evidence — for or against — is a
preregistration violation.

The actual hypothesis test waits for N ≥ 5 / T ≥ 20.

## §8 — Code-Commit-Before-Run (§10.5 Mirror)

Mirroring the external-instrument-audit preregistration §10.5 pattern:

`scripts/phase_coherence_pilot.py` MUST be committed to the styxx
repository BEFORE the pilot is executed. The commit must include:

- the CC estimator (Pearson r at lag 0 on z-scored composite series)
- the DTW robustness estimator
- the windowing/alignment logic (msg_id-ordered, sender-interleaved)
- the shuffled-pairs null model implementation
- the within-agent autocorrelation secondary null
- the bootstrap CI procedure
- explicit data-loading code that reads from chart.jsonl

The commit hash of this scoring code is recorded in this document at
lock-time (§10) BEFORE any data is pulled through it.

**Without the commit-first step, the pilot does not validate anything —
it is just a measurement that happens to exist.**

## §9 — Reporting

All runs (pilot and corpus) deposit results to:
`papers/cooperative-agent-regime/results/phase_coherence_<date>_<commit>.json`

Result files include:
- input pulse_trace msg_id ranges (provenance)
- scoring-code commit hash
- preregistration document hash (this file at lock-time)
- raw CC values, null distribution, bootstrap CI
- pilot vs corpus tag (pilots are non-evidentiary)

No selective reporting. All runs against the locked scoring code are
deposited, regardless of outcome.

## §10 — Provenance

**Preregistration lock protocol:**
1. This document is reviewed by both authors.
2. Sign-off recorded as a final commit to this file with the line
   `## §10 Lock — SIGNED` appended.
3. The signing commit hash IS the preregistration lock-hash.
4. `scripts/phase_coherence_pilot.py` is committed separately, its hash
   recorded in §8 above by amendment-commit referencing the lock-hash.
5. From lock onward, this document is immutable. Methodology changes
   require a new preregistration with a new lock-hash; the old document
   remains in the repository as historical record.

**Msg_id chain (origin):**
- 34767 — styxx.attest proposal (sibling primitive, independent track)
- 34808 — styxx.pulse proposal + phase-coherence framing (initial)
- 34810 — pushback: rename resonance→phase-coherence, hypothesis-not-finding
- 34811 — shipping sequence + three locks adopted
- 34814 — schema locked, null model locked, multiple-comparison lock
- 34815 — preregistration-first ordering adopted; this draft authorized
- 34821 — pre-sign review: intermediate-zone deposit-mandate added;
  permutation threshold lifted p<0.05 → p<0.01 for chain consistency

**Lock-date:** 2026-05-20
**Lock-commit-hash:** the commit that appends the SIGNED block below is
the binding lock-hash. See `git log --follow papers/cooperative-agent-regime/phase_coherence_preregistration_2026_05_20.md`
for the exact value (self-reference paradox in inline recording is
avoided by deferring to git history, same convention as
`external_instrument_audit_preregistration_LOCKED_2026_05_20.md` at
commit `7265770`).
**Scoring-code commit hash:** `23b7912` (recorded by amendment 2026-05-20;
this commit applies the three peer-review fixes from msg_id 34850 — loader
schema renamed to `cogn_*`-prefixed field-reads, `permutation_pvalue`
bootstrapping null-medians at corpus N, per-agent chart-path architecture).
Loader was verified against `~/.styxx/chart.jsonl` on the maintainer's
machine BEFORE this amendment landed — 17 PulseSamples populated cleanly
with all four cognometric instrument scores, `needs_revision`, and
`construct_ceiling_fires` fields. §7 question 1 ("does the scoring code
run end-to-end on real chart.jsonl?") is therefore answered YES at the
methodology-validation pilot scope, before any cross-agent corpus collection.

---

## §10 Lock — SIGNED

**Signed by:** Flobi via Claude Opus 4.7 (1M context), 2026-05-20

**Sign-off statement (verbatim, from the pre-sign review):**

> *"document reads paper-grade end-to-end. all seven pushbacks landed
> plus darkflobi's own §3 single-channel multiple-comparison lock that
> i didn't ask for — that's tighter discipline than i proposed. ready
> to sign."*

**Lock decision per section:**
- §1 (REGISTER-rhythm position at load-bearing top) — accepted
- §2 (PulseSample contract, verbatim from msg_id 34814 schema) — accepted
- §3 (H_phase_coherence + composite-only primary lock) — accepted
- §4 (CC operational definition before naming) — accepted
- §5 (shuffled-pairs primary null, within-agent secondary) — accepted
- §6 (median > 0.5, CI lower > 0.3, p < 0.01 permutation, intermediate
  zone with deposit-mandate, kill-gate < 0.3) — accepted with the p < 0.01
  threshold matching the styxx chain's prior preregistrations (external-
  instrument audit at `7265770`, grounded-arc 8.0 brief)
- §7 (pilot purpose-statement: methodology validation only,
  non-evidentiary) — accepted
- §8 (code-commit-before-run with explicit content checklist) — accepted
- §9 (all runs deposited, no selective reporting) — accepted
- §10 (immutable post-lock, msg_id provenance chain) — accepted

**Post-lock binding:**
- This document is now immutable. Corrigenda may be appended below a
  horizontal rule with timestamps but do not modify §1–§10 above.
- Methodology revisions require a NEW preregistration document with a
  new filename and new lock-commit hash. This document remains in the
  repository as historical record.
- `scripts/phase_coherence_pilot.py` is committed separately per §8;
  its hash is recorded by amendment-commit referencing this lock-hash.
- The pilot per §7 may be executed after the scoring-code commit lands.
- The corpus run (N ≥ 5, T ≥ 20) may be executed after the pilot
  confirms methodology validity per §7's three questions.

**Next concrete step:** commit `scripts/phase_coherence_pilot.py` per §8.


---

## Corrigenda

**2026-05-20 02:00 EDT — §4 alignment clarification (msg_id 34850, 34861)**

§4 specifies pulse-traces are "aligned by `msg_id` ordering (sender-interleaved, not timestamp-resampled — conversations are turn-discrete)."

The operational rule used by the scoring code is: extract the composite scalar from each agent's per-agent `chart.jsonl` in `msg_id`-sorted order, then pair by ordinal position within each agent's trace (kth composite of A with kth composite of B). If trace lengths differ, truncate to the shorter.

Rationale: `chart.jsonl` is per-agent (one log per styxx-instrumented process) and does not carry a cross-agent message-pairing key. Ordinal-within-agent pairing is the implementation of "sender-interleaved, turn-discrete" alignment specified in §4.

This corrigendum clarifies — does not modify — the §4 operational definition. The §10 lock remains binding on §1–§10 above.

**2026-05-20 02:00 EDT — §2 schema field-name clarification (msg_id 34850)**

The §2 `PulseSample` dataclass specifies in-memory field names (`timestamp`, `composite`, `scores`, `needs_revision`, `construct_ceiling_fires`). The underlying `chart.jsonl` writer (`styxx/analytics.py` commit c9d847d) serializes these as `cogn_*`-prefixed names:

| PulseSample field         | chart.jsonl field                |
|---------------------------|----------------------------------|
| `timestamp`               | `ts`                             |
| `composite`               | `cogn_composite`                 |
| `scores`                  | `cogn_scores`                    |
| `needs_revision`          | `cogn_needs_revision`            |
| `construct_ceiling_fires` | `cogn_construct_ceiling_fires`   |
| `msg_id`                  | `msg_id` (set by middleware)     |

The scoring code (`scripts/phase_coherence_pilot.py`) reads the serialized names and constructs `PulseSample` objects with the dataclass names. The §2 contract is the in-memory schema; the serialization mapping is a loader-implementation detail recorded here for provenance.

This corrigendum clarifies — does not modify — the §2 input contract.
