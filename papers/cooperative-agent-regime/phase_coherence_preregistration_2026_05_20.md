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

**Lock-date:** TBD
**Lock-commit-hash:** TBD
**Scoring-code commit hash:** TBD (recorded at §8 by amendment)

---

## §10 Lock — UNSIGNED (draft)

*This document is not yet locked. Methodology may be revised until the
SIGNED line is appended.*
