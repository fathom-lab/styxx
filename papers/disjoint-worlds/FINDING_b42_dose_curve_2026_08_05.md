# FINDING — B42: the bridge replicates on every seed, and the barrier is rank-2 at its core

Fathom Lab · 2026-08-05 · prereg: `PREREG_b42_bridge_dose_2026_08_05.md` (frozen before the
scored run) · receipt: `b42_result.json` · scored by `styxx.protocol` from the frozen gates block.

## Verdict (machine-computed)

**`BRIDGE_REPLICATES_AND_DOSES__barrier_is_low_rank`** — all three gates pass.

| gate | frozen bar | measured | pass |
|---|---|---|---|
| G1_replicates | min bridge at k=20 across 5 seeds ≥ 0.30 | 0.9745 | ✅ |
| G2_null_clean | max matched-null at k=20 across 5 seeds ≤ 0.15 | 0.0026 | ✅ |
| G3_dose_monotone | Spearman(median bridge, k) ≥ 0.6 | 1.0 | ✅ |

B41's single-seed worry is retired: the weakest seed's bridge at k=20 is 0.9745, and the
*strongest* null anywhere at k=20 is 0.0026. The gap between the worst bridge and the best null
is not a margin — it is the whole scale.

## The dose curve (median across 5 seeds per rank)

| rank k | median bridge | median null |
|---|---|---|
| 1 | 0.1709 | 0.0 |
| 2 | 0.5128 | 0.0051 |
| 3 | 0.523 | 0.0 |
| 5 | 0.5281 | 0.0026 |
| 8 | 0.9133 | 0.0026 |
| 12 | 0.9592 | 0.0026 |
| 20 | 0.9821 | 0.0026 |
| 40 | 1.0 | 0.0026 |

**k\* = 2** under the pre-stated definition (minimum rank with median bridge ≥ 0.30 and median
null ≤ 0.15). The headline stands: *the barrier between these two minds is, at its core, two
directions wide.*

## The shape is the real finding: the barrier is hierarchical

The curve is not a smooth ramp — it is **two stages with a plateau between them**:

1. **One direction is not enough** (median 0.1709 at k=1) — but **two directions buy half of
   full legibility** (0.5128). Whatever the barrier is, it has a dominant 2-dimensional core.
2. **A plateau from k=2 to k=5** (0.5128 → 0.523 → 0.5281; essentially flat). Directions 3–5
   add nearly nothing.
3. **A second rise at k=8** (0.9133) filling in to 0.9592 at k=12, 0.9821 at k=20, and exactly
   1.0 at k=40 — a secondary band of correctable structure beyond the core.

So the honest geometry is: **a rank-2 core carrying about half the causal barrier, plus a
secondary shell of roughly 6–10 directions carrying most of the rest.** "Low-rank" was the
prereg's word; the measured refinement is *hierarchically* low-rank.

Disclosed per the prereg's non-monotonicity note: the median curve is perfectly monotone
(Spearman 1.0), but individual seeds do invert within the plateau — seed 1001 dips from 0.5714
at k=2 to 0.3954 at k=3, and seed 1002 reads 0.3214 at k=5 below its own 0.5128 at k=2. The
plateau is flat enough that seed noise reorders it; the two-stage structure itself appears in
every seed.

## What this closes

The island arc's causal chain is now complete and multi-seed:

- the island is **real** (b37 matrix), survives whitening and re-measurement (b38, b39),
- the barrier is **causal** — correcting named contrast directions in concept space turns the
  island legible while matched random frames do nothing (B41, now replicated ×5),
- the barrier is **low-rank and dose-dependent** — rank-2 core, ~rank-8–12 completion, perfect
  monotone dose (this finding),
- and the directions are **sub-symbolic** — they do not align with nameable concept categories
  (b43).

Two minds can share a concept vocabulary, agree on gross relational structure, and still be
mutually unreadable because of a causal difference two directions wide that has no name in
human language — and that difference can be measured, corrected by rank, and dosed.

## Limits, stated plainly

- One model pair (llama_3b → qwen), one concept inventory, one discovery procedure. "Two
  directions wide" is a fact about this pair's geometry, not yet a law of model populations.
- k\* uses the prereg's fixed 0.30/0.15 definition; by construction it says where legibility
  *begins*, not where it saturates (saturation is the k=8–12 shell).
- The null is a random-orthonormal frame of matched rank. It kills "any k directions would do";
  it does not test structured-but-wrong frames (a plausible successor).

*Prereg frozen before the run; verdict computed from the frozen gates block; every number above
grounds in `b42_result.json`. Sealed before commit.*
