# FINDING — B45: the shared frame is visible to the naked eye of geometry — and the island is a modest, perfectly consistent rotation

Fathom Lab · 2026-08-06 · prereg: `PREREG_b45_frame_geometry_2026_08_06.md` (frozen at
`22ad3e2` before the scored run) · receipt: `b45_result.json` · scored by `styxx.protocol`.

## Verdict (machine-computed)

**`SHARED_FRAME_CONFIRMED_GEOMETRICALLY__island_rotated_away`** — both gates pass, with no
discovery machinery, no fitting, and about four seconds of compute.

| gate | frozen bar | measured | pass |
|---|---|---|---|
| G1_clique_frames_shared | median clique affinity − null 95th pct ≥ 0.0 | 0.7914 | ✅ |
| G2_island_separated_all_seeds | qwen below clique in ≥ 5 of 5 seeds | 5 | ✅ |

## The numbers

Affinity = mean squared cosine of principal angles between label-aligned concept-Gram
eigenframes (the committed b40 statistic). A Haar-random frame pair at this size scores
0.051 in expectation, 0.0566 at the 95th percentile.

| pairing | k=20 median | k=2 median |
|---|---|---|
| clique ↔ clique | 0.848 | 0.9257 |
| qwen ↔ clique | 0.7166 | 0.8167 |
| random null (95th pct) | 0.0566 | 0.0117 |

Two facts, and both carry:

1. **Every frame — the island's included — sits an order of magnitude above random.** The
   clique's frames share about eighty-five percent of their squared-cosine mass; qwen's
   frames share about seventy-two percent with the clique. The shared cross-family geometry
   B44 inferred through surgery is directly visible as subspace alignment, and it includes
   the island. Convergence is the rule for all four models.
2. **The island's deficit is modest and perfectly consistent.** Qwen's affinity to the clique
   sits below the clique's self-affinity in **every seed, at both ranks** — a stable gap of
   roughly 0.13 in squared-cosine mass at k=20. The island is not in another galaxy; it is a
   reliable, repeatable rotation a measured distance away from a frame everyone else shares.
   Even at the rank-2 core (reported ungated), the same ordering holds in all five seeds
   (0.9257 vs 0.8167).

## What this closes, and what it sharpens

The B44 story survives with its instrument removed: the clique shares a frame; the island is
rotated away from it. But B45 sharpens the picture in a way B44 could not — through the
discovery lens the island looked *unreadable* (near-zero legibility until corrected), while
in raw frame geometry it is **mostly aligned**. A modest, consistent rotation is enough to
collapse legibility from near-perfect to near-zero. Discovery is a cliff; geometry is a
slope. That nonlinearity — small frame deviation, catastrophic legibility loss — is the
quantitative shape of the barrier, and it is why RSA-style similarity (a slope measure)
never predicted readability (a cliff phenomenon) anywhere in this arc.

This is also the arc's cheapest receipt: any replicator can verify the clique/island frame
structure from the committed banks in under a minute, CPU-only, with no discovery fits.

## Limits

Four models, one concept inventory, one frame construction (top-k concept-Gram eigenvectors
on the b42 training rows). "Rotated away" is measured as affinity deficit; the specific
rotation is not characterized here (b43 says its concept content is not nameable). Whether
other islands, if found, show the same modest-rotation/catastrophic-legibility signature is
the open recurrence question.

*Prereg frozen before the run; both gate shapes chosen against named prior failures (b37-G2's
noise-passable floor repaired as a null-exceedance; the ordering claim gated on all-seeds sign
consistency). Every number grounds in `b45_result.json`. Sealed before commit.*
