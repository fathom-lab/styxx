# FINDING — B46: the cliff is mapped — legibility demands near-exact frame agreement

Fathom Lab · 2026-08-06 · prereg: `PREREG_b46_cliff_function_2026_08_06.md` (frozen at
`060f8b3` before the scored run) · receipt: `b46_result.json` · scored by `styxx.protocol`.

## Verdict (machine-computed)

**`CLIFF_MAPPED__legibility_rises_monotonically_with_dose`** — all three gates pass.

| gate | frozen bar | measured | pass |
|---|---|---|---|
| G0_baseline_low | max disc at t=0 ≤ 0.15 | 0.0612 | ✅ |
| G1_bridge_high | min disc at t=1 ≥ 0.30 | 0.9745 | ✅ |
| G2_monotone | Spearman(median disc, t) ≥ 0.6 | 1.0 | ✅ |

## The cliff function (median across three seeds)

| dose t | median discovery |
|---|---|
| 0.0 | 0.0408 |
| 0.2 | 0.0434 |
| 0.4 | 0.1122 |
| 0.6 | 0.3622 |
| 0.8 | 0.9566 |
| 1.0 | 0.9821 |

Knee at **t½ = 0.8** (first dose crossing half the t=1 median); transition width
(quarter-to-three-quarter) **0.2**. Sixty percent of the way to the reader's frame buys about
a third of full legibility; the step from t=0.6 to t=0.8 buys most of the rest at once.

## What the shape settles

B45 left a paradox: the island's frame shares roughly seventy-two percent of its
squared-cosine mass with the clique, yet discovery reads it near zero. B46 supplies the
mechanism: **the legibility function is flat almost everywhere and nearly vertical close to
alignment.** An island can be "mostly aligned" and still sit far below the knee — which is
exactly where qwen sits. Three consequences, each now measured rather than argued:

1. **Slope measures cannot predict cliff phenomena.** RSA and frame affinity vary smoothly;
   readability switches. The arc-long failure of similarity metrics to predict legibility is
   a geometric necessity, not a puzzle.
2. **Partial frame correction is nearly worthless; near-exact correction is nearly perfect.**
   Practical translation between representationally divergent models is all-or-nothing in
   the frame coordinate.
3. **Small divergences hide big barriers.** Two training runs need only drift a modest,
   consistent rotation apart — invisible to gross similarity — to lose mutual readability.

Disclosed, per the prereg's shape-honesty note: individual seeds are noisy inside the
transition zone (seed 343 reads 0.3214 at t=0.2 then 0.0281 at t=0.4; seed 1002 dips to
0.0944 at t=0.6), which is expected where a near-vertical function meets seed-level frame
variation. The medians are perfectly monotone (Spearman 1.0); the per-seed grid ships in the
receipt.

## Limits

One island, one reader, one blend path (QR of the linear frame interpolation; a geodesic
blend could shift the knee's numeric location, though not the flat-then-vertical shape that
three seeds reproduce). Knee and width were reported ungated per the prereg — they are
measurements, not passed bars — and carry no replication claim beyond these seeds.

*Prereg frozen before the run; every number grounds in `b46_result.json`; sealed before
commit. The island arc now ends with a measured function: real, causal, rank-2 at core,
below language, shared-frame-relative — and switch-like in the frame coordinate.*
