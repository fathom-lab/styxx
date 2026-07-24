# FINDING -- the buried-judge boundary: the ladder prices where its interval stays covered, and refuses below the gate

**Cycle 61 (autopilot, 2026-07-24). Prereg `PREREG_buried_judge_2026_07_24.md` (commit `4be0433`),
frozen before the scored run. Verdict: `SURVIVED__prices_where_covered_refuses_below_gate`.
Receipt: `buried_judge_result.json`. Simulation on the sealed Stage-A DGP; shipped instrument
`styxx.anchors.audit_panel`.**

## What was asked

Cycle 60 closed the four-family PRICE/REFUSE partition (attr + numeric price; chain + temporal
refuse) and named the sharpening step: a genuinely-informative-but-hard family, a real judge buried
under noise, to locate where PRICE turns into REFUSE. Every prior family sat cleanly on one side;
this one is built to sit on the boundary. A single informative judge (alpha 0.30, beta 0.30 + sep)
sits among three deaf judges, with honest exchangeable anchors, and its true separation is swept
from deaf (0.00) through the effective noise-margin gate (~0.255 at K=400) to strong (0.40), at
R=60 replicates per cell.

## The result: an honest transition, no over-pricing

The instrument's verdict moves with the signal, and the interval stays covered wherever it prices:

| sep | ESTIMATED rate | VOID rate | pi-CI coverage (covered / estimated) |
|-----|----------------|-----------|--------------------------------------|
| 0.00 | 0.0 | 1.0 | -- (deaf, all refuse) |
| 0.16 | 0.017 | 0.983 | 1/1 |
| 0.22 | 0.133 | 0.867 | 8/8 |
| 0.25 | 0.517 | 0.483 | 28/31 (coverage 0.903) |
| 0.28 | 0.817 | 0.183 | 47/49 (coverage 0.959) |
| 0.34 | 1.0 | 0.0 | 58/60 (coverage 0.967) |
| 0.40 | 1.0 | 0.0 | 59/60 (coverage 0.983) |

**PD_MOVE passed** (deaf VOID rate 1.0; strong ESTIMATED rate 1.0 -- the output reads the signal).
**PD_DANGER passed**: every cell the ladder prices (ESTIMATED rate >= 0.50: sep 0.25, 0.28, 0.34,
0.40) keeps pi-CI coverage of the true 0.35 at or above 0.903 -- comfortably above the frozen 0.80
floor (the instrument's own worst measured regime). **PD_HONEST passed**: no refusal replicate
leaked a pi. **PD_BOUNDARY** (reported, not gated): the smallest buried-but-recoverable separation
is 0.25 -- coverage clears 0.90 at the very cell where pricing first crosses 50 percent.

## Why this is not a free pass -- the kill was live

The sep 0.25 cell sits at the effective gate: 31/60 replicates cleared the noise-margin keep
(ESTIMATED rate 0.517) and 29/60 refused -- a near-even split driven by the anchor draw. That
split is exactly the over-pricing danger the prereg named: the keep decision selects on the anchor
draw (kept only when beta_hat - alpha_hat is large), which biases the point estimate; at sep 0.22
the priced replicates' pi_median is 0.4274, off the true 0.35, showing the selection bias is real
and not small. PD_DANGER could therefore have fired. It did not, because the selective bootstrap
widens the near-gate interval enough to price that selection uncertainty: coverage at the split
cell is 28/31, and the bias visibly shrinks back toward truth as separation grows (pi_median 0.4274
-> 0.3462 -> 0.354 -> 0.3443 -> 0.3477 across sep 0.22..0.40).

## What this sharpens (and does not claim)

The price/refuse boundary is **well-placed and honest at the boundary**: where an informative judge
is buried but detectable the ladder prices it with a covered interval; where it drops below the
gate the ladder refuses (VOID) rather than emitting a confident wrong number. This closes the shape
of the partition -- attr/numeric price, chain/temporal refuse, and now the boundary between them is
characterized rather than assumed. Scope is unchanged and narrow: simulation on the sealed DGP,
honest exchangeable anchors, a single informative judge; it says nothing about correlated
multi-judge panels, non-exchangeable anchors, or real LLM judges (Stage B). The instrument surface
and its bars are untouched; `styxx/certify.py` was not modified.
