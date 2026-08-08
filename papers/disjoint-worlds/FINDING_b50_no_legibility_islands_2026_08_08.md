# FINDING — B50: legibility does not partition ten models into islands, and B48's "null leak" was our bar, not our null

Fathom Lab · 2026-08-08 · prereg: `PREREG_b50_legibility_islands_2026_08_08.md` (frozen before
the fresh nulls were drawn) · receipt: `b50_result.json` · scored by `styxx.protocol`.

## Verdict (machine-computed)

**`NO_LEGIBILITY_ISLANDS__the_first_island_does_not_generalize`**

| gate | frozen bar | measured | pass |
|---|---|---|---|
| G0_coverage | ≥ 45 pairs | 45 | ✅ |
| G1_signal_present | max pair legibility ≥ 0.0521 | 0.2396 | ✅ |
| G2_null_at_chance | median null ≤ 0.0208 | 0.0104 | ✅ |
| G3_null_tail_bounded | fraction of nulls ≥ 5× chance ≤ 0.10 | 0.0444 (2 of 45) | ✅ |
| G4_islands | bimodality p ≤ 0.05 | 0.1774 | ❌ |

Three of four gates pass. The battery carries real discovery signal — the strongest pair reads at
0.2396, **23.0385× the 0.0104 chance rate** — and the nulls sit exactly on chance. What fails is the
island claim: ten per-member mean legibilities descend smoothly from 0.0914 to 0.0370, a range of
0.0544 whose largest internal gap is 0.0215. A ramp, not a cliff.

## The retraction this run was built to settle

B48 asked the same question on 2026-08-06 and returned `INVALID__null_leaks`. **That verdict was
about our arithmetic, not our data.** B48's G2 judged the **maximum of 45 null draws** against a
bar written for a single draw. With chance at 0.0104, one draw landing on 0.0521 is ordinary luck
across 45 tries; the machinery dutifully called it a leak.

B50 re-drew all 45 nulls at seed 8080 — a different family from b48's 343, because a criterion
chosen after seeing a null distribution is not a preregistration. The two families:

| | b48 (seed 343) | B50 (seed 8080) |
|---|---|---|
| median null | 0.0104 | 0.0104 |
| max null | 0.0521 | 0.0521 |

**Identical to four decimal places on both statistics.** The maximum reaching 5× chance is not a
leak that recurred; it is a stable property of the largest of 45 draws from a discrete
distribution whose floor is the 0.0104 chance rate. Two independent families agreeing this precisely is the strongest
available evidence that b48's nulls were always clean. **B48's `INVALID__null_leaks` is hereby
superseded: the correct reading of that data is the verdict above.** The receipt and the finding
stay in the tree — a retraction that deletes its own evidence is not a retraction.

## What this costs the island programme

The island arc's nine acts established affinity-defined islands in a ten-model cohort. B47 found
no affinity islands under a stricter screen; B50 now finds no *legibility*-defined ones either.
The first island does not generalize to this cohort under either definition.

**Stated in advance, and it binds us here:** at n = 10 the gap screen has little power against a
lone island, so `NO_LEGIBILITY_ISLANDS` is **weak evidence of absence**. We are not entitled to
say legibility islands do not exist. We are entitled to say this cohort does not show them, and
that the honest next step is a cohort large enough for the screen to have teeth — the same
conclusion the C-series reached from the other direction.

## What is new about how this was run

B50 is the first preregistration in this program written entirely under the v2/v3 protocol
machinery. Every gate declares a `power_basis` — the evidence that its bar is reachable — and a
`metric_means` sentence naming what its metric is. The runner resolves all five metric paths
against the result **before** scoring and refuses if any is absent or non-comparable.

G2 and G3 exist as a pair for a specific reason. B48's error was gating a maximum against a
single-draw bar, so B50 gates the **median** (stable under a lucky draw) and adds an **explicit
tail bound**, permitting 4 such draws, to catch what a median would hide. That decomposition is the
direct machinery-level lesson of the b48 failure, and it is the first bar in this program derived
from the distribution it judges rather than from a round number.

## Limits

Ten models, one 96-concept battery, one transfer construction. The legibility matrix is reused
from b48 unchanged — only the nulls are fresh, so this is not an independent replication of the
matrix itself. G4's non-detection is weak by the power argument above, and it is the gate the
finding turns on.

*Frozen before the nulls were drawn; the losing branch named in advance; a prior verdict of our
own overturned in public. Every number grounds in `b50_result.json`, with the three
derived quantities recomputed from it by `build_b50_derived.py` into `b50_derived.json`. Sealed before commit.*
