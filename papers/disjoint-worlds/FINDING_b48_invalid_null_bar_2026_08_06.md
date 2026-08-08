# FINDING — B48: INVALID, and the fault is our own gate — a max-of-45 statistic judged against a single-draw bar

Fathom Lab · 2026-08-06 · prereg: `PREREG_b48_legibility_matrix_ten_2026_08_06.md` (frozen
before the run) · receipt: `b48_result.json` · scored by `styxx.protocol`.

> **SUPERSEDED 2026-08-08 by B50** — `FINDING_b50_no_legibility_islands_2026_08_08.md`. B50 redrew
> all 45 nulls at a different seed and got a null family **identical to four decimal places on
> both statistics** (median 0.0104, max 0.0521). The nulls below never leaked; the bar that judged
> them was wrong, exactly as this document says. On a correctly specified bar the same data reads
> `NO_LEGIBILITY_ISLANDS__the_first_island_does_not_generalize`. This finding and its receipt stay
> in the tree unedited — a retraction that deletes its own evidence is not a retraction.

## Verdict (machine-computed)

**`INVALID__null_leaks`** — the run licenses nothing about islands, and the reason is a
preregistration design error we are recording rather than repairing after the fact.

| gate | frozen bar | measured | pass |
|---|---|---|---|
| G0_coverage | ≥ 45 pairs | 45 | ✅ |
| G1_signal_present | max pair legibility ≥ 0.0521 | 0.2396 | ✅ |
| G2_null_clean | **max** null legibility ≤ 0.0208 | 0.0521 | ❌ |
| G3_legibility_islands | gap-screen p ≤ 0.05 | (moot) | ❌ |

## What actually went wrong

The nulls did not leak. They behaved almost exactly as chance demands: **median 0.0104 —
identical to the chance floor of 0.0104 — and mean 0.0118**, with only 5 of 45 draws above the bar.
The failure is that G2 gated the **maximum of 45 draws** against a bar set for a *single* draw
(2× chance). Legibility here is a discrete statistic in steps of 1/96; a single null landing on
5/96 = 0.0521 across forty-five independent draws is an ordinary outcome under the null, not
evidence of leakage. We wrote a multiple-comparisons error into our own gate and the machinery
correctly refused the run.

This is the same class of mistake as b37's G2 — a bar specified without reference to the
distribution of the statistic it judges — committed again, four days later, by the same lab that
published the first one. That is worth stating plainly: naming a failure mode does not immunize
you against it.

## What the run showed anyway, and why none of it counts

Reported because the prereg said the numbers ship regardless of verdict, and **flagged as
uninterpretable** because the verdict is INVALID:

- **There is real discovery signal on this battery** (G1 passed): the strongest pair read 0.2396,
  twenty-three times chance. The pilot's pessimism about a 96-item battery was too gloomy.
- **Per-member mean legibility spans 0.037 to 0.0914** — a smooth spread with no bimodality, the
  same shape B47 found in frame affinity.
- **The direction-blindness is near-total: 42 of 45 pairs returned exactly equal scores in both
  directions.** Our "both directions" design bought essentially nothing, because the committed
  discovery machinery cannot see direction — precisely the limitation the b37 finding impeached
  in its own author's gates. A design that spends half its compute on a distinction its
  instrument cannot make is a design flaw, and it is ours.
- The registered ungated cliff check fit a threshold better than a line (0.1141 vs 0.0656, a
  difference of 0.0485, best threshold near affinity 0.7524) — but **both fits are weak**, and on
  an INVALID run this is a scatter plot, not a result. It is recorded so that a later, valid run
  cannot be presented as having predicted it.

## What happens next, and what must not

The tempting move is to correct the bar, re-score the existing numbers, and report the verdict
that comes out. **That is gate-shopping and it is refused.** The nulls have now been seen; a bar
chosen after seeing them is not a preregistration.

The successor (B48-v2) will freeze a correctly specified null criterion *before* running —
judging the null distribution rather than its maximum (a false-discovery-rate bound, or a bar
calibrated for max-of-N) — and must be scored on **freshly drawn nulls under a different seed**,
with this document cited as the reason the design changed. The legibility matrix itself may be
reused; the null draws may not.

## Limits

Ten models, one 96-concept battery, one extraction, k = 60, single seed, and an INVALID verdict
that makes every number above a lead rather than a finding. The question B48 was built to answer
— whether legibility-defined islands recur — remains open, and this run did not move it.

*The prereg was frozen before the run, the gate it contained was wrong, the machinery refused
the run because of it, and the refusal is published with the error named. Every number grounds
in `b48_result.json`. Sealed before commit.*
