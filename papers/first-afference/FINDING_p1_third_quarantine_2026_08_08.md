# FINDING — P1: the third quarantine, and the reason it is a finding rather than an embarrassment

Fathom Lab · 2026-08-08 · prereg: `PREREG_p1_power_refusal_2026_08_08.md` (frozen before the
module existed) · receipts: `p1_result.json`, `sense_collector_audit.json` · module quarantined
at `styxx/power.py`, implementation retained at `styxx/power_QUARANTINED.py.txt`.

## Verdict

**`DO_NOT_SHIP`** — `styxx.power` is quarantined and was never released. The
`PROCEED_TO_RED_TEAM__not_yet_shippable` verdict its own exam computed is **unsupported**, and
`p1_result.json` is annotated as such in place.

The preregistration named this branch in advance: *"a third quarantined instrument would itself
be a finding — it would say that this lab's failure classes are diagnosable in retrospect and
not detectable in advance, which is a real and publishable limit on the whole preregistration
programme."* That is what happened, and this document is the finding.

## What the module got wrong

The module claimed to decide whether a preregistered bar is reachable. Three of its six functions
are wrong on discrete statistics — the only kind its own flagship case produces.

| defect | measured |
|---|---|
| `order_stat_bar` returns a bar `reachable` then rejects | claims α 0.05, scored at 0.1165 here and 0.9996 by the auditor |
| realized false-positive rate at that "corrected" bar | 0.0537 (n=10), 0.2198 (n=45), 0.1814 (n=200) against 0.05 |
| `min_detectable_bar` exceeds its promised α | 0.0788 on a Poisson(1)/96 null at a target of 0.05 |
| the same function on a sparse null | returns 0.0, a bar conceding **α = 1.0** |
| `effective_n` on AR(1) ρ=0.8, analytic truth 33.33 | p5–p95 of **18.1 to 59.5**, 48.8% of draws within 25% |
| disagreement with the shipped `styxx.anchors` | **4× on α, 0.15 on power** for the same question |

The root cause of the first three is one line of carelessness with real consequences:
`np.quantile` interpolates between attainable values, and a proportion over 96 trials has no
values between them. `styxx.calibration` already implements the correct primitive — the
`ceil((n+1)(1-α))` order statistic, uninterpolated, with a refusal path. I wrote a fourth
quantile convention into a package that already had three.

## The decisive result: the module fails its own frozen gates on someone else's battery

A third adversary built 31 cases the author never saw, every ground truth a closed-form
probability cross-checked by Monte Carlo, and scored `styxx.power` against **its own
preregistration**:

| predictor | balanced accuracy |
|---|---|
| a correct six-line implementation of the same idea | **1.0** |
| a three-line heuristic with no order-statistic logic at all | **0.8104** |
| `styxx.power.reachable` | **0.75** |
| a constant answer | 0.5 |

G2 demanded 0.85 and got 0.75. G3 demanded a 0.30 margin over the best constant and got 0.25.
**Under its own frozen outcome table that is `DO_NOT_SHIP__no_better_than_a_constant`** — a
branch this lab wrote before the module existed, reached by an adversary two days later.

Read the table again in the order that matters: **deleting the module's flagship correction makes
it more accurate.** On the `n_draws > 1` subset that is the module's entire reason to exist, it
scores 0.5833 against 0.5 for a constant. The contribution is negative.

The mechanism, and I flagged it as a suspicion when commissioning the audit: for a `<=` gate the
docstring promises the bar the *maximum* of n draws must clear, and the code computes the
*minimum*. On b48 — the case the module was built for, which is a `<=` gate on a max — the two
readings differ by a factor of **6.08e20**. There is no argument through which the module can
express the gate it was written to catch.

Three further defects the first two audits missed: the verdict flips on null size alone (wrong on
**36.5%** of resamples at 20 draws, where `MIN_DRAWS` licenses it); `effective_n` silently reports
*no correction whatsoever* on a series with negative lag-1 and strong lag-2 autocorrelation,
returning 5000 where the truth is 128, a **39×** overstatement with no flag; and the
independent-draws assumption overstates alpha by up to 18× when the draws are correlated, which
45 layer-pairs from one model run always are.

## The worst single number

**On its default call path the module scores balanced accuracy 0.5000** — identical to a constant
answer, and the same figure that quarantined `apparatus` v2 at 0.5625. `n_draws` defaults to 1 and
`n_items` defaults to `None`, so every check separating this module from a coin sits behind an
optional argument the caller must know to supply and supply correctly. At `n_items=10000` on a
45-trial proportion, the spacing check silently passes. That is the apparatus `floor`-omitted
defect rebuilt from scratch — and **the author's own battery could not see it**, because the
battery supplies both optionals on every case where they matter.

| further defect | measured |
|---|---|
| `reachable` rejects the bar `order_stat_bar` recommends | 24 of 24 trials, up to 200,000 draws; the advice loop never converges |
| alpha is order-statistic corrected, power is not | power reported 0.01425 where the gate has 0.47579 — off by 33×, verdict inverted |
| `reachable` validates `n_draws` not at all | `0`, `-45`, `True` all return REACHABLE; `detail` records −45 as a sample count |
| `min_detectable_bar` does not validate `op` | a trailing space flips the returned bar from +1.640822 to −1.650391 |
| entry points returning a verdict on a zero-variance null | 3 of 5, contradicting the module's own docstring |
| `bounded_ceiling["chance"]` for 2AFC over 100 trials | reports 0.01; the truth is 0.50 |

## Three of five frozen gates were satisfiable without testing what they named

This is the part that generalises past one module, and it is the most useful thing P1 produced.

- **`G1_historical_in_sample`** passed without exercising the order-statistic machinery at all —
  the b48 case flags unconditionally because `alt=None`.
- **`G3_beats_constant`** hardcoded its baseline at 0.5 instead of scoring one.
- **`G4_refuses_degenerate`** scored a perfect 1.0 while the harness exercised only
  `reachable()`, leaving three of five entry points untested — all three of which fail the exact
  property G4 was written to guarantee.

A gate can be frozen, machine-scored, and still measure nothing, if the harness that feeds it
only visits the paths the author already trusted. `power_basis` and `metric_means` do not catch
this: both gates *had* them. **The missing artifact is a declaration of what the harness must
exercise**, checkable against what it did exercise. That is a concrete, buildable successor and
it is the one thing here worth building next.

## What the exam got wrong, which matters more

`run_p1.py` states that its historical cases are *"reconstructed from their committed receipts."*
**It never opens a receipt.** The script performs no file reads; every historical null is an
`rng` draw and every `source` field names a file the code does not touch.

- **b48-G2.** The 45 real null draws sit in `b48_result.json["pair_null"]`, already committed and
  already sufficient for the module's own minimum draw count. The exam invented
  `rng.integers(0,6)/96` instead, carrying **6× the real tail mass**. It also inverted the gate
  direction, so its reported α of 1.0 is the probability that at least one of 45 draws is
  *clean*. And with `alt=None` the case **could not fail for any bar, any null, or any
  statistics.** A gate that cannot fail measures nothing.
- **C5-G1.** Multiplied a Beta draw by `6.9/300` — a ratio of sample sizes applied as a scale to
  a fraction of pairs — producing values spanning 0.02 to 0.04 judged against a bar of 0.80. The
  real metric is `frac_real_coupled = 2/21`. `effective_n`, the module's headline construction
  and this case's entire premise, **was never called**; 6.9 was hardcoded.
- **b37-G2.** The defect description is **the exact inverse of the truth.** The real b37-G2 was a
  `> 0.0` floor that our own finding calls *"a noise-passable gate, and I wrote it"* — too
  permissive, and it **passed**. P1 reconstructed it as a bar of 0.90 and called it a bar that
  "demanded an effect the apparatus cannot produce."
- **The perfect score was hand-tuned.** The REACHABLE control bars are floored with
  `max(corr, 0.0625)`, lifting them past the module's own correction so the ground-truth labels
  hold. Remove the floor and specificity falls from 1.0 to 0.75, balanced accuracy from 1.0 to
  0.875. The floor concealed exactly the central bug.
- **`G3_beats_constant` scored nothing.** Its baseline is the literal 0.5, not a measurement,
  making the gate algebraically `G2 ≥ 0.80`. The `styxx.apparatus` guard it imitated worked
  *because a constant was actually run against the same battery.*

## The error that is now permanent

The b37 inversion was carried into `PREREG_p1_power_refusal_2026_08_08.md`, which is frozen and
committed. **It is wrong on the public record and it stays wrong**, because a preregistration
that gets edited after the fact is not one. The correction lives here and in the cycle log. This
is the cost of the freeze discipline and it is the discipline working, not failing: the error is
visible precisely because the document could not be quietly repaired.

## An open question this raised about a sealed finding

`styxx.power.effective_n` and the committed `c5_effective_df_addendum.json` disagree on **all
seven subjects** — the published C5 range of 6.9 to 45.8 becomes 4.84 to 53.84 under the new
implementation. They use different autocorrelation estimators (a biased ACF versus the Pearson
correlation of lagged pairs).

**We do not yet know which is correct, and the difference is load-bearing.** On the auditor's
numbers the median effective n moves from 28.5 to 30.82, the implied critical correlation moves
from 0.3705 to 0.3561, and C5's strongest pair at 0.3742 crosses from *below* its bar to *above*
it. C5's verdict was a null, and this would not overturn it into a positive — but the sentence
that carries its argument would reverse.

This is **recorded as an open defect against a sealed finding**, not resolved here. Resolving it
requires an independent third implementation checked against an analytic case, which is exactly
the kind of work that must not be done by the person who wants a particular answer. Until then
C5's effective-n range should be read as implementation-dependent. `FINDING_c5` is annotated
with a pointer to this section.

## The general lesson, stated as narrowly as the evidence allows

Three instruments, three quarantines, one shared mechanism: **an author's own battery cannot
license their own instrument, because a battery that generates its own inputs encodes the
author's misconceptions on both sides of the test.** In P1 the only component with ground truth
outside my own head was the set of three historical cases, and all three were fabricated while
being labelled as receipts. The 1.0 was not evidence. It was a test agreeing with itself.

What survives is procedural and it is not nothing: the module never reached a user, the release
path requires an explicit tag that was never cut, and the adversarial pass that caught this ran
*before* announcement rather than after. The pre-release rule now reads three modules shipped
before it existed (all three broken within hours, two recalled) against **three stopped by it
with zero users exposed**.

*Frozen before the module existed; the losing branch named in advance and taken; the author's own
exam discredited by an adversary and the discrediting published at the same volume as the build.*
