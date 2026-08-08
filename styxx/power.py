"""styxx.power — QUARANTINED ON THE DAY IT WAS WRITTEN. DO NOT SHIP. Never released.

**Not part of any release. Not importable.** The third instrument this lab has quarantined
pre-release, after `styxx.apparatus` v1 and v2. Kept in the tree because the audit is worth more
than the module was.

VERDICT 2026-08-08: **DO NOT SHIP.** Three of six functions are numerically wrong on discrete
statistics — which is the only kind of statistic the module's own flagship case produced — and
the exam that licensed it fabricated all three of its historical cases.

## What is wrong with the module

* **`order_stat_bar` contradicts `reachable` inside the same module.** On a b48-shaped null it
  returns `corrected_bar = 0.052083` claiming alpha 0.05, and `reachable()` scores that same bar
  at **0.1165** on the same data — and at **0.9996** on the auditor's draw. Two functions, one
  module, one question, answers differing by up to 20x. Measured realized max-of-N false-positive
  rates at the bar it returns: **0.0537 at n=10, 0.2198 at n=45, 0.1814 at n=200**, against a
  target of 0.05. The construction the docstring named as "b48 died here" fails on b48's own data
  type, because `np.quantile` interpolates between attainable values of a discrete statistic.
* **`min_detectable_bar` violates its own docstring.** It promises the most permissive bar
  conceding at most alpha and does not deliver: 0.0788 at a target of 0.05 on a Poisson(1)/96
  null, 0.0775 on Poisson(3)/21. On a sparse null it returns **0.0**, a bar that concedes
  **alpha = 1.0** under `>=`. That pathology is printed in the module's own shipped receipt as
  the string `(corrected bar 0.0)`.
* **`effective_n` disagrees with this lab's own committed Bartlett numbers on all seven
  subjects**, and not by rounding: the published C5 range **6.9 to 45.8 becomes 4.84 to 53.84**.
  On AR(1) with rho=0.8 (analytic effective n 33.33) its p5-p95 over 500 replicates is
  **18.1 to 59.5**, with only 48.8% of draws within 25% of truth. The single realization shipped
  in `p1_result.json` as `effective_n_demo` reads 18.1455 — a 5th-percentile draw presented
  without an interval. **This is a live threat to a sealed finding**: on the auditor's numbers
  the median effective n moves 28.5 to 30.82, the critical correlation moves 0.3705 to 0.3561,
  and C5's strongest pair at 0.3742 crosses from below its bar to above it. Which Bartlett
  implementation is right is now an open question and is tracked as such — see the finding.
* **It contradicts `styxx.anchors`, which is exported and shipped.** `anchors._critical_count`
  walks the exact binomial survival function; `power.min_detectable_bar` interpolates an
  empirical quantile. On the same null and alternative they disagree by **4x on alpha and 0.15
  on power** (K=21, J=2, fp=0.20: anchors 0.0497/0.7706 versus power 0.2038/0.9196). A package
  that answers one question two ways has no answer.
* **It is a fourth incompatible quantile convention** in a package that already had three, and
  `styxx.calibration` already implements the correct primitive — the `ceil((n+1)(1-alpha))`
  order statistic, uninterpolated, with a finite-sample guarantee and a refusal path.

## The decisive result: it fails its OWN frozen gates on an independent battery

A third auditor built 31 cases with closed-form ground truth, none written here, and scored this
module against its own preregistration: **0.75 balanced accuracy against a frozen bar of 0.85
(G2 fails), and a 0.25 margin over a constant against a frozen 0.30 (G3 fails).** Under this
module's own outcome table that is `DO_NOT_SHIP__no_better_than_a_constant`.

The comparison that ends the argument:

* a correct six-line implementation of the same idea — **1.0**
* a three-line heuristic containing none of this module's order-statistic logic — **0.8104**
* this module — **0.75**
* a constant answer — 0.5

**Deleting the flagship correction improves accuracy.** On the `n_draws > 1` subset that is the
whole reason this module exists, it scores 0.5833 against 0.5 for a constant.

The mechanism: for a `<=` gate the docstring above promises the bar the *maximum* of n draws must
clear, and the code computes the *minimum*. On b48 — a `<=` gate on a max, the case this was
built for — the two readings differ by **6.08e20**. No argument to this module expresses the gate
it was written to catch. Also: the verdict flips on null size alone (wrong on 36.5% of resamples
at 20 draws); `effective_n` reports *no correction at all* on a series with negative lag-1 and
strong lag-2 structure, returning 5000 where the truth is 128; and the independent-draws
assumption overstates alpha by up to 18x on correlated draws, which layer-pairs from one model
run always are.

## The single worst finding, from the second audit

**On its default call path the module scores balanced accuracy 0.5000 — identical to a constant
answer, and the same number that quarantined `apparatus` v2.** `n_draws` defaults to 1 and
`n_items` defaults to None, so every check that distinguishes this module from a coin is behind
an optional argument the caller must know to supply *and supply correctly*. Supplying `n_items`
wrong is as bad as omitting it: at `n_items=10000` on a 45-trial proportion the spacing check
silently passes. This is the apparatus `floor`-omitted defect rebuilt from scratch, and the
author's own battery could not see it because that battery supplies both optionals on every case
where they matter.

Also from that audit, each measured:

* **`reachable` rejects the corrected bar `order_stat_bar` recommends, 24 times out of 24**,
  across every sample size tested up to 200,000 draws — and then recommends the same bar again.
  The advice loop does not converge. An instrument that is not self-consistent forecloses the
  question of whether it is correct.
* **Alpha is order-statistic corrected and power is not.** At `n_draws=45` the module reported
  power 0.01425 where the gate actually has 0.47579 — under-reported by a factor of 33, with the
  verdict inverted as a result.
* **`reachable` performs no validation on `n_draws` at all.** `0`, `-45` and `True` all return
  `REACHABLE`, with `detail["n_draws"]` faithfully recording `-45` as a sample count.
  `n_draws=0` is the exact typo class that produced b48.
* **`min_detectable_bar` does not validate `op`.** A trailing space silently flips the returned
  bar from +1.640822 to -1.650391 — a bar roughly 95% of the null clears — and that function is
  the one whose output an author writes straight into a preregistration.
* **Three of five public entry points return a verdict on a zero-variance null**, directly
  contradicting this docstring's own promise. `order_stat_bar` reports `bars_differ: False` on a
  point mass, which reads as "no correction needed".
* **`bounded_ceiling["chance"]` is not chance.** It returns `1/n_items`, identical to `spacing`.
  For two-alternative forced choice over 100 trials it reports 0.01 where the truth is 0.50 — a
  wrong number under a right-sounding name, in the field most likely to be copied into a prereg.
* At `alpha=1e-9` with 20 draws it returns `REACHABLE` with a reported alpha of 0.0, when the
  rule-of-three upper bound on the true rate is 0.15 — eight orders of magnitude away.

## Three of five frozen gates were satisfiable without testing what they named

This is the part that generalises beyond one module. `G1_historical_in_sample` passed without
exercising the order-statistic machinery at all, because the b48 case flags unconditionally with
`alt=None`. `G3_beats_constant` hardcoded its baseline instead of scoring one. `G4_refuses_
degenerate` scored a perfect 1.0 while the harness exercised only `reachable()` — leaving three
of five entry points untested, all three of which fail the property G4 was written to guarantee.

## What is wrong with the exam, which is worse

`papers/first-afference/run_p1.py` states its historical cases are "reconstructed from their
committed receipts." **It never opens a receipt.** The file performs no reads at all; every
historical null is an `rng` draw, and the `source` fields name files the script does not read.

* **b48-G2**: the 45 real null draws are committed in `b48_result.json["pair_null"]` and are
  sufficient for the module's own minimum. The exam invented `rng.integers(0,6)/96` instead,
  which carries **6.0x the real tail mass**. It also inverted the gate direction, so the reported
  alpha of 1.0 is the probability that at least one of 45 draws is *clean*. And with `alt=None`
  the case **cannot fail for any bar, any null, any statistics** — the gate measured nothing.
* **C5-G1**: multiplied a Beta draw by `6.9/300`, a ratio of sample sizes used as a scale on a
  fraction of pairs, producing arrays spanning 0.02 to 0.04 judged against a bar of 0.80. The
  real metric is `frac_real_coupled = 2/21`. `effective_n` — the module's headline construction,
  and the entire premise of this case — **was never called**; 6.9 was hardcoded as a scalar.
* **b37-G2**: the defect description is **the exact inverse of what happened.** The real b37-G2
  was a `> 0.0` floor that this lab's own finding calls "a noise-passable gate, and I wrote it" —
  too *permissive*, and it **PASSED**. The exam reconstructs it as a bar of 0.90, too demanding,
  and labels it "bar demanded an effect the apparatus cannot produce." That error was carried
  into the frozen preregistration, which is now permanently wrong on the public record.
* **The perfect score was hand-tuned.** `run_p1.py` floors its "REACHABLE" control bars with
  `max(corr, 0.0625)`, raising them past the module's own correction so the ground-truth label
  holds. Without that floor, specificity falls 1.0 to 0.75 and balanced accuracy 1.0 to 0.875.
  The undisclosed floor conceals precisely the central bug.
* **`G3_beats_constant` measured nothing.** The baseline is hardcoded to 0.5 rather than scored,
  making the gate algebraically identical to `G2 >= 0.80`. The apparatus guard it was imitating
  worked *because a constant was actually run*.

## The conclusion worth keeping

The prereg named this branch in advance: a third quarantined instrument would itself be a
finding. It is. The lesson is not that this implementation was sloppy — it is that **an author's
own exam cannot license their own instrument**, and that a battery which generates its own inputs
will encode the author's misconceptions on both sides of the test. The three historical cases
were the only part of P1 with ground truth outside my own head, and all three were fabricated
while being labelled as receipts. The perfect 1.0 was not evidence; it was the sound of a test
agreeing with itself.

Implementation retained at ``power_QUARANTINED.py.txt``. Full audit in
``papers/first-afference/FINDING_p1_third_quarantine_2026_08_08.md``.
"""

raise ImportError(
    "styxx.power is QUARANTINED and was never released. order_stat_bar contradicts reachable by "
    "up to 20x on the discrete nulls it was built for, min_detectable_bar can return a bar "
    "conceding alpha=1.0, effective_n disagrees with this lab's own committed Bartlett numbers on "
    "all seven subjects, and it contradicts the shipped styxx.anchors by 4x on alpha. Its exam "
    "fabricated all three historical cases while claiming to read them from receipts. See the "
    "module docstring."
)
