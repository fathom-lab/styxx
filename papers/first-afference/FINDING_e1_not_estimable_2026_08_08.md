# FINDING — E1: the effective sample size is not estimable at this series length, C5's conclusion survives and C5's number does not

Fathom Lab · 2026-08-08 · prereg: `PREREG_e1_effective_n_bakeoff_2026_08_08.md` (frozen before
the bake-off ran) · receipt: `e1_result.json` · scored by `styxx.protocol`.

## Verdict, and an immediate correction to it

The machinery returned **`RESOLVED__winner_selected_and_c5_recomputed`**. **I do not think that
verdict is right, and the reason is a defect in a gate I wrote.**

`G1_a_candidate_is_usable` asked whether the best median error over the candidates was at most
0.20. It resolves to **0.1436** and passes. But that minimum belongs to `ar1_closed_form`, which
`G2` **disqualified** in the same run for failing silently. The candidate that actually won has a
pooled median error of **0.2554** — it would have failed G1 outright. Written over *eligible*
candidates, G1 fails and the frozen outcome table returns
**`NOT_ESTIMABLE__c5_effective_n_range_is_unusable_at_this_length`**.

**I take the NOT_ESTIMABLE reading as operative.** The prereg is frozen and stays as written; the
gate is faithfully implemented and its `metric_means` honestly says "over candidates", so nothing
here is a lie. It is simply the wrong quantity. **This is the sixth bar mis-specification in a
week and the most instructive**, because it is the first where the gate, the metric path, the
power basis and the implementation were all individually correct and the *composition* was wrong:
G1 and G2 were written as independent gates when G1's population depends on G2's outcome. No
check in `styxx.protocol` looks at relationships between gates. That is now the top of the
backlog.

## What was measured

Five estimators against analytic AR(1) truth, 400 replicates per cell, n=300 to match C5.

| ρ | analytic n_eff | truncate (biased) | truncate (Pearson) | AR(1) closed form | Bartlett √n | initial positive seq |
|---|---|---|---|---|---|---|
| 0.50 | 100.00 | 101.5 | 101.1 | 102.2 | 115.1 | 100.2 |
| 0.70 | 52.94 | 56.1 | 55.8 | 55.0 | 68.0 | 55.7 |
| 0.80 | 33.33 | 38.0 | 37.6 | 35.8 | 49.2 | 38.0 |
| 0.90 | 15.79 | 20.1 | 19.7 | 17.8 | 30.9 | 20.0 |
| 0.95 | 7.69 | 11.6 | 11.1 | 9.9 | 24.3 | 11.7 |

**Every estimator overstates the effective sample size at every autocorrelation on the grid.**
That direction matters more than the ranking: overstating n_eff *understates* the correlation
needed for significance, so the entire family of estimators is anti-conservative, and the error
grows exactly where BOLD lives. At ρ=0.95 the best eligible estimator reports 11.1 against a
truth of 7.69.

`ar1_closed_form` scored best on accuracy (0.1436) and was disqualified anyway: on a process with
negative lag-1 and strong lag-2 structure it reports an effective n *above* nominal. An estimator
whose failure is silent is unsafe at any bias, which is what G2 exists to say.

## The answer to the question that started this

C5's argument rests on a critical correlation implied by the median effective n, and its
strongest pair sits within 0.0004 of it. Three defensible numbers:

| basis | median n_eff | threshold r | does the strongest pair (0.3742) clear it? |
|---|---|---|---|
| the committed C5 addendum | 28.5 | 0.3746 | **no** — by 0.0004 |
| the winning estimator, raw | 30.4799 | 0.3581 | **yes** |
| the winner corrected for its own measured bias | 24.279 | 0.4020 | **no** |

**C5's conclusion survives; C5's number does not.** Under the published figure and under the
bias-corrected one, the strongest pair fails to reach significance — which is what C5 claimed.
Under the raw winner it clears. Since every estimator on the grid is biased upward, and the
bias-corrected reading is the one consistent with that measurement, the null stands. But a
conclusion that flips on which of three defensible estimators you pick is **not** a conclusion
that should have been stated to four decimal places.

The C5 finding is annotated with a pointer here. Its verdict was a null and remains one. The
sentence *"The strongest pair, at 0.3742, sits essentially on it"* is now known to be
estimator-dependent, and "essentially on it" turns out to have been more accurate than the
precision around it implied.

## A defect in an existing receipt

`c5_effective_df_addendum.json` has **no generator script**, and this run could not identify what
produced it. The closest of five standard candidates differs from its per-subject values by up to
**7.4805** — far too large to call a match. The committed range 6.9 to 45.8 therefore has
**unidentifiable provenance**: it is a number in the tree that nobody can currently reproduce.
That is a worse defect than a wrong number, because a wrong number can be checked.

Every receipt in this program should be regenerable by a committed script. This one is not, and
the rule is now explicit rather than assumed.

## What this licenses

Narrowly: **the effective degrees of freedom of a 300-timepoint BOLD series are not estimable to
the precision C5's argument required.** Any future gate on this data must either use a statistic
that does not need an effective-n estimate — a phase-randomised surrogate, which is what
`styxx.coupling` already does and why it does it — or state its conclusion as robust across the
estimator family, or not be run.

What this does **not** license: any claim that a particular estimator is correct for BOLD. AR(1)
agreement disqualifies; it cannot establish correctness on a process that is not AR(1). Every
number above is a statement about estimators on AR(1) and about the spread they produce on real
series, not about the truth of either.

*Frozen before the bake-off ran; the losing branch named in advance and, on the operative
reading, taken; a gate of my own reported as defective in the document its own result produced.
Every number grounds in `e1_result.json`.*
