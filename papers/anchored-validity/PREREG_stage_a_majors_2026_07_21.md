# PREREG — the last Stage-A majors: the licensing fork, DS coverage, the partial-keep path

date: 2026-07-21
status: FROZEN before the scored run; committed with the code.
subject: the three still-owed majors from the 2026-07-19 panel — fix 6 (R5 licensing fork,
enforced with a dose-response sweep), fix 7 (R2 check rename + DS bootstrap CI + coverage
check), fix 8's remaining half (the R4 partial-keep path, which has had zero coverage).

## stream discipline (unchanged rule)

All new randomness comes from a dedicated generator or from the main stream STRICTLY AFTER
the R8 block, so every existing check re-evaluates on draws identical to the committed scored
run. Any drift in a pre-existing check's detail line voids the run.

## frozen changes and bars

**Fix 7 — R2.** (a) RENAME `R2:ds_fails_the_same_bar_anchored_meets` to
`R2:ds_misses_the_recovery_bar` — the name now states what the condition tests (DS misses the
0.03 recovery bar at the correlated doses); condition unchanged. (b) DS gains a bootstrap CI
(dedicated rng, resamples of the same panels). NEW GATE `R2:ds_ci_fails_coverage_where_anchored_covers`:
at rho 0.30 AND 0.45 the DS interval does NOT cover the true prevalence (the anchored
interval's coverage at every dose is already gated) — "confidently wrong" becomes a checked
sentence or dies.

**Fix 6 — R5b.** Dose-response sweep, doses {0.05, 0.10, 0.15} on two channels: anchor-alpha
pessimism (alpha_shift +d) and anchor-beta optimism (organic betas -d). Per cell: 1-parameter
anchored error, bound d/min-informativeness, naive (majority-vote) bias, the licensing verdict
(licensed iff bound < naive bias) and the realized outcome (correct iff anchored error < naive
bias). GATES: `R5b:degradation_within_bound` (err <= bound + 0.03, every cell);
`R5b:degradation_grows_with_dose` (one-sided +0.005, per channel);
`R5b:fork_is_two_sided` (the sweep must contain at least one cell where correction is licensed
AND realized-correct, and at least one cell where it is NOT licensed — a fork that always says
one thing is not a fork). Bound-tightness ratios reported unbarred.

**Fix 8 — R4b.** Mixed panel: two informative judges (alpha 0.15/0.20, beta 0.85/0.80) and two
deaf (0.45/0.52), n and K as everywhere. GATES: `R4b:deaf_judges_dropped` (kept mask exactly
informative-true, deaf-false); `R4b:subset_recovers` (verdict ESTIMATED, |pi - 0.35| <= 0.05,
interval covers). This exercises the idx-subset moment assembly (two first moments + one pair
moment) that has never been under test. Single-draw fixtures, disclosed as such; the rate
machinery precedent (R9) covers their future.

A missed bar is CLOSED_NEGATIVE verbatim. No existing bar moves; the rename changes a name,
not a condition.
