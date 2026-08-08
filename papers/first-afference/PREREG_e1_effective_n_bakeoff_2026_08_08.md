# PREREG — E1: which effective-sample-size estimator is right, and does C5 survive it?

Fathom Lab · 2026-08-08 · frozen before the bake-off runs.

## Why

`styxx.power.effective_n` (quarantined 2026-08-08) and the committed
`c5_effective_df_addendum.json` disagree on **all seven** C5 subjects. The published range 6.9 to
45.8 becomes 4.84 to 53.84 under the other implementation. On sub-001 alone there are three
candidate answers: 28.5 (the addendum), 30.82 (`power.effective_n`), 32.56 (the AR(1) closed form
from the same lag-1). This is an **open defect against a sealed finding** and it is load-bearing:
C5's argument turns on the critical correlation implied by the median effective n, and its
strongest pair sits within 0.0004 of that threshold.

The addendum has **no generator script** — a committed number with no reproducible path. Part of
this run is identifying what produced it.

## Method

Candidate estimators are scored against **analytic ground truth on AR(1)**, where the integrated
autocorrelation is exactly `(1+ρ)/(1-ρ)` and the effective n is exactly `n(1-ρ)/(1+ρ)`. Grid:
ρ ∈ {0.5, 0.7, 0.8, 0.9, 0.95}, n = 300 to match C5, 400 replicates per cell, fixed seed.

**Stated before running, and it binds the interpretation:** AR(1) agreement **cannot establish
correctness on BOLD**, which is not AR(1). It can only *disqualify* — an estimator badly biased
where truth is known is not one to trust where truth is unknown. The winner is therefore "least
disqualified", not "correct", and the finding must say so.

A separate **disqualification** applies regardless of accuracy score: an estimator that reports
*no correction at all* on a series it should correct is unsafe at any bias, because its failure
is silent. The probe is a process with negative lag-1 and strong lag-2 structure, where the
true effective n is far below nominal.

```gates
{"gates": {"G0_grid_complete": {"metric": "n_cells_scored", "op": ">=", "value": 5,
             "power_basis": "one cell per rho on a five-point grid, 400 replicates each; achievable by enumeration and falsifiable only if an estimator raises",
             "metric_means": "count of rho values on which every candidate returned a finite estimate"},
           "G1_a_candidate_is_usable": {"metric": "best_median_abs_rel_error", "op": "<=", "value": 0.20,
             "power_basis": "RELATIVE, not absolute: the best of several candidates must land within 20% median error somewhere on the grid. The red team measured power.effective_n at +11% (rho=0.8), +57% (0.95) and +270% (0.99), so 0.20 is above the best observed and below the worst -- it can be met and it can be failed. If NO candidate clears it the honest verdict is that this quantity is not estimable at n=300 and C5's range is unusable, which is a real outcome and is named below.",
             "metric_means": "over candidates, the minimum of the per-candidate median absolute relative error pooled across the rho grid"},
           "G2_winner_not_silently_blind": {"metric": "winner_flags_the_silent_probe", "op": ">=", "value": 1.0,
             "power_basis": "a boolean the implementation controls completely; the probe is constructed so the true effective n is far below nominal, and any estimator returning effective==nominal there has failed in the one way that cannot be noticed in the field",
             "metric_means": "1.0 if the winning estimator returns an effective n materially below nominal on the negative-lag-1 probe, else 0.0"},
           "G3_c5_range_recomputed": {"metric": "n_c5_subjects_recomputed", "op": ">=", "value": 7,
             "power_basis": "the seven committed eac_sub-*_L.1D series are in the tree and load; achievable by enumeration",
             "metric_means": "count of C5 subjects for which the winning estimator produced a finite effective n"}},
 "outcomes": [{"when": {"G0_grid_complete": false}, "verdict": "INVALID__grid_incomplete"},
              {"when": {"G0_grid_complete": true, "G1_a_candidate_is_usable": false}, "verdict": "NOT_ESTIMABLE__c5_effective_n_range_is_unusable_at_this_length"},
              {"when": {"G0_grid_complete": true, "G1_a_candidate_is_usable": true, "G2_winner_not_silently_blind": false}, "verdict": "NO_SAFE_ESTIMATOR__best_accuracy_fails_silently"},
              {"when": {"G0_grid_complete": true, "G1_a_candidate_is_usable": true, "G2_winner_not_silently_blind": true, "G3_c5_range_recomputed": false}, "verdict": "INVALID__c5_series_did_not_load"},
              {"when": {"G0_grid_complete": true, "G1_a_candidate_is_usable": true, "G2_winner_not_silently_blind": true, "G3_c5_range_recomputed": true}, "verdict": "RESOLVED__winner_selected_and_c5_recomputed"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

## What the winning branch does NOT license

`RESOLVED__winner_selected_and_c5_recomputed` licenses **a recomputed range and a statement about
whether C5's conclusion is sensitive to it.** It does not license a claim that the new range is
correct for BOLD. It does not overwrite the sealed C5 finding — the recomputation lands as an
addendum with its own receipt, and C5 gains a pointer.

**Named in advance:** whether C5's strongest pair crosses its threshold under the winner is
recorded either way. If it crosses, that is reported as prominently as if it had not, and C5's
argument is amended in public. C5's verdict was a null and a crossing would not make it a
positive — but the sentence that carries it would have to change, and this document commits us to
changing it.
