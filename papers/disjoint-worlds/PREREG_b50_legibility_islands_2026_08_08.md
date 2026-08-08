# PREREG — B50: do LEGIBILITY-defined islands recur? — b48 re-asked with a null bar that can be met

Fathom Lab · 2026-08-08 · frozen before the fresh nulls are drawn.

B48 asked whether legibility-defined islands recur across ten models and returned
`INVALID__null_leaks` — not because the nulls leaked, but because its G2 judged the **maximum of
45 null draws** against a bar set for a *single* draw. The nulls were textbook: median 0.0104,
exactly the 1/96 chance floor. B50 re-asks the question with a criterion that accounts for the
distribution it judges, per the C5 lesson that a bar fixed without computing achievable power is
decoration.

**The legibility matrix from B48 is reused** (it was never in question). **The nulls are drawn
fresh under a different seed**, because B48's were seen and a criterion chosen after seeing them
is not a preregistration.

## Method (frozen)

Ten models, `normeq_reps.npz`, 96-concept battery, 45 unordered pairs, mean of both directions —
identical to B48. Fresh matched shuffled-geometry nulls at seed 8080. Chance is 1/96 = 0.0104.

```gates
{"gates": {"G0_coverage": {"metric": "n_pairs", "op": ">=", "value": 45,
             "power_basis": "the complete pair set of a ten-model cohort; achievable by construction and falsifiable if any fit fails",
             "metric_means": "count of unordered model pairs scored"},
           "G1_signal_present": {"metric": "max_pair_legibility", "op": ">=", "value": 0.0521,
             "power_basis": "5x chance, the multiple the original b37 matrix used to call a read real; B48 measured 0.2396 on this battery so the bar is known reachable",
             "metric_means": "highest mean-of-both-directions discovery accuracy over all pairs"},
           "G2_null_at_chance": {"metric": "median_null_legibility", "op": "<=", "value": 0.0208,
             "power_basis": "the MEDIAN of 45 draws against 2x chance -- B48's error was gating the MAXIMUM of 45 draws against this same single-draw bar, which one draw at 5/96 clears by ordinary luck; a median is stable under that",
             "metric_means": "median across the 45 matched shuffled-geometry null fits"},
           "G3_null_tail_bounded": {"metric": "frac_nulls_above_5x_chance", "op": "<=", "value": 0.10,
             "power_basis": "an explicit tail bound replacing the max: under a true null at most a small fraction of 45 discrete draws should reach 5x chance, and 0.10 allows 4 of 45 -- generous enough that ordinary luck cannot fail it, tight enough that a real leak does",
             "metric_means": "fraction of the 45 null fits at or above 0.0521"},
           "G4_islands": {"metric": "bimodality_p_member_legibility", "op": "<=", "value": 0.05,
             "power_basis": "gap screen on ten per-member means; the C5 lesson applies -- at n=10 this test has little power against a lone island, so a NON-detection here is weak evidence of absence and the finding must say so",
             "metric_means": "gap-screen p on the vector of ten per-member mean legibilities"}},
 "outcomes": [{"when": {"G0_coverage": false}, "verdict": "INVALID__incomplete_matrix"},
              {"when": {"G0_coverage": true, "G1_signal_present": false}, "verdict": "INVALID__battery_carries_no_discovery_signal"},
              {"when": {"G0_coverage": true, "G1_signal_present": true, "G2_null_at_chance": false}, "verdict": "INVALID__null_off_chance"},
              {"when": {"G0_coverage": true, "G1_signal_present": true, "G2_null_at_chance": true, "G3_null_tail_bounded": false}, "verdict": "INVALID__null_tail_heavy"},
              {"when": {"G0_coverage": true, "G1_signal_present": true, "G2_null_at_chance": true, "G3_null_tail_bounded": true, "G4_islands": true}, "verdict": "LEGIBILITY_ISLANDS_RECUR__structure_generalizes"},
              {"when": {"G0_coverage": true, "G1_signal_present": true, "G2_null_at_chance": true, "G3_null_tail_bounded": true, "G4_islands": false}, "verdict": "NO_LEGIBILITY_ISLANDS__the_first_island_does_not_generalize"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

Stated in advance: at n=10 the gap screen has little power against a lone island, so
`NO_LEGIBILITY_ISLANDS` is **weak evidence of absence** and the finding must say so in those
words. Every gate declares both a power basis and what its metric means — the first prereg in
this program written entirely under the v2/v3 machinery.
