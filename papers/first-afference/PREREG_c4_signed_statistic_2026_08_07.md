# PREREG — C4: the signed statistic, composed with the trend guard — frozen before the run

Fathom Lab · 2026-08-07. C3 failed because |r| is a fold (E[|cos|] != 0) and because the trend
guard was omitted. C4 fixes exactly those two things and nothing else.

## Method (frozen)
Statistic: **signed** mean matched-column Pearson r (no absolute value; sign is shared in
matched space, so surrogate expectation is genuinely zero). Guards, all required: surrogate p
<= 0.01 (500 phase-randomized draws, C2 machinery unchanged), confound-matched permutation p
<= 0.01, and the `_trend_r2` shared-trend refusal from `couple()` (both streams >= 0.2 linear
R^2 -> refused, not licensed). Data, pairs, vertices, seeds, attacks: identical to C2/C3.

```gates
{"gates": {"G1_finds_isc": {"metric": "frac_real_coupled", "op": ">=", "value": 0.80},
           "G2_rejects_reversed": {"metric": "frac_reversed_coupled", "op": "<=", "value": 0.10},
           "G3_rejects_independent_ar": {"metric": "frac_independent_ar_coupled", "op": "<=", "value": 0.10},
           "G4_rejects_shared_trend": {"metric": "frac_shared_trend_coupled", "op": "<=", "value": 0.10}},
 "outcomes": [{"when": {"G1_finds_isc": false}, "verdict": "MECHANISM_INCOMPLETE__signed_statistic_also_blind"},
              {"when": {"G1_finds_isc": true, "G3_rejects_independent_ar": false}, "verdict": "REGRESSION__ar_false_positives"},
              {"when": {"G1_finds_isc": true, "G3_rejects_independent_ar": true, "G4_rejects_shared_trend": false}, "verdict": "REGRESSION__trend_false_positives"},
              {"when": {"G1_finds_isc": true, "G3_rejects_independent_ar": true, "G4_rejects_shared_trend": true, "G2_rejects_reversed": false}, "verdict": "LEAK__reversed_pairs_license"},
              {"when": {"G1_finds_isc": true, "G2_rejects_reversed": true, "G3_rejects_independent_ar": true, "G4_rejects_shared_trend": true}, "verdict": "RECALIBRATED__signed_isc_licenses_and_holds_all_refusals"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

Stated in advance: `MECHANISM_INCOMPLETE` would mean C2's story is genuinely wrong and would be
the week's biggest result. Success ships only after a red team, per the standing rule.
