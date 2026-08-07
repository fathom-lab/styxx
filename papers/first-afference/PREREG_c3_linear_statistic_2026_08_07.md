# PREREG — C3: the linear statistic under spectral surrogates — the exam the algebra predicts it passes

Fathom Lab · 2026-08-07 · frozen before the scored run. C2 proved analytically that a squared
statistic (RV/CKA) has a spectral-surrogate null floor at half the cross-power. The corollary:
a statistic **linear** in cross-covariance has surrogate expectation zero. For matched-space
streams that statistic is the **mean matched-column Pearson correlation** — the field's classic
ISC measure. C3 tests that corollary under the identical four-gate exam, on the identical data.

## Method (frozen)

`isc_stat(A, B)` = mean over matched columns j of |Pearson r(A[:,j], B[:,j])|. Licensing null:
500 phase-randomized surrogates of B (the C2 `phase_randomize`, unchanged); p = fraction of
surrogate statistics ≥ observed, add-one smoothed. "Licensed" = p ≤ 0.01. Confound-matched
permutation is retained as a second required gate (both must pass). Everything else — data,
pairs, vertices, seed, attacks — identical to C2.

## Gates

```gates
{"gates": {"G1_finds_isc": {"metric": "frac_real_coupled", "op": ">=", "value": 0.80},
           "G2_rejects_reversed": {"metric": "frac_reversed_coupled", "op": "<=", "value": 0.10},
           "G3_rejects_independent_ar": {"metric": "frac_independent_ar_coupled", "op": "<=", "value": 0.10},
           "G4_rejects_shared_trend": {"metric": "frac_shared_trend_coupled", "op": "<=", "value": 0.10}},
 "outcomes": [{"when": {"G1_finds_isc": false}, "verdict": "ALGEBRA_WRONG__linear_statistic_also_blind"},
              {"when": {"G1_finds_isc": true, "G3_rejects_independent_ar": false}, "verdict": "REGRESSION__ar_false_positives"},
              {"when": {"G1_finds_isc": true, "G3_rejects_independent_ar": true, "G4_rejects_shared_trend": false}, "verdict": "REGRESSION__trend_false_positives"},
              {"when": {"G1_finds_isc": true, "G3_rejects_independent_ar": true, "G4_rejects_shared_trend": true, "G2_rejects_reversed": false}, "verdict": "LEAK__reversed_pairs_license"},
              {"when": {"G1_finds_isc": true, "G2_rejects_reversed": true, "G3_rejects_independent_ar": true, "G4_rejects_shared_trend": true}, "verdict": "ALGEBRA_CONFIRMED__linear_statistic_licenses_isc_and_holds_refusals"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

## Stated before the run

- `ALGEBRA_WRONG` is a live branch: the cos-squared argument predicts success, and if the data
  refuses anyway, the mechanism story of C2 is incomplete and must be amended.
- Success licenses a **matched-space** ISC mode only. It does not resurrect RV for licensing,
  does not reopen mind↔brain by itself, and ships only after a red team, per the standing rule.
- The statistic requires equal column counts; that constraint is stated as a limitation, not
  hidden. Mismatched-space licensing under surrogates remains open.
