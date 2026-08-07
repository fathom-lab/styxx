# PREREG — C5: the same exam, on the tissue where ISC actually lives

Fathom Lab · 2026-08-07 · frozen before the ROI data is downloaded. C4 left one design constant
standing: the vertex set. C5 changes exactly that and nothing else — the streams become each
subject's early-auditory-cortex (EAC) mean timeseries, left and right hemisphere (two matched
columns), from the dataset's own shipped ROI extractions (`roi-EAC_desc-mean_timeseries.1D`,
pieman run-1, same seven subjects). Statistic, guards, nulls, gates, attacks: identical to C4
(signed mean matched-column r; surrogate p and matched p both <= 0.01; shared-trend refusal).

```gates
{"gates": {"G1_finds_isc": {"metric": "frac_real_coupled", "op": ">=", "value": 0.80},
           "G2_rejects_reversed": {"metric": "frac_reversed_coupled", "op": "<=", "value": 0.10},
           "G3_rejects_independent_ar": {"metric": "frac_independent_ar_coupled", "op": "<=", "value": 0.10},
           "G4_rejects_shared_trend": {"metric": "frac_shared_trend_coupled", "op": "<=", "value": 0.10}},
 "outcomes": [{"when": {"G1_finds_isc": false}, "verdict": "FRAMEWORK_WRONG__blind_even_on_the_right_tissue"},
              {"when": {"G1_finds_isc": true, "G3_rejects_independent_ar": false}, "verdict": "REGRESSION__ar_false_positives"},
              {"when": {"G1_finds_isc": true, "G3_rejects_independent_ar": true, "G4_rejects_shared_trend": false}, "verdict": "REGRESSION__trend_false_positives"},
              {"when": {"G1_finds_isc": true, "G3_rejects_independent_ar": true, "G4_rejects_shared_trend": true, "G2_rejects_reversed": false}, "verdict": "LEAK__reversed_pairs_license"},
              {"when": {"G1_finds_isc": true, "G2_rejects_reversed": true, "G3_rejects_independent_ar": true, "G4_rejects_shared_trend": true}, "verdict": "DILUTION_CONFIRMED__four_exams_of_blindness_were_the_wrong_tissue"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

Stated in advance: `DILUTION_CONFIRMED` collapses C1–C4 into "a working instrument pointed at
the wrong tissue"; `FRAMEWORK_WRONG` means the licensing framework itself is broken and is the
larger result. Ships only after a red team either way.
