# PREREG — B46: the cliff function — how does legibility fall with frame rotation?

Fathom Lab · 2026-08-06 · frozen before the scored run. B45 measured the paradox: qwen's
frame shares most of its squared-cosine mass with the clique's, yet discovery reads it near
zero until the frame is fully corrected. So the legibility-vs-rotation function must be
steep somewhere between "mostly aligned" and "aligned." B46 maps it: interpolate the B41
surgery between the island's own frame and the reader's, and measure discovery at each step.

## Design (frozen)

The B41 surgery with an interpolated frame: at dose t ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0},
`U(t)` = the orthonormalized (QR) blend `(1−t)·U_T + t·U_S` of the target's and the reader's
k=20 concept-Gram eigenframes, and `X(t) = X_T − U_T L + U(t) L` with the target's own
loadings `L` (B41 machinery verbatim; t=0 is the untouched island, t=1 is the full bridge).
Discovery per (t, seed): the committed b34-v3 machinery, seeds {343, 1001, 1002}, training
rows and locked k* as in B42. 18 fits, CPU-from-cache.

## Gates

Endpoint bars are inherited from B42's frozen gates (0.15 null ceiling, 0.30 replication
floor); the monotonicity bar is B42's real-effect Spearman. The *shape* of the transition —
the knee's location and width — is reported ungated: no measured prior exists, and inventing
a bar would repeat the b37-G2 error.

```gates
{"gates": {"G0_baseline_low": {"metric": "max_disc_at_t0", "op": "<=", "value": 0.15},
           "G1_bridge_high": {"metric": "min_disc_at_t1", "op": ">=", "value": 0.30},
           "G2_monotone": {"metric": "spearman_mediandisc_vs_t", "op": ">=", "value": 0.6}},
 "outcomes": [{"when": {"G0_baseline_low": false}, "verdict": "INVALID__baseline_not_reproduced"},
              {"when": {"G0_baseline_low": true, "G1_bridge_high": false}, "verdict": "INVALID__bridge_not_reproduced"},
              {"when": {"G0_baseline_low": true, "G1_bridge_high": true, "G2_monotone": true}, "verdict": "CLIFF_MAPPED__legibility_rises_monotonically_with_dose"},
              {"when": {"G0_baseline_low": true, "G1_bridge_high": true, "G2_monotone": false}, "verdict": "CLIFF_NONMONOTONE__transition_has_structure"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

Reported ungated, per the shape question: the median-curve values at every t; the knee
location t½ (first t where the median crosses half the t=1 median); and the transition width
(t between one-quarter and three-quarters of the t=1 median). A sharp knee near t=1 would
say legibility demands near-exact frame agreement (the strictest reading of the cliff); a
gradual rise would say partial correction buys partial reading and B44's wrong-donor medians
were points on a smooth curve.

## Discipline

Smoke = t ∈ {0, 1} × 1 seed, INVALID-only. Result `b46_result.json`; scored by
`styxx.protocol`; certified + sealed before commit. Full per-(t, seed) grid ships regardless
of verdict.
