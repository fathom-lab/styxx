# PREREG — the confound in the flagship knob: rotation or timescale diversity, on permuted MNIST — 2026-09-02

**FROZEN before confirmatory data.** Runner: `run_pmnist_untied.py` (imports
`run_pmnist_ablation.py` verbatim for the data, the FREE and CLAMPED arms, the training loop and
the evaluation). Follows `RESULT_untied_magnitudes_2026_09_02.md`, which found on the ordered-copy
task that untying the clamped bank's magnitudes recovers none of the capacity (recovery 0.0), and
`RESULT_pmnist_ablation_2026_07_23.md`, the arc's flagship (+0.312, FREE 0.920 vs CLAMPED 0.607),
measured with the same knob and carrying the same confound.

## The confound, restated

θ≡0 removes rotation and ties each mode's two real channels to one magnitude. REAL2 — a real
bank with 2·D_SSM modes and 2·D_SSM independent magnitudes, no rotation — has the same real state
width and, by the red-team check, the same parameter count as FREE, inside the same three-block
classifier with the same projection, norms, feed-forward and head. Everything else is the
flagship runner's: the fixed pixel permutation (seed 1234), H=64, D_SSM=64, three blocks, 4000
steps of AdamW with cosine decay, batch 64, seeds 0 and 1, full 10,000-image test accuracy.

## Question

> On the benchmark where oscillation's causal advantage was largest, does a real bank with FREE's
> parameters and twice its independent timescales recover FREE's accuracy, or is the +0.312 the
> rotation's?

## Gates

```gates
{"gates": {"G_P_anchors": {"metric": "anchor_max_abs_dev", "op": "<=", "value": 0.03,
                           "power_basis": "FREE and CLAMPED re-run on CPU must land within 0.03 of the committed GPU receipt (0.9199, 0.6067); two seeds of this task moved by under 0.01 between the arc's own runs, and 0.03 is the allowance for device-level nondeterminism in a 784-step scan"},
           "G_C_gap": {"metric": "gap_free_minus_clamped", "op": ">=", "value": 0.15,
                       "power_basis": "the receipt's gap is 0.312; the flagship RESULT's own load-bearing threshold was 0.02 and half the measured gap is a conservative floor for 'the effect this control exists to explain reproduced'"},
           "G_R_recovers": {"metric": "free_minus_real2", "op": "<=", "value": 0.03,
                            "power_basis": "REAL2 within 0.03 of FREE — the anchor tolerance — is a tie at this task's noise; a tie is the diversity claim"},
           "G_R_fails": {"metric": "real2_minus_clamped", "op": "<=", "value": 0.03,
                         "power_basis": "REAL2 within 0.03 of CLAMPED means untying bought nothing the task can see"}},
 "outcomes": [{"when": {"G_P_anchors": false}, "verdict": "INVALID__plumbing_anchors_drifted"},
              {"when": {"G_P_anchors": true, "G_C_gap": false}, "verdict": "INVALID__gap_did_not_reproduce"},
              {"when": {"G_P_anchors": true, "G_C_gap": true, "G_R_recovers": true}, "verdict": "TIMESCALE_DIVERSITY__rotation_not_load_bearing_here"},
              {"when": {"G_P_anchors": true, "G_C_gap": true, "G_R_recovers": false, "G_R_fails": true}, "verdict": "ROTATION_LOAD_BEARING__beyond_diversity"},
              {"when": {"G_P_anchors": true, "G_C_gap": true, "G_R_recovers": false, "G_R_fails": false}, "verdict": "PARTIAL__diversity_recovers_some"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

`recovery_fraction = (REAL2 − CLAMPED) / (FREE − CLAMPED)` is reported beside the gates.

## Disclosed prior

Toward ROTATION_LOAD_BEARING, because the ordered-copy control recovered nothing; but permuted
MNIST is a classification over 784 steps where a spread of decay rates might carry more than it
does for ordered recall, and the arc's own scarcity result is a warning that easy-task findings do
not always travel. The bar is frozen either way.

## Discipline

Committed before the run. Smoke (`--smoke`: 200 steps, one seed) is INVALID-only. Result →
`pmnist_untied_result.json`, scored through `styxx.protocol`, RESULT sworn to the receipt. No bar
moves after data. If this sandbox cannot afford the run, the preregistration stands frozen for
whoever can, and its verdict is theirs to report.
