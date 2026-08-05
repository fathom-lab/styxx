# PREREG — B42: the bridge dose curve — how few directions, and does it replicate?

Fathom Lab · 2026-08-05 · frozen before the scored run. B41 built the bridge at rank-20, single
seed: correcting qwen's top-20 concept-contrasts turned the island legible (0.9745) while a
random 20-frame did nothing (0.0000). Two honest weaknesses remain: **single seed**, and **why
20?** This run hardens both — a rank sweep across five seeds, each with its matched random-frame
null.

## Design (frozen)

The B41 surgery verbatim (llama_3b → qwen, rank-k concept-space contrast swap), swept over
**k ∈ {1, 2, 3, 5, 8, 12, 20, 40}** and **seeds {343, 1001, 1002, 1003, 1004}** (the split +
row-shuffle seed; each seed is its own complete discovery). Per (k, seed): bridge discovery and
a random-orthonormal-k-frame null discovery. 8 ranks × 5 seeds × 2 arms = 80 discovery fits,
CPU-from-cache.

## Pre-stated structure

- **Replication:** at k=20, the bridge holds across all five seeds (each ≥ 0.30) and every
  matched null stays low (each ≤ 0.15). This is the B41 result made non-single-seed.
- **Dose:** report the full mean±spread bridge curve over k, and the **minimum rank k\*** at
  which median bridge ≥ 0.30 while median null ≤ 0.15. k\* is the headline number — *how few
  directions the barrier actually is.* Reported with its curve; not gated to a specific value
  (no measured prior exists for k\*, and inventing a bar would be the b37 G2 sin).

```gates
{"gates": {"G1_replicates": {"metric": "min_bridge_disc_at_k20", "op": ">=", "value": 0.30},
           "G2_null_clean": {"metric": "max_null_disc_at_k20", "op": "<=", "value": 0.15},
           "G3_dose_monotone": {"metric": "spearman_medianbridge_vs_k", "op": ">=", "value": 0.6}},
 "outcomes": [{"when": {"G1_replicates": false}, "verdict": "BRIDGE_FRAGILE__single_seed_artifact"},
              {"when": {"G1_replicates": true, "G2_null_clean": false}, "verdict": "INVALID__null_leaks_across_seeds"},
              {"when": {"G1_replicates": true, "G2_null_clean": true, "G3_dose_monotone": true}, "verdict": "BRIDGE_REPLICATES_AND_DOSES__barrier_is_low_rank"},
              {"when": {"G1_replicates": true, "G2_null_clean": true, "G3_dose_monotone": false}, "verdict": "BRIDGE_REPLICATES__dose_nonmonotone"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

G3 tests that more corrected directions help monotonically (≥0.6 Spearman over 8 ranks — a
real-effect bar, one or two inversions tolerated), distinguishing "a low-rank barrier that
fills in with k" from "a single magic k". A non-monotone pass is reported honestly, not spun.

## Outcome reading

- **`BRIDGE_REPLICATES_AND_DOSES`**: the bridge is a robust, low-rank, dose-dependent
  phenomenon — the barrier between these two minds is a small correctable subspace whose size
  we now quantify (k\*). The connection-of-minds paper gains its causal capstone.
- **`BRIDGE_FRAGILE`**: k=20 did not hold across seeds — B41 was a lucky draw, demoted at full
  volume, and the naming stays correlational.

## Discipline

CPU-from-cache, zero model loads; ~80 fits. Smoke = k∈{1,20} × 2 seeds, INVALID-only. Result
`b42_result.json`; scored by `styxx.protocol`; certified + sealed before commit. Null uses an
independent seed stream (7000+index) so bridge and null never share a shuffle.
