# PREREG — B38: the legibility cliff — does alignment LEVEL explain the island, or is the island structural?

Fathom Lab · 2026-08-04 · frozen before the scored run.

## Disclosed reconnaissance (committed data, run before this prereg, per the cycle-81 rail)

Two island hypotheses died in recon on the banked extractions: (1) **outlier dimensions** —
qwen has the LOWEST variance concentration of the four models (top-1 dim share 0.007 vs
gemma's 0.036), effective dimensionality and distance-CV indistinguishable from the clique;
(2) **gross relational dissimilarity** — qwen's RSA to the clique is 0.877–0.886, HIGH in
absolute terms, versus clique-internal 0.934–0.963. The recon leaves one sharp pattern: a
~0.05–0.08 RSA gap separates full mutual legibility (disc 0.59–0.83) from near-total
blindness (disc ≤ 0.17). This prereg turns that observation into a causal dose-response.

## Design (frozen)

Baseline pair: llama_3b → gemma_2b (the B37 anchor cell, disc 0.5918 at RSA 0.9593 as
measured on the fit rows). Degrade ONLY the target: add isotropic Gaussian noise to gemma's
462 concept points at doses σ = f × (mean per-dimension std of gemma's centered points),
f ∈ {0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0} (10 doses, one seeded draw each,
noise seed 3801+dose-index). Per dose: recompute pair RSA (committed `distmat` metric on the
392 fit rows), run the full label-free discovery (b34-v3 machinery verbatim, seed 343 split
and shuffle rail), record `disc` (seed_acc). Control curve: the same protocol on
llama_3b → llama_1b (starts higher, 0.7959) — is the response pair-general?

**The interpolated quantity that decides the island question:**
`disc_at_qwen_rsa` = the discovery accuracy linearly interpolated from the gemma dose curve
at RSA = 0.881 (the measured mean qwen→clique RSA). If no two adjacent doses bracket 0.881,
the nearest measured dose stands in and that substitution is disclosed in the result.

## Gates (frozen; scored by styxx.protocol)

```gates
{"gates": {"G0_baseline": {"metric": "gemma_dose0_disc", "op": ">=", "value": 0.30},
           "G1_monotone": {"metric": "spearman_disc_vs_dose", "op": "<=", "value": -0.8},
           "G2_island_matched": {"metric": "disc_at_qwen_rsa", "op": "<=", "value": 0.20},
           "G2b_island_not_matched": {"metric": "disc_at_qwen_rsa", "op": ">=", "value": 0.30}},
 "outcomes": [{"when": {"G0_baseline": false}, "verdict": "INVALID__baseline_not_reproduced"},
              {"when": {"G0_baseline": true, "G1_monotone": false}, "verdict": "INVALID__dose_response_nonmonotone"},
              {"when": {"G0_baseline": true, "G1_monotone": true, "G2_island_matched": true}, "verdict": "ISLAND_EXPLAINED_BY_ALIGNMENT_LEVEL"},
              {"when": {"G0_baseline": true, "G1_monotone": true, "G2_island_matched": false, "G2b_island_not_matched": true}, "verdict": "ISLAND_IS_STRUCTURAL__alignment_level_insufficient"},
              {"when": {"G0_baseline": true, "G1_monotone": true, "G2_island_matched": false, "G2b_island_not_matched": false}, "verdict": "AMBIGUOUS__between_bars"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

## Outcome reading, pre-committed

- **`ISLAND_EXPLAINED_BY_ALIGNMENT_LEVEL`**: a clique member noise-matched to qwen's RSA
  becomes island-blind → legibility is governed by relational-alignment *amount*; qwen is
  quantitatively, not qualitatively, different. The B37 theory question closes to "why is
  qwen's alignment lower," a training-data/objective question.
- **`ISLAND_IS_STRUCTURAL`**: discovery survives at qwen's RSA level under isotropic
  degradation → alignment level does NOT explain the island; qwen's misalignment is
  *structured* in a way isotropic noise is not — the difference is in kind. The successor is
  a structured-perturbation study (what KIND of distortion kills discovery at matched RSA?).
- **Shape of the curve** (cliff vs slope: the transition width in RSA units between 80% and
  20% of baseline disc) is REPORTED with the curve, not gated — no measured base rate exists
  to size a width bar, and the b37 G2 lesson (no noise-passable or unearned bars) applies.

## Discipline

CPU-from-cache, zero model loads; ~22 discovery fits ≈ 80–100 min. Smoke = 3 doses at 40/10,
INVALID-only. Result `b38_result.json`; scored by `styxx.protocol`; recon receipt
(`b38_recon_addendum.json`) committed beside this prereg; certified + sealed before commit.
