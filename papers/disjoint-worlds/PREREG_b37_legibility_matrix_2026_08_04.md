# PREREG — B37: the mutual-legibility matrix — which minds can find each other, and what predicts it

Fathom Lab · 2026-08-04 · frozen before the scored run. The arc's standing theory question:
label-free discovery works Llama-3B→gemma (seed acc 0.59–0.88 across seeds) and fails
Llama-3B→qwen (0.036–0.094), while RSA-isometry predicts neither (rung-2: the highest-RSA
pair read at chance). **What property of a representation pair makes its correspondence
discoverable from geometry alone?** Four models' concept extractions are banked
(`_b31v2_ptsA.npz` = Llama-3.2-3B, `_b31v2_pts_{llama_1b,gemma_2b,qwen_1p5b}.npz`), so every
one of the **12 directed pairs** runs CPU-from-cache. This is the first all-pairs map of
mutual legibility between artificial minds, and a falsifiable test of one candidate law.

## Design (frozen)

For each ordered pair (S→T) of the four models, the b34-v3 pipeline verbatim (seed 343 split:
392 anchors / 70 held-out; seeded target-row shuffle; `TransferMap.fit` label-free discovery;
one b31v2 MLP on the discovered pairing; 70-way read; chance 1/70). Recorded per pair:
`seed_acc` (discovery), `read_top1`, and a pairing-shuffled null read.

**Predictors, computed per pair** (analysis MAY use true labels — it explains discovery, it
never feeds it):
- **RSA** — the committed `distmat` correlation (the rung-2 metric; the incumbent that already
  failed for reads, now formally tested for discovery).
- **kNN-Jaccard** — for k=10: each concept's k nearest neighbors within its own space; the
  mean Jaccard overlap of the two spaces' neighbor sets under the TRUE correspondence. The
  candidate law: **discovery is gated by local neighborhood preservation, not global isometry.**
- **Spectral profile similarity** — cosine between the two spaces' log PCA-eigenvalue decay
  profiles (first 50 components) — the "shape of the space" control predictor.

## The pre-stated prediction (what makes this falsifiable, not a fishing trip)

**P1:** Across the 12 pairs, Spearman(kNN-Jaccard, seed_acc) > Spearman(RSA, seed_acc), and
Spearman(kNN-Jaccard, seed_acc) ≥ 0.60. If local-structure preservation does not out-predict
the already-falsified global-isometry metric, the candidate law dies here.

**P2 (symmetry, reported + gated weakly):** discovery is approximately symmetric —
|seed_acc(A→B) − seed_acc(B→A)| median across the 6 unordered pairs ≤ 0.15. A strong
asymmetry would be its own discovery (direction matters to legibility).

## Gates (frozen; scored by styxx.protocol)

```gates
{"gates": {"G0_reproduce": {"metric": "llama3b_to_gemma_seed_acc", "op": ">=", "value": 0.30},
           "G0b_reproduce_weak": {"metric": "llama3b_to_qwen_seed_acc", "op": "<=", "value": 0.20},
           "G1_law": {"metric": "spearman_knn_vs_disc", "op": ">=", "value": 0.60},
           "G2_beats_rsa": {"metric": "knn_minus_rsa_spearman", "op": ">", "value": 0.0},
           "G3_symmetry": {"metric": "median_abs_asymmetry", "op": "<=", "value": 0.15}},
 "outcomes": [{"when": {"G0_reproduce": false}, "verdict": "INVALID__anchor_cell_not_reproduced"},
              {"when": {"G0_reproduce": true, "G0b_reproduce_weak": false}, "verdict": "INVALID__weak_cell_not_reproduced"},
              {"when": {"G0_reproduce": true, "G0b_reproduce_weak": true, "G1_law": true, "G2_beats_rsa": true, "G3_symmetry": true}, "verdict": "LEGIBILITY_LAW_CANDIDATE__local_structure_symmetric"},
              {"when": {"G0_reproduce": true, "G0b_reproduce_weak": true, "G1_law": true, "G2_beats_rsa": true, "G3_symmetry": false}, "verdict": "LEGIBILITY_LAW_CANDIDATE__local_structure_ASYMMETRIC"},
              {"when": {"G0_reproduce": true, "G0b_reproduce_weak": true, "G1_law": true, "G2_beats_rsa": false}, "verdict": "PREDICTOR_TIE__knn_no_better_than_rsa"},
              {"when": {"G0_reproduce": true, "G0b_reproduce_weak": true, "G1_law": false, "G2_beats_rsa": true}, "verdict": "CANDIDATE_LAW_DIES__knn_below_floor"},
              {"when": {"G0_reproduce": true, "G0b_reproduce_weak": true, "G1_law": false, "G2_beats_rsa": false}, "verdict": "CANDIDATE_LAW_DIES__nothing_predicts"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

## Honesty rails, stated before data

- **n = 12 pairs.** A Spearman over 12 points is coarse; the 0.60 floor is a strong-effect
  bar, and the matrix itself (all 12 cells published) is the primary artifact regardless of
  which outcome fires. No p-value theater at this n; the licensed claim on a pass is
  "candidate law, strong in-battery," never "law of nature."
- The 12 pairs share 4 underlying models — observations are not independent; disclosed, not
  correctable at this scale. The successor with more model families is the real test of any
  candidate this run licenses.
- G0/G0b force the matrix to reproduce both known cells before any correlation is read —
  the b35-b lesson (an unvalidated apparatus licenses nothing) applied in advance.
- The one-seed scope note: the matrix runs at seed 343 (b35-a already measured seed variance
  for the anchor pairs; per-pair seed sweeps are a successor if a law candidate survives).
- Source read layers: every model uses the SAME extraction bank it already has (Llama-3B at
  its G0-locked layer; targets at their frac-rule layers). When a model serves as SOURCE, its
  banked extraction is used as-is — **this is the b35-b confound accepted openly as scope**:
  cells with a non-optimized source layer are labeled `source_layer_unvalidated` in the
  matrix, G0/G0b anchor cells are unaffected (Llama-3B source IS G0-locked), and any
  cross-pair conclusion is stated over the matrix as-measured, not as-optimal.

## Discipline

CPU-from-cache, zero model loads. Smoke = 3 pairs × 40/10, INVALID-only. Result
`b37_result.json`, scored by `styxx.protocol`; certified + sealed before commit.
