# PREREG — B45: the shared frame, stated as pure geometry — no discovery in the loop

Fathom Lab · 2026-08-06 · frozen before the scored run. B44 concluded "the clique shares a
concept-frame geometry and the island is rotated away from it" — but concluded it *through the
discovery machinery* (wrong-donor surgeries restoring legibility). B45 states the same claim
as direct geometry: the models' top-k concept-Gram eigenframes are label-aligned n×k objects
in a common concept-index space, so frame-to-frame alignment is measurable with principal
angles alone. If B44's reading is right, the clique members' frames should align with each
other far above random, and qwen's frame should sit farther from the clique than the clique
sits from itself. No fitting, no Hungarian, no transfer — just subspace angles.

## Design (frozen)

Models: the clique {llama_3b, gemma_2b, llama_1b} and the island {qwen_1p5b}, from the
committed `.npz` banks. Per seed s ∈ {343, 1001, 1002, 1003, 1004}: the B42 training-row
subset for s; frames `gram_eigvecs(X, k)` for each model; pairwise affinity
`affinity(Ua, Ub, k)` (the committed b40 statistic — mean squared cosine of principal angles)
for all six model pairs; **k ∈ {2, 20}** (the measured core and the full bridge rank).

**Null:** 1000 pairs of Haar-random k-frames at the same (n, k); the null's 95th percentile is
computed inside the run by frozen procedure (data-independent by construction; the analytic
expectation for a random pair is k/n).

## Gates

Two gate shapes, each chosen against a named prior failure:

- **G1** is a 95th-percentile exceedance — a significance statement, not an effect-size floor
  (the b37-G2 sin was a noise-passable magnitude floor; an exceedance over an explicit null
  distribution is the repaired form).
- **G2** is all-seeds sign consistency (probability 1/32 under sign exchange) — an ordering
  claim gated on replication, not on an invented margin.

```gates
{"gates": {"G1_clique_frames_shared": {"metric": "clique_affinity_minus_null_p95_k20", "op": ">=", "value": 0.0},
           "G2_island_separated_all_seeds": {"metric": "seeds_qwen_below_clique_k20", "op": ">=", "value": 5}},
 "outcomes": [{"when": {"G1_clique_frames_shared": false}, "verdict": "FRAMES_NOT_SHARED__b44_transfer_is_not_frame_alignment"},
              {"when": {"G1_clique_frames_shared": true, "G2_island_separated_all_seeds": true}, "verdict": "SHARED_FRAME_CONFIRMED_GEOMETRICALLY__island_rotated_away"},
              {"when": {"G1_clique_frames_shared": true, "G2_island_separated_all_seeds": false}, "verdict": "SHARED_FRAME__island_not_distinct_in_frame_affinity"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

Metrics: `clique_affinity_minus_null_p95_k20` = (median of the three clique-pair affinities
across all seeds at k=20) − (null 95th percentile at k=20). `seeds_qwen_below_clique_k20` =
count of seeds where the median qwen-involving affinity < the median clique-pair affinity at
k=20. k=2 affinities are reported ungated (b43 showed the rank-space carving is seed-unstable;
a rank-2 frame comparison has no measured prior).

## What each branch means (written before data)

- **`SHARED_FRAME_CONFIRMED_GEOMETRICALLY`** — B44's discovery-level story survives with the
  discovery machinery removed: the clique's frames co-align and the island's frame is the
  outlier. The arc's final sentence gets its cheapest, most portable receipt (a statistic any
  replicator can compute in one minute on the committed banks).
- **`SHARED_FRAME__island_not_distinct`** — the clique co-aligns but qwen's frame affinity is
  not separably lower. Then B44's wrong-donor transfer worked *despite* frame-level alignment
  being uninformative about the island — the deviation lives in the loadings, not the frame
  angles, and that relocation is the finding.
- **`FRAMES_NOT_SHARED`** — the most surprising branch: donors substitute in surgery while
  their raw frames do not co-align. The correction would then be about what the loadings
  express through *any* well-formed frame — a deeper claim than B44 made, reported as such.

## Discipline

CPU, seconds (SVDs of k×k matrices; no fitting). Smoke = one seed, k=20, INVALID-only. Result
`b45_result.json`; scored by `styxx.protocol`; certified + sealed before commit. Full
per-pair, per-seed, per-rank affinity table ships in the result regardless of verdict.
