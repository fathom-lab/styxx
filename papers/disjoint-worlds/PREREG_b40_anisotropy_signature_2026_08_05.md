# PREREG — B40: the anisotropy signature — do the clique minds share a dominant concept subspace that qwen does not?

Fathom Lab · 2026-08-05 · frozen before the scored run. B39 established (by destroying its own
instrument) that **legibility between minds is carried by shared anisotropy** — flatten the
covariance and even mutually legible minds go blind to each other. Five explanatory routes for
the qwen island are dead or invalid, all of them treatments. This run is a **measurement**: is
the island directly visible as a difference in *which concept contrasts dominate* each mind's
geometry?

## The comparable object (dimension-independent by construction)

The four models have different ambient dimensions (3072/2048/2304/1536) — raw feature
subspaces cannot be compared. But every model represents the SAME 462 concepts, so the
eigenvectors of each model's double-centered concept Gram matrix live in the shared
**concept space** (R^462) and are label-aligned across models. The top-k Gram eigenvectors are
"the concept contrasts this mind's geometry is built around." Subspace agreement between two
minds is then the standard **subspace affinity**: aff(A,B) = ||U_Aᵀ U_B||²_F / k over the
top-k eigenvector subspaces, k = 20 (frozen; covers the bulk of variance at the measured
effective dimensionalities of 16–19, `b38_recon_addendum.json`). Computed on the 392 anchor
rows of the committed seed-343 split (the discovery machinery's own fitting surface), all four
models, all 6 unordered pairs.

## Pre-stated predictions

**P1 (the separation):** the clique's internal affinities and qwen's affinities to the clique
are disjoint ranges — min(clique-internal) > max(qwen-to-clique). No magic thresholds: a pure
order statement, falsifiable by a single overlapping pair.

**P2 (the mechanism link):** across the 6 unordered pairs, affinity tracks measured legibility
— Spearman(affinity, b37 discovery averaged over the two directions) ≥ 0.7. Six points is
coarse and is stated as such; with 6 points a perfect ordering gives 1.0, one inversion ~0.77,
so the bar tolerates at most one adjacent inversion. This gate is honest about being blunt.

```gates
{"gates": {"G1_separation": {"metric": "min_clique_internal_affinity_minus_max_qwen_affinity", "op": ">", "value": 0.0},
           "G2_tracks_legibility": {"metric": "spearman_affinity_vs_disc", "op": ">=", "value": 0.7}},
 "outcomes": [{"when": {"G1_separation": true, "G2_tracks_legibility": true}, "verdict": "ISLAND_NAMED__qwen_dominant_subspace_differs_and_affinity_tracks_legibility"},
              {"when": {"G1_separation": true, "G2_tracks_legibility": false}, "verdict": "SUBSPACE_SEPARATES__but_affinity_does_not_track_legibility"},
              {"when": {"G1_separation": false, "G2_tracks_legibility": true}, "verdict": "AFFINITY_TRACKS__but_no_clean_island_separation"},
              {"when": {"G1_separation": false, "G2_tracks_legibility": false}, "verdict": "ANISOTROPY_SIGNATURE_NOT_FOUND__island_cause_still_open"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

**The b37 G2 lesson, applied:** G1 is an order statement over measured quantities (no
noise-passable margin — it demands full range separation, which one bad pair breaks); G2's
bluntness at n=6 is declared. There is no INVALID branch beyond smoke because this run has no
treatment to break and no baseline to fail — it is a measurement over committed data; every
outcome is informative and all four are pre-committed.

## Outcome reading

- **`ISLAND_NAMED`**: the island has a location — qwen builds its geometry around measurably
  different dominant concept contrasts, and that difference tracks who can read whom. The B39
  mechanism (anisotropy carries legibility) gains its signature; the successor is
  representational (WHICH contrasts differ — inspect the top discordant eigenvectors, a
  concept-level story) and a candidate pre-screen ("compute affinity before attempting
  discovery") enters the toolbox.
- **`ANISOTROPY_SIGNATURE_NOT_FOUND`**: the Gram-spectral top-k signature does not carry the
  island — the difference is subtler than dominant-subspace orientation (higher-order,
  local, or nonlinear), and the reported per-pair table bounds where it is not.

## Discipline

CPU-from-cache, seconds of compute (four 392×392 eigendecompositions). Smoke = k=5 on 40
anchors, INVALID-only. Result `b40_result.json`; scored by `styxx.protocol`; certified +
sealed before commit. Affinity is symmetric by construction — no direction claims (the B37
direction-blindness lesson).
