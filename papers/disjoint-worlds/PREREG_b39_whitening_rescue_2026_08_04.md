# PREREG — B39: naming the island — can whitening rescue qwen's legibility?

Fathom Lab · 2026-08-04 · frozen before the scored run.

**Provenance note (the collision, disclosed):** a parallel Fathom process drafted this design
as `PREREG_b38_island_cause_2026_08_04.md` and was cut off mid-sentence (the draft is
truncated at its Design section, has no gates block, and was frozen-by-accident inside commit
`05545bb`; it is left byte-intact as a draft). The same numeral B38 was independently used by
the noise-dose cliff run (INVALID, cycle 117). This document completes the draft's design as
**B39**, credits the whitening-rescue idea to that draft, and is the runnable freeze.

## Standing evidence (all committed)

Three island routes are dead: outlier dimensions and gross relational dissimilarity
(`b38_recon_addendum.json` — qwen is internally healthy: lowest top-dim variance share 0.007,
highest effective dimensionality 18.7, normal distance-CV), and isotropic alignment-matching
(`b38_result.json` — max-dose noise cannot reach qwen's RSA and gemma stays 30× legible).
The island is a functioning space that no one can match.

## The causal candidate (from committed machinery — the draft's idea)

B28/B29 established that apparent cross-model structure failures can be **covariance
artifacts**: ZCA-whitening resolved axis entanglement (off-diagonals to chance, diagonals
intact) and mapped-space shrunk whitening cleared the cross-model basis (`zca_shrink` ships in
`styxx.crossmind` with the B29-validated shrink λ=0.5). If qwen's illegibility lives in its
**feature-covariance reweighting of the distances that GW and Procrustes consume**, then
running discovery in per-model ZCA-shrunk-whitened spaces should rescue it. If whitening does
not rescue, the surviving explanation is that qwen's concepts are **intrinsically differently
arranged** — a difference in kind that no linear reweighting removes.

## Design (frozen)

Treatments applied per-model to the concept points (anchors + held-outs) BEFORE the standard
pipeline; everything downstream is the b34-v3 machinery verbatim (seed 343 split, shuffle
rail, `TransferMap.fit` discovery, seed_acc vs truth):

- **T0 raw** — baseline (must reproduce B37).
- **T1 zca** — per-model ZCA-shrink whitening (`styxx.crossmind.zca_shrink`, λ=0.5, fit on
  the 392 anchor rows, applied to all 462).
- **T2 diag** — per-dimension standardization (mean/std per dim) — the diagonal-only control
  arm: if T2 rescues as much as T1, the cause is scale, not correlation structure.

Cells per treatment: llama_3b→qwen_1p5b (the rescue target), llama_3b→gemma_2b (the
must-not-break sanity), qwen_1p5b→llama_3b (island, other direction — reported). 9 discovery
fits, CPU-from-cache. Chance for seed_acc ≈ 1/392; no null arm is needed for an assignment
accuracy scored against ground truth.

## Gates (frozen; scored by styxx.protocol)

```gates
{"gates": {"G0_baseline": {"metric": "t0_llama3b_to_gemma", "op": ">=", "value": 0.30},
           "G0b_whiten_preserves_clique": {"metric": "t1_llama3b_to_gemma", "op": ">=", "value": 0.30},
           "G1_rescue": {"metric": "t1_llama3b_to_qwen", "op": ">=", "value": 0.30},
           "G2_no_rescue": {"metric": "t1_llama3b_to_qwen", "op": "<=", "value": 0.15}},
 "outcomes": [{"when": {"G0_baseline": false}, "verdict": "INVALID__baseline_not_reproduced"},
              {"when": {"G0_baseline": true, "G0b_whiten_preserves_clique": false}, "verdict": "INVALID__whitening_breaks_the_clique"},
              {"when": {"G0_baseline": true, "G0b_whiten_preserves_clique": true, "G1_rescue": true}, "verdict": "ISLAND_CAUSE_COVARIANCE__whitening_rescues"},
              {"when": {"G0_baseline": true, "G0b_whiten_preserves_clique": true, "G1_rescue": false, "G2_no_rescue": true}, "verdict": "ISLAND_INTRINSIC_ARRANGEMENT__no_linear_reweighting_rescues"},
              {"when": {"G0_baseline": true, "G0b_whiten_preserves_clique": true, "G1_rescue": false, "G2_no_rescue": false}, "verdict": "PARTIAL_RESCUE__between_bars"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

T2's role is interpretive, reported not gated: on `ISLAND_CAUSE_COVARIANCE`, T2-vs-T1 says
whether scale alone suffices; on no-rescue, T2 corroborates. The qwen→llama_3b direction is
reported for symmetry context (the B37 lesson: this machinery is direction-blind up to
truncation, so no symmetry gate is posed).

## Outcome reading, pre-committed

- **`ISLAND_CAUSE_COVARIANCE`**: the island was a linear artifact all along — qwen's concept
  arrangement matches the clique once its covariance is equalized. The legibility clique
  extends to qwen under a whitened metric; `read_cross_model`'s whitened read-path becomes
  the canonical discovery recipe, and the B37 matrix gains a "whitened" column as a successor.
- **`ISLAND_INTRINSIC_ARRANGEMENT`**: no linear reweighting (full or diagonal) rescues —
  qwen's concepts are genuinely arranged differently. The island is a difference in kind at
  the relational level, and the successor is representational (what does qwen's training do
  differently), not metric.

## Discipline

CPU-from-cache, zero model loads. Smoke = T0/T1 on one pair at 40/10, INVALID-only. Result
`b39_result.json`; scored by `styxx.protocol`; certified + sealed before commit.
