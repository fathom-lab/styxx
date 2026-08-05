# PREREG — B41: the bridge — does correcting the 20 named contrasts causally restore legibility?

Fathom Lab · 2026-08-05 · frozen before the scored run. B40 named the island: qwen's top-20
dominant concept contrasts (anchor-Gram eigenvectors) differ from every clique member's, with
disjoint affinity ranges and ρ=0.71 tracking of legibility. Naming is correlational. This run
is the causal test: **surgically replace qwen's top-20 concept-contrast pattern with
llama_3b's and re-run label-free discovery.** If the naming is causal, a rank-20 concept-space
correction — twenty directions out of 392 — should disproportionately restore a legibility
that outlier-taming, isotropic noise, full whitening, and per-dim scaling all failed to touch.

## Epistemic status, declared up front

The surgery is **label-aligned** (it uses the true concept correspondence to compare and swap
eigenvector patterns in concept space) — this is an INTERVENTION CEILING in the b31v2 sense,
not a label-free protocol. What stays label-free is everything downstream: the discovery
machinery still receives shuffled rows and must find the correspondence itself. The claim on
success is causal ("the named contrasts are what block discovery"), not deployable
telepathy-without-labels.

## Design (frozen)

Anchor rows of the committed seed-343 split (392 concepts). For source llama_3b and target T:
U_S, U_T = top-20 double-centered anchor-Gram eigenvectors (the B40 objects, 392×20).

**The surgery** (rank-20 concept-space swap): X_T' = (I − U_T U_Tᵀ) X_T + U_S (U_Tᵀ X_T) —
qwen keeps its own per-contrast content loadings but expresses them through llama's dominant
contrast pattern. Then the standard label-free pipeline (seeded row shuffle →
`TransferMap.fit` → assignment) measures discovery seed_acc against truth.

**Arms:**
- A0 baseline: llama_3b → qwen, no surgery (must reproduce ~0.05).
- A1 bridge: llama_3b → qwen', surgery with U_S = llama's.
- A2 specificity null: llama_3b → qwen'', surgery with U_S replaced by a random orthonormal
  392×20 frame (QR of seeded Gaussian) — same rank, same machinery, wrong directions.
- A3 no-harm control: llama_3b → gemma', the identical surgery applied to gemma — the bridge
  recipe must not damage an already-legible pair (gemma keeps ≥ 0.30).

## Gates (frozen; scored by styxx.protocol)

```gates
{"gates": {"G0_baseline": {"metric": "a0_baseline_disc", "op": "<=", "value": 0.15},
           "G3_no_harm": {"metric": "a3_gemma_bridged_disc", "op": ">=", "value": 0.30},
           "G1_bridge": {"metric": "a1_bridge_disc", "op": ">=", "value": 0.30},
           "G2_specificity": {"metric": "a2_random_frame_disc", "op": "<=", "value": 0.15}},
 "outcomes": [{"when": {"G0_baseline": false}, "verdict": "INVALID__baseline_not_reproduced"},
              {"when": {"G0_baseline": true, "G3_no_harm": false}, "verdict": "INVALID__surgery_breaks_legible_pairs"},
              {"when": {"G0_baseline": true, "G3_no_harm": true, "G1_bridge": true, "G2_specificity": true}, "verdict": "BRIDGE_BUILT__named_contrasts_causally_block_legibility"},
              {"when": {"G0_baseline": true, "G3_no_harm": true, "G1_bridge": true, "G2_specificity": false}, "verdict": "INVALID__any_rank20_surgery_rescues__not_specific"},
              {"when": {"G0_baseline": true, "G3_no_harm": true, "G1_bridge": false, "G2_specificity": true}, "verdict": "BRIDGE_FAILS__naming_not_causal_at_rank20"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

## Outcome reading, pre-committed

- **`BRIDGE_BUILT`**: the B40 naming is causal — twenty concept-contrast directions are the
  barrier, and correcting them alone un-islands qwen where five broader interventions failed.
  First engineered legibility bridge between an illegible mind and the clique; the successor
  quantifies the dose curve (how few directions suffice?) and the concept-level identity of
  the blocking contrasts.
- **`BRIDGE_FAILS`**: the named signature is a correlate, not the cause — the block lives
  deeper than the top-20 contrast pattern (residual spectrum, local structure). The naming
  demotes to a marker and the arc says so.
- The G2-fail branch is INVALID by design: if random directions rescue too, the surgery
  machinery itself manufactures legibility and nothing is licensed (the B35-c null lesson,
  applied in advance).

## Discipline

CPU-from-cache, ~5 discovery fits. Smoke = k=5 at 40 anchors, INVALID-only. Result
`b41_result.json`; scored by `styxx.protocol`; certified + sealed before commit.
