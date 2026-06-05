# PREREG — does causal transfer survive NONLINEAR WARPING? (the adversarial test of my own causal result)

**Date (frozen before data):** 2026-06-05. The causal cross-world result (`RESULT_causal_crossworld`) has one
honest weakness, which I flagged: the worlds were related by a **rotation** of shared latent z (linear
embedding), so "attribute directions transfer through the recovered rotation" is partly built in. This goes
straight at it: make each world apply a **different nonlinear warp** to the shared structure — geometry
becomes a *warp*, not a rotation — and test whether causal transfer survives. This is self-falsification:
either the result strengthens (transfer survives genuine warping) or it breaks (it was rotation-dependent —
a correction I will report as loudly as the positive).

## Method (frozen)
- Shared latent z (N=100, K=8 attributes, distinctive/anisotropic — the regime transfer worked in).
- **Per-world nonlinear warp:** `E = normalize( (1-α)·(zR) + α·g(zR) ) + σ·noise`, where g is a per-world
  random 2-layer MLP (tanh), R orthonormal (diff dims D_A=32, D_B=24), σ small (high faithfulness). **Warp
  strength α swept {0.0, 0.25, 0.5, 0.75, 1.0}** — α=0 reproduces the linear result, α=1 is a full nonlinear
  warp, different per world. 3 seeds.
- Pipeline (unchanged): unsupervised GW correspondence (zero pairs) → two maps: **(a) orthogonal Procrustes**
  (rotation-only, the original) and **(b) a learned LINEAR map W** (least-squares on recovered pairs — can
  capture affine/shear an orthogonal map can't). Causal transfer = diagonal advantage of the attribute
  transfer matrix; shuffled-map null; true-correspondence ceiling.

## Hypotheses & predictions (frozen)
- **P1 — survival under warp (does my result hold?):** orthogonal-Procrustes diag advantage at α=1 stays
  ≥ **0.30** and beats the null by ≥ 0.30. If it COLLAPSES toward null as α→1, the causal-transfer result was
  rotation-dependent (construction artifact) — reported as a self-falsification.
- **P2 — richer-map rescue:** if Procrustes collapses under warp, does the learned linear map W rescue
  transfer (diag advantage_W ≥ Procrustes + 0.15 at high α)? Tests whether the limit is the *geometry* (no
  shared structure survives) or the *aligner* (rotation-only is too weak for warped geometry).
- **P3 — geometry sharing under warp:** does RSA-sharing itself survive (stay > 0.3) as α→1, or do the
  worlds diverge into uncorrelated geometries?

## Decision rule (frozen)
- **TRANSFER SURVIVES WARP** iff P1 (or P2) holds at α=1 — causal meaning transfers across genuinely warped,
  zero-shared-data worlds: the strong claim, no longer rotation-bound.
- **ROTATION-BOUND / SELF-FALSIFIED** iff transfer collapses with warp under both maps while RSA stays high —
  the geometry is shared but the *causal direction* transfer required a rotation relationship; the linear
  result does not generalize to warps. An honest correction to the prior result, reported plainly.
- **GEOMETRY DIVERGES** iff even RSA collapses (P3 fails) — different nonlinear warps destroy shared geometry
  entirely; the question is ill-posed past mild warps.
- Report diag-advantage(α) for both maps, RSA(α), recovery(α), and the null/ceiling regardless.

## Caveats (frozen)
- Still synthetic; warp = random MLP (one family of nonlinearity); attribute directions linear; matched
  faithfulness. This tests whether the causal-transfer result is an artifact of the rotation construction —
  the single most important robustness check on the prior finding. Real-model embeddings remain downstream.
