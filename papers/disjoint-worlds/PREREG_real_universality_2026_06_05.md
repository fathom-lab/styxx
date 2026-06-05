# PREREG — are real differently-trained models near-isometric enough for ZERO-ANCHOR alignment?

**Date:** 2026-06-05. The real-model frontier raised by the warp result. Synthetic showed: shared geometry
+ near-isometry → zero-anchor alignment works; shared geometry alone (warped) → recovery collapses. Question:
where do REAL differently-trained models fall? Five models on DIFFERENT training data — distilgpt2 (WebText),
pythia-410m (the Pile), Qwen2.5-0.5B, opt-125m, bloom-560m (ROOTS) — templated concept reps (100 concrete
nouns, 6 context templates, last-token), pairwise RSA + unsupervised GW recovery.

**Transparency:** a 2-model smoke (distilgpt2↔pythia-160m, 30 concepts) already ran: RSA 0.76, zero-anchor
recovery 0.10 (chance 0.033). So this is a CONFIRMATION + MAPPING run, not blind. The prediction below is
grounded in the synthetic warp result AND that smoke.

## Prediction (frozen)
- **P1:** real differently-trained models SHARE concept geometry strongly (mean pairwise RSA ≥ 0.40).
- **P2 (the frontier):** zero-anchor unsupervised recovery is LOW (max pairwise recovery < 0.30, most ~chance)
  — shared but NOT near-isometric enough for zero-pair alignment. This would reproduce the warp regime on
  real brains: cross-model meaning needs anchors; pure zero-anchor recovery is fragile on real decoder LMs.
- **Mapping:** report which pairs (if any) clear the near-isometry bar; note scale/family dependence.

## Decision rule
- **SHARED-BUT-ANCHOR-BOUND** (P1 ∧ P2): the practical answer — real models share meaning geometry but
  zero-anchor (vec2vec-style) alignment does not recover correspondence for small decoder LMs; anchors or
  near-isometric (embedding-trained / scaled) models are required. Reproduces the synthetic warp boundary.
- **NEAR-ISOMETRIC** (some pairs recovery ≥ 0.30): zero-anchor alignment works on real brains for those pairs.
- Report RSA + recovery for every pair regardless.

## Honest scope (frozen)
Real models all saw English/web text → shared geometry does NOT isolate data-independence (the synthetic
disjoint-worlds controlled that; this cannot). RSA is also susceptible to lexical/frequency inflation
(prior real-convergence work showed the convergence is semantic-not-lexical only with controls) — so the
**load-bearing metric is RECOVERY (near-isometry), not RSA magnitude.** Last-token templated reps, final
layer; layer/pooling choices are a known lever. This answers the PRACTICAL frontier (does zero-anchor
alignment work on real brains), not the metaphysical one.
