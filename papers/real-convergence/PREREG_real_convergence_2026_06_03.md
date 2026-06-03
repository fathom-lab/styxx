# PREREG — Do REAL LLMs converge to the same concept geometry?

**Date:** 2026-06-03 · **Status:** PRE-REGISTERED (gate frozen before data).
**The non-circular, real-meaning version** of the geometry experiment — built to answer the
audit's central criticism (the synthetic Disjoint-Worlds test passed the same latent `z` to
both worlds, making convergence near-tautological). Here nothing is built in: **6 genuinely
different LLMs**, different architectures and training data, trained by different orgs, are
asked whether they represent **real concepts** with the **same geometry.**

## Setup (frozen)

- **96 real concepts**, 8 clear semantic categories × 12 (animals, fruits, vehicles,
  professions, body parts, weather, furniture, instruments) — `concept_set` in the runner.
- **6 models** (independent architectures/training): Qwen2.5-1.5B, Qwen2.5-3B,
  Llama-3.2-1B, Llama-3.2-3B, Phi-3.5-mini, gemma-2-2b.
- **Representation:** mean-pooled hidden state of each word's tokens at a fixed relative depth
  (~0.66 × n_layers). Per model → a 96×d representation, then a 96×96 distance matrix.
- **Metric:** cross-model **RSA** = Pearson correlation of two models' distance matrices over
  concept pairs (concepts are identical words → correspondence is trivial; this is the
  honestly-scoped CKA/RSA-style measure, NOT unsupervised translation).

## KILL-GATE (frozen)

- **CONVERGENT:** mean cross-model RSA **≥ 0.30** AND **≥ 5×** the shuffled-concept control
  (which must sit ≈ 0). → independently-trained real models DO develop correlated concept
  geometry on real meaning — the non-circular version of structure-driven convergence.
- **CROSS-FAMILY (stronger):** the convergence holds for **cross-architecture** pairs (Qwen↔
  Llama↔Gemma↔Phi), not just same-family — reported separately.
- **NOT CONVERGENT:** mean cross-model RSA ≈ control ≈ 0 → real models do NOT share concept
  geometry; the convergence literature (CKA, vec2vec) would not replicate on our zoo.

**Sanity:** shuffled-concept control must be ≈ 0; within-category distances must be < across-
category (the geometry must actually encode the semantic categories, or the test is moot).

## Honest framing / caveats (frozen)

- This **reproduces an established phenomenon** (cross-model representational convergence —
  CKA Kornblith 2019; vec2vec 2025) on our own model zoo. It is **not novel**; it is the
  **non-circular replacement** for the synthetic toy the audit (correctly) flagged. Its value
  is that our geometry claim now rests on *real models + real concepts* rather than a
  hand-passed latent.
- RSA uses the trivial word↔word correspondence (same concept = same word); it is a
  *similarity* measure, not unsupervised translation. Scoped accordingly.
- Single mean-pool layer, single representation choice; 96 concepts. Honest prior:
  CONVERGENT (the literature predicts it), with cross-family the load-bearing check.

— frozen 2026-06-03
