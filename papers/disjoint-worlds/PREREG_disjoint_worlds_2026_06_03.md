# PREREG — Disjoint-Worlds: is the geometry of meaning universal structure, or shared data?

**Date:** 2026-06-03
**Status:** PRE-REGISTERED (gate frozen before any data).
**The decisive experiment of the geometry sub-question** (Half A of the ancient-question
program — the modern, testable form of Plato's "universal forms"). Cross-model
representational convergence is established (stitching, CKA, vec2vec). The unresolved crux:
is the shared geometry a structure of **meaning/the world** (recoverable from shared
*structure* with **zero shared data**), or an artifact of **shared training data + biases**?

## The idea

Build **two synthetic worlds with IDENTICAL latent relational structure but DISJOINT
surface tokens and INDEPENDENTLY-sampled corpora** — *zero shared data by construction*.
Train an embedding model on each, independently. Then attempt **unsupervised** alignment
of the two learned geometries (no paired examples) and measure whether it recovers the
hidden token correspondence. If geometry is *structure-determined* (Platonic), alignment
succeeds with zero shared data; if it's a data artifact, it fails.

## Worlds (frozen)

- **N = 100 concepts** with a true latent geometry **z ∈ R^8** (random Gaussian). Concept
  co-occurrence ∝ `exp(-||z_i − z_j||² / τ)` → a corpus of co-occurring pairs.
- **World A:** latent z, corpus_A (independent sample), tokens a_1..a_N.
- **World B (same-structure):** the **same** latent z, **independent** corpus_B, **disjoint**
  tokens b_1..b_N. Ground-truth correspondence a_i ↔ b_i is *hidden* from the aligner.
- **World B′ (control, different-structure):** a **different** random latent z′, disjoint
  tokens. Same-method embedding.

## Models (frozen)

Independently-trained **skip-gram (negative-sampling) embeddings** per world, **different
embedding dims for A vs B** (e.g., 32 vs 24) so the two are not trivially identical — only
the *structure* is shared, not the frame, dim, corpus, or tokens.

## Alignment (frozen)

**Unsupervised Gromov-Wasserstein** (entropic, numpy) on the two embeddings' intra-world
distance matrices D_A, D_B — GW aligns metric spaces by internal structure, with **no
correspondence and no shared dimension**. Recover correspondence = argmax of the coupling;
refine the hard assignment with the Hungarian algorithm. Multiple random initializations,
keep the lowest-GW-cost solution. Metric: **top-1 correspondence accuracy** (chance = 1/N
= 0.01).

## KILL-GATE (frozen)

- **UNIVERSAL (structure determines geometry):** same-structure top-1 accuracy **≥ 0.50**
  (≥ 50×... chance) AND **≥ 5×** the different-structure control accuracy, robust across ≥3
  world seeds. → the geometry of meaning is recoverable from shared *structure* with **zero
  shared data** → universal forms supported in the only testable sense.
- **ARTIFACT (negative):** same-structure accuracy ≈ control ≈ chance → independent models on
  disjoint data do **not** converge to an alignable geometry → the strong Platonic reading is
  falsified; observed convergence elsewhere is shared-data/biases.
- **PARTIAL:** same-structure clearly beats control and chance but < 0.50 → geometry is
  *partly* structure-determined; report the degree (how much of meaning is universal vs
  idiosyncratic).

**Sanity controls (must pass for the run to count):** (i) aligning a world to **itself**
with permuted tokens recovers ~100% (the aligner works); (ii) the different-structure
control sits near chance (no spurious alignment).

## Honest caveats (frozen)

- Synthetic worlds, skip-gram embeddings, N=100, GW is non-convex (multi-init). Tests the
  *in-principle* claim (does shared structure determine geometry with zero shared data) —
  the cleanest decisive form. Real cross-modal / cross-substrate (LLM↔brain) is the larger
  follow-up.
- Honest prior: same-structure *should* align (geometry is structure-determined and skip-gram
  recovers it) — so the load-bearing measurements are the **degree** (top-1 accuracy: 0.5?
  0.9?) and robustness to **different dims/independent corpora**, plus the control failing.
  A weak same-structure result (poor alignment despite shared structure) would itself be the
  surprising, informative outcome.

— frozen 2026-06-03
