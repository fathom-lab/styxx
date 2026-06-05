# RESULT — real models robustly SHARE meaning geometry but NONE is zero-anchor (GW) recoverable; my embedder-dividing-line prediction is FALSIFIED

**Date:** 2026-06-05 · **Verdict: SHARED-BUT-NOT-GW-RECOVERABLE, universally.** Real differently-trained
models share concept geometry strongly, but unsupervised Gromov-Wasserstein recovery fails on *every* pair —
decoders, embedders, same-family, cross-family alike. The synthetic warp/isotropic regime (high RSA, ~chance
recovery) reproduces on real brains. Frozen: `PREREG_real_universality_2026_06_05.md`. 8 models, 100
templated concepts.

## Numbers — RSA (sharing) and zero-anchor recovery, by pair kind
| kind | RSA range | recovery range | max recovery |
|---|---|---|---|
| decoder ↔ decoder (5 LMs, diff data) | 0.48 – 0.88 | 0.00 – 0.05 | 0.05 |
| embedder ↔ embedder (MiniLM/mpnet/BGE) | 0.77 – 0.79 | 0.00 – 0.05 | 0.05 |
| decoder ↔ embedder (mixed) | 0.40 – 0.78 | 0.00 – 0.04 | 0.04 |
| **all** | mean **0.69** | ~chance | **0.05** (chance 0.01) |

## What it shows (read straight, including the falsified prediction)
**P1 CONFIRMED — real models robustly share concept geometry.** Every pair of differently-trained models —
GPT-2 (WebText), Pythia (the Pile), Qwen, and three sentence-embedders — shares concept geometry, RSA mean
**0.69**, up to 0.88 (same-family) and still 0.48–0.56 cross-family (GPT-2↔Qwen). Shared geometry of meaning
is real and universal on real brains (modulo the shared-data + lexical caveats below).

**P2 CONFIRMED — zero-anchor recovery fails, and it's DECOUPLED from RSA.** No pair clears chance on
unsupervised GW correspondence recovery (max 0.05 vs chance 0.01). Same-family pairs with RSA 0.85–0.88
recover no better than cross-family pairs at RSA 0.50. High sharing does **not** yield recoverability —
exactly the synthetic dissociation (isotropic RSA 0.96 → recovery 0.10). Identity recovery is the wall, and
on real models the simple geometric matcher hits it everywhere.

**MY PREDICTION FALSIFIED — there is no embedder-vs-decoder dividing line.** I predicted meaning-trained
embedders would be near-isometric (recover) where next-token decoders are not. **They are not:** emb-emb
recovery (0.05) = dec-dec (0.05) = chance. Meaning-trained models share geometry *more cleanly* (RSA ~0.78
among embedders) but are **no more GW-recoverable** than decoders. The bottleneck is not the model's training
objective — it is that **simple geometric (RDM) matching is insufficient for real-model zero-anchor
alignment across the board.** Reported as a clean falsification of my own hypothesis.

## The crucial honest bound — this does NOT contradict vec2vec
Published vec2vec achieves zero-anchor alignment on real embedding models (cosine 0.92) — using a **learned
adversarial/cycle-consistent NONLINEAR translator** over the full distribution, not GW on a 100-concept RDM.
So the honest claim is **method-bounded:** *simple geometric matching (GW on RDMs) cannot recover real-model
correspondence — even for embedders that a stronger learned translator can align.* The shared geometry is
real but not in a form GW exploits; recovering it needs a learned nonlinear map, not distance-matrix
matching. Our result locates the wall for the *simple* method and is fully consistent with vec2vec clearing
it with a *stronger* one.

## The arc, unified (synthetic → real)
1. The geometry of meaning is **shared** across minds — zero-shared-data synthetic worlds (RSA 0.42) and
   real differently-trained models (RSA 0.69). Universal, robust.
2. Causal/attribute meaning **transfers given correspondence** — robust to imperfect recovery and to nonlinear
   warp (synthetic, true-corr 0.97).
3. **Recovering the correspondence unsupervised is the hard, fragile, method-sensitive step** — a sharp
   near-isometry threshold synthetically; total failure of simple GW matching on real models (decoders AND
   embedders); cleared only by stronger learned translators (vec2vec, on embedders).

**"Identity is harder than meaning" — confirmed on real brains.** Meaning is shared and transferable; finding
which concept is which, blind, is the wall — and on real models, simple geometry can't scale it.

## Honest scope
Real models all saw English/web text → shared geometry does not isolate data-independence (the synthetic
disjoint-worlds did). RSA is susceptible to lexical inflation; the load-bearing metric was recovery, and it
is floored. One aligner (GW), one rep (templated last-token / mean-pool, final layer), 100 concepts. The open
frontier is now precise: a learned nonlinear translator (vec2vec-style) on these same real models — does it
clear the wall GW cannot, and for which model types? That is the next decisive run.
