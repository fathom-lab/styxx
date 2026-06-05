# RESULT — under nonlinear warp, causal transfer is RECOVERY-BOUND, not transfer-bound (a refinement, with one strong claim bounded)

**Date:** 2026-06-05 · **Verdict: RECOVERY-BOUND.** The causal-transfer result refines, it does not fully
survive and does not fully fall. Causal *transferability* is **warp-robust** (the meaning rides through a
genuine nonlinear warp given correspondence); the *unsupervised, zero-pairs* version is **bounded to
near-rotation geometry** (correspondence recovery fails under warp). Frozen:
`PREREG_warp_crossworld_2026_06_05.md` (+ a true-correspondence diagnostic added after the first smoke
showed collapse, to disentangle recovery-failure from transfer-failure). 3 seeds.

## Numbers — diagonal advantage vs warp strength α
| α | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---:|---:|---:|---:|---:|
| RSA-sharing | 0.97 | 0.95 | 0.86 | 0.79 | 0.79 |
| recovery (unsup top-1) | 0.18 | 0.12 | 0.04 | 0.04 | 0.02 |
| **UNSUP** transfer (Procrustes) | 0.62 | 0.28 | 0.25 | 0.30 | **0.15** |
| **UNSUP** transfer (affine) | 0.62 | 0.34 | 0.29 | 0.32 | 0.18 |
| **TRUE-corr** transfer (Procrustes) | 0.97 | 0.96 | 0.92 | 0.88 | **0.88** |
| **TRUE-corr** transfer (affine) | 0.99 | 0.98 | 0.97 | 0.97 | **0.97** |
| null | 0.00 | 0.09 | −0.06 | 0.10 | 0.09 |

## What it shows (read straight — a refinement, both directions stated)
**The part that SURVIVES and even strengthens: causal transferability is warp-robust.** Given the
correspondence, attribute directions transfer through a *full nonlinear warp* almost undiminished — the
affine map holds at **0.97** across every warp level (0.99 → 0.97), the orthogonal map at 0.88. The
attribute/causal structure of meaning is not a rotation artifact: it persists under genuine warping. This
extends the causal claim *beyond* the rotation regime — stronger than the original linear result showed.

**The part that is BOUNDED (a real limit on the prior result's strongest reading): the unsupervised,
zero-pairs pipeline does NOT survive warp.** Unsupervised correspondence recovery collapses (0.18 → 0.02 —
GW cannot match differently-warped geometries), so the zero-pairs transfer degrades from 0.62 to **0.15
(≈ the null 0.09)** by full warp, sliding below the pre-registered P1 bar by α=0.25 already. **So the
strongest reading of `RESULT_causal_crossworld` — "recoverable with zero shared data, even across warps" —
is bounded to near-rotation geometry.** Reported as a genuine limit, not softened.

**Why it's RECOVERY-bound and not TRANSFER-bound (the diagnostic).** RSA-sharing survives the warp (0.79 at
α=1) and true-correspondence transfer survives (0.88–0.97). So the geometry is still shared and still
alignable — what fails is *unsupervised recovery of which concept is which* under warp, not the
transferability of meaning. The matcher is the bottleneck, not the geometry.

## The unifying theme across both experiments
Causal/attribute **transfer, given correspondence, is robust** — to imperfect recovery (causal arm) and now
to nonlinear warping (this arm). **Recovering the correspondence unsupervised is the fragile step** — a
sharp ~0.6-ceiling threshold under rotation, and near-total failure under warp. *What the structure means
transfers easily; finding which instance is which, with zero pairs, is the hard, geometry-sensitive part.*
This is the honest shape of "universal forms recoverable": the meaning is universal and transferable; the
zero-supervision recovery of the mapping is bounded.

## Pre-registration accounting
- **P1 (unsupervised Procrustes survives, ≥0.30 & beats null by 0.30): FAILS at α=1** (0.15 vs null 0.09) —
  and already by α=0.25. The unsupervised-under-warp claim is not met. Honest.
- **true-correspondence diagnostic (added post-first-smoke): transfer HOLDS** (0.88–0.97) → locates the
  failure as recovery, not transfer. **P3 (RSA survives): holds** (0.79). The frozen "ROTATION-BOUND /
  self-falsified" branch is half-right (the *unsupervised* pipeline is rotation-bound) but its "transfer-
  bound" premise is wrong (transfer survives with correspondence) — hence the nuanced RECOVERY-BOUND reading.

## Honest bounds & next
Synthetic; one warp family (random MLP); linear attribute directions. The robust, load-bearing findings: the
**warp-flat true-correspondence transfer** (meaning is warp-robust) and the **unsupervised collapse**
(zero-pairs recovery is rotation-bound). Real-model embeddings are the downstream test — and the practical
implication for cross-model work (styxx, vec2vec, cross-model council): aligning two models' *meaning* is
easy once you have any anchors; doing it with *zero* anchors needs near-isometric geometry, which real
differently-trained models may or may not have. That is the real open question this points to.
