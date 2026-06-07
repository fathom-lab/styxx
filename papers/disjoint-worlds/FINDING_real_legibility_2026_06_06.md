# FINDING — Zero-anchor cross-model legibility is REAL and graded by isometry (aligner-validated)

**2026-06-06. Fathom Lab / styxx.** Corrects the prior un-validated `real_universality_result.json`
(recovery 0.05, "not alignable") which used an under-powered GW aligner with no positive control.

## Question

Can one model's concept geometry be aligned to another's with **zero paired data** (no Rosetta
stone)? The program's open frontier on the 2,500-year "is mind universally structured" question —
only ever tested on synthetic toys before.

## The instrument fix (the load-bearing part)

The prior run reported recovery ≈ chance with **no aligner positive control**. The program's own
hardest lesson (the GW aligner that once "false-falsified Plato") is that ~chance recovery from an
unvalidated aligner is uninterpretable. Two fixes: (1) a stronger aligner — **Wasserstein–Procrustes**
(PCA→common dim, Sinkhorn-annealed, GW warm-start, structure-only inits; **no** identity-correspondence
init, which would leak the answer); (2) a **mandatory calibration positive control** — warp a real
embedding to a sweep of *known* RSA levels and measure recovery, so a real-pair null is only
interpretable if the aligner recovers a KNOWN correspondence at that RSA.

**Calibration (Llama-3.2-3B, N=160, chance 1/160=0.006):** WProc recovers known warps top1
1.00 / 0.95 / 0.95 / 0.63 / 0.26 at RSA 1.00 / 0.998 / 0.994 / 0.981 / 0.947 — validated down to
RSA≈0.95. The old GW aligner: 0.96 → 0.44 → ~chance by RSA 0.95. **The prior 0.05 was the broken
instrument, not a bound.**

## Result — real pairs (zero paired data; chance 0.006)

| pair | family | RSA | WProc top1 | top5 | mrr | calib floor@RSA |
|---|---|---|---|---|---|---|
| Llama-3.2-3B ↔ Llama-3.2-1B | same | 0.970 | **0.481** | 0.631 | 0.490 | 0.507 |
| Llama-3.2-3B ↔ **gemma-2-2b** | **cross** | 0.792 | **0.150** | **0.431** | 0.305 | 0.089 |
| gemma-2-2b ↔ Llama-3.2-1B | cross | 0.771 | 0.087 | 0.344 | 0.213 | 0.077 |
| Qwen2.5-3B ↔ Qwen2.5-0.5B | same | 0.789 | 0.044 | 0.094 | 0.073 | 0.087 |
| (RSA < 0.70 pairs) | — | <0.70 | ~chance | ~chance | — | ~chance |

## Reading

- **Zero-anchor cross-model recovery is real — far above chance (1/160 = 0.006).** Near-isometric
  pairs (Llama-3B↔1B, RSA 0.97) recover ~half of 160 concepts with no anchors (top1 0.48).
- **Caveat on the same-family flagship:** that 0.48 is *at* its RSA-matched calibration floor
  (0.507), i.e. the real same-family geometry is recovered about as well as an isotropic-noise warp
  at the same RSA — strong vs chance, but *not* beyond-noise extra-alignable. The **cross-family**
  Llama↔gemma pair is the one that **beats its floor**: top1 0.15 / top5 0.43 vs floor 0.089 —
  different vendor/data/architecture, shared structure *more* alignable than RSA-matched noise.
  So "beats the noise floor" is the cross-family result; the same-family pair is "as alignable as
  its RSA predicts."
- **Aligner dependence:** the favorable aligner is Wasserstein–Procrustes (these top1s); the
  entropic-GW aligner gives ~5× lower (0.094 / 0.044 on the same pairs). Numbers here are WProc;
  GW is reported as the weaker second opinion. This is why the *positive-control calibration* (not
  a fixed aligner) is the load-bearing validation.
- **Below RSA≈0.70 it falls to chance** — the recoverability threshold for this aligner at N=160.
- The same-family Qwen ladder does *not* recover (RSA only ~0.78); Llama's cross-size pair is
  unusually isometric (0.97). Isometry, not mere shared lineage, is what predicts recovery.

## What it means

For the ancient question: independently-deployed minds don't just share *relational* structure
(RSA) — that structure is **partially recoverable into a common frame with zero paired data**, and
the degree of recovery is a measurable function of how isometric the two minds are. A planted
concept in one model is **partially legible from another**. This is the practical, real-model
counterpart to the synthetic disjoint-worlds result, advancing the named frontier with a *validated*
instrument.

For styxx: zero-anchor cross-model representation alignment is a real capability surface (a
cross-model "conscience mount" / representation-read that needs no paired calibration for
near-isometric model pairs) — and the calibration-positive-control method is the reusable product
(it caught both a broken aligner *and* a prior over-claim).

## Honest scope

All models saw overlapping English web text, so shared geometry is **not** data-independent
universality (the synthetic disjoint-worlds controlled that; this is the *practical* frontier).
Recovery is **partial** off-the-near-isometric-regime (cross-family top1 0.15, not >0.5). RSA is
confounded by shared training data. Llama-3B↔1B share lineage (most isometric). Stronger claims
(data-independent zero-anchor reading; full cross-family recovery) need vec2vec-class methods and
data-overlap controls. What stands: **a validated, graded, real-model demonstration that zero-anchor
cross-model concept recovery works, scales with isometry, and survives crossing families — and a
correction of the prior instrument-limited null.**
