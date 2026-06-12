# FINDING — the cross-model basis CLEARS: cycle-5's miss was a source-whitening artifact (BASIS-CLEARED)

**2026-06-12 · Fathom Lab / styxx. Pre-registered: `PREREG_mapped_whitening_2026_06_12.md` (frozen
pre-run, committed 891b8fa). Receipt: `mapped_whitening_result.json`. Backlog B29. Cycle 5
(`FINDING_truth_danger_basis_2026_06_12.md`) cleared the (truth × danger) basis in gemma and Qwen-3B but
missed the primary Llama-3B by 0.0062 (c_truth_invariant_H = 0.6562 vs the 0.65 ceiling). This tests
whether the miss is a whitening-metric artifact or real geometry — and it is an artifact.**

## Result — re-whiten in the mapped metric and the miss clears, stably

A whitened difference-of-means readout is an LDA-style direction `Σ⁻¹ d`; AUROC depends only on which
covariance Σ supplies the metric. Cycle 5 used gemma's covariance for everything, including reads of
MAPPED Llama points whose covariance is not gemma's (the ridge map distorts it). B29 recomputes the
whitening metric on the mapped anchor distribution (shrunk toward scaled identity, since the anchor count
is far below the hidden dimension) and reads the same factorial through it.

**Port check (only the metric changes):** the source-whitened readout reproduces cycle 5 exactly —
gemma 0.7656 / 0.5347 / 1.0 / 0.5104 and Llama-3.2-3B 0.9288 / **0.6562** / 1.0 / 0.5069. The 0.6562 miss
is reproduced bit-for-bit, so any change below is the whitening metric alone.

**The fix (mapped-space whitening), Llama-3.2-3B:**

| readout | source-whitened (cycle 5) | mapped-whitened (λ=0.5) |
| --- | --- | --- |
| c_truth recovers T | 0.9288 | 0.8351 |
| c_truth invariant to H | **0.6562** (miss) | **0.6059** (clears) |
| c_danger recovers H | 1.0 | 1.0 |
| c_danger invariant to T | 0.5069 | 0.5087 |

`c_truth_invariant_H` across the shrinkage sweep: λ=0.2 → 0.5729, 0.35 → 0.5868, 0.5 → 0.6059,
0.65 → 0.6146, 0.8 → 0.6128 — **all five under the 0.65 ceiling (stability 5/5)**, so the pass does not
hinge on one lucky shrinkage. gemma passes (reproduced); Qwen-3B passes under mapped whitening too
(λ=0.5: 0.7691 / 0.5729 / 1.0 / 0.5122). **Verdict per the frozen gate: BASIS-CLEARED.**

## What this establishes — cycle 5 upgrades to a cleared cross-model basis

The cycle-5 0.0062 miss was a **source-whitening artifact**, not real geometry. The ridge map distorts
the covariance, so whitening mapped points in gemma's metric left a residual anisotropy that bled the
danger factor into the truth coordinate; whitening in the mapped distribution's own metric removes it.
With the right metric the full (truth × danger) coordinate system holds cross-model — gemma, Llama-3.2-3B,
AND Qwen-3B — each coordinate recovering its own factor and blind to the other. The danger axis stays
perfect (1.0) throughout. **The conscience-coordinate basis is real cross-model once each read uses the
metric of the space it reads in.**

## The honest trade and the instrument implication

Mapped-space whitening trades a little on-target discriminability for much better invariance: c_truth
recovers T drops from 0.9288 to 0.8351 (still well clear of 0.75) while its danger-leak drops below the
ceiling. That is the right trade for a basis claim (we want each coordinate blind to the other), and it
is a concrete recipe for `styxx.crossmind`: a CROSS-MODEL read should whiten in the mapped-target
distribution, not the reference distribution — an owed enhancement to the module's `read` path (currently
it whitens in the reference/background metric). The within-model read is unaffected.

## Honest bounds (what is NOT claimed)

Methodological consolidation of a certified arc, not a new axis: the danger axis's 1.0 readout remains an
n=48 existence result, and the whole basis is linear, whitened, last-token, register-bounded, on local
same-cluster models (gemma source; Llama-3.2-3B + Qwen2.5-3B targets). This does NOT touch the
content-vs-value distinction — cross-model CONTENT identity still does not transport (cycle 6
CONTENT-WEAK stands); B29 improves the readout of the VALUE basis, nothing more. The mapped covariance is
a heavily-shrunk estimate (the anchor count is far below the hidden dimension), which is exactly why the
shrinkage sweep + stability requirement were pre-registered; the 5/5 stability is the evidence the result
is not a regularization accident. No model generates; the (false,danger) stimuli are flagged-as-false
probe reads only.
