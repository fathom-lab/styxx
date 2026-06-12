# PRE-REGISTRATION — B29: does MAPPED-space whitening (with shrinkage) clear cycle-5's 0.0062 cross-model basis miss? (frozen)

**2026-06-12 · Fathom Lab / styxx. Frozen before any score is seen. Runner: `run_mapped_whitening.py`
(SEED=0). Receipt: `mapped_whitening_result.json`. Backlog B29. Cycle 5
(`FINDING_truth_danger_basis_2026_06_12.md`, PARTIAL-STRUCTURED) cleared the (truth × danger) basis in
gemma and Qwen-3B but missed on the primary Llama-3B by a whisker: c_truth_invariant_H = 0.6562 vs the
0.65 ceiling, by 0.0062. Cycle 5 whitened only in the SOURCE (gemma) space and read MAPPED Llama states
through gemma's covariance. Hypothesis: the ridge map distorts covariance, so gemma's whitening
mis-calibrates mapped points, leaving a residual anisotropy that is the 0.0062 leak. This tests whether
whitening in the MAPPED distribution's own covariance clears it — or confirms the miss as real geometry.**

## The mechanism under test

A whitened difference-of-means readout is equivalent to an LDA-style direction `Σ⁻¹ d` (d = raw DiM,
Σ = the whitening covariance); AUROC depends only on the direction, not the mean offset. Cycle 5 used
Σ = gemma's training covariance for ALL reads, including reads of MAPPED Llama points whose covariance is
NOT gemma's (the label-free ridge map shrinks/rotates it). B29 recomputes the whitening metric on the
MAPPED anchor distribution (map the Llama anchor train into gemma space, estimate its covariance), and
reads the mapped factorial through THAT metric. Because the mapped anchor count (≈136) is far below the
hidden dimension, the covariance is rank-deficient and MUST be shrunk — which is itself the B29 point.

## Design

- Reuse cycle-5 stimuli EXACTLY: truth-train, the danger-train statement set, and the UNCHANGED cycle-4
  2×2 factorial. gemma L12; per-target map layer by held-out R² (as cycle 5).
- Raw DiM directions d_truth, d_danger from gemma's labeled train (unchanged).
- Map Llama→gemma on the union anchor (truth+danger train, paired, label-free). Map the factorial and the
  anchor train into gemma space.
- **Whitening configs**, each reads the mapped Llama factorial:
  - **SOURCE (cycle-5 baseline):** Σ = gemma pooled-train covariance (eps=1e-3). Must reproduce cycle-5's
    Llama readout (≈0.9288 / 0.6562 / 1.0 / 0.5069) — a port check.
  - **MAPPED-λ:** Σ = covariance of the MAPPED anchor points, shrunk toward its scaled identity:
    `Σ_λ = (1−λ)·Σ̂ + λ·(tr Σ̂ / d)·I`, for λ in {0.2, 0.35, 0.5, 0.65, 0.8}.
  - RAW (no whitening) reported as descriptive context.
- For each config: whiten gemma-labeled train + mapped factorial by the same W = Σ⁻¹ᐟ²; fit the unit DiM
  direction in that whitened space; read the mapped factorial; compute the four-cell readout matrix
  (c_truth recovers T, invariant to H; c_danger recovers H, invariant to T). Qwen-3B reported descriptive.

## Frozen gates (verdict precedence)

The pre-registered gate uses **MAPPED-λ at λ = 0.5** as the primary, with a stability requirement.

- **BASIS-CLEARED** iff: gemma's own-factorial readout passes all four cells (reproduced from cycle 5,
  unchanged) AND the mapped-Llama readout under MAPPED-λ=0.5 passes all four
  (c_truth recovers T ≥ 0.75, discrim(c_truth,H) ≤ 0.65, c_danger recovers H ≥ 0.75,
  discrim(c_danger,T) ≤ 0.65) AND c_truth_invariant_H ≤ 0.65 holds for ≥ 3 of the 5 swept λ (so the pass
  is not one lucky shrinkage). → the cycle-5 miss was a source-whitening artifact; the full cross-model
  (truth × danger) basis holds. Upgrade cycle 5.
- **MISS-REAL** iff MAPPED-λ=0.5 does NOT bring c_truth_invariant_H ≤ 0.65, or the stability requirement
  fails. → the 0.0062 miss is real geometry (the truth coordinate genuinely leaks danger on Llama, not a
  metric artifact); cycle 5 stands as PARTIAL-STRUCTURED. Honest negative.
- **PARTIAL** — mixed (e.g. clears at 0.5 but unstable, or clears truth-invariance but breaks another
  cell). Report the exact matrices.

Thresholds (0.75 / 0.65), the λ=0.5 gate, and the ≥3/5 stability rule are frozen. Bars do not move.

## What it does NOT claim

Methodological consolidation of an existing certified arc, not a new scientific axis. A BASIS-CLEARED
result tightens cross-model readout (a real instrument improvement for styxx.crossmind); it does not
touch the content-vs-value distinction (concept identity still does not transport — cycle 6 CONTENT-WEAK).
Linear, whitened, local same-cluster models; the danger axis's 1.0 readout remains an n=48 existence
result. No model generates; the (false,danger) stimuli are flagged-as-false probe reads only.
