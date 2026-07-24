# PREREG — the consistency horizon (distance dose-response + mechanism confirmation)

**Frozen:** 2026-07-24, before this evaluated run. Extends `RESULT_consistency_oscillation_2026_07_23`
(SUPPORT: long-range consistency-checking requires oscillation) from a single point (gap 255) to a curve.
Same model, same phase-clamp knob (FREE θ-learnable vs CLAMPED θ≡0, matched/RNG-matched); the only new
axis is the premise→claim gap.

## What is swept

The delayed consistency comparison (label = claim==premise), premise placed at position T-1-gap, claim at
the final position, T=256 fixed. Gaps swept: 1, 2, 4, 8, 16, 32, 64, 128, 255. 2 seeds, 2500 steps.
Reported: FREE and CLAMPED test accuracy at each gap; and, for each trained CLAMPED model, `mag_max` = the
largest eigenvalue magnitude it learned (`exp(-exp(nu))`, max over all units/blocks) — the slowest-decaying
channel, which is what a decay model must use to hold the premise across the gap.

## Predictions (frozen)

- **Oscillation is range-free:** FREE ≥ 0.95 at every gap.
- **Decay has a finite horizon:** CLAMPED ≈ 1.0 at gap 1, falling monotonically to ≤ 0.60 by gap 255. The
  **consistency horizon** H = the gap at which CLAMPED mean accuracy crosses 0.75 (linear interpolation on
  log-gap). Prediction: 1 < H < 255.
- **Mechanism:** the premise signal a decay model retains scales as `mag_max^gap`. So CLAMPED accuracy
  should collapse exactly where `mag_max^gap` falls below the level needed to separate the four
  premise/claim cases. Frozen check: across gaps, Spearman rank-correlation between CLAMPED accuracy and
  `mag_max^gap` ≥ 0.80; and every gap with CLAMPED ≥ 0.90 has a larger `mag_max^gap` than every gap with
  CLAMPED ≤ 0.60 (clean separation).

## Frozen verdict logic

- **ABSTAIN** iff FREE < 0.95 at any gap (oscillation not range-free → the contrast is not clean) OR
  CLAMPED < 0.90 at gap 1 (decay can't even compare adjacent → prior result not reproduced here).
- **CONFIRM** iff (both controls hold) AND a finite horizon 1 < H < 255 exists AND the mechanism check
  passes (Spearman ≥ 0.80 and clean high/low separation).
- **PARTIAL** iff controls hold and a horizon exists but the mechanism check fails (curve without a
  confirmed cause) — reported verbatim.
- **NULL / anomaly** iff CLAMPED ≥ 0.90 at all gaps (decay never fails → contradicts the prior SUPPORT;
  investigate before any claim).

## Scope

Unchanged from the parent result: a controlled state-space-model characterization of when a pure-decay
channel loses a fact it must stay consistent with, and its mechanism. NOT a real-LLM honesty claim.

## Red-team asserts

1. `lin_scan == seq_scan` < 1e-4 (reused). 2. FREE/CLAMPED share `B_re`/`nu` at init (only θ differs).
3. Premise sits at exactly T-1-gap for each gap; claim at T-1; labels 50/50. 4. `mag_max` is read from the
trained model's `nu` (not assumed) and lies in (0,1).
