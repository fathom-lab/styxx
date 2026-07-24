# PREREG v2 — the consistency horizon as a SUCCESS-PROBABILITY curve

**Frozen:** 2026-07-24, before this evaluated run. Supersedes the mean-accuracy horizon
(`PREREG_consistency_horizon_2026_07_24`), which **ABSTAINED**: its gap-1 control failed because a
decay (theta==0) model's outcome is BIMODAL — each seed either lands in a solving basin (accuracy 1.0)
or a stuck basin (~0.5), so the mean of two seeds is not a meaningful "accuracy," and 2 seeds cannot
estimate the fraction that solves. The honest characterization of a bimodal outcome is a
success-probability curve.

## What is measured

Same model / phase-clamp / consistency-comparison task (label = claim==premise; premise at T-1-gap; T=256).
FREE (theta learnable) confirms range-freedom; the horizon lives in CLAMPED (theta==0). Per gap:
- **CLAMPED solve rate** `p_solve(gap)` = fraction of independent seeds reaching test accuracy >= 0.90
  within the training budget (6 seeds, 1500 steps — a winning seed converges in <500 steps in smoke, so
  this is `P(decay finds a solving representation within budget)`, operationally defined, not a claim of
  permanence).
- **FREE solve rate** (2 seeds) — expected 1.0 at every gap.
- **mag_max(gap)** — largest learned eigenvalue magnitude (mean over CLAMPED seeds), and the surviving
  premise signal `mag_max^gap`.

## Predictions (frozen)

- **Oscillation is range-free:** FREE solve rate = 1.0 at every gap.
- **Decay has a probabilistic horizon:** `p_solve` ~ 1.0 at gap 1, declining monotonically to <= 0.2 by
  gap 255. **Half-horizon** H* = the gap where `p_solve` crosses 0.5 (linear interpolation on log-gap);
  prediction 1 < H* < 255.
- **Mechanism:** `p_solve` tracks the surviving signal `mag_max^gap` — Spearman rank-correlation
  (`p_solve` vs `mag_max^gap`) >= 0.80.

## Frozen verdict logic

- **ABSTAIN** iff FREE solve rate < 1.0 at any gap (oscillation not range-free) OR `p_solve(gap=1) < 0.80`
  (decay does not even reliably solve adjacent within budget → nothing to characterize).
- **CONFIRM** iff FREE range-free AND `p_solve(1) >= 0.80` AND a finite half-horizon 1 < H* < 255 exists
  AND `p_solve` reaches <= 0.20 by gap 255 AND the mechanism Spearman >= 0.80.
- **PARTIAL** iff controls hold and a half-horizon exists but monotonicity or the mechanism check is not
  clean — reported verbatim.

## Scope

Unchanged: a controlled state-space-model characterization of when a pure-decay channel probabilistically
loses a fact it must stay consistent with, and its mechanism. NOT a real-LLM honesty claim.

## Red-team asserts

1. `lin_scan == seq_scan` < 1e-4. 2. FREE/CLAMPED share `B_re`/`nu` at init (only theta differs).
3. Premise at exactly T-1-gap; labels 50/50. 4. Solve rate uses a fixed 0.90 threshold and independent
seeds (0..5); `mag_max` read from trained `nu`, in (0,1).
