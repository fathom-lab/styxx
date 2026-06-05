# RESULT — the phase code is more noise-robust than the decay code (oscillation's first sovereign regime)

**Date:** 2026-06-04 · **Verdict: NOISE ROBUSTNESS CONFIRMED (control-cleared, understated).** Within
recurrent codes, the oscillatory (phase) code loses less capacity under state noise than the decay
(magnitude) code — and survives the scale-fairness control in the *strong* direction. Frozen:
`PREREG_noise_2026_06_04.md`. LRU-free vs LRU-clamped, ordered copy, D=256, 3 seeds, Gaussian state noise
at every timestep (train + eval).

## Numbers — kcap(σ) and ratio
| σ | 0.0 | 0.05 | 0.10 | 0.20 | 0.40 |
|---|---:|---:|---:|---:|---:|
| free kcap | 6.00 | 6.00 | 6.00 | 4.00 | 4.00 |
| clamped kcap | 2.67 | 2.67 | 2.00 | 2.00 | 1.33 |
| ratio | 2.25 | 2.25 | 3.00 | 2.00 | 3.00 |

Fractional capacity loss σ=0 → σ=0.4: **free −33%, clamped −50%.** Spearman(σ, ratio) = 0.40.

## What it shows (honest, two parts — and the control)
**The clean signal (P2).** The decay net erodes from the first noise step (2.67 → 2.67 → 2.00 → 2.00 →
1.33) and loses **half** its capacity under heavy noise. The oscillatory net is **noise-insensitive up to
σ=0.1** (holds 6.0), then steps down to 4.0, losing only **a third**. The phase code degrades more
gracefully than the magnitude code.

**The fairness control — the load-bearing check.** Absolute noise is only fair if both arms hold the same
state magnitude (`run_noise_control.py`). Measured state RMS: **free 1.33, clamped 1.72 (clamped/free =
1.29).** The clamped net carries the *larger* signal — so the oscillatory net is more noise-tolerant
**despite a worse noise-to-signal ratio.** The result is not a scale artifact; it is, if anything,
understated. This is the REVERSE of the confound I pre-flagged, and it strengthens the finding.

**The bounded part (P1).** The *ratio* curve is jagged and non-monotone (2.25, 2.25, 3.0, 2.0, 3.0) on
tiny integer kcaps (3-seed means of 1.33–6.0) — quantization-limited. The robust claim rests on the
fractional-degradation gap + the control, **not** on the ratio trend. Spearman 0.40 barely clears the
0.3 bar and should not be over-read.

## Scope (honest)
- **This is oscillation vs decay, not oscillation vs attention.** The transformer was not in this sweep:
  state-noise-at-every-timestep is a recurrent-specific noise model with no fair transformer analogue
  (residual-stream noise is a different injection). So the claim is *within recurrent codes, the phase
  code beats the magnitude code under noise* — not that rhythm beats attention under noise (untested).
- One task (ordered copy), one noise model (additive Gaussian on the recurrent state), capacity at 0.80.

## Arc meaning — the honest reframe is taking shape
This is the **first regime in the entire arc where oscillation is not dominated by its baseline.** Across
the *capacity* axis, rhythm is real but beaten (attention 15.3 vs free 6.0), resonant, length-tuned,
non-generalizing, and a constant-not-scarcity multiplier — six results saying *rhythm is not the secret
of memory capacity.* But on **robustness**, the phase code wins, cleanly and control-checked. That is
exactly what modern neuroscience increasingly says brain rhythms are *for* — coordination and robustness,
not raw storage. The operator's intuition that frequency is fundamental to mind lands here, demarcated:
fundamental to **robust coding**, not to memory capacity. Timing (the native-domain steelman) is the next
test of whether a second sovereign regime exists.
