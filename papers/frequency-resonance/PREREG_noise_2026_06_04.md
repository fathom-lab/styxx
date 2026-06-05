# PREREG — does oscillation's edge GROW under noise? (the phase-vs-rate robustness claim)

**Date (frozen before data):** 2026-06-04. The second scarcity axis, and the most-cited reason
phase-coding is said to beat rate-coding in biology: **noise robustness.** The multiplexing run showed
oscillation's advantage is flat in *dimension*-scarcity. This tests the other resource the brain is
starved for — a clean *signal*. If the phase-code is genuinely more noise-robust than the decay
(magnitude) code, oscillation's advantage should GROW as state noise rises: the clamped net's
magnitude code degrades, the free net's phase code holds.

## Method (frozen)
- LRU-FREE (oscillatory) vs LRU-CLAMPED (decay only), ordered copy, D=256 fixed, 3 seeds, the
  `run_rhythm_rescue.py` rig + **Gaussian state noise**: at every recurrent timestep, after the state
  update, add `noise_std · N(0,1)` to the real and imaginary state (same noise to both arms — fair).
  Train AND evaluate under the noise (each arm learns its most noise-robust code).
- **Sweep noise_std σ ∈ {0.0, 0.05, 0.1, 0.2, 0.4}.** kcap = largest K with mean acc ≥ 0.80.

## Hypotheses & predictions (frozen)
- **P1 — oscillation's edge grows under noise:** `ratio(σ) = kcap_free / kcap_clamped` RISES with σ —
  specifically Spearman(σ, ratio) ≥ **+0.3** AND `ratio(0.4) ≥ ratio(0.0) + 0.5`. Mechanism: the phase
  (2-D angle) code survives additive corruption the 1-D magnitude/decay code cannot.
- **P2 — clamped degrades faster (the strong form):** clamped kcap falls a larger fraction from σ=0 to
  σ=0.4 than free kcap does (free is more noise-robust in absolute terms).

## Decision rule (frozen)
- **OSCILLATION'S NICHE IS NOISE ROBUSTNESS** iff P1 — its advantage grows under noise. This would be
  the genuine reconciliation: oscillation is not special for raw or dimension-bound capacity, but IS the
  noise-robust code — *why a noisy, resource-bound brain uses rhythm.* The first real "why."
- **NOISE-FLAT / REFUTED** iff ratio(σ) does not clearly rise (Spearman < 0.3 or both collapse together).
  Oscillation's edge is noise-independent; the robustness reconciliation does not hold either.
- Report the full ratio(σ) and absolute kcap(σ) curves.

## Caveats (frozen)
- One task (ordered copy), one noise model (additive Gaussian on the recurrent state — not input or
  readout noise, not spiking/Poisson noise), one oscillation knob. Read the *trend* in the ratio. This
  tests additive-state-noise robustness specifically; other noise models (multiplicative, input-side,
  quantization) remain separate untested doors. A clean P1 here would be the strongest pro-oscillation
  result of the arc; a refutation narrows the "why biology uses rhythm" question further.
