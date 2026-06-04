# THEORY — why memory is resonant in frequency, and what sets the optimum

**Date:** 2026-06-04 · companion to `RESULT_frequency_sweep_2026_06_04.md` (resonance, observed)
and `PREREG_scaling_law_2026_06_04.md` (theta* x window, running). A scaling/order-of-magnitude
phase-coding argument — not an exact capacity theorem. Written before the scaling result; it does
not change the frozen rule, it frames the interpretation.

## 1. A single complex mode is a clock with a fade

LRU diagonal mode: h_t = lambda * h_{t-1} + b x_t, with lambda = r e^{i theta}, 0<r<1,
theta in [0, pi]. An impulse (item) written at time tau contributes at readout time t:

    lambda^{t-tau} = r^{t-tau} * e^{i theta (t-tau)}.

Two channels. **Magnitude** r^{t-tau}: a fading memory, horizon ~ 1/(1-r). **Phase**
theta*(t-tau): a clock that tags *when* the item was written.

## 2. Ordered recall needs phase separation -> the resonance falls out

To recall K items *in order*, the readout must tell them apart by their write-time. The phase is
the only channel that encodes order (magnitude alone gives a monotonic recency gradient, not
identity-at-position). K items written one step apart carry phases spaced by theta. Two failure
modes bound the useful range:

- **theta -> 0:** no phase code; items pile up near phase 0; only the decay gradient separates
  them. Low capacity. (Observed: theta=0 baseline kcap ~2.7.)
- **theta -> pi (Nyquist):** lambda = -r, the mode flips sign every step. Adjacent items land at
  opposite signs at *every* readout time — maximally confusable, the phase carries barely one bit,
  and the rapid sign-flips are hard for the readout to decode. Worse than pure decay. (Observed:
  Nyquist collapses to the floor, at or below baseline.)
- **interior theta\*:** the held items tile the phase circle ~once — maximally separated without
  wrapping. (Observed: interior peak in the ~0.06-0.5pi band.)

**So the resonance is the generic signature of a phase code: a band-limited optimum with collapse
at both ends.** Section 2 is not a new claim — it is the mechanism behind the result we already
measured, and it is robust.

## 3. What sets theta\*? Two budgets — and the delay sweep adjudicates them

The "tile the circle once" condition is `theta* x (phase-relevant span) ~ 2 pi alpha`. Everything
hinges on *which span* binds:

**(a) Write-span budget (item-count).** The K items are written across ~K consecutive steps; their
pairwise phase differences are fixed at write time (= theta x index-gap). Anti-aliasing across K
items -> **theta\* ~ c / K**. Crucially, a pure *delay* (hold the same K items longer) rotates all
modes by the same theta*D and **preserves the relative phases** between items. Under this budget,
delay does not change the optimal theta -> **the delay sweep returns NULL.**

**(b) Hold-window budget.** Over a hold of W steps the phase must stay coherent against accumulated
phase-noise/drift and survive decay (forcing r -> 1); the readout's absolute-phase reference also
advances by theta*W. Longer holds favor a *slower* clock (less total phase to keep coherent, less
aliasing of the absolute reference) -> **theta\* ~ c / W** -> **the delay sweep returns the SCALING
LAW theta\* x W ~ const.**

These make **opposite** predictions. That is what makes the running delay sweep diagnostic:

- **NULL** => theta* is *item-count*-bound — relative phases are preserved under uniform rotation;
  the resonance is set by how *many* items, not how *long* they are held.
- **SCALING** => theta* is *window*-bound — coherence/decay over the hold sets the clock.

## 4. Honest prior (tempered from the frozen PREREG)

The PREREG bet SCALING. Deriving the mechanism *tempers* that bet: the relative-phase-preservation
argument (3a) is clean, and makes **NULL a principled, plausible outcome — not a failure.** I still
lean SCALING — because over long holds, decay pressure (r->1) and phase-coherence couple theta to
W, and the readout's absolute-phase reference drifts with the hold — but the theory moves me from
the PREREG's confident bet toward roughly a coin flip. **Either outcome identifies the mechanism.**
This is the value of doing the theory: it converts a yes/no bet into a two-way mechanistic readout.

## 5. The constant, if SCALING holds

"Tile once" => `theta* x W ~ 2 pi alpha`, alpha = the cycle-fraction the optimal mode sweeps over
the window. Anchoring on the frequency sweep (theta* ~ 0.375pi at W ~ 3.5):
`alpha ~ 0.375 pi * 3.5 / 2 pi ~ 0.66` — about two-thirds of a cycle over the window. So if SCALING
holds: `theta* x W ~ 1.3 pi rad`, predicting theta*(D=6,12,24) ~ 0.14pi, 0.085pi, 0.047pi. (This is
the same number the PREREG guessed, now with a reason attached.)

## 6. Why a spectrum beats a single tone (the FREE result)

A single theta is one clock; it tiles the circle at exactly one scale. Ordered memory spans
multiple scales (recent vs old items, short vs long holds). A *spectrum* of theta across modes
(what FREE learns) tiles many scales at once — a Fourier-like positional code. Hence FREE >= best
single tone. (Observed direction; our +1-item bar was not cleared because at this task size the
single-tone optimum already nears the kcap ceiling.)

## 7. Limits (honest)

- Order-of-magnitude scaling argument, not an exact capacity theorem. alpha and the exact theta*
  depend on r, the 2-layer-MLP readout (which can partially decode aliased phases), and the 0.80
  threshold.
- Single-mode idealization; the trained net couples 256 modes + nonlinear readout.
- The hold-window coherence claim (3b) is argued, not derived; a rigorous version needs the
  phase-noise model of the trained dynamics.
- In-silico, one task/architecture. **Section 2 (resonance) is observed and robust; Section 3 (what
  sets theta*) is the open test the running sweep adjudicates.**
