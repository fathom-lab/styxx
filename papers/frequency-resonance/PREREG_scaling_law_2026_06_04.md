# PREREG — Does the resonant optimum scale as 1/window?  (theta* x window = const?)

**Date (frozen before data):** 2026-06-04 · **Status:** pre-registration. No delay-sweep data
exists at write time. Builds on `RESULT_frequency_sweep_2026_06_04.md` (capacity is RESONANT in
frequency; peak theta*~0.375pi at the base retention window).

## The question

The frequency sweep found a resonant optimum theta* for holding ordered items. If the mechanism
is *phase-coding over the retention window* (items tile the phase circle without wrapping), then
the optimum should be set by **how long items must be held**: a longer window needs a *slower*
oscillation to avoid wrapping. Sharp prediction:

> **theta\*(W) x W = constant** — the optimal mode completes a fixed fraction of a cycle over
> the retention window, regardless of window length.

This turns the resonance curve into an **equation**. It is falsifiable: if theta* does not move
with the window, the resonance is a fixed architectural property, not a window-tuned code.

## Method (frozen — same rig/config as the frequency sweep, one addition)

Ordered-copy with an inserted **delay**: sequence = [K symbols][**D PAD tokens**][K GO], recall
the K symbols in order. The first item is held **W = K + D** steps before its recall slot.
Per delay D, sweep frozen theta and locate the optimum at the **capacity edge**:

- **peak signal:** kcap = largest K (over KGRID=[1,2,3,4,5,6,8,10]) with mean recall acc >= 0.80.
  (Revised pre-data from "mean acc over K in {2,3,4,5}": that mid-load signal ceiling-saturates
  at full training and goes flat; kcap reads the discriminating capacity edge, as the frequency
  sweep did. No scaling data seen at revision — this is a method fix, not a post-hoc move. Bonus:
  D=0 should reproduce the frequency sweep's theta*~0.375pi, an internal consistency control.)
- **theta\*(D):** argmax kcap over the theta grid.
- **window:** W(D) = D + kcap*(D)  (delay plus the items actually held at the optimum — the real
  retention window the system operates at).

Frozen config (matches rhythm-rescue / frequency sweep): D_model=256, V=12, STEPS=4000,
BATCH=64, LR=2e-3, 3 seeds, ACC_THR not used (we read accuracy, not kcap, for peak location).
**Delays:** D in {0, 6, 12, 24} (W = 3.5, 9.5, 15.5, 27.5; ~8x lever). **theta grid (rad/step,
low-resolved where large-D optima are predicted):** theta/pi in
{0.025, 0.05, 0.09, 0.15, 0.25, 0.375, 0.5}.

## Hypotheses & pre-stated predictions

- **H_scale (mechanism — my prior):** theta* falls with D; the product theta\*(D)x W(D) is
  ~constant. Predicted constant ~ 0.375pi x 3.5 ~= **1.3pi rad (~0.65 cycle)** over the window.
  So predicted optima: D=0 -> ~0.375pi, D=6 -> ~0.14pi, D=12 -> ~0.085pi, D=24 -> ~0.047pi.
- **H_null:** theta* invariant to D (peak does not move) -> resonance is architecture-fixed,
  not window-tuned.
- **H_partial:** theta* moves with D but not as 1/W (e.g., constant offset).

## Decision rule / kill-gate (frozen)

Locate theta*(D) for each D. Let rho = Spearman(D, theta*), CV = std/mean of {theta\*(D)xW(D)}.

1. **SCALING LAW (H_scale)** iff **rho <= -0.80 AND CV(theta\*xW) <= 0.35.**
2. **NULL** iff theta* spans <= 1 grid step across all D (no movement).
3. Otherwise **PARTIAL — theta* moves but the 1/W law does not hold at bar; report shape.**

**Pre-registered bet:** SCALING LAW, constant ~ 0.65 cycle over the window. If NULL fires, the
resonance is a fixed property and the 1/W story is wrong — reported as such.

## Caveats (frozen)

- Requires the delay task to remain *learnable* at each D within 4000 steps; if accuracy at the
  largest D collapses to chance for all theta, theta* is undefined there and that D is dropped
  (logged, not hidden).
- One task/arch/hidden-size, in-silico, n=3 seeds. Read whether theta* **moves with the window**
  and whether the **product is stable** — not the exact constant.
- A confirmed law here is a research finding; instrument productization is a separate per-feature
  validation (cf. the geometry probe).
