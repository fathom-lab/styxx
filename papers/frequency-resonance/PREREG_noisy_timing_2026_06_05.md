# PREREG — can rhythm BEAT attention (not tie) at timing under corruption? (the ceiling-breaker)

**Date (frozen before data):** 2026-06-05. The frontier the closed map named. Timing (exp 8) was a
*ceiling tie*: free 1.00 ≈ transformer 0.99, both saturate — so "rhythm is undominated at timing" could
not be sharpened to "rhythm beats attention." The fix is a harder temporal task where neither saturates.
The sharp version unifies the two axes where rhythm won: **periodic prediction under input corruption.**
If oscillation's noise-robustness (exp 7) is real, its distributed phase integration should track a
*corrupted* periodic stream better than attention's position-lookup, which breaks when the looked-up
symbol is the corrupted one. And input corruption is **fair to all three arms** (unlike exp-7 state noise),
so this also closes that scope gap.

## Method (frozen)
- Periodic next-symbol prediction (the exp-8 task) + **input corruption**: each input symbol is replaced
  with a uniform-random symbol with probability ρ. **Target = the CLEAN periodic continuation** (predict
  motif[(t+1) mod P], not the corrupted token). Score after two full periods.
- Train with ρ ~ Uniform(0, 0.4) per batch (each arm learns to denoise across the range); evaluate at
  fixed **ρ ∈ {0.0, 0.1, 0.2, 0.3, 0.4}**, accuracy averaged over periods **P ∈ {4, 6, 8}** (the regime
  where decay already fails clean and rhythm/attention both worked clean — so the ceiling is broken only
  by corruption, not period length).
- Three matched-param arms (LRU-clamped, LRU-free, transformer/learned positions), 3 seeds.

## Hypotheses & predictions (frozen)
- **P1 — rhythm beats attention under corruption (the ceiling-breaker):** `acc_free − acc_transformer ≥
  +0.05` at high corruption (ρ ≥ 0.3). Mechanism: a distributed phase bank integrates the periodic signal
  over many cycles (robust to per-symbol corruption); attention's lookup of position t−P is fragile when
  that position is corrupted.
- **P2 — the honest counter:** attention's ability to attend to *every* prior instance of the period and
  majority-vote is itself a strong denoiser and may TIE or BEAT rhythm — `acc_transformer ≥ acc_free −
  0.05` across ρ. (I hold P1 loosely; this outcome is fully plausible.)

## Decision rule (frozen)
- **RHYTHM BEATS ATTENTION — ceiling broken** iff P1: rhythm strictly more corruption-robust at timing.
  This would be the first regime where oscillation *beats*, not ties, attention — turning "undominated"
  into "superior" on noisy temporal structure, the strongest possible landing of the arc.
- **ATTENTION TIES-OR-WINS — demarcation tightened** iff P2: rhythm never strictly beats attention; the
  honest map sharpens to "rhythm ties attention at timing, loses elsewhere; its only clean win is over its
  decay baseline." Still a real, publishable tightening.
- Report acc(ρ) for all three arms regardless. Either outcome is decisive and kept.

## Caveats (frozen)
- One fair noise model (input-symbol corruption — not embedding-space Gaussian, not jittered periods),
  averaged over P ∈ {4,6,8}, one task family, matched params. This tests robustness of *temporal*
  prediction specifically, the intersection of the two axes rhythm won on. A clean P1 is the ceiling-
  breaker the map asked for; a P2 tightens the map by bounding rhythm to "at best co-equal with attention."
