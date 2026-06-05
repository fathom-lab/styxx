# RESULT — oscillation is a length-SPECIALIZATION, not generalization: it fails extrapolation at chance

**Date:** 2026-06-04 · **Verdict: NO REGIME FLIP — my "rhythm = generalization" bet is REFUTED.**
Frozen: `PREREG_extrapolation_2026_06_04.md`. Train on K≤8 ordered copy, test on K∈{10,12,15,18,20}.
3 seeds, matched params, sinusoidal-pos transformer.

## Numbers (mean acc)
| arm | in-distribution (K≤8) | EXTRAPOLATION (K>8) |
|---|---|---|
| LRU-free (oscillatory) | 0.978 | **0.088** (= chance, 1/12 = 0.083) |
| LRU-clamped (decay) | 0.805 | 0.175 |
| TRANSFORMER (attention) | 1.000 | 0.165 |

## What it shows (and it's deeper than the bet)
- **Nothing extrapolates here** — all three collapse toward chance past the trained length (the
  predicted "recurrence holds where attention collapses" did NOT happen; P1 false).
- **Oscillation extrapolates *worst* — at dead chance** (0.088), despite near-perfect in-distribution
  accuracy. The phase code is **completely length-specific**: it transfers to *zero* new lengths.
- Decay (0.175) and attention (0.165) are comparably poor; the **decay net actually generalizes slightly
  better than the oscillatory one** (P2 false, and reversed).

**Mechanism (coherent with the whole arc):** the resonance is a *tuning*. The oscillatory net learns a
phase code optimized for the trained item-counts (θ\* is item-count-bound — the scaling NULL), so on
unseen lengths it is exactly at chance. **Oscillation is a SPECIALIZATION mechanism: it buys in-distribution
capacity by over-fitting the length regime, and pays for it with zero generalization.** A smooth decay
code is more length-agnostic; attention with absolute positions also fails, but no worse.

## The frequency arc, closed — the complete honest map of rhythm in computational memory
1. **rhythm-rescue:** oscillation ≈ doubles recurrent ordered-memory capacity (real).
2. **frequency-resonance:** capacity is resonant in frequency, peaks then collapses (real; reproduces the
   theta-gamma inverted-U).
3. **scaling NULL:** the optimum is item-count-bound — length-tuned, not window-scaling.
4. **necessity:** a rhythm-free transformer beats oscillation 15.3 vs 6.0 at matched params (rhythm not
   special, not optimal for capacity).
5. **extrapolation (this):** oscillation fails to generalize — at chance on new lengths, *worse* than
   decay or attention. It is a length-SPECIALIZATION.

**Net:** the romantic claim — "rhythm is the secret of memory/mind" — is taken apart brick by brick, by
controlled experiments, and replaced with a precise, bounded, true account: **oscillation is a real,
length-tuned, capacity-limited, non-generalizing recurrent mechanism that a rhythm-free architecture
dominates.** Real, modest, and honestly demarcated — the woo→rigor move, completed for this corner.

## Honest scope
One synthetic task, small matched scale; absolute/sinusoidal positions for the transformer (RoPE/ALiBi
extrapolate better and would likely beat all three — noted). The strong, robust finding is the one against
my own hypothesis: **oscillation does not aid length-generalization; it is the worst of the three at it.**
