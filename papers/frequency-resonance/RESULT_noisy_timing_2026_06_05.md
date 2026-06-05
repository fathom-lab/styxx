# RESULT — rhythm ties attention under corruption; it never strictly beats it (the ceiling-breaker came back negative)

**Date:** 2026-06-05 · **Verdict: ATTENTION TIES — P1 refuted, the map tightens.** Under input corruption,
the oscillatory net and the rhythm-free transformer degrade *identically*; rhythm does not become superior
where exact-lookup is stressed. Frozen: `PREREG_noisy_timing_2026_06_05.md`. Periodic prediction + input
corruption, three matched-param arms, ρ swept 0→0.4, averaged over P∈{4,6,8}, 3 seeds.

## Numbers — acc(ρ)
| ρ | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 |
|---|---:|---:|---:|---:|---:|
| **free** | 0.995 | 0.958 | 0.902 | 0.825 | 0.725 |
| **transformer** | 1.000 | 0.970 | 0.915 | 0.829 | 0.717 |
| clamped | 0.632 | 0.530 | 0.453 | 0.387 | 0.330 |

free − transformer at ρ≥0.3 = **+0.002**. The two curves are within ±0.013 at every ρ and cross at ρ=0.4.

## What it shows (read straight)
**P1 refuted.** I bet oscillation's distributed phase bank would denoise the periodic stream better than
attention's position-lookup. It does not. The curves are statistically indistinguishable across the whole
corruption sweep — same graceful degradation (1.0 → ~0.72 at 40% noise), same robustness. The smoke's
+0.05 tease was undertrained noise; it vanished with 3 seeds at full training.

**The honest consequence — the map tightens, hard.** Timing (exp 8) was a *ceiling* tie, which left open
"maybe rhythm beats attention on a *harder* temporal task." This was that test. The answer is no: rhythm
ties attention at clean timing AND at noisy timing. **Across every fair comparison with attention in this
program — capacity, clean timing, noisy timing — rhythm never strictly wins. It loses (capacity) or ties
(timing).** Its only clean victories are over its own *decay* baseline (which collapses to 0.33 here). So
the sharpened statement: *within recurrence, oscillation is the better temporal/robust mechanism than pure
decay; against attention, it is at best co-equal and never superior on anything we could test fairly.*

**The interesting thing in the null (not a win — a finding).** Two architectures with nothing in common —
recurrence-with-rhythm and attention-with-no-rhythm — produce the *same robustness curve to three decimal
places* under corruption. That is architectural convergence: they appear to learn the same underlying
solution to noisy periodic prediction, expressed in different machinery. Rhythm buys no robustness edge
over attention because attention already solves it as well; the convergence, not a rhythm advantage, is the
real observation.

## Scope (honest)
One corruption model (uniform input-symbol replacement), one task family, P∈{4,6,8}, matched params. Other
hardness axes that could still separate them — jittered/drifting periods, much longer horizons,
compositional/multi-period structure, continuous-time inputs — are untested and remain genuinely open. But
the corruption axis, the one most directly tied to oscillation's claimed noise-robustness, is clear: no
rhythm advantage over attention.

## Arc meaning — the demarcation, in its tightened final form
**Rhythm is a real, efficient recurrent mechanism — and a dominated/co-equal one.** It beats pure decay at
capacity (~2×), robustness (−33% vs −50% under noise), and timing (decisive). But a rhythm-free
architecture (attention) matches or beats it on every axis tested fairly: better at capacity, equal at
clean timing, equal at noisy timing. The folk claim "frequency underlies mind" does not survive even its
own steelman: oscillation's home turf (temporal structure) is shared, not owned. The honest landing of the
operator's intuition is its smallest true form — *frequency is one efficient way a recurrent system can
encode time and resist noise, not a privileged or superior substrate for cognition.* The value is the
boundary, drawn here as tight as controlled experiment can draw it.
