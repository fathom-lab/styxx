# PREREG — rhythm's real niche: does recurrence (and oscillation) extrapolate where attention collapses?

**Date (frozen before data):** 2026-06-04. Deeper than the necessity showdown. That result (attention
beats oscillation on in-distribution ordered-copy capacity, 15.3 vs 6.0) was on attention's home turf.
The honest open question it leaves: **is there a regime where rhythm/recurrence wins something attention
structurally cannot?** The canonical one is **length extrapolation** — transformers notoriously fail
beyond training length; recurrence carries a structured state forward. If recurrence holds where
attention collapses — and if *oscillation* (a phase code) extrapolates better than *decay* — then
rhythm's role is **structure/generalization, not raw storage.** A deeper, more interesting answer than
"attention just wins."

## Method (frozen)
- Same three arms, matched params (~168k): **LRU-FREE** (oscillatory), **LRU-CLAMPED** (decay only),
  **TRANSFORMER** (attention; **sinusoidal** positional encoding — defined for all positions, the *fair*
  extrapolation-capable choice, not learned embeddings). Ordered copy, 3 seeds, identical training.
- **Train on K ∈ [1, 8] only.** Evaluate on K ∈ {1..8} (in-distribution) AND K ∈ {10,12,15,18,20}
  (EXTRAPOLATION — strictly longer than anything seen in training). Acc = ordered-recall accuracy;
  chance ≈ 1/12 = 0.083.

## Hypotheses & predictions (frozen)
- **P1 — recurrence extrapolates, attention collapses:** mean extrapolation accuracy (K>8) for the
  recurrent arms (LRU-free, LRU-clamped) exceeds the TRANSFORMER's by ≥ **0.15**, and the transformer's
  extrapolation accuracy falls to ≤ **0.30** (toward chance) despite high in-distribution accuracy.
- **P2 — oscillation aids generalization (the deep bet):** LRU-FREE extrapolation accuracy ≥ LRU-CLAMPED
  by ≥ **0.05** — the phase code carries structure to unseen lengths better than pure decay.

## Decision rule (frozen)
- **RHYTHM'S NICHE IS GENERALIZATION** iff P1 — recurrence retains accuracy past training length where
  attention collapses. (Completes the honest picture: attention wins in-distribution capacity; recurrence
  wins extrapolation — different strengths, no single winner.)
- **+ OSCILLATION AIDS STRUCTURE** iff additionally P2 — oscillation specifically extrapolates better than
  decay. (Would be the genuinely novel, deeper finding: rhythm = structured generalization.)
- Otherwise report the in-dist vs extrapolation curves for all three.

## Caveats (frozen)
- One task, matched-param controlled, sinusoidal-pos transformer (a fair but not exhaustive extrapolation
  baseline — RoPE/ALiBi might do better; noted). The claim is about a *qualitative regime difference*
  (in-dist capacity vs out-of-length generalization), not exact magnitudes. A transformer collapse here
  reflects the known limitation of absolute/sinusoidal position under length shift, which is the point:
  recurrence has no such positional dependence.
