# PREREG — does the phase mechanism generalize to canonical K-way delayed recall?

**Frozen:** 2026-07-24, before this evaluated run. Extends the consistency-horizon result from a bespoke
binary comparison to the standard **delayed recall / selective-copy** primitive used across the SSM
literature (S4, Mamba) — the mechanistic substrate of in-context recall in real models. Same model, same
single phase-clamp knob (FREE θ-learnable vs CLAMPED θ≡0, matched, RNG-matched); the new axis is that the
model must RETRIEVE which of K values appeared earlier, not just compare a binary claim.

## Task — delayed K-way recall

K=16 content values. Input channels: a one-hot content value ±placed at position T-1-gap (channels
0..K-1), and a query marker at the final position (channel K). Rest zero. Readout = final-position hidden
state; K-way head. Label = the content value. Chance = 1/K = 0.0625. This is "recall the fact you were
given, across distance" — a K-way generalization of the consistency task and the induction/selective-copy
primitive real SSM-LLMs are benchmarked on. Gaps swept: 1, 4, 16, 32, 64, 128, 255.

Because a decay model's trainability is bimodal (per the consistency-horizon result: solve vs stuck),
the reported metric is the **success rate** `p_solve(gap)` = fraction of independent seeds reaching
accuracy ≥ 0.80 (>> chance 0.0625) within budget (5 seeds, 2000 steps). FREE solve rate confirms
range-freedom.

## Predictions (frozen)

- **Oscillation is range-free:** FREE solve rate = 1.0 at every gap.
- **Decay has a recall horizon:** CLAMPED `p_solve` ≈ 1.0 at gap 1, declining to ≤ 0.2 by gap 255; a
  finite half-horizon H* (crossing 0.5) with 1 < H* < 255. Prediction: the K-way recall horizon is at or
  BELOW the binary consistency horizon (~gap 32), since distinguishing K values needs more retained signal
  than a binary sign.
- **Mechanism:** CLAMPED `p_solve` tracks the surviving signal `mag_max^gap` (Spearman ≥ 0.75).

## Frozen verdict logic

- **ABSTAIN** iff FREE solve rate < 1.0 at any gap (oscillation not range-free) OR `p_solve(gap=1) < 0.80`
  (decay can't even recall adjacent → nothing to characterize).
- **CONFIRM** iff FREE range-free AND `p_solve(1) ≥ 0.80` AND finite half-horizon 1 < H* < 255 AND
  `p_solve` reaches ≤ 0.2 by gap 255 AND mechanism Spearman ≥ 0.75.
- **PARTIAL** otherwise — reported verbatim.

## Scope (non-negotiable)

A controlled state-space-model result on the standard recall primitive: a pure-decay channel's probability
of retrieving a fact declines with distance; the oscillatory channel's does not. It connects the mechanism
to the benchmark real SSM-LLMs are measured on, and it explains why real decay-SSMs (Mamba-class) need
help (attention hybrids) for exact recall. It is NOT itself a real-LLM measurement — no language model is
run here. The real-model test (a trained Mamba/LinOSS checkpoint) remains the next rung.

## Red-team asserts

1. `lin_scan == seq_scan` < 1e-4 (reused). 2. FREE/CLAMPED share `B_re`/`nu` (only θ differs).
3. Content one-hot at exactly T-1-gap; query marker at T-1; labels ≈ uniform over K. 4. `mag_max` read
from trained `nu`, in (0,1). 5. chance-level baseline 1/K confirmed on an untrained readout.
