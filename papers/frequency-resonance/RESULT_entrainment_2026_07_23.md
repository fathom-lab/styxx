# RESULT — ENTRAIN-OSS kill-gate: KILL (a real niche, below the registered bar) — 2026-07-23

Frozen by `PREREG_entrainment_2026_07_23.md`. Receipt: `entrainment_result.json`. Runner:
`run_entrain_timing.py` (parallel-scan recurrence, red-team-verified `scan==seq` and `κ=0==STATIC`
bit-for-bit). 3 seeds, 1500 steps, RTX 4070 Laptop. **Verdict: KILL** at the pre-registered primary
budget D=8.

## One line

A layer that learns to **tune its own oscillation frequency to the input online** (ENTRAIN-OSS) does
**not** beat the best static-frequency bank of equal budget at the registered operating point. The
effect is *real but confined to severe mode scarcity* (D=4), vanishes at D=8, and *reverses* (hurts)
when the bank is rich (D=16) — a monotone, mechanistically-coherent decay that sits below the frozen
bar. Attention dominates everything, as everywhere in this arc.

## Results (drifting-period next-symbol accuracy, seed-averaged)

| D (modes) | STATIC | ENTRAIN | ORACLE (pos. ctrl) | ENTRAIN−STATIC | ORACLE−STATIC |
|---|---|---|---|---|---|
| 4  | 0.247 | 0.318 | 0.351 | **+0.0718** | +0.104 |
| **8** (primary) | 0.460 | 0.500 | 0.631 | **+0.0396** | +0.1703 |
| 16 | 0.678 | 0.592 | 0.895 | **−0.0863** | +0.217 |

Floor / context (D=8): CLAMPED (pure decay) 0.348 — oscillation clearly beats decay. TRANSFORMER
(attention, context only, **not** gated) 0.916 — dominates all recurrent arms. No-harm (fixed
period, D=8): ENTRAIN−STATIC = **+0.0304** (≥ −0.03, passes — entrainment does not break the
stationary case). STATIC vs ENTRAIN are matched-param (D=8: 2052 vs 2196, the +144 = the detector
`a,b` + `κ,g`) and matched-compute (same scan recurrence).

## The frozen gate

Primary D=8: `oracle−static = +0.1703 ≥ 0.10` → the positive control **fires** (a diverse bank
locked to the true drifting period is worth +0.17, so adaptation is genuinely rewarded — no
ABSTAIN). `entrain−static = +0.0396 < 0.05` → **KILL**: the learnable loop captures only **23%** of
the oracle's available adaptation gap, and a static bank of equal budget is as good.

## The honest nuance (reported, NOT used to rescue the verdict)

The advantage falls monotonically with the mode budget — `{D4: +0.0718, D8: +0.0396, D16: −0.0863}`
— exactly the pre-registered corroboration direction. So ENTRAIN-OSS **does** have a genuine niche:
at the extreme-starved D=4 it beats static by +0.072 and captures ~69% of the oracle gap. But D=4 is
the pre-registered *fallback* (used only if the positive control failed at D=8, which it did not),
so per the frozen decision tree this niche **does not** license a greenlight — surfacing it as a
rescue would be exactly the favourable-direction post-hoc move this program forbids. The niche is
real and too narrow: it lives only where the bank is too starved to cover the range, and it inverts
into a *penalty* once the static bank has enough modes (D=16), because the noisy single-projection
frequency estimate perturbs an already-adequate bank.

The **oracle ceiling is real and grows with D**: ORACLE reaches 0.351, 0.631, 0.895 against STATIC's
0.247, 0.460, 0.678 — the gap a bank locked to the true timescale opens over the static bank *widens*
as modes increase. Re-pointing frequency to the true timescale is worth a lot, and worth *more* with
more modes; the learnable mechanism simply cannot reach it except under starvation — the gap between
what perfect entrainment buys and what the learned loop delivers *widens* as the problem gets easier
for a static bank.

## Positioning

This corroborates **TIDES** (arXiv 2605.09742), which put input-dependence on the decay/step-size
rather than the imaginary eigenvalue *because* input-dependent frequency hurt. Our slow-loop,
coherence-preserving PLL was the specific untested escape hatch; it does **not** clear the bar with a
single-projection detector. The static broad phase bank remains the thing to beat — consistent with
this arc's own timing finding (`RESULT_timing_2026_06_04.md`: oscillatory nets win timing via a
static distributed bank, not by tuning θ→2π/P).

## Scope, caveats, and the one un-taken lever

Toy scale (D≤16, L=96, integer-symbol streams, 3 seeds). The KILL is specific to **this detector** —
a *single* learned complex projection `z(t)=a·x(t)+i·b·x(t)` feeding a per-mode PLL. It does not
prove entrainment is dead in general: a richer phase estimator (a small learned filterbank /
multi-projection analytic signal) is the one lever not pulled here, and the D=4 niche + the large
oracle ceiling are the honest reasons that lever might be worth one more disciplined swing before the
mechanism is closed. What is settled: **the obvious, cheap version of input-frequency entrainment
does not beat a static bank at a sensible budget.**

## What ships

Per the plan's KILL branch: the honest negative is itself a contribution (the oscillation-vs-decay +
entrainment causal ablation the field's whole-architecture benchmarks skip), and the durable artifact
is the **resonance profiler** — the causal phase-clamp/κ-clamp instrument generalized to any trained
complex-eigenvalue SSM. That, not the layer, is STYXX's on-brand deliverable here.

Discipline note: a frozen gate + a load-bearing positive control + no post-hoc D-selection turned an
exciting "invent a super-tool" directive into a clean, one-afternoon negative with a precisely
located real niche — STYXX run on its own most-wanted idea.
