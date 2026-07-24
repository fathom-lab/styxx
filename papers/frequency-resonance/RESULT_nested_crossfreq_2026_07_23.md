# RESULT — NEST-OSS (theta-gamma coupling): KILL — the flat bank is already at the ceiling — 2026-07-23

Frozen by `PREREG_nested_crossfreq_2026_07_23.md`. Receipt: `nest_capacity_result.json`. Runner:
`run_nest_capacity.py` (parallel-scan recurrence; red-team `scan==seq`, `α=0==FLAT` bit-for-bit,
oracle slotting one-hot). 3 seeds, 3000 steps, RTX 4070. **Verdict: KILL** at the pre-registered
primary budget D=8.

## One line

Implementing the brain's theta-gamma memory code — a slow clock gating fast modes into ordered
phase-slots — as an SSM primitive does **not** beat a flat oscillatory bank of equal budget. Capacity
headroom is real (doubling modes buys **+0.183**), but explicit nesting captures essentially none of
it (**+0.009**, ~5%). The flat learnable oscillatory bank is already at the phase-multiplexing
ceiling; explicit cross-frequency coupling is the redundant thing no SSM implements *because* it is
redundant.

## Results (ordered-copy mean recall accuracy over K=2..10, seed-averaged)

| D | FLAT | NEST | ORACLE (slotting) | kcap (flat/nest) |
|---|---|---|---|---|
| **8** (primary) | 0.376 | 0.385 | 0.385 | 0 / 0 |
| 16 | 0.558 | 0.578 | 0.678 | 3 / 3 |

Positive control **WIDE** = FLAT with twice the mode budget (16 modes): mean-acc **0.558** (kcap 3). Context: CLAMPED
(decay, D=8) 0.344; TRANSFORMER (attention) **0.980**, kcap **10** — dominates every recurrent arm, as
everywhere in this arc. FLAT vs NEST matched-param (D=8: 2116 vs 2126) and matched-compute.

## The frozen gate (primary D=8)

`wide−flat = +0.1825 ≥ 0.05` → capacity **headroom fires** (more modes genuinely buy capacity, so the
task can detect a capacity gain). `nest−flat = +0.0091 < 0.02` → **KILL**: explicit nesting captures
only **5%** of the headroom a 2×-wider bank buys. The slotting-oracle diagnostic at D=8 is `+0.0095`
— hard input-gating (routing each item to one mode) is *lossy*: it destroys the distributed encoding
a flat bank exploits, so it does not even serve as a positive control there (a redesign the smoke
caught before freezing — the wide bank is the valid control).

## The honest nuance

At **D=16** (modes ≥ items for small K), the picture is not flat-zero: NEST reaches mean-acc 0.578
against FLAT's 0.558, and the perfect-slotting ORACLE reaches 0.678 (kcap 4 vs FLAT's 3). So
phase-slotting *can* help once there are enough modes to hold the items — but the **learnable** nested
coupling captures only a sliver of the oracle's margin, and at the registered D=8 budget it is null. The consistent story across D: **capacity is bought by more modes, not by
cleverer coupling of the modes you have.**

## Positioning

Corroborates this arc's own multiplexing result (`project_frequency_resonance_2026_06_04`): a flat
oscillatory bank already realizes a constant phase-multiplexing multiplier — so nesting adds nothing a
flat learnable bank cannot already do. And it answers *why no state-space model implements
cross-frequency coupling*: not an oversight — it is redundant with what a flat complex-eigenvalue bank
learns for free. Attention remains the dominant mechanism for this capacity task (0.980 vs ≤0.58
recurrent).

## Scope & caveats

Toy scale (D≤16, K≤10, integer symbols, 3 seeds). The KILL is for the **SSM-native operationalization**
of nesting — a slow clock modulating fast-mode *input gates* (which stays parallel-scannable). A
fundamentally different coupling (state-dependent phase modulation, or spiking discrete slots) is not
tested and would break the linear scan; but input-gating is the natural, deployable form, and it is
redundant. What is settled: **explicit theta-gamma nesting, as an efficient SSM layer, does not beat a
flat oscillatory bank — the flat bank is already at the multiplexing ceiling.**

Discipline note: a firing capacity-headroom positive control + a frozen gate + no post-hoc turned the
moonshot into a clean, honest negative. STYXX ran the craziest version of the question and it held the
line — the discipline does not relax because the idea is big.
