# PREREG — mode-scarcity scaling of the adaptive-frequency win — 2026-07-23

**FROZEN before confirmatory data.** Runner: `run_scarcity_scale.py`. 1× RTX 4070. Deepens the one
clean result from the adaptive-frequency arc.

## Question

The arc found a *clean* adaptive-frequency win only under mode-scarcity (D=4: the windowed-conv detector
beat static by +0.129 and beat the diverse-bank oracle). That was a reported observation at the
non-primary budget of the ENTRAIN-RICH experiment. This is a **new, dedicated** pre-registration whose
primary budget IS the starved regime:

> On a HARDER, LONGER drifting-period task, does the windowed-conv adaptive-frequency detector cleanly
> beat a static bank at a starved budget — and does the advantage fall monotonically as the mode budget
> grows (the scarcity curve)?

This is not a post-hoc rescue of the ENTRAIN D=8 KILL: it is a fresh confirmatory test of a specific,
pre-stated hypothesis (adaptive frequency wins under scarcity), on a *harder* task than the original.

## Setup (frozen)

Harder task vs `run_entrain_rich`: `L=192` (was 96), `SEG_LEN=48` → **4 segments / 3 change-points**,
periods **[3,20]** (was [3,12]); score ≥2 local periods in. Detector: the arc's peak — a single
windowed conv (`KW=7`). Arms: **STATIC**, **RICH** (conv detector), **ORACLE** (diverse bank locked to
the true period; positive control). Single knob `κ=0`→STATIC bit-for-bit (red-team verified). 3 seeds,
1500 steps. `D_SWEEP = [2,4,6,8,12]`, **primary = the starved D=4**.

## Frozen gate (primary D=4)

`adv = RICH−STATIC`, `orc = ORACLE−STATIC` on drift mean-acc.

- **ABSTAIN** iff `orc < 0.10` (positive control silent at D=4).
- **CONFIRMED** iff `orc ≥ 0.10` **and** `adv ≥ 0.10` **and** `adv ≥ 0.5·orc` → the mode-scarcity
  adaptive-frequency win **scales** to a harder task.
- **KILL** iff `adv < 0.05` → the scarcity win did not scale.
- **WEAK** otherwise.

**Reported (corroboration):** the full scarcity curve `adv(D)` over `D∈{2,4,6,8,12}`; the advantage
should **fall** as D grows (adaptation matters most when modes are scarce) — a rising curve is a red
flag.

## Discipline

New prereg, frozen gate, firing positive control, no post-hoc D-selection (primary fixed at D=4 before
data). Smoke has over-predicted the full run three times this arc; the frozen 3-seed gate decides.
Result → `scarcity_scale_result.json` + `RESULT_`, OATH-certified.
