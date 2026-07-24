# PREREG — ENTRAIN-RICH: the richer-detector swing at the entrainment prize — 2026-07-23

**FROZEN before confirmatory data.** Runner: `run_entrain_rich.py`. 1× RTX 4070. Follows the ENTRAIN
KILL (`RESULT_entrainment_2026_07_23.md`) and pulls its single named un-taken lever.

## Question

The ENTRAIN kill-gate KILLed learned frequency entrainment at D=8 — but the ORACLE proved the prize is
real and large: a diverse bank locked to the true drifting period beats a static bank by **+0.17**
(growing to +0.22 at D=16). The cheap single-projection detector `ω̂=angle(z(t)·conj(z(t−1)))`
captured only **23%** of it. Frozen question:

> Does a RICHER phase detector — a learned causal 1-D conv over a short input window (`KW=7`)
> producing a PER-MODE target frequency, feeding the same slow PLL — capture the oracle's adaptation
> gap that the single-projection detector could not?

## Setup (frozen)

Identical drifting-period task, ORACLE (diverse bank locked to true period), and STATIC baseline as
`run_entrain_timing.py` (L=96, S=3 segments, periods [3,12], score ≥2 local periods in; 3 seeds, 1500
steps). Arms: **STATIC**, **RICH** (conv→per-mode freq→PLL), **ORACLE** (positive control). Single
knob preserved: `κ=0` reduces RICH to STATIC **bit-for-bit** (conv drawn after the read head; red-team
verified). Matched-compute (same scan recurrence); RICH adds the conv + κ (params reported). D∈{4,8},
primary **D=8** (the +0.17 prize regime).

## Frozen gate (primary D=8)

`adv = RICH−STATIC`, `orc = ORACLE−STATIC` on drift mean-acc.

- **ABSTAIN** iff `orc < 0.10` (positive control silent — should not happen, ENTRAIN measured +0.17).
- **GREENLIGHT** iff `orc ≥ 0.10` **and** `adv ≥ 0.10` **and** `adv ≥ 0.5·orc`. → **the win**: the
  first controlled demonstration that learned frequency adaptation gives an SSM a real edge on
  drifting-timescale sequences. Proceed to scale-up.
- **KILL** iff `orc ≥ 0.10` **and** `adv < 0.05` — the richer detector still fails; the KILL is
  **robust to detector richness**, a stronger negative (the static bank is genuinely the thing to beat).
- **WEAK** otherwise — improvement over the single-projection KILL, but below the bar; not a greenlight.

No-harm (fixed period): `RICH−STATIC ≥ −0.03` reported.

## Discipline

The prize is real (the oracle fired at +0.17 under a frozen gate); this pursues it with the one named
lever, not by torturing the data. Honest prior is uncertain: a windowed per-mode conv is far richer
than one projection and could plausibly capture the gap, but STYXX's timing receipt (oscillatory nets
win via a *static* bank, not by tuning θ→2π/P) leaves KILL live. The frozen gate decides. Result →
`entrain_rich_result.json` + `RESULT_`, OATH-certified.
