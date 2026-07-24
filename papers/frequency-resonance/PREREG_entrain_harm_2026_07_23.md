# PREREG — ENTRAIN-HARM: dose-response point #3 (the oracle's inductive bias) — 2026-07-23

**FROZEN before confirmatory data** (smoke/plumbing only). Runner: `run_entrain_harm.py`. 1× RTX 4070.
Completes the adaptive-frequency dose-response (single-projection 23% → windowed-conv 50% → this).

## Question

The ORACLE wins (+0.17) by placing a DIVERSE bank at a geometric spread around the TRUE fundamental.
This detector is given that exact inductive bias:

> A rich 2-layer causal conv estimates a SINGLE fundamental `f_hat(t)` (constrained to the plausible
> band `[2π/PMAX, 2π/PMIN] ≈ [0.52, 2.09]`), and each mode tracks `spread_j · f_hat(t)` with a LEARNED
> per-mode spread (init geometric 0.5..2.0 — the oracle's). Does it capture more of the oracle gap than
> the plain windowed conv (which captured 50%)?

## Pre-data design choices (from smoke plumbing, no confirmatory data used)

The **band constraint** on `f_hat` is a principled fix, not a tuned knob: an unconstrained `π·sigmoid`
fundamental averages ~π/2 and, times the spread, clamps high-spread modes to **π = Nyquist** — which
this arc's resonance finding (`project_frequency_resonance`) established is the capacity *minimum*.
Constraining `f_hat` to the plausible-period band keeps the detector out of the Nyquist-collapse
region. Single knob preserved: `κ=0` reduces HARM to STATIC **bit-for-bit** (detector drawn after the
read head; red-team verified).

## Setup & gate

Identical drifting-period task, ORACLE, STATIC, and gate as `run_entrain_rich.py` (L=96, periods
[3,12], 3 seeds, 1500 steps, D∈{4,8}, primary **D=8**). `adv = HARM−STATIC`, `orc = ORACLE−STATIC`:

- **GREENLIGHT** iff `orc ≥ 0.10` **and** `adv ≥ 0.10` **and** `adv ≥ 0.5·orc` → **the flagship win**:
  learned frequency adaptation with the oracle's inductive bias gives an SSM a real edge.
- **WEAK** iff `0.05 ≤ adv < 0.10` or capture < 50% → dose-response point #3, still sub-threshold.
- **KILL** iff `adv < 0.05`.

## Pre-commitment (anti-fishing)

This is the **LAST** detector swing. Whatever the full 3-seed gate returns, the 3-point dose-response
(23% → 50% → this) is the reported finding; **there will be no detector #4** selected to cross the bar.
Honest prior: the smoke is over-optimistic — the windowed-conv detector read +0.120 in smoke and
**+0.085** in the full run, so a smoke GREENLIGHT (+0.110) can land WEAK. The frozen gate, not the
smoke, decides. Result → `entrain_harm_result.json` + `RESULT_`, OATH-certified.
