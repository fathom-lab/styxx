# SYNTHESIS — The Adaptive-Frequency Ceiling in State-Space Models — 2026-07-23

Three controlled causal ablations of oscillation in sequence models, run in one day on one 8 GB GPU,
each frozen before data, each with a firing positive control, each OATH-certified. This is the first
*within-architecture* causal map of adaptive and nested frequency in state-space models — the
ablations the field's whole-architecture benchmarks (LinOSS vs Mamba) structurally cannot do.

## Why this is the open question

Oscillatory state-space models are a live frontier: LinOSS (ICLR 2025) beats Mamba ~2× on 50k-length
sequences with a bank of forced harmonic oscillators. But every such model uses **static** (fixed
after training) frequencies, and none isolates *the oscillation itself* from a decay baseline. STYXX's
own prior arc showed the optimum frequency is **resonant and locked to the input's timescale** — which
poses two obvious next primitives no SSM has tried:

1. **Adaptive frequency** — tune each oscillator to the input's timescale online.
2. **Nested coupling** — a slow rhythm gating fast modes into ordered slots (the brain's theta-gamma
   working-memory code).

STYXX has the one instrument that can test them cleanly: the **phase-clamp** — a single knob (θ→0, or
κ/α→0) that turns the mechanism off *bit-for-bit*, isolating its causal contribution while holding
everything else fixed.

## The three results

| Primitive | Mechanism | Verdict | The number |
|---|---|---|---|
| **ENTRAIN** | adaptive frequency, single-projection detector | **KILL** | +0.040 (23% of oracle) |
| **ENTRAIN-RICH** | adaptive frequency, windowed-conv detector | **WEAK** @ D=8 / **GREENLIGHT** @ D=4 | +0.085 (50%) / +0.129 |
| **ENTRAIN-HARM** | adaptive frequency, deep conv + harmonic spread | **KILL** | +0.002 (1%) — 8× params, collapse |
| **NEST** | theta-gamma nested coupling | **KILL** | +0.009 (5% of headroom) |

Positive controls fired in every case (the oracle bank locked to the true period beats static by
+0.17; doubling modes buys +0.18 of capacity). Attention dominated raw capacity throughout (kcap 10,
mean 0.98, vs ≤0.58 recurrent).

## The unifying finding

**In state-space models, oscillation is a real but bounded lever — and its two most-promising
extensions are each limited by something specific, not by the idea.**

- **Adaptive frequency genuinely works — but the detector has an *optimum*, not a monotone frontier.**
  The prize is real (a bank locked to the true drifting period beats static by +0.17). The
  dose-response over detector complexity is **non-monotone**: a single-lag projection captures 23%
  (KILL); a simple windowed conv captures **50%** (WEAK, the peak), and under mode-scarcity (D=4)
  captures *all* of it and beats the oracle (+0.129); but a deeper conv-with-harmonic-spread detector —
  8× the params, the oracle's own inductive bias — **collapses to 1%** (+0.002, KILL). More detector
  is worse. So the mechanism is sound and bounded: it is captured by a *simple* estimator, peaks at
  WEAK at a comfortable budget, wins cleanly only when the mode budget is starved — and cannot be
  bought with detector complexity.
- **Nested (theta-gamma) coupling is redundant.** A flat learnable oscillatory bank already sits at
  the phase-multiplexing ceiling; explicit slow-gates-fast coupling captures 5% of the capacity a
  wider bank buys. This answers *why no SSM implements cross-frequency coupling* — not an oversight,
  it is redundant with what a flat complex-eigenvalue bank learns for free. Capacity comes from **more
  modes, not cleverer coupling**.
- **The honest one-liner:** oscillation *adapts* where the budget is tight (a real, controlled win at
  D=4); everywhere with room to spare, a flat bank already captures it, and attention captures more.

## Positioning

Corroborates and sharpens the frontier: LinOSS/D-LinOSS (static frequency) leave the adaptive-frequency
regime open — we show it is reachable but detector-limited. TIDES (2605.09742) found *per-token*
frequency modulation hurts; our slow-loop entrainment converts that negative into a bounded positive
(the loop timescale matters). The nested-coupling KILL explains an absence in the literature rather
than filling a gap. Throughout, the contribution the benchmarks cannot make is the **causal**
one — clamp the oscillation off, hold all else fixed, measure ΔAcc.

## The discipline (the part that makes it trustworthy)

- **Three** over-optimistic smokes (1-seed / 400-step) were **caught** by the frozen 3-seed /
  1500-step gates — ENTRAIN −0.037→+0.040, RICH +0.120→+0.085, HARM **+0.110→+0.002**. No bar was
  moved; no favorable D was promoted to headline (D=4 wins are reported, not promoted); the deepest
  detector was pre-committed as the *last* swing (no detector #4 fished to cross the bar); the KILLs
  were shipped straight.
- The NEST positive control was **redesigned in smoke** (per-item slotting was found lossy → the
  wide-bank headroom control adopted) *before* freezing — no verdict was drawn from a silent control.
- Every recurrence is a parallel associative scan **red-team-verified equal** to the O(T) reference,
  and every reported number is OATH-certified against its receipt.

## The durable artifact

The phase-clamp / κ-clamp / α-clamp is a general **resonance profiler**: give it any trained
complex-eigenvalue SSM and it returns that model's *causal* reliance on oscillation (clamp θ→0) and on
adaptation (clamp the drive off), the within-architecture ablation the field skips — the causal
complement to correlational eigenvalue-spectrum analyses. That instrument, not any single layer, is
STYXX's on-brand deliverable here.

## Honest scope & the live next steps

Toy scale (D≤16, L≤96, integer symbols, 3 seeds). The "bigger detector" lever is now **falsified** —
the deep conv collapsed to +0.002 — so the open questions are about *regime*, not detector size, in
priority order: (1) the mode-scarcity budget where the clean D=4 win lives, at scale; (2) a
param-matched wider-static *efficiency* control for the RICH advantage; (3) ship the resonance profiler
and run it on a real LinOSS/Mamba checkpoint. Receipts: `PREREG_/RESULT_/CERT_` for `entrainment`,
`entrain_rich`, `nested_crossfreq`; `POSITIONING_entrainment_lit`.
