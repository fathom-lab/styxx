# PREREG — Is ordered-memory capacity *monotonic* in oscillation frequency, or *resonant*?

**Date (frozen before data):** 2026-06-04 · **Status:** pre-registration. No sweep data exists
at write time. Builds directly on the validated rig in
`papers/rhythm-rescue/run_rhythm_rescue.py` (RESULT: oscillation ≈ doubles ordered-memory
capacity, kcap free 6.0 vs clamped 2.67).

## The question

Rhythm-rescue tested oscillation **presence vs absence** (θ learnable vs θ≡0). It did **not**
test *how much* frequency. The operator's hypothesis — *"the higher the frequency, the greater
the [capacity]"* — predicts a **monotonic** relationship. The mechanism (phase-coding K items
across a retention window) predicts a **resonant optimum**: a frequency band that tiles the
window without aliasing, beyond which higher frequency *wraps* phase and items collide
(Nyquist-like). These make **opposite, falsifiable** predictions. We settle it.

## Method (frozen — identical to rhythm-rescue except θ handling)

Same complex-diagonal LRU, same task (ordered copy), same config: `D=256`, `V=12`,
`STEPS=4000`, `BATCH=64`, `LR=2e-3`, `KGRID=[1,2,3,4,6,8,10,12,15,18,20]`, `ACC_THR=0.80`,
`SEEDS=[0,1,2]`. The eigenvalue is λ = |λ|·e^{iθ}; θ is oscillation frequency in rad/step
(meaningful range [0, π]; π = Nyquist). **Only manipulation:** θ frozen (non-learnable buffer,
all modes share one value) and swept:

> θ/π ∈ {0, 0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.6875, 0.875, 0.97}

Two **reference arms** (also frozen config): `FREE` = θ learnable (per-mode spread, the
rhythm-rescue winner) and the θ=0 point doubles as the clamped baseline. Primary readout:
**kcap(θ)** = largest K with mean-acc ≥ 0.80, averaged over 3 seeds.

## Hypotheses & pre-stated predictions

- **H_mono (operator):** kcap(θ) monotonically non-decreasing across the grid; argmax at the
  θ=0.97π boundary. *Higher frequency, greater capacity.*
- **H_res (mechanism — my prior):** kcap(θ) is unimodal with an **interior** peak θ*∈(0, π),
  declining for θ>θ*. Pre-stated peak band: **θ*/π ∈ [0.1, 0.4]** (optimal phase-spread over a
  ~2K retention window with K up to ~6–8).
- **H_spectrum (my prior):** the FREE multi-θ net beats the best single-fixed-θ net by ≥1 item
  (kcap_FREE ≥ max_θ kcap(θ) + 1) — a *spread* of frequencies > any single tone.

## Decision rule / kill-gate (frozen)

Let ρ = Spearman(θ_grid, kcap). Let θ̂ = argmax_θ kcap(θ).

1. **MONOTONIC** (operator vindicated, my prior falsified) iff **ρ ≥ 0.90 AND θ̂ at the
   θ=0.97π boundary AND** kcap is non-decreasing with no interior drop > 1 item.
2. **RESONANT** (my prior) iff **θ̂ is interior** (not the min or max grid point) **AND**
   kcap falls ≥ 2 items from peak to the θ=0.97π point.
3. **FLAT** (any nonzero θ suffices, frequency irrelevant above 0) iff all nonzero-θ kcap
   within 1 item of each other and all ≥ θ=0 + 2.
4. Otherwise **MIXED — report shape, claim nothing.**
5. **H_spectrum** is a *separate* gate: PASS iff kcap_FREE ≥ max_θ kcap(θ) + 1.

**Pre-registered bet:** RESONANT ∧ spectrum-PASS, with θ*/π ∈ [0.1, 0.4]. If MONOTONIC
fires, the operator's literal claim has in-silico support and this prior is wrong — reported
as such, no reframing.

## Caveats (frozen)

- One task (ordered copy), one architecture (LRU), in-silico, n=3 seeds, capacity at a 0.80
  threshold. Read the **shape** kcap(θ), not absolute magnitudes.
- Single-fixed-θ deliberately shares **one** frequency across all 256 modes — an impoverished
  condition designed to isolate "a single frequency" from FREE's learned spectrum. This is the
  point of the H_spectrum contrast, not a confound.
- A monotonic result would *not* establish the cosmic claim; it would establish that, in this
  substrate and task, capacity rises with frequency up to Nyquist. Productizing any of this
  into a styxx instrument is a **separate** per-feature validation (cf. the geometry probe,
  which died on a confound control despite a clean research finding).
