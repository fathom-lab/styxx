# PREREG — Phase-clamp ablation with rescue: is oscillation NECESSARY for ordered memory?

**Date:** 2026-06-03
**Status:** PRE-REGISTERED (gate frozen before any training).
**The decisive experiment of the rhythm sub-question** (Half B of the ancient-question
program). The literature says rhythm is "necessary-in-tissue, not in-principle," but the
clean controlled test — *remove the oscillation, hold everything else fixed, see if the
function is rescued* — has not been run on the function oscillation is MOST credited with:
**capacity-limited ordered working memory** (theta-gamma phase coding; Lisman & Jensen 2013).

## The operationalization (the key idea)

A complex-diagonal linear recurrence (LRU; Orvieto et al. 2023) has eigenvalues
**λ = magnitude · e^{iθ}**. The **magnitude is the "rate"** (decay/integration timescale);
the **phase θ is the "rhythm"** (rotation = oscillation). So "suppress oscillation while
holding rate & dynamics fixed" is *literally* **clamping θ = 0** — which collapses the
recurrence to real-eigenvalue pure decay (exactly Mamba-1, which we verified cannot
oscillate). This is the cleanest possible phase-clamp ablation, and it maps directly onto
our own eigenvalue hierarchy (FREE = complex like Mamba-3; CLAMPED = real like Mamba-1).

## Two networks (matched in everything but rhythm capacity)

Identical architecture: input embedding → single complex-diagonal LRU (hidden d, **complex
input projection B = B_re + iB_im in both arms**) → [Re;Im] (2d real state) → MLP readout →
vocab logits. Identical hidden size, magnitude parameterization (`mag = exp(-exp(ν))`),
**identical parameter count**, optimizer, and training budget.
- **FREE:** eigenvalue phase **θ is learnable** (complex eigenvalues → can rotate/oscillate;
  Re and Im channels mix through the rotation).
- **CLAMPED:** **θ ≡ 0** (real eigenvalues → no rotation; Re and Im become two *independent*
  real-decay channels). The **only** difference is whether the eigenvalues can rotate.

**Confound control (locked in):** because B is complex in *both* arms, both have the same
**2d real state dimensions** and the same parameters (θ exists in both; it is simply frozen
at 0 for CLAMPED). So a FREE advantage cannot be "more state" or "more params" — it can only
be the **rotation (oscillation) itself.** This is the clean isolation the design requires.

## Task (frozen)

**Ordered copy / delayed sequence recall** — the canonical capacity-limited ordered-memory
task. Present K random symbols (vocab V=12), a short delay + go signal, then require the
network to reproduce the K symbols **in order**. Trained on **mixed K ∈ [1, 20]** (one model
spans all K); evaluated **per-K**. Metric: **per-position accuracy** vs K. 3 seeds per arm.

Define **K_cap(model)** = the largest K at which mean per-position accuracy ≥ **0.80**.

## KILL-GATE (frozen — readings fixed before data)

- **RESCUE → rhythm is a substrate-specific MECHANISM:** `K_cap(CLAMPED) ≥ K_cap(FREE) − 2`
  AND CLAMPED mean-accuracy ≥ FREE − 0.05 across K ≤ K_cap(FREE), over ≥3 seeds. Oscillation
  is **not necessary**: real-eigenvalue (multi-timescale attractor) dynamics rescue ordered
  memory. *(Honest prior: LIKELY — real-diagonal S4/LRU/Mamba are strong on copy — which is
  exactly why a clean win here would confirm our hypothesis, not assume it.)*
- **NECESSARY → rhythm is a FUNCTION:** `K_cap(FREE) − K_cap(CLAMPED) ≥ 6`, robust across
  seeds. Oscillation/phase-coding extends ordered-memory capacity by a theta-gamma-scale
  margin that real dynamics cannot match → oscillation is necessary for this function. *(The
  surprising, theta-gamma-vindicating outcome — and a real possibility, since phase is a
  genuine extra coding dimension.)*
- **ADVANTAGE (quantified, neither extreme):** capacity gap of 3–5 items → oscillation
  *helps* but is not strictly necessary; report the magnitude.

**Capacity-fingerprint arm (secondary):** report the *shape* of accuracy-vs-K for each. Does
FREE show a sharper capacity cliff (a 7±2-like bound) while CLAMPED degrades gracefully — or
vice versa? Tests whether the capacity bound is an oscillatory fingerprint.

## Controls & honest caveats (frozen)

- Both arms trained to the same fixed budget; training curves reported (a CLAMPED failure
  must not be an optimization artifact — and real LRUs are *easier* to optimize, so this cuts
  against false "necessary").
- Matched hidden size; θ is a negligible parameter difference (noted above). A
  larger-d CLAMPED control can be added if FREE wins, to rule out raw-capacity confound.
- Single function (ordered memory) — the one most tied to oscillation. Binding and routing
  (which transformers already do without rhythm) are pre-registered **follow-up arms**.
- Linear recurrence + nonlinear readout (standard LRU); synthetic task; in-silico only.
- This tests *necessity for a function*, not biological mechanism — the in-silico arm of the
  "phase-clamp ablation with rescue" design; the in-vivo arm (rate-clamped optogenetics) is
  out of scope.

— frozen 2026-06-03
