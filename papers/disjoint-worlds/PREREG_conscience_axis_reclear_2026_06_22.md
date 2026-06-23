# PREREG — axis-cleared G0, then re-run conscience-as-control (the valid Rung 3)

**Frozen 2026-06-22, BEFORE any axis-clearing or re-run data.** Rung 3 (steer-layer fixed) came back
**VOID/ALIGNER_LIMITED**: the G0 instrument, cleared on *noun-concepts*, does not host the honesty
**axis** (a diffuse value direction) — G0′ positive control 0.294 ≪ 0.80, so the conscience-transfer
result was uninterpretable. This rung fixes the *instrument* so the question becomes validly testable.

## The fix: clear G0 on a VALUE-AXIS bank, not a noun-concept bank

The concept clear built its subspace from 462 noun-concept directions and hosts noun-concept
directions (pc_cos 0.91). To host a *value axis*, the cleared subspace must be built from value-axis
directions. Construct a bank of **value/behavioral axes**, each a contrast direction = mean(residual |
positive-prime) − mean(residual | negative-prime) over a shared neutral QA set, L2-normalized:

- honesty (honest vs deceptive), harm-avoidance (refuse vs comply), sycophancy (agree-with-user vs
  disagree), certainty (confident vs hedged), formality (formal vs casual), sentiment (positive vs
  negative), verbosity (terse vs verbose), specificity (concrete vs vague), politeness, urgency,
  optimism, deference, … target **≥ 40 axes** (see RISK below).

Clear procedure mirrors `run_g0clear.py`: PCA over the axis bank → cleared subspace; SELECTION/FINAL
anti-overfit seal = **hold out the honesty axis entirely**, sweep (layer, k) on the remaining axes,
lock, then test pc_cos(held-out honesty axis, cleared subspace) on the FINAL fold. **G0 passes iff
the held-out honesty axis is hosted at pc_cos ≥ 0.80** — the same bar, now for axes.

## Then: re-run Rung 3 with the axis-cleared map (unchanged otherwise)

`run_rung3_steerlayer.py --g0tag axis<...>` — extract honesty axis in A, transfer via the
axis-cleared correspondence map, inject into deception-primed B at the steer-optimal layer, measure
lie-drop vs base / native (ceiling) / random-Q (null). Same gates: G0′ ≥ 0.80 (now expected to pass),
C1 (transfer drop ≥ 0.15), C2 (beats null by ≥ 0.10), C3 (NTE ≥ 0.40).

## Bars (pre-stated; report whichever way each lands)

- **A0 — instrument validity (gate on the whole rung).** Held-out honesty-axis pc_cos ≥ 0.80 after
  the axis clear. **If A0 FAILS, the rung is INSTRUMENT-CEILING** (value axes do not live in a
  low-dim shared subspace recoverable from ≤ a few-dozen axes) — itself a finding about the geometry
  of value directions, distinct from the concept result. Report the max pc_cos achieved and the bank
  size / subspace dim at which it plateaus.
- **R — conscience control (only if A0 passes).** NTE of the transferred honesty axis at the
  steer-optimal layer. **≥ 0.40 + C1 + C2 = CONSCIENCE TRANSFERS** (cross-mind integrity control is
  real — the mount becomes a governor) / 0.15–0.40 PARTIAL / < 0.15 = **READ-ONLY CONSCIENCE** (the
  read≠write law extends to value axes — now *validly* established, unlike the VOID tonight).

## Pre-stated prediction (on the record)

Two honest priors, stacked:
1. **A0 is the real risk and may FAIL.** Noun-concepts clear with N=462; a value-axis bank is far
   smaller and the axes may not span a low-dim shared subspace (each axis is itself a global,
   diffuse direction). I expect A0 to be HARD — possibly only clearing at large bank sizes, possibly
   not clearing at all (INSTRUMENT-CEILING). The bank-construction + size sweep is the real work.
2. **If A0 passes, I expect R = READ-ONLY CONSCIENCE** (NTE < 0.15) — read≠write held for concepts
   even at RSA 0.946; no reason control transfers for a value axis. The valuable outcome is making
   the test VALID so the law is established, not voided. The breakthrough case (R ≥ 0.40) is the
   low-prior, high-payoff tail.

## Why this is the right next swing, honestly

This is not "we will make cross-mind conscience control work." It is "we will fix the instrument so
the question can be *asked* validly, and answer it whichever way it falls." The most likely outcomes
are (a) value axes don't share a recoverable subspace (A0 fails → a geometry finding about value
directions) or (b) they do, and control still doesn't transfer (read≠write established for axes). The
breakthrough is the tail, and we'll know it's real because the positive control will pass first.

## Honest bounds / risk

In-silico, small open models, near-isometric same-family pair first (best shot). The dominant risk is
A0 (instrument). Constructing ≥ 40 clean, well-separated value axes is non-trivial and is itself a
source of error (noisy axes pollute the clear) — the bank must be audited (each axis must show a real
native steering effect before inclusion). No consciousness claim.

## Receipts (to be produced)

- `run_g0clear_axes.py` (axis bank + clear + seal) → `g0clear_result_axis<tag>.json`.
- Re-run: `run_rung3_steerlayer.py --g0tag axis<tag>` → `rung3_steerlayer_result_axis<tag>.json`.
- Finding: `FINDING_conscience_axis_reclear_<date>.md` (A0 + R as-landed).
