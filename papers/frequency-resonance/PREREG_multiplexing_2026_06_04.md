# PREREG — oscillation's real niche? capacity-per-resource: does the advantage grow under state scarcity?

**Date (frozen before data):** 2026-06-04. The deepest legitimate test of oscillation's role — the one
my prior experiments skipped. They used a generous fixed state (D=256) and matched parameters; attention
"won" partly because it has *unbounded* memory. But the actual theta–gamma claim is **capacity per
limited resource**: nested oscillations let scarce neurons hold many items by *multiplexing them across
phases of a cycle*. The test: **starve the state dimension D and watch whether oscillation's advantage
GROWS** — phase-multiplexing should matter most exactly when modes are scarce.

## Method (frozen)
- Two recurrent arms (the controlled oscillation knob), the rig from `run_rhythm_rescue.py`:
  **LRU-FREE** (θ learnable → oscillatory) vs **LRU-CLAMPED** (θ≡0 → decay only). Ordered copy, 3 seeds,
  identical training (STEPS=4000, BATCH=64, LR=2e-3).
- **Sweep the state dimension** D ∈ {16, 24, 32, 48, 64, 128, 256} (2D real state; the "neural resource").
  Readout hidden scales with D. kcap = largest K with mean acc ≥ 0.80.

## Hypotheses & predictions (frozen)
- **P1 — oscillation multiplexes under scarcity:** the RELATIVE advantage `ratio(D) = kcap_free / kcap_clamped`
  is LARGER at small D than at large D — specifically `ratio(16) ≥ ratio(256) + 0.5` (and the trend is
  monotone-ish downward in D). Mechanism: with few modes, only phase lets you pack >1 item per mode.
- **P2 — small-D oscillation beats large-D decay (the strong form):** there exists a small D where
  LRU-FREE kcap ≥ the LRU-CLAMPED kcap at a SUBSTANTIALLY larger D (oscillation buys capacity a decay net
  needs many more resources to match) — quantified post hoc, logged.

## Decision rule (frozen)
- **OSCILLATION'S NICHE IS RESOURCE-CONSTRAINED MULTIPLEXING** iff P1 — its relative advantage grows as
  the state is starved. This would be the genuinely *positive*, deeper finding: oscillation is not special
  for raw capacity (attention wins) but IS the mechanism for capacity-per-resource — *why biology, which
  is resource-bound, uses rhythm even though unconstrained silicon need not.* The reconciliation.
- **ADVANTAGE IS RESOURCE-FLAT** iff ratio(D) is roughly constant across D — oscillation gives a fixed
  multiplier regardless of scarcity (less interesting; multiplexing-under-scarcity not supported).
- Report the full ratio(D) curve regardless.

## Caveats (frozen)
- One task (ordered copy), recurrent-only comparison (the clean oscillation knob; attention has no fixed
  state to starve fairly here). Capacity at the 0.80 threshold; read the *trend* in the ratio, not exact
  kcaps. This tests whether oscillation's advantage is *resource-dependent*, the core of the biological
  multiplexing claim — a genuinely positive-direction test after four refutations.
