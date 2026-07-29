# PREREG -- `--dry` THRESHOLD SENSITIVITY CENSUS (coupling attempt 4 harness)

**Frozen 2026-07-19, autopilot cycle 47. CPU-only, zero GPU, treatment-blind (no scored run exists).**
**Committed BEFORE the probe is written or run.**

## The owed item this discharges

Panel #8 left four zero-GPU items owed before freeze. This is one of them, stated verbatim in the
cycle-46 ledger:

> `--dry` fixtures that can DETECT a threshold change (suite returns 52/52 with the bar halved -- so
> "52/52" is not evidence about any constant).

## Why this and not another owed item

Panels #7 and #8 both killed **threshold moves** -- and the ledger already names why that class is
the dangerous one: it is *"the class of change invisible after the run"*. The harness's own
regression suite is currently the only thing standing between a frozen constant and a silent edit,
and the one spot-check on record says it is blind. This is the program's own standing rule --
*make the gate prove it READS THE DATA: vary the input, require the output to move* -- applied to
the test suite itself. The suite is a gate; its input is the threshold block; nobody has ever asked
whether its output depends on that input.

## What is measured

For each constant, perturb it in the module namespace, re-run `run_dry()`, and ask whether the
suite noticed. Nothing in `coupling_confirm_v4.py` is edited: the probe imports the module and
patches globals, so the committed harness is byte-identical before and after.

### Scope (mechanical, fixed here so it cannot be curated later)

**IN SCOPE -- the THRESHOLD census.** Every entry of `_thresholds()` whose value is a number or a
list of numbers, MINUS the exclusions below. Enumerated from the function, not hand-picked.

**EXCLUDED as NON-THRESHOLD, and excluded HERE, before any result is seen:**

- `k_perm`, `estimator_pseudo_runs`, `estimator_rng_seed` -- **resolution constants**. Perturbing
  them changes Monte-Carlo draws, so a detected change would demonstrate resampling noise, not a
  guard. Probed and reported in a separate block; counted in neither census column.
- `derive_seeds`, `gate_seeds`, `arms_key` -- design partition / identity, not thresholds. A change
  here is a different experiment, not a moved bar.

### Perturbation rule (frozen -- this is the constant that could be gamed)

Magnitude is the one knob that decides the answer: large perturbations flatter the suite, small ones
condemn it. It is fixed now, mechanically, and applied to every constant alike:

- **scalar float**: `0.5x` and `2.0x`.
- **integer count**: `-1` and `+1` (floored at 1).
- **`rank_span` (2, 8)**: `(2, 6)` and `(2, 10)`.
- **`injection_grid`**: all entries `0.5x`, and all entries `2.0x`.

Both directions are applied to every constant, and a constant counts as guarded if **either**
direction is detected. This is the reading most favourable to the suite; it is chosen deliberately,
so that an UNGUARDED verdict cannot be an artifact of having probed the wrong way.

### Detection rule

A perturbation is **DETECTED** iff, versus the unperturbed baseline, any of:

1. `all_ok` flips True -> False;
2. any individual check's `ok` field differs;
3. any fixture's verdict string differs;
4. the run raises (a crash is the suite noticing).

A constant is **GUARDED** if >= 1 of its perturbations is DETECTED, **UNGUARDED** if none is.

## Frozen falsifiable prediction

**`min_effect_slope` will come back UNGUARDED** -- this is panel #8's observation ("52/52 with the
bar halved") and the reason the item is owed.

**If it comes back GUARDED, panel #8's claim is wrong on its own headline example and this prereg
says so in those words.** The census is not built to confirm the panel.

## Verdict form -- this is a CENSUS, not a hypothesis test

The deliverable is the per-constant GUARDED/UNGUARDED partition. Pre-committed rails on how it may
be read:

- **No aggregate pass/fail, and no guarded-fraction reported as reassurance.** "22 of 26 guarded"
  is not a result; *which four* is the result.
- **Every UNGUARDED claim-bearing constant is a real defect**, whatever the count. It means the
  frozen value could be edited -- before the run or after it -- and `--dry` would still print
  `all_ok=True`.
- A census finding **no** unguarded constants is a legitimate outcome and would retire the item.

## Forbidden remedies (frozen before the numbers exist)

- **FORBIDDEN: changing any threshold's VALUE in response to this census.** The remedy for a blind
  spot is a fixture. R5.8 already forbids moving `DISJOINT_FLOOR_CLEAN` / `MIN_DISJOINT`; this
  extends the same rule to every constant in scope, for the duration of attempt 4.
- **FORBIDDEN: counting a value-assertion as a guard.** `assert MIN_EFFECT_SLOPE == 0.0152` would
  detect every perturbation and prove nothing about whether the constant is load-bearing -- it is
  the obvious cheat against this exact census. Such a pin is recorded in a separate, explicitly
  WEAKER remedy class and is never counted in the GUARDED column.
- A guard that counts must be **BEHAVIOURAL**: a fixture whose verdict moves because the constant
  moved.

## Scope of the cycle

The census is the deliverable. Fixtures for whatever it finds are the named next step, not this
cycle's work -- writing the guard and grading it in the same pass is how a suite gets tuned to
its own probe.

## Standing state unaffected

`PARTIAL__coupling_seed_split` remains the program's position on coupling. The static (c36) /
adaptive (c37) / 3B (c40) SURVIVES results are untouched -- none gates on this harness. Attempt 4
remains UNFROZEN and its GPU run OPERATOR-GATED; this probe touches neither.
