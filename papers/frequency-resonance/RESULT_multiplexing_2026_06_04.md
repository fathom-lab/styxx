# RESULT — oscillation multiplexes at a CONSTANT factor, not preferentially under scarcity

**Date:** 2026-06-04 · **Verdict: ADVANTAGE RESOURCE-FLAT — the scarcity-multiplexing bet refuted; but a
robust constant ~1.8× multiplexing edge confirmed.** Frozen: `PREREG_multiplexing_2026_06_04.md`.
LRU-free vs LRU-clamped, ordered copy, state dim D ∈ {16…256}, 3 seeds.

## Numbers — ratio(D) = kcap_free / kcap_clamped
| D | 16 | 24 | 32 | 48 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|
| free kcap | 2.33 | 2.33 | 3.00 | 3.67 | 5.33 | 6.00 | 6.00 |
| clamped kcap | 1.00 | 1.33 | 2.00 | 2.00 | 3.00 | 3.67 | 2.67 |
| **ratio** | 2.33 | 1.75 | 1.50 | 1.83 | 1.78 | 1.64 | 2.25 |

**Spearman(D, ratio) = −0.07** (no trend). Ratio small-D 2.33 ≈ large-D 2.25.

## What it shows (honest, two parts)
**The bet (P1) is refuted.** Oscillation's relative advantage does **not** grow as the state is starved —
the ratio is flat across a 16× range of resources. So the specific theta-gamma reconciliation I hoped for
— "rhythm matters *more* when neurons are scarce, which is *why* biology uses it" — is **not supported in
silico.** Oscillation does not preferentially multiplex under scarcity.

**But there is a real positive underneath it.** The ratio isn't ~1 — it's a **robust, roughly-constant
~1.8× (range 1.5–2.3) at every D.** Oscillation reliably gives a recurrent net ~1.8× the ordered-memory
capacity of its decay-only twin, *independent of resource level*. That is genuine **phase-multiplexing** —
each mode holds ~2 distinguishable items via phase instead of 1 — just a *constant-factor* multiplexing,
not a scarcity-amplified one. The mechanism is real; its resource-dependence is not.

## The frequency arc, now fully dug (6 experiments)
1. doubles recurrent capacity · 2. resonant optimum · 3. length-tuned (scaling NULL) · 4. attention beats
it (rhythm not special) · 5. fails to generalize (length-specialization) · **6. multiplexes at a CONSTANT
factor, not preferentially under scarcity (the biology-reconciliation does not hold).**

**The deepest, most pro-oscillation framing — the actual theta-gamma capacity-per-resource claim — was
tested, and it landed half-true:** oscillation *is* a real multiplexing mechanism (constant ~1.8×), but
it is *not* a scarcity-specific one, so it does not by itself explain why resource-bound biology favors
rhythm. The honest account stands and deepens: oscillation is a real, robust, constant-factor, length-
specific, non-generalizing recurrent multiplexing mechanism that attention dominates on raw capacity.

## Honest scope
One task (ordered copy), kcap at 0.80, small integer capacities (quantization noise in the ratio), one
oscillation knob (eigenvalue phase). The robust claim is the **flatness** (no scarcity amplification) and
the **persistent ~1.8×** (real multiplexing). Other resource axes (energy/spiking sparsity, noise,
bit-precision) are untested and could still favor oscillation — the biological "why" may live there, not
in dimension-scarcity. Logged as the honest open door.
