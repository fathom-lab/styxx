# FINDING — protocol v2: INVALID, because I could not count consistently between my own prereg and my own harness

Fathom Lab · 2026-08-07 · prereg: `PREREG_protocol_power_basis_2026_08_07.md` (frozen before
implementation) · receipt: `protocol_power_basis_result.json` · scored by `styxx.protocol`.

**`INVALID__breaks_existing_preregs`.** The code is fine. The exam is not.

| gate | bar | measured | pass |
|---|---|---|---|
| G1_backward_compatible | ≥ 33 | 32 | ❌ |
| G2_strict_mode_refuses | ≥ 1 | 0 | ❌ |
| G3_declared_passes_strict | ≥ 1 | 1 | ✅ |
| G4_verdict_strings_unchanged | ≤ 0 | 0 | ✅ |

## Two harness errors, both mine, and the first is the same error a fourth time

**G1.** The bar was set from a census of files containing a gates block — thirty-four, as it
turns out, not the thirty-three I wrote down. The harness then measured only files *named*
`PREREG_*.md`, of which there are thirty-two. The two gates-bearing documents outside that
naming convention are `FINDING_b36_write_door_2026_08_01.md` and
`PREDICTION_h1_human_islands_2026_08_06.md`. So the bar counted one population, the metric
counted a smaller one, and the census itself was miscounted. **Nothing was broken; I compared
two different numbers and called the difference a regression.**

That is a bar mis-specified against what was actually measured — the fourth instance this week
after b37 G2, b48 G2 and C5 G1, and it happened *on the preregistration written specifically to
stop it*. The `power_basis` string I attached to G1 says "count of gates-bearing preregs in the
repo at freeze time; the bar IS the census, so achievable by construction." Declaring a power
basis did not make the declaration true. **The machinery now records the claim; it cannot check
it.** That limit is the most useful thing in this document.

**G2.** The harness tested strict mode on `preregs[0]` — alphabetically the *declared* prereg
itself, which correctly does not refuse. A test that picks its specimen by sort order tested the
wrong specimen. The mechanism does work: verified by hand, `Experiment(<any legacy prereg>,
require_power_basis=True)` raises and names every undeclared gate.

## What actually holds

**G4 passed at exactly zero.** Every committed result was re-scored against its prereg and no
verdict string changed anywhere in the repo, which was the one non-negotiable requirement — the
33 sealed findings keep verifying byte-identically. And 31 of 32 preregs are now measured as
carrying undeclared gates, which is the census the change existed to produce.

## What happens next

The verdict stands. **The harness is not re-run with adjusted numbers** — that would be
gate-shopping, and the whole point of this change is that a bar chosen after seeing the data is
not a bar. A successor prereg fixes the metric definition to match the population it names,
selects its strict-mode specimen explicitly rather than by sort order, and states plainly that
`power_basis` is an *unverified declaration* rather than a checked computation.

*Frozen before implementation; the losing branch landed; the failure is in the exam I wrote to
prevent this exact failure. Every number grounds in `protocol_power_basis_result.json`.*
