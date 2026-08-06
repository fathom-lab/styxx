# FINDING — R0: the instrument failed its own exam, and R1 stays blocked

Fathom Lab · 2026-08-05 · prereg: `PREREG_r0_instrument_validation_2026_08_05.md` (frozen
before the scored run) · receipt: `r0_result.json` · scored by `styxx.protocol`.

## Verdict (machine-computed)

**`INSTRUMENT_BLIND__cannot_detect_planted_coupling`** — the first failing gate names the
verdict, and the first gate failed.

| gate | frozen bar | measured | pass |
|---|---|---|---|
| G1_detects_planted_coupling | ≥ 0.10 | 0.0083 | ❌ |
| G2_absorbs_pure_clock | ≤ 0.05 | -0.0042 | (moot) |
| G3_clock_beats_free_null | ≥ 0.10 | -0.0042 | (moot) |
| G4_silent_on_nothing | ≤ 5.0 | 1.0 | (moot) |

On a **fully coupled** synthetic world — same latent driving both streams, generous
signal-to-noise — discovery reached 0.0167 at best against a 0.0042 chance floor: four times
chance, a factor of twelve short of the licensing margin. The power surface is flat
(margins -0.0083 / 0.0042 / 0.0083 from the weakest planted coupling to the strongest). The
pipeline that would have been pointed at darkflobi and a room cannot see coupling that is
*planted, maximal, and known to be there*.

## Why it is blind (the diagnosis, stated for the successor)

The disjoint-worlds machinery identifies points in a cloud of **distinct concepts** — 392
points with distinctive local geometry, which is what makes a destroyed pairing recoverable.
A resampled time series is not such a cloud. It is a **smooth trajectory**: the AR(1)
structure that makes the streams realistic also makes adjacent bins near-duplicates, so
exact-bin 240-way assignment is unidentifiable from geometry — any alignment that slides
along the trajectory fits almost as well. The pairing is destroyed by the world's own
autocorrelation before any shuffle touches it. Identification (*which minute is which*) is
simply a much harder task than the question R1 actually needs answered (*do these streams
share structure beyond the clock*).

Had R0 not existed, R1 would have run on real hardware, returned `ROOM_NOT_LEGIBLE`, and we
would have published "the coil delivers data the agent's state does not track" — an
instrument artifact wearing the costume of a scientific negative. This is the same class of
catch as W1-v1's warm jar: the apparatus, not the world, would have authored the verdict.

## What happens next (per the frozen prereg)

R1 stays blocked. Its gates are not touched. The pipeline is redesigned under a successor
prereg (R0-v2) and must pass the same three-world exam before anything faces reality. The
redesign lane, recorded here before it is built: replace exact-bin identification with a
**detection statistic** (dependence between the paired streams against the hour-matched
permutation distribution), with coarse temporal tolerance on any identification-style
secondary. Detection is the honest form of R1's question; identification was borrowed
prestige from the concept-cloud arc, and the exam caught the borrowing.

*The instrument asked for a license and was refused by its own frozen gates. That refusal is
the system working, at the cheapest possible price: three CPU-hours instead of a false
negative published about a mind and a room.*
