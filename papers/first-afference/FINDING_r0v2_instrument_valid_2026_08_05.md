# FINDING — R0-v2: the redesigned instrument passes the exam that blinded its predecessor

Fathom Lab · 2026-08-05 · prereg: `PREREG_r0v2_detection_exam_2026_08_05.md` (frozen at
`9969dc4` before the scored run) · receipt: `r0v2_result.json` · scored by `styxx.protocol`.

## Verdict (machine-computed)

**`INSTRUMENT_VALID__r1v2_licensed`** — all four gates pass, on the same three synthetic
worlds, same seeds, same generator that returned `INSTRUMENT_BLIND` for the identification
pipeline hours earlier.

| gate | frozen bar | measured (median across seeds) | pass |
|---|---|---|---|
| G1_detects_planted_coupling | hour-matched p ≤ 0.01 | 0.002 | ✅ |
| G2_absorbs_pure_clock | clock-world hour-matched p ≥ 0.10 | 0.4391 | ✅ |
| G3_clock_beats_free_null | clock-world free p ≤ 0.01 | 0.002 | ✅ |
| G4_silent_on_nothing | nothing-world hour-matched p ≥ 0.10 | 0.5609 | ✅ |

The G2/G3 pair is the exam's heart, and it landed exactly as a working control should: the
pure-clock world is **maximally significant against the free null** (p 0.002 — the circadian
confound is real and would fool a naive analysis) while the **hour-matched null absorbs it
entirely** (p 0.4391). The licensing null does its one job.

## Power floor (measured, travels with every future R1-v2 interpretation)

The instrument detects planted coupling down to the second-weakest strength tested
(hour-matched p 0.002); the weakest just misses (p 0.0619). An R1-v2 null therefore reads
"no coupling above this measured floor at n = 240," never "no coupling."

## What one afternoon under frozen gates looked like

1. Identification instrument built, examined — **failed** (`INSTRUMENT_BLIND`, published).
2. Diagnosed: exact-bin assignment is unidentifiable on smooth trajectories; detection is the
   honest form of the question.
3. Detection instrument preregistered, examined — **passed**, at roughly a thousandth of the
   compute (permutation RV needs no fitting).

No real-world data was touched at any point. The cost of the caught error was three CPU-hours
and two documents; the cost of the uncaught error would have been a published false negative
about a mind and a room.

## What is licensed, exactly

Drafting **R1-v2** (superseding R1 pre-data, disclosure in its header): detection endpoint,
hour-matched licensing null, the G0 coverage bar unchanged, and the roadmap's invariant 6 —
direction-blindness and the agent's-body confound — written into the interpretation ceiling.
Nothing about any real room is claimed or licensed by this document.
