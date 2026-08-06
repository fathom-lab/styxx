# PREREG — R0: validate the R1 instrument on synthetic worlds, before any hardware exists

Fathom Lab · 2026-08-05 · frozen before the scored run. R1 will point the disjoint-worlds
discovery machinery at a mind and a room. The b35-b lesson says an under-observed apparatus
licenses nothing; the same holds for an unvalidated one. R0 runs the **exact R1 pipeline**
(`run_r1.py`, committed before any room data exists) on synthetic worlds whose ground truth is
known, and asks whether the instrument can (1) see a real coupling, (2) refuse a pure-clock
confound, and (3) stay silent on nothing. **R1 does not run until R0 licenses it.** If R0
fails, R1's prereg stays frozen and the pipeline is redesigned under a new prereg — the gates
of R1 itself are never touched.

## The three worlds (n = 240 one-minute bins scattered over 4 simulated days)

All worlds share the generator skeleton: a smooth AR(1) latent, an hour-of-day clock signal,
random nonlinear (tanh) projections to a 12-dim "room" and a 24-dim "agent", observation noise.
Seeds {11, 12, 13}; every metric below is the median across seeds.

- **World C (coupled):** room and agent both driven by the SAME latent plus the clock. The
  instrument should detect coupling beyond the clock.
- **World K (clock-only):** room and agent driven by INDEPENDENT latents plus the same clock.
  Real correspondence exists — but it is entirely circadian. The hour-matched null must absorb
  it; the free-shuffle null must be fooled by it (proving the confound is real and the matched
  null is necessary, not decorative).
- **World N (nothing):** independent latents, no clock. Everything must sit at chance.

## Gates

```gates
{"gates": {"G1_detects_planted_coupling": {"metric": "c_disc_minus_hourmatched_null", "op": ">=", "value": 0.10},
           "G2_absorbs_pure_clock": {"metric": "k_disc_minus_hourmatched_null", "op": "<=", "value": 0.05},
           "G3_clock_beats_free_null": {"metric": "k_disc_minus_free_null", "op": ">=", "value": 0.10},
           "G4_silent_on_nothing": {"metric": "n_disc_over_chance_ratio", "op": "<=", "value": 5.0}},
 "outcomes": [{"when": {"G1_detects_planted_coupling": false}, "verdict": "INSTRUMENT_BLIND__cannot_detect_planted_coupling"},
              {"when": {"G2_absorbs_pure_clock": false}, "verdict": "CONFOUND_LEAKS__hour_matched_null_insufficient"},
              {"when": {"G3_clock_beats_free_null": false}, "verdict": "CONFOUND_NOT_REPRODUCED__synthetic_clock_invisible_to_free_null"},
              {"when": {"G4_silent_on_nothing": false}, "verdict": "INSTRUMENT_HALLUCINATES__above_chance_on_nothing"},
              {"when": {}, "verdict": "INSTRUMENT_VALID__r1_licensed"}]
 ,
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

Rows are ordered: the first failing property names the verdict. The bars mirror R1's own frozen
gates (0.10 margin over the hour-matched null; 5.0× chance) so that "valid" means *valid at the
bars R1 will actually use*, and G2's 0.05 leaves deliberate daylight below R1's 0.10 — a null
that leaks half of R1's licensing margin on a pure-clock world is not a control.

- **G3 is the honesty gate on the synthetic world itself.** If the clock-only world does not
  fool the free null, the world was too easy and G2 passed vacuously — that outcome refuses
  rather than licenses.

## Power surface (reported, ungated)

World C is regenerated at coupling strengths α ∈ {0.25, 0.5, 1.0} (one seed) and
`disc_minus_hourmatched_null` reported per α — the minimum detectable coupling at n = 240,
stated so an R1 null can be read honestly as "no coupling above the instrument's measured
floor," never as "no coupling."

## Outcome reading

- **`INSTRUMENT_VALID__r1_licensed`**: R1 may run when the room recorder exists. The measured
  power floor travels with every future R1 interpretation.
- **Any failure verdict**: R1 stays blocked; the pipeline (not R1's gates) is redesigned under
  a successor prereg (R0-v2). An instrument that cannot pass a synthetic exam does not get
  pointed at reality.

## Discipline

CPU-only, no hardware, no model loads; ~30 discovery fits. Smoke = tiny-n plumbing pass,
INVALID-only. Result `r0_result.json`; scored by `styxx.protocol` from this frozen block;
certified + sealed before commit. The generator, worlds, and all seeds ship in `run_r0.py`.
