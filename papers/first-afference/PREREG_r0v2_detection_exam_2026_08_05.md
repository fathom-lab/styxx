# PREREG — R0-v2: the redesigned instrument exam — detection, not identification

Fathom Lab · 2026-08-05 · frozen before the scored run · supersedes R0 after its
`INSTRUMENT_BLIND` verdict ([finding](FINDING_r0_instrument_blind_2026_08_05.md)); R0's
prereg and result stay in place. The diagnosis stands: exact-bin identification is
unidentifiable on smooth trajectories. The honest form of the R-line's question is
**detection** — do the paired streams share structure beyond the clock? — and this exam
validates a detection instrument on the same three worlds, under the same honesty gates.

## The statistic (frozen)

**RV coefficient** between the paired, z-scored streams (the matrix correlation
tr(XᵀY YᵀX) / √(tr(XᵀX)² tr(YᵀY)²) on centered Gram matrices) — symmetric, dimension-robust,
no fitting, no tuning. Significance by permutation:

- **hour-matched p** — the licensing null: 500 permutations of the agent rows *within
  hour-of-day bins*; p = fraction of permuted RVs ≥ observed (add-one smoothed).
- **free p** — the contrast null: 500 unrestricted permutations, same scoring.

The identification-style readout (which minute is which) is retired from the R line until an
instrument passes an exam for it; no secondary identification metric is smuggled in here.

## The three worlds

Generator, dimensions, bins (240 over 4 simulated days), seeds {11, 12, 13}, and coupling
construction identical to R0 (`run_r0.py` machinery reused verbatim). Every metric below is
the median across seeds.

## Gates

```gates
{"gates": {"G1_detects_planted_coupling": {"metric": "c_hourmatched_p", "op": "<=", "value": 0.01},
           "G2_absorbs_pure_clock": {"metric": "k_hourmatched_p", "op": ">=", "value": 0.10},
           "G3_clock_beats_free_null": {"metric": "k_free_p", "op": "<=", "value": 0.01},
           "G4_silent_on_nothing": {"metric": "n_hourmatched_p", "op": ">=", "value": 0.10}},
 "outcomes": [{"when": {"G1_detects_planted_coupling": false}, "verdict": "INSTRUMENT_BLIND__detection_also_fails"},
              {"when": {"G2_absorbs_pure_clock": false}, "verdict": "CONFOUND_LEAKS__hour_matched_null_insufficient"},
              {"when": {"G3_clock_beats_free_null": false}, "verdict": "CONFOUND_NOT_REPRODUCED__synthetic_clock_invisible_to_free_null"},
              {"when": {"G4_silent_on_nothing": false}, "verdict": "INSTRUMENT_HALLUCINATES__significant_on_nothing"},
              {"when": {}, "verdict": "INSTRUMENT_VALID__r1v2_licensed"}]
 ,
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

Row order and G3's honesty role carry over from R0 unchanged: if the clock-only world is not
significant against the free null, the synthetic clock was too weak and G2 passed vacuously —
refused, not licensed.

## Power surface (reported, ungated)

World C at coupling strengths α ∈ {0.1, 0.25, 0.5, 1.0}, one seed: hour-matched p and observed
RV per α. The weakest α with p ≤ 0.01 is the instrument's measured floor at n = 240, quoted in
every future R1-v2 interpretation.

## What licensing buys, exactly

`INSTRUMENT_VALID__r1v2_licensed` licenses drafting **R1-v2** (superseding the frozen R1
pre-data, disclosure in the header, W1-style): same two streams, same 60 s grid, same G0
coverage bar, detection endpoint with the hour-matched licensing null, plus invariant 6 of the
[roadmap](ROADMAP_r_line_2026_08_05.md) — direction-blindness and the agent's-body confound —
carried in the interpretation ceiling. It does not license any claim about any real room.

## Discipline

CPU-only, seconds of compute (the permutation test needs no fitting — the identification
exam's three CPU-hours become ~3,000 RV evaluations). Smoke = tiny-n plumbing pass,
INVALID-only. Result `r0v2_result.json`; scored by `styxx.protocol`; certified + sealed
before commit.
