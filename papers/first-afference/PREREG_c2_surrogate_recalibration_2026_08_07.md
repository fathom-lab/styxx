# PREREG — C2: recalibrating `styxx.coupling` with spectral surrogates — the exam in both directions

Fathom Lab · 2026-08-07 · **frozen before the implementation exists.** The surrogate method is
specified below; no code implementing it has been written at the time of this commit, and the
Narratives data on disk has not been touched by any surrogate analysis.

## Why

`styxx.coupling` is publicly shipped and documented broken in both directions:

- before its autocorrelation refusal, it licensed **independent** first-order autoregressive
  streams at the permutation floor on every seed tried (red team, 2026-08-06);
- with that refusal, it refuses **genuine intersubject correlation** in 20 of 21 real fMRI
  pairs (`FINDING_c1_instrument_blind_to_isc_2026_08_06.md`), because a circular shift of
  heavily autocorrelated BOLD leaves shared slow structure, inflating the shift null.

The literature-standard cure is a **spectral surrogate** (Theiler et al. 1992; Schreiber &
Schmitz 1996): randomize Fourier phases while preserving each stream's full power spectrum, so
the null carries the data's exact autocorrelation — not the white noise a permutation implies,
and not the misaligned-but-still-shared structure a circular shift leaves. One change addresses
both failure directions, **if** it passes both exams. That conditional is this document.

## The method (specified before implementation)

`phase_randomize(B, rng)`: real FFT along time per column; multiply every frequency bin by a
random unit phase, the **same** phase across columns (preserving the stream's internal
cross-column structure); DC and Nyquist bins kept real; inverse FFT. The licensing p becomes
`max(matched_p, surrogate_p)` with `surrogate_p` from ≥ 200 phase-randomized draws of B scored
against A. The circular-shift null and the `INVALID__autocorrelation_defeats_the_permutation_null`
refusal are **replaced** by this (the refusal exists because circular shift was the wrong tool;
the surrogate is the right one and licenses or refuses on its own merits). All other refusals —
coverage, trend, leverage, sampling density, degenerate confound — are unchanged.

## The exam (all four gates must pass; any failure keeps the current refusal in place)

Data: the seven Narratives pieman subjects already on disk (all 21 pairs, 500 fixed vertices,
seed 343, quarter-of-run confound — identical to C1); plus synthetic attacks reproducing the red
team's constructions exactly.

```gates
{"gates": {"G1_finds_isc": {"metric": "frac_real_coupled", "op": ">=", "value": 0.80},
           "G2_rejects_reversed": {"metric": "frac_reversed_coupled", "op": "<=", "value": 0.10},
           "G3_rejects_independent_ar": {"metric": "frac_independent_ar_coupled", "op": "<=", "value": 0.10},
           "G4_rejects_shared_trend": {"metric": "frac_shared_trend_coupled", "op": "<=", "value": 0.10}},
 "outcomes": [{"when": {"G1_finds_isc": false}, "verdict": "STILL_BLIND__surrogate_does_not_recover_isc"},
              {"when": {"G1_finds_isc": true, "G3_rejects_independent_ar": false}, "verdict": "REGRESSION__autoregressive_false_positives_return"},
              {"when": {"G1_finds_isc": true, "G3_rejects_independent_ar": true, "G4_rejects_shared_trend": false}, "verdict": "REGRESSION__trend_false_positives_return"},
              {"when": {"G1_finds_isc": true, "G3_rejects_independent_ar": true, "G4_rejects_shared_trend": true, "G2_rejects_reversed": false}, "verdict": "FALSE_POSITIVE_ON_REVERSED__surrogate_leaks_alignment"},
              {"when": {"G1_finds_isc": true, "G2_rejects_reversed": true, "G3_rejects_independent_ar": true, "G4_rejects_shared_trend": true}, "verdict": "RECALIBRATED__licenses_isc_and_holds_every_prior_refusal"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

- **G3**: 20 pairs of independent AR(1) streams, rho 0.98, disjoint RNGs, n=336 hourly bins —
  the red team's exact critical attack. "Coupled" means `licensed == True`.
- **G4**: 10 pairs of independent streams with independent linear drifts (the 21/21 attack).
  The trend refusal stays, so these should be refused by it; the gate verifies no path around it.
- **G2**: time reversal on the real pairs — the hardest honest null from C1, unchanged.

## Stated before the run

- **Failure keeps the current state.** If any gate fails, the surrogate does not ship and
  `coupling` keeps its documented double-broken status with the C1 withdrawal in place. A
  recalibration that trades one failure direction for the other is not a fix.
- Even full success does **not** reopen the mind↔brain claim by itself: that requires the
  bin-resolution defaults fixed for the domain and a fresh exam on a second dataset. Success
  here licenses removing the `INVALID__autocorrelation...` refusal in favour of the surrogate
  null, nothing more.
- Per the standing rule (cycle 143), the implementation is **red-teamed before any release**,
  regardless of the exam verdict.

## Discipline

CPU, ~30 minutes. Smoke = 3 real pairs + 3 AR pairs, INVALID-only. Result `c2_result.json`;
scored by `styxx.protocol` from this frozen block; certified + sealed before commit.
