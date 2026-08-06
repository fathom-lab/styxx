# PREREG — B48: the 45-pair legibility matrix — do *legibility*-defined islands recur?

Fathom Lab · 2026-08-06 · frozen before the scored run. B47 found no **affinity**-defined
islands in a ten-model cohort and recorded that as a negative against this lab's own
human-islands prediction. But islands were originally defined by **legibility failure**, not by
frame affinity, and b46 showed the two are joined by a switch — so a smooth affinity gradient can
still hide a bimodal legibility distribution. B48 measures the variable that actually matters, on
the same ten models, and either resurrects H1 or kills it.

## Design (frozen)

All **45 unordered pairs** of the ten models in `papers/mind-instrument/normeq_reps.npz` over the
shared 96-concept battery. Per pair, in **both directions** (legibility need not be symmetric):
destroy the item pairing by uniform shuffle, recover it label-free with the committed
`TransferMap.fit` + Hungarian machinery (`styxx_transfer.py`, k = min(60, dims, n−1), the arc's
defaults unchanged), and score against the truth the shuffle hid. Chance is 1/96 = 0.0104.

Per pair a **matched shuffled-geometry null**: one member's rows are independently permuted
before extraction of its frame, so a null fit sees the same dimensionality and scale with the
geometry destroyed. Seed 343 throughout; 90 discovery fits + 45 null fits.

## The honest hazard this design must survive, stated first

A single exploratory fit on this battery read a *same-family* pair at roughly six times chance —
far below the same-family reads on the 462-concept battery that produced the original matrix. **A
96-item battery may simply not carry enough geometry for label-free discovery.** If so, the whole
matrix sits near the floor and says nothing about islands: an all-floor matrix is an
uninformative instrument, not evidence of absence. G1 exists to catch exactly that and route it
to INVALID rather than to a conclusion.

## Gates

```gates
{"gates": {"G0_coverage": {"metric": "n_pairs", "op": ">=", "value": 45},
           "G1_signal_present": {"metric": "max_pair_legibility", "op": ">=", "value": 0.0521},
           "G2_null_clean": {"metric": "max_null_legibility", "op": "<=", "value": 0.0208},
           "G3_legibility_islands": {"metric": "bimodality_p_member_legibility", "op": "<=", "value": 0.05}},
 "outcomes": [{"when": {"G0_coverage": false}, "verdict": "INVALID__incomplete_matrix"},
              {"when": {"G0_coverage": true, "G1_signal_present": false}, "verdict": "INVALID__battery_carries_no_discovery_signal"},
              {"when": {"G0_coverage": true, "G1_signal_present": true, "G2_null_clean": false}, "verdict": "INVALID__null_leaks"},
              {"when": {"G0_coverage": true, "G1_signal_present": true, "G2_null_clean": true, "G3_legibility_islands": true}, "verdict": "LEGIBILITY_ISLANDS_RECUR__islands_generalize_beyond_the_first_cohort"},
              {"when": {"G0_coverage": true, "G1_signal_present": true, "G2_null_clean": true, "G3_legibility_islands": false}, "verdict": "NO_LEGIBILITY_ISLANDS__the_first_island_does_not_generalize"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

G1's bar is 5× chance, the same multiple the original matrix used to call a read real. G2's is
2× chance. G3 uses `styxx.islands`' gap screen on the ten per-member mean legibilities, reported
under its own name (not Hartigan's dip — see the erratum in the H1 prediction).

## Reported, ungated: the cliff on an independent cohort

B47 measured frame affinity for all 45 pairs; B48 measures legibility for the same 45. Their
relationship is a **direct, independent test of b46's switch** — does legibility rise sharply
with affinity, or smoothly? We report the scatter, a sigmoid fit, a linear fit, and the R²
difference. **Ungated**, because no measured prior exists for this construction (cross-pair,
not within-pair interpolation) and inventing a bar would repeat the b37-G2 error. This is also
the model-side analogue of H2 in the human prediction, and it is registered here as a
measurement rather than smuggled in later as a confirmation.

## Stated before the run

- **`NO_LEGIBILITY_ISLANDS` kills H1's model-side support.** It would mean the first island was a
  property of that cohort or that model, and the human prediction registered this morning should
  be amended a second time — downward, in place, publicly.
- **`INVALID__battery_carries_no_discovery_signal` is the most likely outcome given the pilot
  read**, and it is a statement about our battery, not about islands. It licenses nothing in
  either direction and the finding must say so plainly.
- Per-member and per-pair numbers ship in the receipt regardless of verdict, including the
  identity of any flagged member.

## Discipline

CPU-from-cache, zero model loads, ~10 minutes. Smoke = 3 pairs, INVALID-only. Result
`b48_result.json`; scored by `styxx.protocol` from this frozen block; certified + sealed before
commit.
