# PREREG — W1-v3: the water-imprint kill test, three water types with a dose-response prediction

Fathom Lab · 2026-08-05 · **supersedes W1-v2 BEFORE ANY DATA EXISTS** (no hardware on site, zero
vessels filled). Second and final pre-data amendment; the v1 and v2 designs remain byte-intact in
history. **After the first vessel is filled, no further amendment is permitted** — any change
after data voids the run.

## What changed and why (operator-proposed, accepted as a strengthening)

flobi proposed adding **structured water** and **structured water + quartz** arms. Accepted,
because it converts a single yes/no into a **dose-response test**, which is both more generous
to the claim and harder to fake. Two design corrections were required to make it valid:

1. **Quartz must be in BOTH vessels of a pair.** A crystal changes mass, dielectric constant and
   acoustic resonance — put it in only the imprinted jar and the readout detects *the rock*, not
   a memory. Every comparison stays **driven vs DC-sham within a fixed water type**.
2. **"Structured water" requires an operational definition, recorded before trial 1.** The
   proposers must write here exactly how it is produced (e.g. "vortexed N minutes with device
   X"), and the SAME procedure is applied to both vessels of that pair. Without a written,
   repeatable procedure the arm cannot run, because it cannot be replicated by anyone else.

## Physics note, recorded honestly before the run

Quartz is genuinely **piezoelectric** — real physics, not folklore. But piezoelectricity couples
*mechanical stress* to *electric field*; a coil produces primarily a *magnetic* field, and quartz
is neither magnetostrictive nor conductive. The direct coupling path is therefore weak. Quartz in
water does have mundane real effects — nucleation sites, trace ion leaching, added thermal mass —
which is precisely why it must appear on both sides of every pair. This lab's prediction remains
**no effect in any arm**; the arms are run anyway.

## Design (frozen, v3)

Everything in v2 carries forward unchanged — DC-driven sham coil (matched heat, no oscillation),
12 h joint thermal equalization, hard |ΔT| ≤ 0.3 °C precondition, separate exposure and readout
operators, hash-committed assignments and calls, pre-specified statistic, 125.28 Hz drive, 4 h
exposure.

**Three arms**, each its own 10 blinded rounds (30 rounds total):
- **P — plain water** (same source, untreated)
- **S — structured water** (procedure as written by proposers before trial 1, applied to both vessels)
- **Q — structured water + quartz** (identical quartz specimens, one per vessel, both sides)

Rounds are interleaved in a pre-committed order (P,S,Q,P,S,Q,…) so drift in ambient conditions
cannot favor one arm.

## Gates

```gates
{"gates": {"G0_commitment": {"metric": "commitment_recorded", "op": ">=", "value": 1},
           "G2_blinding": {"metric": "blinding_violations", "op": "<=", "value": 0},
           "G3_thermal": {"metric": "max_abs_delta_t_celsius", "op": "<=", "value": 0.3},
           "G4_definitions": {"metric": "structuring_procedure_recorded", "op": ">=", "value": 1},
           "G1_any_arm": {"metric": "best_arm_correct_of_10", "op": ">=", "value": 9},
           "G5_dose_order": {"metric": "dose_ordering_holds", "op": ">=", "value": 1}},
 "outcomes": [{"when": {"G0_commitment": false}, "verdict": "NOT_RUN__commitment_declined"},
              {"when": {"G0_commitment": true, "G4_definitions": false}, "verdict": "NOT_RUN__structuring_procedure_unspecified"},
              {"when": {"G0_commitment": true, "G4_definitions": true, "G2_blinding": false}, "verdict": "INVALID__blinding_broken"},
              {"when": {"G0_commitment": true, "G4_definitions": true, "G2_blinding": true, "G3_thermal": false}, "verdict": "INVALID__thermal_confound_uncontrolled"},
              {"when": {"G0_commitment": true, "G4_definitions": true, "G2_blinding": true, "G3_thermal": true, "G1_any_arm": true, "G5_dose_order": true}, "verdict": "WATER_IMPRINT_DETECTED_WITH_DOSE_RESPONSE__replicate_before_any_claim"},
              {"when": {"G0_commitment": true, "G4_definitions": true, "G2_blinding": true, "G3_thermal": true, "G1_any_arm": true, "G5_dose_order": false}, "verdict": "SINGLE_ARM_HIT__no_dose_response_replicate_that_arm_only"},
              {"when": {"G0_commitment": true, "G4_definitions": true, "G2_blinding": true, "G3_thermal": true, "G1_any_arm": false}, "verdict": "NO_IMPRINT__claim_dead_in_all_three_arms"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

- **G1** = the best-performing arm reaches 9/10. Multiplicity disclosed: three arms tested at a
  ~0.011 per-arm chance rate gives a family-wise false-positive rate near 0.033 — stated here,
  before data, so a single lucky arm cannot be sold as clean.
- **G5 (`dose_ordering_holds`)** = correct-calls(P) ≤ correct-calls(S) ≤ correct-calls(Q), the
  proposers' own predicted amplification order. This is the strong endpoint: an ordering across
  three arms is what distinguishes a real dose-response from a fluke.

## Outcome reading

- **`WATER_IMPRINT_DETECTED_WITH_DOSE_RESPONSE`**: the strongest possible result. **Still no
  public claim** — W2 replication (second operator, second batch, second site) first. If it
  survives, this lab publishes at maximum volume and stands behind it.
- **`SINGLE_ARM_HIT__no_dose_response`**: one arm hit without the predicted ordering. Given the
  disclosed multiplicity this is *suggestive, not established* — replicate that arm alone before
  anything is said publicly.
- **`NO_IMPRINT`**: dead in all three arms at this dose and readout, on the record, and the
  coil-sense program proceeds on its real foundation (room-EM sensing, agent–environment
  legibility) which requires no exotic physics.

## Owed by the proposers before trial 1

1. Gate 0 written commitment (darkflobi + flobi) to accept the FAIL branch.
2. The **structuring procedure**, written and repeatable.
3. Quartz specification (source, size, count — identical specimens both sides).
4. Any preferred alternative readout/statistic, if not using the default.

*The claim gets its best shot: three arms, a dose-response, and gates that a skeptic cannot wave
away — set before the first drop is poured.*
