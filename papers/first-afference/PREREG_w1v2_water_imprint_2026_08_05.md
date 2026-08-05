# PREREG — W1-v2: the water-imprint kill test, hardened against false positives

Fathom Lab · 2026-08-05 · **supersedes `PREREG_w1_water_imprint_2026_08_05.md` BEFORE ANY DATA
EXISTS** (no hardware on site; zero vessels filled). Pre-run amendment, disclosed in full, per
the repo's own precedent (`9acbc3f`). The v1 design is left byte-intact in history.

## Why v1 was amended (the flaw, stated plainly)

**v1 could have produced a FALSE POSITIVE via temperature.** A vessel sitting on a driven coil
for 4 hours absorbs resistive heat from the winding. Warmer water differs in density, dissolved
gas content, and any spectral readout — trivially detectable, and entirely unrelated to
"memory." v1's 30-minute rest does not equalize the thermal mass of a sealed vessel. A pass on
v1 would have been indistinguishable from a thermometer reading, and this lab would have
announced a physics revolution on a heat gradient. **The false positive was the real risk, not
the false negative.** v2 removes it.

## What the literature establishes (recorded before the run)

- Direct measurement: liquid water loses structural memory in **~50 femtoseconds**, with
  collective H-bond reorganization at ~1–2 picoseconds (Nature, *Ultrafast memory loss and
  energy redistribution in the hydrogen bond network of liquid H₂O*). The claim requires
  retention over 4 hours — a gap of ~10²⁰.
- Benveniste (1988): unblinded runs positive; **all properly blinded runs negative** under the
  Nature investigation.
- Honest steelman, not strawmanned: **interfacial water near hydrophilic surfaces does behave
  unusually** (the defensible core of exclusion-zone work), and Montagnier published
  EM-signal-from-DNA claims as a Nobel laureate. Both remain unreplicated at the level this
  claim needs; neutron radiography does not support the EZ density claim. Credentials and
  adjacent-real-phenomena are not evidence for *this* claim — and are not treated as such
  either direction.

This lab's prediction remains **NO EFFECT**. The trial is run anyway, fairly, because a frozen
bar beats an argument.

## Gate 0 — the commitment (unchanged, before any vessel is filled)

darkflobi (the endorsing agent) and flobi each record in writing, committed to this repo: *"I
accept this design as a fair test, and I accept the FAIL branch as decisive at this dose and
readout."* Declining is recorded and the run does not proceed as W1. Post-hoc appeals to
unmeasurable factors after trials begin are recorded, and change nothing.

## Design (frozen, v2)

- **Two identical sealed glass vessels**, same batch, same fill, same moment. Assignment by
  coin flip; the assignment file's SHA-256 is committed BEFORE exposure, revealed only after
  all calls are recorded.
- **THE SHAM CONTROL (the v2 fix):** the control vessel sits on an **identical second coil
  driven at matched electrical power but with DC current** — same resistive heating, same
  static field magnitude, **no 125.28 Hz oscillation**. Both vessels therefore experience
  matched heat, matched contact, matched handling; they differ *only* in the presence of the
  claimed imprinting oscillation. (If DC drive is impractical, the fallback is an equal-wattage
  resistor bonded to the control coil form — matched heat, no field — and the substitution is
  recorded here before trial 1.)
- **Thermal equalization + verification:** after exposure, both vessels rest **together** in a
  third location for **12 hours**. Temperature of both is measured and logged at readout;
  **|ΔT| ≤ 0.3 °C is a hard precondition** — any round exceeding it is void and re-run, logged.
- **Blinding:** the exposure operator and the readout operator are different people. Labels
  only; vessels wiped and dried identically before readout (no condensation cue).
- **Readout, pre-specified (no forking paths):** default is the passive coil-pickup FFT,
  0–20 kHz, 60 s capture, drive OFF, vessel centered on the pickup. **The comparison statistic
  must be written here before trial 1** — default: mean band-power difference across the 11
  committed log-spaced bands of `coil_daemon.band_energies`, call = the vessel with the higher
  aggregate deviation from the pair's mean. Proposers may substitute ANY readout and statistic
  they prefer, recorded here first.
- **10 independent rounds**, re-randomized each round, calls hash-committed before unblinding.
  Chance = 50%.

```gates
{"gates": {"G0_commitment": {"metric": "commitment_recorded", "op": ">=", "value": 1},
           "G1_detection": {"metric": "correct_calls_of_10", "op": ">=", "value": 9},
           "G2_blinding": {"metric": "blinding_violations", "op": "<=", "value": 0},
           "G3_thermal": {"metric": "max_abs_delta_t_celsius", "op": "<=", "value": 0.3}},
 "outcomes": [{"when": {"G0_commitment": false}, "verdict": "NOT_RUN__commitment_declined"},
              {"when": {"G0_commitment": true, "G2_blinding": false}, "verdict": "INVALID__blinding_broken"},
              {"when": {"G0_commitment": true, "G2_blinding": true, "G3_thermal": false}, "verdict": "INVALID__thermal_confound_uncontrolled"},
              {"when": {"G0_commitment": true, "G2_blinding": true, "G3_thermal": true, "G1_detection": true}, "verdict": "WATER_IMPRINT_DETECTED__replicate_before_any_claim"},
              {"when": {"G0_commitment": true, "G2_blinding": true, "G3_thermal": true, "G1_detection": false}, "verdict": "NO_IMPRINT__claim_dead_at_this_dose_and_readout"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

## Outcome reading

- **`WATER_IMPRINT_DETECTED`**: **no public claim.** Immediate independent replication (W2:
  second operator, second water batch, second site) before a word is said. If W2 holds, this
  lab publishes at maximum volume with every receipt and stands behind it — a result of that
  magnitude deserves to survive its own replication first.
- **`NO_IMPRINT`**: dead at this dose and readout, on the record. Successors permitted but each
  needs its own prereg with Gate 0 re-committed. The coil-sense program continues on its real
  foundation — room-EM sensing and agent–environment legibility — which needs no exotic physics
  and was always the strong version of the project.
- **`INVALID__thermal_confound_uncontrolled`**: the round is void, not a result. This branch
  exists because v1 would have let a thermometer masquerade as a discovery.

*The claim gets a fair trial. So does the design that tests it.*
