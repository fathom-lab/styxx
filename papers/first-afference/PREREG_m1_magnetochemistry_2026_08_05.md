# PREREG — M1: does an audio-frequency magnetic field measurably alter a radical-pair reaction?

Fathom Lab · 2026-08-05 · frozen before any hardware or reagent is on site. Companion to the
W1 water-imprint test — **this is the version of the same intuition that has a known
mechanism.** Where W1 tests a claim mainstream physics predicts will fail, M1 tests a claim
mainstream physics predicts will *succeed at some field strength*, in a frequency band that is
genuinely underexplored.

## The mechanism (real, cited, not folklore)

The **radical pair mechanism** is established chemistry: when two radicals form together their
unpaired electron spins are correlated (singlet or triplet), the two spins interconvert under
hyperfine and applied magnetic fields, and recombination is spin-selective — so **a magnetic
field changes the product yield**. This underpins the leading account of avian magnetoreception
(cryptochrome/FAD; *Nature Communications* 2024, quantum-Zeno-enabled magnetosensitivity of
tightly bound pairs) and has been demonstrated as a synthetic chemical compass at
Earth-strength fields.

Critically for us: **magnetic field effects on the luminol chemiluminescent reaction in aqueous
solution are documented** — the field alters the recombination rate constant of luminol
radicals and thereby the light yield, and the radical lifetime in aqueous alkali is of order
**milliseconds**, measured via the phase shift of the MFE under amplitude-modulated fields.

A millisecond radical lifetime sits exactly in the audio band. **125 Hz has an 8 ms period.**
That is the coincidence worth testing: most magnetochemistry is DC or RF; the audio-frequency
regime where the modulation period is comparable to the radical lifetime is thinly studied.

## The question

**Does an audio-frequency (125.28 Hz) magnetic field measurably change the light output of the
luminol/H₂O₂ reaction, relative to field-off and to a DC field of matched magnitude?**

## Design (frozen)

- **Reaction:** standard luminol + hydrogen peroxide in alkaline aqueous solution with a
  catalyst (the common demonstration chemistry; exact recipe and concentrations recorded here
  before trial 1). Reagents mixed by syringe pump or fixed-volume pour into a cuvette seated in
  the coil's center bore, so mixing timing is identical across conditions.
- **Readout:** a photodiode/PMT (or a light-tight phone camera in a fixed rig with fixed
  exposure, if that is what is available — recorded before trial 1) integrating total light
  over a fixed window from mixing. **Primary measure: integrated luminescence per trial.**
- **Conditions, interleaved in a pre-committed order, N = 15 trials each:**
  - **F0** field off (coil present, unpowered)
  - **FAC** 125.28 Hz drive at recorded amplitude
  - **FDC** DC drive at matched RMS current — the control that separates *oscillation* from
    *field magnitude and heat*
- **Blinding:** the operator mixing reagents does not control which condition is active; a
  second person (or a script) sets the drive from a hash-committed randomized sequence, revealed
  after all trials are scored. Light readings are recorded automatically.
- **Temperature:** cuvette temperature logged per trial; **|ΔT| ≤ 0.3 °C across condition means
  is a hard precondition** — luminol yield is temperature sensitive and heat is the obvious
  confound (the same trap found in W1-v1).

## Gates (frozen; scored by styxx.protocol)

```gates
{"gates": {"G0_reaction_works": {"metric": "f0_mean_signal_over_dark", "op": ">=", "value": 10.0},
           "G1_thermal": {"metric": "max_abs_delta_t_celsius", "op": "<=", "value": 0.3},
           "G2_ac_effect": {"metric": "abs_ac_vs_off_percent", "op": ">=", "value": 5.0},
           "G3_ac_specific": {"metric": "abs_ac_vs_dc_percent", "op": ">=", "value": 5.0}},
 "outcomes": [{"when": {"G0_reaction_works": false}, "verdict": "INVALID__reaction_not_detected"},
              {"when": {"G0_reaction_works": true, "G1_thermal": false}, "verdict": "INVALID__thermal_confound"},
              {"when": {"G0_reaction_works": true, "G1_thermal": true, "G2_ac_effect": true, "G3_ac_specific": true}, "verdict": "AUDIO_FREQUENCY_MFE__oscillation_specific_effect"},
              {"when": {"G0_reaction_works": true, "G1_thermal": true, "G2_ac_effect": true, "G3_ac_specific": false}, "verdict": "FIELD_EFFECT_NOT_FREQUENCY_SPECIFIC__dc_matches_ac"},
              {"when": {"G0_reaction_works": true, "G1_thermal": true, "G2_ac_effect": false}, "verdict": "NO_MFE_AT_THIS_FIELD_STRENGTH"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

- **G0** is the positive control: the reaction must actually glow ≥10× dark baseline, or nothing
  downstream means anything (the b35-b lesson — an unvalidated apparatus licenses nothing).
- **G2** asks whether the field does anything at all; **G3** asks whether *oscillation* matters
  beyond raw field magnitude. Both use a 5% effect floor on integrated yield — small enough to be
  fair to a real effect, large enough that it cannot be shot noise at N=15.

## Outcome reading, pre-committed

- **`AUDIO_FREQUENCY_MFE`**: a genuinely publishable result — an oscillation-specific magnetic
  field effect on an aqueous radical-pair reaction at audio frequency, where the modulation
  period is comparable to the radical lifetime. Replicate (M2, second reagent batch, second
  operator) before any public claim.
- **`FIELD_EFFECT_NOT_FREQUENCY_SPECIFIC`**: the field matters, the frequency doesn't — still a
  real MFE observation, honestly bounded; kills the "sacred frequency" framing specifically.
- **`NO_MFE_AT_THIS_FIELD_STRENGTH`**: the coil's field is too weak to move this reaction.
  Reported with the measured field strength (mT at the cuvette) so the null has a number
  attached and a stronger-field successor can be sized from it.

## Why this prereg exists

The water-memory claim (W1) has no mechanism and 10²⁰ of timescale against it. **This claim has
a mechanism, a literature, and a live question.** If flobi and Riccardo want a genuinely novel
physics result from a copper coil, this is the one with a real chance — and unlike W1, a null
here is *informative about field strength* rather than merely confirming textbook physics.

Reagents are commodity (luminol demonstration kits are inexpensive); the coil, amplifier, and
photodiode are the same hardware the coil-sense program already needs.
