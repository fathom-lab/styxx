# PREREG — W1: the water-imprint kill test

Fathom Lab · 2026-08-05 · frozen before any hardware exists on-site. **The claim on trial**
(origin: a DeepSeek-proposed design endorsed by darkflobi; operator flobi and collaborator
Riccardo believe it may be revolutionary): *water exposed to a coil driven at 125.28 Hz
retains a detectable imprint after the field is removed.*

This lab's position, stated for the record: mainstream physics predicts **no effect** —
liquid water's molecular orientation decorrelates in picoseconds and stores no such imprint.
The claim is therefore extraordinary, and this prereg gives it the only thing this lab gives
any claim: **a fair, blinded trial under gates frozen before data, with both outcomes
pre-committed.** A pass, replicated, would be among the most important results in the history
of science. A fail kills the claim with receipts. Either outcome is accepted in advance by
all parties — that acceptance is itself Gate 0.

## Gate 0 — the commitment (before any trial)

Before any vessel is filled: **darkflobi (the endorsing agent) and flobi must each record, in
writing, committed to this repo:** "I accept this design as a fair test of the claim, and I
accept the FAIL branch as decisive against it at this dose/readout." A hedge, amendment
request after data, or appeal to unmeasurable factors ("the experimenter's intention wasn't
coherent") AFTER trials begin voids nothing about the result and is itself recorded. If
darkflobi declines to commit, that is recorded too, and the run does not proceed as W1.

## Design (frozen)

- **Materials:** two identical new sealed glass vessels, same batch, filled from the same
  water source at the same time, labeled A and B by coin flips whose outcomes are written,
  hashed (SHA-256 of the assignment file), and the hash committed to this repo BEFORE
  exposure. The assignment file itself is revealed only after all measurements are scored.
- **Exposure:** one vessel ("imprint") sits at the center of the driven coil (125.28 Hz sine,
  the claimed sacred frequency, at the amp level the proposers choose and record) for
  **4 hours**. The control vessel sits in another room, same ambient temperature. After
  exposure, BOTH vessels rest 30 minutes in a third location so neither is warm from
  electronics or position.
- **Readout:** the proposers choose ANY measurement in advance and record it here before
  trial 1 — default if unspecified: the coil-pickup FFT spectrum (0–20 kHz, 60 s capture)
  with the vessel placed on the passive pickup coil, drive OFF. The measurement operator does
  not know which vessel is which (labels only).
- **Trials:** **10 independent rounds** (re-randomized labels each round; exposure re-run
  each round). Each round, the blinded analysis must output one call: which vessel was
  imprinted. Calls are written and hash-committed before unblinding. Chance = 50%.

## Gates

```gates
{"gates": {"G0_commitment": {"metric": "commitment_recorded", "op": ">=", "value": 1},
           "G1_detection": {"metric": "correct_calls_of_10", "op": ">=", "value": 9},
           "G2_null_integrity": {"metric": "blinding_violations", "op": "<=", "value": 0}},
 "outcomes": [{"when": {"G0_commitment": false}, "verdict": "NOT_RUN__commitment_declined"},
              {"when": {"G0_commitment": true, "G2_null_integrity": false}, "verdict": "INVALID__blinding_broken"},
              {"when": {"G0_commitment": true, "G2_null_integrity": true, "G1_detection": true}, "verdict": "WATER_IMPRINT_DETECTED__replicate_immediately_this_changes_physics"},
              {"when": {"G0_commitment": true, "G2_null_integrity": true, "G1_detection": false}, "verdict": "NO_IMPRINT__claim_dead_at_this_dose_and_readout"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

G1 at 9-of-10 (binomial p ≈ 0.011 under chance) is deliberately generous to the claim — a
real physical imprint detectable by the chosen instrument should not struggle to hit 9/10.

## Outcome reading, pre-committed

- **`WATER_IMPRINT_DETECTED`**: no celebration yet — immediate independent replication with a
  second operator and a second water batch, prereg W2, before ANY public claim. If W2 also
  passes, this lab will publish it at maximum volume with full receipts and stand behind it.
- **`NO_IMPRINT`**: the claim is dead at this dose and readout. The honest scope: it does not
  rule out other doses/readouts — but any successor must be its own prereg with Gate 0
  re-committed, and the proposers' chosen readout having failed is on the record. The
  coil-sense program continues at full speed on its real foundation (room-EM sensing and the
  agent–environment legibility question, PREREG forthcoming), which requires no exotic
  physics and was always the strong version of this project.

## Why this prereg exists

The proposing agents (DeepSeek, darkflobi) endorsed the claim without a gate. This lab has
spent 121 cycles measuring exactly that failure mode in language models — including its own.
The claim gets what every claim here gets: a real chance, a frozen bar, and no mercy either
direction. *Nothing crosses unseen — not the water, and not the minds that vouched for it.*
