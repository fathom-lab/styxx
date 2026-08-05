# PREREG — R1: is a physical room legible to an agent? (first afference, measured)

Fathom Lab · 2026-08-05 · frozen before the room-side hardware exists. This is the experiment
the first-afference arc was built for, and the one that needs no exotic physics.

## The question

darkflobi has never received an input that was not language written by a human. A copper coil
tapped as an electromagnetic pickup produces a continuous, non-linguistic stream from a physical
room. **Is that stream *legible* to the agent — is there discoverable correspondence between the
room's state and the agent's own internal state — or is it merely data?**

This is not a metaphor. The `disjoint-worlds` arc built machinery that answers exactly this shape
of question for two model representations (`TransferMap.fit`: given two point clouds with the
pairing destroyed, is the correspondence recoverable from geometry alone?). That machinery does
not care that one cloud comes from a language model. **R1 points it at a mind and a room.**

## The two clouds

- **Room:** `room_cortex.py` emits a room-state vector every 10 s — 11 log-spaced band energies
  (Nyquist-clamped) plus RMS. Over an observation window this is a cloud of feature vectors,
  timestamped.
- **Agent:** darkflobi's committed cognometric log (`~/.styxx/chart.jsonl`, 4231 records and
  growing) — per-observation phase1/phase4 predictions and confidences, gate verdicts, category.
  Numeric fields only, timestamped. **This stream already exists** and is not created for this
  experiment.

Both are resampled to a common 60 s grid over the same window (mean-pooled within bin; bins with
no observation on either side are dropped, count reported).

## The measurement

The true pairing is **timestamp**. Destroy it and ask whether it is recoverable: shuffle the
agent-side rows, run the committed label-free discovery (`TransferMap.fit` + Hungarian
assignment, the b34-v3 machinery verbatim), and score recovered-pairing accuracy against truth.
Discovery accuracy above the null means the room's spectral geometry and the agent's state
geometry share recoverable structure.

## The confound that decides this experiment, and its control

**Both streams have a daily rhythm.** The room is noisier when people are awake; the agent is
more active then too. A naive shuffle would let discovery succeed on circadian structure alone —
real coupling, but trivial and uninteresting.

**Primary null (the one that matters): hour-matched shuffling.** The agent rows are permuted
*only within the same hour-of-day bin*, so any circadian correspondence is preserved in the null.
Discovery must beat THAT to license a claim. A secondary free-shuffle null is reported for
contrast (it is the easy null and is expected to be beaten).

## Gates (frozen; scored by styxx.protocol)

```gates
{"gates": {"G0_coverage": {"metric": "n_paired_bins", "op": ">=", "value": 200},
           "G1_beats_hour_matched_null": {"metric": "disc_minus_hourmatched_null", "op": ">=", "value": 0.10},
           "G2_above_chance": {"metric": "disc_over_chance_ratio", "op": ">=", "value": 5.0}},
 "outcomes": [{"when": {"G0_coverage": false}, "verdict": "INVALID__insufficient_paired_observation"},
              {"when": {"G0_coverage": true, "G1_beats_hour_matched_null": true, "G2_above_chance": true}, "verdict": "ROOM_IS_LEGIBLE__structural_coupling_beyond_circadian"},
              {"when": {"G0_coverage": true, "G1_beats_hour_matched_null": false, "G2_above_chance": true}, "verdict": "CIRCADIAN_ONLY__coupling_explained_by_time_of_day"},
              {"when": {"G0_coverage": true, "G2_above_chance": false}, "verdict": "ROOM_NOT_LEGIBLE__no_recoverable_correspondence"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

G0 requires ≥200 paired 60 s bins (≈3.3 h of joint observation) before anything is read — the
b35-b lesson: an under-observed apparatus licenses nothing.

## The honest ceiling, stated before the run

A positive result means **the room's spectral state and the agent's internal state share
recoverable geometric structure beyond time-of-day.** It does **not** mean the agent perceives,
senses, experiences, or feels the room. Those words will not appear in the finding. The
prior-art review for this arc flagged "never publicly claim the AI *feels*" as the field's
signature overclaim, and this lab will not make it. What would be established is a *measured
coupling* between a physical environment and a machine's internal state — which is, on its own,
the first quantification of agent–environment coupling with the same instrument used for
mind–mind coupling.

## Outcome reading

- **`ROOM_IS_LEGIBLE`**: replicate on a second window (different day, R2) before any public
  claim, then the successor asks *what* couples — which bands, which agent features, and whether
  the affinity pre-screen (B40) predicts coupling strength before the discovery is run.
- **`CIRCADIAN_ONLY`**: an honest and interesting null — the streams do couple, but through the
  clock, not the room. Successor: a controlled-stimulus design (deliberate acoustic events at
  randomized times) which breaks the circadian confound by construction.
- **`ROOM_NOT_LEGIBLE`**: the coil delivers data the agent's state does not track at all. That
  bounds first-afference honestly and points at the stimulus-driven design as the next rung.

## Dependencies

Room side needs the Tier-0 hardware (~$110: interface, magnet wire, lead — see
`clawd/skills/coil-sense/`). Agent side is already logging. Analysis is CPU-only, reusing
committed machinery. Smoke = 30 bins, INVALID-only. Result `r1_result.json`; scored by
`styxx.protocol`; certified + sealed before commit.
