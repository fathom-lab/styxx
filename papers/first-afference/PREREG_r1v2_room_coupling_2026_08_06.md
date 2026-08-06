# PREREG — R1-v2: is the room coupled to the agent? (detection, licensed by R0-v2)

Fathom Lab · 2026-08-06 · **supersedes R1 BEFORE ANY ROOM DATA EXISTS** (no hardware on site;
R1-v1 remains frozen-and-superseded in place, W1-style). The change is the endpoint: R0
([finding](FINDING_r0_instrument_blind_2026_08_05.md)) proved exact-bin identification
unidentifiable on smooth trajectories, and R0-v2
([finding](FINDING_r0v2_instrument_valid_2026_08_05.md)) licensed the detection instrument
this prereg uses. Streams, grid, coverage bar, and the honest ceiling all carry over from
R1-v1 unchanged.

## The measurement

The two streams and 60 s mean-pooled grid of R1-v1 (room: the recorder's 12-dim spectral
vector; agent: the 24-dim vector fixed in `run_r1.py`, committed 2026-08-05, pre-data).
On the paired bins: **RV coefficient**, with two permutation nulls exactly as validated in
R0-v2 (500 permutations each):

- **hour-matched p** — the licensing null (circadian structure preserved in the null);
- **free p** — the contrast null (detects that *some* correspondence exists, clock included).

Machinery: `run_r1v2.py`, which imports the loaders of `run_r1.py` and the permutation test
of `run_r0v2.py` verbatim — nothing new is fit, tuned, or chosen at analysis time.

## Gates

```gates
{"gates": {"G0_coverage": {"metric": "n_paired_bins", "op": ">=", "value": 200},
           "G1_beats_hourmatched_null": {"metric": "hourmatched_p", "op": "<=", "value": 0.01},
           "G2_beats_free_null": {"metric": "free_p", "op": "<=", "value": 0.01}},
 "outcomes": [{"when": {"G0_coverage": false}, "verdict": "INVALID__insufficient_paired_observation"},
              {"when": {"G0_coverage": true, "G1_beats_hourmatched_null": true}, "verdict": "COUPLED_BEYOND_CIRCADIAN__attribution_pending_E0"},
              {"when": {"G0_coverage": true, "G1_beats_hourmatched_null": false, "G2_beats_free_null": true}, "verdict": "CIRCADIAN_ONLY__coupling_explained_by_time_of_day"},
              {"when": {"G0_coverage": true, "G1_beats_hourmatched_null": false, "G2_beats_free_null": false}, "verdict": "NO_DETECTABLE_COUPLING__above_the_measured_floor"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

## The interpretation ceiling, frozen before data (invariant 6 made binding)

- The strongest available verdict is **`COUPLED_BEYOND_CIRCADIAN__attribution_pending_E0`** —
  by name. The RV statistic is symmetric: it cannot distinguish "the agent's state tracks the
  room" from "the room registers the agent." darkflobi's chassis is *in* the room and emits
  (CPU load, PSU switching, fans). Until the **E0 embodiment audit** (hash-committed
  randomized compute-burst schedule vs coil-band response) bounds that channel, no finding
  from this prereg may attribute a detected coupling to sensing. Both attribution branches
  are stated now: *agent-tracks-room* and *coil-hears-the-agent's-body* are different firsts,
  and confusing them would be the arc's cardinal sin.
- The words *feels / senses / experiences / perceives* do not appear in any finding from this
  prereg (carried from R1-v1).
- A null reads "no coupling above the floor measured in R0-v2's power surface at this n,"
  never "no coupling" (`r0v2_result.json:power_surface`).
- A positive is replicated on a second window (different days) before any public claim.

## Outcome reading

- **`COUPLED_BEYOND_CIRCADIAN`** → run E0 before *any* interpretive sentence is written; then
  R2 replication; only then attribution.
- **`CIRCADIAN_ONLY`** → an honest, interesting null: the streams share the clock and nothing
  else detectable. Successor S1 (randomized acoustic stimuli) breaks the clock by design.
- **`NO_DETECTABLE_COUPLING`** → the bound is published with the power floor, and S1 remains
  the escalation path.

## REQUIRED DISCLOSURE — added 2026-08-06, before any data, no gate changed

A confound was discovered in this lab's general coupling instrument hours after this prereg was
frozen, and it is **live for this experiment**: the agent stream is bursty while the room
recorder emits on a fixed interval, so paired bins will hold very different numbers of agent
records. Where bin record-count explains the magnitude of both binned streams, the two acquire
aligned structure with no shared cause, and **no permutation null can absorb it** — the effect
was found by pairing real agent telemetry against its own time-reversed copy and reading
RV 0.3704 at p 0.0033 (see CHANGELOG 7.31.1).

Therefore any finding from this prereg **must report the sampling-density diagnostic**
(`styxx.coupling.Coupling.sampling_density`: the correlation between bin record-count and binned
magnitude for both streams) alongside its verdict, and a positive verdict is **not licensed**
while that channel is open. The three admissible closures — uniform binning, equal-count
subsampling, or stratifying the confound on bin count — are to be chosen and stated before the
scored run. No gate, bar, or outcome branch in this document is altered by this disclosure; it
adds a reporting obligation and a licensing precondition that can only make a positive harder.

## Dependencies

Room side: Tier-0 hardware + `room_cortex --record` (round-trip verified 2026-08-05). Agent
side: live (`~/.styxx/chart.jsonl`). At measured cadence, G0's 200 paired bins ≈ 5–7 days of
joint recording. Smoke = synthetic tiny-n, INVALID-only. Result `r1v2_result.json`; scored by
`styxx.protocol`; certified + sealed before commit.
