# PREREG — the param-matched wider-static efficiency control — 2026-09-02

**FROZEN before confirmatory data.** Runner: `run_efficiency_control.py` (imports the arc's own
`run_entrain_rich.py`: same task, same arms, same red-team checks, same seeds). Follows
`RESULT_efficiency_control_from_receipts_2026_09_02.md`, which answered this question from committed
receipts *without* a frozen rule and said so. This preregistration is the frozen rule. Its verdict
outranks that document's.

## Disclosed prior

The author has read the recombination: on the identical task a static bank at D=16 (3580 params)
scores 0.678 against RICH at D=8 (5652 params) scoring 0.545. The honest prior is therefore strongly
toward CAPACITY_IN_DISGUISE. That is exactly why the bars below are frozen now: a run whose outcome
is expected is still a run whose outcome can surprise, and the only reading that counts is the one
the gates give.

## Question

> When the parameters RICH spends on its frequency detector are instead spent on more static modes,
> does the static bank match or beat RICH on the drifting-period task?

The static model's parameter count is `2D² + 143D + 780` (checked against three receipts). The
smallest static width with at least RICH-at-D=8's 5652 parameters is **D=26** (5850). The smallest
with at least RICH-at-D=4's 3184 is D=15 (3375). The matched arm is chosen to have *at least* the
adaptive model's parameters, never fewer, so a static win cannot be attributed to a lighter model.

## Setup (frozen)

Identical to `PREREG_entrain_rich_2026_07_23.md`: L=96, three segments, periods [3,12], 1500 steps,
seeds 0/1/2, batch 64, drift mean-accuracy on 1024 held-out sequences. Arms at the **primary width
D=8**: STATIC(8) and RICH(8) as anchors, ORACLE(8) as the positive control, and **STATIC(26)** as the
matched arm. Secondary, reported not gated: STATIC(4), RICH(4), STATIC(15). Device: CPU (this
sandbox); the anchors' agreement with the GPU receipts is a plumbing gate, not an assumption.

## Gates

```gates
{"gates": {"G_P_anchors": {"metric": "anchor_max_abs_dev", "op": "<=", "value": 0.03,
                           "power_basis": "STATIC(8) and RICH(8) re-run on CPU must land within 0.03 of the committed GPU receipt (0.4604, 0.5451); three seeds of this task move by about 0.01-0.02 between devices in the arc's experience, and a larger gap means the run is not the same experiment"},
           "G_C_oracle": {"metric": "oracle_minus_static_8", "op": ">=", "value": 0.10,
                          "power_basis": "the arc's standing positive-control bar (PREREG_entrainment, PREREG_entrain_rich): the diverse bank locked to the true period must beat static by 0.10, measured at 0.1703 on the receipt"},
           "G_E_static_wins": {"metric": "static_matched_minus_rich_8", "op": ">=", "value": 0.0,
                               "power_basis": "CAPACITY_IN_DISGUISE means the matched static bank is at least as good; zero is the natural bar for 'the parameters would have bought as much as modes', and the recombination puts the expected value near +0.13"},
           "G_E_rich_wins": {"metric": "rich_minus_static_matched_8", "op": ">=", "value": 0.05,
                             "power_basis": "the arc's own KILL/WEAK boundary (PREREG_entrain_rich): an advantage under 0.05 has never been read as real in this lane"}},
 "outcomes": [{"when": {"G_P_anchors": false}, "verdict": "INVALID__plumbing_anchors_drifted"},
              {"when": {"G_P_anchors": true, "G_C_oracle": false}, "verdict": "INVALID__positive_control_silent"},
              {"when": {"G_P_anchors": true, "G_C_oracle": true, "G_E_static_wins": true}, "verdict": "CAPACITY_IN_DISGUISE"},
              {"when": {"G_P_anchors": true, "G_C_oracle": true, "G_E_static_wins": false, "G_E_rich_wins": true}, "verdict": "EFFICIENCY_REAL"},
              {"when": {"G_P_anchors": true, "G_C_oracle": true, "G_E_static_wins": false, "G_E_rich_wins": false}, "verdict": "WEAK_EFFICIENCY"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

`anchor_max_abs_dev = max(|STATIC(8) − 0.4604|, |RICH(8) − 0.5451|)`. `oracle_minus_static_8 =
ORACLE(8) − STATIC(8)`. `static_matched_minus_rich_8 = STATIC(26) − RICH(8)`;
`rich_minus_static_matched_8` is its negative. All on drift mean-accuracy, seed-averaged.

## Outcome reading

`CAPACITY_IN_DISGUISE` confirms the recombination under a frozen rule and closes the adaptive-
frequency line's last positive: the frequency-resonance INDEX row gains no new claim, and the
synthesis's "capacity comes from more modes" covers adaptation. `EFFICIENCY_REAL` overturns the
recombination — the detector earns its parameters — and the recombination RESULT gets a back-pointer
the same day. `WEAK_EFFICIENCY` is reported as such and licenses nothing. Either INVALID verdict
ships as INVALID with the number that tripped it.

## Discipline

The prereg is committed before the runner is run for data; the smoke (`--smoke`: 400 steps, one seed)
is INVALID-only by the gates block. Result → `efficiency_control_result.json`, scored through
`styxx.protocol`, RESULT sworn to the receipt. No bar moves after data. No favourable width is
promoted: D=8 is primary because it was primary in the parent prereg.
