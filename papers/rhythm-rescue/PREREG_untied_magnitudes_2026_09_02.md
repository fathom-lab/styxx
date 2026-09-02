# PREREG — the confound in the knob: rotation, or timescale diversity? — 2026-09-02

**FROZEN before confirmatory data.** Runner: `run_untied_control.py` (imports
`run_rhythm_rescue.py` verbatim for the task, the FREE and CLAMPED arms, and evaluation). Follows
`RESULT_rhythm_rescue_2026_06_03.md` (oscillation roughly doubles ordered-memory capacity, 6.0 vs
2.67 items) and the flagship `RESULT_pmnist_ablation_2026_07_23.md` (+0.312 on permuted MNIST),
both of which rest on the same single knob.

## The confound, stated

The phase clamp sets θ≡0. In a complex-diagonal mode, that removes rotation — and it also ties the
mode's two real channels to one magnitude, because both decay by the same |λ|. A FREE bank at D
complex modes has D magnitudes and D rotations; a CLAMPED bank has D magnitudes across 2D real
channels and no rotation. So the clamp changes two things at once, and "CLAMPED loses capacity"
has two readings: rotation was load-bearing, or timescale diversity was and the clamp halved it.
No document in the arc separates them.

## The arm that separates them

**REAL2**: a real-eigenvalue bank with 2D modes and 2D *independent* magnitudes, no rotation.
It has exactly FREE's real state size (2D) and exactly FREE's parameter count (`nu` 2D against
`nu` D plus `theta` D; `B` 2D×d_in against `B_re` plus `B_im`), and strictly more timescale
diversity than either existing arm. The red-team check asserts the parameter equality before any
training. Everything else — task, vocabulary, D=256, 4000 steps, seeds 0/1/2, the K grid, the 0.80
capacity threshold, the read head — is `run_rhythm_rescue.py`'s.

## Question

> Does a real bank with FREE's parameters and twice its independent timescales recover FREE's
> ordered-memory capacity, or does rotation buy capacity that timescale diversity cannot?

## Gates

```gates
{"gates": {"G_P_anchors": {"metric": "anchor_max_abs_dev", "op": "<=", "value": 1.5,
                           "power_basis": "FREE and CLAMPED re-run on CPU must land within 1.5 items of the committed receipt's seed-mean capacities (6.0, 2.67); the K grid steps by 2 above K=4, so one grid step of drift is plumbing, two is a different experiment"},
           "G_C_gap": {"metric": "gap_free_minus_clamped", "op": ">=", "value": 2.0,
                       "power_basis": "the receipt's gap is 3.33 items; the arc's own ADVANTAGE reading started at a 3-item gap and a gap under 2 would mean the effect the control exists to explain did not reproduce"},
           "G_R_recovers": {"metric": "free_minus_real2", "op": "<=", "value": 1.0,
                            "power_basis": "REAL2 within one item of FREE is the rescue rule the parent prereg used (kcap within 2) tightened by one item, because REAL2 carries MORE diversity than FREE and a tie is the claim"},
           "G_R_fails": {"metric": "real2_minus_clamped", "op": "<=", "value": 1.0,
                         "power_basis": "REAL2 no more than one item above CLAMPED means untying the magnitudes bought nothing the grid can see; one item is the grid's resolution at the low end"}},
 "outcomes": [{"when": {"G_P_anchors": false}, "verdict": "INVALID__plumbing_anchors_drifted"},
              {"when": {"G_P_anchors": true, "G_C_gap": false}, "verdict": "INVALID__gap_did_not_reproduce"},
              {"when": {"G_P_anchors": true, "G_C_gap": true, "G_R_recovers": true}, "verdict": "TIMESCALE_DIVERSITY__rotation_not_load_bearing_here"},
              {"when": {"G_P_anchors": true, "G_C_gap": true, "G_R_recovers": false, "G_R_fails": true}, "verdict": "ROTATION_LOAD_BEARING__beyond_diversity"},
              {"when": {"G_P_anchors": true, "G_C_gap": true, "G_R_recovers": false, "G_R_fails": false}, "verdict": "PARTIAL__diversity_recovers_some"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

All metrics are seed-mean capacities in items. `recovery_fraction = (REAL2 − CLAMPED) / (FREE −
CLAMPED)` is reported beside them, not gated.

## Outcome reading

`TIMESCALE_DIVERSITY` means the arc's causal claim about *rotation* needs a scope: the clamp
measured timescale diversity, of which rotation is one source, and the flagship permuted-MNIST
number becomes the next thing to re-run with this arm. `ROTATION_LOAD_BEARING` sharpens the arc's
headline: rotation buys ordered-memory capacity that no amount of independent real decay recovers
at matched size, and the theta-gamma reading strengthens. `PARTIAL` is reported as such, with the
fraction. Either INVALID ships as INVALID with the number that tripped it.

## Disclosed prior

Uncertain, and honestly so. Phase coding is the mechanism the literature credits for ordered
memory (Lisman & Jensen), which favours ROTATION_LOAD_BEARING; but a mode with two independent
decay rates can also stagger items in time, and nothing in the arc has tried it. This is a real
bet, which is why it is worth freezing.

## Discipline

Committed before the run. Smoke (`--smoke`: 200 steps, one seed, a short K grid) is INVALID-only.
Result → `untied_control_result.json`, scored through `styxx.protocol`, RESULT sworn to the
receipt. No bar moves after data. The permuted-MNIST re-run, if this verdict calls for it, gets its
own preregistration.
