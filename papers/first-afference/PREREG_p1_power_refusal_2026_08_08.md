# PREREG — P1: can a machine catch an unreachable bar before the compute is spent?

Fathom Lab · 2026-08-08 · frozen before `styxx.power` exists.

## Why

Five preregistration defects in one week. Three of them are the same defect: **a bar fixed before
anyone computed whether it was achievable.** b37-G2 demanded an effect the apparatus could not
produce. b48-G2 judged the maximum of 45 draws against a bar written for one draw. C5-G1 demanded
80% of individual pairs reach significance at an effective sample size between 6.9 and 45.8, which
no instrument could deliver. Each was written up. Each recurred.

Protocol v2 added `power_basis` — a *declaration* that a bar is reachable. A declaration is not a
computation, and 4 of 35 gated preregs carry one. **P1 asks whether the computation can be
mechanised**: given a null distribution and a candidate bar, can a module return REACHABLE or
UNREACHABLE and be right?

The proposed module is `styxx.power`. It does not exist yet. Per the standing rule of 2026-08-06,
it will not be announced or released unless an adversary tries to break it and fails.

## The battery

Two families, scored separately and reported separately.

**Historical (n=3, IN-SAMPLE — this is a disclosure, not a result).** The three documented bar
failures above, reconstructed from their committed receipts. The module is being written by
someone who knows these three cases, so passing them is a *sanity check on the implementation*
and carries no evidential weight about generalisation. It is gated anyway, because a module that
cannot catch the cases it was designed for should not proceed to the ones it was not.

**Synthetic (held-out, analytic ground truth).** Bars whose reachability follows from the
construction rather than from our judgement: bars beneath a null median, bars above the maximum
attainable value of a bounded statistic, single-draw bars applied to order statistics of known
draw count, and matched REACHABLE controls at each construction. Ground truth is assigned by the
construction before the module sees the case.

A **constant-answer baseline** (always UNREACHABLE, and always REACHABLE) is scored on the same
synthetic battery. A detector that cannot beat a constant is not a detector — this is the lesson
of the twice-quarantined `styxx.apparatus`, where balanced accuracy 0.5625 was measured against
0.5000 for a constant string.

```gates
{"gates": {"G0_coverage": {"metric": "n_cases_scored", "op": ">=", "value": 24,
             "power_basis": "3 historical plus at least 21 synthetic; the synthetic count is fixed by the construction list before the run and is achievable by enumeration",
             "metric_means": "total cases the module returned a non-error verdict on"},
           "G1_historical_in_sample": {"metric": "historical_caught", "op": ">=", "value": 3,
             "power_basis": "n=3, IN-SAMPLE by construction and disclosed as such; a module written from these cases that cannot flag all three is broken rather than merely weak, so the bar is the maximum and its failure is informative while its success is not",
             "metric_means": "count of the three documented historical bar failures returned UNREACHABLE"},
           "G2_synthetic_balanced_accuracy": {"metric": "synthetic_balanced_accuracy", "op": ">=", "value": 0.85,
             "power_basis": "synthetic ground truth is analytic, so a correct implementation should approach 1.0 and 0.85 leaves room for boundary cases; the apparatus red team measured 0.6964 and 0.5625 for instruments that were quarantined, so 0.85 is deliberately above anything this lab has shipped and below the analytic ceiling",
             "metric_means": "mean of sensitivity and specificity on the held-out synthetic battery"},
           "G3_beats_constant": {"metric": "synthetic_balanced_accuracy_minus_best_constant", "op": ">=", "value": 0.30,
             "power_basis": "a constant answer scores exactly 0.5000 balanced accuracy on any battery by construction, so this bar is equivalent to requiring 0.80 and is stated as a margin because the apparatus failure was invisible until a constant was scored alongside it",
             "metric_means": "the module's balanced accuracy minus the better of the two constant-answer baselines"},
           "G4_refuses_degenerate": {"metric": "degenerate_refusal_rate", "op": ">=", "value": 1.0,
             "power_basis": "refusal on a null with too few draws, zero variance, or non-finite values is a code path the implementation controls completely; anything below 1.0 means a case reaches a verdict it has no basis for, which is the failure mode this whole program exists to prevent",
             "metric_means": "fraction of deliberately degenerate inputs on which the module refused rather than returning a verdict"}},
 "outcomes": [{"when": {"G0_coverage": false}, "verdict": "INVALID__battery_incomplete"},
              {"when": {"G0_coverage": true, "G4_refuses_degenerate": false}, "verdict": "DO_NOT_SHIP__answers_without_basis"},
              {"when": {"G0_coverage": true, "G4_refuses_degenerate": true, "G1_historical_in_sample": false}, "verdict": "DO_NOT_SHIP__fails_its_own_design_cases"},
              {"when": {"G0_coverage": true, "G4_refuses_degenerate": true, "G1_historical_in_sample": true, "G3_beats_constant": false}, "verdict": "DO_NOT_SHIP__no_better_than_a_constant"},
              {"when": {"G0_coverage": true, "G4_refuses_degenerate": true, "G1_historical_in_sample": true, "G3_beats_constant": true, "G2_synthetic_balanced_accuracy": false}, "verdict": "DO_NOT_SHIP__below_the_frozen_accuracy_bar"},
              {"when": {"G0_coverage": true, "G4_refuses_degenerate": true, "G1_historical_in_sample": true, "G3_beats_constant": true, "G2_synthetic_balanced_accuracy": true}, "verdict": "PROCEED_TO_RED_TEAM__not_yet_shippable"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

## What the winning branch does NOT license

`PROCEED_TO_RED_TEAM__not_yet_shippable` is the best available outcome and it is **not a release
decision**. The apparatus module passed its own author's battery twice and was quarantined twice
by adversaries. A synthetic battery scores an implementation against the failure modes its author
imagined; the red team's job is the ones they did not. **No version of `styxx.power` ships on the
strength of this document.**

## The losing branch, named in advance

If the module cannot beat a constant, or cannot refuse degenerate input, the verdict is
`DO_NOT_SHIP` and the module is quarantined in-tree with its audit in its own docstring, exactly
as `styxx.apparatus` was — twice. **A third quarantined instrument would itself be a finding**:
it would say that this lab's failure classes are diagnosable in retrospect and not detectable in
advance, which is a real and publishable limit on the whole preregistration programme.
