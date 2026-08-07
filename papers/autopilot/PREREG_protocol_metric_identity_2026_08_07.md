# PREREG — protocol v3: a gate must name a metric that exists, and the machinery must check it

Fathom Lab · 2026-08-07 · frozen before implementation.

## Why

B49 established the limit of protocol v2 empirically: `power_basis` constrains the **bar** and
says nothing about the **metric**. B49's G3 pointed at a field that could never equal what the
gate compared against — it could not have passed under any data — and it carried a power_basis
reading "binary by construction," which was true of the statistic's shape and silent about
whether the right field had been named. That is the fifth mis-specification in this program this
week; the previous four were bars, this one was a metric path, and no existing refusal covers it.

`_resolve()` already raises `GateSpecError` when a metric path is missing from a result — but
only at scoring time, when the run is already spent, and only if the path is absent. A path that
resolves to the *wrong existing field* is invisible.

## The change (frozen)

1. `Experiment.metric_paths` — the set of dotted paths every gate references.
2. `Experiment.check_metrics(result)` — resolves every gate's path against a candidate result
   **without scoring**, returning `{path: present/missing}`. Callable before a run is launched,
   so a mis-named metric costs seconds rather than a whole exam.
3. `Verdict.metric_paths` records what was resolved, so a reader can see which field each gate
   actually read.
4. A gate MAY carry `"metric_means"`: a plain-language statement of what the path is expected to
   contain. Recorded, never verified — because it cannot be, and the honest move is to make the
   author's intent legible beside the path rather than to imply a check that does not exist.

Verdict strings remain untouched; existing preregs keep scoring byte-identically.

```gates
{"gates": {"G1_no_verdict_strings_change": {"metric": "verdict_string_diffs", "op": "<=", "value": 0,
             "power_basis": "exact string comparison over every committed result re-scored against its prereg; zero is the only acceptable value because any diff breaks a committed seal",
             "metric_means": "count of results whose re-scored verdict differs from the verdict stored in the result file"},
           "G2_check_metrics_finds_a_missing_path": {"metric": "missing_paths_detected", "op": ">=", "value": 1,
             "power_basis": "one constructed result with a deliberately absent metric path; a single detection demonstrates the mechanism and more adds no information",
             "metric_means": "number of gate paths check_metrics() reported absent from a result that was built to omit exactly one"},
           "G3_check_metrics_passes_a_real_result": {"metric": "real_result_paths_all_present", "op": ">=", "value": 1,
             "power_basis": "b49_result.json is a committed real result whose gates all resolved at scoring time, so all-present is achievable by construction and falsifiable by regression",
             "metric_means": "1 if check_metrics() reports every path present for a genuine committed result, else 0"}},
 "outcomes": [{"when": {"G1_no_verdict_strings_change": false}, "verdict": "INVALID__changes_committed_verdicts"},
              {"when": {"G1_no_verdict_strings_change": true, "G2_check_metrics_finds_a_missing_path": false}, "verdict": "INVALID__check_does_not_detect"},
              {"when": {"G1_no_verdict_strings_change": true, "G2_check_metrics_finds_a_missing_path": true, "G3_check_metrics_passes_a_real_result": false}, "verdict": "INVALID__check_false_alarms_on_a_real_result"},
              {"when": {"G1_no_verdict_strings_change": true, "G2_check_metrics_finds_a_missing_path": true, "G3_check_metrics_passes_a_real_result": true}, "verdict": "METRIC_IDENTITY_LANDED__paths_checkable_before_a_run_is_spent"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

## Stated before the run

This catches **absent** paths early. It does **not** catch a path that resolves to a real but
wrong field — B49's actual error. Nothing in a gates block can, because the machinery cannot know
what the author meant. `metric_means` makes the intent legible to a reader; it is not a check and
this document will not pretend otherwise. Every gate above declares both `power_basis` and
`metric_means`. Red-teamed before release.
