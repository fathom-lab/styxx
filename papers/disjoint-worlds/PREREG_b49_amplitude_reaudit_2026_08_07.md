# PREREG — B49: do our own published island findings survive the amplitude control we built afterwards?

Fathom Lab · 2026-08-07 · frozen before the re-analysis is run. On 2026-08-06 an adversarial
audit showed that shared per-item **amplitude** — with no shared geometry at all — can produce
frame affinity far above a random-frame null. We added `normalize_items` and made it the default
in `styxx.islands`. The h1a human finding was re-run against that control and survived.

**The island arc's own findings were never re-run.** b45 (frame geometry: clique 0.848 vs
random-null p95 0.0566, island below the clique in 5/5 seeds) and b47 (`SINGLE_LEGIBLE_CLIQUE`
on ten models) were both computed with the un-normalised statistic, and both are published,
sealed, and cited in the connection-of-minds synthesis and the staged arXiv paper. A control that
was decisive enough to apply to human data is decisive enough to apply to our own.

## Design (frozen)

Re-run b45's affinity measurement and b47's ten-model survey with `normalize_amplitude=True`,
everything else identical (same banks, same k, same seeds, same nulls). Compare against the
published values. No re-tuning; a single run of each.

```gates
{"gates": {"G1_b45_clique_still_above_null": {"metric": "b45_norm_clique_minus_null_p95", "op": ">", "value": 0.0,
             "power_basis": "the published margin is 0.7914 and the random-frame null is analytic at k/n; any collapse to <=0 is a qualitative reversal, not a power question"},
           "G2_b45_island_still_lowest": {"metric": "b45_norm_seeds_qwen_below_clique", "op": ">=", "value": 5,
             "power_basis": "all-seeds sign consistency, probability 1/32 under exchange; identical to the bar b45 itself passed"},
           "G3_b47_verdict_unchanged": {"metric": "b47_norm_verdict_matches_published", "op": ">=", "value": 1,
             "power_basis": "exact string match against the sealed verdict; binary by construction"}},
 "outcomes": [{"when": {"G1_b45_clique_still_above_null": false}, "verdict": "PUBLISHED_FINDING_OVERTURNED__b45_was_amplitude"},
              {"when": {"G1_b45_clique_still_above_null": true, "G2_b45_island_still_lowest": false}, "verdict": "PARTIAL__shared_frame_holds_island_ordering_does_not"},
              {"when": {"G1_b45_clique_still_above_null": true, "G2_b45_island_still_lowest": true, "G3_b47_verdict_unchanged": false}, "verdict": "PARTIAL__b45_holds_b47_verdict_moves"},
              {"when": {"G1_b45_clique_still_above_null": true, "G2_b45_island_still_lowest": true, "G3_b47_verdict_unchanged": true}, "verdict": "SURVIVES_THE_CONTROL__island_arc_is_not_amplitude"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

## Stated before the run

`PUBLISHED_FINDING_OVERTURNED` is a live branch. If it fires, b45 and everything downstream —
the synthesis §3 paragraph, the arXiv paper's abstract, the public claim that a cross-family
concept-frame geometry exists — is retracted in place, at the same volume it was published.
Every gate here carries a `power_basis`, the first prereg in this program to do so.
