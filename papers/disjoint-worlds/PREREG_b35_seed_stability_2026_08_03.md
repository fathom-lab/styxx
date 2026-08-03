# PREREG — B35-a: seed stability of the label-free cross-family read

Fathom Lab · 2026-08-03 · frozen before the scored run. b34-v3 licensed the label-free
cross-family read on ONE fresh split (seed 343). A single-seed headline is a candidate, not a
property. This run asks the cheapest, sharpest generality question first: **does the result
hold across five independent fresh splits?** (Second-source and open-vocabulary generality are
B35-b/c, separate preregs — this one is CPU-from-cache, zero model loads.)

## Lesson applied from the b34-v3 erratum

The v3 prereg *asserted* split disjointness and was wrong. This prereg asserts NO disjointness
between seeds — five independent 70-of-462 draws (seeds 1001–1005) overlap each other by
~11 concepts pairwise in expectation, and that is fine and stated: each seed is its own
complete experiment (its maps never see its own held-outs); stability across seeds is the
claim, not disjointness between them.

## Design (frozen; method = b34-v3 verbatim per seed)

Per seed s ∈ {1001, 1002, 1003, 1004, 1005}: permute the 462 concepts with seed s, last 70
held-out, rest anchors; shuffle target anchor rows (order-leak impossible); discover the
correspondence with the committed linear machinery (`TransferMap.fit`, k from the locked G0
record); fit ONE b31v2 MLP on the discovered pseudo-pairs (seed s); read the 70 held-outs
(committed metric, chance 1/70); fit one pairing-shuffled null MLP per target per seed.
Targets: llama_1b, gemma_2b, qwen_1p5b. 15 cells + 15 nulls total.

## Gates (frozen; scored by styxx.protocol)

The null gate is on the MEAN, deliberately: 15 independent 70-way nulls will, with high
probability, contain one 2-or-3-hit draw by pure chance — demanding every null sit at ≤2×
chance would fail honest runs by construction (the b34-v3 knife-edge lesson). The mean of 15
chance-level draws concentrates near 1× chance; a mean at 2× chance signals a real artifact.

```gates
{"gates": {"G0_discovery": {"metric": "median_llama_seed_acc", "op": ">=", "value": 0.30},
           "G2_null_mean": {"metric": "mean_null_top1", "op": "<=", "value": 0.0286},
           "G1_stability": {"metric": "median_gemma_read", "op": ">=", "value": 0.143}},
 "outcomes": [{"when": {"G0_discovery": false}, "verdict": "INVALID__discovery_unstable"},
              {"when": {"G0_discovery": true, "G2_null_mean": false}, "verdict": "INVALID__null_artifact"},
              {"when": {"G0_discovery": true, "G2_null_mean": true, "G1_stability": true}, "verdict": "LABELFREE_READ_SEED_STABLE"},
              {"when": {"G0_discovery": true, "G2_null_mean": true, "G1_stability": false}, "verdict": "LABELFREE_READ_SEED_FRAGILE__single_split_artifact"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

Reported, not gated: per-seed full table (seed_acc, read, null × 3 targets), min/max gemma
read (the honest spread), qwen's per-seed discovery accuracy (is weak discovery a qwen
property or a draw property?).

## Outcome reading

`LABELFREE_READ_SEED_STABLE` upgrades the b34-v3 claim from one-split to
median-of-five-splits; `SEED_FRAGILE` demotes it to a single-split artifact at full volume —
the b34-v3 FINDING would then carry a second erratum, and the family goes back to the bench.
Either way the synthesis §2 sentence gets its stability qualifier updated.

## Discipline

CPU-from-cache; deterministic; smoke (2 seeds × 40/10) INVALID-only; result
`b35a_result.json`; scored through `styxx.protocol`; OATH-certified before commit.
