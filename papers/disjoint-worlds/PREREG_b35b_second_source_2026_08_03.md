# PREREG — B35-b: the second source family — is the label-free read a Llama-source artifact?

Fathom Lab · 2026-08-03 · frozen before extraction. Every disjoint-worlds result reads FROM
Llama-3.2-3B. If the label-free cross-family read only works from one source family, it is a
property of that family, not of minds. Source swap: **Qwen2.5-3B-Instruct** (same scale class,
different family, already used as a target substrate in the agent-conscience arc).

The swap also flips the roles for free: **qwen_1p5b becomes the SAME-family control**
(mirroring llama3B→llama1B) and **llama_1b becomes a CROSS-family target** — so this run
additionally answers whether the weak Llama↔Qwen channel (b35-a: qwen discovery 0.036–0.094
from the Llama side, all five seeds) is **symmetric** (a pair property) or **directional**
(a source property). Reported, not gated.

## Design (frozen)

Extract Qwen2.5-3B at layer round(0.393 × n_layers) — the same committed frac rule every
target has used. The b34-v3 pipeline verbatim from the new source (seed 343 split, shuffle
rail, linear-discover, single MLP, 70-way read, shuffled null). Targets: qwen_1p5b
(same-family G0), gemma_2b (decisive G1), llama_1b (the symmetry probe, reported).

```gates
{"gates": {"G0_same_family": {"metric": "targets.qwen_1p5b.seed_acc", "op": ">=", "value": 0.30},
           "G2_null": {"metric": "max_null_top1", "op": "<=", "value": 0.0286},
           "G1_cross": {"metric": "targets.gemma_2b.read", "op": ">=", "value": 0.143}},
 "outcomes": [{"when": {"G0_same_family": false}, "verdict": "INVALID__second_source_discovery_broken"},
              {"when": {"G0_same_family": true, "G2_null": false}, "verdict": "INVALID__null_artifact"},
              {"when": {"G0_same_family": true, "G2_null": true, "G1_cross": true}, "verdict": "SECOND_SOURCE_READS__not_a_llama_artifact"},
              {"when": {"G0_same_family": true, "G2_null": true, "G1_cross": false}, "verdict": "SECOND_SOURCE_FAILS__source_matters"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

Single-seed scope disclosed up front: this is one seed (343) at the new source — a positive
licenses "not a Llama-source artifact," and its own seed stability inherits the b35-a design
as a later rung if pursued. The Qwen→gemma cell has NO prior base rate; the 0.143 bar is
imported from the arc's standing 10×-chance convention, not sized from measurement — if G0
passes and G1 misses narrowly, that is a CLOSED negative under the frozen bar, per program
rules.

## Discipline

One model load (Qwen2.5-3B — loaded clean repeatedly in the scale3b arc), DETACHED execution
per the shard-kill rail, extraction banked to `_b35b_ptsA_qwen3b.npz`. Smoke = 15 concepts,
INVALID-only. Result `b35b_result.json`, scored by `styxx.protocol`, certified before commit.
