# PREREG — B47: eight minds, one battery — does island structure recur?

Fathom Lab · 2026-08-06 · frozen before the scored run, and **before the cohort has been
surveyed even once**. The island arc's largest open question is recurrence: qwen was one island
in one cohort of four, and four members cannot support an inferential claim about whether
islands are a general feature of independently trained minds or a fact about one model. This
run answers it on an **independent cohort** using the instrument shipped this morning.

## Why this cohort is a genuine test and not a replication

`papers/mind-instrument/normeq_reps.npz` holds norm-equalized representations for **eight**
models — Qwen2.5-1.5B, Qwen2.5-3B, Llama-3.2-1B, Llama-3.2-3B, Phi-3.5-mini, gemma-2-2b, gpt2,
gpt2-large — over the shared 96-concept battery. Compared to the b37 cohort this is a
**different battery** (96 concepts, not 462), a **different extraction** (norm-equalized Atlas
reps, committed 2026-06-10 for an unrelated purpose), **twice the members**, and **four model
families including two GPT-2 scales absent from the original**. Nothing about it was collected
with the island question in mind, which is exactly what makes it a fair test.

Eight members is also the minimum at which `styxx.islands` will return an inferential verdict at
all — below it the instrument reports `UNDERPOWERED__n_below_8` by construction. This cohort sits
exactly at that floor, and the finding must say so.

## Design (frozen)

`styxx.islands.survey` (shipped in styxx 7.30.0, validated against the b37 cohort's published
topology), defaults unchanged: k = 20 concept-Gram eigenframe, 1000-draw Haar-random null,
1000-permutation gap screen, island rule = mean affinity below median − 1·1.4826·MAD. All
eight members, the full 96-concept battery, no subsetting, one run, no seed sweep — the
instrument is deterministic given its seed and the prereg fixes seed 343.

## Gates

```gates
{"gates": {"G0_cohort": {"metric": "n_members", "op": ">=", "value": 8},
           "G1_shared_frame": {"metric": "median_minus_null_p95", "op": ">", "value": 0.0},
           "G2_islands_present": {"metric": "bimodality_p", "op": "<=", "value": 0.05}},
 "outcomes": [{"when": {"G0_cohort": false}, "verdict": "INVALID__cohort_below_instrument_floor"},
              {"when": {"G0_cohort": true, "G1_shared_frame": false}, "verdict": "NO_SHARED_FRAME__cohort_does_not_co_align_and_island_is_the_wrong_word"},
              {"when": {"G0_cohort": true, "G1_shared_frame": true, "G2_islands_present": true}, "verdict": "ISLANDS_RECUR__structure_is_not_specific_to_the_first_cohort"},
              {"when": {"G0_cohort": true, "G1_shared_frame": true, "G2_islands_present": false}, "verdict": "SINGLE_LEGIBLE_CLIQUE__no_islands_in_this_cohort"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

`G2` uses the instrument's own gap screen, reported under its own name — **not** Hartigan's dip
(see the erratum in `PREDICTION_h1_human_islands_2026_08_06.md`; the two are different tests and
this document does not conflate them).

## Stated before the run

- **A `SINGLE_LEGIBLE_CLIQUE` verdict is a real and useful negative**, not a failure. It would
  say the b37 island is a property of that cohort, that battery, or qwen specifically — and it
  would immediately weaken the human-islands prediction registered this morning, which we would
  record in that document rather than leave standing.
- **Which member is flagged is reported regardless of verdict**, including if it is a model
  nobody expects. Predicting the identity now would be storytelling; the raw per-member
  affinities ship in the receipt either way.
- **n = 8 is the floor, not comfort.** A gap screen on eight points has little power; a
  `SINGLE_LEGIBLE_CLIQUE` verdict here is weak evidence of absence and the finding must say so
  in those words.
- The two Qwen models and the two Llama models are same-family pairs; elevated within-family
  affinity is expected and is **not** evidence about islands either way.

## Discipline

CPU, seconds, zero model loads. Result `b47_result.json`; scored by `styxx.protocol` from this
frozen block; certified + sealed before commit. The full pairwise affinity matrix ships in the
receipt regardless of verdict.
