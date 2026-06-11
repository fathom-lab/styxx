# PRE-REGISTRATION — OOD portable conscience, v2: the correct null (frozen pre-run)

**2026-06-11 · Fathom Lab / styxx. Frozen before any v2 score is seen. Runner:
`run_portable_conscience_ood_v2.py` (SEED=0). Receipt: `portable_conscience_ood_v2_result.json`.
This resolves the v_ood_1 VOID-FIT by replacing a mis-specified null, NOT by relaxing a bar.**

## Why v_ood_1 was VOID (honest diagnosis, receipt `portable_conscience_ood_result.json`)

v_ood_1 (leave-families-out) returned transferred OOD AUROC 0.923 (Llama-3.2-3B) / 0.829 (Qwen-3B)
with gemma OOD self-ceiling 0.929 — looked like the conscience survives OOD. But the frozen
floor-concentration guard fired: Llama's random-direction `floor_p95` reached 0.865 (>= 0.78 = VOID).
Diagnosis from the receipt: (a) `floor_median` sat at chance (~0.48-0.51) for every target — the floor
is centred correctly, only its high tail is fat; (b) per-family AUROC showed `geography` = 1.000 almost
everywhere — one OOD family is trivially linearly separable, fattening the floor tail and inflating the
aggregate. The random-direction floor conflates "does ANY direction separate transported truth" (yes,
because truth transports) with the real question "does the SPECIFIC honesty direction transfer OOD."

## The fix — a label-permutation null (the correct test for a DiM direction)

The right null for a difference-of-means direction is the distribution of transfer AUROC from honesty
directions built on RANDOM label bipartitions of the same source data, pushed through the SAME real
map. If truth is so dominant that random-label directions also separate OOD truth, the null rises to
meet the true direction (honest null result). If the true-label honesty direction is special, it beats
the null.

- For K = 1000 permutations: shuffle the gemma TRAIN-FIT labels, refit the source direction by
  difference-of-means on the shuffled labels, score the SAME mapped OOD activations -> `perm_auroc`.
- `perm_p95`, `perm_median`, and `p_value = (1 + #{perm >= transferred}) / (1 + K)`.

## Frozen gates (v2)

- **P1 — OOD-PORTABLE** iff, on the held-out OOD families, **transferred AUROC >= 0.65 AND transferred
  AUROC > `perm_p95` (equivalently p_value < 0.05) for BOTH primary 3B targets** (Llama-3.2-3B,
  Qwen2.5-3B).
- **VOID-FIT** iff gemma OOD self-ceiling < 0.70 ONLY. (The elevated random-direction floor is EXPECTED
  given truth-transport and is no longer the discriminator; the permutation null is. The
  random-direction floor + random-map are retained as DESCRIPTIVE context, not gates.)
- **Verdict ladder:** VOID-FIT > OOD-PORTABLE (both pass) > OOD-PARTIAL (one) > OOD-COLLAPSE (neither).

## Robustness (descriptive, not gated)

- **drop-geography:** recompute transferred AUROC and the permutation p_value on the OOD set with the
  trivially-separable `geography` family removed (3 families, 48 items) — the result must not rest on
  the one easy family.
- Per-OOD-family transferred AUROC; matched in-distribution AUROC + retention; random-direction floor
  + random-map (carried from v_ood_1 for continuity); two smaller secondary targets.

## What each outcome means (pre-committed)

- **OOD-PORTABLE** — the true-label honesty direction transfers across models to unseen fact-families
  better than chance label-bipartitions: a real, specific cross-model OOD honesty readout. Still
  linear, one task; those bounds stand.
- **OOD-PARTIAL / COLLAPSE** — under the correct null, honesty-direction transfer OOD is not
  separable from the map's general truth-transport: the portable conscience is (at least linearly)
  an in-distribution property. Honest frontier, precisely bounded.
- This is methodological self-correction (mis-specified null -> correct null), the same v0->v1->v2
  discipline; the answer stands whichever way it lands.
