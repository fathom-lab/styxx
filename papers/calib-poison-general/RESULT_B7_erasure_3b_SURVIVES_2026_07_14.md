# RESULT — B7: the erasure bound extends to 3B. `SURVIVES__vs_subspace_erasure_3B`

**Fathom Lab · papers/calib-poison-general · 2026-07-14. Prereg frozen and committed BEFORE the
run: `PREREG_B7_erasure_3b_2026_07_13.md`. Receipt: `b7_erasure_3b_result.json`. Scale
comparison operands: `b2_subspace_erasure_result.json` (cycle 36, 1.5B). Attack and audit
byte-identical to B2 by import; only the family constants moved (frozen proportional layer map).**

## Verdict

`SURVIVES__vs_subspace_erasure_3B` — n_admissible = 4, n_survives_cells = 4, n_erased_cells = 0.
Every cell bit, every cell held knowledge, and the private-13 read stayed above the survival
threshold of 0.70 in all four cells. The read/knowledge coupling first measured at 1.5B
(cycles 36 and 37) is not a small-model artifact: the same static gold-subspace eraser with the
same knowledge-replay constraint fails the same way at twice the scale.

## The run

Qwen2.5-3B-Instruct (num_hidden_layers 36), scan layers [15, 18, 21, 23, 26, 28], deploy layer 23
(the frozen proportional image of the 1.5B constants), E1 three-way split (ATTACK 61 / CALIB 53 /
EVAL 66, disjoint stems), seeds {0, 1} x alpha {1.0, 4.0}, lambda 1.0, LoRA r=16, 300 steps.

Clean guard PASSED: clean private-13 EVAL AUROC 0.9853 (rand floor 0.5124), clean EVAL knowledge
0.9242 against the frozen floor (threshold of 0.80), clean frozen-deploy read 0.9069, clean
naive-DoM-6 0.9641. Qwen2.5-3B is a valid base for the dissociation — unlike Llama-3.2-1B, which
VOIDed this same guard in cycle 35.

| cell | private13 | matched13 | naive6 | knowledge | frozen-deploy | bite | parity_gap |
|---|---|---|---|---|---|---|---|
| s0 alpha=1.0 | 0.8065 | 0.7825 | 0.6101 | 0.9091 | 0.6295 | yes | 0.024 |
| s0 alpha=4.0 | 0.8378 | 0.7429 | 0.6166 | 0.9091 | 0.6442 | yes | 0.0949 |
| s1 alpha=1.0 | 0.7484 | 0.7613 | 0.5585 | 0.9394 | 0.5788 | yes | -0.0129 |
| s1 alpha=4.0 | 0.7576 | 0.7512 | 0.5272 | 0.9091 | 0.5806 | yes | 0.0065 |

## The scale split (reported, no bar — operand pairs from the two receipts)

Per matching (seed, alpha) cell, 3B private-13 vs 1.5B private-13: 0.8065 vs 0.7806 (s0 a1),
0.8378 vs 0.7871 (s0 a4), 0.7484 vs 0.7161 (s1 a1), 0.7576 vs 0.7226 (s1 a4) — the 3B read sits
above its 1.5B counterpart in each of the four cells. Knowledge: 0.9091 vs 0.803, 0.9091 vs
0.7576, 0.9394 vs 0.803, 0.9091 vs 0.803 — higher at 3B in each cell. The naive-DoM-6 collapse is
SHALLOWER at 3B (0.6101/0.6166/0.5585/0.5272 vs 0.4848/0.4654/0.4645/0.4636 at 1.5B): scale did
not give the eraser room; it gave the signal room.

Attacker-objective note, same direction: the 3B l_erase tails (0.0285, 0.0244, 0.0238, 0.0304 at
step 299) sit above the 1.5B static run's converged tails (0.0085, 0.0112, 0.0073, 0.008) while
l_rep is satisfied (0.0001 to 0.0005) — at the same LoRA r=16 / 300-step budget the eraser zeroes
the 3B gold subspace LESS completely. The bigger model is a harder erasure target on the
attacker's own loss, and what erasure it does achieve still fails to remove the read.

Parity continuity: the seed structure of the parity arc persists at 3B (s0 parity_gap 0.024 and
0.0949 positive; s1 cells -0.0129 and 0.0065, near zero) — the capacity-dominant attribution
extends to scale.

## Crash-safety provenance (disclosed)

The scored result is a SINGLE-process run (all four cells, one launch, per-cell wall-clock 71.5 to
80.4 minutes, peak VRAM 10862.6 to 10863.9 MiB with WDDM spill). A prior overnight launch died
after two cells with no per-cell persistence; this run's harness added crash-safe per-cell
checkpointing (`b7_checkpoint.py`, runner-only, science untouched) and was relaunched fresh. The
relaunch reproduced the dead run's two seed-0 log lines (private13 0.8065 and 0.8378) to four
decimals — same seeds, same hardware; the disclosed bf16 non-determinism did not manifest at this
precision on these cells.

## Scope (pre-committed)

Qwen2.5-3B-Instruct only; honesty construct; LoRA r=16, 300 steps, lambda 1.0; STATIC rank-2
gold subspace per scan layer (adaptive-at-3B untested); EVAL n=66 (per-cell AUROC SE = 0.06,
unchanged from 1.5B for comparability); one run per cell; the layer map is a frozen proportional
rule, not a per-model optimum; behavioral knowledge invariant on disjoint stems (B4 caveat,
arc-wide). The verdict is `*_3B` and says nothing beyond this attacker class at this budget.

## What this buys

The reviewers' first objection (1.5B-only) is answered with receipts: two scales, same coupling,
wider margins at the larger scale. Together with the coupling dose-response run (same day,
separate RESULT), the pre-paper experiment set named by the backlog is complete.

## Reproduce

```
# prereg (frozen before the run): papers/calib-poison-general/PREREG_B7_erasure_3b_2026_07_13.md
python papers/calib-poison-general/b7_erasure_3b.py
python -m styxx.certify papers/calib-poison-general/RESULT_B7_erasure_3b_SURVIVES_2026_07_14.md \
  papers/calib-poison-general/b7_erasure_3b_result.json \
  papers/calib-poison-general/b2_subspace_erasure_result.json \
  --out papers/calib-poison-general/RESULT_B7_erasure_3b_SURVIVES_2026_07_14.certificate.json
```

*The scale objection was the cheapest one to make and the most expensive to answer — fifteen
GPU-hours, one dead launch, one crash-safe harness. The answer: at 3B the signal does not just
survive the eraser; it survives it more easily.*
