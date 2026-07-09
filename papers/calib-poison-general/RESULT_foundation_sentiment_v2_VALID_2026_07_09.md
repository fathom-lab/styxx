# RESULT — sentiment foundation v2: FOUNDATION_VALID. The generalization test is licensed on Qwen.

**Fathom Lab · papers/calib-poison-general · 2026-07-09. Verdict against the frozen Stage-1 gate of
`PREREG_calib_poison_sentiment_v2_2026_07_09.md` (committed before the run). The corrected K-averaged
controls confirm v1's block was single-draw noise; the sentiment construct clears all four guards on
the decisive model, and Stage 2 (attack + defense) is licensed under its frozen prereg.**

## Verdict: FOUNDATION_VALID — the v1 block was a control artifact, now resolved

On 440 length-matched 2★-vs-4★ Amazon reviews (index-split 220/110/110), the corrected controls land
exactly at chance, and the sentiment construct clears every guard on Qwen2.5-1.5B (the decisive model):

| guard | Qwen (decisive) | bar | verdict |
|---|---|---|---|
| read EVAL AUROC | **0.954** @L22 | ≥ 0.75 | PASS |
| behavioral sentiment accuracy | **0.845** | ≥ 0.80 | PASS |
| K-averaged random-direction floor (200 dirs) | 0.496 | chance band | PASS |
| K-averaged shuffled-behavioral null (500 perms) | 0.494 | chance band | PASS |

The read is graded, not lexical: it rises with depth (L12 0.840 → L22 0.954), and the K-averaged
random-direction floor is 0.496 — a real fitted direction, not high-dimensional geometry. The
behavioral judgment is real: 0.845 against true labels versus a 500-permutation null of 0.494. Both
corrected controls (0.496, 0.494) sit essentially at chance, confirming v1's single-permutation
control value was noise — the fix was to the control, and the construct passes on its merits. The
length confound is negligible.

Llama-3.2-1B corroborates the READ (0.912, graded) but its behavioral sentiment judgment is 0.482
(chance) — a genuine capability gap, not a control artifact — so it VOIDs guard 2 and stays
corroborating-only. The verdict rides Qwen, as the prereg specified.

## What this licenses

Stage 2 is now licensed on the sentiment construct with Qwen decisive: replicate the honesty
attack-and-defense (a LoRA scrub of the frozen sentiment direction with a behavioral-sentiment replay
term; naive ATTACK-calibrated vs private CALIB-calibrated re-locks). The Stage-2 verdicts
(GENERALIZES / NO_GENERALIZATION / PARTIAL) are frozen in
`PREREG_calib_poison_sentiment_2026_07_09.md` and unchanged. The generalization question — is
calibration poisoning a property of substrate linear probes in general, or honesty-specific? — is now
genuinely testable, on a construct fully independent of honesty (sentiment content, human-labeled).

## Bounds

440 reviews, index-split 220/110/110, `Qwen2.5-1.5B` decisive (`Llama-3.2-1B` corroborating-may-VOID,
did), DoM read on last-review-token residuals, behavioral judgment via a yes/no sentiment prompt,
K=200 random directions / K=500 label permutations. Forward passes only; no training in the
foundation.

## Reproducibility

`foundation_sentiment_v2.py` (deterministic; reuses v1's loader/prompt/token helpers and
`run_says_yes_knows_no.py` math) → `foundation_sentiment_v2_result.json`. Prereg frozen before run.

---
*FOUNDATION_VALID — the corrected controls both read chance, the construct clears on its merits, and
the generalization test can finally run on a second, honesty-independent construct.*
