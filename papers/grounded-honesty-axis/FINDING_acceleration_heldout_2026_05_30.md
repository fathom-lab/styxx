# FINDING — ACCELERATION confirmed HELD-OUT: train-frozen route × trim retains detection on unseen items at 6.6× (−0.006 AUC). The 5.15× feasibility number is now a confirmed result

**Run 2026-05-30. Pre-registered in `PREREG_acceleration_heldout_2026_05_30.md` before scoring.**
Inputs SHA-256'd pre-scoring. R=5 stratified 50/50 splits per regime, operating points picked on
train and frozen, evaluated on disjoint test. Per-model calibrated, offline, white-box.

## Why

`FINDING_cascade_acceleration` (route) and `FINDING_resample_earlystop` (trim) reported the 5.15×
compounding with **in-sample** operating points — flagged as needing held-out confirmation. This is
it: per regime, on a stratified TRAIN half, pick `(tau1 = cascade escalation threshold, min_k =
early-stop sample count)` minimizing train compute s.t. train cascade-AUC ≥ train full-N-AUC − 0.02.
FREEZE. Measure on the held-out TEST half. Repeat over 5 splits, aggregate n-weighted.

## Result — HC1 SURVIVED

| aggregate (held-out) | value |
|---|---|
| test cascade-AUC | **0.9413** |
| test full-N-AUC | 0.9470 |
| **AUC delta** | **−0.0057** (bar: ≥ −0.03) ✓ |
| **test passes / item** | **1.51** (bar: ≤ 5.0) ✓ → **6.6×** |

**RESULT = SURVIVED.** The route × trim acceleration **generalizes to unseen items**: operating
points chosen without ever seeing the test fold retain detection (−0.006 AUC, negligible) at 1.51
forward passes/item — a **6.6× collapse**, even better than the in-sample 5.15× because train-picking
found `min_k = 2` suffices in every regime and it held out-of-sample.

## Per-regime — the through-line holds out-of-sample

| regime | cascade AUC | full AUC | passes | speedup |
|---|---|---|---|---|
| logic (Qwen), easy-arith (Llama-1B/3B) | 1.00 | 1.00 | 1.0–1.06 | **9.5–10×** |
| logic (Gemma), logic (Llama-3B) | **1.00** | 0.94 / 0.81 | 1.08–1.18 | 8.5–9.3× (cheap gate **beats** full) |
| code | 0.86 | 0.90 | 1.36 | 7.4× |
| arithmetic (Qwen) | 0.94 | 0.98 | 1.66 | 6.0× |
| arithmetic (Gemma), factual (Llama-1B), gpt-4o-mini | 0.72–0.96 | 0.81–1.00 | 2.0–2.6 | 3.9–5.0× |

**Acceleration = cheap-gate quality, confirmed out-of-sample.** Where the calibrated cheap gate is
strong (derivation), the cascade escalates ~0% and you get the near-full 10× at no loss — sometimes a
*gain*, where one forward pass beats N=10 resampling. Where it is weak (factual recall, Gemma's
soft-capped logits, the closed gpt-4o-mini single-token regime), the cascade escalates most items and
trades a small AUC drop (−0.04 to −0.08 per regime) for a still-real 3.9–5×. The aggregate clears the
bar because the strong regimes dominate; the honest per-regime story is that the speedup and a small
detection cost both grow as the cheap gate weakens.

## What this confirms

The honesty layer can run at **~1.5 forward passes/item instead of 10 — a confirmed 6.6× compute
collapse — at negligible aggregate detection loss**, by routing (cheap gate escalates only the
uncertain fraction) and trimming (early-stop the resamples on escalated items), with all operating
points fit per-model on held-in data. This is the acceleration capstone: the load-bearing detector is
a compute router, and the routing pays off out-of-sample.

## Scope / discipline

Held-out *within* each model/regime (train and test are disjoint items of the same regime) — this
does NOT claim cross-model transport of a fixed threshold (entropy scale is per-model). Counts forward
passes, not wall-clock (batching and the generation-pass reuse shift real numbers). Per-regime test
folds are small (~10–15/class), averaged over 5 splits to reduce noise; the weak regimes' per-fold
AUC is itself noisy. Accelerates detection; corrects nothing (the closed negative). The open frontier
(single-token closed-model confab, where the cheap gate fails) is exactly the low-speedup tail here.
