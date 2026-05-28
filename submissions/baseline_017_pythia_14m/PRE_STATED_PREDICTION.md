# Pre-stated prediction — Baseline-017 (committed BEFORE gauntlet run)

**File created:** 2026-05-27, before any `styxx gauntlet` invocation on Baseline-017.

## What is being tested

Baseline-016 (Pythia-70M) is the strongest detection submission to date — 3/4 with two first-of-kind threshold crossings. The natural next probe: **go EVEN SMALLER.** Baseline-017 uses **EleutherAI/pythia-14m** (14M params, 5× smaller than Pythia-70M).

Two possible outcomes are scientifically valuable:

1. **PASS / near-PASS:** if the curve continues to improve toward small sizes, Pythia-14M might push D1-length-delta over 0.10 — the first real PASS on the leaderboard. The seven-method floor breaks.
2. **Plateau or reversal:** if Pythia-14M is too small to assign meaningful per-token probabilities, signal degrades. We've found the optimum around 70M.

## Honest analytical prior

14M parameters is a toy-scale LM. At this size, the model has very limited capacity to distinguish nuanced token sequences. Two competing forces:

- **Pro-small force:** the inverse-scaling property says smaller LMs are MORE surprised by short canonical truth answers. At 14M, surprise on truth tokens should be even more extreme.
- **Anti-small force:** the model may be so small that its probability estimates are noisy across the board — surprise on misconception tokens might also be high, killing the discrimination gap.

I genuinely don't know which dominates. Pythia-160M → Pythia-70M (2.3× scaling) gave improvement (D2-axis cleared D4, D2-len passed). Pythia-70M → Pythia-14M is a 5× step — much larger.

## Pre-stated bars (the prediction)

| outcome | predicted probability |
|---|---|
| **Clear all 4 bars** (real PASS — first ever on the leaderboard) | **~15%** |
| **3/4 — same shape as Baseline-016** (D1+D2+D4) | ~25% |
| **2/4** (D2-axis still detectable, D1 falls) | ~25% |
| **1/4** (model too small, partial collapse) | ~20% |
| **0/4** (catastrophic small-end collapse) | ~15% |

### Specific AUC predictions

| metric | Pythia-70M actual | Baseline-017 predicted range |
|---|---|---|
| D1 AUC | 0.816 | 0.65–0.85 |
| D2 AUC | 0.907 | 0.75–0.93 |
| D1 − length-oracle | 0.026 | −0.10 to +0.12 |
| D2 − length-oracle | 0.103 | 0.00 to +0.18 |
| D1 − capratio-oracle abs | 0.113 | 0.00 to +0.18 |
| D2 − capratio-oracle abs | 0.115 | 0.00 to +0.20 |

Modal outcome: **2/4 — model too small to clear D1, but still detects D2 axis.** PASS at 15% is non-trivial — Pythia-14M is the smallest LM we've tested.

**Direction prediction:** misc > truth (seven prior confirmations across six baselines). High confidence.

## Why this bet matters

- **PASS (~15%):** the first real PASS. Pythia-14M would be the smallest LM tested AND the first to clear all four v3 bars. The seven-method floor breaks. Paper-grade headline.
- **3/4 (~25%):** confirms small-LM extrapolation; Pythia-70M was on the trajectory but Pythia-14M is at the optimum. Sets up next experiments to probe whether even smaller models exist.
- **2/4 (~25%):** the optimum was around Pythia-70M; Pythia-14M starts to degrade due to insufficient model capacity.
- **1/4 or 0/4 (~35%):** model too small to be useful. Argues the inverse-scaling curve has a floor below which signal collapses.

## Not re-running, not re-tuning

- Model: **EleutherAI/pythia-14m**.
- Algorithm identical to Baselines 011-016.
- Run once.

This document is committed to origin **before** the `styxx gauntlet` invocation on Baseline-017.
