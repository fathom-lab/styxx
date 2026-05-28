# Pre-stated prediction — Baseline-016 (committed BEFORE gauntlet run)

**File created:** 2026-05-27, before any `styxx gauntlet` invocation on Baseline-016.

## What is being tested

The inverse-scaling property is now confirmed across two families and five data points:

| family | size | D2-len-delta | bars |
|---|---|---|---|
| gpt2 | 124M | +0.093 | 3/4 |
| **Pythia** | **160M** | **+0.116** | **2/4** (D2-axis cracked) |
| gpt2 | 355M | +0.040 | 2/4 |
| Pythia | 410M | +0.007 | 1/4 |
| gpt2 | 774M | −0.022 | 1/4 |

The natural next probe: **go SMALLER.** Baseline-016 uses **EleutherAI/pythia-70m** (70M params, 2.3× smaller than Baseline-014's 160M).

If the curve extrapolates: Pythia-70M might give D2-deltas around 0.15-0.20, decisively passing D3 and D4 on the D2 partition. The open question is what happens to D1. If the small-LM advantage extends to D1 as well, **this could be the first real PASS on the leaderboard** — the seven-method floor breaks.

## Honest analytical prior

Three regimes possible at 70M:

1. **Continued improvement** (the linear-extrapolation hypothesis): D2-deltas grow further past 0.10, AND D1-deltas might also improve enough to cross 0.10. PASS becomes plausible.
2. **Plateau** (signal saturates at the small end): Pythia-70M roughly matches Pythia-160M's 2/4 result, possibly with marginally stronger D2-deltas but D1 stuck.
3. **Reversal at the very small end** (curve is non-monotonic at the bottom): the model is too small to discriminate properly. D1 and D2 AUCs drop together.

I weight (1) and (2) roughly equally (~35% each), (3) at ~20%, and various heterogeneous outcomes at ~10%.

## Pre-stated bars (the prediction)

| outcome | predicted probability |
|---|---|
| **Clear all 4 bars** (real PASS — first ever on the leaderboard) | **~15%** |
| **3/4 — D1+D2+D4 pass, D3 fails** (D2-axis strong but D1 still short) | ~25% |
| **3/4 — D1+D2+D3-on-D2 only** (heterogeneous) | ~5% |
| **2/4 — D2 partition clean, matches Baseline-014** | ~25% |
| **2/4 — D2-clean degrades but still wins** | ~10% |
| **1/4 — D2 only** (model too small) | ~12% |
| **0/4** (catastrophic regression at 70M) | ~8% |

### Specific AUC predictions

| metric | predicted range | rationale |
|---|---|---|
| D1 AUC | 0.72–0.88 | extrapolation suggests improvement over Pythia-160M's 0.799 |
| D2 AUC | 0.85–0.96 | strong improvement expected |
| D1 − length-oracle | **−0.05 to +0.12** | first time predicting this might cross 0.10 |
| D2 − length-oracle | **+0.08 to +0.22** | extrapolation: should clear 0.10 |
| D1 − capratio-oracle abs | 0.05 to +0.18 | similar expectation |
| D2 − capratio-oracle abs | +0.10 to +0.25 | similar to D2-len |

Modal outcome: **3/4 (D1+D2+D4 pass, D3 fails)** — D2-deltas clear but D1-length still gets stuck because D1 includes factual-error + pseudoscience that aren't LM-typical the way folklore is.

**Direction:** misc > truth (five prior confirmations; high confidence).

## Why this bet matters

- **PASS (~15%):** the first real PASS. Pythia-70M would be the smallest LM tested AND the first to clear all four v3 bars. The seven-method floor breaks. Paper-grade headline.
- **3/4 (~30% combined):** confirms small-LM extrapolation and identifies Pythia-70M as the new leaderboard leader. Sets the stage for Baseline-017 to push even smaller (Pythia-31M exists).
- **2/4 (~35%):** the curve plateaus near Pythia-160M. The Pythia-160M result was the optimum.
- **1/4 or 0/4 (~20%):** non-monotonic curve at small end. Model too small. Limits the "smaller is better" claim.

## Not re-running, not re-tuning

- Model: **EleutherAI/pythia-70m** (HuggingFace standard checkpoint).
- Algorithm: identical to Baselines 011-015.
- Run once.

This document is committed to origin **before** the `styxx gauntlet` invocation on Baseline-016.
