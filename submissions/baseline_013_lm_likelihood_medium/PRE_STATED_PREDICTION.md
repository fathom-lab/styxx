# Pre-stated prediction — Baseline-013 (committed BEFORE gauntlet run)

**File created:** 2026-05-27, before any `styxx gauntlet` invocation on Baseline-013.

## What is being tested

Two data points on the gpt2 scaling curve so far:

- **Baseline-011** (gpt2-124M): D1=0.811, D2=0.897, D2-length-delta=0.093 → **3/4** (closest to PASS)
- **Baseline-012** (gpt2-large 774M): D1=0.682, D2=0.782, D2-length-delta=−0.022 → **1/4** (degradation)

Baseline-013 fills in the scaling curve with **gpt2-medium (355M parameters, ~2.8× larger than 124M, ~2.2× smaller than 774M)**. The question: is the degradation monotonic with size, or is there a sweet spot?

## The detector

Identical to Baseline-011/012 except for the checkpoint:

```python
_TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2-medium")
_MODEL = GPT2LMHeadModel.from_pretrained("gpt2-medium")
```

Same prefix, same per-token mean log-prob.

## Honest analytical prior

Three competing hypotheses for what gpt2-medium does:

1. **Linear-interpolation hypothesis (monotonic degradation):** gpt2-medium sits roughly halfway between Baseline-011 and Baseline-012 on every metric. D1 ≈ 0.74, D2 ≈ 0.84, D2-length-delta ≈ 0.035. Modal 2/4 (D2 only or D2+D4).
2. **Sweet-spot hypothesis:** gpt2-medium has the best signal-to-noise tradeoff — large enough to assign meaningful probabilities, small enough to retain the small-model truth-surprise discrimination. D1 ≈ 0.83, D2 ≈ 0.91, D2-length-delta ≈ 0.10+. Could PASS or near-PASS.
3. **Non-monotonic-but-not-sweet hypothesis:** gpt2-medium is at a transitional regime — degraded relative to 124M but not as bad as 774M. D1 ≈ 0.75, D2 ≈ 0.85, D2-length-delta ≈ 0.05. Modal 2/4.

I weight the linear-interpolation hypothesis highest (~50%). The sweet-spot hypothesis is intuitive but unsupported by the n=2 data we have; I weight it ~20%. The transitional hypothesis is the dominant alternative (~25%). PASS is ~5%.

## Pre-stated bars (the prediction)

| outcome | predicted probability |
|---|---|
| **Clear all 4 bars** (sweet-spot, real PASS) | **~5%** |
| **3/4 — pass D1+D2+D4 fail D3** (sweet-spot or near it) | ~20% |
| **2/4 — pass D2+D4 fail D1+D3** (intermediate) | ~30% |
| **2/4 — pass D2+D3 fail D1+D4** (unlikely but possible) | ~3% |
| **1/4 — pass D2 only** (monotonic degradation) | ~30% |
| **0/4 — total degradation** (faster degradation than expected) | ~12% |

### Specific AUC predictions

| metric | predicted range |
|---|---|
| D1 AUC | 0.70–0.84 |
| D2 AUC | 0.80–0.92 |
| D1 − length-oracle | −0.10 to +0.06 |
| D2 − length-oracle | **0.00 to +0.12** (modal: just below 0.10 → fail D3) |
| D1 − capratio-oracle abs | −0.05 to +0.13 |
| D2 − capratio-oracle abs | 0.00 to +0.15 |

Modal outcome: **2/4 — D2 + D4 pass, D1 + D3 fail.** The scaling-curve hypothesis predicts gradual degradation; this should be visible as a clearer 2/4 result than Baseline-012's 1/4.

**Direction prediction:** misconception > truth log-prob (high confidence — Baselines 011 and 012 both confirmed).

## Why this bet matters

Three scaling-curve data points (124M, 355M, 774M) is enough to fit a basic curve. If degradation is monotonic, the path forward is clear: stay small or switch families. If gpt2-medium turns out to be a sweet spot, the model-size hyperparameter is non-trivial and the next bet probes around 355M.

The most informative outcome is the **2/4 D2+D4 case at moderate AUC** (D1~0.78, D2~0.87). That falsifies both the sweet-spot hypothesis and the monotonic-degradation hypothesis and supports a "transitional regime" view.

## Not re-running, not re-tuning

- Model: **gpt2-medium** (HuggingFace standard checkpoint).
- All other parameters identical to Baselines 011 and 012.
- Run once.

This document is committed to origin **before** the `styxx gauntlet` invocation on Baseline-013.
