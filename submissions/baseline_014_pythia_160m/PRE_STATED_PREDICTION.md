# Pre-stated prediction — Baseline-014 (committed BEFORE gauntlet run)

**File created:** 2026-05-27, before any `styxx gauntlet` invocation on Baseline-014.

## What is being tested

The [scaling-curve FINDING](../../papers/agent-self-audit/FINDING_lm_likelihood_scaling_curve_2026_05_27.md) established monotonic inverse scaling within the gpt2 family: 124M (3/4) → 355M (2/4) → 774M (1/4). The next disciplined cut: **is this a gpt2-family-specific effect or a general property of small LMs?**

Baseline-014 uses **EleutherAI/pythia-160m** — a 160M-parameter LM at approximately the same size as gpt2-124M, but trained on **The Pile** (mostly Wikipedia, ArXiv, books, code) rather than WebText (Reddit-linked URLs). Different tokenizer (GPT-NeoX BPE), different architecture details (parallel attention, rotary embeddings), different training data distribution.

Same method (mean per-token log-probability), same prefix, same aggregation.

## Honest analytical prior

Two competing hypotheses:

1. **Size-is-what-matters hypothesis.** Small LMs are surprised by short canonical truth answers regardless of family. Pythia-160M should give a 3/4 result similar to gpt2-124M, perhaps slightly weaker because The Pile contains more correct factual content (Wikipedia) than WebText.
2. **WebText-is-what-matters hypothesis.** gpt2's training data (Reddit-linked pages from 2019) is *specifically* rich in folklore-style misconception framings. Pythia trained on The Pile has more academic/Wikipedia content → less common-belief contamination → smaller misconception-typicality signal → underperforms gpt2-124M.

I weight (1) and (2) roughly equally. The composition of The Pile is genuinely different from WebText in ways that should affect this task — but I don't have strong evidence either way about the magnitude.

## Pre-stated bars (the prediction)

| outcome | predicted probability |
|---|---|
| **Clear all 4 bars** (real PASS — size-is-what-matters AND The Pile helps somehow) | **~8%** |
| **3/4 — matches gpt2-124M (D1+D2+D4 pass, D3 fails)** | ~25% |
| **2/4 — D1+D2 pass (like Baseline-013)** | ~25% |
| **2/4 — D2+D4 pass** | ~10% |
| **1/4 — D2 only (family-effect dominates)** | ~20% |
| **0/4 — total degradation** | ~12% |

### Specific AUC predictions

| metric | predicted range |
|---|---|
| D1 AUC | 0.62–0.82 (wide — family-specificity uncertain) |
| D2 AUC | 0.68–0.90 |
| D1 − length-oracle | −0.15 to +0.08 |
| D2 − length-oracle | −0.10 to +0.12 |
| D1 − capratio-oracle abs | −0.08 to +0.15 |
| D2 − capratio-oracle abs | −0.05 to +0.15 |

**Direction prediction:** misc > truth log-prob (high confidence given 3 prior confirmations). If direction flips here, it would be the eleventh in-session falsification and strong evidence that family/training-data matters more than size.

## Why this bet matters

- **PASS (~8%):** Pythia-160M would be the first real PASS on the leaderboard AND would suggest small-LM-detection is robust across families. Major positive result.
- **3/4 matching gpt2-124M (~25%):** confirms size-is-what-matters; the gpt2-WebText story is incidental to a more general "small LMs detect dark-core misconceptions" effect. Strong methodological result.
- **2/4 or worse (~67% combined):** family-specificity matters. The gpt2 result is partly about WebText. Future research must consider training-data composition as a hyperparameter alongside model size.
- **Direction flip (low prior but high value):** would FALSIFY the "LM-typicality detects misconceptions" hypothesis in its current form and force a complete rethink.

## Not re-running, not re-tuning

- Model: **EleutherAI/pythia-160m** (HuggingFace standard checkpoint, the deduped variant if available; standard otherwise).
- Algorithm: identical to Baselines 011-013.
- Prefix: identical.
- Tokenizer: whatever ships with the Pythia checkpoint (GPT-NeoX BPE).
- Run once.

This document is committed to origin **before** the `styxx gauntlet` invocation on Baseline-014.
