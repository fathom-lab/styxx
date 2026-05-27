# Pre-stated prediction — Baseline-012 (committed BEFORE gauntlet run)

**File created:** 2026-05-27, before any `styxx gauntlet` invocation on Baseline-012.

## What is being tested

Baseline-011 (gpt2, 124M) scored 3/4 with D2 length-delta = 0.093 — just **0.007 short** of the D3 threshold of 0.10. The natural follow-up: scale up the language model within the same family (same training data, same tokenizer, same architecture) and ask whether the LM-typicality signal sharpens enough to clear D3.

The bet: same method (mean per-token log-probability of response tokens under prefix `"Question: {q}\nAnswer: "`), but using **gpt2-large (774M parameters, 6× the size of gpt2-124M)** in place of the base gpt2 checkpoint.

This is the cleanest possible scaling experiment for the LM-typicality hypothesis. If the signal scales with model size, D2 length-delta should grow past 0.10 and we get the first real PASS on the leaderboard. If the signal saturates at the gpt2-family scale, deltas stay flat and we learn that the path through D3 requires either a different model family or a different signal entirely.

## The detector

Identical to Baseline-011 except for the model checkpoint:

```python
_TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2-large")
_MODEL = GPT2LMHeadModel.from_pretrained("gpt2-large")
```

Same prefix format. Same per-token mean log-prob aggregation. Same tokenization fix (prefix without trailing space, response with leading space).

## Honest analytical prior

**Why this might clear D3 on D2:**
- Larger LMs assign sharper per-token probabilities — the difference between LM-typical (folklore) and LM-surprising (truth) responses is more pronounced at scale.
- Baseline-011 was *just barely* 0.007 short on D2. Modest improvement in discrimination should close that gap.
- gpt2-large was trained on the same WebText corpus as gpt2; the typicality biases compound at scale rather than getting corrected (no RLHF or instruct-tuning between the two).

**Why this might still fail D3:**
- The length-only oracle's AUC is fixed at 0.79/0.80 regardless of the detector. If D2 detector AUC only grows by a few hundredths (e.g., 0.90 → 0.92), the length-delta moves from 0.093 → 0.117 — clears D3. But if it doesn't grow at all (or grows uniformly with the length oracle), no improvement.
- D1 includes factual-error + pseudoscience categories where LM-typicality may not discriminate truth from misconception as sharply as folklore does. D1 length-delta = 0.022 is much further from 0.10 than D2 was; even a sharper model may not close this gap.

**Direction prediction:** **Same as Baseline-011** — misconception responses score higher mean log-prob than truth. High confidence given Baseline-011 already confirmed this on the same architecture-family.

## Pre-stated bars (the prediction)

| outcome | predicted probability |
|---|---|
| **Clear all 4 bars** (real PASS — D2 length-delta crosses 0.10, D1 length-delta also crosses 0.10) | **~12%** |
| **3/4 — same shape as Baseline-011** (D1+D2+D4 pass, D3 fails because D1-length insufficient) | **~35%** (modal) |
| **3/4 — different shape** (D1+D2+D3 pass on D2 axis, D4 fails) | ~5% |
| **2/4** (D2 + D4) | ~15% |
| **1/4 or 0/4** (signal degrades — unlikely given direction confirmed) | ~8% |
| **D2 alone clears D3** (D2-length passes, D1-length doesn't) — would still count as 3/4 overall because D3 requires BOTH partitions to pass | ~25% |

### Specific AUC predictions

| metric | Baseline-011 actual | Baseline-012 predicted range |
|---|---|---|
| D1 AUC | 0.811 | 0.80–0.92 |
| D2 AUC | 0.897 | 0.88–0.95 |
| D1 − length-oracle | 0.022 | −0.02 to +0.12 |
| D2 − length-oracle | 0.093 | **0.05 to 0.18** (modal: just above 0.10) |
| D1 − capratio-oracle abs | 0.108 | 0.05 to 0.20 |
| D2 − capratio-oracle abs | 0.105 | 0.05 to 0.20 |

Modal outcome: **3/4** (same shape as Baseline-011, possibly with D2-length-delta now CLEARING 0.10 but D1-length-delta still failing). The D3 bar requires *both* partition deltas ≥ 0.10, so even if D2 passes, D1 needs to also.

PASS probability ~12% requires BOTH D1-length-delta and D2-length-delta to clear 0.10 simultaneously. D2 is plausible; D1 is hard.

## Why this bet matters either way

- **PASS (~12% prior):** the first real PASS on the gauntlet. The seven-method floor breaks. Per-token LM log-probability at moderate scale (gpt2-large, 774M) is empirically the path through the dark core's confounds. Paper-grade result.
- **Modal 3/4 (~60% prior with D2-length-delta now clearing 0.10):** strongest signal yet on the D2 axis but D1 still fails. Argues for either:
  - (a) The D1 partition is genuinely harder (factual-error/pseudoscience are not LM-typical the way folklore is), or
  - (b) Even larger LMs (gpt2-xl, 1.5B; Pythia-6.9B; Llama-3-8B-base) might close the D1 gap.
- **Failure to improve over Baseline-011:** would mean the gpt2-family architecture saturates at small scales for this task. Future research should switch model families.

## Not re-running, not re-tuning

- Model: **gpt2-large** (HuggingFace `gpt2-large` checkpoint, 774M params).
- Prefix format: identical to Baseline-011.
- Aggregation: identical (mean per-token log-prob).
- No hyperparameters, no thresholds.
- Run once on the bundled benchmark.

This document is committed to origin **before** the `styxx gauntlet` invocation on Baseline-012. Verifiable from git history.
