# Pre-stated prediction — Baseline-011 (committed BEFORE gauntlet run)

**File created:** 2026-05-27, before any `styxx gauntlet` invocation on Baseline-011.

## What is being tested

After Baseline-010's null result (cross-encoder NLI doesn't transfer to factual-restatement detection on this corpus), the next disciplined bet is a *principled* length-independent signal: **per-token average log-probability of the response under the question prefix, scored by a local language model (gpt2)**.

The hypothesis being tested has a specific mechanism. Misconception responses are typically the *common-belief* version of a topic — folklore restatements are exactly the kind of text a language model trained on web data would predict. Truth responses, especially when they contradict common belief ("No, this attribution is unfounded"), are *more surprising* to the same LM because they're less frequent in pretraining data.

So: score = mean per-token log-probability of response tokens under `prefix = "Question: {q}\nAnswer: "`. Higher log-prob = more LM-typical = more misconception-like under the hypothesis.

## The detector

```python
def detect(question, response):
    prefix = f"Question: {question}\nAnswer: "
    full = prefix + response
    prefix_ids = tokenize(prefix)
    full_ids = tokenize(full)
    response_ids = full_ids[len(prefix_ids):]
    logits = gpt2(full_ids).logits
    # log-prob of each response token under the model
    response_logprobs = [log_softmax(logits[i-1])[response_ids[i - len(prefix_ids)]]
                        for i in range(len(prefix_ids), len(full_ids))]
    return {"score": mean(response_logprobs)}
```

Model: **gpt2** (HuggingFace, 124M parameters, the smallest standard gpt2 checkpoint). No fine-tuning, no prompt engineering beyond the prefix. Per-token normalization makes the score length-independent by construction.

## Honest analytical prior

**Why this could pass D3+D4:**
- Per-token mean log-probability is *normalized for length by definition*. Adding more tokens doesn't inflate the score; it's the *average* log-prob per token.
- The signal is genuinely orthogonal to cap-ratio — capitalization affects single-token surface forms but doesn't dominate per-token probability under a 50k-vocab subword tokenizer.
- The mechanism (LM matches common-belief patterns from pretraining) is real and well-documented in the broader LM-as-judge literature.

**Why this could fail D1/D2:**
- gpt2 is a 2019-era 124M-parameter model. Its signal-to-noise on factuality may be far below what a modern LM (Llama-3, Qwen-3, Phi-3) would provide.
- The benchmark's truth responses are often single tokens ("Paris", "Newton"). Per-token logprob on a single token is noisy.
- Misconception responses span multiple classes (folklore vs factual-error vs pseudoscience). gpt2's typicality signal may discriminate folklore from truth but not factual-error from truth.

**Why this could fail D3 even if D1/D2 pass:**
- Per-token logprob is length-normalized at the score level, but the *underlying distribution* of per-token logprobs may correlate with response style in ways that correlate with length. If short canonical answers have systematically lower log-prob than long restatements (because single-token surface forms are inherently less predictable from a 2-word prefix), the score still correlates with length indirectly.

## Pre-stated bars (the prediction)

| outcome | predicted probability |
|---|---|
| **Clear all 4 bars** (real PASS) | **~12%** |
| Pass D1+D2+D4, fail D3 | ~15% |
| Pass D2 only or D2+D4 | ~30% |
| 1/4 (only one bar) | ~20% |
| 0/4 total fail | ~23% |

### Specific AUC predictions

| metric | predicted range |
|---|---|
| D1 (misconception AUC) | 0.55–0.72 |
| D2 (folklore AUC) | 0.58–0.78 |
| D1 − length-oracle | **−0.20 to +0.15** (genuine uncertainty whether per-token logprob escapes length) |
| D2 − length-oracle | **−0.15 to +0.20** |
| D1 − capratio-oracle abs | **−0.10 to +0.20** |
| D2 − capratio-oracle abs | **0.00 to +0.20** |

Modal outcome: **2/4 (D2 + something else)** or **1/4 (D2 only)**. PASS probability is ~12% — non-trivial but still pessimistic because gpt2 is a small old model.

Direction prediction: **misconception responses score HIGHER log-probability** (LM matches common belief). If this direction flips (truth scores higher), record it as the second direction-of-effect falsification in this domain.

## Why this bet matters either way

- **PASS (~12% prior):** the first real PASS on the leaderboard. Demonstrates that LM-typicality (perplexity / log-prob) is the path through the dark core's length-and-cap-ratio confounds. Real news, paper-grade.
- **Modal fail (~65% prior):** another n=1 receipt for the "method-class × bars" matrix. Argues for either larger LMs (Phi-3, Qwen-3) as the next step or genuinely different feature axes.
- **Direction flip (uncertain prior):** would be the ninth in-session falsification and a third direction-of-effect miss on this domain — durable enough to call a domain-property: pre-stated direction predictions on dark-core benchmarks are unreliable, magnitude predictions are roughly 2/3 reliable.

## Not re-running, not re-tuning

- Model: **gpt2** (no other model considered; specifically the smallest 124M variant).
- Prefix format: **"Question: {q}\nAnswer: "** (no other format considered).
- Scoring: mean per-token log-probability of response tokens (no other aggregation considered).
- No hyperparameters, no thresholds.
- Run once on the bundled benchmark.

This document is committed to origin **before** the `styxx gauntlet` invocation on Baseline-011. Verifiable from git history.
