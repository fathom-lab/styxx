# Pre-stated prediction — Baseline-010 (committed BEFORE gauntlet run)

**File created:** 2026-05-27, before any `styxx gauntlet` invocation on Baseline-010.

## What is being tested

After Baseline-009 demonstrated that length-residualization is *not* the path through D3 (signal subtracted, not signal added), the disciplined next bet is a method that uses **genuinely length-independent information**: a pre-trained NLI cross-encoder scoring whether the response semantically relates to the question.

The hypothesis: misconception responses (especially folklore) **restate the question's premise as if true**. A response like "Marie Antoinette said let them eat cake" to "Did Marie Antoinette say let them eat cake?" is a textual restatement. Truth responses ("No, this attribution is unfounded") are short answers that often *contradict* or *don't restate* the question.

So: feed (response, question) into an NLI cross-encoder. Score by the **entailment probability** — the probability that the response, as a premise, entails the question as a hypothesis. Folklore restatements often score high entailment (they restate the question affirmatively); truth answers often score contradiction or neutral.

The bet: is this signal genuinely orthogonal to length AND cap-ratio (which would let it clear D3 and D4), and is the cross-encoder good enough to surface it at AUC ≥ 0.70?

## The detector

```python
def detect(question, response):
    # premise = response, hypothesis = question
    scores = cross_encoder.predict([(response, question)])
    # scores[0] = [contradiction, entailment, neutral]
    p = softmax(scores[0])
    return {"score": p[1]}  # entailment probability
```

Model: `cross-encoder/nli-deberta-v3-base` (available via `sentence-transformers`).
Frame: R-as-premise, Q-as-hypothesis, entailment probability as score.
No hyperparameters. Run once on the bundled benchmark.

## Honest reasoning

**Why this might pass D3 and D4:**
- NLI entailment is, in principle, content-driven rather than length-driven. A 5-word response that affirms the question scores higher entailment than a 5-word response that contradicts.
- It's not directly correlated with cap-ratio either; entailment doesn't care about proper-noun density.
- Exploratory single-sample probes show the signal exists in the right direction on a few cases (folklore restatements like "Fortune cookies originated in ancient China" entail "Where did fortune cookies originate?" at 0.95; truth "Paris" contradicts "What is the capital of France?" at 0.99).

**Why this might fail D1/D2:**
- The signal is noisy on individual samples. Some truth responses score high entailment (Newton entails "Who discovered gravity?" at 0.94 — correct but it scored "entailment", not "contradiction"). Some folklore responses score neutral (Marie Antoinette restatement, brain 10% restatement). The 108-record AUC depends on whether the signal aggregates above noise.
- Cross-encoder NLI models are trained on SNLI/MNLI, not on this style of factual-restatement detection. The transfer may be poor.

**Why this might still fail D3 even if D1/D2 pass:**
- If the NLI signal happens to correlate with length (perhaps because longer responses contain more entailment-relevant content), the D3 gap may still be < 0.10.

## Pre-stated bars (the prediction)

| outcome | predicted probability |
|---|---|
| **Clear D3 cleanly** (real PASS, 4/4) | **~15%** |
| 3/4 — D1+D2+D4 pass, D3 fails | ~20% |
| 3/4 — D1+D2+D3 pass, D4 fails | ~5% |
| 2/4 — D2 only + one other | ~25% |
| 1/4 — D2 only | ~15% |
| 0/4 — NLI signal isn't there | ~20% |

### Specific AUC predictions

| metric | predicted range |
|---|---|
| D1 (misconception AUC) | 0.55–0.78 |
| D2 (folklore AUC) | 0.65–0.85 |
| D1 − length-oracle | **−0.20 to +0.15** (wide — genuinely uncertain whether NLI escapes length) |
| D2 − length-oracle | **−0.10 to +0.20** |
| D1 − capratio-oracle abs | **−0.10 to +0.20** |
| D2 − capratio-oracle abs | **0.00 to +0.20** |

Modal outcome: 2/4 or 3/4. PASS probability is genuinely uncertain because (a) cross-encoder NLI quality on factual-restatement detection is unproven, and (b) even if the signal is real, it may correlate with length enough to fail D3.

## Why this bet matters either way

- **PASS (~15% prior):** the first real PASS on the leaderboard. The seven-method floor breaks. NLI-based answer-form detection becomes the demonstrated path through the dark core. Real news.
- **Modal fail (~65% prior, in 1/4–3/4):** the NLI signal exists on some axes but doesn't clear all the bars. Each n=1 outcome is diagnostic about *which* axes NLI captures (probably the broad misconception axis less than the folklore-specific axis).
- **Total fail (~20% prior):** the cross-encoder doesn't surface usable signal on this task. Argues for either fine-tuned NLI, a larger model, or a different orthogonal signal as the next bet.

## Not re-running, not re-tuning

- Model: `cross-encoder/nli-deberta-v3-base` (no other model considered).
- Frame: R-as-premise, Q-as-hypothesis, entailment probability (no other frame considered).
- No hyperparameters, no thresholds, no calibration.
- Run once on the bundled benchmark; result goes into submission.json regardless of outcome.

This document is committed to origin **before** the `styxx gauntlet` invocation on Baseline-010. Verifiable from git history.
