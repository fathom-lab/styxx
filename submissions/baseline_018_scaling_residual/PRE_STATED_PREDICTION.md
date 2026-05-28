# Pre-stated prediction — Baseline-018 (committed BEFORE gauntlet run)

**File created:** 2026-05-27, before any `styxx gauntlet` invocation on Baseline-018.

## What is being tested

Seven pre-registered baselines (011-017) traced the LM-likelihood scaling curve: gpt2-124M → 355M → 774M monotonic; Pythia-14M → 70M → 160M → 410M with a peak at 70M. The inverse-scaling finding is now the most-confirmed property of this method-class.

**The weird/innovative bet:** use the inverse-scaling finding itself as a FEATURE. Score each (q, r) pair by the *difference* in mean per-token log-prob between two Pythia models — the small one (70M, the peak) and a larger one (410M, well past the peak).

```
score = logp_pythia_70m(r | q) − logp_pythia_410m(r | q)
```

**Mechanism:** truth responses are dramatically *more surprising* to the smaller LM than the larger LM (the small-LM-truth-surprise is the whole reason inverse scaling works). Misconception responses (especially folklore) are similarly typical at both sizes. The DIFFERENCE isolates the "small-LM-specific surprise" signal that the inverse-scaling curve traced — using the curve's gradient rather than a point on it.

**Numerical example** (rough estimates based on mean scores from prior baselines):
- truth "Paris": logp at 70M ≈ −5.5, logp at 410M ≈ −3.4. Difference: **−2.1** (truth is much more surprising to the small LM)
- folklore restatement: logp at 70M ≈ −2.9, logp at 410M ≈ −2.2. Difference: **−0.7** (similar at both sizes)

Higher difference (less negative) → more misconception-like. The score should discriminate cleanly.

## The detector

Identical algorithm to Baselines 011-017 except the score is the DIFFERENCE between two model evaluations:

```python
def detect(q, r):
    logp_small = mean_per_token_logp(pythia_70m, q, r)
    logp_large = mean_per_token_logp(pythia_410m, q, r)
    return {"score": logp_small - logp_large}
```

No hyperparameters. Both models are off-the-shelf checkpoints already used in Baselines 014/015/016.

## Honest analytical prior

**Why this might clear D3 on D1 — the lone remaining bar:**
- Length is a CONFOUND because long responses have lower per-token logp variance, not because longer = more truth-like. The two LMs are similarly affected by length per token. The DIFFERENCE should be much less length-correlated than either individual logp signal.
- The D1 axis has been blocked because factual-error / pseudoscience are length-typical without being LM-typical the way folklore is. A difference signal should be relatively length-orthogonal *by construction*.
- The signal isolates exactly the discriminative-component that single-model scores entangle with model-fluency.

**Why this might fail:**
- Differences amplify NOISE. If both models have noisy logp estimates with similar mean but different variance, the difference signal could be largely noise.
- The discrimination gap I extrapolated (~−2.1 vs −0.7) is based on means; actual per-record difference distributions may overlap heavily.
- The D1 axis may be fundamentally untracked by LM-typicality regardless of framing (factual-error responses might just not be LM-typical, period).

## Pre-stated bars (the prediction)

| outcome | predicted probability |
|---|---|
| **Clear all 4 bars** (real PASS — the seven-method floor breaks) | **~22%** |
| **3/4 — same shape as Baseline-016** (D1+D2+D4, D3 fails on D1) | ~25% |
| **3/4 — D1+D2+D3** (rare shape; D4 fails) | ~3% |
| **2/4** | ~25% |
| **1/4** | ~12% |
| **0/4 — composite signal is mostly noise** | ~13% |

PASS probability of **22%** is the highest of any baseline this session — genuinely novel signal axis with mechanism-based reasoning suggesting D1-length-orthogonality.

### Specific AUC predictions

| metric | predicted range |
|---|---|
| D1 AUC | 0.70–0.90 |
| D2 AUC | 0.75–0.93 |
| D1 − length-oracle | **−0.05 to +0.20** (the wildcard — wide range because mechanism is novel) |
| D2 − length-oracle | 0.00 to +0.20 |
| D1 − capratio-oracle abs | 0.00 to +0.18 |
| D2 − capratio-oracle abs | 0.00 to +0.18 |

**Direction prediction:** truth has MUCH LOWER score (large negative difference); misconception has MILDLY NEGATIVE score (small magnitude). So misc > truth as before. High confidence.

## Why this bet matters

- **PASS (~22%):** the first real PASS. Validates that "use the scaling curve as a feature" is a generalizable technique for AI evaluation. Paper-grade headline result; goes into the recursive-discipline preprint as Section §11.
- **3/4 modal (~25%):** same ceiling as Baseline-016; argues the D1-length-delta gap is structural rather than method-dependent.
- **Wildcard outcomes:** D1-length-delta could land anywhere from −0.05 to +0.20 with this novel signal. The range is *informative*: if D1-len-delta lands above 0.10, this is a fundamental new direction. If below, the gap is real.

## Not re-running, not re-tuning

- Models: **EleutherAI/pythia-70m AND EleutherAI/pythia-410m** (both already cached from prior baselines).
- Algorithm: difference of mean per-token log-probs.
- No hyperparameters, no weighting, no calibration.
- Run once on the bundled benchmark.

This document is committed to origin **before** the `styxx gauntlet` invocation on Baseline-018.
