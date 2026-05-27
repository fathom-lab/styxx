# Pre-stated prediction — Baseline-009 (committed BEFORE gauntlet run)

**File created:** 2026-05-27, before any `styxx gauntlet` invocation on Baseline-009.

## What is being tested

Baseline-008 (sentence-transformer cosine similarity) failed D3 because the embedding's misconception signal on the D1 partition was nearly identical to the length-only oracle (D1=0.805 vs length=0.790; delta 0.015 << 0.10 threshold). The obvious "smart fix" everyone would propose: **residualize the embedding similarity against length**, removing the length-correlated component, and score on the residual.

The honest question: does this give a method that clears D3, or does residualization just *destroy signal* without escaping the artifact?

## The detector

Same sentence-transformer (`all-MiniLM-L6-v2`) as Baseline-008. New step:

1. Pre-fit a linear regression `sim ~ word_length` on all (q, r) pairs in the benchmark — **unsupervised** (no class labels used in the fit; only the (sim, length) pairs).
2. At scoring time: `score = sim − (intercept + slope · word_length)`.

This is a one-time deterministic fit, committed in `method.py` before the gauntlet run. Coefficients are derived from `expected_consensus` length only, never from `class` labels.

## Honest analytical prior

Length-residualization removes the *length-correlated component* of the score. What remains is the part of similarity not predictable from length. Two scenarios:

- **If embedding similarity is purely length-driven**, the residual will be near-zero noise → D1/D2 AUC drop to ~0.5 → fails D1 + D2 outright.
- **If embedding has class-relevant signal beyond length**, the residual preserves it but at a much lower magnitude than the raw signal → D1/D2 AUC drop but stay above 0.5.

In *neither* scenario does residualization help D3. D3 measures `detector_AUC − length_oracle_AUC ≥ 0.10`. Residualization actively *lowers* detector AUC by removing the length-correlated component; the gap to the length oracle (whose AUC is unchanged) widens.

**The pre-stated bet: this method should FAIL D3 decisively, by a margin much larger than Baseline-008's marginal 0.015 miss.** Confirming this is the science — it argues that the path through D3 requires signal *orthogonal to length*, not signal *with length removed*.

## Pre-stated bars (the prediction)

| outcome | predicted probability |
|---|---|
| **Clear D3 cleanly** (a real PASS, all 4 bars) | **~3%** |
| **Pass D1+D2+D4, fail D3** | ~7% |
| **Pass D2+D4 only** | ~25% |
| **Pass D2 only** | ~30% |
| **Fail all 4 bars** | ~35% |

### Specific AUC predictions

| metric | predicted range |
|---|---|
| D1 (residual misconception AUC) | 0.50–0.62 |
| D2 (residual folklore AUC) | 0.60–0.78 |
| D1 − length-oracle | **−0.30 to −0.15** (residual far below length floor) |
| D2 − length-oracle | **−0.25 to −0.05** |
| D1 − capratio-oracle abs | **−0.20 to −0.05** |
| D2 − capratio-oracle abs | **−0.20 to 0.00** |

Modal outcome: **D2 passes (≥0.70), D1 fails, D3 fails wide, D4 marginal**.

## Why this bet matters either way

- **PASS (3% prior):** would mean length-residualization actually *does* surface clean semantic signal beyond the artifact. Surprise outcome, would prompt re-examination of the D3 bar's geometry. The synthesis gets its first deployable detection-axis positive.
- **Modal fail (≥65% prior):** n=1 evidence that length-residualization is not the path to D3. The "obvious fix" everyone would suggest after seeing Baseline-008 is empirically not sufficient. Argues for: NLI-based answer-form detection, specificity scoring, or genuinely orthogonal semantic features as the next research direction. Baseline-010 would then be the disciplined follow-up.

The valuable outcome is the same in either direction: a published n=1 result, pre-registered, against the same benchmark, under v3 bars (D1+D2+D3+D4).

## Not re-running, not re-tuning

- Same model as Baseline-008 (all-MiniLM-L6-v2, no other model considered).
- Residualization is a simple linear fit on `sim ~ word_length`, no hyperparameters.
- The fit is one-time, deterministic, committed to method.py with the coefficients.
- Run once on the bundled benchmark; result goes into submission.json regardless of outcome.

This document is committed to origin **before** the `styxx gauntlet` invocation on Baseline-009. Verifiable from git history.
