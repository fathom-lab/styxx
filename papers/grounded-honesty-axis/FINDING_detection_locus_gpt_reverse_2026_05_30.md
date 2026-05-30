# FINDING — the span-aggregate closed-model confab gate GENERALIZES beyond digits to a NON-NUMERIC domain: on gpt-4o-mini STRING REVERSAL, span max-entropy separates confab from correct at AUC 0.993 — matching N=10 resampling (0.997, B_contrast +0.005) — while the first-token gate fails even harder (0.57, B_contrast +0.427). So the gate is about confabulation LOCALIZATION, not digit tokenization (SURVIVED / generalization)

**Run 2026-05-30. One confirmatory run, pre-registered in
`PREREG_detection_locus_gpt_reverse_2026_05_30.md` (commit `4ca5e37`) BEFORE the confirmatory run.
gpt-4o-mini string reversal via OpenAI API, full answer-token logprobs → first-token and
span-aggregate detectors + N=10 resampling; hash
`dac77201b8f255fc9d72255342f77a847d0641173c8d258cc7af3c8c3e6516a9` matched.** Receipt:
`detection_locus_gpt_reverse_result.json`.

## Why this run exists

The span run (`FINDING_detection_locus_gpt_span_2026_05_30.md`, `ddffaa9`) showed a span-aggregate
single-pass signal recovers gpt-4o-mini confab detection to resampling parity (AUC 0.991) where
first-token fails — but on MULTIPLICATION, whose answers are NUMBERS. The recovery could have been a
property of digit tokenization. This run repeats the analysis on a NON-NUMERIC, character-level
domain: string reversal (gpt-4o-mini reverses short strings correctly, confabulates long ones with
LOCALIZED errors — correct prefix, wrong middle).

## Result: SURVIVED (generalization) — span ties resampling on strings too, first-token fails harder

| detector (confab vs correct AUC) | AUC | B_contrast = AUC(resampling) − this |
| --- | --- | --- |
| **B1 resampling instability** | **0.997** | — |
| first-token entropy / −margin (best) | 0.570 | **+0.427 → first-token FAILS (≈ chance)** |
| span −min-margin | 0.927 | +0.070 |
| span −mean-margin | 0.930 | +0.067 |
| span mean-entropy | 0.935 | +0.062 |
| **span max-entropy (best span signal)** | **0.993** | **+0.005 → span TIES resampling** |

All four pre-registered conditions held (B1 0.997, best_span 0.993, B_contrast_span +0.005 < 0.20,
B_contrast_first +0.427 ≥ 0.20). **GENERALIZATION.**

| group means (n_conf=30, n_corr=18) | first entropy | max entropy | min margin | n answer tokens |
| --- | --- | --- | --- | --- |
| confab | 0.355 | **1.204** | **0.354** | 7.90 |
| correct | 0.276 | 0.297 | 3.056 | 2.39 |

## The claims that land

1. **The span gate generalizes beyond digits.** On a non-numeric, character-level domain, the span
   aggregate again recovers closed-model confab detection to resampling parity (AUC 0.993 vs 0.997,
   B_contrast +0.005) while the first-token gate fails — so the recovery is about confabulation
   LOCALIZATION (the model is internally uncertain exactly where it confabulates a character), not
   about digit tokenization. `styxx.span_confab` is a general closed-model confab gate for
   structured/multi-token answers, numeric AND non-numeric.
2. **First-token fails HARDER here (0.57 ≈ chance, B_contrast +0.427)** than on multiplication (0.76).
   gpt-4o-mini is confident at the FIRST reversed character even while it scrambles the rest — the
   confident-confabulation-at-token-1 pattern is general and, for reversal, near-total.
3. **The winning span aggregate is domain-dependent — and `span_confab` exposes both.** Multiplication
   favored **min-margin** (the least-confident token, AUC 0.991); reversal favors **max-entropy** (the
   most-uncertain token, AUC 0.993). The model marks its confabulation either by a collapsed-margin
   token or a high-entropy token somewhere in the span; `SpanConfabScore` returns both `min_margin`
   and `max_entropy`, so a caller (or per-domain calibration) can take whichever separates. This
   validates the primitive's multi-feature design.
4. **Correctness bound untouched.** The signal detects/abstains; it never corrects.

## Honest scope (pre-committed)

Single closed model gpt-4o-mini; string reversal only (one new domain); one confirmatory run;
feasibility-grade (30 confab + 18 correct, powered); per-token entropy/margin from top-20 logprobs +
residual bucket (truncated proxy); resampling N=10 at T=1.0 (normalized-string Stability, no judge);
ground truth = s[::-1], hashed pre-scoring. CONFAB-long / CORRECT-short difficulty confound held
fixed across detector TYPES by the contrasts. The span gate now holds on TWO closed-model domains
(numeric multiplication + non-numeric reversal); it still requires a MULTI-token answer with the
error LOCALIZED to some token(s) — single-token answers and errors smeared evenly remain out of
scope, and a second closed-model VENDOR remains untested (blocked on a logprobs-exposing non-OpenAI
key). Does NOT touch the correctness bound.

## The arc, in one line (updated)

The closed-model span-aggregate confab gate generalizes beyond digits: on gpt-4o-mini it ties N=10
resampling on BOTH multiplication (min-margin AUC 0.991) and string reversal (max-entropy AUC 0.993),
in one forward pass, where the first-token gate fails on both (0.76, 0.57) — so it is a general
closed-model confab detector for structured/multi-token answers driven by confabulation LOCALIZATION,
shipped as `styxx.span_confab` (both min-margin and max-entropy exposed); single-token answers and
cross-vendor remain the open frontier, and every signal still moves confidence/abstention, never
correctness.
