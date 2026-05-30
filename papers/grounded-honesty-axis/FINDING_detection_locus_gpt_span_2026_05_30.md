# FINDING — a SPAN-AGGREGATE single-pass signal RECOVERS closed-model confabulation detection to EXACT resampling parity (AUC 0.991 = resampling 0.991, B_contrast 0.000) where the first-token gate fails (0.76): on gpt-4o-mini, the least-confident token's margin across the answer span separates confab from correct in ONE forward pass, no resampling — a cheap closed-model confab gate (SURVIVED / RECOVERY)

**Run 2026-05-30. One confirmatory run, pre-registered in
`PREREG_detection_locus_gpt_span_2026_05_30.md` (commit `44ef6b2`) BEFORE the confirmatory run. Same
50 multiplication items as the first-token closed-model run (hash
`78a3e99c6c42c753b3619a9898378bc9602076bccac5cb31ff8eef189244b896` matched), gpt-4o-mini via OpenAI
API, full answer-token logprobs (`top_logprobs=20`) → per-token entropy/margin aggregated across the
span; N=10 resamples for the Stability baseline.** Receipt: `detection_locus_gpt_span_result.json`.

## Why this run exists

The first-token closed-model run (`FINDING_detection_locus_gpt_2026_05_30.md`, `be55455`) found
single-pass FIRST-TOKEN legibility FAILS on gpt-4o-mini (B_contrast +0.216) — the model confabulates
confidently downstream of the first token (correct leading digits, wrong trailing). But the logprobs
expose the top-20 at EVERY answer token, so a single-pass signal aggregated across the WHOLE answer
span should see the trailing-digit uncertainty the first token misses — still one forward pass.

## Result: SURVIVED (recovery) — span ties resampling, first-token fails

| detector (confab vs correct AUC) | AUC | B_contrast = AUC(resampling) − this |
| --- | --- | --- |
| **B1 resampling instability** | **0.991** | — |
| first-token entropy / −margin (best) | 0.762 | **+0.229 → first-token FAILS** |
| span mean-entropy | 0.954 | +0.037 |
| span max-entropy | 0.960 | +0.031 |
| span −mean-margin | 0.952 | +0.039 |
| **span −min-margin (best span signal)** | **0.991** | **0.000 → span TIES resampling** |

All four pre-registered conditions held: B1 ≥ 0.70 (0.991), B_span ≥ 0.70 (0.991), B_recovery
(B_contrast_span < 0.20 → **0.000**), B_first_fails (B_contrast_first ≥ 0.20 → 0.229). **RECOVERY.**

| group means (n_conf=29, n_corr=18) | first entropy | mean entropy | max entropy | min margin | n answer tokens |
| --- | --- | --- | --- | --- | --- |
| confab | 0.233 | 0.588 | **1.181** | **0.293** | 2.90 |
| correct | 0.089 | 0.073 | 0.140 | **9.594** | 1.89 |

## The claims that land

1. **A one-forward-pass span-aggregate gate recovers closed-model confab detection to EXACT
   resampling parity.** The best span signal — the **minimum per-token margin across the answer
   span** — reaches AUC 0.991, identical to N=10 resampling (0.991), B_contrast 0.000, where the
   first-token gate musters only 0.76 (B_contrast +0.229). The closed-model confabulation IS
   single-pass-legible; it just lives downstream of the first token.
2. **The least-confident token is the tell.** In a confabulated product, the lowest-margin token in
   the span sits at margin 0.293 — the model is at a near-coin-flip exactly where it guesses a digit.
   In a correct answer, even the least-confident token holds margin 9.594. The model "knows" which
   digit it is unsure of; that single token, read in one pass, is as informative as ten full
   resamples. (Max-entropy 1.18 vs 0.14 tells the same story from the entropy side, AUC 0.96.)
3. **This nuances "confident confabulation."** gpt-4o-mini is confident at the FIRST token (magnitude
   is easy) but NOT across the span (the specific digits are not). The bet-0 confident-when-wrong
   regime is first-token confidence, not whole-answer confidence — so confident hallucination is more
   single-pass-detectable than the first-token result alone suggested, provided the answer has
   internal token structure and the error is localized within it.
4. **Product: a cheap closed-model confab gate exists.** It matches resampling-based grounding
   (`grounded_honesty`, 0.991 here) at ~10× fewer forward passes (one greedy call with logprobs vs
   ten resamples). Productized as `styxx.span_confab` (v7.7.14), the span variant of
   `single_pass_confab`.
5. **Correctness bound untouched.** The signal detects/abstains; it never corrects.

## Honest scope (pre-committed)

Single closed model gpt-4o-mini via OpenAI API; multiplication only; one confirmatory run;
feasibility-grade (29 confab + 18 correct, powered); per-answer-token entropy/margin from top-20
logprobs + residual bucket (TRUNCATED proxy — if anything understates uncertainty, so cannot
manufacture the recovery); resampling N=10 at T=1.0 (exact-integer, no judge); same items/hash as
`be55455` (direct first-token-vs-span comparison). **The recovery requires the answer to have
MULTIPLE tokens with the error LOCALIZED to some of them** (here: trailing digits of a multi-token
number). A single-token answer has no span (span = first-token → no recovery), and an error smeared
evenly across all tokens would not localize. So `span_confab` is the closed-model gate for
structured/multi-token answers with localized errors; short single-token factual answers fall back
to the first-token regime. One closed model, one domain; whether min-margin-across-span generalizes
beyond multi-digit arithmetic to other multi-token confabulations (citations, code, multi-fact
claims) is the next test. Does NOT touch the correctness bound.

## The arc, in one line (updated)

Single-pass confabulation legibility is cross-architecture/family/domain on small white-box models
and extends to factual recall; on the strong closed model gpt-4o-mini the FIRST-TOKEN gate FAILS
(error downstream, B_contrast +0.22) but a SPAN-AGGREGATE single pass — the least-confident token's
margin across the answer — RECOVERS it to EXACT resampling parity (AUC 0.991, B_contrast 0.000) in
one forward pass; so confident confabulation is first-token confidence not span confidence, a cheap
(~10×) closed-model confab gate exists for structured answers (shipped as `styxx.span_confab`), and
every signal still moves confidence/abstention, never correctness.
