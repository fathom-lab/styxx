# PRE-REGISTRATION — can a SPAN-AGGREGATE single-pass signal RECOVER closed-model confab detection where the first-token gate fails?

**Written 2026-05-30, BEFORE the confirmatory run, AFTER the first-token result.** The first-token
closed-model run (`FINDING_detection_locus_gpt_2026_05_30.md`, commit `be55455`) found single-pass
first-token legibility FAILS on gpt-4o-mini: resampling instability AUC 0.974 but first-token
entropy/margin only 0.75-0.76, **B_contrast +0.216 ≥ 0.20 (SURVIVED)** — because the model
confabulates CONFIDENTLY downstream of the first token (correct leading digits, wrong trailing).
The first-token signal is blind to a trailing-digit error.

**But the OpenAI logprobs expose the top-20 distribution at EVERY answer token.** If the
confabulation lives in the trailing digits, a single-pass signal aggregated across the WHOLE answer
span — mean / max per-token entropy, min per-token margin — should SEE the uncertainty the first
token misses. Still ONE forward pass, NO resampling. This run tests whether a span-aggregate
single-pass gate RECOVERS closed-model confab detection to resampling parity.

## Item set (SAME as the first-token run — direct comparison)

`run_detection_locus_gpt_span.py`, seed `20260530`, identical 50 multiplication items (30 HARD
4x3/4x4/5x4 + 20 EASY 2x2/3x2), hash
`78a3e99c6c42c753b3619a9898378bc9602076bccac5cb31ff8eef189244b896` (matches the first-token run).
One greedy call per item captures the FULL answer-token logprobs; N=10 resamples give the Stability
baseline. CONFAB = HARD greedy-wrong, CORRECT = EASY greedy-right.

## Signals (label confab=1, correct=0; HIGHER = more-likely-confab)

Per answer token: entropy (top-20 + residual bucket) and margin (top1-top2 logprob), then:
- **B1**: resampling instability `1 − Stability` (baseline).
- **first-token**: `first_entropy`, `−first_margin` (the shipped `single_pass_confab` gate).
- **span-aggregate**: `mean_entropy`, `max_entropy`, `−mean_margin`, `−min_margin` over the span.

`best_first = max AUC(first-token signals)`; `best_span = max AUC(span-aggregate signals)`;
`B_contrast_first = AUC(B1) − best_first`; `B_contrast_span = AUC(B1) − best_span`.

## Predictions / bars

- **B1 (baseline):** AUC(instability) `≥ 0.70`.
- **B_span (core):** `best_span ≥ 0.70` — the span gate detects confab.
- **B_recovery (core):** `B_contrast_span < 0.20` — span ties resampling (no privileged access).
- **B_first_fails (motivating contrast):** `B_contrast_first ≥ 0.20` — first-token loses (replicates
  `be55455` on the same items).

**RECOVERY SURVIVED iff B1 ∧ B_span ∧ B_recovery ∧ B_first_fails** — a one-forward-pass span gate
recovers closed-model confab detection to resampling parity where the first-token gate fails.

**Reading (pre-committed):**
- **SURVIVED (recovery):** there exists a cheap (one forward pass, no resampling) closed-model confab
  signal — the trailing-digit uncertainty is visible in the span even though the first token is
  confident. This would be a deployable closed-model confab gate; productize as a span variant of
  `single_pass_confab`.
- **REPORT_AS_LANDED (no recovery, B_contrast_span ≥ 0.20):** the closed model is confident across the
  WHOLE answer span — its confabulation is single-pass-invisible at every token, and ONLY
  cross-sample resampling (which produces different confident wrong answers each time) reveals it.
  This would confirm resampling-based grounding (`grounded_honesty` / `audit_claim`) as the ONLY
  closed-model path and the single-pass approach as fundamentally white-box. Reported.

I do not know the direction in advance: the bet-0 "confident when wrong" finding argues for
span-invisibility (REPORT_AS_LANDED), but guessed trailing digits may carry genuine per-token
uncertainty (SURVIVED). The experiment decides.

## Honest scope (pre-committed)

Single closed model gpt-4o-mini via OpenAI API; multiplication only; one confirmatory run;
feasibility-grade; per-answer-token entropy/margin from top-20 logprobs + residual bucket (TRUNCATED
proxy — if anything understates entropy, so cannot manufacture a recovery); resampling N=10 at T=1.0
(exact-integer, no judge); ground truth in-code, hashed pre-scoring; same items/hash as the
first-token run. Difficulty confound (CONFAB-hard/CORRECT-easy) held fixed across detector TYPES by
the contrasts. Does NOT touch the correctness bound — every signal DETECTS confabulation, none
CORRECTS it.
