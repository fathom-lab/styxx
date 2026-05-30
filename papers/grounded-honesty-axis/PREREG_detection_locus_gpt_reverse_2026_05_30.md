# PRE-REGISTRATION — does the span-aggregate closed-model confab gate generalize BEYOND digits to a NON-NUMERIC domain (string reversal)?

**Written 2026-05-30, BEFORE the confirmatory run.** The span run
(`FINDING_detection_locus_gpt_span_2026_05_30.md`, `ddffaa9`) found that on gpt-4o-mini
multiplication, the least-confident token's margin across the answer span (`min_margin`) recovers
confabulation detection to N=10-resampling parity (AUC 0.991, B_contrast 0.000) where the first-token
gate fails (0.76) — shipped as `styxx.span_confab`. But multiplication answers are NUMBERS: the
recovery could be a property of digit tokenization rather than of confabulation in general. **This
run repeats the span analysis on a NON-NUMERIC, character-level domain — STRING REVERSAL.**

A greedy probe confirms the structure: gpt-4o-mini reverses SHORT strings correctly (len 4-8) and
confabulates LONG ones (len 11+: 0-1/6 correct), with LOCALIZED errors — e.g. `izhzqmdhkaqwos` →
`sowq…` (correct prefix) then wrong middle. The answers are multi-token (5-11 tokens). If span
`min_margin` still ties resampling here while first-token fails, the gate is about confab
LOCALIZATION, not digits.

## Item set (string reversal; seeded, ground truth = s[::-1])

`run_detection_locus_gpt_reverse.py`, seed `20260530`. HARD = 30 random lowercase strings of length
13/15/17 (gpt-4o-mini greedy-wrong → CONFAB) + EASY = 24 of length 4/5/6 (greedy-right → CORRECT).
System "Reverse the given string … Output only the reversed string, nothing else." Scored by
NORMALIZED exact-string match (strip/lower/dequote), no judge.

**Answer-key SHA-256 (54 items, pinned pre-scoring):**
`dac77201b8f255fc9d72255342f77a847d0641173c8d258cc7af3c8c3e6516a9`

- **CONFAB** = HARD strings greedy-wrong (normalized answer ≠ true reverse, non-empty).
- **CORRECT** = EASY strings greedy-right.

## Signals / bars (identical to the span run)

Per answer token: entropy (top-20 + residual) and margin (top1-top2 logprob). B1 = resampling
instability (normalized-string distinct-count Stability). first-token = best AUC of
`first_entropy`/`−first_margin`. span = best AUC of `mean_entropy`/`max_entropy`/`−mean_margin`/
`−min_margin`. `B_contrast_first = AUC(B1) − best_first`; `B_contrast_span = AUC(B1) − best_span`.

**GENERALIZATION SURVIVED iff B1 ≥ 0.70 ∧ best_span ≥ 0.70 ∧ B_contrast_span < 0.20 ∧
B_contrast_first ≥ 0.20** — the span gate recovers closed-model confab detection to resampling parity
on a non-numeric domain too, where the first-token gate fails. ≥12 usable per group.

**Reading (pre-committed):**
- **SURVIVED:** the span `min_margin` gate generalizes beyond multi-digit arithmetic to character-
  level confabulation — it is about confab LOCALIZATION (the model is uncertain exactly where it
  guesses), not digit tokenization. Strengthens `styxx.span_confab` as a general closed-model gate.
- **REPORT_AS_LANDED:** the recovery is weaker or absent on reversal — either first-token also works
  here (no failure to recover) or span does not tie resampling — bounding the span gate to numeric /
  certain answer structures. Reported either way.

## Honest scope (pre-committed)

Single closed model gpt-4o-mini; string reversal only; one confirmatory run; feasibility-grade;
per-token entropy/margin from top-20 logprobs + residual bucket (truncated proxy); resampling N=10 at
T=1.0 (normalized-string Stability, no judge); ground truth = s[::-1], hashed pre-scoring.
CONFAB-long / CORRECT-short difficulty confound held fixed across detector TYPES by the contrasts.
Does NOT touch the correctness bound — signals DETECT, none CORRECTS.
