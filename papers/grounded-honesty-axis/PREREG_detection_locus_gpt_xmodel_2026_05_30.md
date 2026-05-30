# PRE-REGISTRATION — does the closed-model span confab gate hold across MODEL STRENGTHS (gpt-3.5-turbo, gpt-4o)?

**Written 2026-05-30, BEFORE the confirmatory runs.** The span gate recovered closed-model confab
detection to resampling parity on gpt-4o-mini multiplication (min-margin AUC 0.991,
`FINDING_detection_locus_gpt_span_2026_05_30.md`) and generalized to a non-numeric domain (reversal,
max-entropy AUC 0.993). Open: does it hold across MODEL STRENGTHS — a weaker model (gpt-3.5-turbo)
and a FRONTIER model (gpt-4o)? Cross-VENDOR is blocked (no non-OpenAI logprobs key); this is the
accessible cross-model test.

A greedy probe confirms ALL THREE models share the same arithmetic confab boundary: 2x2 6/6 correct,
4x3 and up 0-1/6 correct, on gpt-3.5-turbo AND gpt-4o (even the frontier model cannot multiply
4+ digit numbers exactly in one pass without chain-of-thought). So the **SAME 50 items / SAME hash**
as the gpt-4o-mini run populate both classes on all three models — a direct cross-model comparison.

## Item set (identical to the gpt-4o-mini span run)

`run_detection_locus_gpt_span.py --model {gpt-3.5-turbo, gpt-4o}`, the same 50 multiplication items
(30 HARD 4x3/4x4/5x4 + 20 EASY 2x2/3x2), hash
`78a3e99c6c42c753b3619a9898378bc9602076bccac5cb31ff8eef189244b896`. CONFAB = HARD greedy-wrong,
CORRECT = EASY greedy-right, per model.

## Signals / bars (identical to the span run)

Per answer token entropy/margin → first-token vs span-aggregate detectors + N=10 resampling.
**RECOVERY SURVIVED iff** (per model) B1 ≥ 0.70 ∧ best_span ≥ 0.70 ∧ B_contrast_span < 0.20 ∧
B_contrast_first ≥ 0.20 — the span gate recovers to resampling parity where the first-token gate
fails. ≥12 usable per group.

**Cross-model reading (pre-committed):**
- **Both SURVIVED:** the span gate is model-strength-invariant — it works on a weak model, gpt-4o-mini,
  AND a frontier model (gpt-4o). The cheap closed-model confab gate holds across the models people
  deploy (within the OpenAI vendor). Strongest possible product evidence short of cross-vendor.
- **A cell REPORT_AS_LANDED:** the gate weakens on that model strength — e.g. gpt-4o confabulates so
  confidently across the whole span that even the span aggregate fails (only resampling reveals it),
  bounding the gate by model strength. Reported per model.

## Honest scope (pre-committed)

Two additional closed models (gpt-3.5-turbo, gpt-4o) via OpenAI API; multiplication only; one
confirmatory run each; feasibility-grade; per-token entropy/margin from top-20 logprobs + residual
bucket (truncated proxy); resampling N=10 at T=1.0 (exact-integer, no judge); same items/hash as the
gpt-4o-mini run. Difficulty confound held fixed across detector TYPES by the contrasts. Still
single-vendor (OpenAI) — cross-VENDOR remains blocked on a non-OpenAI logprobs-exposing key. Does NOT
touch the correctness bound.
