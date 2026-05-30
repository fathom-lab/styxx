# FINDING — the span-aggregate closed-model confab gate is MODEL-STRENGTH-INVARIANT: span ties N=10 resampling on gpt-3.5-turbo (AUC 1.000), gpt-4o-mini (0.991), AND frontier gpt-4o (1.000), B_contrast_span 0.000 on all. The pre-registered RECOVERY verdict is REPORT_AS_LANDED on gpt-3.5/gpt-4o — NOT because span failed but because the FIRST-token gate did not fail there (Bc_first +0.10 / +0.15 < 0.20): first-token reliability is model-dependent, span is the robust universal choice

**Run 2026-05-30. Two confirmatory runs, pre-registered in
`PREREG_detection_locus_gpt_xmodel_2026_05_30.md` (commit `a95ceeb`) BEFORE the runs. gpt-3.5-turbo
and gpt-4o via OpenAI API, the SAME 50 multiplication items / hash
`78a3e99c6c42c753b3619a9898378bc9602076bccac5cb31ff8eef189244b896` as the gpt-4o-mini span run
(all three models share the same arithmetic confab boundary).** Receipts:
`detection_locus_gpt_span_result_gpt-3_5-turbo.json`, `detection_locus_gpt_span_result_gpt-4o.json`.
(NOTE: those two receipts' `honest_scope` prose carries a stale copy-paste label "gpt-4o-mini"; the
authoritative `model` field is correct — gpt-3.5-turbo / gpt-4o. The runner f-string is fixed for
future runs.)

## Result: span model-strength-invariant; first-token model-dependent

| model | B1 resampling | first-token best (B_contrast_first) | span best (B_contrast_span) | RECOVERY verdict |
| --- | --- | --- | --- | --- |
| gpt-3.5-turbo (n 27/18) | 1.000 | 0.899 (**+0.101**) | **1.000** (0.000) | REPORT_AS_LANDED |
| gpt-4o-mini (reference) | 0.974 | 0.76 (+0.216) | 0.991 (0.000) | SURVIVED |
| gpt-4o (n 25/20) | 1.000 | 0.854 (**+0.146**) | **1.000** (0.000) | REPORT_AS_LANDED |

| span means | confab max-entropy | correct max-entropy | confab min-margin | correct min-margin |
| --- | --- | --- | --- | --- |
| gpt-3.5-turbo | 1.243 | 0.025 | **0.157** | 10.42 |
| gpt-4o | 1.562 | 0.021 | **0.197** | 14.08 |

## The claims that land

1. **The span gate is model-strength-invariant.** span-aggregate single-pass ties N=10 resampling
   at confab detection on a weak model (gpt-3.5-turbo, AUC 1.000), a small one (gpt-4o-mini, 0.991),
   and a FRONTIER one (gpt-4o, 1.000) — B_contrast_span 0.000 on all three, one forward pass each.
   The least-confident token sits at margin 0.16–0.20 in a confabulated product vs 10–14 in a correct
   one, on every model. `styxx.span_confab` works across the OpenAI model strength spectrum.
2. **The pre-registered RECOVERY verdict is REPORT_AS_LANDED on gpt-3.5/gpt-4o — and this is reported
   faithfully, not spun.** RECOVERY required the FIRST-token gate to FAIL (B_contrast_first ≥ 0.20).
   On these two models it does NOT: first-token reaches 0.90 / 0.85 (B_contrast +0.10 / +0.15), so
   there is nothing for span to "recover" — both detectors are strong. The verdict is REPORT_AS_LANDED
   because the precondition was unmet, NOT because the span gate failed (it is perfect, 1.000). The
   substantive result — span ties resampling everywhere — is strongly positive; I decline to relabel
   it SURVIVED when the pre-registered condition was not met.
3. **First-token reliability is MODEL-DEPENDENT.** It fails on gpt-4o-mini (0.76, Bc +0.22) and
   reversal (0.57, Bc +0.43) but is adequate on gpt-3.5-turbo (0.90) and gpt-4o (0.85). So the cheaper
   first-token gate (`single_pass_confab`) is sometimes sufficient on a closed model and sometimes
   not — there is no model-invariant guarantee. **`span_confab` is the robust universal closed-model
   choice; `single_pass_confab` is a cheaper fallback that works on some models.** This sharpens the
   product guidance: prefer span on closed models unless you have calibrated first-token on the
   specific deployment model.
4. **Correctness bound untouched.** Signals detect/abstain; never correct.

## Honest scope (pre-committed)

Three OpenAI models now (gpt-3.5-turbo, gpt-4o-mini, gpt-4o); multiplication only; one confirmatory
run each; feasibility-grade (powered); per-token entropy/margin from top-20 logprobs + residual
bucket (truncated proxy); resampling N=10 at T=1.0 (exact-integer, no judge); identical items/hash
across models. The span gate is model-strength-invariant WITHIN the OpenAI vendor on this domain; the
"first-token fails on closed models" premise is itself model-specific. Cross-VENDOR (Claude/Gemini
via a logprobs gateway) remains blocked on a non-OpenAI key. The receipts' honest_scope prose label
is a stale copy-paste artifact (model field authoritative); the runner f-string is fixed going
forward. Does NOT touch the correctness bound.

## The arc, in one line (updated)

The closed-model span-aggregate confab gate ties N=10 resampling across the OpenAI model strength
spectrum — gpt-3.5-turbo, gpt-4o-mini, and frontier gpt-4o, AUC 0.99–1.00, B_contrast_span ~0.00 —
and across numeric and non-numeric domains, so `styxx.span_confab` is a model-strength-invariant cheap
(one-pass) closed-model confab gate for structured answers; the cheaper first-token gate is only
sometimes sufficient (its failure is model-specific), single-token answers and cross-vendor remain
open, and every signal still moves confidence/abstention, never correctness.
