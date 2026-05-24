# Cross-model replication of logprob-validity (H1c)

**grounded-arc · 2026-05-24 · pre-registration `e580964`**

## Question

Bet-0b found logprob-validity ρ=0.73 (refusal, gpt-4o-mini). Is it model-general
or a quirk? Five models, same design, hash-before-score, one-shot each.

## Results

| model | pooled ρ | within-compliance ρ | within-refusal ρ | verdict (locked rule) |
|---|---|---|---|---|
| gpt-4o-mini | +0.734 | +0.575 | +0.437 | PASS |
| gpt-4o | +0.746 | +0.460 | +0.066 | CONFOUNDED |
| gpt-4.1-mini | +0.739 | +0.506 | +0.148 | CONFOUNDED |
| gpt-4.1 | +0.750 | +0.440 | −0.223 | CONFOUNDED |
| Qwen2.5-1.5B-Instruct (cross-family) | +0.578 | +0.294 | +0.163 | CONFOUNDED |

All pooled ρ p<0.0001.

## Verdict (per the locked pre-registration)

Replication rule: HOLDS if ≥3/4 additional models PASS (pooled ≥0.40 AND min
within-class ρ ≥0.20). **0/4 additional models PASS** → the strict cross-model
replication **does NOT hold.** The locked bar is not met; it is not moved.

## What is actually true (honest interpretation, not a re-scored verdict)

1. **The pooled signal replicates everywhere** (ρ 0.58–0.75, all p<0.0001) — but
   it is substantially **class-mediated** (refusals are confident *and* easy).
2. **The within-refusal class is uninformative everywhere.** `refuse_check` is
   near-perfect on refusals (mean |error| 0.01–0.04) → almost no error variance
   to predict → within-refusal ρ is noise. gpt-4o-mini's +0.437 there was a
   low-variance fluke, not signal; the cross-model test exposed it.
3. **The within-compliance class — where the instrument actually errs
   (over-flagging, mean |error| ~0.26) — carries a positive signal in ALL five
   models, including cross-family: 0.575 / 0.460 / 0.506 / 0.440 / 0.294.** This
   is the real, model-general result. But it **attenuates** out of the OpenAI
   family (0.29 on the small open model vs ~0.44–0.58 on OpenAI).

## Bottom line

The "universal reliability oracle" claim is **NOT earned.** The defensible finding
is narrower and honest: *model generation confidence is a model-general (including
cross-family) but attenuating predictor of one specific instrument failure mode —
refusal over-flagging on compliant responses.* The clean both-class signal does
not generalize; it was an artifact of the refusal class having no errors to
predict.

The cross-model + cross-family test did its job: it prevented shipping a
universal claim that would not have survived. Per-call validity from logprobs is
**real but bounded** — a genuine handle on over-flagging, not a universal trust
signal.

## Methodological lesson (for the next pre-registration)

Define the validity test on the **error-bearing subset** of each instrument, not
pooled across a near-perfect class — pooling let a low-variance class both inflate
the headline (via class-mediation) and fail the within-class control (via noise).
The map gains a precise boundary: logprob-validity holds where an instrument has
errors to find, and attenuates across model families.
