# logprob-validity is refusal-specific — it fails on hallucination (confident confabulation)

**Cross-instrument negative · grounded-arc · H1d · 2026-05-24 · prereg `2c71aa0`, holdout `e30543c`**

## Question

Does "model generation confidence predicts instrument reliability" (shown for
refusal over-flagging) generalize to a *different* instrument — hallucination?

## Method

n=450 HaluEval-QA, closed-book gpt-4o-mini responses (logprobs), gold via gpt-4o
judge (validated 0.90 on known right/hallucinated pairs). Instrument = styxx
hallucination `check(use_nli=True)`, grounding the closed-book response against
the unseen knowledge. error = |risk − gold|. Hash-before-score, one-shot.

## Result — FAIL

| measure | ρ(validity_lp, −error) |
|---|---|
| **pooled** | **−0.184** (p=1.0) |
| within correct responses (n=233) | +0.476 |
| within hallucinated responses (n=217) | −0.027 |

Pooled ρ is **negative** — the opposite of the kill-gate. **H1d FAILS.
logprob-validity does NOT generalize to the hallucination instrument.**

## Why — the illuminating part

The signal inverts because of **confident confabulation.** For refusal, an
uncertain generation produced ambiguous text the detector mishandled — so low
confidence flagged unreliability, and the signal worked. Hallucination breaks
that assumption: **the model is frequently *confident* when it is wrong** (high
logprob on a fabricated answer). So:

- within *correct* responses, confidence still tracks reliability (ρ=+0.48 — the
  refusal pattern recurs);
- within *hallucinated* responses, confidence predicts nothing (ρ≈0) — and these
  high-confidence, high-error cases drag the **pooled** correlation negative.

Generation confidence cannot flag where a hallucination detector errs, precisely
because the failure mode you most want to catch (confident falsehood) is the one
where confidence is high. This is the exact case a "validity oracle" would need
to handle, and it doesn't.

## What this settles

The "model-internal confidence grounds cognometry **universally**" thesis is
**closed negative.** logprob-validity is **instrument-specific** — a real but
narrow signal for refusal over-flagging, where instrument error tracks generation
uncertainty. It fails the moment the model is confidently wrong.

The map is now sharp: **model-internal confidence predicts cognometric reliability
only where the instrument's errors are driven by generation uncertainty — not
where they're driven by confident error.** That boundary — refusal yes,
hallucination no, with the mechanism named — is the honest contribution.
