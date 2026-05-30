# FINDING — single-pass FIRST-TOKEN confabulation legibility FAILS on the closed model gpt-4o-mini (the first SURVIVED in the arc — i.e. the BOUNDARY): resampling detects its confabulation at AUC 0.974 but the first-token signal only 0.76, B_contrast +0.216 ≥ 0.20, because gpt-4o-mini confabulates CONFIDENTLY downstream of the first token (correct leading digits, wrong trailing). The `styxx.single_pass_confab` gate is white-box / weak-model only, now proven

**Run 2026-05-30. One confirmatory run, pre-registered in `PREREG_detection_locus_gpt_2026_05_30.md`
(commit `2486ccb`) BEFORE the confirmatory run. The detection-locus protocol on gpt-4o-mini via the
OpenAI API — single-pass entropy/margin from the first generated token's `top_logprobs=20`, N=10
temperature resamples for the Stability baseline; multiplication domain (EASY 2x2/3x2, HARD
4x3/4x4/5x4); hash `78a3e99c6c42c753b3619a9898378bc9602076bccac5cb31ff8eef189244b896` matched.**
Receipt: `detection_locus_gpt_result.json`.

## Why this run exists

The detection-locus arc found single-pass first-token entropy/margin tie N=10 resampling on small
WHITE-BOX open models (Qwen/Llama/Gemma, every B_contrast < 0.20), and `styxx.single_pass_confab`
(v7.7.14) ships that as a ~10x-cheaper gate. The product-critical question: does it survive on a
strong CLOSED model people deploy — gpt-4o-mini, read via API logprobs? This was the one cell where
SURVIVED (B_contrast ≥ 0.20, resampling wins) was the substantive, pre-stated outcome.

## Result: SURVIVED — single-pass FAILS, resampling has privileged access

| signal | AUC (confab vs correct), gpt-4o-mini | role |
| --- | --- | --- |
| **B1** resampling instability | **0.974** | confab IS detectable (by 10 resamples) |
| B2 single-pass first-token entropy | 0.753 | the shipped gate — WEAK |
| B3 single-pass first-token −margin | 0.759 | the shipped gate — WEAK |
| **B_contrast** = 0.974 − 0.759 | **+0.216** (≥0.20) | **SURVIVED → single-pass FAILS** |

| group means (n_conf=29, n_corr=18, powered) | instability | first-token entropy | first-token margin | modal-resample correct |
| --- | --- | --- | --- | --- |
| confab | **0.847** | 0.233 | 4.76 | 0.00 |
| correct | **0.031** | 0.087 | 10.14 | 0.94 |

## The claims that land

1. **First-token single-pass legibility FAILS on the closed model.** This is the FIRST SURVIVED in
   the detection-locus arc, and it is the "single-pass loses" direction: resampling separates
   gpt-4o-mini's confabulation from its correct answers nearly perfectly (AUC 0.974) while the
   first-token signal barely beats chance (0.75-0.76). B_contrast +0.216 clears the 0.20 bar —
   resampling has genuine privileged access here.
2. **The mechanism is confident confabulation DOWNSTREAM of the first token.** Both confab and
   correct have LOW first-token entropy (0.233 vs 0.087): gpt-4o-mini is confident at the leading
   token even when the full answer is wrong, because it estimates magnitude correctly and emits
   correct LEADING digits, then confabulates the TRAILING digits. The error lives where the
   first-token signal cannot see it. This is the exact opposite of the weak white-box models, which
   are uncertain from token one (so first-token worked there) — single-pass first-token legibility is
   a property of EARLY error, not of confabulation in general.
3. **The product boundary, drawn with a number.** `styxx.single_pass_confab` is a white-box /
   weak-model / EARLY-error detector. It does NOT transfer to strong closed models — exactly the
   scope shipped in its v7.7.14 docstring ("does not reach the closed-model confident-hallucination
   regime"), now empirically proven, not asserted. The deployable closed-model path remains
   RESAMPLING-based grounding (`grounded_honesty` / `audit_claim`, AUC 0.974 here), which pays 10x
   the forward passes precisely because it can see the downstream scatter the first token hides.
4. **Open: does a SPAN-AGGREGATE single-pass signal recover it?** If the error is downstream, a
   single-pass signal aggregated across ALL answer tokens (mean/max token entropy) — still one
   forward pass, no resampling — should see the trailing-digit uncertainty the first token misses.
   Tested in `run_detection_locus_gpt_span.py` / `FINDING_detection_locus_gpt_span_2026_05_30.md`.
5. **Correctness bound untouched.** modal_correct 0.00 (confab) vs 0.94 (correct): gpt-4o-mini is
   stably wrong on the hard products; resampling DETECTS, it does not CORRECT.

## Honest scope (pre-committed)

Single closed model gpt-4o-mini via OpenAI API; multiplication only; one confirmatory run;
feasibility-grade (29 confab + 18 correct, powered); resampling N=10 at T=1.0 (exact-integer
Stability, no judge); single-pass entropy/margin from the first generated token's top-20 logprobs
(TRUNCATED — entropy is a lower-bound proxy; OpenAI caps top_logprobs at 20, which if anything
understates the first-token entropy and so cannot manufacture the failure); ground truth in-code,
hashed pre-scoring. SAME CONFAB-hard / CORRECT-easy difficulty confound; B_contrast holds it fixed
across detector types and is load-bearing. Does NOT touch the correctness bound. One closed model,
one domain; whether the boundary is the digit-tokenization of arithmetic specifically or closed-model
confabulation generally is open (the span follow-up speaks to the mechanism).

## The arc, in one line (updated)

Single-pass FIRST-TOKEN confabulation legibility is cross-architecture, cross-family, domain-general
on small white-box models AND extends to factual recall — but it FAILS on the strong closed model
gpt-4o-mini (B_contrast +0.216, the arc's first SURVIVED), because confident confabulation there is
downstream of the first token where the signal is blind; so the shipped `single_pass_confab` gate is
white-box/weak-model/early-error only, resampling-based grounding remains the closed-model path, and
the open question is whether a span-aggregate single pass can recover what the first token loses.
