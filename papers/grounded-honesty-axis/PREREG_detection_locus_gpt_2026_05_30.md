# PRE-REGISTRATION — does single-pass confabulation legibility survive on a CLOSED model (gpt-4o-mini)? (the product-critical cell)

**Written 2026-05-30, BEFORE the confirmatory run.** The detection-locus arc found single-pass clean
first-token entropy/margin detect confabulation as well as N=10 resampling on small WHITE-BOX open
models (Qwen / Llama / Gemma, every B_contrast < 0.20), and `styxx.single_pass_confab` (v7.7.14)
ships that as a primitive. The product-critical question: **does the single-pass signal survive on a
strong CLOSED model people actually deploy — gpt-4o-mini — read via the OpenAI API's
`logprobs` / `top_logprobs=20` at the first answer token?**

## The hypothesis (pre-stated) — and why SURVIVED is the substantive outcome here

Unlike every prior cell (where REPORT_AS_LANDED = single-pass legible was the finding), in this cell a
**SURVIVED** result (B_contrast ≥ 0.20, resampling beats single-pass) is the substantive, expected
outcome: it would mean the closed model confabulates **CONFIDENTLY at the first token** — the
single-pass gate FAILS, and `styxx.single_pass_confab` is a white-box / weak-model / early-error
detector, NOT a closed-model hallucination detector. The mechanism: gpt-4o-mini estimates magnitude
well, so it emits CORRECT LEADING digits and confabulates TRAILING digits — the error is downstream
of the first token, where single-pass entropy is blind. A 24-item greedy feasibility probe is
consistent with this: gpt-4o-mini's WRONG products had first-token entropy ~0.0–0.07 (e.g.
`29159*3898` wrong at entropy 0.005, margin 7.5) — as confident as its correct answers. The probe
informed this expectation (apparatus + class-population check); the bars below are frozen/inherited
and the confirmatory is the powered test. If instead B_contrast < 0.20, single-pass generalizes to
the closed-model API surface. Reported either way.

## Item set (multiplication, difficulty = digit size; seeded, ground truth in-code)

`run_detection_locus_gpt.py`, seed `20260530`. HARD = 30 products of size 4x3 / 4x4 / 5x4
(gpt-4o-mini greedy-wrong → CONFAB) + EASY = 20 of size 2x2 / 3x2 (greedy-right → CORRECT). System
"You are a calculator. Output only the final integer product. No words, no commas." Ground truth =
the in-code product, hashed pre-scoring; exact-integer match (comma/space-stripped), no judge.

**Answer-key SHA-256 (50 items, pinned pre-scoring):**
`78a3e99c6c42c753b3619a9898378bc9602076bccac5cb31ff8eef189244b896`

- **CONFAB group** = HARD items gpt-4o-mini answers WRONG greedily (`v1 != correct`, `v1 is not None`).
- **CORRECT group** = EASY items gpt-4o-mini answers RIGHT greedily.

(Probe confab rates: 2x2 6/6, 3x2 5/6 correct; 4x3 1/6, 4x4 0/6, 5x4 0/6 correct — both classes
populate.)

## Signals (label confab=1, correct=0; HIGHER = more-likely-confab)

1. **Resampling instability** = `1 − Stability` over N=10 answers at T=1.0 (exact-integer, no judge).
2. **Single-pass clean entropy** of the first generated token's distribution — entropy over the
   top-20 logprobs + a single residual bucket for the unseen tail (OpenAI caps `top_logprobs` at 20,
   so this is a lower-bound proxy for the true entropy).
3. **Single-pass −margin** = `−(top1 − top2)` first-token logprob gap.

## Predictions / bars (identical to detection-locus)

- **B1 (core):** AUC(instability) `≥ 0.70`.
- **B2 / B3 (reported):** AUC(entropy), AUC(−margin).
- **B_contrast (core):** `AUC(instability) − max(AUC(entropy), AUC(−margin)) ≥ 0.20`.

**SURVIVED iff B1 ∧ B_contrast.** Powering: `≥ 12` usable per group.

**Closed-model reading (pre-committed):**
- **SURVIVED (B_contrast ≥ 0.20):** single-pass FAILS on gpt-4o-mini — resampling catches the
  confabulation (the full wrong answer scatters across samples) but the first-token signal does not
  (confident leading token). The closed-model error is downstream of the first token. The
  `single_pass_confab` gate is white-box / weak-model only; the closed-model confident-hallucination
  regime remains the open frontier. **This is the pre-registered product boundary.**
- **REPORT_AS_LANDED (B_contrast < 0.20):** single-pass ties resampling on gpt-4o-mini too — the gate
  generalizes to the closed-model API surface. Reported.

## Honest scope (pre-committed)

Single closed model gpt-4o-mini via OpenAI API; multiplication only; one confirmatory run;
feasibility-grade; resampling N=10 at T=1.0 (exact-integer Stability, no judge); single-pass
entropy/margin from the first generated token's top-20 logprobs (TRUNCATED — entropy is a lower-bound
proxy); ground truth in-code, hashed pre-scoring; exact-integer correctness. SAME CONFAB-hard /
CORRECT-easy difficulty confound; B_contrast holds it FIXED across detector types and is the
load-bearing, white-vs-closed-comparable result. Does NOT touch the correctness bound — every signal
DETECTS confabulation, none CORRECTS it.
