# PRE-REGISTRATION — is single-pass confabulation legibility DOMAIN-GENERAL, or arithmetic-specific?

**Written 2026-05-29, BEFORE the code-tracing run.** The detection-locus runs found that
confabulation is internally legible in a SINGLE forward pass on small-model ARITHMETIC — clean
first-token entropy / logit margin separate confab from correct nearly as well as N=10 resampling
on BOTH Qwen2.5-1.5B (AUC 0.92) and Llama-3.2-1B (AUC 1.00), refuting "confident confabulation"
on that domain. Every result so far is arithmetic. The standing open question (synthesis item 8):
is single-pass legibility a property of arithmetic, or does it hold on a structurally different
derivation domain whose difficulty comes from CONTROL FLOW rather than number size?

This run replicates the detection-locus protocol UNCHANGED on **code-output tracing** (the
already-validated second derivation domain from `run_code_tracing_grounding.py`: loops, branches,
nesting, stateful Collatz-style iteration; small per-step numbers), on Qwen2.5-1.5B-Instruct
(white-box).

**Does single-pass internal confidence (clean entropy / logit margin) separate confab from
correct on CODE-TRACING too, or is the legibility arithmetic-specific?**

## Item set (pre-committed — the 36 code-tracing SPECS, ground truth by EXECUTION)

The 36 deterministic, import-free snippets from `run_code_tracing_grounding.SPECS`
(code_ctrl / code_loop / code_nested / code_multistep). Ground truth = `run_snippet(gen_src(...))`
(executed `result`), computed in-code then SHA-256'd pre-scoring. Exact-integer correctness, no
judge. Prompt mirrors the code-tracing run: system "You are a Python interpreter. Output only the
single integer the program prints. No words.", user = the snippet + "What integer does this
program print? Give only the number."

- **CONFAB group** = items Qwen answers WRONG greedily (`v1 != correct`, `v1 is not None`).
- **CORRECT group** = items Qwen answers RIGHT greedily (`v1 == correct`).

Group membership uses the GREEDY one-shot label (difficulty sorts the tiers: ctrl→correct,
multistep/nested→confab). Powering reported, not assumed.

## AMENDMENT (2026-05-29, pilot-driven, BEFORE the confirmatory run — validity-motivated)

Pilot + full greedy pass on the 36 hard SPECS gave **n_conf=35, n_corr=1**: Qwen-1.5B confabulates
essentially every control-flow snippet, so the 36 SPECS contain NO usable CORRECT class and the
confab-vs-correct contrast is undefined. This is a competence-floor / powering failure, NOT a
verdict. To populate the negative class — exactly as the arithmetic detection-locus run used a
separate `EASY_SPECS` pool alongside the hard `SPECS` — I add an **`EASY_CODE`** pool of trivially
traceable, deterministic, import-free snippets (counting loops `total += 1`; constant-increment
`x += c`; doubling `x = x*2`), verified achievable on an 8-item greedy probe (Qwen 5/8). Group
assignment is UNCHANGED in spirit: **CORRECT group = `EASY_CODE` items Qwen answers right greedily;
CONFAB group = the 36 hard SPECS items Qwen answers wrong greedily.** Both groups are genuine
code-tracing (variable state + control flow), preserving the domain claim; the difficulty gap is
the same acknowledged confound as the arithmetic run, and B_contrast (confound held fixed across
detector types) remains the load-bearing result. This amendment is recorded BEFORE the confirmatory
scoring run and is motivated by feasibility, not by any observed detector value.

## Signals (label confab=1, correct=0; HIGHER = more-likely-confab) — identical to detection-locus

1. **Resampling instability** = `1 − Stability`, Stability = `1 − (n_distinct − 1)/(N − 1)` over
   N=10 answers at T=1.0, exact-integer parse, None its own class (no judge).
2. **Single-pass clean entropy** at the first answer-token position (logit-lens, no intervention).
3. **Single-pass −margin** = `−(top1 − top2)` clean logit gap at the first answer token.

## Predictions / bars (identical to detection-locus)

- **B1 (core):** AUC(instability) `≥ 0.70`.
- **B2 / B3 (reported):** AUC(clean entropy), AUC(−margin).
- **B_contrast (core):** `AUC(instability) − max(AUC(entropy), AUC(−margin)) ≥ 0.20`.

**SURVIVED iff B1 ∧ B_contrast.** Powering: `≥ 12` usable items per group.

**Cross-domain reading (pre-committed):**
- If B_contrast FAILS on code too (single-pass entropy/margin separate confab from correct, AUC
  comparable to resampling), single-pass legibility is **DOMAIN-GENERAL** — not an arithmetic
  artifact; the confident-confabulation refutation extends across derivation domains.
- If B_contrast HOLDS on code (single-pass near chance, resampling dominates by ≥0.20), legibility
  is **arithmetic-specific**, and code-tracing is where confabulation IS confident in one pass —
  locating a domain boundary. Reported either way.

## Honest scope (pre-committed)

Single open model Qwen2.5-1.5B-Instruct; code-tracing only (one new domain); one confirmatory
run; feasibility-grade; resampling N=10 at T=1.0; Stability from exact distinct-integer counts
(no judge); single-pass entropy/margin from the clean full-vocab logit-lens at the first answer
token; ground truth by execution then hashed pre-scoring; exact-integer correctness. SAME
difficulty confound as the arithmetic runs (CORRECT easy-control / CONFAB hard-multistep) — so
B1/B2/B3 are difficulty-driven-wrongness detectors, not truth oracles; B_contrast holds the
confound FIXED across detector types (same items) and is the load-bearing, cross-domain-comparable
result. Code answers are short single integers (bare prompt), so the first-token signal is well
defined (NOT the long ANSWER-marker traces used for the path-diverse grounding backend). Does NOT
touch the correctness bound — every signal DETECTS confabulation, none CORRECTS it; the detector
flags abstain, never the answer.
