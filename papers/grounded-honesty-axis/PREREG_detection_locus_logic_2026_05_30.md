# PRE-REGISTRATION — is single-pass confabulation legibility DOMAIN-GENERAL across a THIRD derivation domain (multi-hop logic)?

**Written 2026-05-30, BEFORE the confirmatory logic run.** The detection-locus runs found that
confabulation is internally legible in a SINGLE forward pass on small-model **arithmetic** (clean
first-token entropy / logit margin separate confab from correct nearly as well as N=10 resampling,
AUC 0.92 Qwen / 1.00 Llama) and again on **code-output tracing** (control-flow difficulty, Qwen
single-pass clean entropy 0.906 tracking N=10 resampling 0.908, B_contrast +0.002 → "confident
confabulation" refuted, REPORT_AS_LANDED, commit `eb65cf4`). Two derivation domains down — one
where difficulty is number size, one where it is control flow. The standing open question
(synthesis item 8): does single-pass legibility hold on a structurally different derivation domain
whose difficulty comes from **logical inference depth** rather than number size or control flow?

This run replicates the detection-locus protocol **UNCHANGED** on **multi-hop transitive-ordering
logic**, on Qwen2.5-1.5B-Instruct (white-box).

**Does single-pass internal confidence (clean entropy / logit margin) separate confab from correct
on multi-hop deductive logic too, or is the legibility specific to the prior two domains?**

## Item set (pre-committed — seeded, ground truth in-code)

`run_detection_locus_logic.py`, seed `20260530`. Each item: K people in a secret strict
oldest→youngest order; the K−1 consecutive `"X is older than Y"` facts presented in SCRAMBLED
order; asked `"Among these K people, how many are older than {target}?"` for a non-oldest target.
Ground truth = the target's rank from the oldest (0-based index = count strictly older), computed
in-code then SHA-256'd pre-scoring. Pure transitive deduction — exact-integer answer, no
arithmetic and no control flow. System prompt: "You are solving a logic puzzle about an age
ordering. Use only the given facts. Answer with only the final number, nothing else."

- **HARD pool (24):** long scrambled chains, K ∈ {6,7,8,9} → 5–8 inference hops.
- **EASY pool (24):** short chains, K ∈ {2,3} → 1–2 hops.
- **CONFAB group** = HARD items Qwen answers WRONG greedily (`v1 != correct`, `v1 is not None`).
- **CORRECT group** = EASY items Qwen answers RIGHT greedily (`v1 == correct`).

Both groups are genuine transitive-ordering logic (same task, scrambled-fact deduction); the
HARD/EASY gap is **inference depth**, the same acknowledged CONFAB-hard / CORRECT-easy difficulty
confound as the arithmetic (number size) and code (control flow) runs. B_contrast holds that
confound FIXED across detector types and is the load-bearing, cross-domain-comparable result.

**Answer-key SHA-256 (48 items, pinned pre-scoring):**
`97d816808a6874027637a35a7beeb8a7078aa483f30a01f3ea9e58f9e347e02c`

## Pilot disclosure (apparatus validation, BEFORE the confirmatory run)

A `--n 6` pilot (12 items) was run for apparatus validation only — to confirm the protocol executes
end-to-end and that BOTH classes populate (Qwen confabulates the long scrambled chains and answers
the short ones correctly). It confirmed 5 confab / 6 correct usable at n=6, so the full 24+24 clears
the ≥12/≥12 power bar. The bars, signals, and SURVIVED condition below are **inherited verbatim**
from the arithmetic and code detection-locus preregs and are NOT tunable from the pilot; the
confirmatory run is the full seeded set (same seed, hash above). No detector value motivated any
design choice.

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
- If B_contrast FAILS on logic too (single-pass entropy/margin separate confab from correct, AUC
  comparable to resampling), single-pass legibility is **DERIVATION-DOMAIN-GENERAL across three
  domains** — arithmetic, code-output tracing, AND multi-hop deductive logic; the
  confident-confabulation refutation is not tied to any one kind of difficulty.
- If B_contrast HOLDS on logic (single-pass near chance, resampling dominates by ≥0.20), legibility
  is NOT universal, and multi-hop logic is where confabulation IS confident in one pass — locating
  a domain boundary. Reported either way.

## Honest scope (pre-committed)

Single open model Qwen2.5-1.5B-Instruct; multi-hop transitive-ordering logic only (one new domain);
one confirmatory run; feasibility-grade; resampling N=10 at T=1.0; Stability from exact
distinct-integer counts (no judge); single-pass entropy/margin from the clean full-vocab logit-lens
at the first answer token; ground truth in-code (target rank in a seeded secret order) then hashed
pre-scoring; exact-integer correctness. SAME difficulty confound as the arithmetic and code runs
(CORRECT easy-shallow / CONFAB hard-deep) — so B1/B2/B3 are difficulty-driven-wrongness detectors,
not truth oracles; B_contrast holds the confound FIXED across detector types (same items) and is
the load-bearing, cross-domain-comparable result. Answers are short single integers (bare prompt),
so the first-token signal is well defined. Does NOT touch the correctness bound — every signal
DETECTS confabulation, none CORRECTS it; the detector flags abstain, never the answer.
