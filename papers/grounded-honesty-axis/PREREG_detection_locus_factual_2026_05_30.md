# PRE-REGISTRATION — does single-pass confabulation legibility extend from DERIVATION to FACTUAL RECALL? (the product-relevant boundary)

**Written 2026-05-30, BEFORE the confirmatory run.** The detection-locus arc established that
single-pass clean first-token entropy/margin detect DERIVATION confabulation (arithmetic, code,
logic) as well as or better than N=10 resampling, across three families — every cell B_contrast
< 0.20 (`SYNTHESIS_detection_locus_2026_05_30.md`). Derivation confab is a REASONING error. This run
tests the most product-relevant question the arc has not answered: does the same single-pass signal
catch FACTUAL-RECALL confabulation — a KNOWLEDGE error — or is factual confabulation
single-pass-CONFIDENT, the way closed-model hallucination is?

**This is the first cell where B_contrast could genuinely exceed the 0.20 bar, and I do not know the
direction in advance.** If the model confidently misremembers a fact in one forward pass (sharp
wrong-token distribution, low entropy) while the wrong answer still scatters under resampling, then
single-pass entropy/margin FAIL to flag it while resampling catches it → **B_contrast ≥ 0.20 →
single-pass legibility is DERIVATION-SPECIFIC**, the single-pass gate is a reasoning-error detector
and NOT a knowledge-hallucination detector, and this locates the boundary at derivation-vs-recall.
If B_contrast < 0.20, the legibility is general to white-box generation, not tied to reasoning.

## Model choice (pre-committed, with disclosure)

Qwen2.5-1.5B-Instruct has **near-complete canonical-fact recall** — greedy probes: atomic numbers
(Z 1–118) 39/42 right, famous birth years 28/30, country independence years 28/30, with ~0 refusals.
It does not confabulate canonical facts, so it cannot populate a factual-confab class (an empirical
finding in its own right: small-model confabulation is a derivation phenomenon, not a recall one).
**Llama-3.2-1B-Instruct** has genuine knowledge gaps and confabulates birth years CONFIDENTLY
(greedy: 24/28 obscure figures wrong, 0 refusals), so it is the white-box model that exhibits the
phenomenon under test. (Llama-1B floored on *logic* by incompetence; here it is competent enough to
have both classes — it knows iconic birth years and confabulates obscure ones.)

## Item set (birth years, seeded-free, ground truth canonical)

`run_detection_locus_factual.py --model meta-llama/Llama-3.2-1B-Instruct`. 37 FAMOUS figures (model
greedy-right → CORRECT class) + 28 OBSCURE second-tier scientists/mathematicians (model greedy-wrong
→ CONFAB class). Question: `"In what year was {figure} born?"`; system "Answer with only the year (a
number), nothing else." Ground truth = canonical birth years, hashed pre-scoring; exact-integer
match, no judge.

**Answer-key SHA-256 (65 items, pinned pre-scoring):**
`8d54795e20e8d906d4b78546518c6ac659c8528261aec66d8d98c06c87ac5d45`

- **CONFAB group** = OBSCURE items greedy-wrong (`v1 != correct`, `v1 is not None`).
- **CORRECT group** = FAMOUS items greedy-right (`v1 == correct`).

## Competence gate (pre-confirmatory, BEFORE the confirmatory run)

Greedy on all 65: confab members 24/28, correct members 15/37 — both clear the ≥12 bar. The
familiarity gradient (famous-known vs obscure-confabulated) is the CONFAB-hard / CORRECT-easy
confound, here on KNOWLEDGE rather than derivation depth.

## Signals / bars (identical to detection-locus)

1. Resampling instability = `1 − Stability` over N=10 @ T=1.0 (exact-integer, no judge).
2. Single-pass clean entropy at the first answer token.
3. Single-pass −margin at the first answer token.

- **B1 (core):** AUC(instability) `≥ 0.70`.
- **B2 / B3 (reported):** AUC(entropy), AUC(−margin).
- **B_contrast (core):** `AUC(instability) − max(AUC(entropy), AUC(−margin)) ≥ 0.20`.

**SURVIVED iff B1 ∧ B_contrast.** Powering: `≥ 12` usable per group. NOTE: unlike the derivation
cells (where REPORT_AS_LANDED = single-pass legible was the finding), here a **SURVIVED** result
(B_contrast ≥ 0.20) would be the substantive, expected-possible outcome: it would mean factual
confab is single-pass-confident and the gate does NOT generalize to knowledge errors. Both
directions are pre-committed and reported.

## Honest scope (pre-committed)

Single white-box model Llama-3.2-1B-Instruct; factual recall (birth years) only; one confirmatory
run; feasibility-grade; resampling N=10 at T=1.0; single-pass entropy/margin from the clean
logit-lens at the first answer token; ground truth = canonical birth years, hashed pre-scoring;
exact-integer correctness. CONFAB=obscure / CORRECT=famous: difficulty is FAMILIARITY (a knowledge
gradient); B1/B2/B3 are difficulty-driven-wrongness detectors, B_contrast holds the confound FIXED
across detector types and is the load-bearing, derivation-vs-recall-comparable result. Birth-year
recall is non-systematic (unlike atomic numbers) so genuinely confabulable. Does NOT touch the
correctness bound — every signal DETECTS confabulation, none CORRECTS it; the detector flags abstain.
