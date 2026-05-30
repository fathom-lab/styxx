# PRE-REGISTRATION — does single-pass confabulation legibility on MULTI-HOP LOGIC replicate on a SECOND architecture (Llama-3.2-3B)?

**Written 2026-05-30, BEFORE the confirmatory Llama-3.2-3B run.** The Qwen2.5-1.5B logic
detection-locus run (`FINDING_detection_locus_logic_2026_05_30.md`, commit `2782a8b`) found
single-pass clean entropy separates confab from correct exactly as well as N=10 resampling on
multi-hop transitive-ordering logic (B_contrast 0.000, REPORT_AS_LANDED) — extending
single-pass legibility to a third derivation domain on Qwen. The cross-architecture symmetry
question (arithmetic was confirmed on BOTH Qwen and Llama-3.2-1B): does the LOGIC finding replicate
on a second architecture? A first attempt on **Llama-3.2-1B FAILED at the competence floor**
(`FINDING_detection_locus_logic_llama_2026_05_30.md`, commit `6ffc4be`): the 1B model answers "2"
reflexively to all 24 easy ordering questions, so it has no genuine CORRECT class. This run uses the
more capable same-family **meta-llama/Llama-3.2-3B-Instruct (28 decoder layers)**, which clears the
competence gate, to give the logic domain the same Qwen+Llama two-family support arithmetic has.

**Does single-pass internal confidence (clean entropy / logit margin) separate confab from correct
on multi-hop logic for Llama-3.2-3B too, or is the Qwen legibility architecture-specific here?**

## Item set (same seeded 48 items as the Qwen logic run — identical construction)

`run_detection_locus_logic.py --model meta-llama/Llama-3.2-3B-Instruct`, seed `20260530`. K people
in a secret oldest→youngest order; the K−1 consecutive scrambled "X is older than Y" facts; "how
many are older than {target}?". Ground truth = target rank, in-code, hashed pre-scoring. HARD pool
24 (K∈{6,7,8,9}, 5–8 hops); EASY pool 24 (K∈{2,3}, 1–2 hops).

**Answer-key SHA-256 (48 items, identical to the Qwen run, pinned pre-scoring):**
`97d816808a6874027637a35a7beeb8a7078aa483f30a01f3ea9e58f9e347e02c`

- **CONFAB group** = HARD items Llama-3.2-3B answers WRONG greedily (`v1 != correct`, `v1 is not None`).
- **CORRECT group** = EASY items Llama-3.2-3B answers RIGHT greedily (`v1 == correct`).

## Competence gate (pre-confirmatory feasibility, BEFORE the confirmatory run)

Greedy one-shot on all 48 items, Llama-3.2-3B: **EASY correct-members 18/24, HARD confab-members
21/24** — both clear the ≥12 power bar. Crucially the easy answers are GENUINE, not reflexive: EASY
v1 distribution `{1: 12, 2: 12}` (contrast the 1B's `{2: 24}`), correct-by-true-answer `{1: 12,
2: 6}` — the model returns 1 for single-hop and 2 for two-hop by reasoning, not a constant. So the
CORRECT class is competent, not reflex-matched; the confab-vs-correct contrast is well-posed. This
gate is recorded BEFORE the confirmatory scoring run; the bars, signals, and SURVIVED condition
below are inherited VERBATIM from the arithmetic, code, and Qwen-logic preregs and are NOT tunable
from the gate (which touched only greedy labels, never the detector signals).

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

**Cross-architecture reading (pre-committed):**
- If B_contrast FAILS on Llama-3B too (single-pass entropy/margin separate confab from correct, AUC
  comparable to resampling), single-pass legibility on LOGIC is **cross-architecture** (Qwen +
  Llama), matching the arithmetic result — the confident-confabulation refutation is both
  domain-general AND cross-architecture, and the 1B failure is confirmed as a pure competence floor.
- If B_contrast HOLDS on Llama-3B (single-pass near chance, resampling dominates by ≥0.20), then the
  Qwen logic legibility is architecture-specific, and Llama-3B logic is where confabulation IS
  confident in one pass — locating an architecture boundary. Reported either way.

## Honest scope (pre-committed)

Single second open model Llama-3.2-3B-Instruct; multi-hop transitive-ordering logic only; one
confirmatory run; feasibility-grade; resampling N=10 at T=1.0; Stability from exact distinct-integer
counts (no judge); single-pass entropy/margin from the clean full-vocab logit-lens at the first
answer token; ground truth in-code then hashed pre-scoring; exact-integer correctness. SAME
difficulty confound as every prior detection-locus run (CONFAB hard-deep / CORRECT easy-shallow) —
so B1/B2/B3 are difficulty-driven-wrongness detectors, not truth oracles; B_contrast holds the
confound FIXED across detector types (same items) and is the load-bearing, cross-architecture-
comparable result. Does NOT touch the correctness bound — every signal DETECTS confabulation, none
CORRECTS it; the detector flags abstain, never the answer. Two model sizes within the Llama family
plus Qwen; a third architecture family (e.g. Gemma) remains untested on logic here.
