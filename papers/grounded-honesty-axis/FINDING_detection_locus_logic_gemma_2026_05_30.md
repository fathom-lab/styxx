# FINDING — single-pass confabulation legibility on MULTI-HOP LOGIC is CROSS-FAMILY across THREE pretraining lineages: on Gemma-2-2B (a genuinely different family, with logit soft-capping), clean first-token entropy separates confab from correct as well as / better than N=10 resampling (AUC 1.000 vs 0.936, B_contrast −0.064) — so the Qwen + Llama logic result extends to a third lineage, and "confident confabulation" is refuted on logic across Qwen, Llama, AND Gemma (REPORT_AS_LANDED)

**Run 2026-05-30. One confirmatory run, pre-registered in
`PREREG_detection_locus_logic_gemma_2026_05_30.md` (commit `b1c52cd`) BEFORE the confirmatory run.
The detection-locus protocol UNCHANGED, same 48 seeded items as the Qwen/Llama logic runs (hash
`97d816808a6874027637a35a7beeb8a7078aa483f30a01f3ea9e58f9e347e02c`, matched by the run), on
google/gemma-2-2b-it (white-box), with two pre-committed Gemma-2 accommodations: system folded into
the user turn (no system role) and `attn_implementation="eager"` (attention soft-capping).** Receipt:
`detection_locus_logic_result_gemma-2-2b-it.json`.

## Why this run exists

Single-pass logic legibility had two-family support — Qwen2.5-1.5B (B_contrast 0.000,
`[[FINDING_detection_locus_logic]]`) and Llama-3.2-3B (B_contrast −0.183,
`[[FINDING_detection_locus_logic_llama3b]]`). Both are dense decoder transformers in the common
lineage. This run tests a THIRD, genuinely different pretraining family — Gemma-2 (GQA, dual logit
soft-capping) — to ask whether the legibility is cross-FAMILY, not just cross-model. Gemma cleared
the competence gate (easy correct 18/24 with GENUINE varied answers `v1 {1:12, 2:12}`, hard confab
22/24 varied).

## Result: REPORT_AS_LANDED — B1 holds, B_contrast FAILS (single-pass ≥ resampling)

| signal | AUC (confab vs correct), Gemma-2-2B logic | bar | held |
| --- | --- | --- | --- |
| **B1** resampling instability | **0.936** | ≥0.70 | **HOLD** |
| B2 single-pass clean entropy | **1.000** | reported | — |
| B3 single-pass −margin | **1.000** | reported | — |
| **B_contrast** = 0.936 − 1.000 | **−0.064** | ≥0.20 | **FAIL** |

| group means (n_conf=22, n_corr=18, powered) | instability | clean entropy | logit margin | modal-resample correct |
| --- | --- | --- | --- | --- |
| confab | 0.248 | 1.156 | **0.61** | **0.09** |
| correct | 0.043 | 0.297 | **3.62** | **1.00** |

Soft-capping note (pre-committed): Gemma's entropy/margin are read from soft-capped final logits, so
their ABSOLUTE values are not comparable to Qwen/Llama. B_contrast is a within-model difference of
AUCs and is unaffected — it is −0.064 here, the same direction as Llama-3B (−0.183) and Qwen (0.000):
single-pass entropy detects logic confabulation at least as well as ten resamples.

## The claims that land

1. **Single-pass legibility on multi-hop logic is CROSS-FAMILY across three pretraining lineages.**
   B_contrast FAILS on Gemma-2 (−0.064) exactly as on Qwen (0.000) and Llama (−0.183). Three
   genuinely distinct families — Qwen, Llama, Gemma — all show single-pass clean entropy detecting
   logic confabulation as well as or better than resampling. The confident-confabulation refutation
   on logic is not a lineage artifact.
2. **"Confident confabulation" is refuted on logic across three families.** Gemma-2 logic confabs
   carry a collapsed leading-token margin (0.61 vs the correct group's 3.62) and elevated entropy
   (1.16 vs 0.30) in a single pass — even through logit soft-capping. The model is internally
   uncertain at the first answer token when it confabulates a multi-hop deduction, in every family
   tested.
3. **The boundary remains the instrument, not the architecture, family, or domain.** Six white-box
   confirmatory settings now agree (Qwen arith, Llama-1B arith, Qwen code, Qwen logic, Llama-3B
   logic, Gemma-2 logic): single-pass ≥ resampling. The open confident-when-wrong regime lives in the
   closed-model HALLUCINATION instrument (gpt-4o-mini), not in any open architecture, family, or
   derivation domain. `modal_correct` 0.09 (confab) vs 1.00 (correct) reconfirms the correctness
   bound: resampling and single-pass both DETECT, neither CORRECTS.

## Honest scope (pre-committed)

Single third-family open model Gemma-2-2B-it; multi-hop transitive-ordering logic only; one
confirmatory run; feasibility-grade (22 confab + 18 correct, powered); resampling N=10 at T=1.0;
Stability from exact distinct-integer counts (no judge); single-pass entropy/margin from the clean
full-vocab logit-lens at the first answer token (soft-capped — absolute values not cross-model
comparable; within-model B_contrast is); ground truth in-code then hashed pre-scoring; exact-integer
correctness; system folded into the user turn; eager attention. SAME difficulty confound as every
prior detection-locus run (CONFAB hard-deep / CORRECT easy-shallow) — so B1/B2/B3 are
difficulty-driven-wrongness detectors, not truth oracles; B_contrast holds the confound FIXED across
detector types (same items) and is the load-bearing, cross-family-comparable result. Does NOT touch
the correctness bound — every signal DETECTS confabulation, none CORRECTS it; the detector flags
abstain, never the answer.

## The arc, in one line (updated)

Single-pass confabulation legibility is cross-FAMILY on multi-hop logic across three distinct
pretraining lineages (Qwen, Llama, Gemma — B_contrast 0.000 / −0.183 / −0.064, single-pass entropy ≥
resampling), cross-architecture on arithmetic (Qwen + Llama), and domain-general across three
derivation domains on Qwen (arithmetic, code, logic) — clean first-token entropy/margin ≥ resampling
at AUC 0.91–1.00 everywhere it can be measured, even through Gemma's logit soft-capping; the one
floored cell (Llama-1B × logic) was incompetence, not a boundary; so "confident confabulation" is
family-AND-architecture-AND-domain-general FALSE on small-model derivation, the open regime is the
closed-model hallucination instrument, and every signal here moves confidence/abstention, never
correctness.
