# PRE-REGISTRATION — completing the (architecture-family × derivation-domain) grid for single-pass confab legibility

**Written 2026-05-30, BEFORE the confirmatory runs.** Single-pass legibility is now established
cross-FAMILY on multi-hop logic (Qwen, Llama-3B, Gemma-2; B_contrast 0.000 / −0.183 / −0.064) and
cross-architecture on arithmetic (Qwen, Llama-1B). This run fills the remaining
(family × domain) cells that clear a competence gate, so each measurable domain is confirmed on the
same three pretraining lineages. Protocol UNCHANGED; bars inherited verbatim.

## Cells and competence gate (greedy one-shot, pre-confirmatory; recorded BEFORE confirmatory runs)

Greedy member counts on the existing committed item sets:

| cell | confab (hard) | correct (easy) | gate ≥12/12 | action |
| --- | --- | --- | --- | --- |
| Gemma-2-2B × arithmetic | 35/36 | 24/24 | **PASS** | confirmatory |
| Llama-3.2-3B × arithmetic | 34/36 | 24/24 | **PASS** | confirmatory |
| Gemma-2-2B × code-tracing | 34/36 | **9/20** | **FAIL** | report floor, no run |
| Llama-3.2-3B × code-tracing | 36/36 | **6/20** | **FAIL** | report floor, no run |

**Code-tracing floors on the non-Qwen families.** The `EASY_CODE` tier was tuned for Qwen-1.5B
(which clears 17/20); Gemma-2 (9/20) and Llama-3B (6/20) do not populate a ≥12 CORRECT class on it.
This is a powering limitation of the Qwen-tuned easy tier, reported as a bound (NOT forced with a
re-tuned tier, NOT a legibility verdict) — exactly as the Llama-1B × logic competence floor was
reported (`FINDING_detection_locus_logic_llama_2026_05_30.md`). Code-tracing therefore stays
Qwen-only here.

## Item set — arithmetic (the existing committed SPECS + EASY_SPECS)

`run_detection_locus.py --model {google/gemma-2-2b-it, meta-llama/Llama-3.2-3B-Instruct}`. The 36
hard arithmetic `run_competence_cliff.SPECS` + 24 `run_confabulation_specificity.EASY_SPECS`; ground
truth in-code (`_eval`) then hashed pre-scoring; question `"What is {expr}?"`; system "Answer with
only the final number, nothing else." Same items for both models.

**Arithmetic answer-key SHA-256 (60 items, pinned pre-scoring):**
`0eb5c90d72797150860537048060695a4a2f095be805ae3a8174c7ac255752b1`

- **CONFAB group** = hard SPECS the model answers WRONG greedily (`v1 != correct`, `v1 is not None`).
- **CORRECT group** = EASY_SPECS the model answers RIGHT greedily.

## Apparatus (pre-committed)

Gemma loads with `attn_implementation="eager"` (attention soft-capping) and reads soft-capped final
logits (absolute entropy/margin not cross-model comparable; within-model B_contrast is). The shared
`wb._render_chat` folds system→user only for templates that reject a system role; byte-identical for
Qwen/Llama. The code runner gains a `--model` arg (additive; Qwen default byte-identical).

## Signals / bars (identical to detection-locus)

1. Resampling instability = `1 − Stability` over N=10 @ T=1.0 (exact-integer, no judge).
2. Single-pass clean entropy at the first answer token.
3. Single-pass −margin at the first answer token.

- **B1 (core):** AUC(instability) `≥ 0.70`.
- **B2 / B3 (reported):** AUC(entropy), AUC(−margin).
- **B_contrast (core):** `AUC(instability) − max(AUC(entropy), AUC(−margin)) ≥ 0.20`.

**SURVIVED iff B1 ∧ B_contrast.** Powering: `≥ 12` usable items per group.

**Reading (pre-committed):** If B_contrast FAILS on both arithmetic cells (single-pass ≈/≥
resampling), arithmetic single-pass legibility is **cross-FAMILY** across Qwen, Llama, AND Gemma —
matching logic, so two of the three derivation domains are confirmed cross-family. If B_contrast
HOLDS on a cell, that family×arithmetic is a boundary. Reported either way.

## Honest scope (pre-committed)

Open models Gemma-2-2B-it and Llama-3.2-3B-Instruct; arithmetic only (the new cells); one
confirmatory run each; feasibility-grade; resampling N=10 at T=1.0; single-pass entropy/margin from
the clean logit-lens at the first answer token (soft-capped for Gemma); ground truth in-code then
hashed pre-scoring; exact-integer correctness. SAME difficulty confound as every prior
detection-locus run (CONFAB hard / CORRECT easy) — B1/B2/B3 are difficulty-driven-wrongness
detectors, B_contrast is the load-bearing, cross-family-comparable result. Does NOT touch the
correctness bound. Code-tracing on non-Qwen families is left as a reported competence-floor bound,
not a verdict.
