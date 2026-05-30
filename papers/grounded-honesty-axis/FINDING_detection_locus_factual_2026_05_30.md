# FINDING — single-pass confabulation legibility EXTENDS from derivation to FACTUAL RECALL (it is not derivation-specific), but the detection regime is markedly WEAKER on facts: on Llama-3.2-1B birth-year recall, single-pass margin separates confab from correct (AUC 0.768) at least as well as N=10 resampling (0.756), B_contrast −0.013 — so factual confabulation is NOT single-pass-confident; the cheap gate generalizes to knowledge errors, at attenuated power (AUC ~0.73 vs ~0.95 on derivation) (REPORT_AS_LANDED)

**Run 2026-05-30. One confirmatory run, pre-registered in
`PREREG_detection_locus_factual_2026_05_30.md` (commit `3c2a8fc`) BEFORE the confirmatory run. The
detection-locus protocol UNCHANGED on FACTUAL RECALL (birth years of historical figures — pure
lookup, integer answer, no derivation), on Llama-3.2-1B-Instruct, hash
`8d54795e20e8d906d4b78546518c6ac659c8528261aec66d8d98c06c87ac5d45` matched.** Receipt:
`detection_locus_factual_result_Llama-3_2-1B-Instruct.json`.

## Why this run exists

The detection-locus arc showed single-pass clean entropy/margin detect DERIVATION confabulation
(arithmetic, code, logic) as well as resampling, across three families — every cell B_contrast
< 0.20 (`SYNTHESIS_detection_locus_2026_05_30.md`). Derivation confab is a REASONING error. The
product-relevant question left open: does the cheap single-pass gate also catch FACTUAL-RECALL
confabulation — a KNOWLEDGE error — or is factual confab single-pass-CONFIDENT (the way closed-model
hallucination is)? This was pre-registered as the first cell where B_contrast could genuinely exceed
0.20.

## Elicitation finding (the probes): small-model confab is a DERIVATION phenomenon

Greedy probes establish that **Qwen2.5-1.5B cannot populate a factual-confab class** — it recalls
canonical facts almost completely: atomic numbers (Z 1–118) 39/42, famous birth years 28/30, country
independence years 28/30, with ~0 refusals. It is not confabulating and abstaining; it simply
*knows*. Factual confabulation had to be elicited from a knowledge-gappier model: **Llama-3.2-1B**
confabulates obscure birth years CONFIDENTLY (24/28 greedy-wrong, 0 refusals). So confident factual
confabulation is not a generic small-model behavior on canonical facts — it requires a genuine
knowledge gap. (Llama-1B floored on *logic* by incompetence; on birth years it is competent enough
to anchor a correct class while confabulating the obscure tail.)

## Result: REPORT_AS_LANDED — single-pass ≈ resampling on facts too, but both weak

| signal | AUC (confab vs correct), Llama-1B birth years | bar | held |
| --- | --- | --- | --- |
| **B1** resampling instability | **0.756** | ≥0.70 | **HOLD (barely)** |
| B2 single-pass clean entropy | **0.700** | reported | — |
| B3 single-pass −margin | **0.768** | reported | — |
| **B_contrast** = 0.756 − 0.768 | **−0.013** | ≥0.20 | **FAIL** |

| group means (n_conf=24, n_corr=15, powered) | instability | clean entropy | logit margin | modal-resample correct |
| --- | --- | --- | --- | --- |
| confab (obscure) | 0.519 | 1.619 | 0.89 | 0.04 |
| correct (famous) | 0.289 | 1.129 | 1.93 | 0.93 |

## The claims that land

1. **Single-pass legibility is NOT derivation-specific — it extends to factual recall.** B_contrast
   is −0.013: single-pass margin (0.768) separates factual confab from correct at least as well as
   ten resamples (0.756). Factual confabulation is **not** single-pass-confident; the wrong factual
   commitment is internally uncertain at the first answer token, just as derivation confab is. The
   closed-model "confident hallucination" pattern does NOT appear here. The cheap one-pass gate
   generalizes from reasoning errors to knowledge errors.
2. **But the detection regime is markedly WEAKER on facts.** AUC 0.70–0.77 here vs 0.91–1.00 on
   every derivation cell. The means show why: Llama-1B is shaky *even when right* about famous birth
   years (correct-class instability 0.289, entropy 1.13, modal_correct 0.93 — not the ~0.00 / ~1.00
   of the derivation correct classes). Its factual confidence is low across the board, compressing
   the confab-vs-correct gap. The *relationship* (single-pass ≈ resampling) holds; the *absolute
   power* attenuates. A factual confab detector built on this signal is usable but modest, not the
   near-perfect separator derivation gives.
3. **Correctness bound untouched.** modal_correct 0.04 (confab) vs 0.93 (correct): resampling and
   single-pass both DETECT; neither CORRECTS. The detector flags abstain, never the answer.

## Product implication (stated plainly)

The single-pass first-token signal is a **cheap (one forward pass vs ten), general confab/abstain
gate** that works on BOTH reasoning errors (AUC ~0.95) and factual/knowledge errors (AUC ~0.73) —
single-pass ≈ resampling in both. It is a real, deployable primitive with an honest power gradient:
strong on derivation, modest on facts. It is NOT a near-perfect hallucination oracle on factual
recall, and it cannot run at all where the model simply knows the facts (Qwen) or where confident
hallucination lives in the closed-model regime the arc has repeatedly located as the open frontier.
The honest one-line: a cheap reasoning-error detector that *also* gives a modest factual-hallucination
signal — not a revolutionary universal lie-detector (that remains falsified).

## Honest scope (pre-committed)

Single white-box model Llama-3.2-1B-Instruct; factual recall (birth years) only; one confirmatory
run; feasibility-grade (24 confab + 15 correct, powered, but a WEAK regime — B1 barely clears 0.70);
resampling N=10 at T=1.0; single-pass entropy/margin from the clean logit-lens at the first answer
token; ground truth = canonical birth years, hashed pre-scoring; exact-integer correctness.
CONFAB=obscure / CORRECT=famous: difficulty is FAMILIARITY (a knowledge gradient); B1/B2/B3 are
difficulty-driven-wrongness detectors, B_contrast holds the confound FIXED across detector types and
is the load-bearing, derivation-vs-recall-comparable result. The correct class is itself uncertain on
this model, which both makes the regime weak and is the honest reason the AUCs are low. Does NOT touch
the correctness bound. One model, one factual domain; a sharper correct class (a model confident in
its true facts) could raise the absolute AUC without changing the B_contrast relationship.

## The arc, in one line (updated)

Single-pass confabulation legibility is cross-architecture, cross-family, domain-general on
derivation (B_contrast ≤ +0.056, AUC 0.91–1.00) AND now extends to FACTUAL recall (B_contrast −0.013)
— single-pass ≈ resampling everywhere, so the cheap one-pass gate is a general confab detector, not
a derivation artifact — but its power attenuates sharply on facts (AUC ~0.73, because the model's
factual confidence is low even when correct); confident factual confabulation could not even be
elicited from a knowledge-complete model (Qwen recalls canonical facts), so the confident-hallucination
regime remains the closed-model frontier; and every signal still moves confidence/abstention, never
correctness.
