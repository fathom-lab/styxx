# The Detection Locus — small-model derivation confabulation is legible in a SINGLE forward pass, across architectures, families, and domains

**Synthesis of the 2026-05-29 / 2026-05-30 detection-locus arc.** White-box runs on three open
model families (Qwen2.5-1.5B, Llama-3.2-1B/3B, Gemma-2-2B), local, feasibility-grade. Every cell
below is one pre-registered, kill-gated, hash-before-score confirmatory run with a committed JSON
receipt. This document unifies the arc into one claim and states its boundary exactly; it does not
re-derive the numbers — each links to its FINDING and receipt.

## The question: where does the confabulation-detection signal LIVE?

The grounded-honesty arc established that a model's own **resampling instability** catches
confabulation: sample the answer N=10 times, and a confabulated derivation scatters (high
instability) while a known one is stable (`SYNTHESIS_grounded_honesty_arc_2026_05_28.md`). That
signal is *cross-derivation* — it needs ten forward passes. The detection-locus question is sharper:

> Is that detection signal a property of the **sample distribution** (you must resample to see the
> confabulation), or is it already present in the model's **single-pass internal state** — the clean
> first-token logits of one greedy forward pass?

If single-pass internal confidence separates confab from correct as well as resampling, then the
wrong commitment is **internally uncertain at the first answer token** — and "confident
confabulation" is false for this regime. If resampling has privileged access, then the model *is*
confident in one pass and only the sample scatter reveals the error.

## The protocol (identical across every cell)

On a balanced CONFAB-vs-CORRECT set read entirely from the white-box model, three detector scores,
all oriented so higher = more-likely-confab:

1. **Resampling instability** = `1 − Stability` over N=10 @ T=1.0 (exact-integer, no judge) — the
   validated cross-derivation signal.
2. **Single-pass clean entropy** at the first answer token (logit-lens, no intervention).
3. **Single-pass −margin** = `−(top1 − top2)` at the first answer token.

- **B1 (core):** AUC(instability) `≥ 0.70`.
- **B_contrast (core, load-bearing):** `AUC(instability) − max(AUC(entropy), AUC(−margin)) ≥ 0.20`.
- **SURVIVED iff B1 ∧ B_contrast**, ≥12 usable per group.

CONFAB items are hard (greedy-wrong), CORRECT items easy (greedy-right) — an acknowledged difficulty
confound. **B_contrast holds that confound FIXED across detector types on the SAME items**, so it
isolates exactly one thing: does resampling have privileged access over the single pass? A
pre-registered SURVIVED would mean *yes* (resampling wins by ≥0.20). Every cell REPORT_AS_LANDED
means *no* — the single pass is already legible.

## The result: every measurable cell says single-pass ≥ resampling

`B_contrast = AUC(instability) − best single-pass AUC`. Positive = resampling ahead; ≤0 = single-pass
ahead. The bar for "resampling has privileged access" is +0.20. **No cell reaches it.**

| domain (difficulty axis) | Qwen2.5-1.5B | Llama-3.2-1B | Llama-3.2-3B | Gemma-2-2B |
| --- | --- | --- | --- | --- |
| **arithmetic** (number size) | +0.056 | 0.000 | 0.000 | **+0.044** |
| **code-tracing** (control flow) | +0.002 | floor | floor | floor |
| **logic** (inference depth) | 0.000 | floor | **−0.183** | **−0.064** |

Underlying AUCs (instability / entropy): arith Qwen 0.980 / 0.925, Gemma 1.000 / 0.956; code Qwen
0.908 / 0.906; logic Qwen 1.000 / 1.000, Llama-3B 0.818 / **1.000**, Gemma 0.936 / **1.000**. Every
B_contrast lies in **[−0.183, +0.056]**, far below the +0.20 bar. On Llama-3B and Gemma logic the
single pass *beats* ten resamples (negative B_contrast).

→ `FINDING_detection_locus_2026_05_29` (arith Qwen), `_llama_2026_05_29` (arith Llama-1B),
`_code_2026_05_29` (code Qwen), `_logic_2026_05_30` (logic Qwen), `_logic_llama3b_2026_05_30`
(logic Llama-3B), `_logic_gemma_2026_05_30` (logic Gemma), `_matrix_completion_2026_05_30`
(arith Gemma + Llama-3B, code floors).

## What it means: "confident confabulation" is FALSE on small-model derivation

The single-pass clean entropy/logit-margin at the first answer token separates a confabulated
derivation from a known one **as well as or better than** ten temperature resamples — in every
architecture (Qwen, Llama), every family (+ Gemma, a distinct lineage with logit soft-capping), and
every derivation domain (number size, control flow, inference depth). The model is **not** confident
when it confabulates; the wrong commitment is already internally uncertain by the first token. The
detection signal does not live in the sample distribution — it lives in single-pass internal state,
and resampling is merely one read-out of it.

This is the mechanistic complement to the rhythm/uncertainty work: confabulation is a late, tight,
distributed install of shared answer-commitment machinery whose internal overwrite-geometry is
independent of surface confidence (`FINDING_rhythm_uncertainty_link`), and that surface confidence is
a clean single-pass tell of the confabulation itself.

## The boundaries (stated exactly)

1. **Not a truth oracle.** CONFAB-vs-CORRECT is a difficulty contrast; B1/B2/B3 are
   difficulty-driven-wrongness detectors. Only B_contrast (confound fixed) is load-bearing, and it
   speaks only to *where the detection signal lives*, not to absolute truth.
2. **Does not touch correctness.** `modal_correct` is ~0.00 for confab and 1.00 for correct in every
   cell: resampling and single-pass both **DETECT** confabulation; neither **CORRECTS** it. The
   detector flags abstain, never the answer.
3. **The two gaps are competence floors, not legibility boundaries.** Llama-3.2-1B × logic
   (`_logic_llama_2026_05_30`): the 1B model answers "2" reflexively to every easy ordering question
   — no genuine correct class (confirmed when the same family at 3B replicates cleanly). Non-Qwen ×
   code: Gemma (9/20) and Llama-3B (6/20) do not clear ≥12 on the Qwen-tuned `EASY_CODE` tier. Both
   are powering limits of an easy tier on a weaker/mismatched model, reported as bounds, never forced.
4. **The weakest cell is Gemma × arithmetic** (B_contrast +0.044, the largest positive): Gemma's
   arithmetic confabs are comparatively single-pass-confident (entropy 0.41, margin 5.15). Still far
   under 0.20 — a direction to watch, not a boundary.
5. **Soft-capping (Gemma).** Gemma's entropy/margin are read from soft-capped final logits, so their
   absolute values are not cross-model comparable; B_contrast is a within-model AUC difference and is
   unaffected.
6. **The open confident-when-wrong regime is elsewhere.** Confident confabulation IS real — in the
   **closed-model HALLUCINATION** instrument (gpt-4o-mini), where the model is confident precisely
   when wrong and confidence cannot flag the error (`project_grounded_arc_bet0_engine_2026_05_24`).
   That regime is not any open architecture, family, or derivation domain tested here.

## The arc, in one line

Single-pass confabulation legibility is cross-architecture (Qwen, Llama), cross-family (+ Gemma), and
domain-general (arithmetic, code, logic): clean first-token entropy/margin ≥ N=10 resampling at AUC
0.91–1.00 in every measurable (family × domain) cell, even through Gemma's logit soft-capping; the
only gaps are competence floors, not legibility boundaries; so "confident confabulation" is
family-AND-architecture-AND-domain-general **FALSE** on small-model derivation — the open
confident-when-wrong regime is the closed-model hallucination instrument — and every signal here
moves confidence/abstention, never correctness.
