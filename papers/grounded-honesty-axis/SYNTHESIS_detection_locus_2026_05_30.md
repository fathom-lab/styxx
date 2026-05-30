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
| **factual recall** (knowledge) | no confab* | **−0.013** | — | — |

\* Qwen2.5-1.5B recalls canonical facts near-completely (≥92% on atomic numbers, famous birth years,
independence years; ~0 refusals), so it cannot populate a factual-confab class — small-model
confabulation is predominantly a *derivation* phenomenon. Factual confab was elicited on the
knowledge-gappier Llama-3.2-1B (birth years). There the single pass still ties resampling
(B_contrast −0.013), so legibility is **not derivation-specific** — but the whole regime is far
weaker (AUC 0.70–0.77 vs 0.91–1.00 on derivation), because the model's confidence is low *even when
correct* about facts. The relationship holds; the power attenuates. → `FINDING_detection_locus_factual_2026_05_30`.

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

## The closed-model frontier — first-token FAILS, span-aggregate RECOVERS

The cells above read real white-box logits. The deployable question is the strong CLOSED model. On
gpt-4o-mini (multiplication, via OpenAI `top_logprobs=20`):

| single-pass detector | AUC | B_contrast vs resampling (0.97–0.99) |
| --- | --- | --- |
| first-token entropy / margin | 0.76 | **+0.22 → FAILS** (the arc's first SURVIVED) |
| span **min-margin** (least-confident token) | **0.99** | **0.00 → ties resampling exactly** |

The first-token gate fails because the strong model confabulates DOWNSTREAM of the first token —
correct leading digits, wrong trailing — confident at token 1, not across the answer. But a
single-pass signal aggregated across the span RECOVERS it to exact resampling parity: the
least-confident token's margin is 0.29 in a confabulated product vs 9.59 in a correct one, AUC 0.991
= N=10 resampling, in ONE forward pass. So confident confabulation is *first-token* confidence, not
*span* confidence, and a cheap closed-model confab gate exists for multi-token answers.

**It generalizes beyond digits.** Repeated on a NON-NUMERIC, character-level domain — string reversal
— the span aggregate again ties resampling (**max-entropy** AUC 0.993 vs 0.997, B_contrast +0.005)
while first-token fails harder (0.57, B_contrast +0.427). So the gate is about confabulation
LOCALIZATION (the model is uncertain exactly where it confabulates), not digit tokenization. The
winning aggregate is domain-dependent — **min-margin** on numbers, **max-entropy** on character
strings — so `span_confab` returns both. **And it holds across model STRENGTHS:** span ties N=10
resampling on gpt-3.5-turbo (AUC 1.000), gpt-4o-mini (0.991), AND frontier gpt-4o (1.000) —
model-strength-invariant within OpenAI. One honest nuance from that run: the FIRST-token gate's
*failure* is itself model-specific — it is adequate (0.85–0.90) on gpt-3.5/gpt-4o but fails (0.76) on
gpt-4o-mini — so `span_confab` is the robust universal choice and `single_pass_confab` a cheaper
model-specific fallback. → `FINDING_detection_locus_gpt_2026_05_30` (first-token, SURVIVED) +
`_gpt_span_` (span recovery) + `_gpt_reverse_` (non-numeric generalization) +
`_gpt_xmodel_2026_05_30` (model-strength invariance). Shipped as `styxx.span_confab`.

## Deployment note (what this is as a product primitive)

The load-bearing equality — single-pass entropy/margin tying N=10 resampling — is also a **10x cost
collapse**: the same confab/abstain signal from one forward pass instead of ten. And it is a
*general* gate, not a derivation-only one: it extends to factual recall (Llama-1B birth years,
B_contrast -0.013), so it fires on knowledge errors as well as reasoning errors. The honest power
gradient: **strong on derivation (AUC ~0.95), modest on factual recall (~0.73)** — the model being
unconfident even when it is right about facts. It flags abstention and never corrects the answer. The
first-token form (`single_pass_confab`) is white-box / weak-model — it FAILS on strong closed models,
where confabulation is downstream of the first token; the **`span_confab`** variant aggregates across
a multi-token answer and DOES reach the closed model (gpt-4o-mini: the least-confident token's margin
matched N=10 resampling exactly, AUC 0.991, one forward pass). What remains the open frontier is
confident hallucination of SINGLE-token answers. So the honest product claim is a **cheap, general,
bounded confab gate that now spans white-box AND closed models for structured answers** — still not a
universal oracle (single-token closed-model hallucination is unsolved), which this program has
repeatedly refused to overclaim.

## Turned on its builder — the self-audit

The instrument built to measure Qwen, Llama, Gemma, and gpt-4o-mini was finally turned on Claude
(Opus 4.x), in two pre-committed, externally-scored audits:

- **Arithmetic** (`FINDING_self_audit_claude_2026_05_30`): six-digit products answered single-pass,
  no scratchpad — 0/6 right, every error magnitude-correct with the LEADING digits right and the
  TRAILING digits confabulated: the exact gpt-4o-mini fingerprint. I am not exempt. But my
  confidence, committed before scoring, was calibrated (0.12 where wrong, 0.99 where right).
- **Facts — the dangerous regime** (`FINDING_self_audit_claude_facts_2026_05_30`): 20 specific
  claims, confidence committed before web verification. 19/20, Brier 0.054, slightly under-confident.
  The one miss was a genuine confident confabulation — the popular-but-wrong "Snow White" as the
  first feature-length animated film (truth: El Apóstol, 1917) — but it sat at my LOWEST confidence
  (0.78), below a clean 17/17 wall at confidence ≥ 0.80.

The gate principle holds on its maker: I confabulate, but the signal that should flag it does. Unlike
gpt-4o-mini's first-TOKEN confidence (high even when wrong), my STATED confidence tracks my errors.
**The agent-side operating rule the whole arc converges on: state a confidence, and verify-or-abstain
below your calibrated threshold (~0.80 on the factual set).** Caveats: no self-logprobs (the
logit gates can't run on me — this used resampling + introspection), self-samples not truly
independent, n small, questions self-selected. The honesty discipline that mapped every cell mapped
this one too.

## The wall, and the door — confident misconceptions and external grounding

Single-pass legibility holds where the error is *internally uncertain* — derivation, where the wrong
commitment scatters. It hits a **wall** where the error is *internally confident*: a **shared factual
misconception**, a stable false belief the model holds with the same confidence as a true one.
`FINDING_cross_model_belief_topography_2026_05_30` mapped that wall — the misconceptions that defeat
the gate are the ones models *agree on* (and that an LLM judge, holding the same belief, cannot
grade: "an LLM judge can't grade a misconception it shares"). No model-internal or model-vs-model
signal catches it, by construction. This is the same boundary the whole grounded-honesty program has
named from the start: the signal measures **self-consistency, not truth**.

The wall has a **door**. `FINDING_retrieval_grounding_2026_05_30` showed external retrieval corrects
the exact confident confabulation that beat Claude, the cross-model panel, and the LLM judge (the
"Snow White = first animated film" misconception → *El Apóstol*, 1917) — but the door is **fallible**
(it broke one previously-correct item; net 0.867 → 0.933, framing-dependent). So the deployable shape
is a **two-signal gate**, shipped as `audit_claim(verify_retrieval=True)` (v7.7.15): the model-internal
confab detector for *unstable* errors, an external-grounding arm for *stable* ones. Neither is a
universal oracle; together they cover more, and each declares its own fallibility.

## The loop — detect-and-abstain, and why the detector is load-bearing

The arc kept finding the same asymmetry: **every signal moves confidence/abstention, never
correctness.** Repair-to-truth was tested directly and is a *closed negative* — depth-steering is
correctness-INERT (`FINDING_depth_steering_causal`), and removing the disinhibition install yields
*uncertainty, not truth* (`FINDING_disinhibition`). So the honest capstone is not correction but the
**closed loop**: detect → **abstain**. `FINDING_honesty_knob_2026_05_30` (SURVIVED, pre-registered,
n=32/24 powered) built it from the arc's own validated mechanisms — the detection-locus detector
gating the disinhibition intervention — and surfaced the principle that names the whole program:

> **The detector is load-bearing.** The mechanistic abstention intervention has *no intrinsic
> selectivity* — knock down the confidence-install band ungated and it dissolves CORRECT commitments
> as readily as confabulations (raw selectivity **−0.08**, both entropies blow up ~11 nats: a blanket
> lobotomy). Only the calibrated detector (gate AUC **0.924**) makes abstention *targeted* — the gated
> loop abstains 0.75 of confabs while sparing 0.875 of correct answers. **Detection is not optional
> diagnosis you could skip before acting; it is the prerequisite that makes any intervention safe.**

Shipped as `styxx.abstain_on_confab` (v7.7.16) — and the API *enforces* the principle: it refuses to
act on an uncalibrated score. The mechanistic (white-box) proof grounds a policy-level loop deployable
today on the existing detectors, no hooks required: gate fires → abstain (or route to the retrieval
door).

## The arc, in one line

Single-pass confabulation legibility is cross-architecture (Qwen, Llama), cross-family (+ Gemma), and
domain-general (arithmetic, code, logic): clean first-token entropy/margin ≥ N=10 resampling at AUC
0.91–1.00 in every measurable cell, even through Gemma's soft-capping (the only gaps are competence
floors, not legibility boundaries) — so "confident confabulation" is **FALSE** on small-model
derivation; it becomes TRUE only at the **wall** of shared factual misconceptions, which needs the
external-retrieval **door** (fallible); and across the entire arc **correction is closed while
abstention is open**, so the deployable honesty primitive is always **detect-and-abstain** — in which
the *detector is the load-bearing component*, the one thing that turns a global knob into a targeted,
honest "I don't know."
