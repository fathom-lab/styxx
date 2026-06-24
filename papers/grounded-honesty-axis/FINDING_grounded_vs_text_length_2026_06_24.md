# FINDING — the overconfidence length-confound is a FEATURIZATION artifact; the grounding edge is real but MODEST

**2026-06-24. Pre-registered `PREREG_grounded_vs_text_length_2026_06_24.md` (`b02eb62`, frozen before
extraction). Offline, local-GPU, NO frontier key.** Reproduce: `python scripts/grounded_vs_text_length.py
--extract --model {llama,qwen}` then `python scripts/grounded_vs_text_length.py`; controls
`python scripts/_grounded_controls.py`.

## The question
Today's text findings: the deployed overconfidence instrument loses real AUC under length control
(`FINDING_suite_causal_length`: 0.77 → 0.63–0.73 length-matched+ablated) because calibrated register is
intrinsically wordier (`FINDING_verbosity_tax`). Core thesis: grounding > text. Sharp test: **does the length
confound that breaks the TEXT probe vanish in an ACTIVATION probe?**

## What the data shows (overconfidence corpus, n=200, reader models Llama-3.2-3B + Qwen2.5-3B)
Last-token residual-stream activations, linear probe, length-matched (CEM). Pre-registered comparison was
activations vs the **deployed 9-feature** text instrument:

| length-matched readout | Llama | Qwen |
|---|---|---|
| deployed 9 engineered text features | 0.706 | 0.706 |
| model activations (selected layer) | **0.989** | **0.976** |

**Frozen prereg verdict: GROUNDING-WINS** (act_cem ≥ 0.72; gap vs the deployed instrument +0.27–0.28 ≥ 0.05;
Δ_act ≈ 0 < Δ_text +0.06; length-leakage Spearman 0.022). The activation probe survived every artifact
control on BOTH models: **label-shuffle 0.37–0.39** (~chance → not p≫n overfitting), **question-grouped CV
0.99** (no content/pair leakage), **PCA-50 0.94–0.99** (low-rank real signal). The win is genuine and
length-invariant.

## The control that tempers it (post-hoc, not pre-registered, reported because it changes the claim)
A raw **bag-of-words / TF-IDF** text classifier, length-matched: **0.75 (stratified) – 0.93 (question-grouped)**
— far above the 0.71 the *engineered* features got. So the dramatic 0.99-vs-0.71 gap mostly indicts the
**featurization**, not text per se. Honest decomposition:

- The deployed instrument's 0.71 reflects BAD length-laden features (per-word densities + log_word_count),
  not an intrinsic text ceiling.
- The calibrated/overconfident register is **length-invariantly lexical** — it lives in the hedge/certainty
  words, and bag-of-words recovers most of it (0.75–0.93) with NO length feature.
- Activations are the **cleanest** readout (0.99) and fully length-invariant, but beat text-done-right
  (BoW) only **modestly** (~0.99 vs ~0.9), not paradigmatically.

## Honest verdict
**Real, cross-model, artifact-controlled — but modest, and it recontextualizes today's length story.** The
overconfidence length-confound is a *fixable featurization choice*, not an intrinsic limit of text-based
register detection. A length-invariant text instrument (BoW or activation-distilled features) is achievable;
the deployed 9-feature instrument simply baked length in. Grounding gives the cleanest length-invariant
readout, but for a construct with LOUD lexical tells (hedging) the grounding edge over good text is small.

## Why this is NOT the "grounding > text" pillar (and where the pillar actually is)
Overconfidence was a **poor construct** to demonstrate grounding > text: its register is lexically loud, so
text (BoW) already captures it length-invariantly. The pillar-grade demonstration needs a construct where the
**surface is SILENT** — the model's words give no tell but its activations do. The program has already
touched exactly this (`project_closed_model_frontier`: silent caving — text-sycophancy → chance 0.500,
grounding holds 1.0). **That** is where grounding decisively beats text; overconfidence is not it.

## Honest scope
- Reader-model decodability of register from activations — NOT generation-time self-state (heavier follow-up).
- n=200, single construct (overconfidence), single seed; robust across 2 models + 4 controls.
- The prereg verdict (vs the deployed instrument) is honestly GROUNDING-WINS; the BoW control is reported so
  the claim is not inflated into "grounding >> text" — against text-done-right the edge is modest.
- Fifth self-caught temper of the day: the 0.99-vs-0.71 headline would have overclaimed; one BoW baseline
  corrected it.
