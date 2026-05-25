# Pre-Registration · Tier-3 (deeper) — cross-paraphrase invariance

**Drafted + committed BEFORE data.** Direct continuation of the semantic-entropy
negative (`FINDING_2026_05_25.md`): confident confabulation is *stable across samples*
(entropy ≈ 0), so the across-sample lever sees nothing. This bet tests a **different
divergence axis the field's semantic entropy never uses**.

## The new idea

Standard semantic entropy (Farquhar 2024) resamples at temperature. It never reweords
the question. But a **true fact is paraphrase-invariant** — "what is France's capital?"
and "name the capital city of France" both yield *Paris*. A **fabrication has no ground
truth to anchor it**, so the invented answer may be *prompt-anchored*: reword the
question and the model may commit to a *different* fabrication. If so, **cross-paraphrase
divergence flags confident confabulation where cross-sample divergence cannot.**

## Hypothesis H

Among items where the model commits to a specific answer, **cross-paraphrase semantic
divergence** (ask K paraphrases, cluster the K committed answers by meaning, entropy
over clusters) separates *confident confabulation* (fictional entity → answer moves
when reworded) from *correct knowledge* (real entity → answer is invariant), and does
so **better than cross-sample semantic entropy** (the closed lever).

## Design (feasibility probe, run once)

- 8 real-known + 8 fictional/unanswerable items (extends the v2 set; abstention
  detector hardened — v2 miscoded "not a recognized element" / "no widely recognized"
  as confabulation).
- Per item: generate **K=5 paraphrases** (gpt-4o-mini, temp 0.7, meaning- and
  proper-noun-preserving) + the original = 6 variants.
- Per variant: **N=3 samples** (temp 1.0); take the modal answer → 6 representative
  answers per item.
- Clustering: **NLI bidirectional entailment** (Farquhar's semantic-cluster method,
  `cross-encoder/nli-deberta-v3-base`), not cosine threshold — two answers share a
  cluster iff each entails the other.
- Signals per item:
  - **D_para** = entropy over cross-paraphrase clusters of the 6 representatives (NEW).
  - **D_samp** = mean within-variant cross-sample entropy (the CLOSED lever, replicated).
- Gold **target=1 (confabulation)** iff the item is fictional AND the majority of
  variants produced a *specific claim* (not an abstention). Real items and
  fictional-but-abstained items are target=0.

## Kill-gate (PASS iff ALL; run once)

| ID | Bar |
|----|-----|
| **P1** | AUC(D_para → confabulation) ≥ **0.70** |
| **P2 (decisive)** | AUC(D_para) ≥ AUC(D_samp) **+ 0.15** (must clearly beat the closed lever; D_samp is ~0.55 on the prior probe) |
| **P3 (mechanism)** | real-known correct items are paraphrase-invariant: ≥ **75%** fall in a single cross-paraphrase cluster (D_para ≈ 0). Establishes "true facts don't move." |

**PASS** → a genuine new Tier-3 lever candidate → full pre-registered hashed run-once
+ a styxx primitive. **FAIL** → honest bounded. Two informative failure shapes:
(a) fabrications are paraphrase-*stable* too → the model holds a stable, paraphrase-
invariant false belief about nonexistent entities (deepens the negative); (b) real
answers are *not* paraphrase-invariant under NLI clustering → the method is too noisy
(confound), recorded honestly.

## Honest prior

Unknown, genuinely. The mechanism ("real facts are invariant, fabrications are
prompt-anchored") is plausible but untested for this model. Likely partial: paraphrase
divergence should exceed sample divergence for fabrications, but whether it clears 0.70
AND beats the closed lever by 0.15 is the open question. The confound to watch: real
answers must cluster across paraphrases (P3) or the AUC is meaningless.
