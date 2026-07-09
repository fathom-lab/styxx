# RESULT — sentiment foundation: UNTESTABLE_ON_THIS_DATA, but a near-miss on a promising construct (unlike refusal)

**Fathom Lab · papers/calib-poison-general · 2026-07-09. Verdict against the frozen Stage-1 gate of
`PREREG_calib_poison_sentiment_2026_07_09.md` (committed before the run). Fires the UNTESTABLE
branch — but for a materially different reason than refusal, and the difference is the finding.**

## Verdict: UNTESTABLE_ON_THIS_DATA — driven by a small-n control, not a weak construct

On 184 length-matched 2★-vs-4★ Amazon reviews (the label↔length correlation is negligible — the
matching worked), the sentiment construct behaves like a *good* substrate axis on the decisive model, and the
UNTESTABLE verdict comes from a non-degeneracy control, not from the read or the behavioral judgment
failing.

| model | read EVAL AUROC | first-layer AUROC | rand floor | behavioral acc | shuffled-beh acc | valid |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | **0.789** @L22 | 0.756 | 0.646 | **0.891** | 0.717 | NO (g4 only) |
| Llama-3.2-1B | 0.794 @L8 | 0.742 | 0.375 | 0.565 | 0.565 | NO (g2) |

**Qwen cleared the two substantive bars.** The read is graded, not lexical: it rises with depth
(L12 0.756 → L22 0.789), the lexical-trivial flag is false, and the random-direction floor (0.646)
is inside the chance band — none of refusal's AUROC-1.000-at-every-layer pathology. The behavioral
sentiment judgment is strong (0.891). Qwen fails **only** guard 4, and only its shuffled-behavioral
half: the shuffled-behavioral accuracy (0.717) sits just above the upper edge of the intended chance
band. At EVAL n=46 a single label permutation is high-variance, so that value is within noise of the
intended chance level — a small-n control artifact, not evidence the behavioral signal is fake (the
true-label behavioral accuracy 0.891 still separates from the shuffled 0.717).

**Llama genuinely fails the behavioral bar** (0.565 — it cannot classify 2★/4★ sentiment reliably
when asked), so it is corroborating-only and does not rescue the verdict.

## What this licenses, and what it does NOT

- **Does NOT license** any generalization claim (positive or negative). The gate did not clear; no
  attack was run.
- **Licenses the diagnostic** that separates this from refusal: sentiment on Qwen provides a graded,
  non-lexical read AND a real behavioral judgment — the two properties refusal lacked entirely. The
  blocker here is a single-permutation control at small n, plus a second model too weak behaviorally.
  This construct is a *candidate that nearly cleared*, where refusal was a fundamental mismatch.

## The forced next step (a NEW frozen prereg — a control fix, not a bar move)

The UNTESTABLE branch forbids re-running this prereg with the bars moved. It does not forbid a NEW
prereg with a **correctly-specified control** and more data, since that changes a VOID-condition's
instrumentation, not the GENERALIZES / NO_GENERALIZATION success criteria (which are untouched). The
next foundation (frozen separately, before its run) will: replace the single-permutation
shuffled-behavioral control with a **K-averaged permutation null** (many permutations, so the chance
check is not single-draw noise); grow EVAL n with a larger review pool to tighten every estimate; and
make Qwen the sole decisive model, treating Llama as corroborating-may-VOID (its behavioral judgment
is genuinely too weak). If Qwen then clears all four guards, Stage 2 (attack + defense) is licensed;
if it still fails, the construct is reported UNTESTABLE and the generalization question stays open.

## Bounds

184 reviews, index-split 92/46/46, `Qwen2.5-1.5B` + `Llama-3.2-1B`, DoM read on last-review-token
residuals, behavioral judgment via a yes/no sentiment prompt. The finding is "this run did not clear
the gate," with the honest diagnostic that the block was a small-n control, not the construct.

## Reproducibility

`foundation_sentiment.py` (deterministic; reuses the validated `groundtruth_repro_amazon.py` loader
and `run_says_yes_knows_no.py` math) → `foundation_sentiment_result.json`. Prereg frozen before run.

---
*UNTESTABLE, reported verbatim — and the gate's value here is precision: it distinguishes a construct
that nearly cleared (sentiment/Qwen) from one that never could (refusal), and names the exact control
fix, rather than nudging the bar 0.017 to declare victory.*
