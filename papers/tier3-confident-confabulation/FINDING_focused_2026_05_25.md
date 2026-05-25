# Finding · Tier-3 focused probe — VOID (by gate) + a real caution against NLI clustering

**2026-05-25.** Prereg `preregistration_focused_2026_05_25.md`. Goal: test whether
entailment clustering keeps *paraphrastic-correct* answers low-entropy where a tuned
cosine threshold false-positives. Result: **VOID by the pre-registered validity gate**,
plus a robust descriptive caution that revises the lean of `FINDING_corrected`.

## Why VOID

The validity gate required C2 (real, paraphrastic) items to show genuine surface
variation (mean within-item pairwise cosine < 0.90) and be correct (≥5/6 entail a
reference). **0 of 8 C2 items qualified** — every "why/how" question came back at mean
pairwise cosine **0.93–1.00**. At temperature 1.0, gpt-4o-mini answers explanations
*near-verbatim*, just as it does simple facts. The intended "same fact, many surface
forms" condition does not arise naturally for this model, so F1/F2 are unscoreable. Per
the prereg: re-design, do not reinterpret.

(That non-result is itself informative: for this model, correct = *consistent* almost
regardless of question type; the paraphrastic-correct confound I worried about in
`FINDING_corrected` barely exists at temp 1.0. It would have to be forced.)

## The caution (robust across both runs — descriptive, not a scored bar)

Even though C2 failed the *surface-variation* gate, the answers varied enough to expose
the entailment judge:

| on the 8 C2 (correct) items | value |
|---|---|
| mean pairwise cosine | 0.96 (tight) |
| mean **cosine@0.95** entropy | **0.46** |
| mean **NLI** entropy | **0.79** |
| C2 items NLI flagged (entropy > 0.5) | **5 of 8** |

`nli-deberta-v3-base` **fails to recognize equivalent explanations as mutually
entailing** ("the sky is blue from Rayleigh scattering" vs a reworded correct version →
different clusters → entropy 1.79 on a *correct* item). So NLI clustering has its **own
false-positive mode on free-form correct text**, and here it false-positived *more* than
cosine@0.95 did. Across every condition tested so far, cosine@high-threshold has been at
least as robust as NLI.

## What this does to the corrected finding

- The headline stands: semantic entropy separates confident confabulation from correct
  answers (AUC 0.93–0.95), because **confabulation is inconsistent** and correct answers
  are consistent. Confabulation detection is robust.
- But **"NLI is the right clustering primitive" is now actively doubtful.** NLI is
  threshold-free yet noisy on free-form correct answers; cosine is threshold-sensitive
  yet was at least as robust once tuned. Neither is clean. The clustering step is the
  open problem, not the divergence signal.
- The paraphrase-robustness question (does *any* cheap clustering keep
  surface-varied-but-correct answers together while splitting confabulations?) is
  **unresolved** and harder than `FINDING_corrected` implied.

## Redesign for a real test (not run here)

To actually exercise the crux: (1) **force** surface variation in correct answers
(instruct varied phrasing, or temp ≥ 1.3, or questions with several genuinely distinct
correct answers), so C2 clears a surface-variation bar; (2) judge semantic equivalence
with something **stronger than nli-deberta-base** (a larger NLI model, or an LLM judge
with a calibrated equivalence prompt); (3) re-pre-register F1/F2. Only then is "NLI vs
cosine under paraphrase" answerable. Until then, a shipped `semantic_entropy` primitive
should be treated as **not validated** — and if built, cosine@~0.9–0.95 is the safer
default given NLI's free-text FP mode, with the clustering choice flagged as open.

## Meta

Third self-catch in this arc: tweet (wrong) → first correction (also overclaimed:
"use NLI") → this probe (NLI itself false-positives). The "focused probe first" call was
right — it caught an FP mode that would have shipped in an NLI-based primitive. The
discipline (validity gate honored even when it voids the headline test; lean revised
against my own prior) is the point.
