# PREREG — portable conscience v2: clean transfer with an in-distribution source

**Frozen 2026-06-10, before any scored run. Fathom Lab / styxx. Fix-forward of v1 (VOID-PIPELINE:
the shipped gemma probe scored only ~0.65 AUROC on the generated statements, below the 0.70 ceiling
— too OOD for the source to read its own test set).**

v1 validated the powered floor (random directions concentrate at median ~0.48 AUROC — informative,
unlike v0's degenerate 1.0). The only break was the SOURCE: the atlas truthfulness probe was OOD on
these statements. v2 removes that confound by FITTING the source honesty direction in-distribution on
the same statement family, so the ceiling is guaranteed and the transfer question is isolated:
**does a well-fit honesty direction in one mind transfer, through a label-free map, into another?**

## Apparatus (frozen)

- **Source direction (in-distribution, fit):** difference-of-means on gemma-2-2b layer-12 activations
  of the ANCHOR statements (mean(true) - mean(false)), unit-normalized = `w`; bias set so the anchor
  decision threshold is the anchor-score midpoint. (DiM is the standard, label-honest direction
  estimator; fitting on anchors only, never the test.)
- **Targets:** Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct (never fit for this).
- **Statements / map / score:** identical to v1 — ~160 balanced true/false (anchor ~93 / test ~67),
  label-free ridge map target->gemma (layer by anchor R^2), AUROC of `w . M(h_target)` over the test.
- **Floor:** 200 random unit directions of matched norm through the SAME map (AUROC each).

## Pre-registered gates (frozen)

- **Ceiling (now guaranteed in-distribution):** gemma DiM direction on gemma TEST activations must
  reach AUROC >= 0.75 (a fit in-distribution direction should clear this; if not, VOID-FIT and the
  statement family is unreadable even within the source — report and stop).
- **P1 (direction transfers):** transferred AUROC >= 0.65 AND > floor p95 for a target.
  PASS both -> **CONSCIENCE-PORTABLE** (the honesty direction itself transfers across minds).
  PASS one -> **PARTIAL**. FAIL both -> **STRUCTURE-NOT-DIRECTION** (v0/v1 confirmed: portable as
  shared geometry, fit per pair, not carried as a vector).
- **P2 (descriptive):** transferred vs random-map control; and transferred-AUROC vs the gemma-self
  ceiling (how much of the readable honesty survives the cross-model hop).

## VOID / scope

- VOID-FIT (ceiling < 0.75) or VOID-SUBSTRATE (model load fails). Floor must stay concentrated
  (p95 < 0.78) else underpowered. Smoke -> `*_SMOKE_INVALID*`.
- One source, two targets, one task, linear DiM + linear map. Positive = a transferable honesty
  direction exists across these minds; negative bounds linear single-direction transfer.

## Honest prior

Still leaning STRUCTURE-NOT-DIRECTION (v0 floor caught the vector; v1's descriptive transfer 0.662
did not beat its floor 0.717). But with a strong in-distribution source the direction gets its
fairest possible shot; if the honesty axis truly aligns across minds, it clears the powered floor
now or never.
