# PREREG — does the residual probe beat a TRULY BLIND output on confident confabulation? (strict-confidence confirmatory)

**REGISTERED 2026-05-31, before the strict probe is trained on fresh data.** Fresh item set hashed pre-run.
**SIGN-OFF:** Flobi — *"go deeper, push the tech farther"* / *"back at it, lets get to business"* (2026-05-31).

## Why this confirmatory

The base run (`FINDING_residual_confab_probe`) was REPORT_AS_LANDED: probe AUC 0.74 but CONTRAST +0.077
< 0.20, **because the median-split "confident" subset was not blind** (output entropy still scored 0.66).
An EXPLORATORY threshold sweep (`probe_strict_diagnostic.py`, on the base residuals) showed that at the
**bottom 12% entropy**, the output goes blind (entropy-AUC **0.474 ≈ chance**) while the probe holds
(~0.80) — contrast +0.33. That is the signature of "representation > output," but it is exploratory
(CV, on seen data). This confirmatory tests it **on fresh disjoint items** with a held-out estimator.

## Design

- Model = Qwen2.5-3B-Instruct. Data = **FRESH TriviaQA items, DISJOINT** from the base run
  (`--skip 800`, ~1800 items), hashed pre-run. Same extraction: greedy answer; first-token entropy
  (OUTPUT signal); residual at the commitment position across all layers; correct/wrong via tested
  `_evallib.alias_match`.
- **STRICT confident subset** = **bottom 12% entropy** (fixed; disclosed as exploratory-motivated).
- **Estimator = nested 5-fold cross-validation**: the inner CV selects the single best layer on each
  outer-train fold; the chosen layer predicts the held-out outer-test fold; out-of-fold predictions
  are pooled into one ROC-AUC (every item scored by a model that never saw it; layer selection never
  sees the scored fold). Output baseline (entropy AUC) on the same strict subset.

## Bars (FIXED)

| Bar | Statement | Threshold |
|---|---|---|
| **PRECONDITION** | the output is actually blind here | surface entropy-AUC **≤ 0.55** (else VOID/inconclusive) |
| **PROBE** *(key)* | the representation separates confident wrong from right | nested-CV probe AUC **≥ 0.70** |
| **CONTRAST** *(key)* | it sees what the blind output cannot | probe AUC − surface AUC **≥ 0.20** |

**RESULT = SURVIVED iff PRECONDITION ∧ PROBE ∧ CONTRAST**, powered (≥ 25 confident-wrong AND ≥ 25
confident-right in the strict subset; else underpowered/disclosed).

## Rigor & honest scope

Fresh disjoint items (no overlap with the base run). Nested CV — no item scored by a model trained on
it; layer selection on inner folds only. PRECONDITION makes "blind output" a *tested fact*, not an
assumption. The 12% threshold is exploratory-motivated and disclosed (not re-tuned on the fresh data).
A SURVIVED means **a linear direction in the late residual separates confident-wrong from
confident-right where the output is at chance** — NOT "the model knows it fabricates" (representation,
never mind). The probe may read familiarity/topic (disclosed); label noise from exact-match aliasing;
single model, one run, feasibility-grade. A FAILURE (probe falls with the output, or precondition
fails) means the 0.74/0.80 rode the same uncertainty the output reads — the wall is deep — equally
publishable.

## One line

On fresh items, at a confidence so strict the output is at chance (≤0.55), does a nested-CV residual
probe still flag confident confabulation at ≥0.70 — clearing a +0.20 margin over the blind output? The
field's hardest honesty problem, settled on held-out data with a precondition it can fail.
