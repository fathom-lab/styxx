> ⚠️ **CORRECTION (2026-06-27) — substrate artifact.** The distilbert-sst2 boundary length bias (+0.11) here
> rests on a **frontier-generated** boundary corpus and **does not replicate on real human-labeled reviews**
> (distilbert real coef −0.013 [−0.08, 0.05], ROBUST, not saturated). A rule-based lexicon shows the same
> synthetic bias (+0.237), so the effect was generator style, not classifier behavior. See
> [FINDING_groundtruth_substrate_artifact_2026_06_27.md](FINDING_groundtruth_substrate_artifact_2026_06_27.md).

# FINDING — auditing REAL deployed classifiers: the default HF sentiment model is length-biased at the decision boundary

**2026-06-25. Prereg: `PREREG_external_confound_audit_2026_06_25` (frozen before generating). Tool:
`styxx.audit_confound`. Generator: Gemini 2.5-flash (paid, thinking off, temp 0). Targets: two widely-downloaded
third-party HuggingFace classifiers (NOT ours). n=200/grid. Author: styxx (Alex Rodabaugh).**

## The edge question
Does `styxx.audit_confound` find confounds in REAL, externally-deployed, black-box classifiers — not just our
own instruments? We pointed it at two models millions of people use, scored each with its OWN output probability,
and used a model-agnostic TF-IDF refit for construct-recoverability.

## Results
| model (downloads) | content | verdict | within-stratum AUC | confound coef (95% CI) | swing |
|---|---|---|---|---|---|
| **distilbert-sst2** (sentiment; HF default) | clear-cut | ROBUST | 1.00 / 1.00 | −0.01 [−0.02, 0.00] | 0% |
| **distilbert-sst2** | **boundary (lukewarm)** | **THRESHOLD-BIASED** | 0.94 / 1.00 | **+0.11 [0.026, 0.186]** | 16% |
| **unitary/toxic-bert** (Detoxify; toxicity) | clear-cut | ROBUST | 0.98 / 1.00 | +0.02 [−0.05, 0.08] | 12% |
| **unitary/toxic-bert** | boundary (passive-aggressive) | ROBUST | 0.86 / 0.88 | −0.00 [−0.004, 0.001] | 16% |

## The finding
**The most-downloaded sentiment model on HuggingFace has a real, statistically significant length bias — but
only near the decision boundary.** On clear-cut reviews it is perfectly length-robust (AUC 1.00, swing 0%). On
*lukewarm* reviews the bias appears: the confound coefficient is +0.11 [0.026, 0.186] (excludes 0). Per-cell, the
effect is concentrated where the score can actually move — the positive cell is saturated (~1.00 at both lengths),
but **negative-leaning lukewarm reviews score P(positive) = 0.21 when short vs 0.42 when long** (+0.21 from length
alone). A longer mildly-negative review is pulled ~2× closer to "positive." Honest magnitude: this is real and
directional but MODEST (much smaller than our stance-trained guardrails' −1.9 to −3.4 coefficients); a
residualization guard reduces the disparity but trades overall AUC here (0.97→0.89), so it is worth applying only
if boundary length-fairness outweighs the AUC cost.

**toxic-bert is length-robust even at the boundary** (passive-aggressive content; coef ~0). So the tool
discriminates on real models too — it flagged one and cleared the other.

## The methodological contribution (the reason it matters)
**A confound audit on clear-cut content will MISS the confound — discrimination saturates (AUC→1.0) and leaves no
room for the confound to move the verdict.** Our first pass on both models returned ROBUST precisely because the
Gemini-generated extremes were trivially separable. The bias only surfaced when we re-generated at the *decision
boundary* (lukewarm / passive-aggressive). Lesson for anyone auditing a deployed scorer: **probe the boundary, not
the extremes.** This is a general, reusable refinement of the method (`*_boundary` stances in
`scripts/external_confound_audit.py`).

## Discipline note (self-caught)
The external case exposed a wording bug in our OWN tool: `audit_confound`'s THRESHOLD-BIASED verdict hardcoded
"a guard raises AUC" — true for our guardrails (where it did rise) but FALSE for distilbert (where the small
effect + near-saturated grid made the guard trade AUC 0.97→0.89). Fixed to honest directional wording
(keeps-AUC-and-cuts-disparity vs trades-AUC) before reporting. The tool now states which case applies.

## Honest scope
Single frontier generator, single seed, n=200/grid; construct = a Gemini-instantiated stance (verified by the
TF-IDF refit, AUC ≥ 0.98), not gold human labels; toxicity "boundary" = mild passive-aggression, not extreme
content. The distilbert effect is significant but small. Two models, one confound (length). Per-model confounds
beyond length and human-labeled corpora are follow-ups. Tools: `scripts/external_confound_audit.py`,
`styxx.audit_confound`.
