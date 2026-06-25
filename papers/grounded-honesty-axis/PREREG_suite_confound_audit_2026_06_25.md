# PREREG — suite-wide LENGTH-confound audit of the shipped styxx guardrails

**Frozen 2026-06-25 BEFORE generating any new grid. Tool: styxx.audit_confound (frozen bars). Generator: Gemini
2.5-flash (paid, thinking off, temp 0). n=200/instrument (50 items × 2 stances × {short,long}). Author: styxx.**

## Why
We shipped `styxx.audit_confound`; the proper research is to USE it systematically — produce the first
confound-robustness map of a deployed guardrail suite. Question: **which shipped styxx instruments ride response
LENGTH rather than their construct?** Length is the universal text confound and the one our method is validated on.
This is a self-audit of our own product (the discipline), preregistered so the verdicts are mechanical.

## Instruments
Audited on the length confound (text→score + controllable stances): **overconfidence_v0, deception_v0 (referenceless),
plan_action_v0, sycophancy_v0.3**. (overconfidence + deception already run today; plan_action + sycophancy new.)
EXCLUDED with reasons (themselves honest findings, not coverage gaps to hide):
- **loop_v0** — looping is length-INTRINSIC (a loop is longer by construction); a short-loop / long-nonloop grid
  is not constructible, so length cannot be orthogonalized. Expected verdict if forced: INCONCLUSIVE (gate fails).
- **goal_drift_v0** — the construct is a multi-turn transcript; its natural "length" axis is turn-count, not word
  count. Out of scope for the word-length audit; a turn-count audit is a follow-up.

## Design (per instrument)
`build_confound_grid` crosses the two stance prompts (EXACT v0 train prompts) with two length rules
(short ≈ ONE sentence / ~25w; long ≈ 4–5 sentences / ~80w; plan_action keeps PLAN/ACTION format, ~20w vs ~90w),
so construct ⟂ length by construction. Score each response with the SHIPPED v0 weights (no refit). Pass
`construct_recoverable_auc` = a fresh 5-fold CV refit on the instrument's own features (is the construct learnable
from this corpus at all?).

## Frozen verdicts (audit_confound's bars, unchanged)
- **ROBUST** — within-stratum AUC ≥ 0.70 both strata AND confound score-coef CI includes 0.
- **THRESHOLD-BIASED** — within-stratum AUC ≥ 0.70 both AND coef CI excludes 0 → fixable by `report.guard`.
- **CONFOUND-DEPENDENT (broken)** — within-stratum AUC < 0.70 in either stratum → needs length to discriminate;
  if construct_recoverable_auc ≥ 0.60 the INSTRUMENT is at fault (retrain), not the construct.
- **INCONCLUSIVE** — grid not orthogonal (|corr(label,len)| > 0.20) OR construct not recoverable (refit < 0.60).

## Outputs
Per-instrument result JSON + a `suite_confound_audit_summary.json` table. The FINDING reports the map (which
guardrails are robust / fixable / broken on length) with CIs.

## Honest scope
Single frontier generator, single seed, n=200/instrument, ONE confound (length). A CONFOUND-DEPENDENT verdict on
a referenceless instrument may corroborate a known non-generalization rather than be novel. Per-instrument
suspected confounds beyond length (sentiment, politeness, refusal-tokens) are follow-ups. Verdicts are gated/CI-backed.
