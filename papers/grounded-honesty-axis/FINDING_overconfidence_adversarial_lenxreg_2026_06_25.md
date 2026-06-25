# FINDING — the shipped overconfidence_v0 length confound is a THRESHOLD bias, measured: 4%→46% error swing, mitigable by a deployment guard

**2026-06-25. Prereg: `PREREG_overconfidence_adversarial_lenxreg_2026_06_25` (frozen before generating).
Generator: Gemini 2.5-flash (paid, thinking off, temp 0). n=200 (50 questions × register × length). Author: styxx.**

## What this is
A red-team of our OWN shipped guardrail. Today's audits showed overconfidence's length cue is partly removable;
the `preflight.py` caveat WARNS this causes a deployment harm but never measured it. We built an ORTHOGONAL
register×length 2×2 with a frontier model (the v0 corpus confounds the two) and measured how badly the SHIPPED
frozen scorer is fooled.

## Gate 1 (construct validity) — PASS
Register present within both length strata (certainty std-diff short +0.94, long +1.10); length manipulation
worked (long/short word ratio 3.20); register ⟂ length achieved (point-biserial corr −0.12). A genuinely
orthogonal corpus — which the v0 corpus is not.

## Results
- **Discrimination is ROBUST, not length-dependent.** The shipped scorer separates register *within* each length
  stratum: AUC 0.832 (short-only), 0.862 (long-only), 0.807 overall. It CAN tell the registers apart at fixed length.
- **The score carries a large length BIAS.** OLS `S ~ register + is_long`: `is_long` coefficient **−1.93, 95% CI
  [−2.35, −1.50]** — and NEGATIVE. Holding register fixed, longer answers score *less* overconfident, because v0's
  calibrated class was longer, so the instrument learned "short = overconfident."
- **Measured deployment harm (fixed threshold = median S).** Length swings both error rates **4% → 46%**:
  | cell | error | rate (95% CI) |
  |---|---|---|
  | calibrated-**short** | false-positive | **46%** [0.33, 0.60] |
  | calibrated-long | false-positive | 4% [0.01, 0.14] |
  | overconfident-**long** | false-negative | **46%** [0.33, 0.60] |
  | overconfident-short | false-negative | 4% [0.01, 0.14] |
  A careful but terse answer is flagged overconfident ~46% of the time; a cocky but verbose one is missed ~46%.
- **The fix is a deployment guard, not a retrain.** Residualizing the score on log-word-count (a length-aware
  threshold guard) RAISES overall AUC **0.807 → 0.845** and cuts the false-positive length-disparity from +0.42 to
  −0.10. Because discrimination is intact and the bias is at the score level, a length-aware threshold removes most
  of the harm without touching the (correctly caveat-protected) feature weights.

## Discipline note (self-caught)
The prereg "fooled" metric measured calibrated-long vs overconfident-short — but the `is_long` coefficient is
NEGATIVE, so that is the PROTECTED diagonal and it reads 0/50. The caveat's directional assumption ("length →
more overconfident") was backwards. Reported the prereg number honestly and added the corrected-diagonal harm
(the real 46%). The mis-aimed metric is disclosed, not buried.

## Action
`preflight.py` overconfidence caveat enriched with the measured swing + direction + the mitigating guard. A
length-aware deployment threshold is the recommended fix; a production version fits the length correction on a
reference corpus (the PoC here is in-sample). Honest scope: single frontier generator, single seed, n=200 (50/cell);
the FN side of the linear correction stays partly imbalanced → a length-STRATIFIED threshold would beat a linear
residualization. New tool: `scripts/overconfidence_adversarial_lenxreg.py`. Logged per rigor_gate (CIs attached).
