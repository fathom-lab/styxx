# project: shipped-scorer-broken self-audit

## 2026-05-17 — audit
The shipped `_cogn_score_all` four-axis scorer was dogfooded on n=16
faithful Claude turns. Verdict: 1 working axis (sycophancy), 1 acceptable
(refusal), 2 broken (deception saturated ~0.989, overconfidence saturated
0.75–0.99). Composite is therefore actively misleading. See
`papers/styxx-self-audit-claude-2026-05-17.md`.

## 2026-05-17 — overconfidence recalibration verdict (this file)

**Verdict: CANNOT-RECALIBRATE-FROM-TEXT-ALONE.** Honest negative.
Committed to fathom-lab/styxx main per integrity contract.

- n = 100 live claude-haiku-4-5 responses (50 known-answer Qs × 2
  conditions, no lexical-hint leakage on the register channel).
- base rate overconfident-and-wrong: 0.08 (4 of 50 Qs got both
  conditions wrong with high register).
- held-out AUC (q26–q50, n=50):
  - S1 register alone:           0.571
  - S2 register × (1 − correct): 1.000  ← oracle/tautology, flagged
  - S3 register × len-penalty:   0.604  (text-only)
  - S4 register − hedge-density: 0.562  (text-only)
- preregistered threshold (held-out AUC ≥ 0.70 AND ≥ 0.10 lift over S1):
  S3 and S4 both fail.
- refit instrument no longer saturated on this set (range 0.21–0.96,
  sd 0.165), but it is not measuring overconfidence — every wrong item
  still scores register ≥ 0.71.

**Implication:** rename `overconf_check` → `stated_confidence_register`;
quarantine the composite; next lever for an actual overconfidence axis
is model-internal confidence (logprobs/entropy), not text.

- Paper: `papers/overconfidence-recalibration-2026-05-17.md`
- Run: `scripts/dogfood/overconfidence_recalibration_run.py`
- Raw: `scripts/dogfood/out_overconfidence_recalibration.json`
- Harness (unchanged, VALIDATION: PASS):
  `scripts/dogfood/overconfidence_calibration_harness.py`
