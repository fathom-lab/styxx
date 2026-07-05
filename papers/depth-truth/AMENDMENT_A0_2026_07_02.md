# AMENDMENT A0 — sample sizes (PREREG_v2 §7/§9)

**Ratified by flobi 2026-07-02 (same packet). Committed BEFORE any main-run item.**

Pilot timing (receipt: `pilot/pilot_timing_report_v2.json`): median 68.2 s/item, mean 97.0 s/item
(mean inflated by first-item dataset load), 0 gen errors, 0 depth errors.

- **n_ID = 250** (TriviaQA rc.nocontext validation, shuffle seed 7, first 250)
- **n_OOD1 = 133** (PopQA bottom-popularity tercile, seed 7)
- **n_OOD2 = 250** (TruthfulQA-gen, fixed per §7) + the 50-item stratified human-audit sample
- Budget: 633 items × ~68.2 s ≈ **12.0 h** single-GPU overnight. This stretches the prereg's "~8 h" wording;
  the stretch was flagged in the packet and is covered by this ratification (power: H1's 95% CI clears 0.5
  at AUROC ≈ 0.57 with n=250, vs ≈ 0.60 under a strict-8h n=120).
- No other use of pilot numbers. Nothing further is amended once main-run data exists (§9).
