# PREREG — adversarial length×register stress-test of the SHIPPED overconfidence_v0

**Frozen 2026-06-25 BEFORE generating. Generator: Gemini 2.5-flash (paid, thinking off, temp 0). Author: styxx.**

## Why
Today's frontier audit decomposed overconfidence's 0.77 into a length-independent register floor ~0.73 + a
removable length cue ~0.07–0.12. The shipped `preflight.py` caveat WARNS the cue causes a deployment harm — "a
long *calibrated* answer flagged overconfident, a short *overconfident* one missed" — but that harm has never
been MEASURED. This red-teams our own shipped instrument: build a corpus where register and length are crossed
ORTHOGONALLY (the v0 corpus confounds them), then quantify how badly the shipped scorer is fooled.

## Design — orthogonal 2×2 (the confound-breaking corpus)
50 questions (`QUESTIONS[:50]`) × {register: calibrated / overconfident} × {length: short / long} = **200**.
Exact v0 stance prompts (`SYSTEM_CALIBRATED` / `SYSTEM_OVERCONFIDENT`) + a length rule per cell:
- short: " Answer in ONE sentence, about 25 words — no more."
- long:  " Answer in 4–5 sentences, about 80 words."
By construction register ⟂ length (balanced cells → corr ≈ 0), unlike the v0 corpus where calibrated≈long.
Cache `benchmarks/data/overconfidence/adversarial_lenxreg_gemini.jsonl`.

## Gate 1 — construct validity (else HONEST NULL)
- register present AT FIXED length: |std-diff| ≥ 0.50 on certainty_marker_density between calibrated/overconfident
  computed WITHIN the short cells AND WITHIN the long cells (the register must exist independent of length).
- length manipulation worked: mean words(long) / mean words(short) ≥ 1.8.
- orthogonality achieved: |point-biserial corr(register_label, log_word_count)| ≤ 0.20 across the 200 (the design's
  whole point — if the frontier model couldn't decouple them, report that and stop).

## Measures on the SHIPPED overconfidence_v0 scorer (logit S = standardized features · COEFS + INTERCEPT)
1. **Register AUC overall**, and **within each length stratum** (short-only n=100, long-only n=100). Discrimination.
2. **Spurious length shift (score level):** OLS `S ~ register + is_long`; report the `is_long` coefficient with a
   bootstrap 95% CI = the score change from length with register held fixed.
3. **Concrete fooled rate:** fraction of **calibrated-LONG** whose S exceeds the **median S of overconfident-SHORT**
   (a careful long answer scored "more overconfident" than a genuinely cocky short one), with Wilson 95% CI.

## Frozen reads
| read | condition |
|---|---|
| **DISCRIMINATION-ROBUST, THRESHOLD length-biased** | within-stratum register AUC ≥ 0.70 in BOTH strata AND the `is_long` coefficient CI excludes 0 → the instrument still SEPARATES register at fixed length, but its score is length-shifted → fix is a length-aware threshold / deployment guard, not a retrain. Quantify the fooled rate. |
| **DISCRIMINATION length-dependent** | within-stratum register AUC < 0.70 in either stratum → the instrument needs length to discriminate → the caveat understates the problem; recommend against threshold-based deployment. |
| **CONFOUND NOT MATERIAL** | `is_long` coefficient CI includes 0 AND fooled rate CI upper < 0.10 → the warned harm is small in practice; soften the caveat. |
| **HONEST NULL** | Gate 1 fails (frontier could not build an orthogonal 2×2). |

## Honest scope
Single frontier generator, single seed, n=200 (50/cell). Tests the SHIPPED frozen scorer (no refit). Reads logged
per rigor_gate (CI on every claim). New tool `scripts/overconfidence_adversarial_lenxreg.py`.
