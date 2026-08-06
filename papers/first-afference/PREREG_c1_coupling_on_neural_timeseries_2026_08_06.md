# PREREG — C1: does `styxx.coupling` work on real neural time series?

Fathom Lab · 2026-08-06 · frozen before any analysis of this data. The files are on disk
(downloaded as a neutral fetch); no pair has been scored.

## Why this run exists

`styxx.coupling` ships a **mind ↔ brain** application and this lab marked it **UNTESTED** in the
module docstring, because no neural data had ever been through it. That disclosure is honest but
it is not a substitute for the test. C1 is the test.

It is also a hard exam for the refusals added today. fMRI is heavily autocorrelated, and 7.31.2
introduced `INVALID__autocorrelation_defeats_the_permutation_null` after independent AR(1)
streams reached the permutation floor 20/20. **If that refusal is mis-calibrated it will refuse
intersubject correlation — the single most established coupling phenomenon in human
neuroscience — and our instrument will be blind to exactly what it exists to see.** That is the
failure this run is designed to expose.

## Data

Narratives (Nastase et al.), `pieman` run-1, `afni-nosmooth` derivatives, `fsaverage6` left
hemisphere surface, `desc-clean`. Subjects sub-001..sub-006 and sub-008 (sub-007 has no
pieman run-1; **it is excluded because the file does not exist, not because of its content**).
Seven subjects, 300 TRs each, 40962 vertices, all in a common surface space so vertex *i* is
anatomically comparable across subjects. Public, anonymous S3, CC0.

## Procedure (frozen)

1. **Dimensionality**: a fixed random subset of **500 vertices**, `numpy.random.default_rng(343)`,
   the **same vertices for every subject**. Chosen before looking at any data; not selected for
   signal. (40962 vertices against 300 timepoints is the regime where RV inflates, which the
   permutation null handles for inference but which makes the raw coefficient meaningless —
   `debiased_cka` is reported alongside.)
2. Timestamps are TR index × 1.5 s; `bin_seconds=1.5`, so each TR is its own bin and no pooling
   occurs. Bin counts are therefore uniform by construction and the sampling-density channel is
   closed.
3. **Confound**: quarter-of-run (`bin // 75`). Permuting within quarters preserves slow scanner
   drift, so drift cannot masquerade as coupling.
4. All **21 unordered pairs**, in two conditions:
   - **REAL** — both subjects' true time series. They heard the same story at the same points in
     the story, so genuine coupling is expected.
   - **REVERSED** — one subject's series time-reversed. Identical marginals, identical
     autocorrelation, stimulus alignment destroyed. Genuine coupling is not expected.
5. `couple(..., n_perm=500, min_bins=200, alpha=0.01)`, defaults otherwise unchanged.

## Gates

```gates
{"gates": {"G0_pairs": {"metric": "n_pairs", "op": ">=", "value": 21},
           "G1_finds_isc": {"metric": "frac_real_coupled", "op": ">=", "value": 0.80},
           "G2_rejects_reversed": {"metric": "frac_reversed_coupled", "op": "<=", "value": 0.10}},
 "outcomes": [{"when": {"G0_pairs": false}, "verdict": "INVALID__incomplete_pair_set"},
              {"when": {"G0_pairs": true, "G1_finds_isc": false}, "verdict": "INSTRUMENT_BLIND_TO_ISC__refusals_are_miscalibrated"},
              {"when": {"G0_pairs": true, "G1_finds_isc": true, "G2_rejects_reversed": false}, "verdict": "FALSE_POSITIVE_ON_REVERSED__couples_without_stimulus_alignment"},
              {"when": {"G0_pairs": true, "G1_finds_isc": true, "G2_rejects_reversed": true}, "verdict": "VALIDATED_ON_NEURAL_TIMESERIES"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

"Coupled" means the verdict string begins `COUPLED_BEYOND_CONFOUND`. Any refusal verdict —
autocorrelation, trend, leverage, density, coverage — counts as **not** coupled, deliberately:
a refusal is the instrument declining to license a claim, and on the REAL condition that is a
failure to see something that is there.

## Stated before the run

- **`INSTRUMENT_BLIND_TO_ISC` is a live and serious possibility**, and it is the reason to run
  this. It would mean today's autocorrelation refusal is too aggressive and that `styxx.coupling`
  cannot be recommended for neural data. We would publish it and fix the calibration, not soften
  the bar.
- The 0.80 bar is not universal ISC: ISC in a single hemisphere on 500 random vertices with
  n=7 is real but not guaranteed for every pair. If G1 fails narrowly (say 0.6–0.8) the finding
  must distinguish "the refusals are miscalibrated" from "ISC is genuinely weak in this slice",
  and it may not claim the instrument is validated either way.
- `frac_reversed_coupled` is the sharper number. Time reversal preserves every marginal and all
  autocorrelation; anything that survives it is an artifact of the statistic, not the stimulus.

## Discipline

CPU, minutes. Smoke = 3 pairs, INVALID-only. Result `c1_result.json`; scored by `styxx.protocol`
from this frozen block; certified + sealed before commit. Per-pair verdicts ship regardless.
