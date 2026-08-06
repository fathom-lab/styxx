# PREREG — H1a: the first formal bimodality test on cross-subject neural alignment

Fathom Lab · 2026-08-06 · **frozen before any neural data has been downloaded, let alone
inspected.** No file from this dataset exists on disk at the time of this commit.

## Why this run exists

An external survey of the neuroimaging literature returned a specific gap:

> *No paper applies Hartigan's dip test, a mixture model, or any formal bimodality test to a
> per-subject decoding-performance distribution.*

Distributions get described in words — "varied largely between participants," "covering the full
range" — and low performers are frequently excluded before anyone looks. The question of whether
human cohorts contain **islands** (a subpopulation that a group-trained model cannot read) has,
as far as the survey could establish, never been asked with a statistical test. This lab has the
instrument (`styxx.islands`, shipped 2026-08-06), a frozen prediction that motivated it
(`../disjoint-worlds/PREDICTION_h1_human_islands_2026_08_06.md`), and an ungated public dataset.

## What is measured, and what is NOT

**Measured: cross-subject representational alignment.** For each of the 8 NSD subjects, the
`nsdgeneral` ROI betas for the **shared1000** images — the images every subject saw — averaged
across that subject's repetitions of each image. This yields 8 matrices of (shared images ×
voxels), a cohort over a shared item set, which is exactly `styxx.islands.survey`'s input.

**Not measured: decoding accuracy.** H1 proper is about per-subject *decoding accuracy*; this
run measures the *alignment* that would underlie it. The relationship between the two is
precisely what our own b46 result says is a **switch** rather than a ramp, so **a negative here
does not settle H1** and a positive here would not confirm it. This is H1's precursor, and it
stands in the same relation to H1 as B47 did on the model side. Stating that now, before the
result, so neither branch can be over-read later.

## Source and provenance

`huggingface.co/datasets/pscotti/mindeyev2` — the MindEye2 mirror of the Natural Scenes Dataset
(Allen et al., *Nat. Neurosci.* 2022), GLMsingle single-trial betas restricted to `nsdgeneral`,
ungated and requiring no data-use agreement. Files: `betas_all_subj0{1..8}_fp32_renorm.hdf5`
(~14 GB total), plus the shared-image index. Subject identifiers are the dataset's own; no
personally identifying information is involved at any point.

## Procedure (frozen)

1. Per subject, select rows corresponding to shared1000 images; average repetitions per image.
2. Retain the images present for **all 8 subjects**; drop any image missing for any subject and
   report the count kept.
3. Per-voxel z-score within subject (voxel counts differ between subjects; the instrument works
   in item space, so differing dimensionality is expected and fine).
4. `styxx.islands.survey(reps, k=20, n_null=1000, n_perm=1000, seed=343)` — defaults unchanged
   from the shipped release. No tuning, no k sweep, no subject dropped for any reason.

## Gates

```gates
{"gates": {"G0_cohort": {"metric": "n_members", "op": ">=", "value": 8},
           "G1_shared_frame": {"metric": "median_minus_null_p95", "op": ">", "value": 0.0},
           "G2_islands_present": {"metric": "bimodality_p", "op": "<=", "value": 0.05}},
 "outcomes": [{"when": {"G0_cohort": false}, "verdict": "INVALID__cohort_below_instrument_floor"},
              {"when": {"G0_cohort": true, "G1_shared_frame": false}, "verdict": "NO_SHARED_FRAME__subjects_do_not_co_align_and_island_is_the_wrong_word"},
              {"when": {"G0_cohort": true, "G1_shared_frame": true, "G2_islands_present": true}, "verdict": "HUMAN_ISLANDS_PRESENT__bimodal_cross_subject_alignment"},
              {"when": {"G0_cohort": true, "G1_shared_frame": true, "G2_islands_present": false}, "verdict": "HUMAN_SINGLE_CLIQUE__alignment_is_continuous"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

## Our prediction, stated in advance and against our own earlier position

**We expect `HUMAN_SINGLE_CLIQUE`.** The published human evidence already leans this way — n=175
with no performance exclusions shows an unbroken smear of per-subject accuracy; n=80 with no
exclusions spans chance to perfect with a *linear* predictor relationship; the "BCI illiteracy"
population dissolves when the calibration method changes. We registered the opposite prediction
this morning and have moved the prior down twice since. Predicting our own earlier claim will
fail is the point of writing it down.

**A `HUMAN_ISLANDS_PRESENT` result would therefore be a genuine surprise**, and would be
reported as one — replicated on a second cohort before any public claim, per the standing rule.

## Limits, fixed before the run

- **n = 8 is exactly the instrument's floor.** A gap screen on eight points has very little
  power; `HUMAN_SINGLE_CLIQUE` here is **weak evidence of absence** and the finding must say so
  in those words. NSD has no ninth subject to spare.
- Four of the eight NSD subjects completed fewer sessions than the others. **No subject is
  dropped** — that asymmetry is reported, because dropping the incomplete subjects is exactly the
  survivorship this program is criticizing.
- One dataset, one ROI, one modality, one alignment construction.

## Discipline

Result `h1a_result.json`; scored by `styxx.protocol` from this frozen block; certified + sealed
before commit. The full per-subject affinity table ships regardless of verdict. Raw betas are
not redistributed; the download route is public and cited above.
