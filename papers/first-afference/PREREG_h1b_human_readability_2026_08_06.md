# PREREG — H1b: per-subject *readability* — does the switch turn continuous alignment into bimodal accuracy?

Fathom Lab · 2026-08-06 · frozen before the scored run. H1a measured cross-subject **alignment**
and found it continuous (`HUMAN_SINGLE_CLIQUE`, gap-screen p 0.0779). Its own stated limit was
that H1 is about **decoding accuracy**, not alignment — and this lab's b46 result says the two
are joined by a **switch**: legibility stays flat across most of the frame rotation and turns
nearly vertical only near alignment. **A continuous alignment spread can therefore still produce
a bimodal readability distribution, if the knee falls inside that spread.** H1b measures the
variable H1 actually names, on the same eight subjects, using data already extracted.

## The measurement: can a group-trained template read this subject?

For each held-out subject S, using only the other seven to build the template:

1. Split the 907 shared images into **train** (the profile basis) and **test**, 50/50, fixed seed.
2. For subject S, each test image's **profile** is its vector of correlations to every train
   image, computed within S. Profiles live in item space, so subjects with different voxel
   counts are directly comparable and **no voxel alignment or fitting is performed**.
3. The **group template** is the same profile computed from the mean of the other seven
   subjects' item-similarity structure — S contributes nothing to its own template.
4. Match S's test profiles against the template's test profiles by Hungarian assignment.
   `readability(S)` = fraction of test images correctly identified. Chance = 1 / n_test.

This is leave-one-subject-out identification: exactly "does a model trained on other people
read this person," which is what every cross-subject decoding claim rests on.

## Gates

```gates
{"gates": {"G0_cohort": {"metric": "n_subjects", "op": ">=", "value": 8},
           "G1_group_reads_above_chance": {"metric": "median_readability_over_chance", "op": ">=", "value": 5.0},
           "G2_bimodal_readability": {"metric": "bimodality_p_readability", "op": "<=", "value": 0.05}},
 "outcomes": [{"when": {"G0_cohort": false}, "verdict": "INVALID__cohort_below_floor"},
              {"when": {"G0_cohort": true, "G1_group_reads_above_chance": false}, "verdict": "INVALID__group_template_does_not_read_anyone"},
              {"when": {"G0_cohort": true, "G1_group_reads_above_chance": true, "G2_bimodal_readability": true}, "verdict": "READABILITY_IS_BIMODAL__the_switch_shows_in_humans"},
              {"when": {"G0_cohort": true, "G1_group_reads_above_chance": true, "G2_bimodal_readability": false}, "verdict": "READABILITY_IS_CONTINUOUS__no_switch_at_this_cohort"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

G1 is a positive control at 5× chance — the same multiple the model-side matrix used to call a
read real. If a group template cannot read anybody, the shape of the distribution says nothing
and the run is INVALID.

## Reported ungated

The correlation between each subject's H1a **alignment** and its H1b **readability**. If b46's
switch transfers to humans, that relation should be steep rather than proportional. With eight
points no fit is licensed, so this is a scatter reported for the record and for successors at
larger n — not a claim, and explicitly not gated.

## Our prediction, stated in advance

**We expect `READABILITY_IS_CONTINUOUS`.** Three independent negatives already stand against
island structure, and the published human accuracy distributions are described as unbroken
smears. But we are running it because *this is the variable H1 names* and because our own switch
result makes a bimodal answer genuinely possible — a continuous cause can have a discontinuous
effect. A `READABILITY_IS_BIMODAL` verdict would be the first evidence for H1 all day and would
be treated as a surprise requiring replication at larger n before any public claim.

## Limits, fixed before the run

- **n = 8 remains exactly the instrument's floor.** A gap screen on eight points has very little
  power; `READABILITY_IS_CONTINUOUS` is weak evidence of absence and the finding must say so.
- Identification accuracy is one readability measure among many; a trained decoder could rank
  subjects differently. This one is chosen because it requires no fitting and therefore cannot
  be tuned toward either verdict.
- Same dataset, ROI, modality and seed as H1a; these are not independent runs.

## Discipline

CPU, seconds, no new download — the extracted shared-image matrices from H1a are reused. Result
`h1b_result.json`; scored by `styxx.protocol`; certified + sealed before commit. Per-subject
readability ships regardless of verdict.
