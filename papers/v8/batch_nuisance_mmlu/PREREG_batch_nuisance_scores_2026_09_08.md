# PREREG — does an unreported knob move a benchmark score more than a reported improvement?

Fathom Lab · 2026-09-08 · **A preregistration, frozen before any scored run.** It makes no claim.
Every number in it is a threshold or a parameter chosen in advance, not a measurement. Written and
committed to disk before the harness was pointed at any benchmark item. Not sworn. The operator has
not signed it; it binds this session's analysis and nothing beyond it.

## Why

Two committed receipts from today (`papers/v8/probe_batch_invariance_2026_09_08/`,
`papers/v8/probe_batch_invariance_v2_2026_09_08/`) measured, on one RTX 4070 with greedy decoding
and left padding: a batch-1 rerun is bit-identical, but at batch 8 or 32 **every** sequence
log-probability changes, on both models tested, on 0 of 48 and 0 of 256 items keeping an identical
value. Those probes measured whether an output changed. They did not measure whether a **score**
changed.

Multiple-choice benchmarks are scored by comparing log-probabilities across the options. If every
log-probability moves, then any item whose top two options are close can flip, and the headline
accuracy moves with it. Batch size is not reported in evaluation configurations and is not part of
any run identity we are aware of. So the question is whether an unreported knob moves a published
metric by an amount comparable to the differences papers report as progress.

A prior-art check is running in parallel. **If it returns that this question is already answered,
this preregistration is withdrawn and the answer is cited instead.** Nothing here claims novelty;
novelty is not required for the measurement to be worth having on our own hardware.

## Subjects, battery, and the fixed recipe

| | |
|---|---|
| models | `meta-llama/Llama-3.2-1B-Instruct` and `google/gemma-2-2b-it`, from the local snapshots, offline |
| benchmark | MMLU (`cais/mmlu`), the `test` split of a fixed set of subjects chosen before any run: `high_school_mathematics`, `philosophy`, `professional_law`, `college_computer_science`, `moral_scenarios`. Chosen for a spread of difficulty and length, not for any observed result. |
| items | the first N=200 items per subject in dataset order after a fixed filter (four options present, non-empty question), capped at 1000 items total |
| scoring | the standard log-probability rule: for each option, the sum log-probability of the option's answer token(s) given the same prompt; the prediction is the argmax; accuracy is exact match against the gold label |
| held fixed | weights, revision, prompt text and template, option order, tokenizer, seed 7, `torch.manual_seed`, left padding, one machine, one driver, one process per variant |

## The variants (the nuisance)

Reference: **batch size 1, dataset order, bfloat16**. Then, varying only the nuisance:

1. repeat of the reference (determinism control)
2. batch 8, dataset order
3. batch 32, dataset order
4. batch 8, permuted (seed 11)
5. batch 32, permuted (seed 12)
6. batch 1, float16 (a precision arm, reported separately: this is a *declared* change, not a nuisance)

## Hypotheses, with directions fixed now

- **H1 (primary).** At least one nuisance variant (2–5) produces an accuracy different from the
  reference on at least one model. Directional: |Δ accuracy| > 0.
- **H2.** The number of items whose predicted option changes under a nuisance variant is greater
  than zero and less than 10 per cent of items.
- **H3.** Flipped items have smaller reference margins than unflipped items, where margin is the
  log-probability gap between the top two options. Directional: mean margin of flipped < mean
  margin of unflipped.
- **H4 (the one that matters).** The **spread** of accuracy across nuisance variants 1–5, defined
  as max minus min in percentage points, is at least **0.5 points** on at least one model.

0.5 points is the threshold because model comparisons and leaderboard positions are routinely
argued at that scale. It is chosen now, before any run, and is not adjusted afterwards.

## What would make this null, and it publishes either way

If the spread is below 0.5 points on both models, H4 fails and the honest conclusion is that batch
size is a real but negligible nuisance for this metric at this scale, which is a useful thing to
know and will be published as such. A null on H1 — no accuracy change at all under any nuisance
variant — would additionally contradict today's probes and would send us looking for a harness
defect before it sent us looking for a finding.

## Kill gates, checked before any hypothesis is read

1. **Determinism control.** If variant 1 (the reference repeat) differs from the reference on any
   item, the harness is not deterministic and the whole run is void. No hypothesis is evaluated.
2. **Sanity floor.** If reference accuracy is below 0.25 (chance for four options) on either model,
   the scoring is wrong and the run is void.
3. **Completeness.** If any variant fails to score every item, the run is void rather than reported
   over a surviving subset. A population defined by what survived is a defect this lab has
   catalogued before.

## Analysis, fixed now

Per model, report: accuracy for each variant; the spread across variants 1–5; the flip count and
flip rate versus the reference; the mean reference margin of flipped and unflipped items; and the
full per-item record. Compare the spread to the 0.5-point threshold. The precision arm is reported
beside the nuisance arms and is never merged into the spread, because a precision change is
declared and a batch size is not.

No statistical test is applied to H1–H4: every variant is a deterministic re-run of the same items
on the same machine, so there is no sampling distribution. The quantities are counts and
differences, and they are reported as such. This is stated now so that no test is chosen later.

## Exclusions, fixed now

An item is excluded only if it fails the fixed filter (fewer than four options, empty question, or
tokenization producing an empty option continuation). The exclusion list is recorded with reasons
and its size is reported beside every accuracy.

## What this measurement may never claim

- Not that published results are wrong. It measures one knob on two small models on one GPU.
- Not that the effect transfers to other models, harnesses, kernels, hardware, or benchmarks.
- Not that anyone was careless. Batch size is not reported because nobody established it mattered.
- Not novelty, unless the prior-art check returns that the question is open, and then only in the
  words that check licenses.

## Limits, known before the run

One GPU, one driver, one transformers version, two small models, one benchmark, one scoring rule,
one seed, one process per variant. Multiple-choice log-probability scoring is one of several
scoring conventions and results may not transfer to the others. The subject list was fixed in
advance but is not a random sample of MMLU.
