# RESULT — an unreported knob changes 8% of MMLU answers and moves the score by half a point

Fathom Lab · 2026-09-09 · **A result, not a claim about the literature.** Two models, one GPU, 900
MMLU items, one session. Preregistered at `PREREG_batch_nuisance_scores_2026_09_08.md`, written to
disk before the harness was pointed at a benchmark item; the hypotheses, the 0.5 percentage-point
threshold, the kill gates and the analysis were fixed there and are not adjusted here. Numbers are
read from `receipt.json` (sha256 `b351bfbcf847df5d426affc6212aa56db6bc8155d02f201b5f9464b65b5c30b4`)
produced by `run_batch_nuisance.py` (sha256 `ee63ba2e44670522a1cda00fb5eee9ce9ace00461e8a177d6ef53145bf0e0194`),
analysed by `analyze.py` (sha256 `7961216f81d709284b5e4a6c480b45c9aefe1119a014f311da76fc3f3cd5e241`),
stdout in `stdout.txt`. 2465 seconds of GPU time. Not sworn; not in any log.

**This is not a novel question.** A prior-art check ran in parallel and is summarised in §6. Batch
size and precision were already known to change generated outputs, already measured against scored
generative benchmarks, and the "nuisance exceeds the claimed improvement" comparison has already
been made for seeds and hardware. What was not already done is this benchmark class, and §6 says
exactly which part is new and which is not.

## 1. What was varied

Held fixed: weights, revision, prompt text and template, option order, tokenizer, seed, left
padding, one machine, one driver, one process per variant. Varied: **inference batch size and the
order items are batched in** — factors that evaluation configurations do not report. Precision is
reported beside them as a *declared* change and never enters the nuisance spread.

Scoring is the common MMLU convention: the prompt ends with `Answer:`, the four options are read
off the log-softmax at the final position for the tokens ` A`, ` B`, ` C`, ` D`, and the prediction
is the argmax. One forward pass per question, so the score is a direct function of the
final-position logits.

## 2. Kill gates

All three passed on both models. The batch-1 repeat reproduced the reference on 900 of 900 items
with identical margins, so the harness is deterministic; reference accuracy is above chance; every
variant scored every item.

## 3. Numbers

| | Llama-3.2-1B-Instruct | gemma-2-2b-it |
|---|---|---|
| reference accuracy (batch 1, bf16) | 0.3900 (351/900) | 0.4200 (378/900) |
| batch 8 | 0.3900 | 0.4178 |
| batch 32 | 0.3956 | 0.4244 |
| batch 8, permuted | 0.3956 | 0.4189 |
| batch 32, permuted | 0.3956 | 0.4222 |
| **accuracy spread across nuisance variants** | **0.556 pp** | **0.667 pp** |
| items whose answer changed (union of variants) | **72 of 900 (8.00%)** | 20 of 900 (2.22%) |
| items that changed in all four variants | 13 | 0 |
| mean reference margin, flipped items | 0.082 | 0.116 |
| mean reference margin, unflipped items | 1.058 | 1.986 |
| precision arm (fp16, batch 1), declared not nuisance | 0.3967, 38 flips | 0.4233, 14 flips |

Every preregistered hypothesis passed on both models: accuracy differs under a nuisance variant
(H1); the flip count is above zero and below 10% (H2); flipped items have smaller margins than
unflipped ones (H3); the spread meets the 0.5-point threshold (H4).

## 4. The finding that was not preregistered

The threshold being crossed is the least interesting thing here. The following comparison was added
on the prior-art report's recommendation, **after the run**, and is labelled post-hoc throughout.

At n = 900 the binomial standard error of this metric is about 1.63 points. The largest net
nuisance movement is 0.556 points for Llama and 0.444 for gemma — roughly **0.3 standard errors**,
comfortably inside the benchmark's own precision. But the churn underneath is far larger than the
movement on top:

- Llama: **72 items changed their answer; the headline moved by 5 items' worth. A cancellation
  ratio of 14.4 to 1.**
- gemma: 20 items changed; the headline moved by 4 items' worth. 5.0 to 1.

So the aggregate metric largely self-averages the perturbation. Both readings are true at once and
neither should be quoted without the other:

1. **If you report an aggregate score**, batch size moves it by roughly half a point — below one
   standard error, and comparable to the size of improvements that are argued over.
2. **If you look at individual items**, 8% of one model's answers changed under a knob nobody
   records. That is invisible in the aggregate and it is the number that matters for per-item error
   analysis, for item-level agreement between two runs, for auditing named behaviours, and for any
   evaluation whose unit is the item rather than the mean.

H3 gives the mechanism cleanly and in the direction predicted before the run: flipped items have
reference margins roughly 12 to 24 times smaller than unflipped ones. The close calls move, and
which way they fall depends on what else was in the batch.

## 5. What this does not show

Not that published results are wrong. Not that this transfers to other models, harnesses, kernels,
hardware, or benchmarks — one GPU, one driver, one transformers version, two small models, one
benchmark, one scoring rule, one seed. Not that batch size is the largest unreported nuisance;
published work puts prompt format and backend choice far higher. Nothing here measures correctness
of an answer, only whether it changed. The subject list was fixed in advance but is not a random
sample of MMLU, and one of the five subjects contributed 100 items rather than 200 because that is
the size of its test split.

## 6. Prior art, and what of this is actually new

A prior-art check with web access ran in parallel with the experiment. Its verdict: **partially
answered**, and closer to answered than we would like.

- That batch size and precision change generated outputs at fixed everything is established, with
  the mechanism named as a lack of batch invariance in normalisation, matmul and attention
  kernels — Thinking Machines Lab, September 2025 — and batch-invariant kernels have since shipped
  in vLLM and SGLang. Anyone who cares can set a flag and drive this to zero.
- That they change a **scored** benchmark metric is established for generative reasoning
  benchmarks: Yuan et al., arXiv 2506.09501, report up to 9 points of accuracy variation across a
  grid of GPU type, GPU count and batch size, with float32 driving it to zero.
- That such movement can exceed claimed improvements is established: Hochlehnert et al., COLM 2025,
  for seeds and hardware; Pape et al., arXiv 2605.19537, 16.6 points from inference backend alone.

What was not already done, and is what this result contributes: **the log-likelihood-scored
multiple-choice case**. Every published score-delta we found is on free-generation reasoning sets
with small item counts, where one question is worth several points. Nobody had measured what batch
size does to MMLU-style scoring, where n is large and the metric is an argmax over per-option
log-probabilities. The answer is the cancellation result of §4, which the prior-art report predicted
as the likely and publishable outcome before the numbers existed: aggregate multiple-choice metrics
substantially self-average this noise, while the per-item churn does not average at all.

One further point, which is the prior-art report's and not ours: in common practice batch size is
not merely unreported but **uncontrolled**, because the widely used default sets it from free GPU
memory at run time. We did not test that condition and it is the obvious next arm.

## 7. Limits

Stated in the preregistration before the run and unchanged: one GPU, one driver, one transformers
version, two small models from different families, one benchmark, one scoring convention, one seed,
one process per variant, and one permutation seed per batch size. Multiple-choice log-probability
scoring is one convention among several and these results may not transfer to the others. No
statistical test is applied: every variant is a deterministic re-run of the same items on the same
machine, so there is no sampling distribution, and the quantities are counts and differences.
