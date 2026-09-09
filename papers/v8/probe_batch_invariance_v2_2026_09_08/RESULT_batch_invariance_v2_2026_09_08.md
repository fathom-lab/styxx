# RESULT — the batch-invariance probe replicates in part, and its headline does not

Fathom Lab · 2026-09-08 · **A result, not a claim about models in general.** Two models, one GPU,
two batteries, one session. Every number below is read from `receipt.json`
(sha256 `30f2011ed9e3118f9cda1278a01ac31402ed7afa5eb2fc40590d1c48cee35c5b`) produced by
`probe_batch_invariance_v2.py` (sha256 `310c5ef3a5b39b394fae2d82c5b8ee482ba6cb6cd63ef53142ce66fc38fee121`),
stdout in `stdout.txt`. Predecessor: `papers/v8/probe_batch_invariance_2026_09_08/`
(receipt sha256 `1ef0d4a67c7f3e07b03de3bdd0fbdd8b21e71edad79825888b9d09282b3afa04`). Not sworn; not in
any log. No variant was skipped; `receipt.json` carries an empty `skipped_variants`.

## What was asked

Probe v1 ran on Qwen2.5-0.5B-Instruct with 48 prompts and reported three things. This probe re-ran
the same measurement on a different model family, a different tokenizer, and a battery five times
larger, to see which of the three survive.

| | probe v1 | probe v2 |
|---|---|---|
| model | Qwen2.5-0.5B-Instruct `7ae55760` | google/gemma-2-2b-it `299a8560` |
| battery | 48 prompts | 256 prompts (60 arithmetic, 60 factual, 50 format, 50 instruction, 36 refusal-boundary) |
| decoding | greedy, 16 new tokens, left padding, bf16 reference | identical |
| variants | batch-1 repeat, batch 8 and 32 in natural and permuted order, fp16 at batch 1 | identical |
| GPU | RTX 4070 Laptop, 8 GB | identical |

## Numbers

| variant | flipped vs reference | identical sequence log-probability | mean abs delta | max abs delta |
|---|---|---|---|---|
| repeat, batch 1, bf16 | 0 of 256 | 256 of 256 | 0.00000 | 0.0000 |
| batch 8, natural | 9 | 0 | 0.05047 | 4.0518 |
| batch 32, natural | 8 | 0 | 0.03276 | 0.7445 |
| batch 8, permuted (seed 11) | 9 | 0 | 0.04611 | 3.8880 |
| batch 32, permuted (seed 12) | 13 | 0 | 0.04261 | 1.3641 |
| batch 1, fp16 | 17 | 0 | 0.07439 | 4.2866 |

Derived: the union of items flipped by any batch or order variant is **20 of 256**; the precision
change flips **17**; **10** of those 17 are inside the nuisance union and **7 are not**.

## What replicated

1. **A batch-1 rerun is bit-identical.** 256 of 256 items reproduce their token ids and their
   sequence log-probabilities exactly, as 48 of 48 did in v1.
2. **Batching perturbs every log-probability.** In every batched variant, 0 of 256 items keep an
   identical sequence log-probability, even among the items whose token ids do not change. v1
   reported the same 0-of-48.
3. **Batching flips a small fraction of outputs.** 8 to 13 of 256 per variant here (3.1 to 5.1 per
   cent); 5 to 6 of 48 in v1 (10.4 to 12.5 per cent). Both non-zero, both small, and the rates are
   not equal.

## What did not replicate, and is hereby withdrawn

Probe v1's note observed that every one of its five precision-induced flips was also flipped by
batching, and drew from it the consequence that the draft spec's exclusion rule (§4.4 step 1,
"exclude every item with flip2 > 0") would strip every precision-sensitive item from the battery.

**On gemma-2-2b-it that containment is false.** Seven of the 17 precision flips are in no batch or
order variant: `arith-052`, `bound-011`, `bound-015`, `bound-033`, `instr-029`, `instr-033`,
`instr-039` — one arithmetic, three refusal-boundary, three instruction items. Those seven survive
the §4.4 exclusion and remain sensitive to the precision change, which is exactly what the
selection rule is supposed to keep.

The withdrawn sentence is therefore: *"the selected canaries carry zero sensitivity to the
precision change."* It was true of one battery on one model and does not generalise. What the two
probes jointly support is weaker and still useful: **the exclusion rule removes a majority of
precision-sensitive items — 5 of 5 in v1, 10 of 17 here — so a battery selected under it has less
precision sensitivity than the pool it came from, by an amount that must be measured per subject
rather than assumed.** That is the sensitivity receipt the amended draft already requires at §5.3,
and this probe is the reason the requirement is not optional.

## What it means for the draft

- §2.3's inclusion of `batch_size` and `padding_side` in the recipe stands, strengthened: on a
  second model, batching still moves every log-probability and some outputs.
- §5.6's sentence about the probe must name both models and both rates, not v1's alone.
- §4.1's hypothesis H-canary is unchanged in status: still a hypothesis, now with one measurement
  against its strongest form.
- §5.3's sensitivity receipt moves from "good practice" to "the thing that stops us stating a
  containment that does not hold". A battery's precision sensitivity after exclusion is a number
  to be measured on the subject, and this probe measured it twice and got two answers: 0 of 5, and
  7 of 17.

## Limits

Two models, one GPU, one driver, one session, two battery sizes, one permutation seed per batch
size, 16 new tokens, and one precision pair (bf16 to fp16, both 16-bit; no integer quantisation was
run). The v2 battery is synthetic, written by the probe script minutes before the run, and its item
kinds are not a benchmark. Neither probe measures whether an answer is correct, only whether it
changed. Nothing here transfers to another model, box or battery without being re-run there.
