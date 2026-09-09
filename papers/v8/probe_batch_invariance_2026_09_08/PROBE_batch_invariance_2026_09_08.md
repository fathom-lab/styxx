# PROBE — is greedy decoding invariant to batch size and item order on this box? (δ2 of the v8 draft)

Fathom Lab · 2026-09-08 · **A probe, not a result.** One model, one GPU, one battery of 48 prompts,
one session. Nothing here is a claim about models in general; every number below is read from
`receipt.json` (sha256 `1ef0d4a67c7f3e07b03de3bdd0fbdd8b21e71edad79825888b9d09282b3afa04`), produced by
`probe_batch_invariance.py` (sha256 `aa9956eacc3370cc430fb918b3047b5f63f0cf65ffeebabfe862e8be22499c98`),
stdout in `probe_stdout.txt`. Not sworn; not in any log.

## Why it was run

The v8 draft (§4.2, §4.4, §5.1) treats batch size and item order as *nuisance* (δ2): any item whose
greedy output changes under δ2 is excluded from the canary battery, and the noise floor is built from
runs that vary "only nuisance factors". It treats a precision change (δ1) as *signal*. Nobody had
measured either on this box.

## Setup

| field | value |
|---|---|
| model | `Qwen/Qwen2.5-0.5B-Instruct`, snapshot `7ae557604adf67be50417f59c2c2f167def9a775`, loaded from the local HF cache |
| runtime | torch 2.5.1+cu121, transformers 4.57.3, NVIDIA GeForce RTX 4070 Laptop GPU (8 GB) |
| decoding | greedy, `max_new_tokens=16`, chat template with `add_generation_prompt=True`, left padding, `torch.manual_seed(7)` |
| battery | 48 prompts: 12 factual, 12 arithmetic, 10 instruction, 8 format, 6 refusal-boundary |
| reference | batch size 1, natural order, bfloat16 |
| variants | repeat of the reference; batch 8 natural; batch 32 natural; batch 8 permuted (seed 11); batch 32 permuted (seed 12); batch 1 natural in float16 |

## Numbers (from `receipt.json`, `variants`)

| variant | flipped vs reference (of 48) | mean \|Δ seq_logprob\| | max \|Δ seq_logprob\| | mean \|Δ top-1 lp\| at first token |
|---|---|---|---|---|
| repeat, batch 1, bf16 | 0 | 0.0000 | 0.0000 | 0.00000 |
| batch 8, natural, bf16 | 6 | 0.2452 | 5.2222 | 0.02987 |
| batch 32, natural, bf16 | 5 | 0.2070 | 5.2147 | 0.02752 |
| batch 8, permuted, bf16 | 6 | 0.1645 | 1.7291 | 0.02772 |
| batch 32, permuted, bf16 | 6 | 0.2126 | 5.2351 | 0.02656 |
| batch 1, natural, **fp16** (a δ1-type precision change) | 5 | 0.1027 | 0.9199 | 0.02740 |

Derived from the per-item records (`items[*].variants`):

- The batch-1 repeat reproduces every token id and every sequence log-probability of the reference on 48 of 48 items.
- In every batched variant, 0 of 48 items reproduce the reference sequence log-probability exactly, including the items whose token ids did not change.
- The union of items flipped by any of the four batch/order variants is 9 of 48: indices 3, 9, 22, 30, 34, 38, 39, 41, 42 (format 4, factual 2, refusal-boundary 1, arithmetic 1, instruction 1).
- The fp16 precision change flipped 5 items: 3, 9, 22, 39, 41. **All five are in the batch/order union.** The precision change flipped no item that batching did not also flip.
- Item 22 (arithmetic, reference output `45 * 11 = 495`) reads `595` under batch 8 and batch 32 natural order, and `555` under fp16.

## What the draft's rules do to this battery, mechanically

Under §4.4 step 1 as written (exclude every item with `flip2 > 0`), the 9 batch-sensitive items are
excluded. The 5 precision-sensitive items are among them, so every surviving item has `flip1 = 0` under
the fp16 perturbation: the selected battery would carry no item that this precision change moves. Under
§5.1 as written (floor from runs that vary batch size and order), the exact-channel floor for this battery
is 5–6 flips of 48 per run, the same count the precision change produces at batch 1. The two are not
separable by count on this battery, on this box.

Under a recipe that pins batch size to 1 (not in the §2.3 recipe as drafted), the reference run repeats
bit-for-bit, the exact-channel floor is 0 of 48 and the seqlp floor is 0.0000, and the fp16 change is
5 flips above that floor.

## Limits

One model of 0.5B parameters, one GPU, one driver, one battery of 48, `max_new_tokens=16`, one
permutation seed per batch size, one precision pair (bf16 → fp16, both 16-bit; no integer quantization was
run). Nothing here transfers to another model, box, or battery without being re-run there. The prompts
were written by the harness author minutes before the run and are not a released battery.
