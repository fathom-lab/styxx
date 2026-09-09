# RESULT — the first re-run of a published styxx battery, and what it cost

Fathom Lab · 2026-09-09 · **A re-run, not a reproduction by anyone else.** Same box, same session,
same operator, same local snapshot as the verdict it re-runs. It is not a second party and nothing
below should be read as one. Not sworn.

The subject is the published verdict at `../first_verdict_2026_09_09`: `google/gemma-2-2b-it` at
revision `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`, a 64-prompt battery, greedy, 16 new tokens.
This is the first time any battery in this project has been run twice and the two runs compared.

Artifacts beside this document: `rerun_battery.py` (the driver), `rerun_stdout.txt` (its raw
output), `rerun_raw.json` (every per-item comparison, both timing passes, the environment) and
`rerun_raw_pass1.json` / `rerun_stdout_pass1.txt` (an earlier pass of the same driver before the
one-call and GPU-memory instrumentation was added). That earlier stdout ends in a `SystemExit: 0`
traceback: the driver's top-level handler caught `BaseException`, so a clean exit printed as a
failure. The comparison had already completed and been written; the handler now catches `Exception`
and the shipped output has no such trailer.

## The comparison

The reference is the canonical fingerprint
`sha256:73a09ffa3f1ee69a065f5d3aa28965348558d7d128b1b302fd6909324766a5c3`
(`fp_bf16/fingerprint-canonical-73a09ffa3f1e.json`), which records `batch_size=1`,
`item_order=canonical` in its `body.nuisance`.

| what was compared, per item | reproduced |
|---|---|
| `output_sha256` | **64 of 64** |
| `token_ids_sha256` | 64 of 64 |
| `n_generated` | 64 of 64 |
| `seq_logprob`, compared as an exact float | 64 of 64 |
| `topk` — up to 8 positions × top-5 ids and log-probs | 64 of 64 |

**No item differs.** The list of differing item ids is empty (`rerun_raw.json`,
`result.differing_item_ids`).

The Appendix A.3 exact-channel hash over all 64 item records comes back as the same value the cert
carries:

```
cert   d3372370c68804a099c6e0098038382da23e1a4d24383e129d0576ae312a18c9
re-run d3372370c68804a099c6e0098038382da23e1a4d24383e129d0576ae312a18c9
```

Three separate passes were run (two of the driver as shipped, one of the earlier version). All
three returned 64 of 64 on every field above. The bitwise agreement on `seq_logprob` and `topk` is
the stronger of the two statements: `output_sha256` can survive a small numeric perturbation that
does not change the argmax, and these did not perturb.

The `topk` result carries a known caveat from `../THE_BOUNDARY_2026_09_09.md`: the format records
at most 8 positions per item while items generate up to 16 tokens, so "topk reproduces" covers the
recorded positions and says nothing about the emitted tokens beyond them.

## What it cost, which has never been measured

Wall clock on this box, from `rerun_raw.json` `timing`, `one_call_pass` and `gpu_memory`:

| | seconds |
|---|---|
| **the 64 items, one runner call at batch 1** | **49.9** (0.78 per item) |
| the 64 items, 64 separate runner calls | 51.7 (0.81 per item) |
| model load from the local snapshot | 19.5 |
| recomputing the four Appendix A.2 snapshot hashes | 5.1 |
| **end to end, hashes + load + one battery pass** | **74.5** |

Per item, from the 64 separately-timed calls: min 0.25, median 0.59, mean 0.81, max 1.58. The
spread is decoding length, not per-call overhead: against the certificate's own `n_generated`
(3 to 16 tokens, mean 8.97) the per-item seconds correlate at **r = 0.943**, and the 13 items that
emitted 3 tokens averaged 0.40s against 1.31s for the 21 that ran to the 16-token cap. There is no
warm-up term either — the first item cost 1.17s and the mean excluding it is 0.80s.

Across the three passes the 64-item generate time was 62.8s, 58.5s and 51.7s, in that order, all on
an otherwise idle GPU. So **the number to plan against is about a minute of compute per battery
pass, and about 75 seconds end to end**, with a ±20% spread between passes on this laptop card that
this document does not explain.

GPU memory, which decides what card a challenger needs: 4,987 MiB allocated after load, peak 5,035
MiB allocated / 5,246 MiB reserved. **The transient cost of everything above the weights is about
48 MiB.** `logits_to_keep` was not passed and was not needed, because at batch 1 the
`(1, seq, 256k)` logits tensor is small; that is a batch-size problem, and no claim about
`logits_to_keep` is relied on here or made here.

## The cost gap this run does not explain

`../first_verdict_2026_09_09/transcript_stage4.json` records the fp16 fingerprint command — one
run, 64 items, batch 1, including model load — at **53.8 seconds**, and the five-run bf16
fingerprint command at 127.0 seconds. The comparable number here is 69.4s (19.5 load + 49.9
generate), about 1.3× slower, and the published five-run figure implies a per-item cost well under
what was measured today at batch 1.

The digests are identical, so this is a cost discrepancy and not a result discrepancy. It is
reported rather than explained: the published transcript times a CLI subprocess and this driver
times the runner inside its own process, the two do not decompose the same way, and no measurement
here isolates the difference.

## What was checked before the run, out of the published bytes

Every input came from the published artifact; nothing was retyped.

- **The prompts** came from log entry 0, the battery pool cert
  `sha256:39e7f293e034ca10191cd959f46049762c46a8a093664b575d5b3158c93d23c1`, which carries
  `prompt_text` for all 64 items. The battery is recoverable from the published log alone — no
  separate `items.json` was needed, and none is published.
- **The run order** came from the canonical cert's own `body.items` order. It was checked two ways:
  it equals A.3 order (`item_id` ascending), and `sha256(JCS(order))` recomputes to
  `0e25a46ee7ff6075f85eebb1bd1ce697d1fda9e202bd3a40613a29b5b7470c3d`, the value in the cert's
  `nuisance.item_order_sha256`.
- **The recipe** came from the cert's own signed `recipe` block — decoding
  (`temperature 0, max_new_tokens 16, seed 7, batch_size 1, padding_side left`) and materials
  (the chat template) — not from `recipe.json` sitting beside it, so what ran is what the signature
  covers.
- **The weights** were checked, not assumed: all four Appendix A.2 hashes were recomputed from the
  snapshot on disk and each equals the cert's, and `TransformersRunner.subject()` reported
  `precision: bf16` for the dtype it loaded.
- **The environment** the runner observed matches the cert's on every leaf, and `nvidia-smi`
  reported driver `596.08` at run time. Python 3.12.10, torch 2.5.1+cu121, transformers 4.57.3.

`styxx/v8/runner_hf.py` was imported and called unmodified. Because the canonical run is batch 1,
the runner's own loop already issues one forward pass per item, so calling it 64 times with one
item each issues the same forward passes as calling it once with 64. That is asserted in the
driver and then tested: the one-call pass produced item records **equal to the per-item loop's**
(`one_call_pass.agrees_with_per_item_loop = true`), so the per-item timings decompose a real batch-1
pass and are not an artifact of the measurement.

## What this establishes

- **The mechanism runs end to end from published bytes.** A reader holding only the published
  artifact and the model can reconstruct the battery, the order, the recipe and the subject, run it,
  and compare — with no access to the original working directory. That path had never been walked.
- **This machine reproduces its own reference run exactly, across processes and hours.** The
  published floor already recorded that same-batch pairs had distance 0 among the five runs inside
  one command; this extends it to a run from a separate process, a separate driver and a separate
  code path, on the same day.
- **A battery pass costs about a minute of GPU time and about 5 GB of VRAM.** On the affordability
  question in `../THE_BOUNDARY_2026_09_09.md` — "a challenge that costs a full battery re-run is a
  challenge nobody performs" — the measured cost of this battery on this model is not the obstacle.
  What a challenge additionally costs (obtaining the weights, the floor runs a plan demands, the
  log and key handling) is not measured here.

## What this does not establish

- **It is not a reproduction by a second party.** Same machine, same session, same operator, same
  snapshot, same driver author. The published verdict remains a verdict nobody outside this lab has
  re-run, and the reproduction count for it remains 0.
- **It introduces no byte the issuer did not write.** Per the design consequence in
  `../THE_BOUNDARY_2026_09_09.md`, that is the whole content of the challenge mechanism, and this
  is not it. Nothing here was appended to the published log and no challenge cert was minted.
- **It says nothing about a different card, driver, torch build or transformers version.** The
  published floor names exactly those in `not_covered`, and this run varied none of them. Agreement
  here is agreement inside one machine's covered conditions.
- **It does not re-derive the verdict.** Only the canonical bf16 fingerprint was re-run. The four
  other floor runs, the fp16 fingerprint, the floor arithmetic and the `exceeds_floor` verdict were
  not recomputed, so nothing here confirms or disputes the floor values or the verdict.
- **A reproduction that succeeds bounds nothing about detection.** 64 of 64 agreeing is what a
  correct re-run and a re-run of the wrong thing that happened to agree would both look like; the
  guard against the second is the A.2 hash check above, which is a check on this machine's disk and
  not on the world.

## Limits

One model, one battery of 64 prompts, one machine, one operator, three passes within ten minutes of
each other. The timing spread across those three passes is 20% and is unexplained, so the cost
figures should be read as an order of magnitude and not as a benchmark. The gap against the
published transcript's 53.8s is also unexplained. No part of this was reviewed by anyone outside
the session that produced it.
