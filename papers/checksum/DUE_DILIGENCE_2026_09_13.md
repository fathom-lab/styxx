# DUE DILIGENCE — the plate and the checksum, reviewed adversarially before anyone calls them first

fathom lab · 2026-09-13 · written by the same hand that built them, which is why it has to be blunt

## 1. prior art we must name before the word "first" is used

**Hash-to-picture for human comparison is old.** OpenSSH has rendered host-key fingerprints as
"randomart" (the drunken-bishop walk) since 2008 so people can compare keys by eye; identicons
(2007) do the same for avatars; several "hash visualizer" schemes exist. The hash plate's only
novelties are narrow: the figure is the nodal set of physical plate modes (Chladni, 1787), it is
rendered as sand, and it is applied to sworn receipts in a chat as a reproduction check. We claim
the use, not the idea. Any "first" sentence about the hash plate is wrong.

**Comparing models by log-probabilities on fixed prompts is standard.** KL/log-prob comparison of
two models' outputs on a fixed set is how quantization papers report damage, how "How is
ChatGPT's behavior changing over time?" (Chen, Zaharia, Zou, 2023) documented drift, and how
model-fingerprinting work identifies checkpoints. The checksum's contribution is the packaging —
a hashed canary set carried in the cert, a measured null floor, a degeneracy guard, a bootstrap
resolution, a clock-free digest, a sworn result, a plate, and an anchor — not the measurement.
Before any "first" claim about the checksum, the prior-art survey runs under a frozen procedure,
the way `papers/sworn/PROCEDURE_prior_art_survey` did for sworn.

**Representational similarity is standard.** RDMs and their correlation are the RSA toolkit
(Kriegeskorte, 2008). The geometry plate is a fixed low-pass view of an RDM; it claims no
novelty as a measurement.

## 2. the checksum — weaknesses found, and what was done

| found | consequence | done |
|---|---|---|
| the SAME/DRIFT verdict used a fixed resolution (1e-4 nats) | on GPUs, batched or sampled serving the null pair is not 0; a same-weights pair would read DRIFT | `null_floor()` measures the worst same-weights distance in situ; `distance(..., floor_nats=...)` grades against it; test pins that a floor as large as the drift turns DRIFT into INCONCLUSIVE, never keeps it |
| mean log-prob per token depends on the tokenizer | comparing two models with different tokenizers would report a meaningless number | `tokenizer_id` on every fingerprint; `distance()` refuses across tokenizers (the belief-geometry rdm remains comparable); test pins the refusal |
| the cert digest covered timestamps | two identical re-runs produced different digests, which defeats reproduction and anchoring | the digest now covers the comparison only; `created` rides outside it; test pins that changing the clock does not change the digest |
| `rdm_sha256` hashed the float64 array, not the rounded values written to the file | a stranger re-hashing the json could never match | the hash now covers the written (rounded) values |

## 3. the checksum — weaknesses that remain, stated

- **Canary overfitting.** The 48 canaries are public. A model could be tuned to hold its
  behavior on them while changing elsewhere. This is a drift detector, not a tamper-proof seal.
  The fix is a committed-but-private canary set (hash published, items revealed after each
  epoch) and is not built.
- **Coverage.** 48 short English trivia/logic/code items. Drift outside that surface is invisible
  to this set by construction. The set will be widened under preregistration, and the hash will
  change when it does — which is why the hash is in every cert.
- **The threshold.** The instrument reports magnitude and resolution. "Meaningful" is a
  preregistered decision per use, and no such preregistration exists yet.
- **Nondeterministic serving.** The null floor is measured, not modeled. On an API the floor
  must be re-measured whenever the provider's serving changes — which is itself drift of a kind
  this instrument cannot separate from weight drift. Stated, not solved.
- **The int8 result.** `quantize_dynamic` quantized the tied `lm_head` as well as the blocks;
  per-tensor int8 without calibration on 135M is the crudest case. The result says what it says
  about that case and nothing about GPTQ/AWQ/NF4 on deployed models.

## 4. the four-model geometry claim — a control the post did not have

The pinned post says four models share one concept geometry, with a shuffled-item control at
r ≈ 0.00. A hostile reader would ask: wouldn't any word-embedding table agree with those banks,
since animals cluster with animals? So we ran a static, context-free baseline — the mean input
token embedding of each of the 462 concept strings from a fifth model (SmolLM2-135M), no context,
no layers — and correlated its RDM with the four contextual banks:

| static token embeddings vs | llama-3b | llama-1b | gemma-2b | qwen-1.5b |
|---|---|---|---|---|
| r | 0.308 | 0.339 | 0.304 | 0.285 |

against 0.87–0.96 between the contextual banks themselves and ≈0.00 for random vectors. A
context-free embedding table recovers about a third of the agreement; the rest is what the
models compute in context. The claim survives the control it should have had. Caveat: the
baseline is a different model's table and averages sub-word tokens; the fairer control — each
model's own input embeddings — needs the weights and was not run here.

## 5. the plates — what a hostile reviewer can still say

- The geometry plate's low-pass view (K=8 of a 462-item DCT) mostly reflects category-block
  structure in the canonical item order. That is what the picture shows; the finer agreement is
  the printed r. Never quote the picture without the number.
- Both plate renderers are frozen mappings. Changing either changes every published picture.
  Version them (`plate/v1`, `geoplate/v1`) in the file caption before the next release.
- A plate is not a proof. Two plates that look the same are two hashes that are the same, which is
  a claim about bytes, and the bytes are what a stranger must check.

## 6. what would change our minds

- A same-weights pair reading DRIFT after the floor is measured → the instrument is broken.
- A 1B–3B model under a real 4-bit method reading SAME → the canary set is too coarse to matter.
- The sealed pmnist_untied run recovering FREE's accuracy with the rotation clamped → the
  frequency arc's flagship causal story is dead, and this document will say so first.
