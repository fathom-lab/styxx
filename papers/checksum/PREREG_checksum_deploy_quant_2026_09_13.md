# PREREG — checksum on a deployed-scale model: bf16 vs NF4 vs int8, Qwen2.5-1.5B, GPU

**frozen 2026-09-13, before any run. seal: sha256 of this file anchored in the ledger before
`run_deploy_quant.py` starts. the RESULT reports every number below whatever it says.**

## question

does the checksum, on a model people actually deploy and a quantization people actually use,
(a) read the null pair as SAME once the serving floor is measured on a GPU, (b) read a real 4-bit
method as DRIFT with a magnitude in the band we predict, and (c) not read it as SAME — the one
outcome that would mean the canary set is too coarse to matter?

## design, fixed

- model: `Qwen/Qwen2.5-1.5B` (ungated; the family already in the disjoint-worlds banks).
- canaries: `styxx.checksum.CANARIES`, hash `71f8e5c9973c…` (the full hash is in every cert).
- arms: **A** bf16 loaded (run 1); **A′**, **A″** bf16 loaded again (runs 2, 3) — the null floor is
  the worst pairwise distance among A, A′, A″; **Q4** bitsandbytes NF4 (`bnb_4bit_quant_type="nf4"`,
  double-quant on, compute dtype bf16); **Q8** bitsandbytes LLM.int8; **R** random init, seed 343.
- probe: teacher-forced, greedy, no sampling; `torch.use_deterministic_algorithms` best effort.
- distance: `styxx.checksum.distance(..., n_boot=2000, seed=20260913, floor_nats=<measured>)`.
- sanity: first-token top-1 hits on the 48 canaries per arm.

## hypotheses and predicted bands (written before data)

- **H1 (floor).** The null floor on GPU is > 0 and ≤ 1e-3 nats/token. A vs A′ reads SAME.
- **H2 (NF4).** A vs Q4 reads DRIFT, mean |Δ log-prob| in **[0.01, 0.30] nats/token**, belief
  geometry r ≥ 0.95, top-1 loss ≤ 4 of 48.
- **H3 (int8).** A vs Q8 reads DRIFT with mean |Δ log-prob| below Q4's (int8 gentler than 4-bit).
- **H4 (far control).** A vs R: mean |Δ log-prob| > 5 nats/token, r < 0.3, top-1 ≤ 2 of 48.

## kill gates

- **K1.** Null floor > 1e-2 nats/token → the serving is not deterministic enough for this probe;
  the run is INCONCLUSIVE and says so; no other hypothesis is evaluated.
- **K2.** A vs Q4 reads SAME (upper bound below floor) → the canary set cannot see a deployed
  quantization; the instrument fails its purpose at this coverage; the RESULT is titled with the
  kill.
- **K3.** A vs Q4 > 1.0 nats/token or Q4 top-1 loss > 12 → the quantization pipeline, not the
  model, is suspect (compare against a published NF4 perplexity delta before believing it).
- **K4.** A vs R inside H2's band → the distance cannot tell a random model from a quantized one;
  the metric is broken.

## what is not claimed

A threshold for "meaningful" drift. Anything about API models. Anything beyond this one model and
these two methods. Nondeterministic (sampled) serving.

## outputs

`papers/checksum/deploy_quant_certs.json`, `deploy_quant_fingerprints.json`,
`deploy_quant_plates.png`, and a sworn `RESULT_checksum_deploy_quant_<date>.md` binding every
number above to those files at the commit that carries them.
