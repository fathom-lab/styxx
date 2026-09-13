# RESULT — checksum v0: the null pair reads SAME at exactly zero, int8 reads DRIFT, random reads far — one small model, cpu, 2026-09-13

**status: instrument smoke on one 135M model. not a claim about quantization in general; not a promotion.**
This document is sworn: every figure below is bound to `papers/checksum/smollm_quant_certs.json`
or `papers/checksum/smollm_quant_fingerprints.json` at the commit the receipt names. Rebuild both
with `python papers/checksum/run_smollm_quant.py` (cpu, about a minute).

## what was run

`styxx.checksum` fingerprints a model on a fixed, hashed canary set in the continuous quantity the
model exposes — teacher-forced log-probabilities — and compares two fingerprints with a bootstrap
resolution, a degeneracy guard, and a cert. <sworn r="path:papers/checksum/smollm_quant_certs.json#/sanity/n_canaries" k="numeric">The canary set has 48 items</sworn> and its hash is in every cert. Four
fingerprints of HuggingFaceTB/SmolLM2-135M on cpu, deterministic: float32 loaded once (A), float32
loaded again (A′), int8 dynamic quantization of every nn.Linear (Q, per-tensor, no calibration —
the crudest quantization there is), and the same architecture with random weights (R).

## what it read

- **The null pair.** <sworn r="path:papers/checksum/smollm_quant_certs.json#/reloaded/distance/mean_abs_nats" k="numeric">A vs A′: mean |Δ log-prob| = 0 nats per token</sworn>, <sworn r="path:papers/checksum/smollm_quant_certs.json#/reloaded/distance/verdict" k="quote">verdict `SAME`</sworn>, belief-geometry agreement <sworn r="path:papers/checksum/smollm_quant_certs.json#/reloaded/distance/rdm_r" k="numeric">1.0</sworn>. The instrument does not invent drift.
- **int8.** <sworn r="path:papers/checksum/smollm_quant_certs.json#/int8/distance/mean_abs_nats" k="numeric">A vs Q: 1.51 nats per token</sworn>, 95% bootstrap interval <sworn r="path:papers/checksum/smollm_quant_certs.json#/int8/distance/ci_mean_abs/0" k="numeric">from 1.17</sworn> <sworn r="path:papers/checksum/smollm_quant_certs.json#/int8/distance/ci_mean_abs/1" k="numeric">to 1.90</sworn>, verdict DRIFT, belief-geometry agreement <sworn r="path:papers/checksum/smollm_quant_certs.json#/int8/distance/rdm_r" k="numeric">0.83</sworn>. The plain number agrees: <sworn r="path:papers/checksum/smollm_quant_certs.json#/sanity/first_token_top1_hits/A" k="numeric">the full-precision model gets the first continuation token right on 30 canaries</sworn>, <sworn r="path:papers/checksum/smollm_quant_certs.json#/sanity/first_token_top1_hits/int8" k="numeric">the quantized one on 16</sworn>. The checksum saw the damage without a benchmark being run.
- **The far control.** <sworn r="path:papers/checksum/smollm_quant_certs.json#/random/distance/mean_abs_nats" k="numeric">A vs R: 9.50 nats per token</sworn>, belief-geometry agreement <sworn r="path:papers/checksum/smollm_quant_certs.json#/random/distance/rdm_r" k="numeric">0.07</sworn>, <sworn r="path:papers/checksum/smollm_quant_certs.json#/sanity/first_token_top1_hits/random" k="numeric">0 canaries right</sworn>.

The four belief geometries are drawn as plates in `smollm_quant_plates.png`: A and A′ identical, Q on
the same lines with visible moves, R unrecognisable.

## what this does and does not establish

It establishes that the instrument's positive controls hold on a real model: identical weights
read SAME at exactly zero; a known damage reads DRIFT with a magnitude and an interval; an
unrelated model reads far. `tests/test_checksum.py` pins the same three properties on scripted
probes, plus the degeneracy guard (a fingerprint with no variation across items is INCONCLUSIVE,
never SAME) and the refusal to compare across canary sets.

It does not establish anything about quantization as practiced. Per-tensor dynamic int8 without
calibration on a 135M model is the crudest case; GPTQ, AWQ and NF4 on real deployments are far
gentler and were not run. It does not establish a threshold for "meaningful" drift — the
instrument reports magnitude and resolution, and the threshold belongs to a preregistered use.
It does not establish a noise floor for nondeterministic serving (temperature, GPU kernels);
this run was deterministic on cpu, where the floor is exactly zero.

## next, in order

1. The same run on a model people deploy (1B–3B, bf16 vs a real 4-bit method), on the alienware,
   with the prereg frozen and sealed before the run.
2. A frontier API model with logprobs, fingerprinted daily — the drift observatory.
3. The threshold question, preregistered, not guessed.
