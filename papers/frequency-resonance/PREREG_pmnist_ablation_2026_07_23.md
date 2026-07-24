# PREREG — permuted-MNIST oscillation ablation (the flagship-sharpener) — 2026-07-23

**FROZEN before confirmatory data** (smoke only). Runner: `run_pmnist_ablation.py`. 1× RTX 4070.
Sharpens the seq-MNIST flagship (`RESULT_smnist_ablation`, gap +0.0408).

## Question

Sequential MNIST has local pixel structure, so a pure-decay SSM still reached 94.3% and the
oscillation gap was a clean but modest +4.1 pts. **Permuted MNIST** applies a *fixed* random
permutation to the 784 pixels (same for every image, train and test), destroying locality — the
canonical genuinely-long-range benchmark. Frozen question:

> Does the oscillation's causal advantage (FREE − CLAMPED) **widen** on permuted MNIST relative to the
> +0.041 measured on sequential MNIST — i.e. does oscillation matter *more* when locality can no longer
> substitute for it?

## Setup (frozen)

Identical to `run_smnist_ablation.py` (deep S5/LinOSS-class classifier, `H=64`, 3 blocks; FREE = θ
learnable vs CLAMPED = θ≡0; matched-param, RNG-matched; parallel scan red-team-verified) **plus a fixed
permutation** of the 784 positions (`torch.randperm(784, seed=1234)`), applied to train and test alike.
AdamW `lr=3e-3, wd=0.01`, cosine, `STEPS=4000`, `BATCH=64`, `SEEDS=[0,1]`. Metric: test accuracy;
`gap = FREE − CLAMPED`.

## Frozen verdict

- **OSCILLATION LOAD-BEARING** iff `gap ≥ +0.01` (else NOT NEEDED / TIE).
- **Sharpening** (the headline comparison): `WIDENS` iff `gap > 0.0408 + 0.005`; `NARROWS` iff
  `gap < 0.0408 − 0.005`; else `similar`.

Reported alongside: within-model oscillation reliance (trained FREE model, θ clamped to 0 in place).

## Prior & discipline

Hypothesis is **WIDENS** — locality cannot substitute for oscillation on permuted input. Smoke (200
steps) shows gap +0.131 vs sMNIST's +0.081 at the same point, consistent with WIDENS — but smoke has
over-predicted the full run three times this arc, so the frozen 2-seed / 4000-step gate decides. The
`0.0408` sMNIST comparison value is fixed here before data. Result → `pmnist_ablation_result.json` +
`RESULT_`, OATH-certified.
