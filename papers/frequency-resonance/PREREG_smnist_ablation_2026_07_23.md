# PREREG — seq-MNIST oscillation-vs-decay ablation (the real-task flagship) — 2026-07-23

**FROZEN before confirmatory data** (smoke/plumbing only). Runner: `run_smnist_ablation.py`. 1× RTX
4070. The oscillation-vs-decay causal ablation LinOSS never published, on a REAL long-range benchmark.

## Question

> On real long-range data (sequential MNIST, 784-length pixel sequence → 10-way classification), is the
> **oscillation** in an S5/LinOSS-class SSM *causally load-bearing* — or would pure real-eigenvalue
> decay do as well?

Whole-architecture benchmarks (LinOSS vs Mamba) cannot answer this — they change many things at once.
The single-knob phase-clamp can: hold the entire deep classifier fixed and flip only whether the
eigenvalue phase θ is learnable (rotation → oscillation) or clamped to 0 (real eigenvalues → decay).

## Setup (frozen)

Deep S5/LinOSS-class classifier: `emb(1→H)` → `N_BLK` stacked blocks {complex-diagonal SSM →
Linear(2·d_ssm→H) → residual+LayerNorm → GELU-MLP → residual+LayerNorm} → mean-pool → `head(H→10)`.
`H=64, d_ssm=64, N_BLK=3`. Recurrence via parallel `lin_scan` (red-team-verified `scan==seq`).

Arms — **matched-param, RNG-matched** (θ is a learnable Parameter in FREE, a zeros buffer in CLAMPED;
`th` is drawn in both so every non-θ init is identical under a seed; verified `B_re`, `emb` equal):
- **FREE** — θ learnable (oscillatory): the S5/LinOSS-class model.
- **CLAMPED** — θ≡0 (real eigenvalues → pure decay): a diagonal real SSM. The ablation.

Training: full 60k train, AdamW `lr=3e-3, wd=0.01`, cosine schedule, `STEPS=4000`, `BATCH=64`,
`SEEDS=[0,1]`. Metric: **test accuracy** on the 10k test set. `gap = FREE − CLAMPED`.

## Frozen verdict

- **OSCILLATION LOAD-BEARING** iff `gap ≥ +0.01`.
- **OSCILLATION NOT NEEDED** iff `gap ≤ −0.01`.
- **TIE** otherwise.

Reported alongside: **within-model oscillation reliance** — the trained FREE model re-evaluated with
θ clamped to 0 *in place* (`free_acc − free_θ→0_acc`), the resonance profiler's within-model measure.

## Prior & discipline

The literature (S5, LinOSS) reports oscillatory SSMs beating decay baselines on long-range tasks, so the
honest prior is **LOAD-BEARING** — but this is the first *controlled single-knob within-architecture*
test on real data, and either outcome is a genuine causal result. 2 seeds; smoke was over-optimistic
this arc three times, so the frozen full run decides. Result → `smnist_ablation_result.json` + `RESULT_`,
OATH-certified.
