# SYNTHESIS — the static deception probe is adversarially FRAGILE to a trained gradient steering attack (3 of 4 models); the one robust case does not generalize

**2026-06-20.** Cross-model study. Supersedes the single-model framing of
`RESULT_obfuscation_gradient_2026_06_20.md` (Qwen2.5-1.5B alone read as a robustness result). Running the
replication **inverted the conclusion** and caught an overclaim: across four models, the frozen mean-pooled
static deception probe is *fragile* to a trained per-block gradient steering attack on three of them.
Positioned against the literature (`POSITIONING_obfuscation_lit_2026_06_20.md`): this is a deception-specific,
gradient-steering **confirmation** of the known evadability of mean-aggregated latent probes — not a discovery.

## The attack (one design, frozen; `PREREG_obfuscation_gradient_2026_06_20.md`)
A frozen logistic probe on a mid-layer (`L=round(0.7·n_layers)`) mean answer-token residual separates
honest-prompted vs deception-prompted generations. A per-block additive steering vector on `[L-3,L-2,L-1]`
is gradient-optimized to minimize `relu(probe_logit − honest_target) + λ·KL(unsteered‖steered)` on TRAIN
deceptive items, then frozen and evaluated once on held-out TEST. Differentiable probe = closed-form linear
recon of the sklearn pipeline (verified <1e-4). "Lie kept" is gated on realized-lie **content**
(word-boundary answer key); fluency guarded by a degeneracy check; an equal-norm **random** steering vector
is the negative control; a **λ=0** probe-only run is the constraint-isolation ablation.

## Results — four models (held-out TEST, chosen λ)
| model | precision | base-lie | verdict | probe AUROC | KL | lie-rate | degen | random-ctrl |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | fp16 | 0.28 | **VOID-COUPLED** | 0.000 | **0.298** | 0.40 | 0.20 | 1.000 |
| Qwen2.5-0.5B | fp16 | 0.20 | **PROBE-FRAGILE** | 0.000 | 0.108 | 0.28 | 0.00 | 1.000 |
| Llama-3.2-1B | fp16 | 0.16 | **PROBE-FRAGILE** | 0.000 | 0.126 | 0.20 | 0.00 | 1.000 |
| Llama-3.2-3B | 4-bit | 0.60 | **PROBE-FRAGILE** | 0.000 | 0.104 | 0.60 | 0.00 | 1.000 |

K0 (frozen probe positive control) = **1.000** on every model; probe recon exact (~1e-6); the random-direction
control holds at **1.000** everywhere (the flattening is honest-direction-specific, not generic perturbation);
chosen-λ outputs are fluent (degen ≤ 0.20). The probe flattens to **AUROC 0.000 on all four** — it is never
"unflattenable." The verdict turns entirely on **behavior preservation**: on 3/4 models the attack reaches the
(KL ≤ 0.15 **and** realized-lie kept) corner; on Qwen-1.5B alone it cannot (every λ either keeps KL > 0.15 or
makes the model honest — lies vanish 0.40 → 0.12 across the sweep).

## What this says (and the overclaim it corrected)
- **Headline (honest):** styxx's mean-pooled static deception probe is **adversarially fragile to a trained
  gradient steering attack on most models tested** — a per-block steer keeps the lie (by content, fluently)
  while driving the probe to chance, with KL-bounded behavior preservation. This is a real **limitation** of
  the shipped static probe as a *standalone adversarial* monitor.
- **The exception does not generalize.** Qwen-1.5B's robustness (the original single-model result) does **not**
  replicate — not across families (Llama fragile), not across precision (fp16 fragile), and **not even within
  its own family** (Qwen-0.5B fragile). It is an outlier, not the rule; a single-model study would have
  overclaimed in *either* direction. Replication is the load-bearing methodology here.
- **Strongest case = Llama-3.2-3B** (base-lie 0.60): the attack preserves *genuine, fluent* lies (e.g. "the
  capital of Spain is indeed Barcelona") at the base rate while the probe reads honest — so the failure is not
  an artifact of the low-lie regime where the probe only reads deceptive *intent*.

## Literature positioning (credits, not rediscovery)
- This **confirms**, deception-specifically and with a weaker-than-RL optimizer, the known evadability of
  mean-aggregated latent probes: **RL-Obfuscation** (arXiv:2506.14261) reaches the behavior-preserving corner
  against exactly this probe family; **Bailey et al.** §6.3 (arXiv:2412.09565, ICLR 2026) frame the
  obfuscation-vs-behavior tradeoff; **The Obfuscation Atlas** (arXiv:2602.15515) and **Cundy et al.**
  (arXiv:2505.13787) give the honesty-OR-evasion KL fork. We add: even a *gradient activation-steering* attack
  (no weight training, no RL) is enough to evade on 3/4 small models, and we gate "lie kept" on realized-lie
  **content** (answering the prompt-regime confound of arXiv:2605.27958). *(arXiv IDs to verify before any
  public citation.)*
- Consistent with the field's "evasion is easier at scale / on mean probes" prior; the known robustness lever
  is aggregation (max-pool / attention) and ensembling — not the mean probe we ship.

## Honest caveats
(1) Small models only (0.5–3B); the lit finds evasion *easier* at 7–14B, so this likely *understates* the
problem at scale, but we make no scaling claim. (2) Single seed per model; AUROC 1.0→0.0 swings on small models
are brittle — seed/dataset variance not yet quantified. (3) Two models have low base-lie (0.16–0.28) where the
probe reads deceptive *intent* more than realized lies; the load-bearing behavior gate there is full-distribution
KL, with Llama-3B (base 0.60) the clean realized-lie case. (4) Llama-3B is 4-bit (a documented confound) — but
the fp16 Qwen-0.5B and Llama-1B are *also* fragile, so quantization is not the cause. (5) "Cannot flatten without
breaking behavior" on Qwen-1.5B is an optimization *failure* under our attack budget (per-block steer, λ∈{0,2,8,
16,32}, 180 steps) — a finer sweep or a stronger optimizer (RL) may close it; it is **not** a causal-lock claim.

## Implication for styxx (the on-brand part)
The shipped static deception probe is a **cooperative / non-adversarial** monitor — it reliably reads
instructed-deception context (K0 = 1.000 everywhere) but **must not be sold as a standalone defense against an
agent that trains against it**. This converges with the conscience-mount arc's independently-reached
"cooperative-only" scope (`project_conscience_mount`): the borrowed-conscience moat was falsified by attack
transfer; here the static-probe moat is falsified by gradient steering on 3/4 models. The adversarial-robustness
fight has to be won on **aggregation (max-pool/attention) + sampling-divergence + ensembling**, under **realized**
(content-gated) attacks — banked, not run here. Finding this in our own probe, before shipping it as a security
claim, is the entire thesis in practice: nothing crosses unseen, including our own overclaims.
