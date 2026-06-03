# PREREG — Does representational convergence RISE with model scale? (the falsifiable PRH prediction)

**Date:** 2026-06-03 · **Status:** PRE-REGISTERED (gate frozen before data).
The real-convergence result (`RESULT_real_convergence_2026_06_03.md`) found convergence is
universal in direction but heterogeneous in magnitude, and proposed — as a HYPOTHESIS — that this
is the Platonic Representation Hypothesis (PRH) caught **mid-climb**: small models partway up a
scale-driven convergence curve. That hypothesis makes a sharp, falsifiable prediction. This tests it.

## Prediction
A model's concept geometry aligns **more** with an independent semantic reference as the model
**scales up**, with architecture and training data held constant.

## Design (frozen) — three CONTROLLED scale ladders

| ladder | models (params) | what's held constant |
|---|---|---|
| **Pythia** (gold) | 14M, 70M, 160M, 410M | The Pile, arch, training order — ONLY scale varies |
| **GPT-2** | 124M, 355M, 774M, 1.5B | WebText recipe, arch — scale varies |
| **Qwen2.5** | 0.5B, 1.5B, 3B | modern recipe, arch — scale varies |

- **Concepts:** the pooled **192** (96 original + 96 fresh, both prior sets) for power.
- **Representation:** contextual-template last-token hidden state at fixed 0.66 relative depth
  (the non-fished recipe validated in v3), → 192×192 distance matrix per model.
- **Alignment metric:** **partial-lexical RSA** between each model's distance matrix and an
  **independent semantic reference** embedder's — partialling out char-length + token-count, so the
  scale trend cannot be a lexical artifact. Reference = **all-MiniLM-L6-v2** (primary),
  **all-mpnet-base-v2** (robustness; reported, not gated).

## KILL-GATE (frozen)
- **SCALE EFFECT CONFIRMED** iff **(i)** pooled Spearman ρ(alignment, log10 params) **≥ 0.50**
  across all 11 models, **AND (ii)** within-ladder ρ > 0 in **≥ 2 of 3** ladders. → convergence is
  scale-driven; PRH-as-scaling-limit supported on controlled ladders.
- **NULL** iff pooled ρ ≈ 0 (|ρ| < 0.2) → within 14M–3B, scale does NOT drive convergence; the
  heterogeneity is data/architecture-driven, not a scale climb (would *weaken* the PRH-limit story).
- **Pythia is load-bearing:** it is the only ladder with data + arch + order perfectly controlled.
  A positive pooled result that FAILS on Pythia is reported as confounded, not clean.

## Honest framing / caveats (frozen)
- Tests an **established direction** (PRH/Huh 2024 shows convergence rises with scale/capability);
  value here = doing it on **perfectly-controlled** ladders (esp. Pythia) with **lexical control**
  and an **independent** reference, and being honest if the narrow 14M–3B range is too small to
  resolve it.
- Tiny models (14M, 70M) are EXPECTED to align weakly — that low end IS the predicted signal, not a
  failure. The alignment-to-an-embedder operationalizes "approaching the shared semantic geometry";
  it is a proxy, not a claim about a literal platonic form.
- Honest prior: CONFIRMED, with Pythia the cleanest test and the modern Qwen ladder likely strongest.

— frozen 2026-06-03
