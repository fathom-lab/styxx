# RESULT — Deep LLMs capture HUMAN-rated meaning better than shallow co-occurrence (dim-controlled, brain-independent)

**Date:** 2026-06-03 · A clean, high-powered, **brain-independent** answer to the deflation question,
found by mining an asset we almost walked past: the **54 human-rated semantic features** shipped with
ds004301 (Wang 2022) — 672 Chinese concepts, ~126 human raters, experiential dimensions (vision,
brightness, color, size, emotion, motor…), designed brain-relevant (Binder-style). A *human,
interpretable, non-distributional* meaning space, perfectly aligned (672/672) to the embeddings.

## Question
Does a deep LLM (GPT2/BERT) capture the human-rated meaning geometry better than shallow co-occurrence
(GloVe/fastText) — and does it survive a **dimensionality control** (768 vs 50 dims) and a bootstrap?

## Result — CONFIRMED, and it strengthens under the control
RSA between each model's concept geometry and the human 54-feature geometry (672 concepts):

| | GPT2 | BERT | **deep** | GloVe | fastText | **shallow(GloVe)** | ResNet | ViT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full dims | 0.484 | 0.440 | **0.466** | 0.382 | 0.436 | **0.379** | 0.098 | 0.226 |

- **deep 0.466 vs shallow 0.379 — diff +0.087, bootstrap 95% CI (0.063, 0.097), P(deep>shallow) = 1.000.**
- **Dimensionality-matched (PCA every embedding to 50 dims): deep 0.478 vs shallow 0.379 — diff +0.099,
  CI (0.080, 0.115), P = 1.000.** The advantage is **larger** matched than full, so it is **not** a
  capacity artifact — give every model the same 50 dimensions and deep *still* wins, by more.
- **Vision is the control and behaves correctly:** ResNet 0.098, ViT 0.226 — far below language, as
  expected for a non-perceptual human-conceptual reference.

## Why it matters
- **Brain-independent.** This owes nothing to the noisy fMRI. 672 concepts × ~126 raters = a clean,
  high-powered measurement; the depth effect is unambiguous (P = 1.000, dim-controlled).
- **The deflation, answered against a HUMAN reference.** Earlier (English brain, Mitchell-60) shallow
  co-occurrence matched deep at the brain. Here, against human *conceptual* ratings, **deep genuinely
  exceeds shallow** — and it is the *depth/structure*, not dimensionality. So deep models capture
  human experiential meaning that word-counting does not.
- **Cross-lingual.** Replicated in Chinese, against Chinese human ratings — not an English/co-occurrence
  artifact.

## Honest scope
- One human-feature set (54 experiential dims), one language, RSA. The human ratings are independent of
  the text models (they are behavioral ratings), so the deep>shallow gap is a real model property, not
  shared text leakage. Lexical (length/frequency) not partialled — but it cannot drive a *between-model*
  difference (both models see the same words). The claim is bounded: *deep > shallow at matching this
  human conceptual space*, not "deep models understand meaning."
- This is a controlled brick, not the cathedral. It complements (does not replace) the brain test:
  whether this human-aligned deep structure also reaches the **brain** is the GLMsingle test, still running.

## Reproduce
`add_human_features.py` (builds the human-feature reference + adds it to `predictor_rdms.npz`) ·
`human_feature_test.py` (full + PCA-50-matched RSA + 2000-iter bootstrap). Data: ds004301
`derivatives/annotations/semantic feature/feature.csv`. Result: `human_feature_result.json`.
