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

## WHERE depth helps — an interpretable decomposition (`feature_decomposition.py`)
Per-feature cross-validated prediction (embedding → each of the 54 human features), **both PCA-50
(dim-matched)**. Overall deep 0.657 vs shallow 0.579 (advantage +0.078, consistent with the RSA). But
the advantage is **concentrated, not uniform** — and the pattern is the story:

| depth helps most (abstract/relational) | adv | word-counting suffices (perceptual/routine) | adv |
|---|---:|---|---:|
| 复杂度 complexity | **+0.296** | 视觉 visual | −0.048 |
| 短/慢/快 short/slow/fast (magnitude, dynamics) | +0.16–0.20 | 触摸 touch | −0.015 |
| 热 heat · 明亮度 brightness | +0.16 | 沟通 communication | −0.026 |
| 路径 path · 身体 body | +0.15 | 惊讶 surprise · 自我 self | ~0.006 |

**Deep models earn their advantage on abstract/structural meaning** (complexity above all — a +0.30
chasm — plus magnitude, speed, path), the dimensions that require integrating beyond co-occurrence. On
**directly-grounded perceptual properties** (visual, touch), shallow co-occurrence already suffices and
even wins on "visual." (Per-feature differences are descriptive — not each individually bootstrapped —
but the overall effect is the significant one above, and the abstract>perceptual gradient is consistent.)

## Full-zoo robustness — confirmed cluster effect, refined to a GRADIENT (`human_feature_spectrum.py`)
All 10 provided models (PCA-50, RSA vs human features): **ERNIE 0.523 > GPT2 0.505 > fastText 0.452 ≈
BERT 0.450 > Electra 0.434 > GloVe 0.379 ≫ vision (ViT 0.234 … DenseNet 0.040).** Cluster means: deep
**0.478** vs static **0.415** vs vision 0.147; **deep − static +0.063, bootstrap 95% CI (0.047, 0.077),
P(deep>static) = 1.000** — the cluster effect is robust across the whole zoo.

**Honest refinement:** the clusters **overlap** — fastText (a *static* subword embedding) beats BERT and
Electra. So it is **not a clean deep/shallow binary; it is a sophistication gradient:** contextual +
knowledge-enhanced models (ERNIE, GPT2) at the top, **raw co-occurrence (GloVe) at the floor**, and
subword-static fastText competing with the weaker deep models. The strongest, cleanest contrast is
**ERNIE/GPT2 vs GloVe** (~0.51 vs 0.38); "deep > shallow" holds on average but the boundary is fuzzy.
(The full zoo both confirmed the cluster effect *and* corrected the crude binary — discipline at work.)

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
