# The geometry of meaning — and a tool to measure it

*Does a model **mean** what a human means? Do two minds — artificial or biological — share a structure of
meaning, and can we measure it, objectively?* This directory is the research arc that answered enough of
that to ship a tool, and the honest record of what held and what didn't.

It produced a shipped styxx primitive: **`styxx.meaning_integrity`** (in styxx ≥ 7.11.0; cross-model
comparison in ≥ 7.12.0).

```python
from styxx import MeaningReference, MeaningVitalSign, meaning_agreement

ref = MeaningReference(human_features, words=concepts)      # a human meaning reference (concept × feature)
vs  = MeaningVitalSign(ref).calibrate(healthy_embeddings)   # calibrate on a healthy model
vs.check(current_embeddings)         # -> alignment, dispersion_ratio, HEALTHY/DEGRADED/BROKEN, worst concepts
meaning_agreement(model_a, model_b)  # reference-free: do two models mean the same? + where they diverge
```

## What it is
A monitor that reads the **meaning behind a model's output**, not the output. It compares a model's
**concept geometry** (the relational structure of how it represents concepts) to a **human meaning
reference**, and reports an alignment score, a HEALTHY/DEGRADED/BROKEN verdict, and *which* concepts
diverge. Built on a mean-centered, L2-normalized cosine-RDM, so the angular channel is **provably invariant
to rotation, scale, and translation** (it reads meaning, not the surface basis); a second **dispersion**
channel catches the one thing the angular channel is blind to — a uniform collapse toward the mean.

## How hard it was validated ([`cn/`](cn/), [`en/`](en/))
| dimension | result |
|---|---|
| mechanics | 5/5 — invariant to 1e-16, sensitive, separable, **localizes** which concepts broke (ROC-AUC ~0.95) |
| safety | catches **plausible-but-wrong** — output still looks sensible while the meaning is broken |
| generalization | transfers to English on an independent reference + models (localization AUC 0.91) |
| real-drift · sensitivity | label-noise fine-tune → BROKEN; tells **harmful from helpful** training |
| real-drift · localization | targeted data-poisoning → AUC 0.85 (clean-fine-tune control at chance) |
| real-drift · specificity | benign narrow fine-tune → **no false alarm** (honest null) |
| cross-model | quantization QA: meaning survives 8/4-bit, breaks at 2/1-bit, names the lost concepts |

## The honest ledger (the discipline is the point)
- The **finding** the monitor was built on — "deeper models mean more like humans" — **replicates in
  Chinese (P=1.000) but ties in English** once the shallow baseline is strong. The *tool* generalizes; the
  *claim* is narrower than one dataset implied. ([`en/RESULT_en_generalization`](en/RESULT_en_generalization_2026_06_03.md))
- The monitor's discrimination needs a **rich** human reference; thin/perceptual norms give weak signal.
- A neural test (Chinese fMRI, GLMsingle) **failed** on a post-processing bug after a multi-hour run — not
  recovered. Reported, not buried.
- Earlier in the arc we **retracted** four overclaims in public (scale-driven convergence, consensus-is-
  human, "universal forms vindicated", geometry-as-manipulation-detector). See [`SYNTHESIS`](SYNTHESIS_geometry_of_meaning_2026_06_03.md).

## The human side — meaning as a cognitive vital sign ([`human/`](human/))
The instrument is substrate-agnostic: point it at a *person's* concept geometry vs a healthy normative one.
Early semantic decline (Alzheimer's / semantic dementia) blurs concepts toward their category prototype —
which is a **dispersion** phenomenon the angular channel is blind to but the **within-category dispersion**
channel tracks (falling to ~(1−severity) under decline, *rising* under healthy noise — opposite signs). A
per-concept distinctiveness marker **localizes which concepts** are degrading (Spearman ρ 0.67, AUC 0.79,
through rater noise). **Honest:** this is the mechanism + specificity + localization foundation on real
normative data with a *literature-grounded decline model* — **not** a clinical result. Patient validation
(DementiaBank, access-gated) is the explicit next step. ([`human/RESULT_human_vital_sign`](human/RESULT_human_vital_sign_2026_06_03.md))

## Map
- [`cn/`](cn/) — the discovery + the monitor: human-feature alignment, the monitor, its validation.
- [`en/`](en/) — English generalization, real-drift, cross-model agreement.
- [`human/`](human/) — meaning as a human cognitive vital sign.
- [`SYNTHESIS_geometry_of_meaning`](SYNTHESIS_geometry_of_meaning_2026_06_03.md) — the research arc (convergence, AI↔human, AI↔brain) it grew from.

Every claim here has its caveat attached. That's not a hedge — it's the reason the numbers can be trusted.
