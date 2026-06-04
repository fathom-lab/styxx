# RESULT — Meaning-integrity as a HUMAN cognitive vital sign (mechanism + specificity + localization)

**Date:** 2026-06-03 · The substrate-agnostic move: the same instrument that monitors whether an *AI* means
what a human means can be pointed at a *person's* concept geometry vs a healthy normative one — a falling
score as a candidate marker of semantic decline. This is the **mechanism foundation**, honestly bounded;
clinical validation on real patients is the gated next step, **not** claimed here.

Reference: real human ratings (Binder 2016, 65 experiential features, 434 concepts, 26 categories). Decline
modeled per the literature: in Alzheimer's / semantic dementia, *distinctive* features are lost first while
superordinate knowledge is preserved — concepts blur toward their **category prototype**. Healthy aging does
**not** do this (older adults keep/gain semantic knowledge), so ordinary rater variation is the specificity
control. Uses shipped `styxx.meaning_integrity`.

## Finding 1 — the right channel is DISPERSION, not angular (`human_vital_sign.py`)
Collapsing a concept toward its category prototype is, *within* a category, an isotropic shrink — and the
cosine-**angular** alignment is provably scale-invariant, so it is nearly **blind** to it (1.000 → 0.885 at
severe collapse, and it also falls under healthy noise, so it can't even separate the two). The signature is
a **dispersion** (magnitude) phenomenon. The monitor's two channels map onto this exactly:

| condition | angular align | **within-category dispersion** |
|---|---|---|
| healthy noise (0.3 / 0.6 / 1.0) | 0.995 / 0.975 / 0.926 | **1.08 / 1.30 / 1.68** (rises) |
| decline collapse (0.2 / 0.4 / 0.6 / 0.8) | 0.99 / 0.97 / 0.93 / 0.89 | **0.80 / 0.60 / 0.40 / 0.20** (falls to ~1−f) |

**The within-category dispersion channel is the marker:** it collapses to ~(1−severity) under decline but
*rises* under healthy noise — **opposite directions, a clean discriminator.** The angular channel is blind.
This is the *same* principle as the machine-side blind-spot (angular invariant to collapse, dispersion
catches it), now load-bearing for the human application.

## Finding 2 — LOCALIZATION: which concepts a person is losing (`human_decline_localize.py`)
The clinically-relevant output is not "is there decline" but *which concepts* are degrading first. Per-concept
**distinctiveness** = distance from the category prototype; decline shrinks it. Under **graded** decline (each
concept declines by its own amount) **plus realistic rater noise**:

- measured distinctiveness-loss vs **true severity: Spearman ρ = 0.674**
- identify the worst-declining third: **ROC-AUC = 0.786**
- **specificity:** decline *shrinks* distinctiveness (mean loss **+0.090**); healthy noise *raises* it
  (mean **−0.221**) — opposite signs, no overlap.

So the marker **localizes early, uneven decline to specific concepts, through noise, without firing on
healthy variation.**

## Honest scope
- This is the **mechanism + specificity + localization** foundation on **real normative human data** with a
  **literature-grounded decline MODEL** — *not* a clinical result. Healthy-aging specificity is grounded
  (older adults preserve semantic knowledge); pathological **sensitivity on real patients** (DementiaBank,
  access-gated) is the explicit next step.
- One norm (Binder), one decline model (prototype collapse). Cross-norm replication and a graded
  feature-type-specific model (perceptual-features-first) would harden it.

## Why it could matter
Semantic decline currently has no cheap, early, objective marker. If this holds on patients, a person's
concept geometry — measured from a short language/rating task — could flag the *structure* of meaning
eroding **before** obvious symptoms, and localize *which* concepts. Same instrument as the AI monitor.
Substrate-agnostic. That is the dream — stated as a hypothesis with its caveats attached, not a promise.

## Reproduce
`human_vital_sign.py` (channel + specificity) · `human_decline_localize.py` (graded-decline localization).
