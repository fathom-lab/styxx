# RESULT — Do REAL LLMs converge to the same concept geometry? Above-chance YES, uniformly NO.

**Date:** 2026-06-03 · The **non-circular, real-model** replacement for the synthetic Disjoint-
Worlds toy the audit flagged (which passed the same latent `z` to both worlds). Here nothing is
built in: 6 independently-trained LLMs (different orgs, architectures, data) are asked whether they
represent **real concepts** with the **same geometry**. Method: contextual-template last-token
representations → per-model 96×96 distance matrix → cross-model RSA, lexical-controlled and
anchored to an independent embedder. Reproduces an **established** phenomenon (Platonic
Representation Hypothesis; vec2vec; CKA) — its value here is being *non-circular*, *controlled*,
and honest about *heterogeneity*.

## The disciplined arc (every step committed)

| step | what | result |
|---|---|---|
| **v1 (PRE-REGISTERED)** | bare single-word reps, fixed layer | **NULL** — mean RSA **0.075**, gate ≥0.30 FAILED. *But positive control failed too:* within-model category structure ~1.2 (each model barely separated animals from furniture) → measurement too weak to trust the null. |
| **v2 (exploratory)** | contextual templates, last-token | convergence appears: fixed-layer xfam RSA **0.313**, best-layer **0.580**; 5/6 models 0.80–0.98. The v1 null was **measurement-limited**. |
| **v3 (controls)** | partial out word-length + token-count; independent MiniLM anchor | **SEMANTIC, not lexical:** xfam 0.313→**0.304** after lexical removal; MiniLM-anchor 0.385→**0.384**. The agreement is about meaning, not spelling. |
| **confirm (PRE-REGISTERED, out-of-sample)** | 96 FRESH concepts, 8 NEW categories, gate frozen first | **CONFIRMED:** cross-family partial-lex RSA **0.258** (≥0.25), control −0.002±0.011, MiniLM anchor 0.360. Replicates on unseen words. |

**The v1 pre-registered null STANDS as the pre-registered result.** v2/v3/confirm are the honest
story of *why* it was null (a weak measurement) and what a strong, controlled, out-of-sample
measurement shows. Discipline applied symmetrically: the failed positive control is what licensed
re-measuring — not a dislike of the answer.

## The honest finding: convergence is universal in DIRECTION, heterogeneous in MAGNITUDE

On the fresh, pre-registered set, **every model pair shares geometry above chance** (weakest pair
0.075 ≈ 7× the control SD; control ≈ 0). But the strength ranges over an order of magnitude:

| pair type | partial-lex RSA | reading |
|---|---|---|
| same-family (Qwen↔Qwen, Llama↔Llama) | 0.96 – 0.999 | near-identical |
| **Llama ↔ Gemma** | **0.93** (both sets) | strongest cross-family convergence; robust |
| Qwen ↔ Llama/Gemma | 0.82 (concrete nouns) → **0.10** (fresh/abstract) | **concept-dependent**, fragile |
| **Phi-3.5 ↔ all** | 0.17 – 0.21 (both sets) | **consistent outlier** |
| cross-family aggregate | mean **0.258**, median **0.166** | mean carried by Llama↔Gemma |

So the aggregate "CONFIRMED" is real but must not be read as *uniform* convergence: it is **carried
by Llama↔Gemma**, the median cross-family pair is **0.166**, and Qwen's cross-family agreement
**collapses on abstract concepts**. Reporting the mean alone would have hidden this.

## Interpretation (HYPOTHESIS — tagged, not claimed)

The outliers **track training data**, which is what a *meaning*-convergence account predicts and a
lexical artifact cannot (a spelling effect could not spare Phi, which has the identical words):
- **Phi-3.5** (heavily *synthetic "textbook"* data) sits apart from everyone.
- **Qwen** (heavily multilingual) converges with Llama/Gemma on concrete nouns but diverges on the
  abstract/emotion-heavy fresh set.
- **Llama & Gemma** (similar web-scale English recipes) nearly coincide.

This is consistent with the **Platonic Representation Hypothesis as a *scaling/capability limit*** —
these 1–3B models are only **partway up** the convergence curve, and unevenly, by data recipe. A
clean, falsifiable prediction follows: **convergence should rise with scale** and with **training-
data overlap.** Not tested here.

## What this does for the program
- **Replaces the circular synthetic geometry toy** (audit C2/C3) with a real-model, real-concept,
  lexically-controlled, out-of-sample-confirmed result. The geometry-convergence claim no longer
  rests on a hand-passed latent.
- **Confirms the mechanism is semantic** (lexical-partial + independent anchor) — the thing the
  synthetic toy could only assert.
- **Adds a genuinely new, honest wrinkle** the headline literature underweights: at small scale
  convergence is **partial and data-dependent**, with **Phi-3.5 a clean low-convergence outlier**.

## Caveats (frozen)
- **Not novel.** Reproduces PRH / vec2vec / CKA. Contribution = non-circular + controlled +
  heterogeneity map + disciplined arc, *not* a new phenomenon.
- 6 small **instruct** models; 96+96 concepts; one representation recipe (contextual-template
  last-token). Best-layer numbers use a model-internal layer choice (category separation) and are
  **supplementary**; the **fixed-0.66-layer numbers are primary** (not fished).
- RSA uses the trivial word↔word correspondence — a **similarity** measure, **not** unsupervised
  translation (that is a separate, harder claim; vec2vec does it, we do not here).
- The training-data interpretation is a **hypothesis**, not measured. Scale/overlap sweep is the
  next real experiment.

## Reproduce
`run_real_convergence.py` (v1 null) · `_v2` (strengthened) · `_v3_controls` (lexical + anchor) ·
`_confirm` (pre-registered out-of-sample). Results: `real_convergence*_result.json`,
`contextual_reps.npz`. Prereg: `PREREG_real_convergence_2026_06_03.md`.
