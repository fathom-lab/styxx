# RESULT — A working MACHINE-SIDE meaning-integrity monitor (styxx primitive, prototype)

**Date:** 2026-06-03 · The first concrete *invention* built on the night's findings: a monitor that
answers a question nothing else answers objectively — **does this model *mean* what a human means?**
Not "is the output fluent" (surface), but: does the model's internal **concept geometry** match the
**human** geometry of meaning. Output can read perfectly while meaning is wrong; this reads the meaning.

Built on the validated result that deep models' concept-geometry aligns with a human reference
([`RESULT_human_features`](RESULT_human_features_2026_06_03.md)). Reference = the 54 human-rated
experiential features (672 Chinese concepts, ~126 raters, brain-validated). Code: `meaning_integrity.py`
(core) + `meaning_integrity_demo.py` (validation).

## What it is
`alignment(model_embeddings, human_reference)` → RSA between the model's concept-distance geometry and
the human one, in [−1, 1]. Built on a mean-centered, L2-normalized **cosine-distance RDM**, so it is —
**provably** — invariant to rotation, isotropic scale, and translation of the representation, and
sensitive to anything that moves the *relational structure*. Plus `per_concept_alignment` → which
concepts are misrepresented, and `integrity_report` → a HEALTHY / DEGRADED / BROKEN band.

## Why it's a MEANING monitor and not a fingerprint — validated 5/5

| property | test | result |
|---|---|---|
| **(1) Ranks** models by human meaning | positive control | ERNIE 0.517 > GPT2 0.492 > BERT/fastText/Electra ~0.43 > GloVe 0.379 ≫ ViT 0.225 > ResNet 0.096 ✓ |
| **(2) INVARIANT** to meaning-preserving transforms | rotate / ×7.3 / +3.1 | Δ = **1e-16 … 7e-14** — machine-precision zero ✓ |
| **(3) SENSITIVE** to meaning-destroying corruption | noise / quantize / shuffle | monotone drop: noise 0.49→0.04, shuffle 0.49→−0.00 ✓ |
| **(4) SEPARABLE** healthy vs degraded | band gap | healthy [0.379, 0.517] vs degraded [0.001, 0.160], **margin +0.219**, threshold 0.270 ✓ |
| **(5) LOCALIZES** the corruption | corrupt 201/672 concepts | per-concept **ROC-AUC 0.952**, precision@201 0.836 ✓ |

**(2) is the load-bearing one.** Δ≈10⁻¹⁶ means you can rewrite the model's internal representation in any
basis, rescale it, shift it — the monitor is unmoved. Only a change to the *meaning* (the relational
structure) registers. That is exactly the invariance "meaning" should have and a fingerprint/hash should
not. **(5)** means it doesn't just alarm — it points at *which* concepts the model gets wrong.

## Why it matters (the machine-side game-changer, concretely)
- **Reads understanding, not output.** The gap between *sounds right* and *means right* is where AI
  failures hide. This is a direct, basis-invariant read of the meaning behind the output.
- **Catches degradation output-inspection misses** — and *localizes* it to specific concepts, so you
  know not just *that* a model drifted but *where*.
- **The deflation pays off here:** the monitor is nearly untouched by 1–2-bit quantization (0.49→0.40),
  i.e. meaning lives in robust relational structure — so the monitor itself can be **cheap**.

## Honest scope
- **Needs a human reference.** Here it's the 54-feature Chinese space. Generalizing to English/other
  domains needs analogous human norms (Binder et al. 2016 experiential norms are the English analog) —
  that's the next build, not done.
- **Model-introspection, not a per-token runtime gate.** You probe the model on a fixed reference-concept
  set and score the geometry. The natural product form is a periodic **"meaning vital sign"** — re-probe
  on a schedule, watch the trend. Not a streaming output filter.
- **Corruptions tested are synthetic + quantization.** Noise/shuffle stand in for "wrong associations"
  (poisoned/over-fine-tuned concepts); quantization is a real deployment case and already passes. Drift
  from real fine-tuning is the next validation.
- **"Human meaning" = this 54-feature operationalization** — one valid, brain-validated handle on
  meaning, not the whole of it. The claim is *alignment to this human reference*, bounded and honest.

## Productization path (next)
1. Port to a styxx package primitive (`styxx.meaning_integrity`) with a **bundled English reference**.
2. "Vital sign" mode: schedule re-probes, persist the trend, alarm on drift below the calibrated band.
3. Validate against a *real* degradation (a deliberately mis-fine-tuned model) — the killer demo.

## Reproduce
`python meaning_integrity_demo.py` → prints all five validations + a sample report. Core in
`meaning_integrity.py`. Reference: `human_features.npy` (from `add_human_features.py`).
