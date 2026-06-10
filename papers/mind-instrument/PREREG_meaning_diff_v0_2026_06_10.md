# PREREG — styxx.meaning_diff v0: the meaning-regression instrument

**Frozen 2026-06-10, before the validation run. Fathom Lab / styxx.**

Productizes tonight's validated science into one shippable capability: given two models' concept
representations, report whether they MEAN the same, how much, and WHICH concepts diverge most —
norm-equalized (so the number is trustworthy), with a reliability flag (so the user knows when not
to trust it). No new science; this validates the PORT of the validated apparatus, not the claims.

## Public surface (frozen)

`styxx.meaning_diff(reps_a, reps_b, *, words=None) -> dict` with keys:
- `agreement`: cross-model partial-lexical-free RSA of the two concept geometries (Pearson of the
  upper-triangular cosine-distance RDMs; the program's core measure, no lexical control needed when
  the caller supplies their own concept set).
- `verdict`: HEALTHY (agreement >= 0.80) / DRIFTED (0.50–0.80) / BROKEN (< 0.50). Bands frozen here.
- `divergent_concepts`: the words whose per-concept geometry moved most between the two models,
  ranked — each model's row of the RDM compared (1 - correlation), top-k.
- `reliability`: a 0–1 flag = min over the two models of split-half geometry self-consistency when
  templates are available (caller passes `template_reps`), else `null` with a stated caveat; a
  `reliable` boolean = reliability >= 0.5 OR null-with-caveat acknowledged.
- `norm_equalized`: bool — whether inputs were per-row L2-normalized on entry (default True, the
  validated convention).

Helper `meaning_diff_templates(template_reps_a, template_reps_b, words)` accepts (T, N, d) arrays,
applies the frozen per-template-L2-then-average convention, computes split-half reliability, and
calls `meaning_diff`.

## Validation gates (frozen; all must pass or the port does not ship)

- **D1 — apparatus equivalence:** on the Atlas anchors (`normeq_reps.npz`), `meaning_diff`'s
  `agreement` for every cross-model pair reproduces the direct distmat-RSA used in the anatomy/atlas
  pipeline to ±1e-9 (same math, packaged).
- **D2 — divergence sanity:** `divergent_concepts` for a model vs ITSELF is empty (perfect
  self-agreement, no false divergence); for two real models it is non-empty and the agreement on a
  random 50% concept subset stays within 0.15 of the full agreement (stability).
- **D3 — verdict monotonicity:** agreement(M, M) = 1.0 -> HEALTHY; agreement(LLM, shuffled-LLM)
  near 0 -> BROKEN; a known mid pair (Qwen2.5-1.5B vs Llama-3.2-1B from the receipts) lands in its
  receipt-consistent band.
- **D4 — reliability wiring:** with template reps supplied, `reliability` equals the frozen
  split-half SB-corrected battery reliability used in anatomy v2 (Qwen2.5-3B -> ~0.94 under
  norm-eq) to ±0.01; without them, `reliability is None` and `reliable` reflects the caveat path.
- **D5 — determinism + provenance:** two calls bit-identical; pure-Python+numpy, no torch import at
  module load (the instrument must install in the core wheel).

## VOID / scope

- Validation uses ONLY stored receipts/anchors; no model runs, no API.
- v0 scope: concrete-noun battery convention; the agreement bands are frozen heuristics calibrated
  to the receipts, disclosed as such (not universal thresholds). Tests live in `tests/test_meaning_diff.py`.
- Ships in the core package (no torch dep); the heavy template extraction stays caller-side.
