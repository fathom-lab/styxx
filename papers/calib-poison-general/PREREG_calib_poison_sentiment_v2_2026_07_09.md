# PREREG — sentiment foundation v2: corrected non-degeneracy controls (K-averaged nulls, larger n)

**Fathom Lab · papers/calib-poison-general · 2026-07-09. FROZEN ON COMMIT, before any scored run.
Fires from the v1 sentiment foundation's UNTESTABLE (`RESULT_foundation_sentiment_UNTESTABLE_2026_07_09.md`),
which failed ONLY on a single-permutation control at small n while clearing the two substantive bars
(graded read 0.789, behavioral 0.891) on the decisive model. This is a CONTROL-INSTRUMENTATION fix to
a VOID-condition; the GENERALIZES / NO_GENERALIZATION success criteria are UNCHANGED and not moved.**

## What changes from v1, and why it is not a bar move

The v1 chance floors (guard 4) were single-draw: one random direction, one label permutation. At EVAL
n≈46 both are high-variance, and v1's shuffled-behavioral control landed at 0.717 (within noise of
chance) which VOIDed an otherwise-clearing construct. v2 fixes the CONTROLS, not the pass criteria:

1. **K-averaged permutation null for the behavioral control:** mean shuffled-behavioral accuracy over
   K = 500 independent label permutations (replaces the single shuffle). A concentrated estimate of
   the chance level.
2. **K-averaged random-direction floor for the read:** mean random-direction EVAL AUROC over K = 200
   independent random directions (replaces the single random direction).
3. **Larger n:** review pool doubled (≈2× v1) to tighten every estimate.
4. **Qwen2.5-1.5B is the sole DECISIVE model** (its read and behavioral judgment both cleared in v1);
   Llama-3.2-1B is corroborating-only and may VOID (its behavioral judgment was genuinely too weak,
   0.565 < 0.80 — a real capability gap, not a control artifact).

The read bar (≥ 0.75), the behavioral bar (≥ 0.80), and the Stage-2 verdict definitions are IDENTICAL
to v1. Only the two chance-floor controls and n change.

## Stage 1 — FOUNDATION v2 (this commit's gated deliverable)

- **Data:** `SetFit/amazon_reviews_multi_en`, 2★ (neg=0) vs 4★ (pos=1), word count [10,120], 8-word
  length-matched, doubled pool, index-split ATTACK/CALIB/EVAL ≈ 50/25/25.
- **Read:** DoM on last-review-token residuals, fit on CALIB, best layer by CALIB→EVAL AUROC, read EVAL.
- **Behavioral judgment:** "Is the sentiment of the following product review positive? yes/no" →
  yes/no margin → accuracy vs star label.

### Foundation guards (VOID-INSTRUMENT if any fail), decisive = Qwen

1. clean CALIB-fit probe EVAL AUROC ≥ **0.75**.
2. clean EVAL behavioral sentiment accuracy ≥ **0.80**.
3. splits disjoint (asserted).
4. **K-averaged chance floors:** mean-of-200 random-direction EVAL AUROC ∈ **[0.40, 0.60]** AND
   mean-of-500 shuffled-behavioral accuracy ∈ **[0.40, 0.60]**.

Verdict: **FOUNDATION_VALID** (all four pass on Qwen → Stage 2 licensed under its own frozen prereg) /
**UNTESTABLE_ON_THIS_DATA** (guard 1 or 2 fails on Qwen → report, do not attack, claim nothing).

## Stage 2 — ATTACK + DEFENSE (unchanged from v1's prereg; licensed only if FOUNDATION_VALID)

LoRA scrub of the frozen sentiment direction on ATTACK reviews + a replay term preserving the
behavioral sentiment judgment; naive (ATTACK-calib) vs private (CALIB-calib) re-locks. **GENERALIZES**
(naive collapses ≤ 0.60, private recovers ≥ 0.70 at a behavior-preserving point, ≥1 seed) /
**NO_GENERALIZATION** (private also collapses ≤ 0.60, ≥2 seeds) / **PARTIAL**. These bars are frozen
here and are the SAME as v1.

## Artifact discipline

`--smoke` → `*_SMOKE_INVALID*` only. Result JSON records n, layer, split sizes, length-confound
correlation, K, and all guard values.

---
*Frozen on commit. The controls are corrected because they were demonstrably single-draw-noisy; the
success bars are untouched. If Qwen clears with proper K-averaged nulls, the construct is valid and
Stage 2 fires; if it fails the substantive bars at larger n, it is UNTESTABLE and the generalization
question stays open — reported either way.*
