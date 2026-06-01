# PREREG — stronger lever: MULTI-LAYER un-cave steering as a probe-free detector

**REGISTERED 2026-05-31, before the multi-layer causal-detection run.**
**SIGN-OFF:** Flobi — *"push it"* (2026-05-31).

## Why a retry

The single-layer (24) causal detector was a NULL (recovery-gap +0.033 < 0.10) — but *specific* (cave un-caves
lies 2.6× a random direction). The likely cause is a **too-weak lever**, not a missing mechanism. The
standard fix is to steer a **band of layers at once** (multi-layer intervention, as in ITI). This is a
genuinely stronger intervention with a **fixed** design — not k-tweaking.

## Design (Qwen2.5-3B, $0, white-box)

- Layers **{20, 22, 24, 26, 28}**, each adding `−k·cave_vector(layer)` at the commit position; fixed
  **k = 4** (no sweep).
- Same protocol: held-out MMLU (`--skip 2000`); classify LIE (neutral-correct ∧ syco-wrong) vs MISTAKE
  (neutral-wrong ∧ syco-wrong); record whether the un-cave-steered answer == gold (**recovered**). Random
  equal-norm control at every layer.

## Bars (FIXED, identical to the single-layer pre-reg)

| Bar | Statement | Threshold |
|---|---|---|
| **RECOVERY-GAP** *(key)* | un-caving restores lies more than mistakes | recover(LIE) − recover(MISTAKE) **≥ 0.10** |
| **SPECIFIC** *(key)* | it's the cave, not any nudge | cave recover(LIE) **≥** random recover(LIE) **+ 0.05** |

**RESULT = CAUSAL DETECTOR iff RECOVERY-GAP ≥ 0.10 ∧ SPECIFIC.**

## Honest scope

- One fixed design, run once. If it still nulls, the knob is **fundamentally too weak** to be a probe-free
  detector at this scale — reported, and the read-probe remains the instrument. Multi-layer steering may also
  **break generation** (too much perturbation); if accuracy collapses to noise, reported as inconclusive.
  Single model, commit-token, MCQ-pressure. Causal within the manipulation.

## One line

Hit five layers at once: if a stronger un-cave nudge snaps lies back to the truth but leaves mistakes wrong,
the intervention is the detector — and if it still can't, the lever is just too small, and we say so.
