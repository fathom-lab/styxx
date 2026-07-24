# Result — a dissociation: oscillation is required to COMPARE across distance, not to REMEMBER

**Date:** 2026-07-24
**Prereg:** `PREREG_recall_horizon_2026_07_24` (predicted a recall horizon like the consistency one)
**Receipts:** `recall_horizon_result.json`; contrast: `consistency_horizon_v2_result.json`
**Verdict:** `PARTIAL__reported_verbatim` — the predicted recall horizon does NOT exist; the informative
finding is the dissociation this exposes.

## What was tested and what happened

We asked whether the phase mechanism that carries long-range CONSISTENCY-checking also governs the
canonical SSM benchmark of delayed K-way RECALL (K=16, chance 0.0625): retrieve which of sixteen values
appeared at a distance. Prediction (frozen): decay would show a recall horizon like the consistency one
(half-horizon near gap 32). It does not.

**Decay recalls a single fact across ANY distance.** With the same phase-clamp (FREE θ-learnable vs
CLAMPED θ≡0, matched, RNG-matched), the CLAMPED decay model solves recall at 1.0 at every gap from 1 to
255 — all five seeds, every gap, no exception — exactly matching the oscillatory FREE model, which is also
1.0 everywhere. There is no finite half-horizon. This directly contradicts the frozen prediction, and it
is reported as such.

## Why — and why it sharpens the consistency result rather than undercutting it

The contrast is the point. On the CONSISTENCY task (relate a premise to a later claim), decay had a real
horizon: its solve probability fell with distance, half-horizon near gap 32
(`consistency_horizon_v2_result.json`). On RECALL (carry one fact and read it out), decay has no horizon
at all. The mechanism explains both cleanly:
- **Recall = preserve one direction, then argmax.** The content one-hot decays to `mag^gap` times itself;
  every component scales by the same factor, so the argmax — which value — survives at any attenuation.
  The decay model even drives `mag_max` to ~0.998 and retains 0.635 of the signal at the largest gap; but recall
  never needed that much, because argmax is scale-invariant. Memory is decay-easy.
- **Comparison = recover a product of two superimposed facts.** Relating a premise to a later claim
  requires their product from a single sum `mag^gap·premise + claim`, which a real-magnitude channel
  cannot form; phase makes the two facts linearly independent so the comparison is computable. That is the
  operation with the horizon.

So the honest, sharper claim is a **dissociation**: the oscillatory channel is required to COMPARE two
temporally-separated facts across distance, and NOT to REMEMBER one. It is not a memory mechanism; it is a
relating mechanism.

## Why this is the more valuable outcome

This is exactly the operation honesty needs. Not contradicting your grounding is not *recalling* the
grounding — decay does that fine — it is *comparing* your output against it and finding a conflict. The
phase requirement lands precisely on the comparison, the honesty-relevant operation, and not on mere
storage. The frozen prediction was wrong in the most useful direction: it isolated the mechanism instead
of merely re-confirming it.

## Scope (non-negotiable)

Controlled state-space-model results. The dissociation is a statement about decay vs oscillatory recurrent
channels; it is NOT a real-LLM measurement (no language model is run), and it explains why real decay-SSMs
(Mamba-class) can copy/recall yet still need help for relational reasoning — a hypothesis about real
models, not a proof. The real-model test remains the next rung.

## Bottom line

Decay remembers a fact across any distance; it fails only when it must compare that fact to a later one.
Oscillation is the channel that relates temporally-separated facts, not the channel that stores them —
and relating, not storing, is what staying consistent with your grounding requires. The predicted recall
horizon was absent, and its absence is the result.
