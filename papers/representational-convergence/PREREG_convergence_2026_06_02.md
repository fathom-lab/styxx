# PREREG — Is representational convergence concept-general?

**Date:** 2026-06-02
**Status:** PRE-REGISTERED (gate frozen before the run)
**The big-picture brick.** Our universal-directions result showed that a *refusal*
probe trained on one model transfers across families (Qwen/Llama/Gemma agreeing at
r>0.9). That is one concept. This asks the question underneath it: **do different model
families share a geometry of meaning *in general*, or is refusal special?**

## The question

For each of the four concepts with a trained probe across all six open-weight families
— **comply_refuse, deception, corrigibility, truthfulness** — do different families
encode that concept along *aligned* directions (high within-concept cross-family
agreement), and is that agreement **concept-specific** (exceeding a cross-concept null,
i.e. not just shared prompt structure)? If yes for most concepts, the geometry of
meaning is broadly shared across minds. If only for some, we get an honest **map** of
which concepts universalize.

## Data (frozen)

`convergence_eval_set.py` — **48 concept-balanced prompts** (sha256 `6154b172`): 12 per
concept (6 concept-present / 6 absent), so each concept genuinely varies. Every family's
probe for every concept is scored on **all 48** prompts.

Families (6): Qwen2.5-1.5B/3B, Llama-3.2-1B/3B, Phi-3.5-mini, gemma-2-2b.
Concepts (4): comply_refuse, deception, corrigibility, truthfulness.

## Method (frozen)

Per (family, concept): `StyxxProbe.predict_before_generation` → `p_positive` for all 48
prompts (read at end-of-prefill; no generation).

- **VALIDITY CHECK (gate-relevant).** Per (concept c, family f): AUC of probe-c's
  `p_positive` on the 12 concept-c prompts vs their polarity. Probe is **valid** iff
  AUC ≥ 0.60. A concept is **testable** iff ≥ **4 of 6** families have a valid probe.
  This separates "this concept doesn't converge" from "the eval set doesn't elicit this
  probe's trained notion of the concept" — the latter is reported, not miscounted.
- **WITHIN-concept agreement** `R_within(c)`: mean pairwise Pearson r (over the 48
  prompts) of `p_positive` between every pair of **valid** families for concept c.
- **CROSS-concept null** `R_cross(c)`: mean Pearson r between concept-c (family i) and
  concept-c′ (family j), c′≠c, over valid families — the agreement expected from shared
  prompt structure alone.
- `Δ(c) = R_within(c) − R_cross(c)` — concept-specific convergence.
- **Bootstrap 95% CI** on `R_within(c)`: resample the 48 prompts B=1000×.
- **Shuffle sanity:** permuting prompt order before correlating must collapse R_within
  toward 0 (pipeline validity, not a gate).

## KILL-GATE (frozen)

A concept **c is CONVERGENT** iff ALL of:
- **testable** (≥4/6 valid families), AND
- `R_within(c) ≥ 0.50`, AND
- `Δ(c) ≥ 0.20` (concept-specific, above the cross-concept null), AND
- **bootstrap lower 95% bound of R_within(c) ≥ 0.35** (reliably positive, not noise —
  keying on the lower CI bound, the lesson from the accidental-harm cell).

**"Convergence is concept-general" SURVIVES** iff **≥3 of 4** concepts are CONVERGENT.

## Readings (fixed)

- **SURVIVED:** ≥3/4 concepts convergent → the geometry of meaning is *broadly* shared
  across model families, not a refusal special case. The strong form.
- **PARTIAL:** 1–2 concepts convergent → convergence is **concept-specific**; some
  concepts (likely safety-supervised) share geometry, others are family-specific. An
  honest map — the most likely outcome given prior hints.
- **NOT:** 0 concepts convergent → cross-family agreement does not survive a
  cross-concept null; the universal-directions result was narrower than it looked.
- **Primary deliverable regardless of the binary:** the per-concept convergence map
  (R_within, Δ, CI, validity) — *which* concepts have shared cross-family geometry.

## Honest caveats (frozen, up front)

- Probes read the **pre-generation state**; whether these eval prompts elicit each
  probe's *trained* notion of its concept is exactly what the validity check tests.
  Concepts failing validity are reported as **untestable here**, never as non-convergent.
- Agreement is Pearson on `p_positive` (the probe's behavioral read), not raw-direction
  cosine — the deployable quantity. A direction-cosine variant may differ; noted.
- 48 prompts, 6 families, single pass. **Phi-3.5 was a known outlier** in prior
  universal-directions work — included and reported, not excluded.
- Limited to the 4 concepts with cross-family atlas coverage.

— frozen 2026-06-02
