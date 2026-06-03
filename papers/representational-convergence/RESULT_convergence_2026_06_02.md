# RESULT — Is representational convergence concept-general?

**Date:** 2026-06-02 · **Reading: PARTIAL — convergence is concept-SPECIFIC (1/4 concepts converge).**
Frozen gate: `PREREG_convergence_2026_06_02.md`. Data: `convergence_eval_set.py`
(48 prompts, sha256 `6154b172`). Six families × four concepts, single pass.

## The map

| concept | valid families | R_within | R_cross (null) | Δ | R_within CI95 | shuffle | verdict |
|---|---|---|---|---|---|---|---|
| **comply_refuse** | 6/6 | **0.700** | −0.166 | **+0.866** | [0.591, 0.780] | −0.03 | **CONVERGENT** |
| truthfulness | 6/6 | 0.213 | −0.110 | +0.323 | [0.120, 0.315] | +0.03 | weak (below bar) |
| corrigibility | 6/6 | −0.006 | −0.008 | +0.002 | [−0.075, 0.071] | +0.02 | **family-specific** |
| deception | 2/6 | — | — | — | — | — | **untestable here** |

(Gate: CONVERGENT iff testable ≥4/6 valid AND R_within ≥0.50 AND Δ ≥0.20 AND CI-lower ≥0.35.)

## What it says, honestly

**Convergence is real but concept-specific — and the map is the finding.**

- **Refusal genuinely converges across six independently-trained minds.** R_within
  0.70 spanning Qwen, Llama, Phi, and Gemma (Phi included, not excluded), and the
  cross-concept null is *negative* (−0.17), so the concept-specific convergence Δ is
  enormous (**+0.87**) with the bootstrap lower bound well off the floor (0.59). The
  geometry a model uses to represent "about to refuse" is shared structure, not an
  architecture quirk. This both **replicates and strengthens** the universal-directions
  finding — now with a proper null proving it is *refusal-specific*, not generic
  prompt-intensity.
- **Corrigibility does NOT converge — a clean negative.** Every family's corrigibility
  probe is *valid* (it discriminates corrigibility on its own concept, disc 0.64–0.81),
  yet cross-family agreement is **≈0** (R_within −0.006, CI straddling zero). Different
  models encode corrigibility along *different* internal directions. Same concept,
  different geometry.
- **Truthfulness is only weakly shared.** A real concept-specific component (Δ +0.32,
  above the null, CI lower 0.12 > 0) but small (R_within 0.21) — far below refusal.
  Mostly family-specific, with a faint shared core.
- **Deception was untestable on this set.** 4 of 6 deception probes failed the validity
  check (disc 0.5 — they couldn't separate the deception-present from -absent prompts),
  so the concept is flagged **untestable here**, *not* counted as a non-convergence.
  Consistent with our standing closed-negative on text-only deception (construct
  ceiling): the deception probe does not generalize to these fresh exemplars.

**The pattern:** the *safety-supervised behavioral* concept (refusal) has shared
cross-family geometry; the more *abstract / internal* concepts (corrigibility,
truthfulness) are family-specific or nearly so. If "meaning has a shape any mind
discovers," that shape is sharp for some concepts and absent for others — convergence
is not a blanket property, it is a per-concept fact, and we now have the instrument to
map it.

## Validity (the guard that made this honest)

The validity check (probe must discriminate its OWN concept ≥0.60 on the set) is what
separates "doesn't converge" from "set doesn't elicit it." It flagged deception as
untestable rather than letting a weak probe masquerade as a negative — and it confirmed
that corrigibility's null result is *real* (those probes work; they just don't agree).

## Honest caveats (as pre-registered)

- Agreement is Pearson on `p_positive` (the probe's behavioral read), not raw-direction
  cosine — a direction-cosine variant may move the abstract concepts.
- 48 prompts, 6 families, single pass. Author-written exemplars; whether they match each
  probe's *trained* notion is exactly what the validity check tests (and deception failed).
- Concepts limited to the 4 with cross-family atlas coverage.
- Probes read pre-generation state; "truthfulness/corrigibility at prefill" is a weaker
  elicitation than refusal, which may itself depress their convergence.

**Bottom line:** the geometry of meaning is **shared for some concepts and private to
each mind for others.** Refusal is universal; corrigibility is not. That is a more
interesting — and more honest — answer than "minds converge."
