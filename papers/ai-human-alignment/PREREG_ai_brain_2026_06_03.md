# PREREG — Does LLM concept geometry match the HUMAN BRAIN geometry of meaning?

**Date:** 2026-06-03 · **Status:** PRE-REGISTERED (gate frozen before model RSAs computed).
The behavioral test (`RESULT_ai_human_2026_06_03.md`) showed LLMs share the human *behavioral*
geometry. This is the literal **AI↔brain** test: does the LLM concept geometry match the geometry of
meaning measured in an actual **human brain**?

## Brain ground truth
**Mitchell et al. 2008** fMRI: 9 subjects, 60 concrete nouns × 6 presentations. Per subject: mean
voxel pattern per noun over the top-500 stability-selected voxels; RDM = 1 − Pearson correlation.
Group RDM = mean of 9 subject RDMs. **Noise ceiling = [0.394 lower, 0.557 upper]** (inter-subject;
the max RSA any model can reach). Inter-subject mean RDM corr 0.226. (`brain_rdm.npz`.)

## Machine / behavioral sides
LLM concept RDMs at the fixed **final layer** over the 60 nouns; cohort = the 13 models from the
behavioral study. MiniLM/mpnet embedders. **VICE human behavioral** RDM over the 60∩THINGS subset.
All alignments **partial-lexical** RSA. Brain alignment judged **relative to the noise ceiling.**

## Hypotheses & KILL-GATES (frozen)
- **H1 (AI↔brain):** LLM final-layer geometry aligns with the human brain RDM above chance and
  reaches a substantial fraction of the ceiling. **Gate:** best-model RSA_brain ≥ 0.20 AND ≥ 50% of
  the ceiling lower bound (≥ 0.197), with shuffled-noun control ≈ 0 (|ctrl| < 0.05).
- **Three-way comparison (pre-stated, descriptive):** over the shared 60∩THINGS subset, compare
  **AI↔brain** vs **behavioral-human(VICE)↔brain** vs **embedder↔brain.** Does AI match the brain as
  well as human *behavior* does? (Behavioral-human↔brain is itself a strong reference.)
- **Consistency:** across models, does brain-alignment track behavioral(VICE)-alignment?

## Falsifiers (frozen)
- RSA_brain ≈ 0 / shuffle-level for all models → LLM geometry does NOT match the human brain (dead).
- best model < 50% of ceiling → alignment exists but is weak relative to what the brain supports.

## Honest framing (frozen)
- **Known direction** (text models predict brain semantic responses — Mitchell 2008 itself,
  Schrimpf, Goldstein). Contribution = a clean, **noise-ceiling-relative**, lexical-controlled,
  three-way (brain/behavior/AI) comparison on our convergence cohort, with the same discipline.
- **Visual confound:** Mitchell stimuli were word + line-drawing, so the brain RDM carries some
  **visual** similarity, not pure semantics — flagged, not corrected here.
- 60 nouns; noisy fMRI (ceiling 0.4–0.56); single final-layer read-out; one classic dataset.
- Honest prior: H1 holds; the open question is how close to ceiling and how AI compares to behavior.

— frozen 2026-06-03
