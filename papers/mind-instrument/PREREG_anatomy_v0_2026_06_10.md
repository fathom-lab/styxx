# PREREG — Anatomy of Convergence v0: which parts of meaning are universal?

**Frozen 2026-06-10, before any scored run. Fathom Lab / styxx. Moonshot M4 territory.**

The capstone established THAT independent minds converge on the geometry of meaning; the Atlas v0
showed family signatures in HOW MUCH. Nobody has decomposed WHERE: which concept domains carry the
convergence, and is that anatomy itself shared? If the per-domain profile of convergence is the same
across independently-built minds, the universal structure has internal anatomy — a map of which
meanings are most universal. If profiles are idiosyncratic, "citizenship" is one number and the
anatomy reading dies.

## Apparatus (all frozen, all validated)

- Battery: the frozen 96-concept / 8-category set (`styxx.mind.BATTERY`; categories: animal, fruit,
  vehicle, profession, body, weather, furniture, instrument — 12 concepts each).
- Geometry: `styxx.mind.distmat` + `partial_corr` with the frozen lexical controls (equivalence-gated
  M2), restricted per category: sub-RDM over that category's 12 concepts (66 pairs), lexical design
  restricted to the same pairs.
- Anchors: the 6 stored convergence models (`contextual_reps.npz`, fixed layer).
- Subjects: the 4 OUT-OF-ANCHOR minds of Atlas v0 (gpt2, gpt2-large, pythia-410m,
  Qwen2.5-0.5B-Instruct) — families gpt2/pythia never touched anchor construction; live reps at the
  frozen 0.66-layer convention, persisted to `atlas_live_reps.npz` (receipt).

## Measure

For each subject mind m and category c: `A(m,c)` = mean partial-lexical RSA between m's category-c
sub-RDM and each CROSS-FAMILY anchor's category-c sub-RDM. Each subject yields an 8-long anatomy
profile; ranks within subject are the object of the test.

## Pre-registered gates (frozen before the run)

- **P1 (shared anatomy):** Kendall's W of the category RANKINGS across the 4 independent subjects
  satisfies `W >= 0.60` AND exceeds the 95th percentile of a 10,000-draw null (each subject's ranks
  independently permuted, seed 0). SUPPORTED iff both; else CLOSED_NEGATIVE.
- **P2 (anchor reference, descriptive only):** the same W computed across the 6 anchors is reported
  as the in-set reference. No bar; context only.
- **Validity control:** for one subject (gpt2), the full-battery citizenship recomputed from the
  persisted reps must match its Atlas v0 receipt value to ±0.0005 (the pipeline reproduces itself).
  Mismatch = VOID-PIPELINE, no anatomy claim.
- Stated prior (honest): SUPPORTED, with concrete high-coherence categories (animal, body) ranking
  above weather/profession — but the per-category n is small (66 pairs), so a noise-driven
  CLOSED_NEGATIVE is a live possibility and will be reported as such.

## VOID conditions

- VOID-PIPELINE (control above fails).
- VOID-SUBSTRATE: any live model fails to load at the frozen layer convention.
- Smoke output only to `*_SMOKE_INVALID*`; never read as results.

## What this can and cannot establish

CAN: whether the internal anatomy of meaning-convergence is shared across independent lineages, and
the first map of which domains are most universal. CANNOT: anything about WHY (training-data
frequency vs perceptual grounding vs structure — confounds named, untested here); anything beyond
this battery and layer convention.
