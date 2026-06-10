# PREREG — GAVAGAI v0: radical translation between artificial minds

**Frozen 2026-06-10, before any scored run. Fathom Lab / styxx.**

Quine's radical-translation problem (1960): an interpreter with NO shared language, NO paired
examples, and NO ostension cannot, allegedly, determine what a foreign speaker's terms refer to.
Harnad's symbol-grounding problem (1990): symbols allegedly cannot carry meaning without grounding
outside the symbol system. Both have stood as philosophy for decades because no one could read two
independent minds. We can. If meaning is position in a CONVERGENT relational geometry — the
program's central, newly norm-corrected result — then concept identity should be RECOVERABLE from
geometry alone, across minds that share no architecture, no training data, and no construction
history. That is a falsifiable, quantitative version of "translation without a shared language."

## Apparatus (frozen)

- Reps: `normeq_reps.npz` (10 minds × 96 battery concepts, norm-equalized convention of
  `PREREG_anatomy_v2_normeq_2026_06_10.md`; persisted receipt of the v2 run).
- Old-convention contrast: anchors from `contextual_reps.npz` (`fixed__*`) + subjects from
  `atlas_live_reps.npz` — same minds, artifact-bearing convention.
- RDMs: `styxx.mind.distmat` (cosine distance). NO lexical partialling (the translator must work
  from raw geometry; lexical structure is part of what a real translator would NOT share).
- **Translator (unsupervised, label-free):** for the ordered pair (A→B), labels of B are hidden by
  a seeded random permutation (seed 0). (1) Initial signature match: each concept's
  SORTED distance profile (permutation-invariant); cost = Euclidean distance between sorted
  profiles; Hungarian assignment. (2) Refinement, at most 50 iterations or convergence: given
  current mapping π, cost C[i,j] = negative Pearson correlation between RDM_A[i, k] and
  RDM_B[j, π(k)] over k ≠ i; re-solve Hungarian; stop when π stable.
- **Accuracy** = fraction of concepts mapped to their true counterpart. **Category accuracy** =
  fraction mapped within the correct category (chance 12/96 = 0.125).

## Pre-registered gates (frozen)

Primary population: the 24 cross-family ordered pairs between the 4 independent subjects and the
out-of-family anchors are NOT the focus — the focus is harder: ALL 45 unordered mind pairs scored
as A→B with A,B in battery order; headline = mean over the 33 CROSS-FAMILY pairs.

- **G1 (translation exists):** mean cross-family concept accuracy ≥ 10× chance (≥ 0.1042) AND a
  label-shuffle null (translator run on B with rows+columns of RDM_B independently re-indexed by a
  random derangement of concept IDENTITIES — i.e., signatures decoupled from true identity — 100
  runs, seed 0) has 95th percentile below the observed mean. PASS → **TRANSLATION-POSSIBLE**;
  FAIL → **QUINE-UPHELD** (geometry under-determines reference at this scale; the negative is
  reported as the bound on "meaning = position").
- **G2 (the artifact ordering, prediction):** norm-equalized accuracy > old-convention accuracy on
  the same pairs (paired mean difference > 0). Descriptive of the apparatus discovery; no bar.
- **G3 (category floor):** mean cross-family CATEGORY accuracy ≥ 2× its chance (≥ 0.25) even if G1
  fails — partial translation (kind, not individual) is a distinct, reportable grade.

## VOID

- VOID-PIPELINE: translator run with labels NOT hidden (identity start) must reach accuracy 1.0 on
  a self-pair (A→A); else the matcher is broken.
- Smoke (`--smoke`, 2 minds) writes `*_SMOKE_INVALID*` only.

## Honest prior

Uncertain and exciting in both directions. The healed convergence (0.61–0.85) is second-order
structure; assignment recovery needs DISTINCTIVE structure (disjoint-worlds showed recovery is a
sharp threshold). Plausible outcome: strong category-level translation, partial concept-level.
Either branch advances a 65-year-old question from intuition to measurement.
