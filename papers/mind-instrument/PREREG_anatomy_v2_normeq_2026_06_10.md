# PREREG — Anatomy v2: does the anatomy survive the norm-equalized apparatus?

**Frozen 2026-06-10, before any scored run. Fathom Lab / styxx. Successor to v1 (VOID-APPARATUS,
`anatomy_v1_result.json`): the unweighted template average is norm-dominated by the bare `{w}`
template (row-norm 531.6 vs 80.8 on the other templates, Qwen2.5-3B), so the validated convention
measures a bare-word-dominated geometry. This is the strongest critique we can currently aim at the
same-day anatomy v0 finding — so we aim it.**

## Apparatus (frozen)

- **Norm-equalized reps:** for each mind and battery word, compute the 0.66-layer last-token state
  per template SEPARATELY, L2-normalize each template's state, THEN average over the 8 frozen
  templates. No template can dominate by magnitude.
- All 10 Atlas minds recomputed under this convention (anchors live-recomputed; npz NOT reused —
  it carries the old convention).
- Anatomy measure exactly as v0: per-category cross-family partial-lexical RSA of a subject's
  category sub-RDM vs each cross-family anchor's, frozen battery/categories/lexical constants
  (`styxx.mind`), subjects = the 4 out-of-anchor minds.

## Pre-registered gates (frozen)

- **G1 (anatomy robustness):** Kendall's W of the category rankings across the 4 independent
  subjects under the norm-equalized apparatus: `W >= 0.60` AND permutation `p < 0.05`
  (10,000 draws, seed 0). PASS -> **ANATOMY-ROBUST** (the v0 finding survives its strongest
  apparatus critique). FAIL -> **APPARATUS-ARTIFACT** (v0's SUPPORTED is publicly downgraded to
  convention-bound; the FINDING gets a correction notice).
- **G2 (descriptive, no bar):** rank-correlation (Spearman) between the v0 mean anatomy and the v2
  mean anatomy over the 8 categories — how much of the map was carried by the bare-word component.
- **G3 (validity):** under v2, the within-mind odd/even split-half battery reliability for
  Qwen2.5-3B must EXCEED its v1 value (0.291 SB-corrected); the norm fix should restore coherence
  (reliability no longer mechanically floored by norm dominance). If it does not exceed it, the
  norm-domination diagnosis was wrong -> VOID-DIAGNOSIS, both v1 and v2 readings reopened.

## VOID

- VOID-SUBSTRATE: any model fails to load under the frozen convention.
- Smoke writes `*_SMOKE_INVALID*` only; smoke = 2 minds.

## Stated prior (honest)

Genuinely uncertain, leaning ANATOMY-ROBUST: category structure is strong in both bare and
contextual geometry, so the per-domain profile plausibly survives. If it does not, that is the
bigger finding: the published convergence convention's anatomy is an artifact of token norms — a
result with consequences for every template-averaged RSA in the literature.
