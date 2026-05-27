# Pre-stated prediction — Baseline-006 (committed BEFORE the gauntlet run)

**File created:** 2026-05-27, before any `styxx gauntlet` invocation on Baseline-006.

## What is being tested

Baseline-005 (word-level TF-IDF + LR) **failed every folklore-axis metric** worse than Baseline-002 (sentence-transformer + LR):

| metric | Baseline-002 | Baseline-005 | delta |
|---|---|---|---|
| K1 folklore F1 in-dist | 0.42 | 0.30 | −0.12 |
| K3 cross-corpus F1 | 0.36 | 0.22 | −0.14 |
| cross-corpus recall | 0.17 | 0.067 | −0.10 |

That result motivated the claim "classical bag-of-tokens cannot pick up the dark-core signature; embedding-based methods can at least partially." n=1 evidence at submission time of Baseline-005.

Baseline-006 (char-level TF-IDF, same classifier head, same training corpus) is the n=2 test.

## Pre-stated bars (the prediction)

| outcome | what it would mean |
|---|---|
| **K3 < 0.30 and recall < 0.17** | n=2 confirmation: classical bag-of-tokens at the word level *and* char level both fail to see the folklore signature. Strengthens the "semantic features required" claim. |
| **K3 ≥ 0.36 and recall ≥ 0.17** | char-level partially refutes the n=1 finding from Baseline-005. The "word abstraction is wrong" claim is partially supported; the dark core may be visible to sub-word features after all. The "semantic features required" claim weakens. |
| **K3 between 0.30 and 0.36** | char-level beats word-level but doesn't reach embedding-level. Suggests char-level captures *some* signal that word-level misses (perhaps cultural-marker n-grams) but not all of what embeddings see. Genuinely interesting middle outcome. |

## Honest prior

Best guess before running: char-level will land in the middle outcome region (0.30–0.40 K3), slightly better than Baseline-005 but worse than Baseline-002. Reasoning: char-level catches morphological markers like "rabbit's foot" or "Marie Antoinette" that word-level tokenization fragments, but sub-word features still don't encode semantic categories (the council's "competitor availability" is a property of training-data semantics, not surface morphology).

## Not re-running, not re-tuning

The vectorizer config is locked: `analyzer="char_wb"`, `ngram_range=(3, 5)`, `min_df=1`, `max_df=1.0`, `sublinear_tf=True`, `norm="l2"`. The classifier config is identical to Baseline-005. The training corpus is unchanged. Run once.

This document is committed to origin before the `styxx gauntlet` invocation on Baseline-006, verifiable from git history.
