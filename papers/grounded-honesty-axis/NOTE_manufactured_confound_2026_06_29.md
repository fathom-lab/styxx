# NOTE — two ways a confound gets manufactured (deepening the substrate-artifact thesis)

*fathom-lab · 2026-06-29 · extends [FINDING_groundtruth_substrate_artifact_2026_06_27](FINDING_groundtruth_substrate_artifact_2026_06_27.md).
Repro: `manufactured_confound_repro.py` (numpy + scikit-learn only; no model download, no network).*

The parent finding showed that an LLM-generated "orthogonal" eval corpus can manufacture the confound it is
used to test for, and that a label-recovering bag-of-words is the artifact's *fingerprint* rather than a
validity control. Going deeper, we pin down **two distinct, independently sufficient mechanisms** by which a
significant confound signal appears with **zero** real model bias — one in the data, one in the auditor itself.

## A — the generator entangles the confound LEXICALLY (first-order)

On styxx's bundled, LLM-generated sentiment boundary corpus, we fit a binary bag-of-words logistic model and,
for every construct-discriminative token, measure how its **presence** co-varies with length *within each
label class*. The positive-discriminative vocabulary systematically appears more in longer texts:

| token | construct coef | within-label corr(presence, length) |
|---|---|---|
| `perfectly` | +0.82 | **+0.63** |
| `solid` | +0.74 | +0.43 |
| `pleasant` | +0.82 | +0.23 |
| `generally` | +1.38 | +0.19 |
| + hedges/function words | (`not`, `without`, `for`, `in`, `that`) | +0.3–0.6 |

Net over construct-strong tokens: **+0.42 → long ⇒ positive.** Asked for "long positive / short negative,"
the generator wrote its longer positives with specific positive-and-hedge vocabulary (`perfectly`, `solid`,
`pleasant`, `generally`, `not …`, `without …`). So *within each label*, longer text carries more
positive-discriminative tokens — and any reader of the words (a dumb lexicon, or the audited classifiers) maps
long ⇒ positive. The "orthogonal" grid was lexically **non-orthogonal**; that is the concrete origin of the
report card's phantom "longer reviews score more positive."

## B — the validator manufactures the confound via its representation (second-order)

We then audit the **auditor**. Build a TRUE NULL: synthetic texts where the construct-word count is drawn
**independently** of total length within label (construct ⟂ length by construction). Sweep length variance and
compare two construct-recoverability probes — the conventional TF-IDF (L2-normalized) margin vs a
length-invariant binary/`norm=None` margin — on the within-label correlation of the margin with length:

| length CV | conventional L2-tfidf | length-invariant (binary, norm=None) | BoW-AUC |
|---|---|---|---|
| 0.22 | corr **0.437**, perm-p 0.002 — FALSE-POS | corr 0.121, p 0.142 — ok | 1.000 |
| 0.38 | corr **0.605**, perm-p 0.002 — FALSE-POS | corr 0.064, p 0.641 — ok | 1.000 |
| 0.47 | corr **0.658**, perm-p 0.002 — FALSE-POS | corr 0.029, p 0.915 — ok | 1.000 |
| 0.50 | corr 0.643, perm-p 0.002 — FALSE-POS | corr 0.042, p 0.783 — ok | 1.000 |
| 0.54 | corr 0.638, perm-p 0.002 — FALSE-POS | corr 0.056, p 0.691 — ok | 0.991 |

The conventional probe fabricates a within-label length signal that **grows with length variance** (perm-p
0.002 throughout) under *zero* true entanglement, because L2 normalization dilutes the construct-token weight
as filler accumulates, mechanically coupling the margin to length. The length-invariant probe stays at the
null. Crucially, **BoW-AUC ≈ 1.0 the whole time** — the construct is fully recoverable — so high
construct-recoverability says *nothing* about confound-orthogonality, and the standard probe doesn't just fail
to validate the corpus, it invents the confound.

## The deepened principle

A significant "confound" can be produced with no model bias by (A) the generator entangling construct
vocabulary with the confound, or (B) the validator's representation coupling its own score to the confound.
Both clear the usual sanity check (a refit recovers the label). Therefore:

1. **Construct-recoverability (BoW AUC) is not a validity control** — necessary at best, and high precisely
   where these artifacts live.
2. The entanglement diagnostic must be **length/confound-invariant** (binary, `norm=None`), **permutation-
   tested** (within-label shuffle null), and **shuffle-folded** — which is exactly the design shipped in styxx
   7.23.0 `_lexical_entanglement` (it is the only probe in Part B that returns the correct null).
3. The decisive gate remains **ground truth**: re-run the audit on length-matched real human labels
   (`styxx.validate_against_ground_truth`). Neither (A) nor (B) survives it.

This is the same thesis at a second order: we manufactured a confound, then caught our own auditor
manufacturing one — and the fix that survives both is the one now in the tool.
