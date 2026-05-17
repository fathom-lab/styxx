# Corpus-Coverage Law — High-Resolution Replication: FAILS Strict Criterion (2026-05-17)

**Status:** honest walk-back. Higher-resolution replication of the
2026-05-17 "LAW SUPPORTED" result FAILS the preregistered replication
criterion. The core effect survives as a *threshold*; the prior "clean
law / strongest result / paper-shaped" framing was too strong and is
corrected here.
**Script:** `scripts/dogfood/corpus_coverage_law_fine.py`
**Raw:** `scripts/dogfood/out_corpus_coverage_law_fine.json`

## Preregistered (before run)

12 size-controlled (n=360) label-free corpora, domain-fraction graded
0.00→1.00 + pure-narrative anchor; FIXED cached behavioral labels (no
new generation). H1': cross-family Spearman(overlap,AUC) ≥ 0.60.
H2': same-family |Spearman| < 0.40 (flat control). Replication holds
iff H1' ∧ H2'.

## Result

- Cross-family (mpnet) Spearman = **+0.692** → H1' PASS. AUC rises
  0.687 (overlap 0.184) → 0.847 (overlap ~0.36). **Sufficiency
  threshold located: overlap ≥ 0.31 ⇒ cross-family AUC ≥ 0.80.**
- Same-family (te3-small) Spearman = **−0.406** → **H2' FAIL**
  (preregistered limit 0.40; mild but real negative drift, not the
  clean null claimed at 5 points).
- Preregistered verdict: **REPLICATION FAILED** (H2' not met).

## Honest interpretation

1. **The operational core survives.** Cross-family transport needs
   corpus↔domain overlap ≳0.31; below it (narrative/generic corpora,
   overlap 0.18–0.22) AUC collapses to ~0.69. This threshold is real
   and is a usable pre-flight diagnostic.
2. **It is a threshold, not a smooth law.** The mean-max-cosine metric
   *saturates*: every domain-containing corpus clusters at overlap
   0.31–0.37. The relationship is a step (low vs high), not a graded
   dose-response. The prior +0.83 Spearman was partly an artifact of
   the 5 levels spreading overlap better; at resolution it is +0.69
   and step-shaped.
3. **The control is confounded.** Same-family AUC drifts mildly
   *down* with overlap (ρ=−0.41). The effect is strongly
   cross-family-dominant (swing ~0.16 vs same-family ~0.02) but **not
   cross-family-exclusive** as previously claimed.

## Correction to the record

The 2026-05-17 prior note (`corpus-coverage-law-2026-05-17.md`) and the
session summary called this "LAW SUPPORTED … strongest, cleanest result
… genuinely paper-shaped." **That was an over-claim.** Corrected status:
a real cross-family overlap **threshold** with (a) a saturating metric
that can't resolve the high regime, (b) a confounded same-family
control, (c) a weaker correlation than reported, (d) a FAILED strict
preregistered replication. Discovered by replicating our own result at
higher resolution — which is exactly why you replicate.

## What still stands (unchanged)

- Same-family universal cognometric transport: validated (refusal AUC
  1.000 clear / 0.89–0.94 live). Unaffected.
- Cross-family overlap **threshold** (~0.31): real, operational, useful
  as a deployment guardrail — just not the clean graded law claimed.
- All prior negatives stand (zero-paired closed; Brick #1 null;
  instrument-agnostic bounded).

## Gates to a real paper (now LONGER, honestly)

1. A better overlap metric that resolves the high-overlap regime
   (mean-max-cosine saturates).
2. Understand/clean the same-family control drift (why does same-family
   degrade slightly with higher-overlap corpora?).
3. More sub-threshold points to characterize the step precisely.
4. Cross-vendor (still blocked on a 2nd-vendor key) — the real
   external-validity gate.

No Zenodo/OSF. Further from paper-grade than implied last turn. The bar
holds; the honest distance to it is longer than I said.
