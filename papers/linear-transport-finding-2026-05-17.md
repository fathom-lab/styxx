# Linear Transport of a Cognometric Instrument Across Embedding Spaces

**Date:** 2026-05-17
**Status:** directional dogfood — NOT paper-grade (see Limitations)
**Script:** `scripts/dogfood/universal_directions_linear_transport.py`
**Raw:** `scripts/dogfood/out_universal_directions_linear_transport.json`

## Question

mini-vec2vec (arXiv 2510.02348) / vec2vec (NeurIPS 2025) report that the
map between text-embedding spaces is approximately *linear* and recoverable
without paired data. Does that hold for *cognometric directions* — i.e. can
the styxx refusal instrument be fit once in a home space and linearly
transported into other embedding spaces (including closed models / other
families) with no weights, no retraining, no labels?

## Method

Home space A = `text-embedding-3-large` (3072d); diff-of-means refusal axis
fit on the 20 obvious eval prompts only (the shipped probe). Transport
`M: B -> A` learned **only** from a 435-sentence generic corpus, disjoint
from the eval, with **no behavior labels**. Scored as AUC vs the saved
behavioral refusal of `gpt-4o-mini` / `gpt-4.1-mini` from the 2026-05-14
closed-model run (no new model calls).

## Results (AUC vs closed-model behavioral refusal; gt = clear cases only)

| transport into | naive | paired-ridge | paired-procrustes | unpaired-procrustes | native ceiling |
|---|---|---|---|---|---|
| text-embedding-3-small (1536d, OpenAI) | 0.585 | 0.890 (gt 1.000) | **0.935** (gt 1.000) | 0.365 | 1.000 |
| all-mpnet-base-v2 (768d, different family) | 0.300 | 0.860 (gt 1.000) | **0.885** (gt 1.000) | 0.595 | 1.000 |

## Findings

1. **Label-free linear transport works, including cross-family.** An
   orthogonal map learned from generic unlabeled text moves the instrument
   into a different model family and still perfectly separates clear
   refuse/comply (AUC 1.000) and predicts live closed-model refusal at
   0.885–0.935. Naive direct transfer (0.30–0.59) is at/below random,
   confirming the transport is doing the work.

2. **Fully-unpaired (zero-correspondence) transport FAILED: 0.365 / 0.595.**
   The lightweight Conneau-style NN-Procrustes proxy did not recover the
   alignment on a small generic corpus. Honest negative. Consistent with
   the literature (real vec2vec/mini-vec2vec require more machinery), but
   the strong "from nothing" claim is **not demonstrated**.

## Verdict

Shippable/citable claim: *label-free linear transport from a styxx-owned
generic corpus* (one instrument → any embedding space, no labels/weights/
retraining). NOT yet: zero-paired-data transport — requires a real linear
mini-vec2vec port before any claim.

## Limitations

n=30 eval (AUC quantised, noisy); unpaired arm is a lightweight proxy, not
a mini-vec2vec reimplementation; corpus is generic English; behavioral
labels reused from 2026-05-14. This artifact decides whether the
linear-transport upgrade warrants a paper-grade build; it is not that paper.

## Next

1. Port real linear mini-vec2vec (whitening init + cycle term, ≥5k corpus);
   retest the unpaired arm.
2. Repeat transport for sycophancy / deception instruments.
3. If (1) clears: universal-cognometric-transport paper + closed-model
   independent-audit wedge (EU AI Act GPAI external-evaluator obligation,
   enforcement 2026-08-02).
