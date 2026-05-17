# Zero-Paired-Data Cognometric Transport — Honest Negative (2026-05-17)

**Status:** negative result. Records a boundary, not a breakthrough.
**Script:** `scripts/dogfood/universal_directions_unsupervised_transport.py`
**Raw:** `scripts/dogfood/out_universal_directions_unsupervised_transport.json`

## Question

Can a styxx cognometric instrument transport into a foreign embedding
space with **zero paired data**, using the real unsupervised linear
alignment pipeline (vec2vec / MUSE line: center → PCA → ZCA-whiten →
CSLS → mutual-NN → iterative Procrustes → multi-restart with
unsupervised model selection) — not the lightweight NN-Procrustes proxy
that failed on the morning of 2026-05-17?

## Result

| transport into | zero-paired AUC | gt-clear | prior proxy | native ceiling |
|---|---|---|---|---|
| text-embedding-3-small | 0.600 | 0.500 | 0.365 | 1.000 |
| all-mpnet-base-v2 | 0.615 | 0.690 | 0.595 | 1.000 |

The aligner recovered ~780–790 mutual pairs with a positive unsupervised
selection score — it **did** align generic semantic geometry — but the
transported refusal axis is at/near chance on the clear cases
(gt-clear 0.500 for te3-small). The instrument is intact in-space
(native ceiling 1.000); the **alignment does not preserve the behavioral
direction**.

## Conclusion

**Two principled methods now agree: zero-paired-data cognometric
transport does NOT work with a single linear orthogonal map on a modest
generic corpus.** This is consistent with the 2026 literature — global
representational convergence is being walked back; only *local* semantic
structure transfers cheaply; vec2vec reaches its result with adversarial
+ cycle-consistency machinery and large in-distribution corpora, which a
linear map deliberately omits.

This bounds the claim cleanly:

- **Validated (2026-05-17 AM):** *paired, label-free* linear transport
  works — AUC 1.000 on clear cases, 0.885–0.935 vs live closed-model
  refusal, including cross-family. One instrument → any space, no
  labels, no weights, no retraining, given a generic corpus embedded
  through both encoders (no model cooperation needed — you only need to
  embed text).
- **Open problem (this note):** *zero-paired-data* transport. Requires
  vec2vec-grade nonlinear/adversarial machinery and larger
  in-distribution corpora. A separate, heavier research bet — not a
  linear-map dogfood.

## Why the negative is useful

A documented boundary strengthens the eventual paper. The novel
instrument is "**universal cognometric transport (paired, label-free)**"
with this exact failure mode stated as its honest limit. Overclaiming a
zero-shot version would have been refuted; this closes that door
deliberately.

## Caveats

n=30 eval (AUC quantised/noisy); corpus 2,715 generic+concept sentences
(smaller and less in-distribution than vec2vec-scale); orthogonal-linear
map only by design; behavioral labels reused from 2026-05-14.

## Next (honest)

1. Package the **validated paired transport** as a real styxx module
   (`styxx.transport`) — converts the proven result into the "new tool"
   half of the paper bar. Correct, honest, shippable.
2. Treat zero-paired-data transport as a scoped research project (real
   nonlinear vec2vec + ≥50k in-distribution corpus), not a dogfood.
3. Paper = paired universal cognometric transport + this boundary. Only
   then Zenodo/OSF (publishing bar; DOI is irreversible).
