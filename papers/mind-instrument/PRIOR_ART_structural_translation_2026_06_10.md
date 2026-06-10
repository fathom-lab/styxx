# Prior art — structural translation between minds (due diligence, 2026-06-10)

**Fathom Lab / styxx. Companion to FINDING_anatomy_v2, FINDING_gavagai_v0, and the telepathy-v0
arc. Purpose: name everyone who has been close, what they showed, what they required — and state
the deltas checkably. Web-verified 2026-06-10.**

## The nearest neighbors

| work | what they showed | what they required / didn't do |
| --- | --- | --- |
| **vec2vec** — Jha, Zhang, Shmatikov, "Harnessing the Universal Geometry of Embeddings" (arXiv:2505.12540, 2025) | Unsupervised translation between TEXT-EMBEDDING spaces (CycleGAN-style adapters into a shared latent), no paired data; strong cross-model cosine fidelity; security implications | Trains a neural translator on large unpaired corpora; operates on purpose-built embedding models, not causal-LM internal states; no pre-registration; no novel-concept relational channel |
| **Relative representations** — Moschella et al. (ICLR 2023, arXiv:2209.15430) | Representing points by similarity to ANCHOR samples enables zero-shot latent communication and model stitching across architectures | Anchor CORRESPONDENCE is given (same labeled anchors in both spaces). Our oracle-alignment ceiling (novel-concept decode 1.0 cross-model) reproduces their core insight inside causal-LM concept geometry; our addition is bootstrapping the correspondence with NO labels |
| **MUSE** — Conneau et al. 2017; **GW alignment** — Alvarez-Melis & Jaakkola (EMNLP 2018, arXiv:1809.00013) | Word translation across LANGUAGES without parallel data (adversarial alignment; Gromov-Wasserstein over word-embedding metric structure) | Static word embeddings within one embedding technology; large vocabularies needed; cross-LINGUAL not cross-MIND; no certified pipeline |
| **Blind Match** — (CVPR 2025, arXiv:2503.24129) | Vision-language correspondence WITHOUT parallel data as a quadratic assignment problem on pairwise distances — methodically the closest assignment-from-structure relative | Cross-modal encoder embeddings; small solved instances; no causal-LM internal geometry, no apparatus correction, no novel-content transmission |
| **Platonic Representation Hypothesis** — Huh et al. 2024; **Representational Alignment Hypothesis** (arXiv:2602.16584, 2026) | Convergence of representations across models/modalities (the WHY behind any of this working) | Convergence claims, not translation protocols; no identity-recovery rates, no transmission channel |
| **Rogue dimensions** — Timkey & van Schijndel (EMNLP 2021); **Massive activations** — Sun et al. 2024; **attention sinks** — Xiao et al. 2023, ICLR-2025 empirical study (arXiv:2410.10781) | A few dimensions/tokens carry extreme norms and distort similarity measures; first-token states have outsized norms | Dimension- and token-level diagnosis. Nobody applied it to TEMPLATE-AVERAGED concept geometry, showed it suppresses measured cross-model convergence 2-60x, or showed that per-template L2 normalization heals reliability (0.29 -> 0.94) and quadruples unsupervised translation (0.044 -> 0.166) |
| **Hyperalignment** — Haxby et al.; **RSA** — Kriegeskorte 2008 | Cross-SUBJECT functional alignment of human brains; second-order similarity as the lingua franca of mind comparison | The neuroscience ancestors of everything here; human brains, supervised/functional alignment, no label-free identity recovery bars |

## What tonight's arc adds (each clause checkable against receipts)

1. **The norm-domination artifact in template-averaged LM concept geometry** — bare-prompt states
   ~7x the norm of contextual ones dominate the unweighted average; healing it (per-template L2)
   raises split-half reliability 0.291→0.9411 and reveals cross-family convergence of 0.61–0.85 in
   every domain (`anatomy_v1_result.json`, `FINDING_anatomy_v2_2026_06_10.md`). Adjacent fields
   diagnosed token/dimension outliers; the template-averaging consequence and its repair appear
   unpublished.
2. **Training-free, label-free concept-identity recovery between CAUSAL-LM internal geometries**
   with pre-registered bars and an identity-decoupled null — 16x chance cross-family
   (`FINDING_gavagai_v0_2026_06_10.md`). vec2vec translates embedder spaces with a trained network;
   Blind Match solves cross-modal QAP; MUSE/GW align word embeddings cross-lingually. None recover
   concept identity between independent LLMs' hidden-state geometries, training-free, under frozen
   gates.
3. **Novel-concept transmission over an unsupervised-bootstrapped relational channel**
   (telepathy-v0, in flight): relative-representation messages whose anchor correspondence is
   DISCOVERED, not given — with the oracle ceiling already observed at 1.0 in smoke (perfect
   cross-model transfer of unseen concepts given alignment). Moschella's stitching assumed the
   correspondence; vec2vec learns a translator per pair; this channel is assignment-only and
   training-free.
4. **The certified pipeline** — every claim above is pre-registered before its data existed,
   machine-verified against receipts (OATH), with same-day public self-corrections in the chain.
   No neighbor operates under this discipline; it is what makes "first" claims checkable rather
   than rhetorical.

## What we have NOT shown (so the deltas stay honest)

vec2vec translates arbitrary continuous embeddings at corpus scale — we identify/transmit over a
96-concept battery and a 96-candidate decode set. Relative representations achieve high TASK
accuracy with given anchors — we measure identification, not downstream tasks. Cross-lingual BLI
operates over 100k+ vocabularies — our scale is 2-3 orders smaller. Telepathy-v0's verdict (and its
abstract-content profile) is pending; until its gates pass, only the smoke observation exists.
Scaling the channel to open vocabularies is the standing rung, not a result.
