# A rank-1 probe can pass accuracy, cross-domain generalization, and a silence gate — and still be orthogonal to the concept

*styxx / fathom-lab — technical note, 2026-06-24. Offline, two open 3B models, fully reproducible. This is a
small-scale replication-plus-packaging, honestly scoped — not a new phenomenon.*

**▶ Try it in 60s (one-click Colab):**
https://colab.research.google.com/github/fathom-lab/styxx/blob/main/examples/probe_validity_colab.ipynb
— run `styxx.validate_probe` on a planted surface artifact, then on a real model, then on your own probe.

## Abstract
Representation-based oversight trains linear probes on activations to read latent properties (truthfulness,
deception). It is known that such probes can latch onto surface features rather than the concept, especially
in smaller models (Marks & Tegmark 2023, Fig. 3b; Levinstein & Herrmann 2023). We add one specific wrinkle and
one piece of packaging. **The wrinkle:** the artifact survives *three* checks a practitioner might treat as
sufficient together — high leave-one-domain-out accuracy (mass-mean **0.977 / 0.984** on Qwen2.5-3B /
Llama-3.2-3B), cross-domain generalization, **and** an adversary-fair text-silence gate (bag-of-words at
chance) — while using a **rank-1 difference-of-means probe that structurally cannot overfit** and therefore
passes Hewitt & Liang (2019) control-task selectivity. The direction is nonetheless orthogonal to the model's
natural-data truth direction (cosine **−0.05 / +0.14**) and does not transfer to natural statements (Qwen
fair-AUC 0.55, permutation p=0.63 — at the random-direction floor; Llama 0.67, p=0.24 — non-significant but
underpowered, CI [0.52, 0.80]). **The packaging:** we combine the field's standard cross-dataset transfer test
with a direction-cosine check and a permutation-correct floor into one pre-deployment battery, and ship it as
a tool. Scope is small (two 3B models, template constructs, n=61 natural test); we do not claim a new failure
mode, only a clean small-model demonstration and a reusable check.

## Setup
- **Models:** Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct (reader-side, last-token residual stream).
- **Probe:** mass-mean (difference of class means; rank-1, cannot overfit) — chosen precisely because it
  passes control-task selectivity. Logistic probes give the same qualitative result.
- **Construct:** template true/false statements, cyclic-derangement so answer tokens are balanced across
  true/false. The narrow construct (136 statements / 6 domains) yields the headline probe; its adversary-fair
  bag-of-words silence is **0.505** (leave-one-domain-out). A wide construct (260 statements / 14 domains)
  tests robustness to width; its silence is **0.501**. Both ~chance.
- **Natural OOD test:** 70 curated misconceptions(false)/surprising-truths(true) → **independently
  fact-checked by three agents**, 9 contested items dropped (Mpemba, "Eiffel >15cm", honey-never-spoils,
  blue-whale-largest-ever, …) → **61 verified** (34 false / 27 true).
- Controls: 1000× permutation null, cosine to the natural-data direction, natural-axis leave-one-out ceiling.

## Results
**1. High in-construct AUC, orthogonal to the concept.** Mass-mean leave-one-domain-out: **0.977 (Qwen, L19) /
0.984 (Llama, L14)**. Applied to natural OOD statements the same direction gives fair-AUC 0.55 / 0.67. Cosine to
the natural-data truth direction: **−0.05 / +0.14** (orthogonal). The high in-construct number is template
structure.

**2. Transfer is not established — clearly null for one model, underpowered for the other.** Against a 1000×
permutation null of shuffled-label directions (the adversary-fair floor is ≈0.6 at n=61, not 0.5):
- **Qwen:** fair-AUC 0.554, **p = 0.63** — at the random-direction floor; cosine −0.05. No transfer.
- **Llama:** fair-AUC 0.667, **p = 0.24**, bootstrap CI **[0.52, 0.80]** — non-significant but **underpowered**;
  the CI does not exclude real transfer. We report this as "cannot reject random," not "equivalent to random."

**3. The concept axis exists — recoverable from natural data.** A direction fit on natural statements separates
held-out natural statements at **0.88 / 0.94** (leave-one-out; permutation p ≤ 0.003). The model encodes truth;
the construct simply finds an orthogonal direction.

**4. Widening the construct does not fix it; the recoverable axis is model-dependent.** Expanding 6→14 template
domains (n 25→61) leaves the construct direction orthogonal and non-transferring on both models. The natural-
data direction generalizes *back* to the templates for Llama (0.87) but the template direction fails on natural
(0.66) — a clean general-vs-surface asymmetry. **For Qwen, neither direction transfers** (natural→template 0.58,
cosine −0.06): truth is encoded distribution-specifically and no single linear axis spans both. So "fit on
natural data and recover the general axis" holds for one model and only partially for the other.

## What this means for oversight
The components here are not new — cross-dataset OOD transfer is the standard check in this literature (Marks &
Tegmark run exactly these transfer experiments and already report approximately-orthogonal truth directions
across datasets), and surface-confounded probes in small models are documented (Levinstein & Herrmann 2023).
Our narrow, defensible point: the artifact **survives a *rank-1* probe** — the one probe class control-task
selectivity (Hewitt & Liang 2019) is designed to clear, since a difference-of-means direction cannot overfit a
control task — *and* an adversary-fair silence gate, *and* cross-domain generalization. So passing
accuracy + cross-domain + control-task-selectivity + silence-gate is jointly insufficient; you still need the
cross-dataset transfer test and a cosine-to-natural-direction check. We package those into one pre-deployment
battery (`styxx.probe_validity`): silence gate + in-construct AUC + natural-OOD transfer with a permutation
floor + orthogonality-to-natural-direction + concept-exists ceiling → a VALID / SURFACE-ARTIFACT verdict. The
assembly and the silence-gate-still-fails demonstration are the contribution; the individual diagnostics are
prior art.

## Related work
LLM activations linearly encode truth in-distribution but with surface confounds and cross-dataset/negation
transfer failures, especially in smaller models: Azaria & Mitchell 2023; Marks & Tegmark 2023 (transfer
experiments; approximately-orthogonal dataset-specific truth directions; small-model surface clustering);
Levinstein & Herrmann 2023 ("Still No Lie Detector"); Bürger et al. 2024. On probe validity generally: Hewitt
& Liang 2019 (control tasks / selectivity); Belinkov 2022 (probing survey). We contribute a small-model
replication where the artifact survives a rank-1 probe + a silence gate, plus a packaged battery.

### References
- Azaria, A. & Mitchell, T. (2023). *The Internal State of an LLM Knows When It's Lying.* EMNLP Findings.
- Marks, S. & Tegmark, M. (2023). *The Geometry of Truth: Emergent Linear Structure in LLM Representations of
  True/False Datasets.* (Cross-dataset transfer experiments; approximately-orthogonal dataset-specific truth
  directions; small-model surface clustering, Fig. 3b.)
- Levinstein, B. & Herrmann, D. (2023). *Still No Lie Detector for Language Models: Probing Empirical and
  Conceptual Roadblocks.* (Negation/generalization failures of truth probes.)
- Bürger, L. et al. (2024). *Truth is Universal: Robust Detection of Lies in LLMs.*
- Hewitt, J. & Liang, P. (2019). *Designing and Interpreting Probes with Control Tasks.* EMNLP. (Selectivity;
  control tasks — which a rank-1 difference-of-means probe passes by construction.)
- Belinkov, Y. (2022). *Probing Classifiers: Promises, Shortcomings, and Advances.* Computational Linguistics.

## Limitations
Two open 3B models; template constructs (a frontier-scale test wants thousands of natural statements and
larger models); single seed; OOD n = 61; linear probes. The Llama transfer result is non-significant but
underpowered, not a demonstrated null. The "recoverable from natural data" claim is model-dependent (clean for
Llama, fails for Qwen). All numbers are permutation-tested, the OOD ground truth is multi-agent fact-verified,
and every figure reproduces offline.

*Reproduce: `scripts/truth_axis_settling.py` (prints the fair-AUC + 1000× permutation-p values cited above),
`scripts/{build_controlled_truthset,build_wide_truthset,build_ood_naturals,truthset_probe,truth_diligence,
negation_augmented_truth}.py`, and `styxx.probe_validity` (the packaged battery). Pre-registrations and full
findings under `papers/grounded-honesty-axis/`.*
