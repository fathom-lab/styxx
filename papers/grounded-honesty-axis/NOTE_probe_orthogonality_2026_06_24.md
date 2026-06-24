# A validated probe can be orthogonal to the concept: a cautionary result for representation-based AI oversight

*styxx / fathom-lab — technical note, 2026-06-24. Offline, two open 3B models, fully reproducible.*

## Abstract
AI-safety increasingly validates oversight by training **linear probes on model activations** to read latent
properties — truthfulness, deception, harmfulness — on the premise that a model's representation reveals what
its text can hide. We show a failure mode that survives the field's standard validation. On a controlled
true/false statement construct, a mass-mean probe reaches **0.98 leave-one-domain-out** accuracy and passes an
**adversary-fair text-silence gate** (a bag-of-words classifier is at exactly chance, 0.50, so the signal is
genuinely not lexical). Yet that 0.98 direction is **orthogonal** (cosine ≈ 0) to the model's actual truth
axis and predicts truth on **natural** statements no better than a random direction (permutation p = 0.24–0.63).
The failure is **robust to construct width** (6 → 14 domains, n = 25 → 61, permutation-tested) and is invisible
to every standard check; it is exposed only by (a) out-of-distribution transfer to *natural* statements and
(b) a cosine check against a direction fit on natural data. We recommend both as a required validation battery
for any probe used in oversight.

## Setup
- **Models:** Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct (reader-side, last-token residual stream, one layer).
- **Controlled construct:** template true/false statements (e.g. *"The capital of France is Paris/Lyon"*),
  cyclic-derangement so each answer token is **balanced** across true/false → text carries no usable truth
  signal by construction. Silence verified: adversary-fair `max(AUC, 1-AUC)` bag-of-words, leave-one-domain-out
  = **0.501**.
- **Natural OOD test:** 70 hand-curated *misconceptions* (familiar-but-false) and *surprising-truths*
  (unfamiliar-but-true), then independently fact-checked by three agents; **9 contested items dropped** (e.g.
  the Mpemba effect, "Eiffel Tower grows >15 cm", honey-never-spoils, blue-whale-largest-ever) → **61 verified**.
- **Probes:** mass-mean (difference-of-class-means; rank-1, cannot overfit) and L2-logistic. Controls:
  label-shuffle, 1000× permutation null, PCA, cosine to the natural-data direction.

## Results
**1. The 0.98 probe is orthogonal to the concept.** In-construct leave-one-domain-out AUC 0.977 (Qwen) /
0.984 (Llama). But applied to natural OOD statements: AUC 0.55 / 0.67, with permutation-null p = 0.63 / 0.24 —
**not above a random direction.** Cosine between the construct direction and the natural-data truth direction:
**−0.05 / +0.14** (orthogonal).

**2. The truth axis exists — but only natural data recovers it.** A direction fit on natural statements
separates held-out natural statements at **0.88 / 0.94** (leave-one-out; permutation p ≤ 0.003). The model
linearly encodes truth; the construct simply finds a different, orthogonal direction.

**3. Widening the construct does not help.** Expanding 6 → 14 template domains and n = 25 → 61 leaves the
construct direction orthogonal (cosine ≈ 0) and non-transferring (permutation-tested). The fix is not more
template domains.

**4. The general axis is model-dependent.** For Llama the natural-data direction generalizes back to the
templates (0.87) while the template direction fails on natural data (0.66) — a clean "general vs surface"
asymmetry. For Qwen *neither* direction transfers to the other distribution (natural→template 0.58, cosine
−0.06): truth is encoded in a distribution-specific way, and no single linear axis spans both. So the strong
constructive claim ("fit on natural data and you recover the general axis") holds for one model and only
partially for the other — reported honestly.

## What this means for oversight
Representation-based oversight is only as trustworthy as the probe's *validity* — whether it tracks the
concept or a correlated artifact. We show a probe can pass **high accuracy, cross-domain generalization, and a
verified text-silence gate** and still be orthogonal to the concept, failing silently exactly where it would
be deployed (natural inputs unlike the construction set). A "deception detector" validated this way could read
template surface and miss real deception.

**Recommended validation battery (the actionable contribution):**
1. **Natural-OOD transfer, permutation-tested.** Test the probe on natural statements unlike the construction
   set; compare to a permutation null of shuffled-label directions (the `max(AUC,1-AUC)` floor is well above
   0.5 at small n — a naive 0.5 baseline overstates significance).
2. **Orthogonality check.** Compute cosine between the construct-fit direction and a direction fit on natural
   data. Near-zero cosine means the construct found a surface artifact, no matter how high its in-construct AUC.
3. **Silence is necessary, not sufficient.** A passed text-silence gate does *not* certify the probe found the
   concept — both of our orthogonal directions came from a silence-gated construct.

## Related work
LLM activations linearly encode truth in-distribution (Azaria & Mitchell 2023; Marks & Tegmark 2023; Bürger
et al. 2024). The probing-pitfalls literature (Hewitt & Liang 2019 control tasks; Belinkov 2022) cautions that
high probe accuracy can reflect probe capacity or spurious structure. Our contribution is specific and
oversight-relevant: a construct that passes a *silence gate* and *cross-domain* generalization can still yield
a direction *orthogonal* to the concept, the failure is *robust to construct width*, and we give a concrete
transfer-plus-orthogonality battery that detects it.

## Limitations
Two open 3B models; the "wide" construct is 14 *template* domains (a frontier-scale test wants thousands of
natural statements and larger/frontier models); single seed; linear probes; OOD n = 61. The natural-data
direction's generality is model-dependent (clean for one model, partial for the other). All claims are
permutation-tested and the ground-truth OOD set is multi-agent fact-verified; every number reproduces offline.

*Reproduce: `scripts/{build_controlled_truthset,build_wide_truthset,build_ood_naturals,truthset_probe,truth_diligence,negation_augmented_truth,truth_axis_settling}.py`. Pre-registrations and full findings under `papers/grounded-honesty-axis/`.*
