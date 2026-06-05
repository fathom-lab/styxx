# Representational universality — the unified capstone (real-now vs speculative, defense vs attack)

**2026-06-05 · fathom-lab / styxx.** Produced by a 7-agent adversarial assault (internal audit · external
literature · two independent experiment designs · steelman-the-null · theory · implications) synthesized,
fused here with the same-day real-model run (`papers/disjoint-worlds/RESULT_real_universality_2026_06_05.md`).

## The synthesis, in one line
**Representational universality is real but NARROW — exploitable in essentially one place, the refusal/
safety direction — and that single fact has *opposite sign* for defense (an asset) vs. attack (a liability).**

## What the real-model run added (today, fresh)
8 real models (5 decoders trained on different data + 3 sentence-embedders), 100 templated concepts:
**robust shared concept geometry** (RSA mean 0.69, up to 0.88) but **zero-anchor GW recovery fails on every
pair** (max 0.05 ≈ chance), decoders and embedders alike — *falsifying my own embedder-dividing-line
prediction.* Broad concept geometry is shared but not recoverable by simple geometric matching. This is the
empirical floor under the synthesis: the *general* geometry is shared-but-not-blindly-alignable; what IS
exploitable is the narrow, specific safety direction the assault identified.

## Per-angle verdict (real-now vs speculative)
1. **Trust / styxx instrument.** *Real-now:* zero-access refusal prediction on closed models from the prompt
   embedding alone (AUC ~0.95); an observer-model that reads another model's confident errors from
   activations (~0.65, replicated Qwen+gemma). *Speculative / locally-falsified:* "translate a concept of
   deception across models" — our own data say deception/truthfulness are **family-specific** (r~0.30),
   error-self-knowledge is fragmented (off-diagonal AUC 0.474, below chance), and the white-box geometry
   detector is **DEAD** (benign-behavioral confound).
2. **Interoperability.** vec2vec (arXiv 2505.12540 — nonlinear adversarial + cycle) achieves the unsupervised
   embedding translation our LINEAR map failed at twice, and that GW failed at on real models today — exactly
   the "heavy nonlinear bet" the linear-transport memo flagged. Validated scope: **same-family only;
   cross-vendor is pre-registration-killed.**
3. **Attack surface (the sharpest result).** "Jailbreak Transferability Emerges from Shared Representations"
   (arXiv 2506.12913) demonstrates refusal-suppression steering A→B **with no access to B**, on the exact
   Llama/Gemma/Phi families styxx uses. Exposure lands on open-weight / self-hosted agents (Hermes,
   darkflobi) — the same surface styxx's guardian needs, now **symmetric for the attacker.** Benign steering
   also raises jailbreak risk (arXiv 2602.04896).

## The theory that ties it together
The *general* concept geometry is broadly shared but not blindly recoverable (today's real-model floor;
the synthetic near-isometry threshold). The *specific safety/refusal* direction is the exception that IS
cross-model transferable — which is precisely why it is simultaneously styxx's best zero-access signal AND
the attacker's best zero-access lever. Universality is narrow, and the narrow band is dual-use.

## THE decisive next experiment (concrete, $0, weights cached)
Run the **cross-model steering attack on our own atlas**: take a refusal-suppression direction from Qwen,
apply it to **gemma with no labels**, measure attack success — then test whether **styxx's observer probe
detects the steered state**, with a **pre-registered kill-gate against the benign-behavioral-steering
confound** that already killed the geometry probe. This is the experiment where the science (cross-model
transfer) becomes the product (styxx detecting a cross-model attack) in one run.

## Honest scope flag (carried from the synthesis)
The assault did **not** re-verify cited styxx code against current `fathom-lab/styxx` HEAD; several grounding
memory files carry 18–26-day staleness warnings. The empirical *findings* are our own pre-registered
results, but any "currently shipped" / file:line claim must be checked against HEAD before it enters a paper
or the site.

## External anchors
arXiv 2405.07987 (Platonic Representation Hypothesis + its named counterexamples) · 2505.12540 (vec2vec) ·
2506.12913 (jailbreak transfer from shared representations) · 2503.01865 (transfer ASR 18.4→50.3%) ·
2602.04896 (benign steering raises jailbreak risk).
