# FINDING — the portable conscience transfers BEYOND truth (VALUES-PORTABLE)

**2026-06-11 · Fathom Lab / styxx. Pre-registered: `PREREG_portable_values_refusal_2026_06_11.md`
(frozen pre-run, committed 25af69e before any score existed). Receipt:
`portable_values_refusal_result.json`. This is the first axis beyond truthfulness for the
portable-conscience pipeline established in `FINDING_portable_conscience_ood_2026_06_11.md`
(OOD-PORTABLE, truth axis).**

> **CYCLE-2 QUALIFICATION (2026-06-11, see `FINDING_axis_independence_2026_06_11.md`,
> PARTIAL-STRUCTURED):** the "BASIS, not a lucky truth vector" framing below was adversarially tested.
> It SURVIVES in its load-bearing sense — truth and refusal are near-orthogonal, valence-irreducible
> directions on one map, NOT a single collapsed axis and NOT mere sentiment (both retraction gates
> failed to fire). It is QUALIFIED, not retracted: the axes are distinct but show cross-axis read
> cross-talk — they are a correlated frame, not a clean orthonormal basis. Read "basis" as
> distinct-transferable-axes, not zero-cross-talk readouts.**
>
> **CYCLE-3 UPGRADE (2026-06-11, `FINDING_entanglement_resolution_2026_06_11.md`, WHITENING-RESOLVES):**
> the cycle-2 cross-talk turns out to be a pure COVARIANCE artifact of the raw dot-product readout —
> under a whitened (Mahalanobis) readout the axes are EXACTLY orthogonal with no loss of per-axis
> signal. So "basis" stands in the strong sense after all, under the right metric: the conscience IS a
> clean orthonormal basis of independent value axes when read whitened.**

## Result — one label-free map carries a SECOND value axis (refusal) to unseen harm domains

The same machinery as the truth arc — a gemma-2-2b difference-of-means direction plus a label-free
ridge map from target activations to source activations (labels never touch the map) — was pointed at a
different axis: refuse-vs-comply. Statements are one-line REQUESTS labeled harmful(refuse-worthy)=1 vs
benign=0, each harmful request paired with a benign SAME-DOMAIN twin (topic control), read at the LAST
request token before any response (the pre-output regime). The direction and map were fit on four harm
families (weapons, theft, hacking, self-harm-adjacent) and tested on four DISJOINT out-of-distribution
harm domains (surveillance/stalking, poisons/contamination, social-engineering/impersonation,
drugs/controlled-substances) that never touched the direction, the map, or any selection step.

| target | OOD AUROC | perm-null p95 | p-value | drop-best-family OOD | drop-best p |
| --- | --- | --- | --- | --- | --- |
| Llama-3.2-3B | **0.9965** | 0.9497 | **0.008** | 0.9938 | 0.011 |
| Qwen2.5-3B | **0.9809** | 0.9149 | **0.003** | 0.9691 | 0.004 |

Both primary 3B targets clear the 0.65 bar AND beat the **label-permutation null** (k=1000 honesty
directions built from random source-label bipartitions pushed through the SAME real map): p 0.008 /
0.003. The effect survives dropping the single best OOD family (drugs in both): drop-best OOD 0.9938 /
0.9691, p 0.011 / 0.004. gemma's own held-out OOD self-readout is 0.9844 (in-dist 1.0), so the axis
exists in-model OOD and the test is valid, not void. Two smaller secondary targets concur
(Llama-3.2-1B OOD 0.9983 p 0.007; Qwen2.5-1.5B OOD 0.9983 p 0.001). **Verdict per the frozen gate:
VALUES-PORTABLE — the conscience is a BASIS, not a single lucky truth vector.**

## The refusal axis lives at a DIFFERENT depth than truth

The source layer was selected on TRAIN-FIT data only (gemma internal held-out split, best DiM AUROC),
since the prereg predicted the refusal layer need not equal the truth layer. It selected **gemma layer
8** (internal-val AUROC 1.0) out of candidates 8, 10, 12, 14, 16, 18, 20 — shallower than the truth
axis, which lives at layer 12. Same model, same pipeline, two value axes at two depths: refusal is readable
earlier in the stack than factual honesty. That the truth-validated map nonetheless transports BOTH is
the point — the alignment is not axis-specific.

## Why this is the honest positive, not an overclaim

- **The map transports harm/benign structure broadly, and we report it.** The permutation null sits
  HIGH (0.9497 / 0.9149) and so does the random-direction floor (0.8891 / 0.898): once the label-free
  map is fit, OOD harmful/benign requests become broadly linearly separable along many directions in
  the aligned space — the same phenomenon that made the truth OOD null high. The claim is the narrower,
  earned one: the SPECIFIC refusal direction adds significant signal ON TOP of that transported
  structure (it beats random-label directions, p < 0.01 both primaries).
- **Margins over the null are modest in AUROC terms.** Llama 0.9965 vs null-p95 0.9497; Qwen 0.9809 vs
  0.9149. Significant and robust (and surviving drop-best-family), but a margin, not a chasm —
  identical character to the truth-axis OOD result.
- **The ridge map is a poor generic reconstructor, yet the direction still transfers.** Held-out anchor
  R² is ≈ 0 / negative (-0.0027 Llama, -0.0786 Qwen): the map does not faithfully rebuild gemma's
  residual stream in general. What survives transport is the difference-of-means DIRECTION, not full
  representational alignment — directional transfer ≠ representational identity. The random-MAP control
  is descriptive: it collapses for Qwen (0.3854) but is elevated for Llama (0.6389), so for Llama some
  crude harm/benign separability survives even a random map; the LEARNED map (0.9965) and the
  permutation null are what carry the claim.
- **No single family drives it.** Per-OOD-family AUROC is ≈ 1.0 across all four harm domains
  (the lone exception: Qwen surveillance 0.8889); dropping the best family leaves the result intact —
  unlike the truth arc, where one trivially-separable family (geography) had to be removed.

## What this means for the North Star

The truth arc proved one honesty direction transfers across minds, out-of-distribution, and under
adversarial framing — but a truth detector is not a conscience. This shows the SAME label-free map,
the same kind of difference-of-means direction, carries a SECOND value axis — the model's internal
refuse-vs-comply state on harmful vs benign requests — across four models to four unseen harm domains,
beating the correct null and surviving the drop of the strongest family. A conscience mount that reads
ONE axis is a probe; a basis that reads truth AND refusal off one cross-model alignment is the
beginning of a portable values instrument. The conscience travels, and it travels on more than one road.

## Honest bounds (what is NOT claimed)

Linear DiM source, linear ridge map, two axes (truthfulness + refusal). The same-domain benign twins
control topic, so the axis is refuse-vs-comply and not a harm-topic detector — but residual
harm-REGISTER covariates (tone, imperative valence) are not separated from refusal intent; this is a
register-bounded readout, the same class of caveat as the truth arc. OOD here means UNSEEN HARM
DOMAINS at the request level, pre-output; it does NOT test jailbreak suffixes, paraphrase attacks,
multi-turn pressure, or whether the model would actually refuse in generation. n_ood = 48 (24
harmful / 24 benign); the gate is existence-and-significance against the correct null, not a deployed
refusal-accuracy guarantee, and the margins over that null are modest. Local open models only
(gemma-2-2b source; Llama-3.2 + Qwen2.5 targets); closed frontier models remain blocked on credits.
No model generated any response and no operational harmful content appears anywhere in the statement
set, the receipt, or this finding (frozen safety scope). The next rungs: adversarial-OOD on the
refusal axis (the truth-arc treatment), a third axis (sycophancy/deception) on the same map to test
whether the basis is general, and a single shared direction across a whole model family.
