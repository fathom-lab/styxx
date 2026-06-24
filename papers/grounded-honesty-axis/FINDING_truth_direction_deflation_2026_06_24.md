# FINDING — the "truth direction" deflates: it's surface/plausibility, not truth (fails negation + OOD)

**2026-06-24. Due diligence on `FINDING_truthset_grounding` (the within-construct 0.98 result).** Offline.
Reproduce: `python scripts/truth_diligence.py --extract --model {qwen,llama}` then `python scripts/truth_diligence.py`.

## What was tested
The mass-mean truth direction fit on the controlled silent set hit 0.98 leave-one-domain-out. The classic
threat: is it TRUTH, or FAMILIARITY/surface? Fit the direction on the controlled set, then SCORE three
dissociation probe sets where truth and familiarity DISAGREE.

## Result — it largely fails (both models)
| test | Qwen (L19) | Llama (L14) |
|---|---|---|
| within-construct LODO (in-distribution) | 0.98 | 0.98 |
| **negation polarity** ("Paris is not the capital of France" = FALSE) | **0.000** | **0.000** |
| **misconceptions (familiar-false) vs surprising-truths (unfamiliar-true)** | 0.387 | 0.560 |

- **Negation: perfectly INVERTED (0.000).** The direction ignores "not" entirely — it reads the surface
  tokens (Paris+France) and assigns truth by co-occurrence, so a negated-true statement reads false and
  vice-versa. A real truth representation handles polarity; this does not.
- **Natural true/false (misconceptions vs surprising): chance** (0.39-0.56). The direction does NOT
  generalize from the template construct to natural statements.

## Honest conclusion
The within-construct 0.98 was a **construct-specific surface/plausibility direction**, not a general truth
representation. It generalizes across the 6 template domains (which share structure) but collapses on
negation and natural OOD statements. The earlier "the model's words are at chance while its mind is at 98%"
framing OVERSOLD it — corrected here.

## Why (and what the rigorous version requires)
The mass-mean direction was fit on a construct with NO negations and only 6 templates, so it learned the
template-truth surface correlation, not truth-value. The literature gets negation-robust truth directions
(Marks-Tegmark, Bürger) using LARGER, diverse, negation-INCLUSIVE training. The definitive next test:
fit a direction on negation-AUGMENTED training (so it must encode truth-value not surface), then test OOD
generalization to misconceptions + held-out negations. If a negation-robust direction generalizes -> real
truth axis; if still flat -> these small models encode plausibility, not robust truth.

## Honest scope
2 local 3B models, n=37 probe statements, single seed. The negation failure is unambiguous (0.000 both
models); the misconception result is small-n but flat both ways. 9th self-caught deflation of the session,
and the most important: it corrects the headline I called "most mind-blowing."

## DEFINITIVE follow-up — negation-augmented training (commit appended)
Fit the direction on the controlled set PLUS its negations (labels flipped), so polarity is forced into
training (`scripts/negation_augmented_truth.py`):
| | Qwen | Llama |
|---|---|---|
| in-construct LODO (polarity in train) | 0.947 | 0.968 |
| no-negation dir -> negations (raw) | 0.020 | 0.392 (inverts — confirms deflation) |
| augmented dir -> held-out NEGATION polarity | 1.000 | 1.000 (CAN learn polarity if trained on it) |
| augmented dir -> OOD natural misconceptions-vs-surprising | **0.533** | **0.600** (still FLAT) |
**Definitive: the model can encode polarity when trained on it, but even the polarity-robust direction does
NOT generalize to natural OOD true/false statements. The truth direction is CONSTRUCT-BOUND, not a transferable
truth representation, on these 3B models + this 6-template construct.** Open question (under multi-agent
triangulation): is the OOD failure because the construct is too narrow (6 templates) or because 3B models lack
a transferable truth axis at this scale? The settling experiment = a large diverse negation-inclusive dataset
(Marks-Tegmark scale).

## TRIANGULATION (5-agent ultracode, all converge) — the real, richer conclusion
Five independent probes (method, layer, reverse-direction, calibration, power) agree:
- **The 0.98 construct direction is ORTHOGONAL to the model's real truth axis** — cosine(controlled-template
  direction, OOD-internal truth direction) = **−0.066 (Qwen) / +0.037 (Llama)**. The probe that scores 0.98
  leave-one-domain-out is measuring something unrelated to the concept it claims to measure.
- **A truth axis DOES exist in these 3B models:** natural true/false statements are linearly separable
  IN-DISTRIBUTION at in-OOD leave-one-out **0.833 (Qwen) / 0.913 (Llama)**, permutation **p ≤ 0.003**. The
  models encode truth; the controlled construct just recovered the wrong (orthogonal, surface) direction.
- **Transfer failure is robust** (no method/layer/regularization/negation-augmentation reaches 0.70 OOD),
  BUT honestly **underpowered at n=25** (every OOD bootstrap CI ~0.4 wide, none excludes 0.70). So this
  REFUTES a usable generalizing direction at this construct width; it cannot, on power alone, prove no axis
  exists in principle.

## THE GENUINE CONTRIBUTION (more valuable + more defensible than the original claim)
**A linear probe can hit 0.98 leave-one-domain-out on a silence-gated, cross-domain controlled construct and
be ORTHOGONAL to the model's actual concept axis.** High in-distribution probe accuracy — even with
cross-domain generalization AND a verified text-silence gate — does NOT establish you found the concept; the
construct's surface regularities yield a high-AUC direction unrelated to the real feature. The required
checks: OOD transfer to natural statements AND orthogonality to the in-OOD-internal direction. This is a
clean, general cautionary result for activation-probing / interpretability, demonstrated with receipts.

## SETTLING EXPERIMENT (pre-registered next, to run cold)
Wide naturalistic paired true/false corpus (hundreds of statements, many domains/forms/lengths, truth
decorrelated from template surface) + large balanced OOD test (≥100 misconceptions vs ≥100 surprising-true),
fit mass-mean+logistic, pre-registered 0.70 bar with bootstrap CI required to EXCLUDE 0.70, and track cosine
to the in-OOD-internal direction. (a) transfers + cosine→1 = "construct was too narrow, axis is recoverable";
(b) stays flat with large n = "no linearly-transferable truth axis from a held-out construct at 3B".
