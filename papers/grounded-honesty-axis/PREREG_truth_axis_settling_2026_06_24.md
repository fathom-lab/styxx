# PREREG — the settling experiment: does a WIDE construct recover the transferable truth axis?

**Frozen 2026-06-24 BEFORE extracting wide-construct/large-OOD activations. Offline, local-GPU, NO frontier
key.** Settles the open question from `FINDING_truth_direction_deflation` (triangulation): the 6-template
construct's 0.98 direction is ORTHOGONAL to the model's real truth axis (cosine ≈0) and fails OOD (0.53–0.67),
but at n=25 the OOD test is underpowered. Two non-exclusive causes: construct-too-narrow vs no-transferable-
axis-at-3B. This kills BOTH the power problem AND the surface-confound simultaneously.

## Design
- **Wide training construct:** ≥15 domains, cyclic-derangement minimal pairs (false BY CONSTRUCTION; the
  TRUE statements are basic verified facts), diverse templates/forms/lengths. Silence RE-VERIFIED:
  adversary-fair BoW leave-one-domain-out ≤ 0.55.
- **Large OOD natural test:** ≥35 misconceptions (familiar-but-FALSE) + ≥35 surprising-truths
  (unfamiliar-but-TRUE), each INDEPENDENTLY cross-verified (multi-agent fact-check; drop any disputed/ambiguous).
  Balanced, natural sentence forms (NOT the template).
- Reader models: Qwen2.5-3B + Llama-3.2-3B, last-token activations at the truth layer (qwen 19 / llama 14).
- Fit mass-mean AND L2-logistic on the wide construct; test the large OOD set.

## Decision thresholds (FROZEN), per model, BOTH must agree for a robust verdict
Let `ood` = best of {mass-mean, logistic} OOD misconception-vs-surprising AUC; `ci_lo` = its 2000x bootstrap
95% CI lower bound; `cos` = cosine(wide-construct direction, in-OOD-internal direction).
| Verdict | Condition |
|---|---|
| **AXIS RECOVERABLE** (failure was construct-narrowness; the fix is wide data) | `ood ≥ 0.75` AND `ci_lo > 0.70` AND `cos ≥ 0.50` |
| **ROBUST FAILURE** (no transferable truth axis from a held-out construct at 3B) | `ood < 0.65` AND CI upper bound < 0.75, with the in-OOD-internal ceiling still high (axis exists, just not transferable) |
| **PARTIAL** | otherwise |

## Required controls (frozen)
- **in-OOD-internal LOO ceiling** (fit+test within the large OOD set): confirms a truth axis EXISTS on
  naturals (expected ≥0.80) — the result is about TRANSFER, not absence.
- **label-shuffle** on the wide-construct fit → OOD ≈ chance.
- **BoW silence** on BOTH the wide construct (LODO ≤0.55) AND a BoW-OOD floor (BoW trained on construct → OOD
  must be ≈chance; text carries no transferable truth).
- **cosine to real axis** reported regardless (the headline diagnostic).

## What each outcome means (pre-committed interpretation)
- AXIS RECOVERABLE → "the orthogonality failure is curable: narrow constructs find surface directions, but a
  wide diverse construct recovers the model's real, transferable truth axis. Here is the validation recipe."
  (Constructive, actionable, the stronger contribution.)
- ROBUST FAILURE → "even a wide, well-powered held-out construct does not yield a linearly-transferable truth
  direction in 3B models, though the axis demonstrably exists in-distribution. Probe transfer is fundamentally
  fragile at this scale." (A hard, important bound.)

## Honest scope
2 local 3B models; OOD n≥70 (kills the n=25 caveat); single seed; linear probes. Pre-registered; verdict
mechanical. The wide-construct facts are false-by-construction; the OOD naturals are the fact-critical part and
are multi-agent cross-verified before use.
