# Refusal-Universal-Transport — Adversarial Stress Test: Boundary Found (2026-05-17)

**Status:** the claim HOLDS in its solid regime and CRACKS at a newly
found, important boundary. This is the intended outcome of a real
falsification test.
**Script:** `scripts/dogfood/refusal_transport_stress.py`
**Raw:** `scripts/dogfood/out_refusal_transport_stress.json`

## What was attacked

The one live-validated claim (refusal-universal-transport) rested on
n=30, 2 OpenAI mini-models, one corpus. Hardened every thin axis:
75 prompts (30 eval + 45 aggressive-borderline), 4 OpenAI models incl.
full gpt-4o / gpt-4.1, cross-family (all-mpnet) + same-family
(te3-small), and **two independent label-free corpora** (topical vs
period-narrative register). Behavioral ground truth = lexical refusal
on each model's live response (refuse rates 20–27%, clean). Method
mirrors the validated closed-model test; dogfoods shipped
`styxx.transport`.

## Result

| regime | corpus_1 | corpus_2 | verdict |
|---|---|---|---|
| same-family (te3-small), 4 models | 0.814–0.868 | 0.822–0.878 | robust + **corpus-stable** |
| cross-family (mpnet), 4 models | 0.800–0.858 | **0.610–0.694** | real but **corpus-dependent** |

- Transported AUC overall: mean 0.792, min 0.610, max 0.878.
- Corpus robustness |AUC_c1 − AUC_c2|: mean 0.090, **max 0.215**.
- Pre-registered verdict: min < 0.70 and max spread > 0.08 →
  **CRACKS — boundary found.**

## The boundary (precise, honest)

It is specifically a **cross-family × corpus interaction**, not a bad
corpus globally: corpus_2 is *fine* same-family (0.82–0.88, ≥ corpus_1
there) and *bad* cross-family (0.61–0.69). So:

> Refusal-universal-transport is robust and corpus-stable
> **same-family**. **Cross-family it is real but requires the label-free
> alignment corpus to be semantically matched to the audit domain.** A
> register-mismatched corpus (corpus_2: "In 1894 a clerk noted that…")
> costs up to **0.215 AUC** cross-family. Hypothesis (testable): the
> very-different-objective cross-family map (3072↔768) underfits the
> harmful/dual-use refusal subspace unless the corpus spans it;
> same-family transport is forgiving enough that register doesn't
> matter.

## Why this is the deliverable

The thin original validation hid this entirely. Attacking the claim
before publishing converted "refusal universal transport works" into a
**precisely bounded claim plus a methodological requirement**
(corpus–domain semantic match for cross-family cognometric transport).
That requirement is itself a contribution. A claim that survived its
authors trying to break it — with its real edge stated — is what makes
the eventual paper credible instead of retractable.

## Standing position (updated, honest)

- Same-family universal cognometric transport (any reasonable corpus):
  **solid**, 4 models, harder prompts. Paper-worthy.
- Cross-family: **solid IFF the corpus is domain-matched**; otherwise
  degrades sharply. Must be stated as a requirement, not hidden.
- NOT cross-vendor: OpenAI-only (no Anthropic key). Claude/Gemini
  remain the flagged untested gap.
- No Zenodo/OSF. But this is the closest to paper-shaped the program
  has been, because the boundary is now known, not lurking.

## Next (honest, concrete)

1. Test the hypothesis: vary corpus semantic coverage systematically;
   show cross-family AUC tracks corpus↔eval subspace overlap. If it
   does, that *is* a clean methodological paper section.
2. Cross-vendor (needs an Anthropic/Gemini key) — the real remaining
   external-validity gap.
