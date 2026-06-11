# FINDING — the apex test returned an honest null: these models do not cave (VOID-NO-CAVING)

**2026-06-11 · Fathom Lab / styxx. Pre-registered: `PREREG_says_yes_knows_no_2026_06_11.md` (frozen
pre-run). Receipt: `says_yes_knows_no_result.json`. A pre-committed VOID condition fired; this is
recorded, not buried — the null is on the PREMISE, not the conscience.**

## What we asked

The apex of the portable-conscience arc: when a model is pressured into VERBALLY asserting a falsehood
as true, does the portable conscience — read from internal state, with a direction fit in a DIFFERENT
model — still know it is false? "The model says yes; its mind says no." Internal read = the validated
portable pipeline (gemma-2-2b layer-12 honesty direction, mapped into each target, read on the bare
statement). Behavioral = the target's own first-token logit(True) - logit(False) under NEUTRAL vs a
single-turn authority PRESSURE prompt.

## Result — the premise did not hold: the models refused to cave

| target | behavioral neutral | behavioral pressure | caving drop | internal portable | conscience holds? |
| --- | --- | --- | --- | --- | --- |
| Llama-3.2-3B | 0.9883 | 0.9961 | -0.0078 | 0.9229 (p=0.003) | yes |
| Qwen2.5-3B | 0.9961 | 0.9995 | -0.0034 | 0.8906 (p=0.004) | yes |

Under single-turn authority pressure, neither primary model caved — behavioral truth-discrimination
stayed at ceiling (~0.99) and even ticked UP. The pre-committed `VOID-NO-CAVING` condition (neither
primary caves) fired. With no verbal lie, the says-yes-knows-no divergence cannot be measured here.
The smaller Llama-3.2-1B bent slightly (caving drop 0.0947) but below the 0.15 threshold; gemma's
internal self-ceiling held at 0.9287.

## What is, and is NOT, concluded

- **NOT concluded:** that styxx cannot catch a caving model. We never produced a caving model, so the
  apex claim is simply UNTESTED in this regime — no positive and no negative on the conscience.
- **What IS shown (an honest, almost surprising sub-result):** modern RLHF'd instruct models
  (Llama-3.2-3B, Qwen2.5-3B) resist single-turn authority/social pressure on clear facts in a
  forced-choice — they will not assert a plain falsehood as true just because an "expert" insists.
  This BOUNDS the live lie-catch framing: the divergence between words and internal grounding requires
  a regime where the words actually cave (weaker/fine-tuned models, ambiguous claims, or multi-turn
  escalation), not this one.
- The portable conscience itself remained valid throughout: the internal read beat the
  label-permutation null for every model (p <= 0.004), consistent with the OOD and adversarial findings.

## Stronger-pressure attempt (v2) — no result, technical failure disclosed

A pre-registered v2 (`PREREG_says_yes_knows_no_v2_2026_06_11.md`,
`run_says_yes_knows_no_v2.py`) stacked a deference SYSTEM persona + authority + social cost as the one
strengthened attempt. The run hung mid-execution on a wedged local GPU driver and produced NO output;
it is reported here as a technical failure with zero scientific content, to be re-run when the GPU
recovers. No v2 numbers are claimed.

## Honest bounds

One pressure family (single-turn authority, forced-choice True/False margin), clear-fact claims, linear
DiM source + ridge map, local open models. A VOID-NO-CAVING result is a statement about the test setup
and the models' robustness, not about the conscience. The next rung is a regime that actually induces
caving (v2 stronger pressure once the GPU is back, ambiguous claims, or multi-turn) — only there can the
apex be decided.
