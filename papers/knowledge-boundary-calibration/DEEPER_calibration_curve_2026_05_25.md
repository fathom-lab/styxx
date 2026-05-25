# Deeper direction · The Epistemic Psychometric Function — a calibration CURVE, not a score

**2026-05-25, queued behind the KBC probe.** KBC (binary real-vs-fake) measures a
model's knowledge boundary with **two points**. The deeper instrument measures the whole
**curve** — borrowing the method psychophysics uses for perception.

## The analogy (and why it's the right one)

Psychophysics never asks "can you perceive it?" It sweeps **stimulus intensity** and fits
a **psychometric function** to the response rate, yielding two parameters:
- **threshold** — the intensity where the response crosses 50%
- **slope** — how sharply it transitions (discrimination acuity)

Map this onto knowledge. The "stimulus" is **epistemic distance from solid fact**; the
"response" is **abstention**. Sweep a reality gradient, plot abstention-rate vs level,
fit the curve. You get, per model:
- **boundary location** — where it starts saying "I don't know" (its knowledge edge)
- **boundary sharpness** — how cleanly it separates known from unknown

This turns "is the model calibrated?" from a yes/no into a **2-parameter profile of where
its knowledge ends and confabulation begins.** Nobody measures models this way.

## The reality gradient (the stimulus axis)

| level | example | well-calibrated response |
|---|---|---|
| L0 real-common | capital of France | answer |
| L1 real-obscure | deepest point of the Mediterranean | answer (if known) |
| L2 real-rare | the 1908 Olympic marathon's disqualified leader | answer or honest "unsure" |
| L3 plausible-fake | capital of the Republic of Vorland | **abstain** |
| L4 absurd-fake | capital of the Kingdom of Glorptaxia-9 | **abstain** (easy) |

The discriminating signal lives at **L2→L3** — the real/fake seam. A model that abstains
on L4 (absurd) but confabulates L3 (plausible) has its boundary in the *wrong place*:
it's pattern-matching surface absurdity, not tracking real knowledge.

## What the curve diagnoses (failure modes a score can't)

- **boundary too far right** (answers plausible-fakes) → overconfident confabulator
- **boundary too far left** (abstains on obscure-reals) → over-cautious / useless-humble
- **shallow slope** → fuzzy boundary, poor knowledge discrimination
- **non-monotonic** → abstains on absurd-fakes but not plausible-fakes → surface-cue
  matching, not epistemic awareness (the most revealing failure)

## The second curve (fuse the Tier-3 detector)

At each level, when the model *answers*, measure cross-sample **inconsistency** (the
validated confabulation signal, cosine@0.90 entropy). Plot inconsistency vs level. It
should be ~0 on L0–L1 (knows it), rising through L2–L4 (inventing). Two psychometric
functions over the same axis:
- **abstention(level)** — what the model *admits* it doesn't know
- **inconsistency(level)** — what the model *betrays* it doesn't know (even while
  answering confidently)

**The gap between them is self-knowledge failure made visible:** the region where the
model still answers (low abstention) but is already inventing (high inconsistency) is
exactly **confident confabulation** — the Tier-3 frontier — now localized on a curve.

## Why this is the culmination, not a tangent

The session's arc: Tier-1 form → Tier-2 grounding → Tier-3 confident error (partially
crossed: confabulation is inconsistent) → and now a **psychometric instrument for the
knowledge boundary itself**. Both behavioral signals — *abstention* (against controlled
ground truth) and *inconsistency* (across samples) — sidestep the model's unreliable
self-report. The curve unifies them: one axis (epistemic distance), two response
functions, a complete map of where a model knows, where it admits ignorance, and where
it confidently makes things up.

## Discipline note

Queued, not pre-registered yet — on purpose. **Run KBC first.** If KBC fails K1 (no model
abstains on fakes at all), the curve has no threshold to find and this is moot; if it
passes, the gradient refines the boundary into a curve. Pre-register the curve's bars
(monotonic abstention rise; threshold between L2 and L3 for the best model; the
abstention/inconsistency gap is non-empty) only after KBC confirms the binary signal.
No grand theory ahead of data — the lesson of this session, four self-corrections deep.
