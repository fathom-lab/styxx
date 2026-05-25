# Finding · The Epistemic Psychometric Function — PASS, and it's a clean instrument

**2026-05-25.** Prereg `preregistration_curve_2026_05_25.md`. Sweeps a reality gradient
under a fixed neutral prompt and fits two response curves per model. **Verdict: PASS**
(B1 ∧ B2; B3 confirmed). The clearest single artifact of the whole arc:
`epistemic_curve.png`.

## The two curves (N=5, neutral prompt "Answer the question in one short sentence.")

| level | gpt-4o abstain / inconsist | gpt-4o-mini abstain / inconsist |
|---|---|---|
| L0 real-common | 0.00 / 0.00 | 0.00 / 0.00 |
| L1 real-obscure | 0.00 / 0.00 | 0.00 / 0.17 |
| L2 plausible-fake | 0.60 / 0.76 | 0.43 / **1.24** |
| L3 absurd-fake | 1.00 / — | 0.77 / 0.95 |

- **abstention(level)** = what the model *admits* it doesn't know.
- **inconsistency(level)** = cosine@0.90 cross-sample entropy on *answered* items = what
  it *betrays* while still answering.

## What the curve says (the whole thesis in one picture)

- **gpt-4o has a located, well-placed knowledge boundary.** Abstention is flat-zero
  across both real tiers, then rises sharply (0.60 → 1.00) across the fake tiers;
  threshold at plausible-fake. It **admits** ignorance — and because it rarely answers a
  fake, its inconsistency stays low. A model that knows the edge of its knowledge.
- **gpt-4o-mini's boundary is "too far right."** It *answers* 57% of plausible-fakes
  (abstention only 0.43) and only mostly-rejects the absurd ones. It won't **admit**
  ignorance — but it **betrays** it: inconsistency spikes to 1.24 at plausible-fake,
  where its two curves *cross* (amber above cyan). **That crossing is confident
  confabulation, localized on the curve** (B3: gpt-4o-mini @ plausible-fake, abstention
  < 0.5 ∧ inconsistency ≥ 1.0).
- **The betrayal precedes the admission.** gpt-4o-mini already shows nonzero
  inconsistency at real-*obscure* (0.17) — the inconsistency signal starts revealing
  shaky knowledge a full tier before abstention would.

## Why this matters

Two behavioral channels, neither trusting the model's self-report, jointly map a model's
epistemic state across the knowledge frontier:
- where it **knows** (both curves flat-zero),
- where it **admits ignorance** (abstention rises),
- where it **confidently invents** (abstention low, inconsistency high — the danger
  zone).

The **gap between the curves** is the operational definition of *confident
confabulation*, and it's exactly where single-response confidence (closed) and the
model's own abstention both fail — but cross-sample inconsistency succeeds. This is a
psychometric instrument for the knowledge boundary, and it ranks models by epistemic
shape, not just score: gpt-4o (admits) vs gpt-4o-mini (betrays).

## Honest scope

Feasibility-grade: 2 models, N=5, 6 items/level, single run, OpenAI-only, neutral prompt
(abstention is prompt-elastic — KBC finding — so the curve is the *default*-disposition
boundary, not the only one). Coarse thresholds. The abstention regex may undercount soft
hedges. PASS is on the pre-registered descriptive bars; a full run (more items/level,
≥3 models, multiple prompts to map the prompt-elasticity surface) is the next step. No
overclaim: this shows the *method works and is interpretable*, not that these exact
thresholds are a model's final number.

## Place in the arc

Form → grounding → confident error (partially crossed: confabulation is inconsistent) →
KBC (grounded humility, prompt-elastic) → **the psychometric curve** that unifies
abstention + inconsistency into one map of where a model's knowledge ends. The session's
through-line, made visual.
