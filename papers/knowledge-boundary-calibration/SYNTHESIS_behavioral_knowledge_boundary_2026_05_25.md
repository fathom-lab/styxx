# The Behavioral Knowledge Boundary — what this session built

**2026-05-25.** Five pre-registered probes, one through-line. Extends
`SYNTHESIS_cognometric_boundary_2026_05_25.md` at the Tier-3 frontier.

## The question

"Does the model know what it doesn't know?" — equivalently, "is this confident answer
real or confabulated?" Both routes that *ask the model* are closed: token-logprob dies on
hallucination (grounded-arc, ρ≈0); text-only calibration recal failed held-out. The
model's self-report is unreliable **exactly when it's wrong.**

## The move: stop asking, start watching divergence

A fact is a **shared attractor**; a fabrication has **no attractor**. So truth is
*convergent* and fabrication is *divergent* — and divergence is measurable behaviorally,
three ways, none trusting self-report:

| signal | axis of divergence | finding | status |
|---|---|---|---|
| **intra-model inconsistency** | across N samples of one model | a model invents a *different* fact each sample → semantic entropy detects confabulation, AUC 0.88–0.95 cross-model | PASS (feasibility, 3 models) |
| **abstention** | answer vs "I don't know," against *controlled* fake entities | abstain-on-fake / answer-on-real = knows its boundary; but **prompt-elastic** (one clause: 0%→97% abstention) | construct-valid; prompt-bound |
| **inter-model agreement** | across independent models | models converge on real, scatter on fake → reference-free fabrication detection, AUC 1.0 | PASS (consensus, not yet truth) |

## The unifying picture (the psychometric curve makes it visual)

Plot abstention and inconsistency across a reality gradient and you get a model's
epistemic profile:
- where it **knows** — both flat-zero (convergent, no abstention);
- where it **admits** — abstention rises;
- where it **confidently invents** — abstention low but inconsistency high. **That gap is
  confident confabulation**, the Tier-3 case single-response confidence misses.

gpt-4o *admits* (abstains before inventing); gpt-4o-mini *betrays* (answers fakes, but its
answers scatter). Same instrument, two epistemic shapes.

## The recurring adversary: FORM impersonating MEANING

Every error this session lived in the *measurement*, not the phenomenon:
- cosine@0.70 clustering called six different lies "the same answer" (form ≈ surface) →
  manufactured a null; entailment/meaning clustering fixed it.
- a validity gate keyed on cosine (a tested method) was circular.
- an abstention-inviting prompt ceilinged calibration (form = prompt wording).

The cognometric boundary — *form is cheap, truth needs grounding* — reappears **inside
the instruments**, one level down. The signal is always a meaning/divergence quantity;
the trap is always a surface proxy for it.

## Honest scope (the whole arc is feasibility-grade)

OpenAI-only (cross-vendor key-blocked — and same-vendor consensus is the *weak* form of
the Council); small n; single runs; pre-registered feasibility bars, not hashed
production validations. Four self-corrections happened along the way (tweet → "use NLI" →
NLI-also-FPs → threshold sensitivity), each by honoring a bar over momentum.

## What's earned vs open

- **Earned:** confident confabulation is *behaviorally detectable* (it's inconsistent),
  cross-model; the clustering step is characterized; epistemic humility is prompt-elastic
  (and "say if unsure" is a near-free guardrail); reference-free *fabrication* detection
  works via inter-model agreement.
- **Open:** reference-free **truth** (vs *consensus/fame*) — needs an ultra-rare-real
  tier; **cross-vendor** councils; **full hashed runs**; a shipped `semantic_entropy` /
  `council` primitive. These are the pre-registerable next bets.

**The line that holds:** a model's knowledge boundary is dark to its own self-report but
*bright to divergence* — across its samples, and across its peers. That is the session's
contribution, and its honest edge.
