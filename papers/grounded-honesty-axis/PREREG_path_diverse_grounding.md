# Pre-registration — does METHOD-DIVERSE grounding catch confident confabulation? (within a single model)

**Stated 2026-05-28, BEFORE the path-diverse set is scored.** Direct follow-on to
the competence-cliff run (FINDING_competence_cliff_2026_05_28.md), which proved that
on hard derivation gpt-4o-mini is frequently **stably wrong**: plain resampling of
"What is 517 × 283?" returned 146051 on all 10 samples (Stability 1.0, fully wrong;
truth 146311). Plain-resample Stability therefore certifies a sharp BELIEF, not a
correct value (D1 gate failed, high-stability AUC 0.778), and grounded AUC decayed
to 0.667 / chance on the hardest tier. Methodology: recursive discipline. One-shot
confirmatory run; answer key SHA-256'd before scoring; exact ground truth computed
in-code; core signal is exact integer match (no judge).

## The hypothesis being tested

Plain resampling re-runs the *same* one-shot computation path, so a systematic
miscalculation reproduces as a sharp wrong attractor. **Hypothesis: confident
confabulation is method-specific.** If the model derives the same product through
**independent reasoning paths** — explicit step-by-step CoT, hundreds/tens/ones
decomposition, long multiplication with partial products, estimate-then-exact,
digit-by-digit-with-carry — then either:

1. The paths **converge on truth** (the model can compute it correctly given a
   reasoning scaffold; the one-shot answer was a shortcut error). Then **method-
   diverse self-consistency is a confabulation-catching validity gate that recovers
   the truth signal WITHIN a single model** — a strictly stronger grounding backend
   for styxx, and a within-model substitute for the cross-vendor signal the
   same-vendor council (C2) could not provide.
2. The paths **agree on the same wrong value** (the error is robust across methods).
   Then confident confabulation is a deep systematic error and **cross-vendor
   grounding is genuinely required** — also a decisive, reportable boundary.

Either outcome maps the boundary. The decisive kill-gate is P2.

## Method (stated before scoring)

Reuses the competence-cliff dataset (`SPECS` in run_competence_cliff.py): the same
register-matched hard arithmetic pairs (3×2 control, 3×3, 4×3, multi-step a×b±c),
correct values computed in-code and hashed. Two grounding backends are scored on
the IDENTICAL items, N=10 samples each, temp=1.0, gpt-4o-mini:

- **plain** — bare "What is X? Give only the number." (the cliff backend).
- **path-diverse** — N derivations, each prompted with a DIFFERENT method
  instruction (5 methods rotated), each ending "ANSWER: <number>"; the final
  integer is parsed (ANSWER: line, fallback to last integer).

Per item and backend: parsed-integer mode, Stability = 1−(n_distinct−1)/(N−1),
concordance-with-truth = (#samples == computed truth)/N, grounded g = Stability ×
concordance-with-the-claim, modal-correctness = (mode == truth).

## Pre-registered predictions

- **P1 — method-diversity self-corrects (descriptive).** On the items where the
  PLAIN modal answer was WRONG, the path-diverse backend's modal-correctness rate is
  **≥ +0.40 absolute higher** than plain's (which is 0 on those items by
  definition). i.e. given a reasoning path the model recovers the truth on a large
  fraction of its one-shot confabulations.

- **P2 — path-diverse grounding restores the truth signal (DECISIVE kill-gate).**
  Path-diverse grounded AUC (TRUE vs FALSE) is **≥ 0.85 overall** (vs plain 0.667),
  AND the high-Stability stratum recovers to **≥ 0.85** (vs plain 0.778). If P2
  holds, method-diverse self-consistency is a within-model validity gate that catches
  the confident confabulation plain Stability missed.

- **P3 — concordance-with-truth jumps (mechanism).** Mean per-item
  concordance-with-truth under path-diverse exceeds plain by **≥ +0.30** on the hard
  (non-control) tiers — the mechanism behind P1/P2 is more samples landing on the
  true value, not just lower Stability.

- **K — control tier unharmed.** On the ctrl_3x2 tier (where plain already grounded
  at AUC 1.0), path-diverse grounded AUC stays **≥ 0.90** — method diversity does not
  degrade items the model already computes reliably.

## What counts as what (no reframing)

- **P2 holds** → method-diverse grounding is a stronger backend that certifies truth
  (not just belief) on derivation within a single model: a materially stronger claim
  and a concrete tool upgrade.
- **P2 fails** (path-diverse AUC < 0.85 or high-stratum does not recover) → the
  confabulation is robust across reasoning methods; within-model grounding cannot
  escape it and cross-VENDOR grounding is required. Reported against the optimistic
  P1/P2.

## Honest scope

Single model (gpt-4o-mini), OpenAI-only, one run, feasibility-grade, n≈36 × 2
backends. Exact arithmetic ground truth computed in-code and hashed pre-scoring;
core signal exact integer match (no judge). CoT answers parsed from an explicit
ANSWER: line. Inherits all grounded-axis scope: self-consistency not external truth,
injection-blind, one axis-family. I commit to reporting whichever way it lands.
