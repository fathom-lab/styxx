# Pre-registration — does grounding extend from RETRIEVED to COMPUTED facts?

**Stated 2026-05-28, BEFORE the computed-fact set is scored.** Follows the grounded
honesty arc: the axis grounds a factual self-claim against the model's OWN resampled
belief and breaks the text-only ceiling (AUC 0.97 vs 0.50) — but every dataset so
far is **retrieved** facts (capitals, element symbols, famous facts the model
MEMORIZED). Methodology: recursive-discipline. One-shot confirmatory run; answer key
SHA-256'd before scoring.

## The boundary being probed

The grounding mechanism assumes a **convergent attractor**: the model's stable
resampling mode is the truth. For retrieved facts that holds (the council work
showed truth-tracking, not fame-tracking). But a COMPUTED fact (e.g. 23 × 19 = 437)
is not stored — the model must DERIVE it each sample. Two open possibilities:

1. The attractor mechanism generalizes: for computations the model can reliably
   do, every sample lands on the same correct value (high Stability), so grounding
   works exactly as for retrieval.
2. It does NOT generalize: derivation is noisy, samples scatter even when the
   model "could" get it right, Stability is low across the board, and grounding
   degrades to chance — meaning the axis is a *retrieval* signal, not a *truth*
   signal.

Either is a real, reportable result that maps the boundary of the grounded axis.

## The computed-fact set (construction stated before scoring)

~36 arithmetic self-claims of graded difficulty, each a register-matched TRUE/FALSE
pair (same confident template, one substituted value), exact author-supplied ground
truth hashed before scoring:

- **easy** (2-digit ± / 1-digit ×): model expected reliable → high Stability.
- **medium** (2-digit × 2-digit): mixed.
- **hard** (3-digit × 2-digit, multi-step): model expected less reliable → lower
  Stability, more confident-wrong derivations.

FALSE sibling = a plausible wrong value (off-by-one-digit / common carry error).

## Pre-registered predictions

- **R1 — grounding extends to computation (descriptive).** Grounded AUC on the full
  computed set is **>= 0.75** (the mechanism is not retrieval-only).

- **R_kill — grounded beats text-only (decisive).** Grounded AUC exceeds the
  text-only deception axis by **>= 0.15** on the computed set (the gain is the
  grounding, not register).

- **R2 — the Stability gate replicates on a NEW domain (the prize).** Stratify the
  computed items by per-item resample Stability. HIGH-stability AUC **>= 0.85**;
  LOW-stability AUC collapses below it (and toward chance). If R2 holds, the
  report-or-abstain self-calibration found on retrieved facts is a GENERAL property
  of the grounded axis, not a retrieval artifact: it abstains on computations the
  model can't stably do.

- **K3 — register-matched (confound guard).** Text-only deception does NOT separate
  the arms (Welch p >= 0.05).

## What counts as what (no reframing)

- If R1 + R_kill + R2 hold, grounding is a TRUTH signal that spans retrieval AND
  derivation, self-calibrating in both — a materially broader claim than "works on
  memorized facts."
- If grounded AUC < 0.75 or Stability does NOT gate validity on computation, I
  report that the grounded axis is (at least partly) a RETRIEVAL signal that does
  not transfer to derivation — a boundary, reported against the optimistic R1.

## Honest scope

Single model (gpt-4o-mini), OpenAI-only, one run, feasibility-grade, n≈36. Exact
arithmetic ground truth hashed pre-scoring. Inherits all grounded-axis scope:
self-consistency not external truth, injection-blind, one axis-family. I commit to
reporting whichever way it lands.
