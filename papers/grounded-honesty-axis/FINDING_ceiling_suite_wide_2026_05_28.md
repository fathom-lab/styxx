# FINDING — The construct ceiling is suite-wide, not a quirk of the deception axis

**2026-05-28. Pre-registered (PREREG_ceiling_suite_wide.md), one confirmatory run.
Fully OFFLINE and reproducible: deterministic text heuristics, no model sampling.**
Receipt: `ceiling_suite_result.json`. Dataset: the same 48 register-matched
factual-self-claim pairs as the grounded-honesty run (`DATASET` in
run_grounded_honesty.py).

## Result (SURVIVED)

Every text-only cognometric axis styxx ships sits at chance separating a TRUE
confident self-claim from a register-matched FALSE one:

| text-only axis | AUC (TRUE vs FALSE) | Welch t | p |
| --- | --- | --- | --- |
| deception | 0.498 | −0.74 | 0.46 |
| overconfidence | 0.449 | −0.58 | 0.56 |
| sycophancy | 0.505 | +0.04 | 0.97 |
| refusal | 0.537 | +0.50 | 0.62 |
| **grounded** `Stability×Concordance` (companion run, same data) | **0.966** | — | — |

- **S1 held:** all three substantive text-only axes fall in [0.35, 0.65] — the max
  deviation from the 0.50 chance line across the suite is **0.051**. The whole
  text-only instrument suite is blind to the substituted fact.
- **S_kill held (decisive):** grounded beats the best text-only axis by **+0.461**
  (0.966 vs 0.505) — far past the pre-registered +0.15 bar.
- **K held (no leakage on any axis):** every text-only axis is register-matched
  across the arms (all p ≥ 0.46). The two arms are identical confident self-reports
  with one substituted fact, so the grounded gain cannot be register leakage on
  *any* dimension the text suite measures.

## What this means

The grounded-honesty finding showed the *deception* axis is a register detector
here. This run shows that is not a quirk of one axis: **register-bound is a property
of text-only cognometrics as a class.** Deception, overconfidence, and sycophancy
all read how the sentence *sounds* — and the two arms sound identical — so all
three are pinned at chance by construction. The only thing that moves is the axis
grounded in an EXTERNAL signal (the model's own resampled belief distribution),
which jumps to 0.966.

This is the construct ceiling quantified across the suite, on a hashed,
register-matched dataset, with a fully offline and reproducible analysis (no API,
no randomness — re-run `run_ceiling_suite.py` and the numbers are identical).

## Honest bounds (stated, not hidden)

- **One axis-family.** The pairs are *factual* self-claims. This says nothing about
  value claims, predictions, or non-factual self-reports — only that on factual
  self-claims, the text-only suite is uniformly register-bound.
- **The grounded comparator is the companion run** on the identical dataset, not
  recomputed here (it needs live sampling; the text axes do not). The construction,
  dataset, and hash are the same; the comparison is apples-to-apples by design.
- The grounded axis inherits all its own scope: single model, self-consistency not
  external truth, injection-blind, feasibility-grade.

## Net

styxx's text-only cognometrics are, as a class, register detectors with a validated
construct ceiling — now shown across deception, overconfidence, AND sycophancy on a
register-matched factual-self-claim set, not just the worst-case deception axis.
The grounded honesty axis (shipped as `styxx.grounded_honesty` in 7.7.13) is the
one signal that escapes the ceiling, because it grounds in an external,
sampling-based belief distribution rather than the register of the text. That is
the precise, falsifiable boundary between what text-only scoring can and cannot do.
