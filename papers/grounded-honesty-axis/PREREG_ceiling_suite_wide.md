# Pre-registration — the construct ceiling is suite-wide (not just the deception axis)

**Stated 2026-05-28, BEFORE the text-only suite is scored on the register-matched
pairs.** Follows the SURVIVED grounded-honesty finding
(FINDING_grounded_honesty_2026_05_28.md): on 48 register-matched factual
self-claims, the **deception** axis sits at AUC 0.498 (chance) while the grounded
axis (g = Stability × Concordance) reaches 0.966. Methodology:
recursive-discipline. One-shot confirmatory run; dataset is the SAME 48-pair set
(DATASET in run_grounded_honesty.py), answer key already hashed in the companion
run.

## The question

The grounded finding showed the *deception* axis is a register detector here. But
styxx ships **four** text-only cognometric axes (deception, overconfidence,
sycophancy, refusal). If the construct ceiling is real, it should be **suite-wide**:
none of the text-only axes should separate a TRUE confident self-claim from a FALSE
one, because the two arms are register-matched (identical confident template, one
substituted fact). Only an axis grounded in an EXTERNAL signal (sampling
divergence) should separate them.

This run scores all four text-only axes on the identical register-matched pairs.
The text-only axes are **deterministic text heuristics** (`styxx.attack.score_all`)
— no model sampling — so this is a fully OFFLINE confirmatory analysis on the
already-constructed dataset; the grounded number is the established 0.966 from the
companion confirmatory run on the same pairs.

## Pre-registered predictions

- **S1 — the whole text-only suite is at chance (descriptive).** Each text-only
  axis (deception, overconfidence, sycophancy) separates register-matched TRUE vs
  FALSE factual self-claims at AUC in **[0.35, 0.65]** (i.e. indistinguishable from
  the 0.50 chance line in either direction). Refusal is reported descriptively
  (these self-claims contain no refusal, so it is expected to be near-degenerate).

- **S_kill — grounded beats the BEST text-only axis decisively.** The grounded
  axis (0.966, companion run, identical dataset) exceeds the best of the three
  text-only axes by **>= 0.15 AUC**. If any single text-only axis already separates
  the arms at > 0.65, the ceiling is NOT clean for that axis and the gain is partly
  register leakage — I would report that axis as a confound, not a clean ceiling.

- **K — register-match holds per axis (confound guard).** For each text-only axis,
  a Welch t-test of the TRUE-arm vs FALSE-arm scores is non-significant
  (p >= 0.05): the arms are matched on every register dimension, so the grounded
  gain cannot be register leakage on any axis.

## What counts as what (no reframing)

- If S1 holds for all three axes, the contribution is: **the construct ceiling is a
  property of text-only cognometrics as a class, not a quirk of the deception
  axis** — quantified on a register-matched, hashed dataset. That strengthens the
  grounded-honesty headline without re-litigating it.
- If some axis separates the arms (AUC > 0.65, p < 0.05), I report that axis is
  NOT register-bound on this construction (a real, reportable negative against my
  own prediction), and scope the suite-wide claim accordingly.

## Honest scope

Offline analysis of deterministic text heuristics on the same 48 register-matched
factual-self-claim pairs as the companion grounded run; feasibility-grade. Says
nothing about value claims, predictions, or non-factual self-reports. The grounded
comparator inherits its own scope (single model, self-consistency not external
truth, injection-blind). I commit to reporting whichever way it lands.
