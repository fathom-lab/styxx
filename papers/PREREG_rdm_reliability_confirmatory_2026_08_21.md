# PREREGISTRATION — confirmatory test on FRESH items

**Frozen 2026-08-21, before the confirmatory run. Exploratory origin disclosed
below; nothing here was chosen after seeing confirmatory data, because none
exists yet.**

## why this is legitimate and not attempt 3

Attempt 2's prereg said: *"no attempt 3 without a new mechanism."* That rule
exists to stop a program running variants until one clears. Two things changed,
and neither is a variant:

**1. G4 was measuring the wrong property, and that is an instrument error
independent of any outcome.** G4 gated on IQR — *absolute spread* — while the
analysis it guards (AUC, Spearman) is **rank-based**. Absolute spread is
irrelevant to a rank statistic. The run it invalidated had **497 distinct values
in 500 items**: near-complete rank ordering, not a constant. The correct sanity
check for a rank-based analysis is **ties / effective distinct values**, and that
correction stands whatever the data say.

**2. A direction is now predicted, from exploratory analysis that is disclosed
rather than laundered.** Post-hoc inspection of the INVALID attempt-2 run showed
rho(reliability, correct) = **-0.1621** (p = 0.0003), surviving partial
correlation controlling prompt length and log popularity at **-0.1321**
(p = 0.003), and near-orthogonal to popularity (rho = 0.027). Higher reliability,
*less* likely correct — **inverted from the original H1**.

That analysis cannot be confirmatory: it came from a run its own gates rejected,
was looked at after the rejection, and found a direction nobody predicted. This
preregistration is the clean test of it, and the exploratory→confirmatory split
is honoured the only way that means anything: **on data the exploratory analysis
never touched.**

## the test

**H1'** (ONE-SIDED, pre-declared): rho(reliability, correct) < 0 on fresh items,
and the partial correlation controlling prompt length and log popularity is also
< 0 at p < 0.05 one-sided.

**H0'**: it is not, or the sign flips.

**Fresh items.** N = 500 PopQA items drawn with seed 20260822, **excluding every
item used in attempts 1 and 2**. Disjointness is asserted in code, not assumed.

**Everything else is attempt 2, unchanged**: Qwen2.5-1.5B-Instruct, layer 21/28,
final-prompt-token representation, 20 feature splits, exact-match grading, same
baseline confidence triple. One thing varies — the items.

## gates

- **G1 PRIMARY** — delta-AUC(baseline + reliability) over AUC(baseline alone),
  5-fold out-of-fold, 95% bootstrap CI, 2000 resamples. CI includes 0 →
  NOT SUPPORTED.
- **G2 CONFOUND** — partial Spearman controlling length and log popularity must
  be **negative** and p < 0.05 one-sided.
- **G3 VALIDITY** — accuracy outside [0.10, 0.90] → INVALID.
- **G4 CORRECTED** — **≥ 90% distinct values** among the reliability scores.
  Replaces the IQR floor, which was the wrong criterion for a rank statistic.
  Zero or near-zero *rank* variation is what makes a measure unusable here, not
  a small absolute range.

## what a pass would and would not license

**Would:** that on this model, this task and this layer, representational
reliability carries error-relevant information beyond token confidence — in the
direction opposite to the original hypothesis, replicated on unseen items.

**Would not:** any claim about other models, scales, tasks or layers; any causal
reading; any deployment. And it would still owe an explanation for **why**
stability predicts being *wrong*, which this design does not test.

**Power, stated in advance:** attempt 2 ran ~11.6% accuracy, about 58 positives.
If the fresh sample lands near that, the delta-AUC CI will be wide. **A wide CI
covering zero is a power statement, not evidence of absence**, and will be
reported as one.
