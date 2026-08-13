# PRE-REGISTRATION — do dead instruments predict retracted results?

**Written:** 2026-08-13, evening, **before any grouping or rate was computed.** The
PROBE E re-runs are still executing; no dead-term rate for any retraction-linked module
has been looked at. This file is committed before the analysis script is written, which
is the practice `PRIOR_ART_2026_08_13.md` demanded after today's ledger was classified
retrospectively.

## AMENDMENT 1 — the first exposure build was not blind, and the harness was the leak

**Attempt 1 is void as a blind classification.** It is recorded here rather than
discarded, because a pre-registration that quietly replaces a failed attempt is worth
nothing.

The blinding rule was given to every agent and every agent obeyed it: none opened a
PROBE E artifact. **The leak was in the harness.** Each phase passed the previous
phase's full JSON downstream, and the agents' own contamination declarations quoted
outcome-side numbers verbatim inside free-text notes — so the adjudicator read
repo-level rates, one module-level deadness fact, and two conscience-chain terms with
their phi values *before* classifying anything.

It was caught only because the adjudicator **declared it unprompted**, quoted exactly
what it had seen, and assessed the direction of the resulting bias. Its assessment: the
leak runs **against H1 or is neutral** — it kept the whole conscience chain (known live,
which dilutes the effect) and declined to add `styxx.weather` to the exposure set
despite knowing it is dead, because the causal path did not justify it. That is the
conservative direction, and it is still not blind.

Two of four harvest arms also died on API 529s, so the exposure set was incomplete
independently of the contamination.

**Attempt 2** passes only a whitelist of structured fields between phases
(`claim`, `status`, `receipt_path`, `quote`, `date`), scrubs percentages, n-of-m ratios
and phi values from those, and re-runs the two failed arms. Free text never crosses a
phase boundary.

Amendments to a pre-registration are themselves suspect, so the boundary is stated: this
amendment changes **only how the exposure variable is transported between agents**. The
hypotheses, the exposure definition, the outcome, the test, `MIN_MODULES = 5`, and the
recorded expectation of H2 are all unchanged from the original commit `c2340ed`.

## The question

PROBE E measures whether a decision term could have gone the other way on the population
that drove it. Today that is hygiene: a property of code, interesting to its authors.

The question that would make it science is whether it **predicts epistemic failure**:

> Do the instruments that produced **retracted or corrected** published claims carry a
> higher dead-term rate than the instruments that produced claims which still stand?

If yes, falsifiability analysis becomes a *screening tool for results*, not a code smell.
If no, PROBE E measures something real about code and nothing about knowledge, and it
should be described that way permanently.

Neither literature answers this. Mutation testing relates test quality to defect
detection; nothing relates instrument falsifiability to **retraction of findings**.

## Hypotheses, stated before looking

- **H1 (primary).** Modules on the causal path of a retracted/corrected claim have a
  higher adjudicative dead-term rate than modules on the path of surviving claims.
- **H0.** No difference; any observed gap is within what the module-size distribution
  and the suite's coverage produce by chance.
- **H2 (the honest alternative, and I expect it to be live).** Retraction-linked modules
  are simply *less exercised* — their terms are `NEVER_REACHED` or `UNDERPOWERED` more
  often — and the dead rate among powered terms does not differ. This would be a
  coverage finding, not a falsifiability finding, and must not be reported as the latter.

## Design, fixed now

**Unit of analysis:** module (`styxx/<path>.py`). Terms are too fine — a retraction
implicates a pipeline, not a line — and papers are too coarse.

**Exposure (retraction-linked):** a module is retraction-linked if it appears on the
causal path of a claim that this repository has **struck, retracted, voided, withdrawn,
or corrected in a way that changed the claim's direction or licence**. The claim must
have a written receipt in `papers/` or `MEMORY.md`. Candidates known to exist before
grouping: the `sensitivity 1.0` strike, the withdrawn paired-McNemar p-value, G2 VOID
(`memory_integrity`), the anchored-validity H1 retraction, `consistency_robustness`
RETRACTED, today's `EXTERNAL_CENSUS` withdrawal, and today's own showcase-row
withdrawal.

**Comparison (surviving):** every other instrumented module.

**Outcome:** `dead_rate_adjudicative` from the corrected PROBE E run — adjudicative
terms only, value-position operands excluded, process-count pseudo-replicates excluded.

**Test:** Mann–Whitney U on module-level rates (module sizes are wildly unequal and the
rates are bounded, so a t-test is wrong). Report the effect size and the CI, not a bare
p. Two-sided.

**Covariates that must be reported alongside, because each could produce H1 spuriously:**
1. powered-term count per module (bigger modules estimate their rate better)
2. exercised fraction per module (H2's mechanism)
3. module age

## Stopping and floors, fixed now

- **MIN_MODULES = 5** retraction-linked modules with at least one powered adjudicative
  term. Below that the comparison is not run and the result is reported as
  `REFUSED__underpowered`. **This floor is not negotiable after seeing the data** — the
  day already contains one instance of a floor being applied to the wrong denominator to
  rescue a p-value, and it was withdrawn.
- No subgroup analysis, no module re-classification, and no swapping the outcome for
  `dead_rate_of_powered` once the primary is computed.
- One test. If it fails to separate, that is the result.

## What each outcome licenses

- **H1 supported:** falsifiability is associated with published error *in this
  repository, retrospectively, on n≈dozens of modules*. It licenses a prospective test,
  nothing more. It does not license "dead gates cause retractions."
- **H0:** PROBE E measures code, not knowledge. Say so in the method file permanently.
- **H2 supported instead:** the finding is about **coverage**, and the falsifiability
  framing must be dropped from it entirely.

## The obvious confound, named now rather than later

Retracted claims got *more attention* — they were investigated, which is how they came
to be retracted. Attention may drive both the retraction and the discovery of dead gates
nearby. This design cannot separate that; a prospective version would. **Any write-up
must carry this as a limit in the same paragraph as the result, not in a footnote.**

## Author's expectation, recorded so it cannot be revised later

I expect **H2**. The retraction-linked modules I can name are ones with weak or absent
test coverage, and the most likely honest outcome is that they are underexercised rather
than differently dead. I am registering H1 because it is the interesting claim and
because the instrument was built to test exactly this — not because I believe it.
