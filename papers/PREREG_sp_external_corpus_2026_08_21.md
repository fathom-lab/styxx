# PREREG — SP-EXT: a silent-pass corpus mined from code we did not write

**Frozen before any candidate commit was inspected. The 14 repositories were
cloned first; not one commit had been read when this was written.**

---

## the gap this closes

Every SILENT-PASS result this project has published is measured on **its own
repository**. The corpus in `benchmarks/silent_pass` is 20 defects, all of them
ours, all found by us. That makes this a lab that audits itself well. It does not
establish that the defect class exists in the field, and the one attempt to show
that it does — `RESULT_flattering_external_2026_08_21.md` — returned 0 genuine
findings from a screen with 10% recall, which established nothing in either
direction.

**SP-EXT** is the corpus that would settle it: silent-pass defects in other
people's code, each anchored to a **real fix commit**, so ground truth is not our
judgment. Somebody else read that code and decided the old behaviour was wrong.

No benchmark of this kind exists. That is the reason to build it and also the
reason to be careful: with no prior art there is nothing to check us against, so
the inclusion rule and the rejection rate both have to be published.

## the corpus, named before any commit was read

14 repositories, 61,702 commits, cloned with full history. Evaluation,
red-teaming, data validation and ML monitoring — the organ where measurements are
the product:

`lm-evaluation-harness` · `ragas` · `deepeval` · `garak` · `trulens` · `giskard` ·
`inspect_ai` · `great_expectations` · `evidently` · `pandera` · `whylogs` ·
`alibi-detect` · `cleanlab` · `deepchecks`

Pinned by HEAD SHA in the RESULT. **No repository is added or removed after any
candidate is seen.**

## INCLUSION RULE, frozen

A commit qualifies as an **SP-EXT case** only if its diff shows that, *before* the
fix, all three held:

1. **A path existed where the measurement did not happen** — no data, an empty
   input, an exception, an unsupported type, an unavailable optional dependency,
   a skipped branch, a platform without the capability.
2. **On that path the code produced a value or verdict INDISTINGUISHABLE from a
   real, healthy measurement.** Not `None`, not `NaN`, not a raise, not a distinct
   sentinel, not a separate `skipped` state — a value a consumer cannot tell apart
   from one that was earned.
3. **The fix made the absence visible** — by raising, returning `NaN`/`None`,
   adding a distinct state or validity flag, skipping instead of passing,
   warning, or failing closed.

All three, or it is not a case.

## EXCLUSIONS, frozen

- Refactors, renames, typing, formatting, performance.
- Corrections to a *computed* value's arithmetic that leave its measured/unmeasured
  status unchanged. Wrong-but-measured is a different defect.
- Fixes where the pre-fix behaviour was **already distinguishable** — `None` → raise
  is hardening, not silent pass. Requirement 2 must fail for the *old* code.
- Documentation, CI config, dependency bumps.
- Test-only changes, **except** where the test itself silently passed (an assertion
  that could not fail, a skip reported as a pass). Those are included and tagged
  `SP-8 INERT_CONTROL`, because a green test that never ran is the defect class
  aimed at itself.

## HARVEST QUERIES, frozen

Declared in advance so candidates cannot be hand-picked, and so recall is a known
quantity rather than an implied one.

**Q1 — commit message** (case-insensitive, any match):
`silent(ly)?` · `always (return|pass|true|zero)` · `never (fire|ran|run|check|trigger)` ·
`was not (check|validat|measur)` · `return(s|ed)? (0|0\.0|True|"pass") when` ·
`empty (list|input|dict|sequence|array)` · `no data` · `fail(s|ed|ing)? (open|closed)` ·
`swallow` · `missing (data|value|logprob)` · `default(s|ed)? to (0|true|pass)` ·
`false negative` · `undetected` · `unreported` · `treated as (valid|success|pass)` ·
`divide by zero` · `division by zero` · `skipped .* report` · `vacuous`

**Q2 — diff shape** (the stronger signal): a hunk that **removes** a line matching
`return\s+(0|0\.0|True|1\.0|"(pass|ok|valid|healthy)")` and **adds** a line matching
`raise|nan|None|warn|skip|log\.(warn|error)` within the same hunk.

Both queries run over all 61,702 commits. Their yields are reported separately.

## ADJUDICATION, frozen

Every candidate is judged by **three independent reviewers, each prompted to
REJECT**, each given a distinct lens, each shown the real diff:

- *lens A* — argue requirement 2 fails: the old value **was** distinguishable.
- *lens B* — argue requirement 1 fails: that path was unreachable, or the input was
  determinate rather than absent.
- *lens C* — argue this is an excluded category: refactor, arithmetic correction,
  hardening, perf.

**ACCEPTED only when rejecters fail to reach a majority. Uncertainty resolves to
REJECT.** Every verdict is published with its rationale and the commit URL, so any
reader can check a case rather than trust the label.

## GATES, frozen

**G1 — YIELD.** If fewer than **12 cases** survive adjudication, SP-EXT is
published as-is with its size stated in the title, and **no claim is made that the
defect class is common in the field.** A small corpus is still a corpus; a small
corpus described as evidence of prevalence is a lie.

**G2 — ACCEPT RATE, two-sided.** The accept rate is reported with equal
prominence to the count. **If it exceeds 80%, that goes in the title** — an
adjudication that rejects almost nothing is not an adjudication, and the same
objection applies here as applied to the 24/24 refutations in the flattering run.
If it falls below 20%, the harvest queries are near-noise and that goes in the
title instead.

**G3 — SPREAD.** Cases concentrated in fewer than 4 repositories are reported as
`NARROW`, and no cross-project claim is made. One project's house style is what
this whole exercise exists to rule out.

**G4 — NO SELF-SELECTION.** Candidates come only from Q1 and Q2 as written above.
Nothing is added because it looked interesting. If a query is wrong, the run is
re-done under a new preregistration with the change disclosed — queries are not
edited mid-harvest.

**G5 — RECALL IS UNKNOWN AND SAID SO.** These queries find commits whose *authors
described* the fix a particular way, or whose diff took a particular shape. They
cannot find silent-pass fixes described some other way. **SP-EXT is a lower bound
on incidence and must never be quoted as a rate.** No sentence of the form "X% of
eval libraries contain this" is licensed by any version of this corpus.

## what SP-EXT is for

A benchmark that detectors can be scored against without being scored on the code
that produced them. `contract` (3/5), `flattering` (10% recall, 0/8 external) and
`edges` (0/20) were all measured on our own corpus or on nothing. SP-EXT lets the
next instrument be measured on someone else's.

It is also the artifact that would make the defect class citable by people who do
not work here, which is the only thing that turns a house finding into a field
finding.
