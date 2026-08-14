# Pre-registered result: dead instruments do NOT predict retracted claims

**Date:** 2026-08-13. **Prereg:** `PREREG_dead_gates_predict_retraction_2026_08_13.md`,
committed `c2340ed` before the exposure ledger was built and before the analysis script
existed. **Analysis:** `analyze_retraction_falsifiability.py`, written before either
variable was visible. **Artifacts:** `RETRACTION_LEDGER.json`, `PREREG_RESULT.json`.

## The result

**H1_NOT_SUPPORTED.**

| | exposed (retraction-linked) | comparison | test |
|---|---:|---:|---|
| **H1** — adjudicative dead rate | median **0.333** (n=23) | median **0.369** (n=108) | U = 1052, **p = 0.248**, rank-biserial 0.153 |
| **H2** — exercised fraction | median **0.700** | median **0.553** | p = 0.100, rbc 0.218 |
| covariate — powered terms/module | median 9.0 | median 6.5 | p = 0.450 |

Modules on the causal path of a retracted, struck, voided or withdrawn claim have a
**slightly lower** dead-term rate than the rest of the package, and the difference is not
significant. The pre-registered hypothesis fails, and it fails in the direction opposite
to the one predicted.

**H2 also fails, and also in the direction opposite to the one recorded.** The prereg
stated in advance that the author expected H2 — that retraction-linked modules would
simply be *less exercised*, making any result a coverage finding rather than a
falsifiability one. They are more exercised (0.700 vs 0.553), not less. Both the
interesting hypothesis and the deflationary one are wrong.

## What this costs PROBE E

The prereg fixed what a null licenses, so this is not a matter of interpretation:

> **H0: PROBE E measures code, not knowledge. Say so in the method file permanently.**

That statement now stands. On this repository, on 131 modules, with 75 verified
retraction entries, **whether an instrument could have failed does not predict whether
the claims it produced were later withdrawn.** A dead gate is a real defect in a real
instrument, and it is not a marker for results that will need retracting.

This is the single most useful thing measured today, precisely because it is negative.
Without it, "43% of our decision terms are dead" sits one rhetorical step away from
"and that is why results get retracted" — a claim nobody had tested, that sounds
obvious, and that the data does not support.

## Why the null is credible rather than merely underpowered

- **n is not the problem for H1's direction.** 23 exposed modules against 108, and the
  exposed group is on the *low* side. More data would have to reverse a sign, not
  sharpen an estimate.
- **The pre-registered floor was met, not adjusted.** MIN_MODULES = 5; 28 distinct
  ledger modules matched, 23 with a powered adjudicative term. No floor was moved after
  seeing data, which is the failure this repository committed and withdrew earlier the
  same day.
- **The exposure set is large and independently verified.** 77 candidate claims
  harvested from papers, MEMORY, git history and closed-negative sweeps; 75 retained
  after an independent adjudicator checked each retraction quote at its cited path and
  spot-checked module mappings; 2 rejected.
- **The outcome is the corrected one.** Adjudicative terms only, value-position operands
  excluded, process-count pseudo-replicates flagged — the definition adversarial review
  forced earlier the same day. Using the old pooled rate would have tested a quantity
  known to be wrong.

## The confound the prereg named, visible in the data

The prereg named it in advance: retracted claims received **more attention** — they were
investigated, which is how they came to be retracted — and attention may drive both the
retraction and the discovery of dead gates nearby.

The H2 row is that confound made visible. Retraction-linked modules are **more
exercised** than the rest of the repository. They are the code this lab looked at hardest,
so they are better tested, which plausibly suppresses their dead rate and works against
H1. A prospective design — freeze the dead rates, then wait to see which claims get
retracted — is the only version that separates this, and it is the obvious successor.

## Blinding: two harness failures, both declared

Neither attempt at the exposure ledger was perfectly blind, and both leaks were in the
**harness**, not in any agent's behaviour.

**Attempt 1 (void).** Each phase passed the previous phase's full JSON downstream, and
the agents' own contamination declarations quoted outcome rates verbatim inside free-text
notes. The adjudicator read repo-level rates and two phi values before classifying. It
declared this unprompted. See amendment 1.

**Attempt 2 (used here).** Payloads sanitised to a field whitelist. Three residual leaks,
all declared by the adjudicator without being asked:
1. an `ls -la` returning **filenames and byte sizes only** for blocked artifacts;
2. **the real one** — `DAY_2026_08_13.md` was not on my blocklist and contains PROBE E
   prose, leaking one repo-level instrument-agreement figure (~16%);
3. the operator's `MEMORY.md` index, **auto-injected into every agent's context by the
   harness**, carrying two term-level facts about non-styxx modules.

None identifies which styxx modules are dead, so none could steer inclusion or exclusion.
The adjudicator's three module strips and two rejections all *shrink* the exposure set,
which is conservative against H1 — and the result is null regardless, so no leak is
rescuing anything.

**The third leak is not fixable by me.** The harness injects the operator's auto-memory
into every subagent's context. Any future blinded classification in this environment
inherits that, and it should be stated rather than assumed away.

## Limits

Retrospective and observational; one repository; one suite; one day's snapshot. Module
level is coarse — a retraction implicates a pipeline, and every module on that pipeline
is marked exposed whether or not it contributed the error. 5 ledger modules could not be
matched to instrumented modules and are named in `PREREG_RESULT.json` rather than
dropped silently.

## Reproduction

```
python analyze_retraction_falsifiability.py --probe probe_e_styxx_v2.json \
    --ledger RETRACTION_LEDGER.json --json PREREG_RESULT.json
```
