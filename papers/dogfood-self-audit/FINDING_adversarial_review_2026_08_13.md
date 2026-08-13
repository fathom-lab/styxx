# Twelve skeptics against one day's findings: four survived, and the flagship did not

**Date:** 2026-08-13, evening. **Method:** seven independent agents, one per claim, each
instructed to *refute* and to default to `refuted=true` under uncertainty. Every wounded
claim then went to a second, independent adjudicator told to check the **skeptic**, not
the author — because skeptics over-reach as readily as authors over-claim, and this
program had already shipped one non-discriminating test inside the fix for another.
Twelve agents, 271 tool calls, 1.14M tokens, 39 minutes.

Every figure in the review was re-derived by the adjudicators from the raw JSON or by
executing the instrument. None was taken from the author's report.

## Score

| claim | verdict |
|---|---|
| styxx dead rate | stands with rewording — **the number changes** |
| numpy dead rate | stands with rewording — **the number changes** |
| census join collisions | **stands as written** (verified identical under a collision-free key) |
| instrumentation validity | stands with rewording; **two published statements withdrawn** |
| receipt attribution | conscience receipt reworded; **gate receipt withdrawn entirely** |
| cross-repo framing | **withdrawn as a comparison** |
| what "dead" means | **withdrawn as written**; replaced with a decomposition |

Three of seven claims survived intact. The one that came through untouched was the one I
had flagged to the reviewers myself as most likely broken — the function-level join key,
which I suspected of basename collisions. It was rebuilt with a collision-free key and
returned **bit-identical results on both repositories**. The suspicion was sound; the
defect was not there.

## The four defects

**1. Renaming a failure is not excluding it.** The exit-code check added hours earlier
set a chunk's status to `no population` and then fell straight through into the merge
with no `continue`. All 30 of numpy's f2py chunks — pytest exit 5, zero tests collected —
had their import-time observations merged anyway: 2,075 terms, 15 crossing the
observation floor, 12 scored dead, from suites that never ran. **The label said one
thing and the code did another**, and it survived because the fix was never tested
against the case it was written for.

**2. 175 of 800 "dead gates" were not decisions.** An exact-`term_id` AST reconstruction
classified all 4,985 terms by consumer position — 4,985/4,985 matched, no fuzzy
matching — and found 21.9% in value position: `float(x or 0.5)` picks a default,
`prefix or "$"` coalesces. **21 are constant by mathematical construction**, movable by
no population that could ever exist. On adjudicative terms alone the rate is 39.9%, and
~36% if defensive guards are also excluded — two careful readers got 36.9% and 36.1%,
and that ~15-point spread on where a guard stops being a gate is itself a finding about
the concept.

The showcase table was the worst part of the document. Row 1, at n=53,771, is
`if w and w[0].isupper()` where `w` comes from `\b\w+\b` — a **logical tautology**, and
50,000 random strings produce no empty match. Rows 2–4 are a value coalesce and two
early-exit guards. Only row 5 survives: `weather:_bucket_for_hour` is genuinely dead and
proves the suite never passed an hour between 22:00 and 05:00.

**3. A claim of mine was simply false.** I published that `tests/test_anthropic_hack.py`
"cannot be measured by this instrument" and that "the crash is the instrument's, not the
subject's." It runs to completion under instrumentation **5 times out of 5** — confirmed
again in the corrected re-run, chunk 13, `ok`. The stated mechanism was wrong too:
`_probe_e_rec(tid, EXPR)` evaluates `EXPR` as a call *argument*, before the recorder's
frame exists, so instrumentation cannot deepen the subject's recursion. **One observed
crash was generalised into a property of the instrument on a sample of one**, and a
`--stack-mb 256` apparatus was built on the diagnosis.

**4. The receipt certified measurements that could not have come out differently.**
Re-scoped on the 58 drafts where the conscience fired every single time (value literally
1.0) and the 25 where it never fired (0.0), it returned `OK__path_could_have_failed` at
37.5% and 25.5% live — while its own rows correctly recorded every terminal decision term
as CONSTANT.

Pooling was the mechanism. Character-level tokenisation loops generate tens of thousands
of live observations with no bearing on the verdict and outvote the handful of terms
that decide. Its single heaviest "live" term was `' ' in phrase` at n=30,174 — a
compile-time property of a lexicon file, **invariant under every possible input**,
certified LIVE.

This is the worst defect of the day and the reviewers said so: it is the only artifact
whose entire purpose is the verdict rather than the number; it violates the lab's own
standing rule that *a leg which cannot fail must not gate*, inside the instrument built
to enforce that rule; and its selftest passed because the fixture was degenerate in the
one way the real population is not — a single pinned term with **no bystanders**.

## The fixes, and the bug in the fix

Term position is now recorded and the headline computed on adjudicative terms alone. The
receipt gained two checks in front of its verdict: `mark_item()` per unit the value
aggregates over, refusing outright when every item produced the same outcome; and a phi
coefficient between each term's per-item value and the result, refusing when no live
term tracks the outcome. Restricting to adjudicative terms does **not** suffice on its
own — a tokenisation loop's `if` is adjudicative too.

On the real conscience receipt this cuts 38 live terms to **17 outcome-linked**, and the
survivors are the actual gates — `overconf_check` `proba >= th` at phi=+0.920,
`tool_cogn_audit_with_advice` `s < 0.4` at phi=−1.000 — rather than the tokenisers.

Validation went 5 → 8 cases on the prober and 5 → 12 on the receipt, including the exact
counterexample the reviewers constructed: a rate of literally 1.0 with dozens of live
adjudicative bystanders, which the old code passed and the new code refuses.

**And the position fix shipped with a bug.** The chunk merge copies a fixed list of
metadata keys; `pos` was not in it. Every term's position was discarded in transit,
`report()` classified all 4,995 terms as value-position, and the run printed
`adjudicative powered: 0` with a headline of `None`. It was caught in a single run
**because it failed loudly**. Had the field defaulted to `"adjudicative"` instead of
`None`, a confident and wrong headline would have shipped — which is this directory's
entire argument, arriving as a bug in the fix for the previous instance of it.

## What the exercise is worth

The review cost more tokens than the work it audited. It overturned the day's flagship
result, corrected two rates, withdrew one artifact entirely, and falsified one statement
that was not merely overstated but **untrue**. Every one of those was invisible to the
author, who had spent the day specifically hunting this defect class and had written
three documents about it.

The load-bearing design choices, for anyone repeating this: skeptics were told to
**refute** rather than evaluate, and to default to refuted under uncertainty; each got
one claim rather than the whole set; and every refutation was checked by an independent
adjudicator who re-ran the decisive test, which is what caught the first skeptic's
overstated 41.6% and its `1-of-15` crash statistic. **Adversarial review without an
adjudicator is just a second author.**
