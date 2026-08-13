# The silence subclass — and why it is not the whole law

**Date:** 2026-08-13, written after the day's defects were counted.
**Status: EXPLORATORY.** The classification below is **retrospective**, which is the
exact practice `PRIOR_ART_2026_08_13.md` said today's ledger should stop doing
("pre-register the defect classification *before* the passes run"). It is offered as a
hypothesis worth pre-registering next time, not as a result. Nothing here is a finding.

## The tempting generalisation

By the end of the day one sentence seemed to cover everything:

> **An instrument's null output is ambiguous between "there is nothing to report" and
> "I could not have reported anything."**

It is a good sentence. A gate that extracts zero claims, a probe that instruments zero
terms, a checker nobody calls, a suite that runs no tests — each emits the same thing a
healthy instrument emits when the world is fine. The failure is invisible *in the
instrument's own output*, which is what makes it survive review by people who are
looking directly at it.

It was tempting to declare that the law of the day. Checked against the day's actual
defect list, it does not hold.

## Testing it against the record

Each defect below has a receipt elsewhere in this directory. Classification is mine and
is the weak link.

**Fits — the null output was indistinguishable from a broken instrument (≈15):**

- `claim_audit` zero-claim gate: `GATE: PASS` after extracting 0 claims from 46 sentences
- `meta_audit.judge`: `conscience_fired OR refused`, conscience firing 12/12 — could not
  come out below 1.0
- G2 `memory_integrity`: 2 of 3 detector terms constant across the population
- the escrow verification that self-agreed across all three share pairs
- `epistemic_surface`'s `i can\b`, which silently ate `can't` and discarded the case
- `repo_epistemic_audit` v2 reporting 0.998 grounding with **no chance floor**
- both gates returning **zero claims** on a status report with a fabricated work item
- the same gates again on subject-dropped register — zero claims, real fabrication
- the topical-evidence veto firing on a source too terse to discriminate
- the gates being **wired into nothing** — silence that was absence, not verdict
- the prober serving stale bytecode: **zero terms instrumented**, suite green
- the prober importing a second copy of itself: 3,349 terms, **zero observations**
- the census join reading the wrong key: a confident zero
- 178 numpy chunks marked `ok` while pytest exited 4 and **no test ran**
- `NEVER_REACHED` collapsed into `UNDERPOWERED` by the chunked merge

**Does not fit (≈8):**

- the `<` tiebreak that let ties re-admit the short-path premium — a strictness bug with
  a non-null, wrong output
- MIN_CELL applied to the paired denominator instead of the discordant one — a real
  number, overstated power
- `_resolve_by_context` labelling arbitrary matches as `context` — a wrong label, not a
  silence
- the receipt check accepting any evidence in the window — a confident *positive*
- ambient evidence discharging a first-person claim — again a positive, not a silence
- the passive pattern reading "nothing new has landed" as a completion claim — a false
  positive, the opposite direction entirely
- the same pattern on "until memory_integrity is rebuilt"
- the recorder name that Python privately mangles inside class bodies — this one would
  have raised `NameError` loudly, and belongs to no interesting class at all

## What survives

The silence framing covers roughly **two thirds**. The day's original framing covers
nearly all of them:

> **An instrument reports more resolution than its method supports, and the
> overstatement is invisible in its own output.**

Silence is the **special case where the overstated quantity is zero** — and it is the
case that dominates the auditors' own tooling. Of the fifteen fitting instances, **ten
are in instruments written today by people who had already published the bug class in
writing**. That skew is worth a hypothesis: null results feel like non-events, so they
recruit less scrutiny than a number does. Nobody re-derives a zero.

## The design rule this suggests

Stated as a rule rather than a finding, because it has not been tested:

> **Every instrument should distinguish its null result from its inability to produce a
> non-null one, in its own output, without the reader having to ask.**

Applied in this directory today, mostly after being forced to:

- PROBE E reports `UNDERPOWERED` and `NEVER_REACHED` as distinct from `LIVE`, and
  excludes both from the denominator rather than counting them healthy
- the chunk runner labels `pytest_exit(N) — no population` instead of `ok`
- the receipt gate carries a three-valued standing — `BACKED`, `AMBIENT_ONLY`,
  `UNBACKED` — because collapsing the middle one would have been a lie in either
  direction
- the falsifiability receipt separates `REFUSED__no_live_terms` from
  `REFUSED__nothing_adjudicable`: the apparatus could not fail, versus the receipt could
  not speak
- `--selftest` everywhere, asserting the instrument can **both** fire and stay quiet

The cheapest version of the rule is the one that keeps working: **print the denominator
next to the rate, and the n next to the verdict.** Most of today's fifteen would have
been visible at a glance.

## What would make this a result

Pre-register the two categories and their boundary, then classify a *future* day's
defects blind — ideally by someone who did not write the code. Retrospective
classification by the author of both the bugs and the taxonomy is the weakest possible
evidence for a taxonomy, and this document is exactly that.
