# PROBE E on styxx: 43% of exercised decision terms could not have gone the other way

**Date:** 2026-08-13. **Instrument:** `probe_e_runtime.py` (method and limits in
`PROBE_E_METHOD_2026_08_13.md`). **Population:** the repository's own test suite, 129
files, one interpreter per file. **Artifacts:** `probe_e_styxx_full.json` (raw),
`probe_e_styxx_joined.json` (verdicts + census join).

## The measurement

4,985 boolean decision terms were instrumented across the `styxx` package and driven by
its own tests.

| verdict | terms | |
|---|---:|---|
| `LIVE` — observed both true and false | 1,048 | |
| `CONSTANT_TRUE` — only ever true | 326 | in an `or`, forces pass |
| `CONSTANT_FALSE` — only ever false | 474 | in an `and`, forces silence |
| `UNDERPOWERED` — 1–7 observations | 1,407 | excluded; says nothing |
| `NEVER_REACHED` — 0 observations | 1,730 | excluded; the suite never ran it |

**Of the 1,848 terms the suite exercised at least eight times, 800 — 43.3% — never took
more than one value.** Those gates could not have decided otherwise on the data their
own test suite provides.

246 of the dead terms were evaluated **more than a hundred times each** and still never
varied. The heaviest:

| n | verdict | site |
|---:|---|---|
| 53,771 | `CONSTANT_TRUE` | `anthropic_hack.text_features:extract_features` L161 |
| 45,016 | `CONSTANT_TRUE` | `analytics:log_stats` L958 `e.get('gate')` |
| 31,554 | `CONSTANT_FALSE` | `crossmind:auroc` L129 `len(pos) == 0` |
| 24,983 | `CONSTANT_FALSE` | `analytics:_read_and_cache_audit` L114 `not line` |
| 16,779 | `CONSTANT_TRUE` | `weather:_bucket_for_hour` L103 `start < end` |

A term evaluated fifty thousand times without ever changing value is not lightly-tested
code. It is a branch the program has been carrying, and paying for, without ever using.

## The static screen was wrong in both directions — and the misses dominate

This is the result that changes how the census should be quoted. Every previous
statement about it — in `EXTERNAL_CENSUS_2026_08_13.md`, in the prior-art assessment —
described it as an **upper bound**: syntactic candidates, some of which would turn out
to be live gates. That was half the story, and the smaller half.

Joining execution against the static census at function level (the granularity both
sides support):

- **108** functions carried an at-risk shape
- **48** confirmed dead by execution
- **9** refuted — flagged, exercised, every term varied
- **51** unmeasured — flagged, but the suite never pressed them hard enough to answer

So the false-positive load is real but modest: 9 of the 57 flagged functions that could
be adjudicated were wrong, about 16%.

The false *negative* load is the story:

| dead terms | count |
|---|---:|
| inside functions the census flagged | 250 |
| **inside functions the census never flagged** | **550** |
| **static screen miss rate** | **68.8%** |

**More than two thirds of the dead gates in this repository are invisible to the static
screen.** They sit in functions with no suspicious vocabulary at all — no
`PRESENCE_TEST`, no text `LENGTH_TEST`, nothing the AST pattern was written to notice.
They are dead for ordinary reasons: a config flag that is always set in practice, a
guard against a condition the data never produces, an `or` arm kept alive by habit.

The census's 20.5% was never an upper bound on the defect rate. It was a **differently
biased sample** of it, and the two errors do not cancel — they point opposite ways and
the larger one was never measured until now. The honest revision:

> the static census counts a syntactic shape. It over-calls by roughly 16% among the
> functions it can adjudicate and misses roughly 69% of real dead terms outright.
> Its number is not a bound on anything. It is a screen for a specific vocabulary, and
> it should be quoted only as that.

`EXTERNAL_CENSUS_2026_08_13.md` compares styxx at 19.4% against a pooled third-party
14.6%. That comparison is now known to be **measuring the wrong quantity on both sides**
and should not be cited as evidence about defect prevalence in either direction.

## What this licenses

**Licensed:** on the data its own test suite provides, 43.3% of the exercised decision
terms in this package are indistinguishable from constants. For any gate that produced a
published number, that is disqualifying on its own — the number carries no information
the constant did not already fix.

**Not licensed:** "43% of styxx's logic is broken." A `CONSTANT` verdict is a statement
about a *pairing* of instrument and population. Many of these terms are defensive checks
that are correct precisely because the condition never arises; a guard that never fires
is doing its job, not failing at it. What the verdict establishes is narrower and still
serious: **the test suite cannot tell these gates from hard-wired ones**, so it provides
no evidence that they work.

Distinguishing a healthy dormant guard from a dead gate needs the next step, and it is
not a static one: re-run against a population *designed* to exercise the term, and see
whether it can be made to move.

## Coverage, stated rather than hidden

1,730 terms (34.7%) were never evaluated at all and 1,407 more fewer than eight times.
**63% of the package's decision terms are not meaningfully exercised by its own suite.**
That figure is not a finding about correctness and it is the necessary context for the
43.3%: the denominator is the exercised minority.

One file, `tests/test_anthropic_hack.py`, cannot be measured by this instrument — it
dies with an access violation under instrumentation while passing 14/14 without it. The
crash is the instrument's, not the subject's, and every term only that file would have
exercised is `NEVER_REACHED` for a reason unrelated to the code under audit.

## The harness defect this run exposed

The first external run of this same instrument, against numpy, reported a confident
**69% dead rate** for a mainstream library. Every one of its 178 chunks was recorded
`ok`. No test had executed: pytest was exiting 4 on `ModuleNotFoundError: hypothesis`,
and the observations came from import time alone. The chunk runner checked only that a
result file appeared, never that the population ran.

That number would have been the day's headline external finding, and it was an artifact
of an empty population certified as a successful run — the exact class of defect this
program exists to catch, committed by the harness built to catch it. The status check now
reads pytest's exit code, and a chunk whose suite did not run is labelled
`pytest_exit(N) — no population` rather than `ok`.

## Reproduction

```
python probe_e_runtime.py --pkg styxx --tests tests --chunked \
    --census census_styxx_broad.json --json probe_e_styxx_full.json
python probe_e_runtime.py --join-only probe_e_styxx_full.json \
    --census census_styxx_broad.json --json probe_e_styxx_joined.json
```
