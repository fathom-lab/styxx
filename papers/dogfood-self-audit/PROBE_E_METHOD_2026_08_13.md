# PROBE E — measuring dead gates instead of guessing at them

**Date:** 2026-08-13. **Instrument:** `probe_e_runtime.py`. **Status of results:** see
`FINDING_probe_e_*.md`; this file is the method and its limits, written before the
numbers so the numbers cannot shape it.

## Why a runtime probe was owed

`falsifiability_census.py` is a static screen. It reports **shapes**: a decision
expression carrying a term that *could* be constant. `census_discrimination_control.py`
established that it cannot distinguish a dead gate from a live one sharing the same
syntax, so every census figure since has been quoted as an **upper bound** on the defect
rate rather than the rate. `PRIOR_ART_2026_08_13.md` closed by naming the missing piece
directly: the surviving contribution is falsifiability analysis applied to *measurement
instruments*, and it needs execution against a population to be worth anything.

Static analysis can say *this gate is shaped like one that cannot fail*. Only execution
can say *this gate did not fail, on this data, and could not have*.

## What it does

The prober rewrites the target package's ASTs in memory at import time. Operands of
`BoolOp` are wrapped in a recorder that notes the truthiness and returns the value
unchanged; so are the tests of `if`/`while`/`assert`/`IfExp` and bare comparisons in
`return` position. The package is then driven by its own test suite, and each term's
observed value set is reported.

**This is an instrument-defined population, not a census of the package's gates.** An
earlier version of this sentence claimed "every operand of every `BoolOp`, plus bare
comparisons in `if` and `return` position," which was wrong in both directions: the `if`
handler accepts a wider node set than `return` does, and there was no handling of
`while`, conditional expressions, or comprehension filters at all. A term count is a
property of what the rewriter reaches, and 106 terms in 15 modules that the suite never
imports are absent from the denominator rather than counted against coverage.

Each term also records its **position**: `adjudicative` if its value is consumed as a
decision (an `if`/`while`/`assert` test, a returned expression, a conditional test), or
`value` if it merely selects a result — `float(x or 0.5)` picks a default and
`prefix or "$"` coalesces. Adversarial review found 175 of styxx's 800 originally
reported dead terms sitting in value position, 21 of them constant by mathematical
construction. **The headline rate is computed on adjudicative terms alone**; pooling the
two counts default-picking as dead logic.

Wrapping **operands** rather than whole expressions preserves short-circuit evaluation
exactly — a right-hand operand records only when Python would have evaluated it. That is
a second measurement rather than a limitation: an operand that never records at all is a
term the population never reached.

Five verdicts, and the distinctions between them are the point:

| verdict | meaning |
|---|---|
| `LIVE` | observed both true and false — the gate could have gone either way |
| `CONSTANT_TRUE` | only ever true; in an `or`, forces pass |
| `CONSTANT_FALSE` | only ever false; in an `and`, forces silence |
| `UNDERPOWERED` | evaluated fewer than `OBS_FLOOR` (8) times — says nothing |
| `NEVER_REACHED` | the population never evaluated it at all |

Only `LIVE`, `CONSTANT_TRUE`, and `CONSTANT_FALSE` enter the denominator. Counting
unreached or barely-reached terms as healthy would credit thin coverage as evidence of
soundness, which is the overstatement this whole program exists to catch.

## PROBE E measures code, not knowledge — tested, pre-registered, and settled

This is stated permanently because a pre-registered test said it should be. **Whether an
instrument could have failed does not predict whether the claims it produced were later
withdrawn.** On 131 modules, against 75 verified retraction entries, modules on the
causal path of a retracted claim have a *slightly lower* dead-term rate than the rest
(median 0.333 vs 0.369, U = 1052, p = 0.248).

The prereg fixed the consequence of a null in advance and this file honours it. A dead
gate is a real defect in a real instrument; it is **not** a marker for results that will
need retracting. Anyone quoting a dead-term rate as evidence about the reliability of a
lab's *findings* is making a claim this repository tested and failed to support.
`FINDING_prereg_retraction_null_2026_08_13.md` has the design and the confound.

## What a CONSTANT verdict licenses, and what it does not

**Licensed:** the population could not distinguish this gate from one with that term
hard-wired. For a gate that produced a published number, that is disqualifying on its
own — the number carries no information the constant did not already determine.

**Not licensed:** "this term can never vary." A different population may falsify it
tomorrow. The verdict is a statement about a *pairing* of instrument and data, and every
row carries its n so the reader can see how hard the question was pressed.

This is mutation testing approached from the opposite side. Mutation asks whether
changing the code changes the tests' verdict; PROBE E asks whether the data ever changed
the gate's inputs. Both are asking what the silence means. `PRIOR_ART_2026_08_13.md`
holds the citation obligation: mutation testing is the origin, and the claim here is the
specialisation to instruments rather than test suites.

## Validation before belief

`--selftest` drives three fixtures with known answers: a term wired true, a term wired
false, and a gate that genuinely varies. It also asserts that a short-circuited operand
is reported rather than dropped. A prober that has never been shown both outcomes on a
known answer is a hope, and this file's own argument is that such instruments are
everywhere.

The selftest earned its keep immediately. Its first version swept the `live` fixture over
a range on which **both** of that gate's terms happen to be constant, and the prober
correctly reported the live gate as dead. The prober was right; the population was the
defect. That is the finding restated as a bug report against its own control, and the
driving range is now part of the fixture rather than an afterthought.

## Four defects found in the prober while building it

Each produced output that looked like a clean result:

1. **Bytecode cache.** `SourceLoader.get_code` served `__pycache__` and never called the
   rewriter. The suite passed and the run reported **zero instrumented terms** — a
   broken instrument and an empty finding are indistinguishable from outside.
2. **Two module copies.** Run as a script, this file is `__main__`, so the subject's
   `import probe_e_runtime` created a *second* module object with its own counters.
   3,349 terms instrumented, zero observations. The recorder is now injected into the
   subject's globals instead of imported.
3. **Name mangling.** The injected identifier began with two underscores, which Python
   privately mangles inside any class body — every gate in a method would have raised
   `NameError`. One leading underscore now.
4. **Silent join.** The census writes `function`; these rows carry `func`, and the census
   records the `BoolOp` line while the rewriter records each *operand's* line. Keying on
   `(func, line)` would have matched almost nothing and printed a confident zero,
   indistinguishable from "the census predicted no real defects." The join is now at
   function level and labelled as such, rather than dressed up as per-term precision the
   data cannot support.

## A methodological error worth recording

The first full run was contaminated by the author. Chunks are separate interpreters that
re-read the script on every spawn, and the script was edited mid-run — adding a code path
that referenced an argparse flag not yet declared. Twelve consecutive chunks died with
`AttributeError` and were recorded as crashes of the subject.

**The instrument was mutated while it was measuring.** The chunk log made it visible
within a minute, and the run was discarded and restarted against a frozen file rather
than repaired. Worth stating plainly because the failure is not exotic: it is the
ordinary consequence of treating a running measurement as editable, and nothing in the
tooling prevented it.

## Population, and its honest limits

The driving population is **the repository's own test suite** — one interpreter per test
file, so a crashing module costs one file instead of the run. Files that fail to run are
listed in the report, because a term left unmeasured by harness failure must not be
confused with a term the code kept quiet.

> **WITHDRAWN — the paragraph below is false and is kept only because deleting a
> retracted claim hides that it was made.** Adversarial verification ran
> `tests/test_anthropic_hack.py` under instrumentation **5 times out of 5**: exit 0,
> full report written each time, 69 terms observed. It is measurable. The stated
> mechanism is also wrong: `_probe_e_rec(tid, EXPR)` evaluates `EXPR` as a call argument
> *before* the recorder's frame is pushed, so instrumentation cannot deepen the
> subject's recursion. One observed crash was generalised into a property of the
> instrument on a sample of one, by the same reasoning this program exists to catch —
> and the `--stack-mb 256` apparatus was built on it.

~~**One file is unmeasurable by this instrument, and the instrument is at fault.**
`tests/test_anthropic_hack.py` dies with an access violation (`0xC0000005`) under
instrumentation. Run without it, the same file passes 14/14 in 23s. The crash is
therefore mine, not the subject's — the extra frame per decision term is enough to break
something in that module's execution — and every term only that file would have
exercised is reported `NEVER_REACHED` **for a reason that has nothing to do with the
code under audit**. Checking this took one command and it inverts the reading of that
file's rows completely; without the control, an instrument-induced crash would have been
silently recorded as dead code in the subject.~~

The suite is a convenience population, not a designed one. It over-represents what the
authors thought to test, which biases *against* finding dead gates in well-tested code
and *toward* `NEVER_REACHED` elsewhere. A gate that is dead under its own test suite is
nonetheless dead where it matters: that suite is the evidence its authors rely on.

## Reproduction

```
python probe_e_runtime.py --selftest
python probe_e_runtime.py --pkg styxx --tests tests --chunked \
    --census census_styxx_broad.json --json probe_e_styxx_full.json
python probe_e_runtime.py --join-only probe_e_styxx_full.json --census <census> --json <out>
```

No network, no model calls. ASTs are rewritten in memory; nothing on disk is modified.
