# SILENT-PASS — a benchmark of failures that report success

**What it is.** A labelled corpus of real defects in which a measurement failed,
or never ran, and the system returned a value indistinguishable from a healthy
one. Every case is drawn from production code, has a confirmed fix, and carries
its pre-fix and post-fix state in version control.

**Why it exists.** Static analysis benchmarks measure crashes, taint, and type
errors. This class produces none of those: the value is well-formed, in range,
and wrong in exactly one direction — toward *fine*. Types check, linters pass,
coverage is green. The defect is invisible to every tool that asks "is this
malformed?" and visible only to one that asks "what does this number mean when
the thing that computes it didn't run?"

We publish it because a class nobody can measure is a class nobody fixes.

## the shape

A case is SILENT-PASS when all four hold:

1. **A measurement path failed or was skipped** — an exception, a missing
   dependency, an empty window, an absent field, a degenerate statistic.
2. **A value was still returned** — no raise, no None, no error field.
3. **That value reads as healthy** — the best or a benign point in its range:
   `trust=1.0`, `risk=0.0`, `gate="pass"`, `r2=1.0`, `divergence=0.0`,
   `valid=True`, `entropy=0.0`.
4. **A consumer acts on it** — a gate, a calibrator, a trainer, a report, a
   human reading a dashboard.

Remove (4) and it is a latent bug. Remove (3) and it is an ordinary one.

## subtypes

| id | name | signature |
|---|---|---|
| SP-1 | HEALTHY_ON_CRASH | an `except` path returns a passing verdict |
| SP-2 | SENTINEL_DEFAULT | an absent *measurement* defaults to a number |
| SP-3 | UNDEFINED_AS_NUMBER | a degenerate statistic returns a value instead of refusing |
| SP-4 | TRUTHY_GATE | a decision made on an object's truthiness, or a disjunct that can never decide |
| SP-5 | CRASH_TO_SENTINEL | a crash swallowed into a sentinel, healthy on the sentinel |
| SP-6 | UNMEASURED_AS_MEASURED | an empty input produces a full, confident result |
| SP-7 | SELF_CONFIRMING | ground truth derived from the system's own verdict, then trusted |
| SP-8 | INERT_CONTROL | a knob, guard or verification leg that cannot change any outcome |

## how to score against it

```python
from benchmarks.silent_pass import load_cases, score

def my_detector(source: str, filename: str) -> set[int]:
    """Return the line numbers you flag."""
    ...

print(score(my_detector, tolerance=10))
```

`score()` reports **recall** (cases whose defect line was flagged, within
tolerance) broken down by subtype, and it reports what it cannot tell you:
this corpus contains only true positives, so **it cannot measure precision**.
A detector that flags every line scores 1.00 here and is worthless. Precision
must be estimated separately, on unlabelled code, by hand — see
`papers/CENSUS_absence_2026_08_19.md` for how we did that on ours (5 of 14).

Reporting recall from this corpus without a precision estimate beside it is
the exact error the corpus exists to document.

## baseline: our own detectors, scored

Run `python -c "from benchmarks.silent_pass import main; main([])"`.

| detector | recall | strong on | blind on |
|---|---:|---|---|
| `styxx.absence` | **9/20 (45%)** | SP-1 1/1, SP-4 2/2, SP-5 1/1 | SP-6 0/5, SP-7 0/2 |
| `styxx.loops` | **5/20 (25%)** | SP-7 2/2 | SP-1, SP-3, SP-4, SP-5 all 0 |
| both | **13/20 (65%)** | | SP-6 1/5 |

Two things this says, both about us:

**The tools are complementary, not redundant.** Each catches cases the other
misses, and the union beats either alone. That is the argument for shipping two
instruments rather than folding one into the other.

**SP-6 is the hole nobody covers — 1 of 5.** "An empty input produces a full,
confident result" is usually the absence of a validation that was never
written, and no pass over source can flag code that does not exist. A forecast
built from an empty trajectory, a truth map of no tokens, a verdict on a
measurement that never completed: those needed a human or a test. If you want
to beat this benchmark, that is where the room is.

We publish our own 45% because a benchmark whose author scores 100% on it is a
benchmark that was fitted, not measured.

## provenance and bias, stated

Every case comes from **one codebase** — styxx itself — found during three
adversarial audit waves in August 2026. That is the corpus's central limitation
and we are not going to bury it:

- **Single-project bias.** These are the shapes one Python measurement library
  produced. Other stacks will have shapes we never saw.
- **Discovery bias.** Cases were found by humans and by our own detectors, so
  the corpus over-represents what those methods find. A defect nobody could
  detect is, by construction, absent.
- **Author bias.** We wrote the code, the fixes, the labels, and two of the
  detectors scored against it. Our numbers on our own corpus should be read
  with that in mind, which is why the scorer prints this paragraph's summary
  alongside every result.

The correct response to all three is contribution: cases from other codebases,
in the same schema, are the thing that turns this from a receipt into a
benchmark. `CONTRIBUTING.md` in this directory has the format.

## what a case looks like

```json
{
  "id": "SP-2026-0007",
  "subtype": "SP-1",
  "module": "styxx/gate.py",
  "fix_commit": "b2b716b",
  "defect_line": 432,
  "what_failed": "an invalid API key raised inside the vendor call",
  "what_was_returned": "GateVerdict(trust_score=1.0, will_refuse=0.0)",
  "why_it_reads_healthy": "1.0 is the maximum of the trust range",
  "consumer": "docs route callers to threshold on verdict.trust_score",
  "fix": "fall back to the text heuristic; neutral 0.5 if that also fails"
}
```

The pre-fix source is not copied into the JSON — it is fetched from
`fix_commit~1` so the corpus cannot drift from the history it claims.
