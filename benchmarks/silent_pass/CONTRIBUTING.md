# contributing a SILENT-PASS case

The corpus is currently **one codebase deep**. That is its biggest weakness, and
cases from other projects are worth more to it than anything we can add.

## what qualifies

All four must hold:

1. a measurement path failed or was skipped
2. a value was still returned — no raise, no None, no error field
3. that value reads as **healthy** — the best or a benign point in its range
4. something acted on it: a gate, a calibrator, a trainer, a report, a person

If it crashed, it is not this class. If nobody reads it, it is a latent bug.

## the format

Add one object to `cases.json`:

```json
{
  "id": "SP-2026-00NN",
  "subtype": "SP-1",
  "module": "path/to/module.py",
  "fix_commit": "abc1234",
  "defect_line": 432,
  "what_failed": "what stopped working, concretely",
  "what_was_returned": "the literal value, with its type",
  "why_it_reads_healthy": "why that value looks fine to a reader",
  "consumer": "who acted on it",
  "fix": "what changed"
}
```

`fix_commit` must exist in the repository the corpus is scored against, because
the pre-fix source is fetched from `fix_commit~1` rather than pasted in. That is
deliberate: a corpus that carries its own snapshots can drift from the history
it claims, and then it is documentation, not evidence.

For a case from another repository, include a `repo` field with a clone URL and
the scorer will fetch against it.

## what we will not accept

- **synthetic cases.** Hand-written examples of the shape are useful for unit
  tests and useless as a benchmark. Every case must have actually shipped.
- **cases without a fix.** If it was not fixed, the label is a claim rather than
  a confirmed defect.
- **cases where the value was plausibly correct.** Plenty of zeros are real
  zeros. If a reasonable engineer would defend the default, it is not this.

## on scoring your own detector

`score()` measures recall and nothing else — the corpus holds only true
positives, so flagging every line scores 1.00. Publish a precision estimate
beside any recall number, measured by hand on unlabelled code. We publish ours
(5 of 14) in `papers/CENSUS_absence_2026_08_19.md`, and our detectors score
45% and 25% here.
