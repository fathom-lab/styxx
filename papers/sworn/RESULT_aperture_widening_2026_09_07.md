# RESULT — the differential generator can now reach the characters that hid four defects

*2026-09-07. A repair to an instrument, and a negative result from it.*

## What was wrong with the number

`RESULT_differential_agreement_2026_09_05.md` and the run beneath it report that two independent
verifiers agree on **100000 of 100000** generated documents. On 2026-09-06 that number was paired
with its aperture for the first time:

```
distinct code points the generator could reach: 98
non-ASCII (7): U+00E9, U+0301, U+0663, U+0665, U+2212, U+FEFF, U+1F600
```

Against the same day's audit findings:

| defect | code point | reachable then |
| --- | --- | --- |
| a dash that dropped a minus sign, `-0.42` HELD against `0.42` | `U+2010` and 25 others | no |
| the override that shows a reader `55.0` where `0.55` was checked | `U+202E` | no |
| the path segment that split the two verifiers on a document verdict | `U+0085` | no |
| the Unicode-version digit skew | `U+10D40` | no |

The agreement was real and it covered a 98-character alphabet. Four defects lived outside it.

## The repair

`APERTURE_ALPHABET` — 68 code points across dashes, format and bidi controls, unusual whitespace,
combining marks and non-ASCII digits — is spliced into span inner text (p=0.22) and receipt targets
(p=0.12).

```
generator alphabet: 98 -> 142 code points   (non-ASCII 7 -> 69)

  U+2010  reachable: True
  U+202E  reachable: True
  U+0085  reachable: True
  U+10D40 reachable: False      <- deliberately, see below
```

### Why one of them stays out of reach

`U+10D40` is a Garay digit: category `Nd` in Unicode 16.0, unassigned in 15.0.0. CPython here is on
15.0.0 and V8's ICU on 16.0, so the two runtimes classify it differently and **no edit to either
implementation can make them agree**. A generator that emitted it would report a disagreement that
is real, unfixable, and not about the verifiers — turning a defect detector into a runtime detector,
red on this machine and green on one whose runtimes match.

So the aperture widens only where the two runtimes already agree. Every one of the 68 was checked
against `conformance/sworn/class_census.py`, and a guard re-checks it rather than trusting this
paragraph.

## The result from the widened instrument

```
seed 20260905, 100000 cases
compared 100000 | agree 100000 | disagree 0 | one-sided errors 0 | reasons 39
```

Thirty-nine distinct reason codes reached, up from thirty-eight. **No disagreement.** The six
repairs of 2026-09-06 hold across an alphabet 44 code points wider than the one that missed them.

That is a negative result and it is worth what a negative result is worth: the space where four
defects hid has now been searched, by an instrument that can demonstrably see into it.

## The instrument lied first, and the lie was mine

The first version of the probe reported **1577 disagreements in 2000 cases**. It was false.

Case 0 was `<sworn r="" k="quote">held ` + "`PASS`" + `</sworn>` — plain ASCII, no injected
character — which is not a document two verifiers should disagree about. The cause was the probe's
own hand-rolled manifest, which used a `bytes_b64` key the format has no such field for. Re-running
the identical documents with `manifest=None` produced exact agreement, which located the fault in
one step.

The repair was to delete the hand-rolled manifest and call the harness's own `_manifest`, removing
the probe as a variable so that a disagreement could only be about the document alphabet — the one
thing being widened.

It is recorded here because a spectacular finding that survives ten minutes of checking is the
failure mode this corpus exists to make expensive, and this one was the author's.

## What is pinned

`tests/test_differential_aperture.py`:

- the three defect-carrying code points are **reachable** — a regression of any of those repairs can
  now be generated, so an agreement number covers them;
- `U+10D40` is **not** reachable, and the test says why, so a later widening cannot make the harness
  unfixably red by accident;
- the alphabet stays materially wider than the 98/7 that made the miss list possible;
- every member of `APERTURE_ALPHABET` is checked against the census, so adding a runtime-skewed
  character fails before it can produce a false disagreement.

Watched to fail: with the injection disabled, all three reach-guards and the width guard go red,
and the two safety guards stay green.

## What this does not claim

That the two implementations agree. It claims they agree on **142 code points** now instead of 98,
and the number still travels with its alphabet. Beyond that alphabet nothing has been searched, and
the region the runtimes disagree about is excluded by construction rather than by evidence.

The conformance set is untouched: `differential.py` is not among `gen_vectors.py`'s `SOURCES`, so no
vector, blob or digest moves. The previous differential run stands as history — it describes the
generator it was run with, and that generator is not this one.
