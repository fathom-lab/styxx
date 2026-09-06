# FINDING — the two-implementation agreement is bounded by Unicode version parity

*2026-09-06. Not a defect in either implementation. A condition on a claim this project makes
prominently, which had never been written down.*

## How it was found

`SPEC_path_segment_class_is_pinned_v01_2026_09_06` closed a divergence where `sworn_verify.js`
hand-expanded Python's `\s` and dropped `U+0085`. That spec ended by naming what it had not done:

> `TOKEN_RE`, `GRAM_RE`, `HEXRUN_RE` and `DIGIT_RE` are hand-mirrored the same way and have **never
> been enumerated** against their Python counterparts. That is the obvious next measurement, and the
> method now exists to make it.

This is that measurement. `conformance/sworn/class_census.py` enumerates every character class both
verifiers implement, over all 1,114,112 scalar values, lifting the JavaScript regexes out of the
shipped file rather than restating them.

## The result

```
python unicodedata 15.0.0   node unicode/icu 16.0

_TOKEN / TOKEN_RE                   property DIFFER  python=137971 node=142975   (5004 node-only)
_DIGIT / DIGIT_RE                   property DIFFER  python=680    node=760      (80 node-only)
_HEXRUN / HEXRUN_RE                 explicit AGREE   python=22     node=22
_PATH_SEG_BAD / PATH_SEG_BAD_RE     explicit AGREE   python=58     node=58
_DIRECTIONAL_OVERRIDE / ...         explicit AGREE   python=2      node=2
```

**Every class defined by a Unicode property diverges. Every class defined by an explicit list
agrees.** The split is exact and it explains itself: CPython here ships Unicode 15.0.0 and V8's ICU
ships 16.0, so both sides correctly implement "category Nd" and "word character" against different
editions of the standard.

It changes verdicts. `U+10D40` is a Garay digit, added in Unicode 16.0:

```
U+10D40  python unicodedata category: Cn (unassigned to this build)
python _DIGIT matches it : False

document: <sworn r="r1" k="numeric">the count was <U+10D40 x3> items</sworn>
PYTHON verdict: SWORN-FAILED  [('MALFORMED','number_count')]
core digests -> python 6ecf74684afd528e   node 35b7ceafd77b6471    *** DIVERGE ***
```

## What it bounds

The project states, prominently and truly, that a second independent implementation agrees on
**1689 of 1689** conformance vectors. That claim is true **because no vector uses a code point the
two runtimes classify differently** — measured: **0 of 3981 blobs do**.

So the agreement is real and it is conditional. The condition is that both runtimes share a Unicode
version, and it was nowhere stated. A reader who takes "1689/1689" as "these two implementations
agree" is reading it more broadly than the evidence supports; the accurate reading is "these two
implementations agree on this corpus, at these two Unicode versions."

## What is NOT proposed

**Hardcoding the property classes.** `\w` has 137,971 members here; writing it out on both sides
would be a large, version-pinned table that must be regenerated whenever either runtime moves, and
would replace a stated boundary with a maintenance burden that fails silently when neglected.

The explicit classes are explicit because they are small and adversarially interesting —
`_PATH_SEG_BAD` (58), `_DIRECTIONAL_OVERRIDE` (2), `_HEXRUN` (22). That was the right call for
those and is the wrong call for `\w`.

## What is shipped

1. `conformance/sworn/class_census.py`, so the measurement is repeatable rather than a number in a
   document. It **fails** when an *explicit* class differs — that is a hand-written defect, as
   `U+0085` was — and **reports** when a *property* class differs, naming both Unicode versions.
2. A guard that the explicit classes agree, and a guard that **no conformance blob uses a code point
   the two runtimes classify differently**. The second is the precise statement of why the 1689 bar
   holds; if a future vector reaches into the skew, it fails and forces the question rather than
   quietly making the bar mean something narrower.

## What is left for the operator

Whether the published claim should carry the condition. "A second implementation agrees on 1689 of
1689 vectors" appears in `sworn_verify.js`'s own header and in the RESULT that shipped it. The
honest form adds *at matching Unicode versions*, and amending shipped documents is not something to
do on the way past.

Also left: `GRAM_RE`, which the census does not cover. It is anchored and matches strings rather
than single characters, so comparing it needs a string generator, not a code-point sweep. That is a
different instrument and it has not been built.
