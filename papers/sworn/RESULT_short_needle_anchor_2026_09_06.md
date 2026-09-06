# RESULT — an exemption for authors who point at less was open to authors who point at everything

Fathom Lab · 2026-09-06 · Spec: `SPEC_short_needle_anchor_v01_2026_09_06.md`, frozen before the
repair, with an erratum appended after the first repair failed. Receipt:
`short_needle_repair.json`, re-derived from git by `repair_receipt.py`. **This document is itself
sworn.**

## The rule and its justification

v0.2 R3: a `quote` needle shorter than sixteen bytes over a whole receipt is refused as
`short_needle`, because two bytes HELD against almost anything and an oath that cannot fail is not
an oath. A line slice is exempt, and the code says why: *"the author narrowed the haystack by naming
it, and the comparison is against that alone."*

## Where the justification was not enforced

The exemption was keyed on whether a slice was **present**, not on whether it **narrowed**.
Confirmed by execution — a two-byte needle `` `ok` `` against a three-line, 53-byte receipt:

| receipt | verdict |
|---|---|
| `r1` (whole receipt) | MALFORMED `short_needle` |
| `r1#L1` (one line) | HELD |
| **`r1#L1-L3` (every line)** | **HELD** |
| `r1#L1-L400` | MALFORMED `anchor_out_of_range` |

`#L1-L3` over a three-line receipt is the whole receipt with an anchor on it. Nothing narrowed;
floor gone. The sidecar battery's adversary named this with the example `#L1-L400` — which is wrong,
refused as out of range — but the claim underneath it holds with a range that exactly spans the
file, which is the shape an author trying it would reach for.

## The first repair was wrong, and the spec that prescribed it was wrong the same way

N1 of the frozen spec said the narrowing test must compare **bytes**, not line counts, so a
trailing-newline off-by-one could not fake a narrowing. The first repair did exactly that. The guard
still failed on the newline-terminated receipt — because `_line_slice` **excludes the last selected
line's terminating LF by design**, so a full-range slice of a 53-byte receipt is 52 bytes, and a
length test calls that narrowed. The hazard N1 feared was real; the remedy it prescribed was the
thing it bites.

An erratum is appended to the spec. The repair that holds asks the slicer itself: a slice narrows if
and only if it differs from `_line_slice(receipt, 1, n_lines)` — what *everything* looks like under
the same function. That is exact for both the terminated and the unterminated receipt, and the
JavaScript verifier asks `lineSlice` the same question, which N5 requires.

## How it was checked

The guard was written first and run against the verifier as shipped. Of
<sworn r="path:papers/sworn/short_needle_repair.json#/guard_tests" k="numeric">8 tests,</sworn>
<sworn r="path:papers/sworn/short_needle_repair.json#/before/failed" k="numeric">2 failed before the repair</sworn>
— the whole-receipt slice, terminated and unterminated — and
<sworn r="path:papers/sworn/short_needle_repair.json#/after/failed" k="numeric">0 failed after.</sworn>
The
<sworn r="path:papers/sworn/short_needle_repair.json#/before/passed" k="numeric">6 that passed in both states</sworn>
are the controls that give the repair its shape: the bare floor still refuses, one line of three and
two of three still earn the exemption, a needle absent from the named line still FAILs rather than
being excused, a pointer leaf is untouched, and a needle at the floor needs no exemption. A repair
that dropped the exemption outright — the tempting one-liner — would have failed three of them.

The numbers are re-derived, not remembered: `repair_receipt.py` reads **both** verifiers at the
repair commit and at its parent from git, runs the guard against each in a scratch copy, and refuses
if the guard ran no tests. A two-sided repair measured with one side old and one side new would be
measuring nothing.

The frozen 1689-vector bar holds. The standing differential guard passes with both sides changed
together. The conformance set regenerated with no moved core.

## What this does not say

**That sixteen bytes is the right floor**, or that one line is always a meaningful narrowing — a
one-line receipt is its own whole, and `#L1` over it is now the floor's business, as it should be.

**That the pointer-leaf exemption is sound.** It is out of scope here and unexamined.

**That the adversary's list is closed.** `commit: null` was probed alongside and did **not**
reproduce as described — with an explicit tree handle it resolves; only the CLI path without
`--commit` goes blind, which is the UNRESOLVED finding already warned about. Recorded as
not-reproduced rather than dropped.

---

*The check asked whether the author had pointed, and not at what. The first repair asked how much
they had pointed at, and the slicer's own definition of "everything" disagreed with a ruler.*
