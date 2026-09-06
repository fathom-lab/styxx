# RESULT — an exemption for authors who point at less was open to authors who point at everything

Fathom Lab · 2026-09-06 · Spec: `SPEC_short_needle_anchor_v01_2026_09_06.md`, frozen before the
repair, with two errata appended — one after the first repair failed, one after the second met a
decision this lab had already written down. Receipts: `short_needle_repair.json` and
`short_needle_repair_2.json`, both re-derived from git by `repair_receipt.py`. **This document is
itself sworn.**

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

## Two repairs, and what each one got wrong

**The first repair compared byte lengths**, as N1 of the frozen spec prescribed to dodge a
trailing-newline off-by-one. The guard still failed on the newline-terminated receipt — because
`_line_slice` **excludes the last selected line's terminating LF by design**, so a full-range slice
of a 53-byte receipt is 52 bytes, and a length test calls that narrowed. N1 named the right hazard
and prescribed the thing it bites. The first erratum withdraws that remedy: a slice narrows if and
only if it differs from `_line_slice(receipt, 1, n_lines)` — what *everything* looks like under the
same function.

**The second repair enforced that strictly, and the conformance generator refused it.** Three tests
in the vector sources broke, and the generator would not regenerate over sources that no longer
passed under the recorder. All three use `#L1` over a nine-byte receipt with a short needle and
expect HELD, and one says why in its own comment: *"a nine-byte receipt cannot hold a sixteen-byte
needle; the author narrows the haystack with a line anchor and the short needle is then exempt."*
That is a documented prior decision, and it is right on the floor's own terms — two bytes over nine
do not hold against almost anything. The spec had taken the opposite position without having read
it. The second erratum narrows the rule rather than withdrawing it: a line slice earns the exemption
when it narrows, **or** when the receipt it was cut from is itself shorter than the floor, where the
floor's danger cannot exist. `#L1` over nine bytes stays HELD. `#L1` over a one-line 44-byte record
is the floor's business. `#L1-L3` over 53 bytes is refused. Bare receipts are untouched, and the
three source tests are not changed.

## How it was checked, in two receipts

The first receipt measures the finding. The guard as first written had
<sworn r="path:papers/sworn/short_needle_repair.json#/guard_tests" k="numeric">8 tests;</sworn>
<sworn r="path:papers/sworn/short_needle_repair.json#/before/failed" k="numeric">2 failed against the shipped verifier</sworn>
— the whole-receipt slice, terminated and unterminated — and
<sworn r="path:papers/sworn/short_needle_repair.json#/after/failed" k="numeric">0 failed after the strict repair.</sworn>

The second receipt measures the correction. The guard grew to
<sworn r="path:papers/sworn/short_needle_repair_2.json#/guard_tests" k="numeric">12 tests</sworn>
to pin both halves of the boundary;
<sworn r="path:papers/sworn/short_needle_repair_2.json#/before/failed" k="numeric">1 failed against the strict repair</sworn>
— the sub-floor idiom the source tests depend on — and
<sworn r="path:papers/sworn/short_needle_repair_2.json#/after/failed" k="numeric">0 failed after the sub-floor clause.</sworn>

The tests that pass in every state are the controls that give the repair its shape: the bare floor
still refuses, one line of three and two of three still earn the exemption, a needle absent from the
named line still FAILs rather than being excused, a pointer leaf is untouched, a needle at the floor
needs no exemption — and a bare tiny receipt with no anchor is still refused, so a repair that
quietly exempted every small receipt would pass the idiom test and fail that one.

Neither receipt is remembered: `repair_receipt.py` reads **both** verifiers at the repair commit
and at its parent from git and runs the guard against each in a scratch copy, refusing if the guard
ran no tests. The frozen 1689-vector bar holds. The standing differential guard passes with both
sides changed together. The conformance set regenerated with no moved core.

## What this does not say

**That sixteen bytes is the right floor.** It is the line the lab drew, and this leg enforces it
where it was not enforced; it does not defend the number.

**That the pointer-leaf exemption is sound.** It is out of scope and unexamined.

**That the adversary's list is closed.** `commit: null` was probed alongside and did **not**
reproduce as described — with an explicit tree handle it resolves; only the CLI path without
`--commit` goes blind, which is the UNRESOLVED finding already warned about. Recorded as
not-reproduced rather than dropped.

**That the spec was right.** It was wrong twice, once about the remedy and once about the boundary,
and both are in it as errata rather than edits. The second was wrong because I had not read a
comment in the tests I was about to break, which is the cheaper of the two mistakes and the one I
would rather have avoided.

---

*The check asked whether the author had pointed, and not at what. The first repair measured with a
ruler the slicer disagreed with. The second forgot that someone had already decided what a tiny
receipt deserves — and had written it down, in the one place a test failure would make me read.*
