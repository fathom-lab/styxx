# SPEC — the short-needle exemption must be earned by narrowing, not by naming

Fathom Lab · 2026-09-06 · **A spec, not a result.** Frozen in its own commit before the repair is
made, so the repair cannot be scored by whether it made the probe go green.

## The rule, and its stated justification

v0.2 R3: a `quote` needle shorter than `SHORT_NEEDLE_BYTES` (16) over a whole receipt is
`MALFORMED / short_needle`, because two bytes HELD against almost anything and an oath that cannot
fail is not an oath. The code exempts two shapes from that floor, and says why:

> *A pointer leaf (above) and a line slice are exempt — the author narrowed the haystack by naming
> it, and the comparison is against that alone.*

The justification is **narrowing**. A line slice is exempt because the author pointed at less than
the whole receipt, so a short needle over that smaller haystack means something.

## Where the justification is not enforced

The exemption is keyed on `res["slice"] is None` — on whether a line anchor is *present*, not on
whether it *narrows*. Confirmed by execution, a two-byte needle `` `ok` `` against a three-line,
53-byte receipt:

| receipt | verdict |
|---|---|
| `r1` (whole receipt) | MALFORMED `short_needle` |
| `r1#L1` (one line) | HELD |
| **`r1#L1-L3` (every line)** | **HELD** |
| `r1#L1-L400` | MALFORMED `anchor_out_of_range` |

`#L1-L3` over a three-line receipt is the whole receipt. Nothing was narrowed, and the floor is
gone. The sidecar battery's adversary named this with the example `#L1-L400`; that example is wrong —
it is refused as out of range — but the claim underneath it holds with a range that exactly spans
the file, which is the shape an author trying this would reach for.

## What the repair must satisfy

**N1 — the exemption is earned by narrowing.** A line slice exempts the floor only when the slice is
strictly shorter than the receipt it was cut from. A slice equal to the whole receipt is the whole
receipt, and the floor applies.
*Attack:* keying the check on line count rather than bytes, so a file whose last line has no
trailing newline is "narrowed" by an off-by-one. *Answer:* the comparison is on the bytes of the
slice against the bytes of the receipt.

**N2 — a slice that genuinely narrows keeps its exemption.** `#L1` over a multi-line receipt is the
case the exemption exists for and must still HELD. A repair that dropped the exemption entirely
would refuse every short quote over a named line, which is the behaviour the rule was written to
permit.

**N3 — the pointer-leaf exemption is untouched.** A `#/pointer` names a leaf, which is always
narrower than the receipt; nothing here changes it. The reason string is the existing
`short_needle`; no new vocabulary.

**N4 — the guard is watched to fail.** It exists before the repair, fails against the shipped
verifier on the whole-receipt slice, passes on the one-line slice and on the whole-receipt
non-anchored case, and passes on all of them after.

**N5 — the JavaScript verifier is held to the same rule.** `sworn_verify.js` implements the same
exemption (`res.slice !== null`), so this is a change on both sides or it is a disagreement the
differential harness will find. Both change in the same commit, and the standing guard
`tests/test_differential_agreement.py` runs at the new behaviour. Any conformance vector whose core
moves is examined and named, not regenerated over.

## What this spec does not say

That 16 bytes is the right floor, or that a single line is always a meaningful narrowing — a
one-line receipt is its own whole. That the pointer-leaf exemption is sound; it is out of scope
here and unexamined. That this closes the adversary's list: `commit: null` was probed alongside and
did **not** reproduce as described — with an explicit tree handle it resolves; only the CLI path
without `--commit` goes blind, which is the UNRESOLVED finding already warned about in #71.

---

*An exemption written for authors who point at less was available to authors who point at
everything, because the check asked whether they pointed and not at what.*
