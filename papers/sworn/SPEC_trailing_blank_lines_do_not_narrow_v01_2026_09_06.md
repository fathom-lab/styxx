# SPEC — a trailing blank line does not earn the short-needle exemption (v0.1)

**Frozen 2026-09-06, before the code.** One rule, B1.

**This is a defect in a repair of my own, shipped earlier today.** PR #73 added the short-needle
narrowing clause; the audit (`wf_9466dcba-f49`, dimension `quote`) found a hole in it within hours.
The clause and this correction are both mine, and that is the point worth recording: the repair was
tested against the case it was written for and not against the question *what would have to be true
for this to be wrong.*

## The defect

The exemption is earned when a line anchor selects less than the whole receipt. "The whole receipt"
was computed with the module's own line convention:

    n_lines = whole.count(b"\n") + (1 if whole and not whole.endswith(b"\n") else 0)
    narrowed = (res["slice"] != _line_slice(whole, 1, n_lines) or len(whole) < SHORT_NEEDLE_BYTES)

A receipt ending `…\n\n` has **two** lines by that convention, the second empty. `#L1` therefore
differs from the full-range slice — by one empty line — and registers as narrowing. The author earns
the exemption while selecting essentially the entire receipt.

Measured at `4e6f10de`, over a one-line 9020-byte blob ending `overall status: FAIL`:

```
  no trailing newline    #L1 -> MALFORMED short_needle   doc=SWORN-FAILED
  one trailing newline   #L1 -> MALFORMED short_needle   doc=SWORN-FAILED
  trailing BLANK line    #L1 -> HELD                     doc=SWORN-HELD
                                needle_bytes 4, haystack_bytes 9020, occurrences 600
```

The sentence *"The run came back `PASS`."* is HELD against a receipt whose status is FAIL, because
`PASS` occurs 600 times inside it. **One blank line at the end of a captured log is the whole
attack**, and blank-terminated logs are ordinary.

`sworn_verify.js` mirrors the clause exactly, comment for comment, so both implementations carry the
same hole and the differential harness sees agreement.

## B1 — "the whole receipt" means its content lines

The full-range comparison is computed over the receipt with trailing newlines removed:

    core = whole.rstrip(b"\n")
    n_content = core.count(b"\n") + (1 if core else 0)

and `narrowed` is false when the author's slice equals `_line_slice(whole, 1, n_content)`.

A receipt with **no** content lines (empty, or newlines only) narrows nothing, so `narrowed` is
false there except through the below-the-floor clause, which is unchanged.

This is a deliberate divergence from `_line_slice`'s own line convention, and the reason is that the
two are answering different questions. `_line_slice` is asked *how many lines does this file have*,
where a trailing empty line is a line. The exemption asks *did the anchor exclude any content*,
where it is not.

The below-the-floor clause (`len(whole) < SHORT_NEEDLE_BYTES`) is untouched: it exists for the
documented tiny-fixture decision and that decision still holds.

## What moves

- A `#L1` anchor over a receipt whose only extra line is blank stops being exempt: `MALFORMED
  short_needle` instead of adjudicated. That is the repair.
- Anchors over receipts with real trailing content are unaffected.
- Both implementations change in one commit; the parity gate decides it.
- Conformance moves by the build pin only unless a vector exercises a blank-terminated receipt with
  an anchor. If one does, that is a real verdict change, it is named, and it is not smoothed.

## Guards, watched to fail before the code

| # | guard | before | after |
|---|---|---|---|
| B-G1 | a 4-byte needle under `#L1` over a blank-terminated one-line blob is not HELD | red: HELD, SWORN-HELD | green: MALFORMED `short_needle` |
| B-G2 | the same blob with no trailing newline, and with exactly one, is MALFORMED | green throughout — the control that localises the cause |
| B-G3 | a genuine `#L1` over a multi-line receipt keeps its exemption | green throughout — catches over-reach |
| B-G4 | selecting every content line of a blank-terminated receipt (`#L1-Ln`) narrows nothing | red | green |
| B-G5 | a receipt below the floor keeps its exemption (the documented tiny-fixture decision) | green throughout |
| B-G6 | Python and the JS verifier agree by core digest across all of the above | green with both wrong, red with one side fixed |

B-G1 is the guard that must be seen red.
