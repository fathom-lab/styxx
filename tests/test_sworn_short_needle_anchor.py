"""The short-needle exemption must be earned by narrowing, not by naming a line range.

THE DEFECT THIS PINS. v0.2 R3 refuses a `quote` needle under SHORT_NEEDLE_BYTES over a whole receipt
— two bytes HELD against almost anything, and an oath that cannot fail is not an oath. A line slice
is exempt, and the code says why: "the author narrowed the haystack by naming it, and the comparison
is against that alone."

The exemption was keyed on whether a slice is PRESENT (`res["slice"] is None`), not on whether it
NARROWS. So against a three-line receipt, a two-byte needle:

    r1          ->  MALFORMED short_needle     the floor, correctly
    r1#L1       ->  HELD                       one line of three: narrowed, exempt, correct
    r1#L1-L3    ->  HELD                       every line: the whole receipt, nothing narrowed,
                                               floor gone

Spec: papers/sworn/SPEC_short_needle_anchor_v01_2026_09_06.md, frozen before the repair. This file
exists before the repair too (N4): it fails against the shipped verifier on the whole-receipt slice
and passes everywhere else, then passes on all of them after. The one-line case is the control — a
repair that dropped the exemption entirely would fail it, and that would be worse than the defect.
"""
from __future__ import annotations

import base64
import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx import sworn  # noqa: E402

# Three lines, 53 bytes, trailing newline. The needle `ok` is two bytes and appears once.
PAYLOAD = b"line one: status ok\nline two: loss 0\nline three: end\n"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
# The same bytes with NO trailing newline: the trap N1 names, where a line count says "3 lines"
# both ways and only a byte comparison tells the slice from the whole.
PAYLOAD_NO_NL = PAYLOAD.rstrip(b"\n")
DIGEST_NO_NL = hashlib.sha256(PAYLOAD_NO_NL).hexdigest()


def _manifest():
    def entry(rid, payload, digest):
        return {"id": rid, "sha256": digest, "kind_of_source": "tool_stdout",
                "captured_at": "2026-09-01T00:00:00Z", "complete": True,
                "bytes": base64.b64encode(payload).decode("ascii")}
    return sworn.Manifest.from_dict({
        "spec": "sworn/manifest/0.2", "harness": "ci", "turn": "t",
        "minted_at": "2026-09-01T00:00:00Z", "rung": "L2", "authored_sha256": [],
        "receipts": {"r1": entry("r1", PAYLOAD, DIGEST),
                     "r2": entry("r2", PAYLOAD_NO_NL, DIGEST_NO_NL),
                     "r3": entry("r3", TINY, hashlib.sha256(TINY).hexdigest()),
                     "r4": entry("r4", ONE_LONG_LINE, hashlib.sha256(ONE_LONG_LINE).hexdigest())},
    })


# The boundary the second erratum draws. TINY is below the floor: the lab's prior, documented
# decision says #L1 over it keeps the exemption, because two bytes over nine cannot be the danger
# the floor targets. ONE_LONG_LINE is a single line AT the floor's size and beyond: #L1 over it is
# the whole receipt, narrows nothing, and is the floor's business.
TINY = b"ok 0.95\n"                                       # 8 bytes, one line
ONE_LONG_LINE = b"status ok; loss 0; reserve 5; end of record\n"   # 44 bytes, one line
assert len(TINY) < sworn.SHORT_NEEDLE_BYTES <= len(ONE_LONG_LINE)


def _span(receipt, sentence="the log says `ok` at the top."):
    doc = ('<sworn r="%s" k="quote">%s</sworn>\n' % (receipt, sentence)).encode("utf-8")
    core = sworn.verify(doc, name="d.md", manifest=_manifest(), commit=None)
    return core["spans"][0]


assert len(b"ok") < sworn.SHORT_NEEDLE_BYTES, "the needle must be under the floor for any of this to mean anything"


# ---------------------------------------------------------------- the floor, unanchored

def test_a_short_needle_over_the_whole_receipt_is_refused():
    s = _span("r1")
    assert (s["verdict"], s["reason"]) == ("MALFORMED", "short_needle"), s


# ---------------------------------------------------------------- N1: the defect

def test_a_slice_covering_every_line_does_not_exempt_the_floor():
    """#L1-L3 over a three-line receipt is the whole receipt. Nothing was narrowed."""
    s = _span("r1#L1-L3")
    assert (s["verdict"], s["reason"]) == ("MALFORMED", "short_needle"), (
        "a line range spanning the entire receipt bypassed the short-needle floor: %r" % s)


def test_the_comparison_is_on_bytes_not_line_counts():
    """r2 has no trailing newline. By line COUNT, #L1-L3 is 'all three lines' either way; only a
    byte comparison sees that the slice equals the receipt. This is the off-by-one N1 names."""
    s = _span("r2#L1-L3")
    assert (s["verdict"], s["reason"]) == ("MALFORMED", "short_needle"), s


# ---------------------------------------------------------------- N2: narrowing still earns it

def test_a_slice_that_genuinely_narrows_keeps_the_exemption():
    """The case the exemption exists for. A repair that refused this would be worse than the
    defect."""
    s = _span("r1#L1")
    assert s["verdict"] == "HELD", s


def test_two_of_three_lines_still_narrows():
    s = _span("r1#L1-L2")
    assert s["verdict"] == "HELD", s


def test_the_needle_must_still_be_present_in_the_narrowed_slice():
    """Narrowing earns the exemption from the FLOOR, not from the comparison."""
    s = _span("r1#L3")                    # `ok` is on line one, not line three
    assert (s["verdict"], s["reason"]) == ("FAILED", "needle_missing"), s


# ---------------------------------------------------------------- N3: pointer leaves untouched

def test_a_pointer_leaf_is_still_exempt():
    payload = b'{"status": "ok"}'
    man = sworn.Manifest.from_dict({
        "spec": "sworn/manifest/0.2", "harness": "ci", "turn": "t",
        "minted_at": "2026-09-01T00:00:00Z", "rung": "L2", "authored_sha256": [],
        "receipts": {"r1": {"id": "r1", "sha256": hashlib.sha256(payload).hexdigest(),
                            "kind_of_source": "tool_stdout", "captured_at": "2026-09-01T00:00:00Z",
                            "complete": True, "bytes": base64.b64encode(payload).decode("ascii")}},
    })
    doc = b'<sworn r="r1#/status" k="quote">the status is `ok`.</sworn>\n'
    s = sworn.verify(doc, name="d.md", manifest=man, commit=None)["spans"][0]
    assert s["verdict"] == "HELD", s


# ---------------------------------------------------------------- a long needle needs no exemption

def test_a_needle_at_the_floor_holds_over_the_whole_receipt():
    s = _span("r1", "the log begins `line one: status ok` as expected.")
    assert s["verdict"] == "HELD", s


# ---------------------------------------------------------------- the boundary, both halves

def test_an_anchor_over_a_receipt_below_the_floor_keeps_the_exemption():
    """The lab's prior, documented decision, found in tests/test_sworn.py after the first repair
    broke it: "a nine-byte receipt cannot hold a sixteen-byte needle; the author narrows the
    haystack with a line anchor and the short needle is then exempt." Two bytes over eight do not
    hold against almost anything; the floor's danger is proportional to the haystack and cannot
    exist below it. #L1 here narrows nothing and is exempt anyway, on purpose."""
    s = _span("r3#L1", "the receipt says `ok` and nothing else.")
    assert s["verdict"] == "HELD", (
        "the sub-floor idiom the source tests depend on was refused: %r" % s)


def test_a_bare_receipt_below_the_floor_is_still_refused_without_an_anchor():
    """The prior decision is about an ANCHOR over a tiny receipt, not about tiny receipts. Bare
    receipts are untouched by this leg, and a repair that quietly widened the exemption to every
    small receipt would pass the test above and fail this one."""
    s = _span("r3", "the receipt says `ok` and nothing else.")
    assert (s["verdict"], s["reason"]) == ("MALFORMED", "short_needle"), s


def test_an_anchor_over_a_one_line_receipt_at_the_floor_is_the_floors_business():
    """The strict half, kept for receipts at or above the floor. A one-line receipt of forty-four
    bytes is its own whole; #L1 over it narrows nothing, and a 10 KB minified blob is the same
    shape at a size where two bytes hold against almost anything."""
    s = _span("r4#L1", "the record says `ok` in it.")
    assert (s["verdict"], s["reason"]) == ("MALFORMED", "short_needle"), (
        "#L1 over a single line at the floor's size bypassed the floor: %r" % s)


def test_a_long_needle_over_the_one_line_receipt_still_holds():
    """And the same receipt is fine once the needle carries enough bytes to mean something."""
    s = _span("r4#L1", "the record reads `status ok; loss 0; reserve 5` verbatim.")
    assert s["verdict"] == "HELD", s
