"""A trailing blank line does not earn the short-needle exemption.

The exemption is earned when a line anchor selects less than the whole receipt. "The whole receipt"
was computed with the module's own line convention, in which a receipt ending `...\\n\\n` has TWO
lines, the second empty. `#L1` therefore differed from the full-range slice -- by one EMPTY line --
and registered as narrowing, so the author earned the exemption while selecting essentially the
entire receipt:

    one-line 9020-byte blob ending 'overall status: FAIL'
      no trailing newline    #L1 -> MALFORMED short_needle
      one trailing newline   #L1 -> MALFORMED short_needle
      trailing BLANK line    #L1 -> HELD   (needle 4B, haystack 9020B, 600 occurrences)

"The run came back `PASS`." HELD against a receipt whose status is FAIL. One blank line at the end
of a captured log is the whole attack, and blank-terminated logs are ordinary.

This is a defect in a repair of my own: PR #73 added the narrowing clause this morning, and the
adversarial audit found the hole in it the same day. Recorded here rather than quietly patched,
because the failure was not in the code but in the test: #73 was tested against the case it was
written for, and never against `what would have to be true for this to be wrong`.

Spec: papers/sworn/SPEC_trailing_blank_lines_do_not_narrow_v01_2026_09_06.md (B1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx import sworn  # noqa: E402

# one long line: says PASS six hundred times and FAIL once, at the end
BLOB = (b"subtest a PASS " * 600) + b"overall status: FAIL"
MULTI = b"first line here\nsecond line here\nthird line here\n"


def _span(receipt: bytes, anchor: str, needle: str = "PASS"):
    m = sworn.Manifest(harness="ci", turn="t1", rung="L2")
    m.add("r1", receipt, "tool_stdout", complete=True)
    doc = ('<sworn r="r1%s" k="quote">The run came back `%s`.</sworn>\n'
           % (anchor, needle)).encode("utf-8")
    core = sworn.verify(doc, name="D.md", manifest=m, tree=None)
    return core, core["spans"][0]


def test_a_trailing_blank_line_does_not_buy_the_exemption():
    """B-G1, the guard that must be seen red."""
    core, span = _span(BLOB + b"\n\n", "#L1")
    assert span["verdict"] == "MALFORMED", (
        "a 4-byte needle was %s over a %d-byte haystack because the receipt ended with a blank "
        "line; the sentence claims PASS and the receipt says FAIL"
        % (span["verdict"], (span.get("detail") or {}).get("haystack_bytes") or 0))
    assert span["reason"] == "short_needle", span
    assert core["document_verdict"] == "SWORN-FAILED"


@pytest.mark.parametrize("tail", [b"", b"\n"], ids=["no-newline", "one-newline"])
def test_the_same_blob_without_a_blank_line_was_always_refused(tail):
    """B-G2, the control: it localises the cause to the blank line and nothing else."""
    _core, span = _span(BLOB + tail, "#L1")
    assert span["verdict"] == "MALFORMED" and span["reason"] == "short_needle", span


def test_a_real_anchor_over_a_multi_line_receipt_keeps_its_exemption():
    """B-G3. The repair must not take the exemption from an anchor that genuinely narrows."""
    _core, span = _span(MULTI, "#L2")
    assert span["verdict"] != "MALFORMED", span
    _core2, span2 = _span(MULTI, "#L2", needle="second")
    assert span2["verdict"] == "HELD", span2


def test_selecting_every_content_line_narrows_nothing():
    """B-G4. `#L1-L3` over a blank-terminated three-line receipt selects all of its content."""
    _core, span = _span(MULTI + b"\n", "#L1-L3")
    assert span["verdict"] == "MALFORMED" and span["reason"] == "short_needle", (
        "selecting every content line is not narrowing, even when a blank line follows: %s" % span)


def test_a_receipt_below_the_floor_keeps_its_exemption():
    """B-G5. The documented tiny-fixture decision, untouched by this repair."""
    _core, span = _span(b"ok\nyes\n", "#L1", needle="ok")
    assert span["verdict"] != "MALFORMED", (
        "a receipt below SHORT_NEEDLE_BYTES cannot be the danger the floor targets; the prior "
        "decision keeps its exemption: %s" % span)


def test_the_javascript_verifier_agrees():
    """B-G6, the parity gate, by core digest through the differential harness's node runner."""
    from shutil import which
    if which("node") is None:
        pytest.skip("node is not available")
    try:
        import conformance.sworn.differential as D
    except Exception as exc:                                     # noqa: BLE001
        pytest.skip("the differential harness is not importable here: %s" % exc)

    cases = [(BLOB + b"\n\n", "#L1"), (BLOB, "#L1"), (BLOB + b"\n", "#L1"),
             (MULTI, "#L2"), (MULTI + b"\n", "#L1-L3"), (b"ok\nyes\n", "#L1")]
    batch = []
    for i, (receipt, anchor) in enumerate(cases):
        m = sworn.Manifest(harness="ci", turn="t1", rung="L2")
        m.add("r1", receipt, "tool_stdout", complete=True)
        doc = ('<sworn r="r1%s" k="quote">The run came back `PASS`.</sworn>\n' % anchor).encode()
        batch.append({"index": i, "document": doc, "manifest": m.to_dict(),
                      "name": "D.md", "commit": None})

    import tempfile
    rows = D.js_digests(batch, Path(tempfile.mkdtemp()))
    bad = []
    for i, (_r, anchor) in enumerate(cases):
        py, py_err, _c = D.python_digest(batch[i])
        js = rows.get(i) or {}
        if py != js.get("digest"):
            bad.append("case %d (%s) python=%s(%s) node=%s(%s)"
                       % (i, anchor, (py or "-")[:12], py_err or "ok",
                          (js.get("digest") or "-")[:12], js.get("error") or "ok"))
    assert not bad, "the two implementations disagree:\n  " + "\n  ".join(bad)
