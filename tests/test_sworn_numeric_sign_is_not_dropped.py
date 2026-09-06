"""A dash before a numeric span's digits never disappears.

`_TOKEN` admitted exactly two dash code points, U+002D and U+2212. Every other dash sat outside the
class, so it did not join the digits it preceded -- it split the token and vanished. The verifier
then adjudicated a number whose sign was gone: a sentence asserting -0.42 was HELD against a receipt
of 0.42, document SWORN-HELD, receipt VERIFIED. A reader sees a dash on the baseline; the verifier
reads a positive number.

Spec: papers/sworn/SPEC_numeric_sign_is_not_dropped_v01_2026_09_06.md (N1). The rule is that a
dash-like code point BINDS to the number it precedes, so the token fails the number grammar and the
span is MALFORMED `number_grammar`. The verifier declines rather than guessing which of 26 dashes
was meant as arithmetic negation -- guessing would manufacture a FAILED accusation out of a
typographic artifact.

Watched to fail: before the repair, every one of the 26 code points below was HELD.
"""
from __future__ import annotations

import json
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx import sworn  # noqa: E402

ACCEPTED = (0x002D, 0x2212)          # the two the format reads as a sign
DASHES = sorted((set(c for c in range(0x110000) if unicodedata.category(chr(c)) == "Pd")
                 | {0x00AD}) - set(ACCEPTED))

JS = ROOT / "styxx" / "_data" / "sworn_verify.js"


def _doc(sign_cp: int | None, number: str = "0.42") -> bytes:
    lead = "" if sign_cp is None else chr(sign_cp)
    return ('<sworn r="r1" k="numeric">the delta was %s%s on the panel</sworn>\n'
            % (lead, number)).encode("utf-8")


def _manifest(value: bytes = b"0.42"):
    """A scalar receipt: a numeric span over a whole JSON object is MALFORMED for a different
    reason (`leaf_type`), which would mask the one this file is about."""
    m = sworn.Manifest(harness="ci", turn="t1", rung="L2")
    m.add("r1", value, "tool_stdout", complete=True)
    return m


def _one(sign_cp, number="0.42", receipt=b"0.42"):
    core = sworn.verify(_doc(sign_cp, number), name="D.md", manifest=_manifest(receipt), tree=None)
    return core, core["spans"][0]


@pytest.mark.parametrize("cp", DASHES, ids=lambda c: "U+%04X" % c)
def test_a_dash_before_the_digits_is_never_dropped(cp):
    """G1. The sentence asserts a negative number; the receipt is positive. HELD is a false verdict."""
    core, span = _one(cp)
    assert span["verdict"] != "HELD", (
        "U+%04X before the digits was dropped: the sentence asserts %s0.42, the receipt is 0.42, and "
        "the span was HELD with printed_token %r"
        % (cp, chr(cp), (span.get("detail") or {}).get("printed_token")))
    assert core["document_verdict"] != "SWORN-HELD"


@pytest.mark.parametrize("cp", DASHES, ids=lambda c: "U+%04X" % c)
def test_the_dash_binds_to_the_number_and_the_verifier_declines(cp):
    """N1's positive half: MALFORMED number_grammar, not a guessed FAILED."""
    _core, span = _one(cp)
    assert span["verdict"] == "MALFORMED", span
    assert span["reason"] == "number_grammar", span


@pytest.mark.parametrize("cp", ACCEPTED, ids=lambda c: "U+%04X" % c)
def test_the_two_accepted_signs_still_adjudicate(cp):
    """G2. The repair must not swallow the signs the format does read."""
    _c, mismatched = _one(cp)                                   # asserts -0.42 against 0.42
    assert mismatched["verdict"] == "FAILED", mismatched
    _c2, matched = _one(cp, receipt=b"-0.42")        # receipt agrees with the sign
    assert matched["verdict"] == "HELD", matched


def test_an_ordinary_numeric_span_is_untouched():
    """G3."""
    _c, held = _one(None)
    assert held["verdict"] == "HELD", held
    _c2, failed = _one(None, number="0.43")
    assert failed["verdict"] == "FAILED", failed


def test_the_javascript_verifier_agrees_on_every_dash():
    """G4, the parity gate. Both implementations hand-mirror the token class, so both are wrong the
    same way; fixing one alone creates a parity defect of exactly the kind this audit found on
    U+0085.

    Compared on the CORE DIGEST through the differential harness's own node runner, not on a
    hand-rolled driver: a digest match is agreement on the whole core, verdicts included.
    """
    from shutil import which
    if which("node") is None:
        pytest.skip("node is not available")
    try:
        import conformance.sworn.differential as D
    except Exception as exc:                                     # noqa: BLE001
        pytest.skip("the differential harness is not importable here: %s" % exc)

    batch = [{"index": i, "document": _doc(cp), "manifest": _manifest().to_dict(),
              "name": "D.md", "commit": None}
             for i, cp in enumerate(DASHES)]
    import tempfile
    rows = D.js_digests(batch, Path(tempfile.mkdtemp()))

    disagree = []
    for i, cp in enumerate(DASHES):
        py_digest, py_err, _census = D.python_digest(batch[i])
        js = rows.get(i) or {}
        if py_digest != js.get("digest"):
            disagree.append("U+%04X python=%s(%s) node=%s(%s)"
                            % (cp, (py_digest or "-")[:12], py_err or "ok",
                               (js.get("digest") or "-")[:12], js.get("error") or "ok"))
    assert not disagree, "the two implementations disagree on %d of %d dashes:\n  %s" % (
        len(disagree), len(DASHES), "\n  ".join(disagree))
