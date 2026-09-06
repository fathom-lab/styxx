"""A HELD numeric span whose comparison was against zero says nothing about its receipt.

WHAT THIS IS NOT. It is not a claim that the rounding rule is wrong. `DECISIONS["rounding"]` says
the receipt scalar is "quantized to the printed fractional digits with ROUND_HALF_EVEN", and that is
deliberate and necessary: an author writing 0.42 against a receipt of 0.4211 is honestly rounding,
and demanding an exact match would FAIL every rounded figure in this corpus.

WHAT IT IS. The rule quantizes to the AUTHOR'S precision and has no floor. At zero fractional digits
it stops rounding and starts erasing:

    receipt {"a_share": 0.4211}   sentence "the A-share is 0."   ->   HELD

Genuine harness-minted L2 receipt, correct digest, `complete`, nothing malformed, nothing forged.
The author simply chose how much of the receipt's value to round away, and chose all of it.

The verdict is deliberately NOT changed — that would break the honest-rounding rule the format needs.
The HEADLINE counts these spans instead, derived from `receipt` and `receipt_rounded`, which the
detail already carries. The line drawn is not a threshold anyone has to argue about: it is the case
where the printed figure carries no information about the receipt at all, a non-zero receipt whose
comparison was against zero.

THE FIRST VERSION OF THIS SIGNAL WAS WRONG AND THE CORPUS CAUGHT IT. It added a `rounded_away` field
to the span's `detail` — and `detail` is INSIDE the digested core, so it moved the core digest of
every affected span and would have put this side out of agreement with `styxx/_data/sworn_verify.js`,
which knows nothing about it. The conformance generator refused the regeneration outright: "a moved
core is a finding about the verifier, never a reason to rewrite the set". A warning that changes what
the format digests is not a warning, it is a format change wearing a warning's clothes.

Found by the adversary in the sidecar attack battery, in its list of areas nobody had attacked.
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

PAYLOAD = b'{"a_share": 0.4211, "big": 4200000, "neg": -0.4211, "zero": 0}'


def _manifest():
    return sworn.Manifest.from_dict({
        "spec": "sworn/manifest/0.2", "harness": "ci", "turn": "t",
        "minted_at": "2026-09-01T00:00:00Z", "authored_sha256": [], "rung": "L2",
        "receipts": {"r1": {
            "id": "r1", "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
            "kind_of_source": "tool_stdout", "captured_at": "2026-09-01T00:00:00Z",
            "complete": True, "bytes": base64.b64encode(PAYLOAD).decode("ascii"),
        }},
    })


def _span(pointer: str, sentence: str):
    doc = ('<sworn r="r1#/%s" k="numeric">%s</sworn>\n' % (pointer, sentence)).encode("utf-8")
    core = sworn.verify(doc, name="d.md", manifest=_manifest(), commit=None)
    return core, core["spans"][0]


@pytest.mark.parametrize("sentence", ["the A-share is 0.4211.", "the A-share is 0.42.",
                                      "the A-share is 0.4."])
def test_honest_rounding_still_holds_and_is_not_flagged(sentence):
    """The rule this format needs. Without these, a fix could 'solve' the problem by refusing every
    rounded figure, which would be worse than the problem."""
    _core, s = _span("a_share", sentence)
    assert s["verdict"] == "HELD", s
    assert "WARNING" not in sworn._headline(_core), sworn._headline(_core)


def test_a_receipt_rounded_entirely_away_is_flagged():
    """The finding: 0.4211 against a sentence printing no fractional digits."""
    core, s = _span("a_share", "the A-share is 0.")
    assert s["verdict"] == "HELD", "the verdict is deliberately unchanged"
    d = s["detail"]
    # The detail already says it: the receipt and what it was compared against.
    assert d["receipt"] == "0.4211" and d["receipt_rounded"] == "0"
    # ...and nothing was ADDED to detail, because detail is inside the digested core. The first
    # version of this signal put a field there and moved a conformance vector's core digest; the
    # generator refused the regeneration and was right to.
    assert "rounded_away" not in d, (
        "detail is inside the digested core; a field added here moves it: %r" % d)
    line = sworn._headline(core)
    assert "WARNING" in line and "compared against 0" in line, line


def test_it_fires_on_a_negative_receipt_too():
    core, s = _span("neg", "the A-share is -0.")
    if s["verdict"] != "HELD":
        pytest.skip("a signed-zero token is not held here; the positive case carries the finding")
    assert "WARNING" in sworn._headline(core), sworn._headline(core)


def test_a_receipt_that_is_genuinely_zero_is_not_flagged():
    """Zero rounding to zero erases nothing, and flagging it would teach a reader to ignore the
    signal. This is the control that keeps the check honest."""
    _core, s = _span("zero", "the count is 0.")
    assert s["verdict"] == "HELD"
    assert "WARNING" not in sworn._headline(_core), sworn._headline(_core)


def test_a_large_receipt_printed_whole_is_not_flagged():
    _core, s = _span("big", "the loss is 4200000.")
    assert s["verdict"] == "HELD"
    assert "WARNING" not in sworn._headline(_core), sworn._headline(_core)


def test_a_failing_span_is_not_flagged():
    """rounded_away is about a HELD span that means nothing. A FAILED span already says so."""
    _core, s = _span("big", "the loss is 7.")
    assert s["verdict"] == "FAILED"
    assert "WARNING" not in sworn._headline(_core), sworn._headline(_core)


def test_the_headline_is_silent_when_nothing_was_rounded_away():
    core, _s = _span("a_share", "the A-share is 0.4211.")
    assert "WARNING" not in sworn._headline(core), sworn._headline(core)
