"""safeText must agree with Python's _safe_text, including on characters outside the BMP.

THE DEFECT THIS PINS. `styxx/_data/sworn_verify.js` mirrored

    str(x)[:limit].encode("utf-8", errors="replace").decode("utf-8")

with three things wrong at once, one of them destructive:

  1. **It destroyed valid astral characters.** The lone-surrogate fixup was
     `s.replace(/[\\ud800-\\udfff](?![\\udc00-\\udfff])/g, ...)`. That class spans d800-dfff — high
     AND low surrogates. For a valid pair `[d83d, de00]` the LOW half is itself in the class and is
     not followed by another low surrogate, so it matched and was replaced: U+1F600 came out of a
     detail field as `[d83d, fffd]`. Every astral character in a `detail.leaf` was corrupted.
  2. **The replacement character was wrong.** The code claimed a lone surrogate becomes U+FFFD.
     Python's `errors="replace"` emits `?` (U+003F) when it cannot ENCODE a surrogate; U+FFFD is
     what it emits on decode. `_safe_text("\\ud800")` returns `"?"`.
  3. **It sliced by UTF-16 code units.** Python's `[:80]` counts code points, so the two sides
     truncated a leaf of astral characters at different places, and the JavaScript cut could land
     between the halves of a pair.

The repair uses `Array.from`, which iterates a JS string by code point: a valid pair arrives as one
element of length 2, a lone surrogate as one of length 1. All three follow from that.

HOW IT WAS FOUND. Not by the 1689 conformance vectors, which pass identically before and after —
1689 ran, 1689 passed, 0 failed, both times. Not by the differential harness at its original
grammar, which ran 150000 cases and found nothing. It surfaced only in the 31 disagreements that
survived the BOM repair, after the mutation study named the payload aperture and the generator was
widened where that study pointed.

A KNOWN BOUNDARY, recorded rather than papered over. Python strings are sequences of code points
and can hold two ADJACENT LONE SURROGATES; JavaScript strings are UTF-16 and cannot tell that pair
apart from one astral character. `_safe_text("\\ud83d\\ude00")` is therefore `"??"` in Python and
U+1F600 in JavaScript, and no repair can reconcile it — the two languages disagree about what the
value IS, not about what to do with it. Such a value cannot cross the JSON transport this suite
uses, so it is asserted as a boundary below instead of as an agreement.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "styxx" / "_data" / "sworn_verify.js"

from styxx.sworn import _safe_text  # noqa: E402

_PROBE = """
const api = require(process.argv[2]);
const fs = require('fs');
const cases = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const out = {};
for (const k of Object.keys(cases)) out[k] = api.safeText(cases[k]);
fs.writeFileSync(process.argv[4], JSON.stringify(out));
"""

# Every case here is expressible identically in both languages: no adjacent lone surrogates.
AGREE = {
    "a valid astral character": "\U0001f600",
    "astral inside a sentence": "value \U0001f600 here",
    "a lone high surrogate": "\ud800",
    "a lone low surrogate": "\udc00",
    "a lone high surrogate inside text": "a\ud800b",
    "ordinary ascii": "hello",
    "a combining mark": "é",
    "a BOM": "﻿marked",
    "90 astral characters, so it truncates": "\U0001f600" * 90,
    "90 ascii, so it truncates": "a" * 90,
    "78 ascii then two astral, cutting at the boundary": "x" * 78 + "\U0001f600\U0001f600",
    "an astral character at the limit": "y" * 79 + "\U0001f600",
}


def _node():
    exe = shutil.which("node")
    if exe is None:
        pytest.skip("node is not on PATH; the two implementations cannot be compared here")
    return exe


@pytest.fixture(scope="module")
def js_results():
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        probe, inp, outp = work / "p.js", work / "in.json", work / "out.json"
        probe.write_text(_PROBE, encoding="utf-8")
        inp.write_text(json.dumps(AGREE), encoding="utf-8")
        r = subprocess.run([_node(), str(probe), str(JS), str(inp), str(outp)],
                           capture_output=True, text=True, encoding="utf-8", timeout=300)
        assert outp.exists(), "the node side wrote nothing: %s" % (r.stderr or "")[-400:]
        return json.loads(outp.read_text(encoding="utf-8"))


@pytest.mark.parametrize("label", sorted(AGREE))
def test_safe_text_agrees(label, js_results):
    want = _safe_text(AGREE[label])
    got = js_results[label]
    assert got == want, (
        "%s: python=%s javascript=%s" % (label, ascii(want), ascii(got)))


def test_a_valid_astral_character_survives_intact_on_both_sides(js_results):
    """The destructive half of the defect, stated on its own so a regression cannot hide inside
    the parametrised sweep: before the repair this returned a high surrogate plus U+FFFD."""
    assert _safe_text("\U0001f600") == "\U0001f600"
    assert js_results["a valid astral character"] == "\U0001f600"


def test_a_lone_surrogate_becomes_a_question_mark_not_a_replacement_character(js_results):
    """Python's errors='replace' emits U+003F when it cannot ENCODE. U+FFFD is the decode-side
    replacement and was the wrong character to copy."""
    assert _safe_text("\ud800") == "?"
    assert js_results["a lone high surrogate"] == "?"
    assert "�" not in js_results["a lone high surrogate"]


def test_truncation_counts_code_points_on_both_sides(js_results):
    """80 astral characters are 80 code points and 160 UTF-16 units. A side that counted units
    would return 40 of them."""
    got = js_results["90 astral characters, so it truncates"]
    assert len(_safe_text("\U0001f600" * 90)) == 80
    assert len([c for c in got]) == 80, "javascript kept %d code points, expected 80" % len(
        [c for c in got])


def test_adjacent_lone_surrogates_are_a_boundary_not_an_agreement():
    """The one case no repair can reconcile, asserted so it is not mistaken for a defect later.

    Python holds two adjacent lone surrogates as two code points and replaces both. JavaScript's
    UTF-16 string cannot distinguish them from one astral character. The two languages disagree
    about what the value IS.
    """
    two_lone = "\ud83d" + "\ude00"          # two code points in Python, one character in JS
    one_astral = "\U0001f600"               # one code point in Python, the same two UTF-16 units
    assert len(two_lone) == 2 and len(one_astral) == 1
    assert two_lone != one_astral, "python keeps these distinct; javascript cannot"
    assert _safe_text(two_lone) == "??", "both halves are unencodable on their own"
    assert _safe_text(one_astral) == one_astral
