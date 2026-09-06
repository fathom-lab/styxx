"""The explicit character classes agree, and the corpus stays out of the version-skewed ones.

Two different claims, pinned separately.

FIRST: a class both verifiers write out by hand MUST agree. `_PATH_SEG_BAD` did not -- the JS
hand-expansion of Python's `\\s` dropped U+0085, and a path: target carrying it produced
SWORN-FAILED in Python and SWORN-HELD in node. That is a defect, and this fails on it.

SECOND: a class defined by a Unicode PROPERTY is bound to each runtime's Unicode version, and here
they differ -- CPython 15.0.0, V8's ICU 16.0. `_TOKEN`/`TOKEN_RE` differ by 5004 code points and
`_DIGIT`/`DIGIT_RE` by 80. That is not a defect in either implementation; it is a condition on the
project's agreement claim. The claim "a second implementation agrees on 1689 of 1689 vectors" is
true BECAUSE no vector uses a code point the two runtimes classify differently. This pins that
reason: if a future vector reaches into the skew, the bar quietly starts meaning something narrower,
and this fails first.

See papers/sworn/FINDING_unicode_version_skew_2026_09_06.md.
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CENSUS = ROOT / "conformance" / "sworn" / "class_census.py"
BLOBS = ROOT / "conformance" / "sworn" / "blobs.json"


def _census_mod():
    from shutil import which
    if which("node") is None:
        pytest.skip("node is not available")
    if not CENSUS.exists():
        pytest.skip("the census is not present in this checkout")
    import importlib
    sys.path.insert(0, str(CENSUS.parent))
    return importlib.import_module("class_census")


@pytest.fixture(scope="module")
def rows():
    mod = _census_mod()
    return mod.census()


def test_every_explicitly_written_class_agrees(rows):
    """A hand-written class differing between the two is a defect, as U+0085 was."""
    bad = []
    for label, kind, py, js in rows:
        if kind != "explicit":
            continue
        only_py, only_js = sorted(py - js), sorted(js - py)
        if only_py or only_js:
            bad.append("%s: python-only %s, node-only %s"
                       % (label, ["U+%04X" % c for c in only_py[:8]],
                          ["U+%04X" % c for c in only_js[:8]]))
    assert not bad, "hand-written classes disagree:\n  " + "\n  ".join(bad)


def test_the_property_classes_are_reported_not_asserted(rows):
    """The census must actually be looking at them, or the finding above is unfounded.

    This asserts the SHAPE of the measurement, not its outcome: a property class may agree (if the
    runtimes' Unicode versions match) or differ (if they do not). What must not happen is that the
    census silently stops covering them.
    """
    kinds = {kind for _l, kind, _p, _j in rows}
    assert "property" in kinds and "explicit" in kinds, kinds
    for label, kind, py, js in rows:
        assert py or js, "%s matched nothing on either side; the lift is broken" % label


def _corpus_code_points() -> set:
    """Every distinct scalar value appearing in any conformance blob."""
    store = json.load(io.open(BLOBS, encoding="utf-8"))
    store = store.get("blobs", store)
    seen = set()
    for v in store.values():
        if isinstance(v, dict):
            v = v.get("b64") or v.get("bytes") or v.get("text") or ""
        try:
            raw = base64.b64decode(v, validate=True)
            text = raw.decode("utf-8")
        except Exception:                                        # noqa: BLE001
            continue
        seen.update(ord(c) for c in text)
    return seen


def test_no_conformance_vector_uses_a_code_point_the_runtimes_disagree_about(rows):
    """Why the 1689 bar holds, stated as a check rather than as an assumption.

    Measured when written: 0 of 3981 blobs reach into the skew. If a vector ever does, the bar stops
    meaning "the two implementations agree" and starts meaning "they agree except where they do
    not" -- and this fails before that happens quietly.
    """
    if not BLOBS.exists():
        pytest.skip("no committed blob store in this checkout")
    skew = set()
    for _label, kind, py, js in rows:
        if kind == "property":
            skew |= (py ^ js)
    corpus = _corpus_code_points()
    assert corpus, "read no code points from the blob store; the reader is broken"
    overlap = sorted(corpus & skew)
    assert not overlap, (
        "%d conformance code point(s) fall where the two runtimes disagree: %s — the 1689-vector "
        "agreement no longer means what it says"
        % (len(overlap), ["U+%04X" % c for c in overlap[:12]]))


def test_the_census_runs_as_a_command():
    """CLI output is behaviour: the tool must work the way its docstring says."""
    from shutil import which
    import subprocess
    if which("node") is None:
        pytest.skip("node is not available")
    p = subprocess.run([sys.executable, str(CENSUS)], capture_output=True, cwd=str(ROOT))
    out = p.stdout.decode("utf-8", "replace")
    assert p.returncode == 0, out[-800:]
    assert "python unicodedata" in out and "node unicode/icu" in out
    assert "explicit" in out and "property" in out
