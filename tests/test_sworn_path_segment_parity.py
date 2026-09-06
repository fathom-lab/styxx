"""The path-segment class rejects the same code points in both implementations.

`sworn.py` writes `_PATH_SEG_BAD = [\\\\\\s\\x00-\\x1f\\x7f*?\\[\\]]`. `sworn_verify.js` cannot write
`\\s` and mean the same thing, so it HAND-EXPANDS the class -- and the expansion omitted one member
of Python's `\\s`, U+0085 NEXT LINE. Over every Unicode scalar value: python rejected 58, node 57,
the difference being exactly U+0085.

A `path:` receipt whose target carried it split the two verifiers on the DOCUMENT VERDICT:

    PYTHON  document_verdict SWORN-FAILED   span MALFORMED/receipt_form
    NODE    document_verdict SWORN-HELD     span UNRESOLVED/no_repository

One implementation refused the document; the other said it held. 1689 replayed conformance vectors
did not see it, because no vector puts a U+0085 in a path.

WHY THIS TEST IS THE REPAIR AND THE CHARACTER IS NOT. The enumeration cannot be avoided: neither
language's `\\s` means the other's -- Python's includes U+0085 and excludes U+FEFF, JavaScript's the
reverse -- so a literal list is required on the JS side, and a literal list drifts unless something
compares it. This compares them, over all 1,114,112 scalar values, and it reads the JavaScript's
regex out of the shipped file rather than restating it, so it cannot pass by testing a copy of the
source it is meant to police.

Spec: papers/sworn/SPEC_path_segment_class_is_pinned_v01_2026_09_06.md (P1).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx import sworn  # noqa: E402

JS = ROOT / "styxx" / "_data" / "sworn_verify.js"
NEL = ""


def _node():
    from shutil import which
    return which("node")


def _python_rejected() -> set:
    return {c for c in range(0x110000) if sworn._PATH_SEG_BAD.search(chr(c))}


def _js_rejected(node) -> set:
    """Evaluate the SHIPPED regex, lifted from the file, over every scalar value."""
    src = JS.read_text(encoding="utf-8")
    m = re.search(r"const PATH_SEG_BAD_RE = new RegExp\((.*?)\);", src)
    assert m, "PATH_SEG_BAD_RE is no longer a `new RegExp(...)` literal; update this lift"
    driver = (
        "const re = new RegExp(%s);\n"
        "const out = [];\n"
        "for (let c = 0; c < 0x110000; c++) { if (re.test(String.fromCodePoint(c))) out.push(c); }\n"
        "process.stdout.write(JSON.stringify(out));\n" % m.group(1)
    )
    d = Path(tempfile.mkdtemp())
    (d / "drv.js").write_bytes(driver.encode("utf-8"))
    p = subprocess.run([node, str(d / "drv.js")], capture_output=True)
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")[:400]
    return set(json.loads(p.stdout.decode("utf-8")))


def test_both_implementations_reject_exactly_the_same_code_points():
    """P-G1, the guard that must be seen red, and the one that outlives this defect.

    Any future edit to either class that does not edit the other fails here.
    """
    node = _node()
    if node is None:
        pytest.skip("node is not available")
    py, js = _python_rejected(), _js_rejected(node)
    only_py = sorted(py - js)
    only_js = sorted(js - py)
    fmt = lambda s: [("U+%04X" % c) for c in s][:20]                       # noqa: E731
    assert not only_py and not only_js, (
        "the path-segment classes disagree: python-only %s, node-only %s (python rejects %d, node "
        "%d)" % (fmt(only_py), fmt(only_js), len(py), len(js)))


def test_the_nel_character_is_rejected_by_python():
    """The specific member the hand-expansion dropped; kept so the regression has a name."""
    assert sworn._PATH_SEG_BAD.search(NEL) is not None


def test_a_path_target_carrying_nel_gets_one_verdict_from_both(tmp_path):
    """P-G2. The class is a means; the document verdict is what a reader acts on."""
    node = _node()
    if node is None:
        pytest.skip("node is not available")
    try:
        import conformance.sworn.differential as D
    except Exception as exc:                                     # noqa: BLE001
        pytest.skip("the differential harness is not importable here: %s" % exc)

    doc = ('<sworn r="path:out%stxt" k="quote">the log says `all done` here</sworn>\n'
           % NEL).encode("utf-8")
    batch = [{"index": 0, "document": doc, "manifest": None, "name": "D.md", "commit": None}]
    rows = D.js_digests(batch, Path(tempfile.mkdtemp()))
    py, py_err, _c = D.python_digest(batch[0])
    js = rows.get(0) or {}
    assert py == js.get("digest"), (
        "a path: target carrying U+0085 gives python=%s(%s) and node=%s(%s)"
        % ((py or "-")[:12], py_err or "ok", (js.get("digest") or "-")[:12], js.get("error") or "ok"))


def test_an_ordinary_path_target_is_unaffected_in_both():
    """P-G3. Catches over-reach: the repair must not start rejecting ordinary paths."""
    node = _node()
    if node is None:
        pytest.skip("node is not available")
    try:
        import conformance.sworn.differential as D
    except Exception as exc:                                     # noqa: BLE001
        pytest.skip("the differential harness is not importable here: %s" % exc)

    doc = (b'<sworn r="path:results/out.txt" k="quote">the log says `all done` here</sworn>\n')
    batch = [{"index": 0, "document": doc, "manifest": None, "name": "D.md", "commit": None}]
    rows = D.js_digests(batch, Path(tempfile.mkdtemp()))
    py, _e, _c = D.python_digest(batch[0])
    assert py == (rows.get(0) or {}).get("digest")
    assert sworn._PATH_SEG_BAD.search("results/out.txt") is None
