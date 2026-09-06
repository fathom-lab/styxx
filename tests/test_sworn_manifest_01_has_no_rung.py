"""A manifest/0.1 resolves at `undeclared`, which is what the verifier's own receipt already says.

`DECISIONS["rung"]` travels inside the digested core and states: "a manifest/0.1, or a 0.2 with no
rung, resolves at rung `undeclared`, never at L2". `Manifest.core()` honours that -- `rung` is
digested only for spec 0.2. `from_dict` and `rung_status()` did not: they read and consulted `rung`
for any spec. So a 0.1 manifest resolved at whatever rung it declared, through a field its own
digest does not cover.

That made a tamper free. On a document whose sentence asserts 0.99 against a receipt of 0.42:

    0.1  rung=None -> SWORN-FAILED  intact=True  manifest_digest=a328205796f6
    0.1  rung=L9   -> SWORN-HELD    intact=True  manifest_digest=a328205796f6   <- identical
    0.2  rung=L9   -> SWORN-HELD    intact=False (digest moved)                 <- tamper visible

WHAT IS NOT CHANGED, and is deliberately asserted here so a later reader does not "fix" it: an
unknown rung making every rN span UNRESOLVED is documented design, and a document with UNRESOLVED
spans and no FAILED ones being SWORN-HELD is documented, tested, and a proposal to change it was
retracted in PR #74. Neither is touched. The defect is only that a 0.1 was honoured as declaring a
rung at all.

Spec: papers/sworn/SPEC_manifest_01_has_no_rung_v01_2026_09_06.md (R1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx import sworn  # noqa: E402

V01, V02 = "sworn/manifest/0.1", "sworn/manifest/0.2"
TRUE_DOC = b'<sworn r="r1" k="numeric">the delta was 0.42 on the panel</sworn>\n'
FALSE_DOC = b'<sworn r="r1" k="numeric">the delta was 0.99 on the panel</sworn>\n'


def _manifest_dict(spec, rung, with_digest=True):
    """A manifest minted with no rung, then given one AFTER its digest was taken."""
    m = sworn.Manifest(harness="ci", turn="t1", rung=None, spec=spec)
    m.add("r1", b"0.42", "tool_stdout", complete=True)
    d = m.to_dict() if with_digest else m.core()
    if rung is not None:
        d["rung"] = rung
    return d


def _verify(spec, rung, doc=TRUE_DOC, with_digest=True):
    man = sworn.Manifest.from_dict(_manifest_dict(spec, rung, with_digest))
    core = sworn.verify(doc, name="D.md", manifest=man, tree=None)
    return core, man


@pytest.mark.parametrize("rung", ["L1", "L2"])
def test_a_v01_manifest_resolves_undeclared_whatever_it_declares(rung):
    """R-G1. The sentence the receipt prints: '...never at L2'."""
    core, _man = _verify(V01, rung)
    assert core["rungs"] == {"undeclared": 1}, (
        "a manifest/0.1 declaring %s resolved at %r, but DECISIONS['rung'] says a 0.1 resolves at "
        "`undeclared`, never at L2" % (rung, core["rungs"]))


def test_a_false_document_cannot_be_laundered_by_appending_a_rung():
    """R-G2, the guard that must be seen red.

    Appending one key to a 0.1 manifest turned SWORN-FAILED into SWORN-HELD with a byte-identical
    manifest_digest and intact() still true.
    """
    plain, man_plain = _verify(V01, None, doc=FALSE_DOC)
    tampered, man_tampered = _verify(V01, "L9", doc=FALSE_DOC)

    assert plain["document_verdict"] == "SWORN-FAILED", plain["document_verdict"]
    assert man_plain.digest_or_none() == man_tampered.digest_or_none(), (
        "the premise of this test is that the rung is OUTSIDE a 0.1 manifest's digest; if the "
        "digests now differ, the tamper became visible by another route and this test needs "
        "rewriting rather than deleting")
    assert tampered["document_verdict"] == "SWORN-FAILED", (
        "appending rung=L9 to a manifest/0.1 turned a false document into %s with an unchanged "
        "manifest_digest" % tampered["document_verdict"])


@pytest.mark.parametrize("rung", ["L1", "L2", "L9", "banana"])
def test_appending_a_rung_to_a_v01_changes_nothing_at_all(rung):
    """R-G3. The field is outside the digest, so it must also be outside the verdict."""
    base, _ = _verify(V01, None)
    got, _ = _verify(V01, rung)
    assert got["rungs"] == base["rungs"]
    assert got["document_verdict"] == base["document_verdict"]
    assert [s["verdict"] for s in got["spans"]] == [s["verdict"] for s in base["spans"]]


@pytest.mark.parametrize("rung", ["L1", "L2"])
def test_a_v02_still_resolves_at_the_rung_it_declares(rung):
    """R-G4. The repair must not swallow the channel that legitimately declares a rung."""
    d = _manifest_dict(V02, None, with_digest=False)
    d["rung"] = rung
    man = sworn.Manifest.from_dict(d)
    core = sworn.verify(TRUE_DOC, name="D.md", manifest=man, tree=None)
    assert core["rungs"] == {rung: 1}, core["rungs"]


def test_a_v02_with_an_appended_rung_is_still_caught_by_its_digest():
    """R-G5. On 0.2 the rung IS digested, and that is what makes the tamper visible."""
    _core, man = _verify(V02, "L9")
    assert man.intact() is False, "a 0.2 manifest's appended rung must break its declared digest"


def test_the_documented_decisions_this_repair_does_not_touch():
    """An unknown rung still declines, and UNRESOLVED still does not fail a document.

    Both are documented; the second had a change proposed and retracted in PR #74. Asserted here so
    that a later reader repairing R-G2 does not quietly take them with it.
    """
    d = _manifest_dict(V02, None, with_digest=False)
    d["rung"] = "L9"
    man = sworn.Manifest.from_dict(d)
    core = sworn.verify(TRUE_DOC, name="D.md", manifest=man, tree=None)
    assert core["spans"][0]["verdict"] == "UNRESOLVED"
    assert core["spans"][0]["reason"] == "rung_unknown"
    assert core["document_verdict"] == "SWORN-HELD", (
        "UNRESOLVED-only documents are SWORN-HELD by a documented, tested decision")


def test_the_javascript_verifier_agrees_on_every_spec_and_rung():
    """R-G6, the parity gate, by core digest through the differential harness's own node runner.

    Both implementations read `rung` for any spec and digest it only for 0.2, so both are wrong the
    same way and this passes until exactly one side is repaired.
    """
    from shutil import which
    if which("node") is None:
        pytest.skip("node is not available")
    try:
        import conformance.sworn.differential as D
    except Exception as exc:                                     # noqa: BLE001
        pytest.skip("the differential harness is not importable here: %s" % exc)

    combos = [(spec, rung, doc)
              for spec in (V01, V02)
              for rung in (None, "L1", "L2", "L9")
              for doc in (TRUE_DOC, FALSE_DOC)]
    batch = [{"index": i, "document": doc,
              "manifest": _manifest_dict(spec, rung), "name": "D.md", "commit": None}
             for i, (spec, rung, doc) in enumerate(combos)]

    import tempfile
    rows = D.js_digests(batch, Path(tempfile.mkdtemp()))
    bad = []
    for i, (spec, rung, _doc) in enumerate(combos):
        py, py_err, _c = D.python_digest(batch[i])
        js = rows.get(i) or {}
        if py != js.get("digest"):
            bad.append("%s rung=%s python=%s(%s) node=%s(%s)"
                       % (spec[-3:], rung, (py or "-")[:12], py_err or "ok",
                          (js.get("digest") or "-")[:12], js.get("error") or "ok"))
    assert not bad, "the two implementations disagree on %d of %d combinations:\n  %s" % (
        len(bad), len(combos), "\n  ".join(bad))
