# -*- coding: utf-8 -*-
"""The certificate sentence a sworn verdict receipt carries says what the verifier checks.

papers/sworn/NOTE_certifies_correction_2026_10_24.md. The /v1 `certifies` sentence opened with a clause
the verifier does not check: its only authorship check is the manifest's. Receipts now issue as
``styxx.sworn.verdict-receipt/v2``, whose `certifies` is the corrected sentence and whose core differs
from a /v1 core in `schema` and `certifies` and nothing else. A /v1 receipt keeps its digest over its
bytes as committed and is compared with those two strings held at their /v1 values, so every committed
/v1 receipt re-derives exactly when it did; a re-worded /v1 receipt still fails.

This file is not a conformance source: it reads the repository, runs node and opens capsules. The /v2
conformance set's only source is tests/test_sworn_receipt_v2.py.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from styxx import sworn
from styxx.sworn import Manifest, issue_receipt, verify, verify_receipt

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "styxx" / "_data" / "sworn_verify.js"
V0, V1, V2 = ("styxx.sworn.verdict-receipt/v%d" % i for i in range(3))

# The /v1 sentence, as every /v1 receipt in history carries it (styxx/sworn.py l.277-283 at 2e98debc).
V1_SENTENCE = (
    "the spans the author bound were checked against bytes the author did not write, at the commit "
    "or manifest the document names and at the rung the manifest declares — NOT a claim that the "
    "document is correct, NOT a claim that the right sentences were bound, NOT a check that the tags "
    "were written at write time, NOT a check of any signature, and only as trustworthy as the harness "
    "that minted the manifest and the history that holds the commit"
)
# The /v2 sentence, word for word (the NOTE quotes it in full).
V2_SENTENCE = (
    "the spans the author bound that read HELD or FAILED were checked, each by its kind, against the "
    "receipt it names (its bytes, or for a hash span on a manifest entry that carries none, the sha256 "
    "the manifest records), at the commit and manifest the receipt records and at the rung the manifest "
    "declares; the only authorship check is against the manifest: an rN receipt is taken as not the "
    "author's when the manifest records it under a kind_of_source from the closed external vocabulary and "
    "does not list its sha256 in authored_sha256, and a path: or prereg: receipt when no manifest is given "
    "or the manifest does not list its sha256 there; authored_sha256 is the harness's record of what the "
    "agent produced in the turn the manifest names, so bytes the author wrote that the harness never "
    "digested, such as a file committed in another turn or under a name the harness never saw, are not "
    "caught; an UNRESOLVED span was not checked — NOT a claim that the author did not write the receipt's "
    "bytes, NOT a claim that the document is correct, NOT a claim that the right sentences were bound, NOT "
    "a check that the tags were written at write time, NOT a check of any signature, and only as "
    "trustworthy as the harness that minted the manifest and the tree that served the commit (git history "
    "from the CLI; a snapshot or in-memory tree on its caller's word)"
)
V1_OPENING = "the spans the author bound were checked against bytes the author did not write"


def _surface_lint():
    spec = importlib.util.spec_from_file_location("surface_lint_for_certifies", ROOT / "scripts" / "surface_lint.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["surface_lint_for_certifies"] = mod
    spec.loader.exec_module(mod)
    return mod


def sp(inner, r="r1", k="numeric"):
    return '<sworn r="%s" k="%s">%s</sworn>\n' % (r, k, inner)


def _manifest(**receipts):
    m = Manifest("test-harness", "turn-1", "2026-09-01T00:00:00Z")
    for rid, data in receipts.items():
        m.add(rid, data, "tool_stdout", complete=True, captured_at="2026-09-01T00:00:00Z")
    return m


def _cores():
    """One fresh core per document verdict and per span verdict this verifier prints."""
    m = _manifest(r1=b"0.5")
    return {
        "held": verify(sp("0.5").encode(), name="d.md", manifest=m),
        "failed": verify(sp("0.6").encode(), name="d.md", manifest=m),
        "unresolved": verify(sp("0.5").encode(), name="d.md"),
        "malformed": verify(sp("0.5", k="nope").encode(), name="d.md", manifest=m),
        "unsworn": verify(b"# nothing sworn here\n", name="d.md"),
        "document_malformed": verify(b"```\nan unbalanced fence\n", name="d.md"),
        "tree": verify(sp("0.5", r="path:a.json").encode(), name="d.md",
                       tree=sworn.MemoryTree({"a.json": b"0.5"}, commit="a" * 40), commit="a" * 40),
    }


# ============================================================================ the strings

def test_the_receipt_schema_strings_are_pinned_verbatim():
    """The exact pins tests/test_sworn.py's doctrine test handed over when it became schema-neutral."""
    assert sworn.SPEC == "sworn/0.1"
    assert sworn.MANIFEST_SPEC == "sworn/manifest/0.2"
    assert sworn.MANIFEST_SPECS == ("sworn/manifest/0.1", "sworn/manifest/0.2")
    assert sworn.RECEIPT_SCHEMA == V2
    # the older strings are still LOADED and never EMITTED (M7)
    assert sworn.RECEIPT_SCHEMAS == (V0, V1, V2)
    assert sworn._CERTIFIES == V2_SENTENCE
    assert sworn._CERTIFIES_V1 == V1_SENTENCE
    assert sworn._CERTIFIES_OF == {V1: V1_SENTENCE, V2: V2_SENTENCE}
    assert len(V2_SENTENCE) == 1345 and len(V2_SENTENCE.encode("utf-8")) == 1347


def test_a_fresh_receipt_issues_at_v2_with_the_corrected_sentence():
    core = verify(sp("0.5").encode(), name="d.md", manifest=_manifest(r1=b"0.5"))
    assert core["schema"] == V2 and core["certifies"] == V2_SENTENCE
    rec = issue_receipt(core, timestamp="2026-10-24T00:00:00Z")
    assert rec["schema"] == V2 and rec["certifies"] == V2_SENTENCE
    # certifies stays inside the digest: a re-worded /v2 sentence breaks it
    assert "certifies" not in sworn._RECEIPT_OUTSIDE_DIGEST
    assert "NOTE_certifies_correction_2026_10_24" in sworn.DECISIONS["receipt_v2"]
    assert core["verifier"]["decisions"]["receipt_v2"] == sworn.DECISIONS["receipt_v2"]


def test_the_v2_sentence_keeps_every_true_clause_and_drops_the_wide_one():
    s = sworn._CERTIFIES
    # kept word for word: tests across the sworn suites pin them
    for kept in ("NOT a claim that the document is correct", "right sentences were bound",
                 "at the rung the manifest declares", "NOT a check that the tags were written at write time",
                 "NOT a check of any signature"):
        assert kept in s, kept
    # the correction
    for new in ("that read HELD or FAILED were checked", "the only authorship check is against the manifest",
                "are not caught", "an UNRESOLVED span was not checked",
                "NOT a claim that the author did not write the receipt's bytes",
                "the tree that served the commit"):
        assert new in s, new
    assert V1_OPENING not in s
    assert "bytes the author did not write" not in s
    assert "\n" not in s and "  " not in s and "—" in s
    hits, _ = _surface_lint().lint_text(s, "certifies.txt")
    assert [h.rid for h in hits if h.kind == "charter"] == []
    assert [h.rid for h in hits if h.rid == "withdrawn/certifies-v1"] == []


def test_the_census_the_note_swears_to_reproduces():
    """papers/sworn/certifies_census_2026_10_24.json is what the NOTE's numeric spans read. Every key
    regenerates from git and styxx.sworn here except one: re-deriving the receipts beside the sidecars
    takes minutes, and tests/test_sworn_dogfood.py re-derives each of them anyway, so here it is held
    to the count it names (python papers/sworn/certifies_census.py --check runs it in full)."""
    spec = importlib.util.spec_from_file_location("certifies_census", ROOT / "papers" / "sworn" / "certifies_census.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    committed = json.loads(mod.OUT.read_text(encoding="utf-8"))
    fresh = mod.census(verify_receipts=False)
    assert fresh.pop("sidecar_receipts_verified") is None
    verified = committed.pop("sidecar_receipts_verified")
    assert committed == fresh
    assert verified == committed["sidecar_receipts"] == committed["sidecars"]
    assert committed["v1_opening"] == V1_OPENING and V1_SENTENCE.startswith(V1_OPENING)
    assert committed["committed_receipts_modified_or_deleted_since_base"] == 0


def test_the_v1_opening_stands_only_beside_its_withdrawal():
    """scripts/surface_lint.py lists the /v1 opening as withdrawn. On shipped surfaces it stands only in
    the two frozen constants, each under its WITHDRAWN marker; papers/ (the receipts, the samples, the
    NOTE) is not a shipped surface and is not linted."""
    L = _surface_lint()
    assert "withdrawn/certifies-v1" in {r.rid for r in L.RULES}
    hits, allowed = L.lint_text(V1_SENTENCE, "x.md")
    assert [h.rid for h in hits] == ["withdrawn/certifies-v1"] and allowed == []
    rep = L.run(L.tree_files(ROOT, L.PROFILES["repo"]))
    assert [h for h in rep.hits if h.rid == "withdrawn/certifies-v1"] == []
    assert sorted(h.path for h in rep.allowed if h.rid == "withdrawn/certifies-v1") == [
        "styxx/_data/sworn_verify.js", "styxx/sworn.py"]


# ============================================================================ the projection

def test_core_as_sets_two_keys_and_nothing_else():
    for label, core in _cores().items():
        assert core["schema"] == V2 and core["certifies"] == V2_SENTENCE, label
        before = sworn._jcs(core)
        for schema in (V1, V2):
            out = sworn._core_as(core, schema)
            assert out is not core
            assert set(out) == set(core), label
            moved = sorted(k for k in core if sworn._jcs(core[k]) != sworn._jcs(out[k]))
            assert moved == ([] if schema == V2 else ["certifies", "schema"]), (label, schema, moved)
            assert out["schema"] == schema and out["certifies"] == sworn._CERTIFIES_OF[schema]
        assert sworn._jcs(core) == before, "the projection must not touch its input"
        assert sworn._core_as(core, V2) == core
    for bad in (V0, "styxx.sworn.verdict-receipt/v9", None, ""):
        with pytest.raises(ValueError):
            sworn._core_as(_cores()["held"], bad)


def test_the_recorder_pin_moves_two_strings_and_no_verdict(monkeypatch):
    """What conformance/sworn/recorder.py does while the /v1 set's sources run (C3b): hold
    RECEIPT_SCHEMA and _CERTIFIES at /v1. The core it gives is the live core projected onto /v1,
    for a document of every verdict, verifier block aside."""
    live = _cores()
    monkeypatch.setattr(sworn, "RECEIPT_SCHEMA", V1)
    monkeypatch.setattr(sworn, "_CERTIFIES", sworn._CERTIFIES_V1)
    pinned = _cores()
    for label in live:
        a = {k: v for k, v in pinned[label].items() if k != "verifier"}
        b = {k: v for k, v in sworn._core_as(live[label], V1).items() if k != "verifier"}
        assert sworn._jcs(a) == sworn._jcs(b), label
        assert pinned[label]["document_verdict"] == live[label]["document_verdict"], label


def _redigest(r):
    return issue_receipt({k: v for k, v in r.items() if k not in ("digest", "timestamp", "coverage_sha256")},
                         timestamp="2026-09-01T00:00:00Z")


def test_a_v1_receipt_re_derives_and_a_re_worded_one_fails():
    doc = sp("0.5").encode()
    m = _manifest(r1=b"0.5")
    fresh = verify(doc, name="d.md", manifest=m)
    v1 = issue_receipt(sworn._core_as(fresh, V1), timestamp="2026-09-01T00:00:00Z")
    res = verify_receipt(v1, doc, manifest=m)
    assert res["status"] == "VERIFIED" and res["schema"] == V1
    assert "held at their /v1 strings" in res["note"]
    for sentence in ("every span here is true", V2_SENTENCE):
        edited = _redigest(dict(v1, certifies=sentence))
        assert edited["digest"] != v1["digest"]
        r = verify_receipt(edited, doc, manifest=m)
        assert r["status"] == "FAILED" and r["digest_match"] is True and r["verdict_reproduces"] is False, sentence
    v2 = issue_receipt(fresh, timestamp="2026-10-24T00:00:00Z")
    assert verify_receipt(v2, doc, manifest=m)["status"] == "VERIFIED"
    assert verify_receipt(_redigest(dict(v2, certifies=V1_SENTENCE)), doc, manifest=m)["status"] == "FAILED"
    assert verify_receipt(dict(v2, certifies=V1_SENTENCE), doc, manifest=m)["digest_match"] is False


# ============================================================================ history

def _receipt_objects(obj, where="$"):
    """Every verdict-receipt object in a JSON value, nested ones included (the walk
    papers/sworn/receipt_provenance_audit.py makes), keyed on its schema string (M7)."""
    if isinstance(obj, dict):
        if isinstance(obj.get("schema"), str) and obj["schema"].startswith("styxx.sworn.verdict-receipt/"):
            yield where, obj
        for k, v in obj.items():
            yield from _receipt_objects(v, "%s.%s" % (where, k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _receipt_objects(v, "%s[%d]" % (where, i))


def test_the_v1_sentence_is_the_one_history_carries():
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "papers"], capture_output=True,
                         check=True).stdout
    paths = [p for p in out.decode("utf-8").split("\0") if p.endswith(".json")]
    seen = {V0: 0, V1: 0, V2: 0}
    for rel in paths:
        try:
            obj = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        except (UnicodeDecodeError, ValueError):
            continue
        for where, rec in _receipt_objects(obj):
            schema = rec["schema"]
            assert schema in sworn.RECEIPT_SCHEMAS, (rel, where, schema)
            seen[schema] += 1
            if schema in sworn._CERTIFIES_OF:
                assert rec.get("certifies") == sworn._CERTIFIES_OF[schema], (rel, where, schema)
    assert seen[V1] >= 71, seen
    assert seen[V0] == 0, "no /v0 receipt was committed at 2e98debc; one appearing is a finding"


# ============================================================================ the browser verifier

def _node():
    exe = shutil.which("node")
    if exe is None:
        pytest.skip("node is not on PATH; the browser verifier's strings and projection cannot be compared here")
    return exe


_JS_PROBE = r"""
const api = require(process.argv[1]);
const input = JSON.parse(require("fs").readFileSync(0, "utf8"));
const doc = new Uint8Array(Buffer.from(input.doc, "base64"));
const core = api.swornVerify(doc, api.jsonPlain(JSON.stringify(input.manifest)), { name: "d.md", commit: null });
let threw = false;
try { api.coreAs(core, "styxx.sworn.verdict-receipt/v9"); } catch (e) { threw = true; }
const before = api.jcs(core);
const v1 = api.coreAs(core, "styxx.sworn.verdict-receipt/v1");
process.stdout.write(JSON.stringify({
  schema: api.RECEIPT_SCHEMA, by: api.CERTIFIES_BY_SCHEMA, frozen: Object.isFrozen(api.CERTIFIES_BY_SCHEMA),
  core_jcs: api.jcs(core), core_digest: api.coreDigest(core), v1_jcs: api.jcs(v1),
  untouched: api.jcs(core) === before, threw }));
"""


def test_the_browser_verifier_carries_the_same_strings_and_projection():
    node = _node()
    doc = sp("0.5").encode()
    m = _manifest(r1=b"0.5")
    core = {k: v for k, v in verify(doc, name="d.md", manifest=m).items() if k not in ("verifier", "coverage")}
    payload = json.dumps({"doc": base64.b64encode(doc).decode("ascii"), "manifest": m.to_dict()})
    p = subprocess.run([node, "-e", _JS_PROBE, str(JS)], input=payload, capture_output=True, text=True,
                       encoding="utf-8")
    assert p.returncode == 0, p.stderr
    got = json.loads(p.stdout)
    assert got["schema"] == sworn.RECEIPT_SCHEMA == V2
    assert got["by"] == {V1: sworn._CERTIFIES_V1, V2: sworn._CERTIFIES}
    assert got["frozen"] is True and got["untouched"] is True and got["threw"] is True
    assert got["core_jcs"] == sworn._jcs(core)
    assert got["core_digest"] == sworn._sha256(sworn._jcs(core).encode("utf-8"))
    assert got["v1_jcs"] == sworn._jcs(sworn._core_as(core, V1))


# ============================================================================ the capsule profile

def test_the_committed_sworn_capsule_still_verifies():
    """The sealed capsule embeds a /v1 receipt and the /v1 browser verifier. Layer 2 re-runs the
    installed verifier, which issues /v2, and compares on the receipt's own schema."""
    from styxx.capsule import verify_capsule
    rep = verify_capsule(ROOT / "papers" / "sworn" / "CANNED_harness_junit_2026_09_05.capsule.html")
    assert rep["ok"] is True, rep["problems"]
    assert any("INSTRUMENT SKEW" in a for a in rep["advisory"]), rep["advisory"]


_LAYER1 = r"""
const fs=require('fs');
const html=fs.readFileSync(process.argv[1],'utf8');
const m=html.match(/<script type="application\/json" id="oath-capsule">([\s\S]*?)<\/script>/);
const P=JSON.parse(m[1].replace(/\\u003c/g,'<'));
const js=html.match(/<script>([\s\S]*?)<\/script>\s*<script>\s*\(function/);
const mod={exports:{}};
new Function('module','exports','globalThis',js[1])(mod,mod.exports,globalThis);
const api=globalThis.swornVerifyApi;
const b64=s=>new Uint8Array(Buffer.from(s,'base64'));
const man=P.manifest?api.jsonPlain(JSON.stringify(P.manifest)):null;
const live=api.swornVerify(b64(P.document.b64),man,{name:P.receipt.document.name,commit:P.receipt.commit});
const core=api.coreAs(live,P.receipt.schema);
console.log(JSON.stringify({schema:P.receipt.schema,live_schema:live.schema,
  core_ok:api.coreDigest(core)===P.core_sha256,unprojected_ok:api.coreDigest(live)===P.core_sha256}));
"""


def test_a_capsule_seals_a_v1_receipt_and_both_layers_re_derive_it(tmp_path):
    from styxx.capsule import create_capsule_sworn, verify_capsule
    report = b'{"passed": 12}\n'
    m = Manifest("pytest", "turn-1", "2026-09-01T00:00:00Z", rung="L1")
    m.add("r1", report, "test_report", complete=True, captured_at="2026-09-01T00:00:00Z")
    man = tmp_path / "m.manifest.json"
    m.write(man)
    doc = tmp_path / "note.md"
    doc.write_bytes(b'# note\n\n<sworn r="r1#/passed" k="numeric">The battery passed 12 checks.</sworn>\n')
    core = verify(doc.read_bytes(), name=doc.name, manifest=m)
    rec = tmp_path / "r.sworn-receipt.json"
    rec.write_text(json.dumps(issue_receipt(sworn._core_as(core, V1), timestamp="2026-09-01T00:00:00Z"), indent=1),
                   encoding="utf-8")
    cap = create_capsule_sworn(doc, man, rec, tmp_path / "note.capsule.html")
    rep = verify_capsule(cap)
    assert rep["ok"] is True, rep["problems"]
    html = cap.read_text(encoding="utf-8")
    assert "api.coreAs(api.swornVerify(" in html, "the page's layer 1 projects onto the sealed schema"
    node = _node()
    p = subprocess.run([node, "-e", _LAYER1, str(cap)], capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr
    got = json.loads(p.stdout)
    assert got == {"schema": V1, "live_schema": V2, "core_ok": True, "unprojected_ok": False}
