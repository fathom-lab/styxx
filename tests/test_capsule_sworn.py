"""The sworn capsule profile: what it refuses to seal, and what each layer catches.

SPEC: ``papers/sworn/SPEC_sworn_browser_verifier_v01_2026_09_05.md`` B6. The capsule seals the
document bytes, the manifest, the verdict receipt and the browser verifier's own bytes. Layer 1
(browser) re-derives the portable core with ``sworn_verify.js``; layer 2
(``python -m styxx.capsule verify``) re-runs ``styxx.sworn``.

The honest claim, and these tests hold it to exactly that: *re-derives sworn span verdicts
offline; a forger controlling the whole file passes both browser layers; the package at the named
commit is the check.* The last test constructs that forger and shows layer 1 believing it.
"""
from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from styxx.capsule import (SPEC_SWORN, SWORN_REFUSALS, create_capsule_sworn, verify_capsule)
from styxx.sworn import Manifest, issue_receipt, verify

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "styxx" / "_data" / "sworn_verify.js"
PAYLOAD_RE = re.compile(r'<script type="application/json" id="oath-capsule">(.*?)</script>',
                        re.S)

DOC = ('# a small note\n\n'
       'Fathom Lab. <sworn r="r1#/passed" k="numeric">The battery passed 12 checks.</sworn>\n')
REPORT = b'{"passed": 12, "note": "the harness wrote this, not the author"}\n'


def _manifest(tmp: Path, *, complete=True, authored=()) -> Path:
    m = Manifest("pytest", "turn-1", "2026-09-01T00:00:00Z", rung="L1")
    for a in authored:
        m.record_authored(a)
    m.add("r1", REPORT, "test_report", complete=complete, captured_at="2026-09-01T00:00:00Z")
    p = tmp / "m.manifest.json"
    m.write(p)
    return p


def _doc(tmp: Path, text: str = DOC) -> Path:
    p = tmp / "note.md"
    p.write_bytes(text.encode("utf-8"))
    return p


def _receipt(tmp: Path, doc: Path, man: Path | None, name="r.sworn-receipt.json") -> Path:
    m = Manifest.from_dict(json.loads(man.read_text(encoding="utf-8"))) if man else None
    core = verify(doc.read_bytes(), name=doc.name, manifest=m)
    p = tmp / name
    p.write_text(json.dumps(issue_receipt(core), indent=1) + "\n", encoding="utf-8", newline="\n")
    return p


def _mint(tmp: Path, **kw) -> Path:
    doc = kw.get("doc") or _doc(tmp)
    man = kw["man"] if "man" in kw else _manifest(tmp)
    rec = kw.get("rec") or _receipt(tmp, doc, man)
    out = tmp / "note.capsule.html"
    return create_capsule_sworn(doc, man, rec, out)


def _payload(cap: Path) -> dict:
    html = cap.read_text(encoding="utf-8")
    return json.loads(PAYLOAD_RE.search(html).group(1).replace("\\u003c", "<"))


def _reseal(cap: Path, payload: dict) -> None:
    """Write a payload back into the page — the forger's move, and no more than that."""
    html = cap.read_text(encoding="utf-8")
    body = json.dumps(payload, indent=1).replace("<", "\\u003c")
    start = html.index('<script type="application/json" id="oath-capsule">') + \
        len('<script type="application/json" id="oath-capsule">')
    end = html.index("</script>", start)
    cap.write_text(html[:start] + body + html[end:], encoding="utf-8")


def _layer1(cap: Path) -> dict:
    """Run what the page's own layer 1 runs, in node, over the page's own payload."""
    if shutil.which("node") is None:
        pytest.skip("node is not on PATH")
    script = r"""
const fs=require('fs');
const html=fs.readFileSync(process.argv[1],'utf8');
const m=html.match(/<script type="application\/json" id="oath-capsule">([\s\S]*?)<\/script>/);
const P=JSON.parse(m[1].replace(/\\u003c/g,'<'));
// layer 1 runs the verifier INLINED IN THE PAGE, which is the forger's opportunity
const js=html.match(/<script>([\s\S]*?)<\/script>\s*<script>\s*\(function/);
const mod={exports:{}};
new Function('module','exports','globalThis',js[1])(mod,mod.exports,globalThis);
const api=globalThis.swornVerifyApi;
const b64=s=>new Uint8Array(Buffer.from(s,'base64'));
const out={};
try{
  const doc=b64(P.document.b64);
  out.js_sealed_ok=api.sha256Bytes(b64(P.verifier_js.b64))===P.verifier_js.sha256;
  out.doc_ok=api.sha256Bytes(doc)===P.document.sha256;
  const man=P.manifest?api.jsonPlain(JSON.stringify(P.manifest)):null;
  const core=api.swornVerify(doc,man,{name:P.receipt.document.name,commit:P.receipt.commit});
  out.verdict=core.document_verdict;
  out.counts=core.counts;
  out.core_ok=api.coreDigest(core)===P.core_sha256;
}catch(e){out.error=String(e);}
out.ok=!!(out.js_sealed_ok&&out.doc_ok&&out.core_ok);
console.log(JSON.stringify(out));
"""
    p = subprocess.run(["node", "-e", script, str(cap)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


# ---------------------------------------------------------------- minting and both layers

def test_a_sworn_capsule_verifies_at_both_layers(tmp_path):
    cap = _mint(tmp_path)
    rep = verify_capsule(cap)
    assert rep["ok"], rep["problems"]
    assert rep["spec"] == SPEC_SWORN
    assert rep["verdict"] == "SWORN-HELD"
    l1 = _layer1(cap)
    assert l1["ok"], l1
    assert l1["verdict"] == "SWORN-HELD"
    assert l1["core_ok"]


def test_the_page_carries_the_label_and_never_calls_itself_self_verifying(tmp_path):
    html = _mint(tmp_path).read_text(encoding="utf-8")
    assert "a forger controlling the whole file passes both browser layers" in html
    assert "the package at the named commit is the check" in html
    low = html.lower()
    for banned in ("self-verifying", "tamper-proof", "immutable"):
        assert banned not in low


def test_the_sealed_verifier_is_the_installed_one_and_is_inlined_byte_for_byte(tmp_path):
    cap = _mint(tmp_path)
    p = _payload(cap)
    js = base64.b64decode(p["verifier_js"]["b64"], validate=True)
    assert js == JS.read_bytes()
    assert js.decode("utf-8") in cap.read_text(encoding="utf-8")


# ---------------------------------------------------------------- the five refusals (B6)

def test_refusal_names_are_the_five_the_spec_froze():
    assert set(SWORN_REFUSALS) == {"sworn_no_manifest", "sworn_receipt_mismatch",
                                   "sworn_manifest_mismatch", "sworn_tree_receipt",
                                   "sworn_document_mismatch"}


def test_sworn_no_manifest(tmp_path):
    doc = _doc(tmp_path)
    man = _manifest(tmp_path)
    rec = _receipt(tmp_path, doc, man)
    with pytest.raises(SystemExit) as e:
        create_capsule_sworn(doc, None, rec, tmp_path / "x.capsule.html")
    assert "sworn_no_manifest" in str(e.value)


def test_sworn_tree_receipt(tmp_path):
    doc = _doc(tmp_path, '# t\n\n<sworn r="path:a/b.json#/n" k="numeric">It is 3.</sworn>\n')
    rec = _receipt(tmp_path, doc, None)
    with pytest.raises(SystemExit) as e:
        create_capsule_sworn(doc, None, rec, tmp_path / "x.capsule.html")
    assert "sworn_tree_receipt" in str(e.value)


def test_sworn_document_mismatch(tmp_path):
    doc = _doc(tmp_path)
    man = _manifest(tmp_path)
    rec = _receipt(tmp_path, doc, man)
    doc.write_bytes(DOC.replace("12 checks", "13 checks").encode("utf-8"))
    with pytest.raises(SystemExit) as e:
        create_capsule_sworn(doc, man, rec, tmp_path / "x.capsule.html")
    assert "sworn_document_mismatch" in str(e.value)


def test_sworn_manifest_mismatch(tmp_path):
    doc = _doc(tmp_path)
    man = _manifest(tmp_path)
    rec = _receipt(tmp_path, doc, man)
    other = json.loads(man.read_text(encoding="utf-8"))
    other["turn"] = "a different turn"                 # a different manifest, a different digest
    m2 = tmp_path / "m2.manifest.json"
    m2.write_text(json.dumps(other, indent=1) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(SystemExit) as e:
        create_capsule_sworn(doc, m2, rec, tmp_path / "x.capsule.html")
    assert "sworn_manifest_mismatch" in str(e.value)


def test_sworn_receipt_mismatch(tmp_path):
    doc = _doc(tmp_path)
    man = _manifest(tmp_path)
    rec = _receipt(tmp_path, doc, man)
    obj = json.loads(rec.read_text(encoding="utf-8"))
    obj["counts"]["HELD"] = 99                          # a receipt that no longer re-derives
    rec.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(SystemExit) as e:
        create_capsule_sworn(doc, man, rec, tmp_path / "x.capsule.html")
    assert "sworn_receipt_mismatch" in str(e.value)


# ---------------------------------------------------------------- the tamper battery

def test_tamper_a_document_byte_is_caught_by_both_layers(tmp_path):
    cap = _mint(tmp_path)
    p = _payload(cap)
    doc = bytearray(base64.b64decode(p["document"]["b64"], validate=True))
    doc[doc.index(b"12 checks")] = ord("1")
    doc[doc.index(b"12 checks") + 1] = ord("3")         # 12 -> 13, a lie about the receipt
    p["document"]["b64"] = base64.b64encode(bytes(doc)).decode("ascii")
    _reseal(cap, p)
    rep = verify_capsule(cap)
    assert not rep["ok"]
    assert any("document bytes" in x for x in rep["problems"]), rep["problems"]
    l1 = _layer1(cap)
    assert not l1["ok"], l1                             # the sealed digest no longer matches


def test_tamper_the_receipt_verdict_is_caught_by_both_layers(tmp_path):
    cap = _mint(tmp_path)
    p = _payload(cap)
    p["receipt"]["spans"][0]["verdict"] = "FAILED"
    p["receipt"]["counts"] = {"HELD": 0, "FAILED": 1, "UNRESOLVED": 0, "MALFORMED": 0,
                              "WITHHELD": 0}
    p["receipt"]["document_verdict"] = "SWORN-FAILED"
    _reseal(cap, p)
    rep = verify_capsule(cap)
    assert not rep["ok"]
    assert any("core does not re-derive" in x or "core_sha256" in x for x in rep["problems"]), \
        rep["problems"]
    # Layer 1 is UNMOVED, and that is the design rather than a gap: it re-derives from the
    # document and the manifest and compares to the sealed `core_sha256`, neither of which this
    # forger touched. A displayed verdict that disagrees with the sealed digest is layer 2's.
    l1 = _layer1(cap)
    assert l1["core_ok"], l1
    # and if the forger moves the sealed digest too, so the page is internally consistent,
    # layer 1 stops agreeing — the two checks close over each other
    p["core_sha256"] = "0" * 64
    _reseal(cap, p)
    assert not _layer1(cap)["core_ok"]
    assert not verify_capsule(cap)["ok"]


def test_tamper_the_manifest_digest_is_caught_by_both_layers(tmp_path):
    """A manifest whose declared digest no longer re-derives is not a sound manifest: every rN
    goes UNRESOLVED, so the core moves and both layers see it."""
    cap = _mint(tmp_path)
    p = _payload(cap)
    p["manifest"]["digest"] = "0" * 64
    _reseal(cap, p)
    rep = verify_capsule(cap)
    assert not rep["ok"]
    assert any("core does not re-derive" in x for x in rep["problems"]), rep["problems"]
    assert any("which is tamper" in x for x in rep["problems"]), rep["problems"]
    l1 = _layer1(cap)
    assert not l1["core_ok"], l1
    assert l1["counts"]["UNRESOLVED"] == 1, l1


def test_tamper_the_inlined_verifier_is_caught_by_layer_2_alone(tmp_path):
    """The heart of the label. A forger who edits only the copy the browser RUNS leaves the
    sealed copy intact: layer 1 is satisfied and lies, and layer 2 names it."""
    cap = _mint(tmp_path)
    html = cap.read_text(encoding="utf-8")
    # The running copy quietly un-does the change to the document before it verifies: the reader
    # sees "13 checks", and the verifier the page runs reads "12 checks" and agrees with itself.
    needle = "function swornVerify(documentBytes, manifestObj, opts) {"
    assert needle in html, "the entry point's signature moved; update this forgery"
    forged = html.replace(needle, needle + "\n"
                          '  { const _s = new TextDecoder().decode(documentBytes)'
                          '.replace("13 checks", "12 checks");\n'
                          "    documentBytes = new TextEncoder().encode(_s); }", 1)
    # and flip the document under it
    p = json.loads(PAYLOAD_RE.search(forged).group(1).replace("\\u003c", "<"))
    doc = base64.b64decode(p["document"]["b64"], validate=True).replace(b"12 checks", b"13 checks")
    p["document"]["b64"] = base64.b64encode(doc).decode("ascii")
    p["document"]["sha256"] = __import__("hashlib").sha256(doc).hexdigest()
    body = json.dumps(p, indent=1).replace("<", "\\u003c")
    start = forged.index('<script type="application/json" id="oath-capsule">') + \
        len('<script type="application/json" id="oath-capsule">')
    end = forged.index("</script>", start)
    cap.write_text(forged[:start] + body + forged[end:], encoding="utf-8")

    l1 = _layer1(cap)
    assert l1["ok"], ("the forgery was supposed to satisfy layer 1 — that is the claim being "
                      f"demonstrated, not a defect: {l1}")
    rep = verify_capsule(cap)
    assert not rep["ok"]
    assert any("inlined in the page is not the sealed one" in x for x in rep["problems"]), \
        rep["problems"]


def test_instrument_skew_is_advisory_and_named_apart_from_tamper(tmp_path):
    """A receipt issued by another build, with every byte in place, is SKEW — reported beside a
    verdict that still re-derives, never as tamper."""
    cap = _mint(tmp_path)
    p = _payload(cap)
    p["receipt"]["verifier"]["sworn_sha256"] = "b" * 64
    _reseal(cap, p)
    rep = verify_capsule(cap)
    assert rep["ok"], rep["problems"]
    assert rep["same_build"] is False
    assert any("INSTRUMENT SKEW" in a for a in rep["advisory"]), rep["advisory"]
