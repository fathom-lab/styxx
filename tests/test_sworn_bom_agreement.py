"""A BOM must survive into both verifiers, or one of them vouches for what the other refuses.

THE DEFECT THIS PINS. `styxx/_data/sworn_verify.js` built its decoders as
`new TextDecoder("utf-8", { fatal: true })`. `ignoreBOM` defaults to **false**, which in WHATWG's
inverted naming means the decoder *strips* a leading U+FEFF. Three consequences, all live in the
shipped file until 2026-09-05:

  1. `jsonStrict`'s explicit BOM refusal was unreachable dead code on every bytes path — the BOM was
     gone before the check looked at it.
  2. A receipt payload prefixed with a BOM made `styxx.sworn` return MALFORMED/`receipt_not_json`
     and the JavaScript return **HELD**. The browser verifier reported a sentence as sworn and
     holding that the reference implementation refuses to read at all.
  3. The same stripping applied to a span's inner text, where Python keeps the character.

HOW IT WAS FOUND, and why that matters more than the bug. The 1689 conformance vectors pass
identically before and after the repair — they could not see it in either direction. The
differential harness at its original grammar ran 150000 cases and found 0 disagreements. Only after
the mutation study named the payload aperture as its largest blind spot, and the generator was
widened in exactly the places that study pointed at, did the two implementations part company: 712
disagreements in 150000 at the same seed and the same size. The blind spot was hiding a live
divergence.

So these tests do not go through the fuzzer. They are the minimal hand-built inputs the divergence
reduces to, pinned so it cannot come back quietly.
"""
from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "styxx" / "_data" / "sworn_verify.js"

from styxx import sworn  # noqa: E402

_RUNNER = r"""
const fs = require('fs');
const api = require(process.argv[2]);
const c = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
let out;
try {
  const doc = new Uint8Array(Buffer.from(c.document_b64, 'base64'));
  const man = c.manifest === null ? null : api.jsonPlain(JSON.stringify(c.manifest));
  const core = api.swornVerify(doc, man, { name: c.name, commit: c.commit });
  out = { core: core };
} catch (e) {
  out = { error: String(e && e.message ? e.message : e).slice(0, 200) };
}
fs.writeFileSync(process.argv[4], JSON.stringify(out));
"""


def _node():
    exe = shutil.which("node")
    if exe is None:
        pytest.skip("node is not on PATH; the two verifiers cannot be compared here")
    return exe


def _manifest_with(payload: bytes) -> dict:
    return {
        "spec": "sworn/manifest/0.2", "harness": "pytest", "turn": "t",
        "minted_at": "2026-09-01T00:00:00Z", "authored_sha256": [], "rung": "L1",
        "receipts": {"r1": {
            "id": "r1", "sha256": hashlib.sha256(payload).hexdigest(),
            "kind_of_source": "tool_stdout", "captured_at": "2026-09-01T00:00:00Z",
            "complete": True, "bytes": base64.b64encode(payload).decode("ascii"),
        }},
    }


def _both(doc: bytes, man):
    """(python_core, js_core) for one hand-built case. Neither side is instrumented."""
    pm = sworn.Manifest.from_dict(man) if man is not None else None
    pcore = sworn.verify(doc, name="d.md", manifest=pm, commit=None)
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        runner, inp, outp = work / "r.js", work / "in.json", work / "out.json"
        runner.write_text(_RUNNER, encoding="utf-8")
        inp.write_bytes(json.dumps({
            "name": "d.md", "commit": None, "manifest": man,
            "document_b64": base64.b64encode(doc).decode("ascii")}).encode("utf-8"))
        r = subprocess.run([_node(), str(runner), str(JS), str(inp), str(outp)],
                           capture_output=True, timeout=300)
        assert outp.exists(), "the node side wrote nothing: %s" % r.stderr.decode(
            "utf-8", "replace")[-400:]
        row = json.loads(outp.read_text(encoding="utf-8"))
    assert "error" not in row, "the node side raised: %s" % row.get("error")
    return pcore, row["core"]


def _same(pcore, jcore, what):
    assert pcore["counts"] == jcore["counts"], (
        "%s: the two verifiers disagree about the VERDICT — python=%r js=%r"
        % (what, pcore["counts"], jcore["counts"]))
    assert pcore["document_verdict"] == jcore["document_verdict"], what
    assert [s.get("reason") for s in pcore["spans"]] == [s.get("reason") for s in jcore["spans"]], (
        "%s: same verdict, different reason — python=%r js=%r"
        % (what, [s.get("reason") for s in pcore["spans"]],
           [s.get("reason") for s in jcore["spans"]]))


def test_the_js_decoder_does_not_strip_a_leading_bom():
    """The root cause, checked directly rather than through a verdict.

    `ignoreBOM: true` is the WHATWG spelling of "hand me the BOM as a character". Left at its
    default, TextDecoder removes it, and every refusal downstream that looks for one is dead code.
    """
    prog = """
const api = require(process.argv[2]);
const b = new Uint8Array([0xef, 0xbb, 0xbf, 0x7b, 0x7d]);
const t = api.decodeStrict(b);
console.log(JSON.stringify({len: t.length, first: t.charCodeAt(0)}));
"""
    with tempfile.TemporaryDirectory() as td:
        runner = Path(td) / "bom.js"
        runner.write_text(prog, encoding="utf-8")
        r = subprocess.run([_node(), str(runner), str(JS)],
                           capture_output=True, text=True, encoding="utf-8", timeout=300)
    assert r.returncode == 0, r.stderr[-400:]
    got = json.loads(r.stdout.strip())
    assert got["first"] == 0xfeff, (
        "decodeStrict stripped the BOM (first code point %#x, length %d). Python's "
        "bytes.decode('utf-8') keeps it, so this side would accept payloads the reference "
        "implementation refuses." % (got["first"], got["len"]))
    assert got["len"] == 3


def test_a_bom_prefixed_receipt_payload_is_refused_by_both():
    """The verdict-changing case. Before the repair: python MALFORMED, javascript HELD."""
    payload = b'\xef\xbb\xbf{"n": 1}'
    doc = b'<sworn r="r1#/n" k="numeric">the value is 1.</sworn>\n'
    p, j = _both(doc, _manifest_with(payload))
    _same(p, j, "a BOM-prefixed receipt payload")
    assert p["counts"]["HELD"] == 0, (
        "a receipt whose bytes are not valid JSON must not hold: %r" % (p["counts"],))
    assert p["spans"][0]["reason"] == "receipt_not_json"


def test_a_payload_without_a_bom_still_holds():
    """The repair must refuse a BOM, not refuse JSON. Without this, the test above passes for the
    wrong reason — a verifier that rejected every payload would satisfy it."""
    payload = b'{"n": 1}'
    doc = b'<sworn r="r1#/n" k="numeric">the value is 1.</sworn>\n'
    p, j = _both(doc, _manifest_with(payload))
    _same(p, j, "an ordinary payload")
    assert p["counts"]["HELD"] == 1, p["counts"]


def test_a_bom_inside_a_span_is_the_same_character_to_both():
    """decodeStrict also produces every span's inner text, so the same stripping reached there."""
    payload = '{"s": "﻿marked"}'.encode("utf-8")
    doc = '<sworn r="r1#/s" k="quote">﻿marked</sworn>\n'.encode("utf-8")
    p, j = _both(doc, _manifest_with(payload))
    _same(p, j, "a BOM inside a span's inner text")


def test_a_bom_prefixed_document_reads_the_same_on_both_sides():
    payload = b'{"n": 1}'
    doc = '﻿<sworn r="r1#/n" k="numeric">the value is 1.</sworn>\n'.encode("utf-8")
    p, j = _both(doc, _manifest_with(payload))
    _same(p, j, "a BOM-prefixed document")
