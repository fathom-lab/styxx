# -*- coding: utf-8 -*-
"""Differential reproduction: styxx.attestation.jcs vs rfc8785 on object key order.

RFC 8785 section 3.2.3 ("Sorting of Object Properties") requires that object
properties be sorted by their UTF-16 CODE UNIT sequence. styxx.attestation._jcs
sorts with Python's default str comparison, which orders by Unicode CODE POINT.
The two orders differ exactly when one sibling key contains a non-BMP character
(U+10000 and above -> a UTF-16 surrogate pair beginning 0xD800) and another
sibling key starts at or above U+E000, because the surrogate lead unit 0xD800
sorts BELOW 0xE000 as a code unit but the code point U+1F600 sorts ABOVE U+E000.

Run:  python papers/v8/jcs_key_order/reproduce.py
Exits 0 when the divergence reproduces on the divergent object AND the two
backends agree on the all-BMP control object.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_REPO))

import rfc8785  # noqa: E402
from styxx.attestation import jcs as styxx_jcs  # noqa: E402

try:  # keep Windows consoles from dying on the non-BMP key
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:  # pragma: no cover
    pass


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def show(label: str, obj: dict) -> tuple[bytes, bytes]:
    s_bytes = styxx_jcs(obj).encode("utf-8")
    r_bytes = rfc8785.dumps(obj)
    print(f"--- {label}")
    print(f"    keys (python repr)        : {[k for k in obj]!r}")
    print(f"    styxx.attestation.jcs     : {s_bytes.decode('utf-8')!r}")
    print(f"    sha256                    : {sha(s_bytes)}")
    print(f"    rfc8785.dumps             : {r_bytes.decode('utf-8')!r}")
    print(f"    sha256                    : {sha(r_bytes)}")
    print(f"    bytes equal               : {s_bytes == r_bytes}")
    print()
    return s_bytes, r_bytes


def utf16_units(s: str) -> list[int]:
    b = s.encode("utf-16-be")
    return [(b[i] << 8) | b[i + 1] for i in range(0, len(b), 2)]


def main() -> int:
    # The minimal divergent object: one non-BMP key, one BMP key >= U+E000.
    #   ""      code point 0xE000     UTF-16 units [0xE000]
    #   "\U0001F600"  code point 0x1F600    UTF-16 units [0xD83D, 0xDE00]
    # code point order : U+E000  <  U+1F600
    # code unit  order : 0xD83D  <  0xE000   -> the emoji key comes FIRST
    divergent = {"": 1, "\U0001f600": 2}
    control = {"": 1, "é": 2, "a": 3, "￿": 4}

    print("styxx JCS key-order differential — RFC 8785 section 3.2.3")
    print(f"python           : {sys.version.split()[0]}")
    print(f"rfc8785          : {getattr(rfc8785, '__version__', 'unknown')}")
    print(f"repo             : {_REPO}")
    print()
    for name, key in (("BMP private-use", ""), ("non-BMP emoji", "\U0001f600")):
        print(f"    {name:16s} code points {[ord(c) for c in key]}  "
              f"utf-16 units {[hex(u) for u in utf16_units(key)]}")
    print()

    d_s, d_r = show("DIVERGENT object {U+E000: 1, U+1F600: 2}", divergent)
    c_s, c_r = show("CONTROL object (all keys BMP)", control)

    assert d_s != d_r, "expected the two backends to DIFFER on the divergent object"
    assert sha(d_s) != sha(d_r), "expected the digests to differ"
    assert c_s == c_r, "expected the two backends to AGREE on the all-BMP control"

    # The JavaScript side. Array.prototype.sort() compares by UTF-16 code unit,
    # so the committed JS canonicalizers may already be RFC-correct. Establish
    # that by running node, not by assuming it.
    js = _HERE.parent / "_js_probe.cjs"
    js.write_text(
        "\n".join(
            [
                "const fs = require('node:fs');",
                "const vm = require('node:vm');",
                "const { createHash } = require('node:crypto');",
                "const repo = process.argv[2];",
                "function loadCjs(rel) {",
                "  const code = fs.readFileSync(repo + '/' + rel, 'utf8');",
                "  const m = { exports: {} };",
                "  const ctx = vm.createContext({ module: m, exports: m.exports,",
                "    require, console, process, Buffer, TextEncoder, TextDecoder });",
                "  vm.runInContext(code, ctx, { filename: rel });",
                "  return m.exports;",
                "}",
                "const sworn = loadCjs('styxx/_data/sworn_verify.js');",
                "const web = loadCjs('web/styxx_verify.js');",
                "const obj = {}; obj['\\ue000'] = 1; obj['\\u{1F600}'] = 2;",
                "const ctl = {}; ctl['\\ue000']=1; ctl['\\u00e9']=2; ctl['a']=3; ctl['\\uffff']=4;",
                "const h = s => createHash('sha256').update(Buffer.from(s,'utf8')).digest('hex');",
                "const out = {",
                "  node_version: process.version,",
                "  sworn_verify_divergent: sworn.jcs(obj),",
                "  sworn_verify_divergent_sha256: h(sworn.jcs(obj)),",
                "  sworn_verify_control_sha256: h(sworn.jcs(ctl)),",
                "  styxx_verify_divergent: web.jcs(obj),",
                "  styxx_verify_divergent_sha256: h(web.jcs(obj)),",
                "  styxx_verify_control_sha256: h(web.jcs(ctl)),",
                "};",
                "process.stdout.write(JSON.stringify(out, null, 1));",
            ]
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    proc = subprocess.run(
        ["node", str(js), str(_REPO)], capture_output=True, text=True, encoding="utf-8",
        cwd=str(_REPO), input="", timeout=60,
    )
    print("--- JavaScript backends (node)")
    if proc.returncode != 0:
        print("    node FAILED:")
        print(proc.stderr.strip()[:4000])
        return 2
    jsout = json.loads(proc.stdout)
    for k in ("sworn_verify", "styxx_verify"):
        print(f"    {k:14s} divergent  : {jsout[k + '_divergent']!r}")
        print(f"    {k:14s} sha256     : {jsout[k + '_divergent_sha256']}")
        print(f"    {k:14s} control sha: {jsout[k + '_control_sha256']}")
    print()

    ref = sha(d_r)
    table = {
        "styxx.attestation.jcs (python)": sha(d_s),
        "rfc8785 0.1.4 (python)": ref,
        "styxx/_data/sworn_verify.js": jsout["sworn_verify_divergent_sha256"],
        "web/styxx_verify.js": jsout["styxx_verify_divergent_sha256"],
    }
    print("--- AGREEMENT on the divergent object (sha256 of the canonical bytes)")
    for name, dig in table.items():
        print(f"    {dig}  {'RFC-8785-order' if dig == ref else 'code-point-order'}  {name}")
    print()
    ctl_ref = sha(c_r)
    ctl_table = {
        "styxx.attestation.jcs (python)": sha(c_s),
        "rfc8785 0.1.4 (python)": ctl_ref,
        "styxx/_data/sworn_verify.js": jsout["sworn_verify_control_sha256"],
        "web/styxx_verify.js": jsout["styxx_verify_control_sha256"],
    }
    print("--- AGREEMENT on the all-BMP control object")
    for name, dig in ctl_table.items():
        print(f"    {dig}  {'agrees' if dig == ctl_ref else 'DIFFERS'}  {name}")
    print()
    assert len(set(ctl_table.values())) == 1, "all four backends must agree on the control"

    js.unlink(missing_ok=True)
    print("OK: divergence reproduced on the non-BMP object; all four backends agree on the control.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
