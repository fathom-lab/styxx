# -*- coding: utf-8 -*-
"""Enumerate every character class both verifiers implement, and compare them member by member.

WHY. The U+0085 defect (SPEC_path_segment_class_is_pinned_v01_2026_09_06) was one member missing
from a hand-expansion of Python's ``\\s`` in sworn_verify.js. It changed a DOCUMENT VERDICT --
SWORN-FAILED in Python, SWORN-HELD in node -- and survived 1689 replayed vectors because no vector
put that byte in a path. It was found by enumerating one class. This enumerates the rest.

WHAT IT FOUND. Every class defined by a Unicode PROPERTY diverges; every class defined by an
EXPLICIT LIST agrees. That is not a coding mistake on either side: it is the two runtimes
implementing the same property against different Unicode versions.

  python unicodedata 15.0.0   node unicode/icu 16.0

  _TOKEN / TOKEN_RE          DIFFER  python=137971 node=142975   (5004 node-only)
  _DIGIT / DIGIT_RE          DIFFER  python=680    node=760      (80 node-only)
  _HEXRUN / HEXRUN_RE        AGREE   python=22     node=22       (ASCII)
  _PATH_SEG_BAD / ...        AGREE   python=58     node=58       (explicit list, pinned)
  _DIRECTIONAL_OVERRIDE/...  AGREE   python=2      node=2        (explicit list)

HOW TO READ THAT. It bounds a claim rather than reporting a bug. "The two implementations agree on
1689 of 1689 vectors" is true, and true *because* no vector uses a code point the two runtimes
classify differently -- measured: 0 of 3981 blobs do. The agreement is conditional on Unicode
version parity, and that condition was never written down.

The JS regexes are LIFTED out of the shipped file rather than restated here, so this cannot pass by
testing a copy of the source it is meant to police.

  python conformance/sworn/class_census.py           # report; exit 1 if an EXPLICIT class differs
  python conformance/sworn/class_census.py --strict  # exit 1 if any class differs at all
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
JS = ROOT / "styxx" / "_data" / "sworn_verify.js"

from styxx import sworn  # noqa: E402

# (label, python pattern, JS const name, kind) — kind "explicit" means the class is written out and
# MUST agree; "property" means it is a Unicode property on at least one side and is version-bound.
PAIRS = [
    ("_TOKEN / TOKEN_RE", sworn._TOKEN, "TOKEN_RE", "property"),
    ("_DIGIT / DIGIT_RE", sworn._DIGIT, "DIGIT_RE", "property"),
    ("_HEXRUN / HEXRUN_RE", sworn._HEXRUN, "HEXRUN_RE", "explicit"),
    ("_PATH_SEG_BAD / PATH_SEG_BAD_RE", sworn._PATH_SEG_BAD, "PATH_SEG_BAD_RE", "explicit"),
    ("_DIRECTIONAL_OVERRIDE / DIRECTIONAL_OVERRIDE_RE", sworn._DIRECTIONAL_OVERRIDE,
     "DIRECTIONAL_OVERRIDE_RE", "explicit"),
]

_DRIVER = [
    "const fs = require('fs');",
    "const src = fs.readFileSync(process.argv[2], 'utf8');",
    "const names = JSON.parse(process.argv[3]);",
    "let prelude = '';",
    "const dash = src.match(/const DASH_BINDS =([\\s\\S]*?);/);",
    "if (dash) prelude += 'const DASH_BINDS =' + dash[1] + ';';",
    "const out = {};",
    "for (const name of names) {",
    "  const m = src.match(new RegExp('const ' + name + ' = ([^;]+);'));",
    "  if (!m) { out[name] = null; continue; }",
    "  let re;",
    "  try { re = eval(prelude + ' (' + m[1] + ')'); }",
    "  catch (e) { out[name] = 'ERR:' + e.message; continue; }",
    "  const hits = [];",
    "  for (let c = 0; c < 0x110000; c++) {",
    "    if (c >= 0xd800 && c <= 0xdfff) continue;",
    "    re.lastIndex = 0;",
    "    if (re.test(String.fromCodePoint(c))) hits.push(c);",
    "  }",
    "  out[name] = hits;",
    "}",
    "process.stdout.write(JSON.stringify(out));",
]


def python_members(pat) -> set:
    """Every scalar value the pattern matches on its own."""
    out = set()
    is_bytes = isinstance(pat.pattern, bytes)
    for c in range(0x110000):
        if 0xD800 <= c <= 0xDFFF:
            continue
        ch = chr(c)
        try:
            if pat.search(ch.encode("utf-8") if is_bytes else ch):
                out.add(c)
        except Exception:                                        # noqa: BLE001
            pass
    return out


def js_members(names) -> dict:
    d = Path(tempfile.mkdtemp())
    (d / "drv.js").write_text("\n".join(_DRIVER), encoding="utf-8")
    p = subprocess.run(["node", str(d / "drv.js"), str(JS), json.dumps(list(names))],
                       capture_output=True)
    if p.returncode != 0:
        raise SystemExit("REFUSED: the node side did not run: %s"
                         % p.stderr.decode("utf-8", "replace")[-400:])
    return json.loads(p.stdout.decode("utf-8"))


def unicode_versions():
    v = subprocess.run(["node", "-p", "process.versions.unicode || process.versions.icu"],
                       capture_output=True)
    return unicodedata.unidata_version, v.stdout.decode().strip()


def census() -> list:
    """[(label, kind, python_set, js_set)] for every pair."""
    js_all = js_members([n for _l, _p, n, _k in PAIRS])
    rows = []
    for label, pat, name, kind in PAIRS:
        hits = js_all.get(name)
        if hits is None or isinstance(hits, str):
            raise SystemExit("REFUSED: could not read %s from the shipped file: %s"
                             % (name, hits or "not found"))
        rows.append((label, kind, python_members(pat), set(hits)))
    return rows


def main(argv) -> int:
    py_v, node_v = unicode_versions()
    print("python unicodedata %s   node unicode/icu %s" % (py_v, node_v))
    if py_v.split(".")[0] != str(node_v).split(".")[0]:
        print("  the two runtimes are on DIFFERENT Unicode versions; every property-defined class")
        print("  below is expected to differ, and the agreement claim is bounded by this line")
    print()
    strict = "--strict" in argv
    bad = 0
    for label, kind, py, js in census():
        only_py, only_js = sorted(py - js), sorted(js - py)
        agree = not only_py and not only_js
        print("%-48s %-8s %-7s python=%-6d node=%-6d"
              % (label, kind, "AGREE" if agree else "DIFFER", len(py), len(js)))
        if agree:
            continue
        if kind == "explicit" or strict:
            bad += 1
        for who, s in (("python only", only_py), ("node only  ", only_js)):
            if s:
                shown = ", ".join("U+%04X" % c for c in s[:12])
                print("      %s (%d): %s%s" % (who, len(s), shown, " ..." if len(s) > 12 else ""))
    print()
    if bad:
        print("EXPLICIT classes that differ: %d — these are hand-written on both sides and a" % bad)
        print("difference is a defect, not a version skew.")
        return 1
    print("every EXPLICIT class agrees. Property-defined classes track each runtime's Unicode")
    print("version and are reported, not failed; see FINDING_unicode_version_skew_2026_09_06.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
