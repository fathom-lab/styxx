# -*- coding: utf-8 -*-
"""unicode_table.py: the pinned A.3 classes of DERIVE v0 (section 18.3 repair).

A.3 cuts the inner text into tokens of letters and numbers (General_Category L* and N*) and calls a
token digit-bearing when it holds an Nd character. Read from the host engine, those classes moved with
the engine's Unicode version, and the two checkers disagreed (Python 3.12.10 reads 15.0.0, Node 24.13.0
reads 16.0). DERIVE v0 now pins Unicode 15.0.0 and carries the two classes as range tables, written
here from Python's own unicodedata (no network, nothing fetched).

    python -I unicode_table.py --write   writes ../unicode_15_0_0.txt and the marked blocks in
                                          ../derive.py and ../derive.js
    python -I unicode_table.py --check   exits 0 when all three equal what this script computes

Both modes refuse (exit 3) unless the running Python's unicodedata.unidata_version is 15.0.0, so the
table can only be regenerated from that version and never drifts with a newer Python.
"""
import hashlib
import os
import sys
import unicodedata

VERSION = "15.0.0"
HERE = os.path.dirname(os.path.abspath(__file__))
DDIR = os.path.dirname(HERE)
TABLE = os.path.join(DDIR, "unicode_15_0_0.txt")
PY = os.path.join(DDIR, "derive.py")
JS = os.path.join(DDIR, "derive.js")
PER_LINE = 12


def ranges(pred):
    out, start = [], None
    for cp in range(0x110000):
        ok = pred(chr(cp))
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, 0x10FFFF))
    return out


def item(r):
    a, b = r
    return "%04X" % a if a == b else "%04X-%04X" % (a, b)


def tables():
    ln = ranges(lambda c: unicodedata.category(c)[0] in ("L", "N"))
    nd = ranges(lambda c: unicodedata.category(c) == "Nd")
    return [item(r) for r in ln], [item(r) for r in nd]


def table_text(ln, nd):
    return ("# DERIVE v0 appendix A.3: the letter-or-number class (General_Category L* and N*) and the decimal\n"
            "# digit class (Nd) of Unicode " + VERSION + ", as inclusive code point ranges in hex.\n"
            "# Written by tools/unicode_table.py from Python's unicodedata " + VERSION + ".\n"
            "LN " + " ".join(ln) + "\n"
            "ND " + " ".join(nd) + "\n")


def chunks(items):
    return [items[i:i + PER_LINE] for i in range(0, len(items), PER_LINE)]


def py_block(ln, nd):
    out = ["# BEGIN UNICODE " + VERSION + " TABLE (written by tools/unicode_table.py; do not edit by hand)"]
    for name, items in (("_UNI_LN", ln), ("_UNI_ND", nd)):
        out.append(name + " = (")
        for c in chunks(items):
            out.append('    "' + " ".join(c) + ' "')
        out.append(")")
    out.append("# END UNICODE TABLE")
    return "\n".join(out)


def js_block(ln, nd):
    out = ["// BEGIN UNICODE " + VERSION + " TABLE (written by tools/unicode_table.py; do not edit by hand)"]
    for name, items in (("UNI_LN", ln), ("UNI_ND", nd)):
        out.append("const " + name + " = \"\"")
        for c in chunks(items):
            out.append('  + "' + " ".join(c) + ' "')
        out[-1] += ";"
    out.append("// END UNICODE TABLE")
    return "\n".join(out)


def splice(src, begin, end, block):
    i = src.index(begin)
    j = src.index(end, i) + len(end)
    return src[:i] + block + src[j:]


def main(argv):
    if unicodedata.unidata_version != VERSION:
        sys.stderr.write("refusing: this Python has Unicode %s, the table is %s\n" % (unicodedata.unidata_version, VERSION))
        return 3
    ln, nd = tables()
    want_table = table_text(ln, nd)
    with open(PY, "r", encoding="utf-8", newline="") as fh:
        py = fh.read()
    with open(JS, "r", encoding="utf-8", newline="") as fh:
        js = fh.read()
    want_py = splice(py, "# BEGIN UNICODE ", "# END UNICODE TABLE", py_block(ln, nd))
    want_js = splice(js, "// BEGIN UNICODE ", "// END UNICODE TABLE", js_block(ln, nd))
    if "--write" in argv:
        for path, text in ((TABLE, want_table), (PY, want_py), (JS, want_js)):
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
    ok = True
    for path, text in ((TABLE, want_table), (PY, want_py), (JS, want_js)):
        if not os.path.exists(path):
            ok = False
            continue
        with open(path, "r", encoding="utf-8", newline="") as fh:
            if fh.read() != text:
                ok = False
    sys.stdout.write("unicode %s: LN %d ranges, ND %d ranges; table sha256 %s; %s\n" % (
        VERSION, len(ln), len(nd), hashlib.sha256(want_table.encode("utf-8")).hexdigest(), "in agreement" if ok else "DIFFERS"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
