"""Decide, mechanically, which catalogue entries are genuine controls.

A control is a deliberately semantics-preserving edit that the differential guard must NOT catch.
G-K of `papers/sworn/SPEC_mutation_coverage_v01_2026_09_05.md` voids a whole run if one is caught,
because a guard that fires on a reworded comment is detecting editing rather than divergence.

That gate voided the first run — and not for the reason it was written. The proposing agents
mislabelled two entries. `findBytes(hay, needle, 0) >= 0` changed to `> 0` was called
semantics-preserving and is nothing of the sort: a needle at byte offset zero would report HELD
instead of FAILED, a false negative on the one kind whose whole purpose is swearing something is
absent. And `d.closer_at -> d.closer_end` re-points the code at a different existing field while
looking exactly like a rename — that one was MISSED, so it sat outside the denominator hiding a
real miss rather than inflating a rate. Two of twelve, wrong in both directions of harm.

So the label cannot be taken from whoever proposed it. This decides it from the edit itself.

THE CRITERION, fixed before it was applied and applied to every entry in BOTH directions — an entry
mislabelled a mutation is corrected too, even though that can only enlarge the denominator and make
the detection rate worse. An entry is a genuine control if and only if:

  (a) the mutated region lies entirely inside a comment or docstring; or
  (b) the snippets differ only in identifier tokens — never in an operator, keyword, numeric or
      string literal — AND every introduced identifier is bound nowhere else: absent from the CODE
      of the rest of the file and not a language builtin. That second half is what separates
      `q -> cursor` (a fresh local name) from `d.closer_at -> d.closer_end` and from
      `Decimal -> float`, both of which merely point at something else that already exists.

Three corrections were needed before this classifier could be trusted, recorded because a
classifier that decides a study's denominator has to be audited like anything else:

  1. It tokenised `0xfeff -> 0xfffe` as an identifier rename, so "the JSON reader stops refusing a
     BOM" read as a control. Numeric, string and escape literals are now tokenised BEFORE
     identifiers, and language keywords are excluded — `delete` in JavaScript is not a variable.
  2. Anchors carry their leading indentation, so a whole-line comment anchor began before the
     comment span and read as code. Membership is tested on the anchor's stripped extent.
  3. The freshness check searched raw file text, so `hay -> haystack` looked like a substitution
     because the word "haystack" appears in three docstrings. It searches CODE only, with comments
     and string literals blanked.

Usage:
    python conformance/sworn/control_audit.py --catalogue <in.json> [--out <corrected.json>]
"""
from __future__ import annotations

import argparse
import builtins as _b
import json
import re
from pathlib import Path

W = Path(__file__).resolve().parent.parent.parent
SRC = {"js": (W / "styxx/_data/sworn_verify.js").read_bytes().decode("utf-8"),
       "python": (W / "styxx/sworn.py").read_bytes().decode("utf-8")}

KEYWORDS = {
    "js": {"var", "let", "const", "function", "return", "if", "else", "for", "while", "do", "new",
           "delete", "typeof", "instanceof", "in", "of", "class", "this", "throw", "try", "catch",
           "finally", "switch", "case", "break", "continue", "null", "true", "false", "undefined",
           "void", "yield", "await", "async", "static", "get", "set", "extends", "super"},
    "python": {"def", "class", "return", "if", "elif", "else", "for", "while", "in", "is", "not",
               "and", "or", "None", "True", "False", "try", "except", "finally", "raise", "with",
               "as", "import", "from", "lambda", "pass", "break", "continue", "global", "nonlocal",
               "assert", "del", "yield", "await", "async"},
}

BUILTINS = {
    "python": set(dir(_b)),
    "js": {"Object", "Array", "String", "Number", "Boolean", "Math", "JSON", "Map", "Set",
           "BigInt", "Uint8Array", "DataView", "ArrayBuffer", "Error", "TypeError", "RangeError",
           "parseInt", "parseFloat", "isNaN", "isFinite", "Symbol", "Promise", "RegExp", "Date"},
}

TOKEN = re.compile(r"""
    (?P<str>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
  | (?P<num>0[xXbBoO][0-9a-fA-F_]+n?|\d[\d_]*\.?[\d_]*(?:[eE][+-]?\d+)?n?)
  | (?P<esc>\\[uxUX][0-9a-fA-F]+)
  | (?P<ident>[A-Za-z_$][A-Za-z_$0-9]*)
  | (?P<op>\S)
""", re.X)


def comment_spans(text: str, side: str):
    pat = (r"/\*(?:.|\n)*?\*/|//[^\n]*" if side == "js"
           else r'"""(?:.|\n)*?"""|\'\'\'(?:.|\n)*?\'\'\'|#[^\n]*')
    return [(m.start(), m.end()) for m in re.finditer(pat, text)]


COMMENTS = {s: comment_spans(SRC[s], s) for s in SRC}


def code_only(text: str, side: str) -> str:
    """The file with comments AND string literals blanked, so a name search sees bindings only."""
    out = list(text)
    for a, b in comment_spans(text, side):
        for i in range(a, b):
            out[i] = " "
    blanked = "".join(out)
    return re.sub(r'"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'',
                  lambda m: " " * len(m.group()), blanked)


CODE = {s: code_only(SRC[s], s) for s in SRC}


def inside_comment(m: dict) -> bool:
    src = SRC[m["side"]]
    i = src.find(m["old"])
    if i < 0:
        return False
    lead = len(m["old"]) - len(m["old"].lstrip())
    trail = len(m["old"]) - len(m["old"].rstrip())
    i, j = i + lead, i + len(m["old"]) - trail          # the anchor's stripped extent
    return any(a <= i and j <= b for a, b in COMMENTS[m["side"]])


def toks(text: str, side: str):
    out = []
    for mo in TOKEN.finditer(text):
        kind, val = mo.lastgroup, mo.group()
        if kind == "ident" and val in KEYWORDS[side]:
            kind = "kw"
        out.append((kind, val))
    return out


def classify(m: dict):
    side = m["side"]
    if inside_comment(m):
        return True, "the mutated region lies entirely inside a comment or docstring"

    o, n = toks(m["old"], side), toks(m["new"], side)
    if len(o) != len(n):
        return False, "token count changed (%d -> %d)" % (len(o), len(n))
    diffs = [(a, b) for a, b in zip(o, n) if a != b]
    non_ident = [(a, b) for a, b in diffs if a[0] != "ident" or b[0] != "ident"]
    if non_ident:
        k = non_ident[0]
        return False, "changes a %s -> %s: %r -> %r" % (k[0][0], k[1][0],
                                                        k[0][1][:26], k[1][1][:26])
    if not diffs:
        return True, "identical outside whitespace and comments"

    introduced = {b[1] for a, b in diffs}
    rest = CODE[side].replace(m["old"], " ", 1)
    already = sorted(
        x for x in introduced
        if x in BUILTINS[side]
        or re.search(r"(?<![A-Za-z_0-9$])%s(?![A-Za-z_0-9$])" % re.escape(x), rest))
    if already:
        return False, ("substitutes existing binding(s) %s - not a fresh local name"
                       % ", ".join(already))
    return True, "consistent rename to fresh name(s): %s" % ", ".join(sorted(introduced))


CRITERION = ("a control changes only comment or docstring text, or renames a local consistently to "
             "a name bound nowhere else (builtins count as bindings); anything touching an "
             "operator, keyword, numeric or string literal is a mutation")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--out", help="write a corrected catalogue here")
    a = ap.parse_args(argv)

    doc = json.loads(Path(a.catalogue).read_text(encoding="utf-8"))
    entries = doc["mutations"] if isinstance(doc, dict) else doc

    to_mut, to_ctl = [], []
    for m in entries:
        ok, why = classify(m)
        if bool(m.get("control")) != ok:
            (to_ctl if ok else to_mut).append((m["name"], why))
        print("%-9s %-7s %-52s %s" % ("control" if ok else "mutation", m["side"],
                                      m["name"][:52], why))
    print()
    print("control -> mutation: %d" % len(to_mut))
    for n, w in to_mut:
        print("   - %s :: %s" % (n, w))
    print("mutation -> control: %d" % len(to_ctl))
    for n, w in to_ctl:
        print("   - %s :: %s" % (n, w))

    if a.out:
        for m in entries:
            ok, why = classify(m)
            m["control"] = ok
            m["control_reason"] = why
        if isinstance(doc, dict):
            doc["counts"]["controls"] = sum(1 for m in entries if m["control"])
            doc["control_audit"] = {
                "criterion": CRITERION,
                "applied_to": "every entry, in both directions, blind to which one failed a gate",
                "control_to_mutation": [n for n, _ in to_mut],
                "mutation_to_control": [n for n, _ in to_ctl],
            }
        out = Path(a.out)
        out.write_bytes((json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False)
                         + "\n").encode("utf-8"))
        assert b"\r" not in out.read_bytes(), "conformance/ is -text pinned"
        print()
        print("wrote %s" % out.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
