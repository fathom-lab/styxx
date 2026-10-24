"""#179 and #178 (fix/diffgate-178-179): the `async def` read, and the `def` lines outside Python source.

#179. Both templates that count `def` lines read `def` and not `async def`: the tests count was `^\\s*def test_` and
the symbol pattern `^\\s*(?:def|class)\\s+NAME\\b`. So a true "Added 1 test." over an added `async def test_x():`
read CONTRADICTED ("diff adds 0 test functions"), and a true "Added function fetch." over `async def fetch():` read
CONTRADICTED, at both doors, in prose and declared, in both ports, while CPython reads an AsyncFunctionDef. Both
ports now read `async` and a run of the white space CPython's tokenizer reads between it and `def` (space, tab, form
feed: `[ \\t\\f]+`, not `\\s`, so U+FEFF opens no new seam between the ports) wherever they read `def`. The PATH-2a
overlay's model of main's count (`_p2a_counted` / `_p2aCounted`) reads it too, so #101 still withholds a changed
async test: without that mirror a changed `async def test_x` reads a wrong VERIFIED.

#178. The templates count `def` lines wherever they sit: a fenced `def test_example():` in `docs/guide.md`, a
`def test_n():` in `notes.txt`, or a `def test_s():` inside a triple-quoted string added to a Python file. So a false
"Added 1 test." read VERIFIED, and a true one beside a documented example read CONTRADICTED. Both ports now abstain
(UNCHECKABLE, #178) on those two claims when a counted line sits in a file whose path does not end in .py or .pyi, or
after an odd number of triple quotes in the added lines of its `+++` section. Both counts are taken from one walk of
the diff (`_diff_sections` / `_diffSections`), which places each added line as `parse_unified_diff_sides` does (held
here on every committed input, in both ports): every section's added lines against the Python-source lines of the same
walk, so the abstention fires only where its reason holds, never where `parse_unified_diff` and
`parse_unified_diff_sides` place a line apart (#172's shapes). The parity and the path test are a section's, not a
key's of `parse_unified_diff_sides`: the two ports build those keys apart (Python's strip() and the port's trim()
remove different code points from a path's tail, and the runtimes lower case differently, U+1C89 on Node 24 and not on
CPython 3.12), so a parity taken per key put a `def test_` after an odd count in one port only, and the two printed
opposite gates (review 1 of fix batch 1); sections both ports read alike. The path's tail is read with every trailing
code point at or below U+0020 or from U+007F up removed, so strip() and trim() cannot part the path test either. The
abstention adds no VERIFIED and no CONTRADICTED.

What is held here: the reporters' cases at the raw door (both ports, both strict modes, the two ports' records equal,
reasons included) and at the git door (a real two-commit repository; skipped without git, failed under CI); the gap
after `async`; the changed async test the overlay withholds; the controls #178 must not touch; the one-parse rule; the
path's tail; the sections a key would merge; the walk placing every added line as the parse does; the named residuals,
pinned as they read; an ast grid over every case with a Python file model; and a plant of each change in each port,
each told by a check here.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import styxx.diffgate as N
from tests import _p2a_ref as R

NODE = shutil.which("node")
GIT = shutil.which("git")


def _in_ci() -> bool:
    return bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))


def _missing(tool: str, half: str):
    """A missing tool skips its half locally and fails it under CI, where a skip would read green."""
    if _in_ci():
        pytest.fail(f"{tool} is not on PATH under CI; {half} did not run")
    pytest.skip(f"{tool} is not on PATH; {half} cannot run here")


# The port, loaded as a page loads it (one script).
RUNNER = r"""
"use strict";
const fs = require("fs");
const vm = require("vm");
const [port, inp, out] = process.argv.slice(2);
const P = vm.runInContext(fs.readFileSync(port, "utf8") + "\n;({gateDiffText})", vm.createContext({}), { filename: port });
const rec = g => ({ verdict: g.verdict, claims: g.claims.map(c => [c.kind, c.verdict, c.why]) });
const res = JSON.parse(fs.readFileSync(inp, "utf8")).map(it => {
  const o = { id: it.id };
  try {
    o.off = rec(P.gateDiffText(it.summary, it.diff));
    o.on = rec(P.gateDiffText(it.summary, it.diff, { strict: true }));
  } catch (e) {
    o.error = String((e && e.message) || e);
  }
  return o;
});
fs.writeFileSync(out, JSON.stringify(res));
"""


def _port(cases: list, tmp_path: Path, port: Path = R.PORT, tag: str = "run") -> dict:
    """The port's records, by id, in both strict modes."""
    if NODE is None:
        _missing("node", "the port half of this check")
    runner = tmp_path / "runner_178_179.js"
    runner.write_text(RUNNER, encoding="utf-8")
    src, out = tmp_path / f"{tag}_in.json", tmp_path / f"{tag}_out.json"
    src.write_text(json.dumps([{k: c[k] for k in ("id", "summary", "diff")} for c in cases], ensure_ascii=True),
                   encoding="utf-8")
    r = subprocess.run([NODE, str(runner), str(port), str(src), str(out)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    got = {x["id"]: x for x in json.loads(out.read_text(encoding="utf-8"))}
    assert not [x for x in got.values() if "error" in x], [x for x in got.values() if "error" in x][:3]
    return {k: {"off": v["off"], "on": v["on"]} for k, v in got.items()}


def _rec(g) -> dict:
    return {"verdict": g.verdict, "claims": [[c.kind, c.verdict, c.why] for c in g.claims]}


def _gate(mod, summary: str, diff: str) -> dict:
    return {"off": _rec(mod.gate_diff_text(summary, diff)), "on": _rec(mod.gate_diff_text(summary, diff, strict=True))}


def _new_file(path: str, lines: list) -> str:
    return (f"diff --git a/{path} b/{path}\nnew file mode 100644\nindex 0000000..27dd614\n--- /dev/null\n"
            f"+++ b/{path}\n@@ -0,0 +1,{len(lines)} @@\n" + "".join("+" + x + "\n" for x in lines))


def _edit(path: str, old: list, new: list) -> str:
    return (f"diff --git a/{path} b/{path}\nindex 1111111..2222222 100644\n--- a/{path}\n+++ b/{path}\n"
            f"@@ -1,{len(old)} +1,{len(new)} @@\n" + "".join("-" + x + "\n" for x in old)
            + "".join("+" + x + "\n" for x in new))


def _text(lines: list) -> str:
    return "\n".join(lines) + "\n"


def _want(claims: list) -> dict:
    """Both strict modes: CONTRADICTED fails; --strict also fails an UNCHECKABLE."""
    off = "FAIL" if any(c[1] == "CONTRADICTED" for c in claims) else "PASS"
    on = "FAIL" if any(c[1] in ("CONTRADICTED", "UNCHECKABLE") for c in claims) else "PASS"
    return {"off": {"verdict": off, "claims": claims}, "on": {"verdict": on, "claims": claims}}


def _tests(got: int, n: int, verdict: str | None = None) -> list:
    return ["tests_added", verdict or ("VERIFIED" if got == n else "CONTRADICTED"),
            f"diff adds {got} test functions, claim says {n}"]


def _symbol(kind: str, name: str, hit: bool) -> list:
    return ["symbol_added", "VERIFIED" if hit else "CONTRADICTED",
            f"added lines {'do' if hit else 'do NOT'} define {kind} {name!r}"]


# The two #178 reasons. Each says only what was measured: where the counted lines sit, and that a template that counts
# `def` lines cannot tell whether CPython defines a test or the name there.
WHY_178_TESTS = ("{k} of the {placed} `def test_` lines added sit in a file not read as .py or .pyi, or after an odd "
                 "number of triple quotes in the added lines of their `+++` section; this template counts `def` lines "
                 "and cannot tell whether CPython defines a test there; claim says {n} (#178)")
WHY_178_SYMBOL = ("every added line that would define {kind} {name} sits in a file not read as .py or .pyi, or after "
                  "an odd number of triple quotes in the added lines of its `+++` section; this template cannot tell "
                  "whether CPython defines it there (#178)")


def _t178(k: int, placed: int, n: int) -> list:
    return ["tests_added", "UNCHECKABLE", WHY_178_TESTS.format(k=k, placed=placed, n=n)]


def _s178(kind: str, name: str) -> list:
    return ["symbol_added", "UNCHECKABLE", WHY_178_SYMBOL.format(kind=kind, name=repr(name))]


# ---- the cases --------------------------------------------------------------------------------------------------------
# Each: id, summary, diff, `py` (the Python files' (base, head) texts, the model ast.parse judges; None where the case
# has none), `claims` (what each claim says, in record order: ("tests", n) or ("symbol", name)) and `want`.

A_TEST = ["async def test_x():", "    pass"]
A_FETCH = ["async def fetch():", "    return 1"]
D_A_TEST = _new_file("tests/test_a.py", A_TEST)
D_A_FETCH = _new_file("src/a.py", A_FETCH)
PY_A = {"tests/test_a.py": (None, _text(A_TEST)), "src/a.py": (None, _text(A_FETCH))}
DECLARED_179 = "Adds an async fetch and its test.\n\n```styxx\nadds_symbol: fetch\ntests_added: 1\n```\n"


def _case(cid, summary, diff, py, claims, want):
    return {"id": cid, "summary": summary, "diff": diff, "py": py, "claims": claims, "want": _want(want)}


CASES_179 = [
    _case("179-test", "Added 1 test.", D_A_TEST, {"tests/test_a.py": PY_A["tests/test_a.py"]}, [("tests", 1)],
          [_tests(1, 1)]),
    _case("179-fetch", "Added function fetch.", D_A_FETCH, {"src/a.py": PY_A["src/a.py"]}, [("symbol", "fetch")],
          [_symbol("function", "fetch", True)]),
    _case("179-prose", "Added 1 test. Added function fetch.", D_A_TEST + D_A_FETCH, PY_A,
          [("tests", 1), ("symbol", "fetch")], [_tests(1, 1), _symbol("function", "fetch", True)]),
    _case("179-declared", DECLARED_179, D_A_TEST + D_A_FETCH, PY_A, [("symbol", "fetch"), ("tests", 1)],
          [_symbol("function", "fetch", True), _tests(1, 1)]),       # a declaration is read in its keys' order
    _case("179-false-count", "Added 0 tests.", D_A_TEST, {"tests/test_a.py": PY_A["tests/test_a.py"]},
          [("tests", 0)], [_tests(1, 0)]),
]


def _gap(cid, line, read, lines=None):
    """`line` defines test_x; both claims are read when `read`, and neither when not."""
    lines = lines or [line, "    pass"]
    return _case(cid, "Added 1 test. Added function test_x.", _new_file("tests/test_g.py", lines),
                 {"tests/test_g.py": (None, _text(lines))}, [("tests", 1), ("symbol", "test_x")],
                 [_tests(1 if read else 0, 1), _symbol("function", "test_x", read)])


CASES_GAP = [
    _gap("gap-tab", "async\tdef test_x():", True),
    _gap("gap-form-feed", "async\x0cdef test_x():", True),
    _gap("gap-two-spaces", "async  def test_x():", True),
    _gap("gap-indented", "", True, ["class TestA:", "    async def test_x(self):", "        pass"]),
    _gap("gap-none", "asyncdef test_x():", False),                 # no gap: CPython refuses it, and it is not read
    _gap("gap-feff", "async﻿def test_x():", False),           # U+FEFF is no gap in either port (not `\s`)
]

# Named residuals of the gap, pinned as they read. A backslash continuation between `async` and `def`: the template
# reads the continued line's `def test_x` (right, by the `def` line alone). A U+FEFF lead: the port's `\s` reads it and
# the Python's does not, #205's residual, now also before `async`.
RESIDUAL_BACKSLASH = _case("residual-backslash", "Added 1 test. Added function test_x.",
                           _new_file("tests/test_g.py", ["async \\", "def test_x():", "    pass"]),
                           {"tests/test_g.py": (None, _text(["async \\", "def test_x():", "    pass"]))},
                           [("tests", 1), ("symbol", "test_x")],
                           [_tests(1, 1), _symbol("function", "test_x", True)])
FEFF_LEAD = ["x = 1", "﻿async def test_x():", "    pass"]
RESIDUAL_FEFF = {"id": "residual-feff-lead", "summary": "Added 1 test.",
                 "diff": _new_file("tests/test_g.py", FEFF_LEAD), "py": {"tests/test_g.py": (None, _text(FEFF_LEAD))},
                 "claims": [("tests", 1)],
                 "want_py": _want([_tests(0, 1)]), "want_port": _want([_tests(1, 1)])}

# A changed async test: base `async def test_x():`, head `async def test_x(y=0):`. main counts the added line, and
# PATH-2a (#101) withholds the count, since a test a removed line defines is not an added one.
CHANGED_OLD, CHANGED_NEW = ["async def test_x():", "    pass"], ["async def test_x(y=0):", "    pass"]
D_CHANGED = _edit("tests/test_a.py", CHANGED_OLD, CHANGED_NEW)
WITHHELD_101 = ("VERIFIED withheld by PATH-2a (#101): a test the added lines count is also defined in the removed "
                "lines, and a changed test is not an added one. main's reading: diff adds 1 test functions, claim "
                "says 1")
CASE_CHANGED = _case("179-changed", "Added 1 test.", D_CHANGED,
                     {"tests/test_a.py": (_text(CHANGED_OLD), _text(CHANGED_NEW))}, [("tests", 1)],
                     [["tests_added", "UNCHECKABLE", WITHHELD_101]])

FENCED = ["# Guide", "", "```python", "def test_example():", "    assert True", "```"]
SRC_OLD, SRC_NEW = ["x = 1"], ["x = 2"]
D_SRC = _edit("src/a.py", SRC_OLD, SRC_NEW)
PY_SRC = {"src/a.py": (_text(SRC_OLD), _text(SRC_NEW))}
NEW_TEST = ["def test_new():", "    assert True"]
STRING_OLD, STRING_NEW = ["x = 1"], ["x = 1", 'S = """', "def test_s():", '"""']

CASES_178 = [
    _case("178a-docs-beside-an-edit", "Added 1 test.", D_SRC + _new_file("docs/guide.md", FENCED), PY_SRC,
          [("tests", 1)], [_t178(1, 1, 1)]),
    _case("178b-docs-beside-a-new-test", "Added 1 test.",
          _new_file("tests/test_new.py", NEW_TEST) + _new_file("docs/guide.md", FENCED),
          {"tests/test_new.py": (None, _text(NEW_TEST))}, [("tests", 1)], [_t178(1, 2, 1)]),
    _case("178b-declared", "Adds a test.\n\n```styxx\ntests_added: 1\n```\n",
          _new_file("tests/test_new.py", NEW_TEST) + _new_file("docs/guide.md", FENCED),
          {"tests/test_new.py": (None, _text(NEW_TEST))}, [("tests", 1)], [_t178(1, 2, 1)]),
    _case("178c-in-a-string", "Added 1 test.", _edit("src/a.py", STRING_OLD, STRING_NEW),
          {"src/a.py": (_text(STRING_OLD), _text(STRING_NEW))}, [("tests", 1)], [_t178(1, 1, 1)]),
    _case("178-symbol-in-markdown", "Added function foo.", D_SRC + _new_file("docs/guide.md", ["def foo():", "    pass"]),
          PY_SRC, [("symbol", "foo")], [_s178("function", "foo")]),
    _case("178-notes-txt", "Added 1 test.", D_SRC + _new_file("notes.txt", ["def test_n():"]), PY_SRC,
          [("tests", 1)], [_t178(1, 1, 1)]),
    _case("178-async-in-markdown", "Added 1 test. Added class Foo.",
          D_SRC + _new_file("docs/guide.md", ["async def test_md():", "class Foo:"]), PY_SRC,
          [("tests", 1), ("symbol", "Foo")], [_t178(1, 1, 1), _s178("class", "Foo")]),
]

DOC_ONE = ['"""Module doc."""', "def test_d():", "    pass"]
DOC_TWO = ['"""Module doc,', 'on two lines."""', "", "def test_e():", "    pass", "", "class Foo:", "    pass"]
STUB = ["def test_stub() -> None: ..."]
CASES_CONTROL = [
    _case("control-plain", "Added 1 test.", _new_file("tests/test_c.py", ["def test_c():", "    pass"]),
          {"tests/test_c.py": (None, _text(["def test_c():", "    pass"]))}, [("tests", 1)], [_tests(1, 1)]),
    _case("control-docstring-one-line", "Added 1 test.", _new_file("src/m.py", DOC_ONE),
          {"src/m.py": (None, _text(DOC_ONE))}, [("tests", 1)], [_tests(1, 1)]),
    _case("control-docstring-two-lines", "Added 1 test. Added class Foo.", _new_file("src/m.py", DOC_TWO),
          {"src/m.py": (None, _text(DOC_TWO))}, [("tests", 1), ("symbol", "Foo")],
          [_tests(1, 1), _symbol("class", "Foo", True)]),
    _case("control-pyi", "Added 1 test.", _new_file("src/m.pyi", STUB), {"src/m.pyi": (None, _text(STUB))},
          [("tests", 1)], [_tests(1, 1)]),
    _case("control-false-count", "Added 2 tests.", _new_file("tests/test_c.py", ["def test_c():", "    pass"]),
          {"tests/test_c.py": (None, _text(["def test_c():", "    pass"]))}, [("tests", 2)], [_tests(1, 2)]),
]

# The path's tail: `+++ b/src/a.py<ch>` beside a plain test file. Python's strip() removes U+001F, U+0085, U+00A0 and
# U+3000 and keeps U+007F, U+200B and U+FEFF; the port's trim() keeps U+001F, U+0085, U+007F and U+200B and removes
# U+00A0, U+3000 and U+FEFF. #178 reads the tail with every trailing code point at or below U+0020 or from U+007F up
# removed, so both ports read the file as Python source and print the same record.
TAIL_CHARS = ["\x1f", "\x85", "\x7f", "​", "﻿", "\xa0", "　"]
CASES_TAIL = [
    _case(f"tail-U+{ord(ch):04X}", "Added 2 tests.",
          _new_file("tests/test_ok.py", ["def test_ok():", "    pass"]) + _new_file("src/a.py" + ch, ["def test_s():"]),
          None, [("tests", 2)], [_tests(2, 2)])
    for ch in TAIL_CHARS
]

# Named residual of #178: a string opened in a line the diff does not show is not seen, so a `def test_` added inside
# an existing docstring is still counted, as on main (a wrong VERIFIED, pinned).
RESIDUAL_OPEN_BEFORE = ['S = """', "x", '"""']
RESIDUAL_OPEN_AFTER = ['S = """', "x", "def test_s():", '"""']
RESIDUAL_STRING = _case(
    "residual-string-opened-out-of-view", "Added 1 test.",
    "diff --git a/src/a.py b/src/a.py\nindex 1111111..2222222 100644\n--- a/src/a.py\n+++ b/src/a.py\n"
    "@@ -2,2 +2,3 @@\n x\n+def test_s():\n \"\"\"\n",
    {"src/a.py": (_text(RESIDUAL_OPEN_BEFORE), _text(RESIDUAL_OPEN_AFTER))}, [("tests", 1)], [_tests(1, 1)])



def _sec(path: str, lines: list) -> str:
    """One `+++` section editing `path`: a context line, then `lines` added."""
    return (f"diff --git a/{path} b/{path}\nindex 1111111..2222222 100644\n--- a/{path}\n+++ b/{path}\n"
            f"@@ -1,1 +1,{len(lines) + 1} @@\n x = 0\n" + "".join("+" + x + "\n" for x in lines))


# The sections a key of `parse_unified_diff_sides` would merge (review 1 of fix batch 1): `tests/test_a.py<ch>` and
# `tests/test_a.py`, where Python's strip() removes U+0085, U+001C and U+001F and the port's trim() keeps them, and the
# port's trim() removes U+FEFF and Python's strip() keeps it; U+1C89 and U+1C8A, which Node 24 lower-cases together and
# CPython 3.12 does not; an ASCII case pair, which both lower-case together; and one path named twice. Read per key, the
# first section's open `"""` put the second section's `def test_x` after an odd count in the port that merged them
# only (13 of 37 such inputs parted the two ports' records and gates); read per section, both ports read each alike.
SECTIONS_OPEN = [_sec("tests/test_a.py{ch}", ['S = """']), _sec("tests/test_a.py", ["def test_x(): pass"])]
SECTIONS_CLOSE = [_sec("tests/test_a.py{ch}", ['S = """', "x"]), _sec("tests/test_a.py", ['"""', "def test_x(): pass"])]
SECTION_CHARS = {"U+0085": "\x85", "U+001F": "\x1f", "U+001C": "\x1c", "U+FEFF": "\ufeff"}
CASES_SECTIONS = [
    _case(f"sections-open-{tag}", "Added 2 tests. Added function test_x.", "".join(SECTIONS_OPEN).replace("{ch}", ch),
          None, [("tests", 2), ("symbol", "test_x")], [_tests(1, 2), _symbol("function", "test_x", True)])
    for tag, ch in SECTION_CHARS.items()
] + [
    # the string the first section opens is closed in the second, and the `def` after it is read after an odd count
    # in its own section: #178 abstains in both ports (main read VERIFIED; the merging port read it so too)
    _case(f"sections-close-{tag}", "Added 1 test. Added function test_x.", "".join(SECTIONS_CLOSE).replace("{ch}", ch),
          None, [("tests", 1), ("symbol", "test_x")], [_t178(1, 1, 1), _s178("function", "test_x")])
    for tag, ch in SECTION_CHARS.items()
] + [
    _case(f"sections-{tag}", "Added 2 tests. Added function test_x.",
          _sec(p1, ['S = """']) + _sec(p2, ["def test_x(): pass"]), None, [("tests", 2), ("symbol", "test_x")],
          [_tests(1, 2), _symbol("function", "test_x", True)])
    for tag, (p1, p2) in {"lowering-U+1C89": ("tests/\u1c89.py", "tests/\u1c8a.py"),
                          "case-pair": ("tests/A.py", "tests/a.py"),
                          "one-path-twice": ("tests/test_a.py", "tests/test_a.py")}.items()
]

# Named residuals of #178's parity and path test (review 1 of fix batch 1), each pinned as it reads, both ports alike:
# - the parity counts three double and three single quotes together, so a `"""` line inside a `'''` string flips it:
#   the `def test_s` inside the string is read as source, a wrong VERIFIED, as on main;
# - the tail rule removes any trailing code point from U+007F up, so `tests/test_a.py<U+00E9>` is read as Python source
#   and main's count stands there, as on main;
# - #169's and #168's header misreads carry into the path test: a path git quotes under its default core.quotePath
#   (any name outside ASCII) and a GNU or svn header carrying a timestamp are read as not .py, so a true count over a
#   `def test_` there beside a plain .py file reads UNCHECKABLE (#178), where main read VERIFIED.
MIXED = ["S = '''", '"""', "def test_s():", "'''"]
RESIDUAL_MIXED_QUOTES = _case("residual-mixed-quotes", "Added 1 test. Added function test_s.",
                              _new_file("tests/test_s.py", MIXED), {"tests/test_s.py": (None, _text(MIXED))},
                              [("tests", 1), ("symbol", "test_s")],
                              [_tests(1, 1), _symbol("function", "test_s", True)])
RESIDUAL_TAIL_ABOVE_7F = _case("residual-tail-above-7f", "Added 1 test.",
                               _new_file("tests/test_a.py\u00e9", ["def test_e():"]), None, [("tests", 1)],
                               [_tests(1, 1)])
RESIDUAL_QUOTED_PATH = _case(
    "residual-quoted-path-169", "Added 2 tests.",
    'diff --git "a/tests/t\\303\\251st.py" "b/tests/t\\303\\251st.py"\nnew file mode 100644\nindex 0000000..1\n'
    '--- /dev/null\n+++ "b/tests/t\\303\\251st.py"\n@@ -0,0 +1,1 @@\n+def test_q(): pass\n'
    + _new_file("tests/test_n.py", ["def test_n(): pass"]), None, [("tests", 2)], [_t178(1, 2, 2)])
RESIDUAL_SVN_TIMESTAMP = _case(
    "residual-svn-timestamp-168", "Added 2 tests.",
    "Index: x.py\n===================================================================\n--- x.py\t(revision 1)\n"
    "+++ x.py\t(working copy)\n@@ -0,0 +1,1 @@\n+def test_x(): pass\n"
    + _new_file("tests/test_n.py", ["def test_n(): pass"]), None, [("tests", 2)], [_t178(1, 2, 2)])
RESIDUALS_178 = [RESIDUAL_MIXED_QUOTES, RESIDUAL_TAIL_ABOVE_7F, RESIDUAL_QUOTED_PATH, RESIDUAL_SVN_TIMESTAMP]

ALL_ALIKE = CASES_179 + CASES_GAP + [RESIDUAL_BACKSLASH, CASE_CHANGED] + CASES_178 + CASES_CONTROL + CASES_TAIL + \
    [RESIDUAL_STRING] + CASES_SECTIONS + RESIDUALS_178
RESIDUALS = {RESIDUAL_STRING["id"], RESIDUAL_FEFF["id"], RESIDUAL_MIXED_QUOTES["id"]}   # what the ast grid names


def _problems(mod, cases: list) -> list:
    return [(c["id"], got) for c in cases if (got := _gate(mod, c["summary"], c["diff"])) != c["want"]]


def _port_problems(cases: list, got: dict) -> list:
    return [(c["id"], got[c["id"]]) for c in cases if got[c["id"]] != c["want"]]


# ---- 1. #179 ----------------------------------------------------------------------------------------------------------

def test_179_the_truth_of_the_reporters_cases():
    """CPython reads an AsyncFunctionDef for each, and one test."""
    for path, (_b, head) in PY_A.items():
        nodes = [n for n in ast.walk(ast.parse(head)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        assert [type(n).__name__ for n in nodes] == ["AsyncFunctionDef"], path


def test_179_the_reporters_cases_python():
    assert _problems(N, CASES_179) == []


def test_179_the_reporters_cases_port(tmp_path):
    got = _port(CASES_179, tmp_path)
    assert _port_problems(CASES_179, got) == []
    assert all(got[c["id"]] == _gate(N, c["summary"], c["diff"]) for c in CASES_179)


def _git(d: Path, *a) -> str:
    return subprocess.run([GIT, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false",
                           "-c", "core.longpaths=true", *a], cwd=d, capture_output=True,
                          check=True).stdout.decode("utf-8")


def _repo(tmp_path: Path, files: dict) -> Path:
    """A two-commit repository: an empty base, and a head adding `files`."""
    if GIT is None:
        _missing("git", "the git door half of this check")
    d = tmp_path / "repo"
    d.mkdir()
    _git(d, "init", "-q")
    _git(d, "config", "core.autocrlf", "false")
    _git(d, "config", "core.longpaths", "true")
    _git(d, "commit", "-q", "--allow-empty", "-m", "base")
    for rel, text in files.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_bytes(text.encode("utf-8"))
    _git(d, "add", "-A")
    _git(d, "commit", "-q", "-m", "head")
    return d


def test_179_the_git_door(tmp_path):
    d = _repo(tmp_path, {p: head for p, (_b, head) in PY_A.items()})
    for c in CASES_179:
        got = {"off": _rec(N.gate_diff(c["summary"], d, "HEAD~1", "HEAD")),
               "on": _rec(N.gate_diff(c["summary"], d, "HEAD~1", "HEAD", strict=True))}
        assert got == c["want"], c["id"]


def test_179_the_gap_after_async_python():
    assert _problems(N, CASES_GAP + [RESIDUAL_BACKSLASH]) == []


def test_179_the_gap_after_async_port(tmp_path):
    cases = CASES_GAP + [RESIDUAL_BACKSLASH, RESIDUAL_FEFF]
    got = _port(cases, tmp_path)
    assert _port_problems(CASES_GAP + [RESIDUAL_BACKSLASH], got) == []
    for c in CASES_GAP + [RESIDUAL_BACKSLASH]:
        assert got[c["id"]] == _gate(N, c["summary"], c["diff"]), c["id"]
    # the residual U+FEFF lead, as it stands: the port's \s reads it, the Python's does not
    assert _gate(N, RESIDUAL_FEFF["summary"], RESIDUAL_FEFF["diff"]) == RESIDUAL_FEFF["want_py"]
    assert got[RESIDUAL_FEFF["id"]] == RESIDUAL_FEFF["want_port"]


def test_179_a_changed_async_test_stays_withheld_python():
    assert _problems(N, [CASE_CHANGED]) == []


def test_179_a_changed_async_test_stays_withheld_port(tmp_path):
    got = _port([CASE_CHANGED], tmp_path)
    assert got[CASE_CHANGED["id"]] == CASE_CHANGED["want"]


# ---- 2. #178 ----------------------------------------------------------------------------------------------------------

def test_178_the_reporters_cases_python():
    assert _problems(N, CASES_178) == []


def test_178_the_reporters_cases_port(tmp_path):
    got = _port(CASES_178, tmp_path)
    assert _port_problems(CASES_178, got) == []


def test_178_controls_python():
    assert _problems(N, CASES_CONTROL) == []


def test_178_controls_port(tmp_path):
    got = _port(CASES_CONTROL, tmp_path)
    assert _port_problems(CASES_CONTROL, got) == []


def _one_parse_cases() -> list:
    """Shapes where `parse_unified_diff` reads an added `def` line the per-file parse does not place (#172's): an empty
    line inside a hunk read as context, so a `--- ` content line ends the file; and a `+++:` header line."""
    w1 = next(x for x in R.repro_cases() if x["id"] == "path2:w1-an-empty-line-inside-a-hunk-is-a-context-line")
    colon = ("diff --git a/tests/test_x.py b/tests/test_x.py\nindex 1111111..2222222 100644\n--- a/tests/test_x.py\n"
             "+++: b/tests/test_x.py\n@@ -1,1 +1,2 @@\n x = 1\n+def test_a():\n")
    return [{"id": "w1", "summary": w1["summary"], "diff": w1["diff"]},
            {"id": "plus-colon", "summary": "Added 1 test. Added function test_a.", "diff": colon}]


def test_178_one_parse(tmp_path):
    """#178 compares two counts of one parse, so it does not fire where the two parse functions place a line apart:
    those shapes are #172's, and the reason #178 gives would be false there."""
    cases = _one_parse_cases()
    got = _port(cases, tmp_path)
    for c in cases:
        for rec in (_gate(N, c["summary"], c["diff"]), got[c["id"]]):
            assert not any("(#178)" in x[2] for k in ("off", "on") for x in rec[k]["claims"]), (c["id"], rec)


def test_178_the_path_tail(tmp_path):
    assert _problems(N, CASES_TAIL) == []
    got = _port(CASES_TAIL, tmp_path)
    assert _port_problems(CASES_TAIL, got) == []


def test_178_named_residual(tmp_path):
    """A string opened in a line the diff does not show is not seen: the count reads the `def test_` as on main."""
    assert _problems(N, [RESIDUAL_STRING]) == []
    assert _port([RESIDUAL_STRING], tmp_path)[RESIDUAL_STRING["id"]] == RESIDUAL_STRING["want"]


def test_178_the_sections_a_key_would_merge(tmp_path):
    """The parity and the path test are a `+++` section's: both ports print the same record where their keys of
    `parse_unified_diff_sides` merge two sections apart."""
    assert _problems(N, CASES_SECTIONS) == []
    got = _port(CASES_SECTIONS, tmp_path)
    assert _port_problems(CASES_SECTIONS, got) == []


def test_178_named_residuals_of_the_parity_and_the_path(tmp_path):
    """Mixed triple quotes, a tail from U+007F up, a quoted path (#169) and an svn timestamp (#168): pinned as read."""
    assert _problems(N, RESIDUALS_178) == []
    got = _port(RESIDUALS_178, tmp_path)
    assert _port_problems(RESIDUALS_178, got) == []


SECTIONS_RUNNER = r"""
"use strict";
const fs = require("fs");
const vm = require("vm");
const [port, inp, out] = process.argv.slice(2);
const P = vm.runInContext(fs.readFileSync(port, "utf8") + "\n;({_diffSections, parseUnifiedDiffSides})",
                          vm.createContext({}), { filename: port });
const res = JSON.parse(fs.readFileSync(inp, "utf8")).map(d => {
  const a = P._diffSections(d).flatMap(([, x]) => x).sort();
  const b = [...P.parseUnifiedDiffSides(d).values()].flatMap(([x]) => x).sort();
  return JSON.stringify(a) === JSON.stringify(b);
});
fs.writeFileSync(out, JSON.stringify(res));
"""


def test_178_the_walk_places_every_added_line_as_the_parse(tmp_path):
    """On every committed input `_diff_sections` holds the added lines `parse_unified_diff_sides` holds, as a multiset,
    in each port: the two counts #178 compares are of one parse's lines, grouped per section."""
    import collections
    rows = R.inputs()
    assert len(rows) == 6672
    bad = [R.uid(i, r) for i, r in enumerate(rows)
           if collections.Counter(x for _p, a in N._diff_sections(r[3]) for x in a)
           != collections.Counter(x for a, _r in N.parse_unified_diff_sides(r[3]).values() for x in a)]
    assert bad == []
    if NODE is None:
        _missing("node", "the port half of this check")
    runner, src, out = tmp_path / "sections.js", tmp_path / "sections_in.json", tmp_path / "sections_out.json"
    runner.write_text(SECTIONS_RUNNER, encoding="utf-8")
    src.write_text(json.dumps([r[3] for r in rows], ensure_ascii=True), encoding="utf-8")
    r = subprocess.run([NODE, str(runner), str(R.PORT), str(src), str(out)], capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    same = json.loads(out.read_text(encoding="utf-8"))
    assert len(same) == len(rows) and [R.uid(i, rows[i]) for i, x in enumerate(same) if not x] == []


def test_the_two_ports_print_the_same_records(tmp_path):
    got = _port(ALL_ALIKE, tmp_path)
    assert [c["id"] for c in ALL_ALIKE if got[c["id"]] != _gate(N, c["summary"], c["diff"])] == []


# ---- 3. the ast grid -------------------------------------------------------------------------------------------------

def _defs(src):
    if src is None:
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    return [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]


def _truth(case: dict, claim: tuple):
    """What ast.parse of the case's Python files says of one claim; None where a file does not parse."""
    bases = [_defs(b) for b, _h in case["py"].values()]
    heads = [_defs(h) for _b, h in case["py"].values()]
    if any(x is None for x in bases + heads):
        return None
    kind, arg = claim
    if kind == "tests":
        added = sum(n.startswith("test_") for h in heads for n in h) - sum(n.startswith("test_") for b in bases for n in b)
        return added == arg
    return any(arg in h for h in heads) and not any(arg in b for b in bases)


def _refuted(case: dict, rec: dict) -> list:
    out = []
    for claim, x in zip(case["claims"], rec["off"]["claims"]):
        t = _truth(case, claim)
        if (x[1] == "VERIFIED" and t is False) or (x[1] == "CONTRADICTED" and t is True):
            out.append((case["id"], claim, x[1]))
    return out


def test_the_ast_grid(tmp_path):
    """For every case with a Python file model, no port prints a VERIFIED or a CONTRADICTED that ast.parse refutes,
    except the named residuals."""
    cases = [c for c in ALL_ALIKE + [RESIDUAL_FEFF] if c["py"] is not None]
    assert len(cases) >= 25
    got = _port(cases, tmp_path)
    refuted = []
    for c in cases:
        refuted += _refuted(c, _gate(N, c["summary"], c["diff"])) + _refuted(c, got[c["id"]])
    assert {r[0] for r in refuted} <= RESIDUALS, refuted
    assert RESIDUAL_STRING["id"] in {r[0] for r in refuted}          # the residual is a residual: ast refutes it
    assert RESIDUAL_MIXED_QUOTES["id"] in {r[0] for r in refuted}


# ---- 4. plants: each change undone, in each port, each told by a check above ------------------------------------------

def _py_plant(old: str, new: str, after: str, name: str):
    text = R.lf(R.INSTRUMENT)
    i = text.index(old, text.index(after))
    return R.module_from(text[:i] + new + text[i + len(old):], "_p178_" + name)


PY_PLANTS = {
    "#179: async dropped from the count": ('tx = r"^\\s*(?:async[ \\t\\f]+)?def test_"', 'tx = r"^\\s*def test_"',
                                          "def _gate(", CASES_179),
    "#179: async dropped from the symbol pattern": ("(?:async[ \\t\\f]+def|def|class)", "(?:def|class)", "def _gate(",
                                                   CASES_179),
    "#179: the overlay's mirror dropped": ('        if line.startswith("async", r):', "        if False:",
                                          "def _p2a_counted(", [CASE_CHANGED]),
    "#178: dropped from the count": ("if inside != placed:", "if False:", "def _gate(", CASES_178),
    "#178: dropped from the symbol": ("if re.search(pat, sides_blob, re.M) and not re.search(pat, inside_blob, re.M):",
                                      "if False:", "def _gate(", CASES_178),
    "#178: the path's tail read as is": ("while e and (ord(p[e - 1]) <= 32 or ord(p[e - 1]) >= 127):", "while False:",
                                         "def _python_source_lines(", CASES_TAIL),
    "#178: the parity per key, not per section": (
        'sections=_diff_sections(diff_text or "")',
        'sections=[(p, a) for p, (a, _r) in parse_unified_diff_sides(diff_text or "").items()]',
        "def gate_diff_text(", CASES_SECTIONS),
}


@pytest.mark.parametrize("name", sorted(PY_PLANTS))
def test_a_planted_python_slip_is_told(name):
    old, new, after, cases = PY_PLANTS[name]
    assert R.lf(R.INSTRUMENT).count(old) == 1, old
    mod = _py_plant(old, new, after, "x" + str(sorted(PY_PLANTS).index(name)))
    assert _problems(mod, cases) != [], name


JS_PLANTS = {
    "#179: async dropped from the count": ("const tx = /^\\s*(?:async[ \\t\\f]+)?def test_/gm;",
                                          "const tx = /^\\s*def test_/gm;", CASES_179),
    "#179: async dropped from the symbol pattern": ("(?:async[ \\\\t\\\\f]+def|def|class)", "(?:def|class)", CASES_179),
    "#179: the overlay's mirror dropped": ("      if (q > r + 5) s = q;\n", "", [CASE_CHANGED]),
    "#178: dropped from the count": ("if (inside !== placed) {", "if (false) {", CASES_178),
    "#178: dropped from the symbol": ("if (pat.test(sidesBlob) && !pat.test(insideBlob)) {", "if (false) {", CASES_178),
    "#178: the path's tail read as is": (
        "    while (e > 0 && (p.charCodeAt(e - 1) <= 32 || p.charCodeAt(e - 1) >= 127)) e--;\n", "", CASES_TAIL),
    "#178: the parity per key, not per section": ("const sections = _diffSections(diffText);",
                                                  "const sections = [...sides].map(([p, [a]]) => [p, a]);",
                                                  CASES_SECTIONS),
}


@pytest.mark.parametrize("name", sorted(JS_PLANTS))
def test_a_planted_port_slip_is_told(name, tmp_path):
    old, new, cases = JS_PLANTS[name]
    text = R.lf(R.PORT)
    assert text.count(old) == 1, old
    planted = tmp_path / "diffgate_planted.js"
    planted.write_bytes(text.replace(old, new).encode("utf-8"))
    got = _port(cases, tmp_path, planted, "planted")
    assert _port_problems(cases, got) != [], name
