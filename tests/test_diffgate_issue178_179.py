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
opposite gates (review 1 of fix batch 1). The tests count abstains only on a claim at or below the larger of its
two counts (the one-sided #178, options judgment of fix batch 1): above every count the template takes, the claim
differs from each of them and from the same counts with changed tests dropped, so main's CONTRADICTED stands there;
below the Python-source count it still abstains, since dropping a changed test can reach the claim there
(p2a-101:20260930:111, where deciding below the range printed a false CONTRADICTED the overlay does not withhold).
Above every count the verdict is main's, on main's counts: where main counts fewer `def` lines than CPython defines
(two spaces after `def`), a true count there reads CONTRADICTED as on main, a named residual pinned here.
The path's tail is read with every trailing code point at or below U+0020 or
from U+007F up removed, and a `+++` header is read as /dev/null on that header with every such code point removed from
either end (`_edge_cut` / `_edgeCut`, review 2 of fix batch 1): every code point strip() or trim() removes is in that
set, so the two ports reach the same path tail where strip() and trim() part (held here over every code point). Read
after strip() and trim(), `+++ /dev/null<U+0085>` was a deletion in Python only and `+++ /dev/null<U+FEFF>` in the
port only, and on a false count the gates parted without --strict. The abstention adds no VERIFIED and no
CONTRADICTED.

#205's seams and the duplicate name (review 3 of fix batch 1). The Python's `\\s` and the port's part on U+FEFF (and on
U+001C to U+001F and U+0085, which #205's _READ_APART holds), strip() and trim() part on the same code points at a
`---` or `+++` header's edge, and the Python's `\\b` ends a name by Unicode where the port's ends it by ASCII. On main
those seams parted the two ports' verdicts where a counted line or a header held one (`RESIDUAL_APART_CHAR` in
test_diffgate_issue205_206.py), and #179's `async def` read reached them on more lines: over a new file saved with a
byte order mark whose line 1 is `async def test_x():`, CPython defines test_x, the CLI read "Added 1 test."
CONTRADICTED and the bookmarklet VERIFIED, so the gates parted without --strict. Both counts now abstain, in both
ports (UNCHECKABLE, #205), where a header holds one of these at an edge of its path, where a line either port's test
count reads holds one before its `def`, and, for the symbol claim, where a line one port reads as defining the name
holds one before the name or runs the name on into a code point from U+0080 up, and no line is one both ports read so.
Both ports read these lines with the same code-point sets, not with `\\s`, `\\b`, strip() or trim(). And where two
lines of one `+++` section the test count reads define one name, one of them after `async`, the count abstains
(UNCHECKABLE, #179): CPython binds the name once in a scope, and #179's read reached main's duplicate-definition limit
there (two right CONTRADICTEDs of main became wrong VERIFIEDs on the truth world). Two `def` lines of one name keep
main's count, a residual pinned here.

What is held here: the reporters' cases at the raw door (both ports, both strict modes, the two ports' records equal,
reasons included) and at the git door (a real two-commit repository; skipped without git, failed under CI); the gap
after `async`; the changed async test the overlay withholds; the controls #178 must not touch; the two sides of the
tests count's range; the one-parse rule; the path's tail; the `+++ /dev/null` header and the set it is read on; the
sections a key would merge; the walk placing every added line as the parse does; #205's seams, at a header's edge,
before a `def` and after a name, and the line both ports read that keeps main's verdict; the name defined twice; the
named residuals, pinned as they read; an ast grid over every case with a Python file model; and a plant of each
change in each port, each told by a check here.
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

# Named residual of the gap, pinned as it reads. A backslash continuation between `async` and `def`: the template reads
# the continued line's `def test_x` (right, by the `def` line alone). A U+FEFF lead before `async`, which the port's `\s`
# reads and the Python's does not (#205's residual, which #179's read reached), abstains in both ports: SEAM_FEFF_LEAD.
RESIDUAL_BACKSLASH = _case("residual-backslash", "Added 1 test. Added function test_x.",
                           _new_file("tests/test_g.py", ["async \\", "def test_x():", "    pass"]),
                           {"tests/test_g.py": (None, _text(["async \\", "def test_x():", "    pass"]))},
                           [("tests", 1), ("symbol", "test_x")],
                           [_tests(1, 1), _symbol("function", "test_x", True)])
FEFF_LEAD = ["x = 1", "﻿async def test_x():", "    pass"]

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

# The two sides of the tests count's range (the one-sided #178). Above every count main's verdict stands; at or below
# the larger count #178 abstains, and below the Python-source count too: there a changed test beside a `def test_` in
# a string makes a true "Added 0 tests." read CONTRADICTED on main's count, which the overlay does not withhold
# (p2a-101:20260930:111's shape: main counts 2, the Python source 1, and the counts with the changed test dropped 1
# and 0).
SIDE_OLD = ["def test_c():", "    pass"]
SIDE_NEW = ["def test_c(x=0):", "    pass", 'DOC = """', "def test_a():", '"""']
CASES_178_SIDES = [
    _case("178-above-every-count", "Added 0 tests. Added 1 test. Added 2 tests. Added 3 tests.",
          _new_file("tests/test_new.py", NEW_TEST) + _new_file("docs/guide.md", FENCED),
          {"tests/test_new.py": (None, _text(NEW_TEST))}, [("tests", 0), ("tests", 1), ("tests", 2), ("tests", 3)],
          [_t178(1, 2, 0), _t178(1, 2, 1), _t178(1, 2, 2), _tests(2, 3)]),
    _case("178-below-the-python-count-beside-a-changed-test", "Added 0 tests.",
          _edit("tests/test_a.py", SIDE_OLD, SIDE_NEW), {"tests/test_a.py": (_text(SIDE_OLD), _text(SIDE_NEW))},
          [("tests", 0)], [_t178(1, 2, 0)]),
]
# Its named residual, pinned as it reads: above every count main's verdict stands on main's counts, so where main
# counts fewer `def test_` lines than CPython defines (two spaces after `def`, which `def test_` does not read), a true
# count above them reads CONTRADICTED, as on main, and as at the head on the same file without the docs file beside it.
UNDER = ["def  test_a():", "    pass", "def  test_b():", "    pass"]
RESIDUAL_UNDERCOUNT_ABOVE = _case("residual-undercount-above-every-count", "Added 2 tests.",
                                  _new_file("tests/test_a.py", UNDER) + _new_file("docs/guide.md", FENCED),
                                  {"tests/test_a.py": (None, _text(UNDER))}, [("tests", 2)], [_tests(1, 2)])

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

# The same residual reached through `async def` (review 3 of fix batch 1), pinned as it reads, at both doors, both ports
# alike. main did not count an `async def` line, so where the line sits inside a string opened in a context line main
# read right by accident, and #179's read makes that verdict wrong either way. A pytest-plugin test adds an
# `async def test_b():` inside `pytester.makepyfile("""...""")`, whose `"""` opens above the hunk, and a real
# `def test_new`: ast of the head file defines test_runs and test_new, so one test is added. "Added 1 test." (true)
# reads CONTRADICTED ("diff adds 2"), which FAILs without --strict, where main read VERIFIED; "Added 2 tests." (false)
# reads VERIFIED, where main read CONTRADICTED. The symbol twin: an `async def fetch():` added inside a `SNIPPET = """`
# string opened in a context line reads "Added function fetch." VERIFIED (false; main read CONTRADICTED). No added-line
# parity sees a string the hunk does not open; #161's A-1 shape (an added `async def test_` abstains the count) would
# avoid the class and lose #179's VERIFIEDs, the operator's choice.
Q3 = '"' * 3
PYTESTER_BASE = ["def test_runs(pytester):", "    pytester.makepyfile(", "        " + Q3, "        import pytest", "",
                 "        @pytest.mark.asyncio", "        async def test_a():", "            pass", "        " + Q3, "    )",
                 "    pytester.runpytest().assert_outcomes(passed=1)"]
PYTESTER_HEAD = PYTESTER_BASE[:8] + ["", "        @pytest.mark.asyncio", "        async def test_b():", "            pass"] \
    + PYTESTER_BASE[8:10] + ["    pytester.runpytest().assert_outcomes(passed=2)", "", "", "def test_new(pytester):",
                             "    assert True"]
D_PYTESTER = (
    "diff --git a/tests/test_plugin.py b/tests/test_plugin.py\nindex 1111111..2222222 100644\n"
    "--- a/tests/test_plugin.py\n+++ b/tests/test_plugin.py\n@@ -6,6 +6,14 @@\n         @pytest.mark.asyncio\n"
    "         async def test_a():\n             pass\n+\n+        @pytest.mark.asyncio\n+        async def test_b():\n"
    "+            pass\n         " + Q3 + "\n     )\n-    pytester.runpytest().assert_outcomes(passed=1)\n"
    "+    pytester.runpytest().assert_outcomes(passed=2)\n+\n+\n+def test_new(pytester):\n+    assert True\n")
PY_PYTESTER = {"tests/test_plugin.py": (_text(PYTESTER_BASE), _text(PYTESTER_HEAD))}
SNIPPET_BASE = ["SNIPPET = " + Q3, "import asyncio", Q3, "", "def run():", "    pass"]
SNIPPET_HEAD = ["SNIPPET = " + Q3, "import asyncio", "", "async def fetch():", "    pass", Q3, "", "def run():",
                "    pass"]
D_SNIPPET = ("diff --git a/src/snippets.py b/src/snippets.py\nindex 1111111..2222222 100644\n--- a/src/snippets.py\n"
             "+++ b/src/snippets.py\n@@ -1,5 +1,8 @@\n SNIPPET = " + Q3 + "\n import asyncio\n+\n+async def fetch():\n"
             "+    pass\n " + Q3 + "\n \n def run():\n")
RESIDUALS_STRING_ASYNC = [
    _case("residual-async-in-a-string-opened-out-of-view-true", "Added 1 test.", D_PYTESTER, PY_PYTESTER,
          [("tests", 1)], [_tests(2, 1)]),
    _case("residual-async-in-a-string-opened-out-of-view-false", "Added 2 tests.", D_PYTESTER, PY_PYTESTER,
          [("tests", 2)], [_tests(2, 2)]),
    _case("residual-async-symbol-in-a-string-opened-out-of-view", "Added function fetch.", D_SNIPPET,
          {"src/snippets.py": (_text(SNIPPET_BASE), _text(SNIPPET_HEAD))}, [("symbol", "fetch")],
          [_symbol("function", "fetch", True)]),
]

# Named residuals of the parity (review 3 of fix batch 1), pinned as they read, both ports alike: it counts `"""` and
# `'''` as substrings, not as tokens, so a `"""` in a comment or inside a one-quote string turns it, and a one-quote
# string continued by a backslash holds no triple quote at all. Each puts a `def test_s` inside a string where the count
# reads it as source: "Added 1 test." reads VERIFIED, as on main, where ast of the head file defines no function.
# Through `async def` each is finding 1's class above: main read CONTRADICTED there.
PARITY_SHAPES = {
    "comment-quote": ["x = 1", "# a " + Q3 + " in a comment", "S = " + Q3, "def test_s():", Q3],
    "one-quote-string": ["x = 1", "Q = '" + Q3 + "'", "S = " + Q3, "def test_s():", Q3],
    "backslash-continued-string": ["x = 1", 'S = "abc\\', 'def test_s():"'],
}
RESIDUALS_PARITY = [
    _case(f"residual-parity-{tag}", "Added 1 test.", _edit("src/a.py", ["x = 1"], lines),
          {"src/a.py": (_text(["x = 1"]), _text(lines))}, [("tests", 1)], [_tests(1, 1)])
    for tag, lines in PARITY_SHAPES.items()
]



def _sec(path: str, lines: list) -> str:
    """One `+++` section editing `path`: a context line, then `lines` added."""
    return (f"diff --git a/{path} b/{path}\nindex 1111111..2222222 100644\n--- a/{path}\n+++ b/{path}\n"
            f"@@ -1,1 +1,{len(lines) + 1} @@\n x = 0\n" + "".join("+" + x + "\n" for x in lines))


# The sections a key of `parse_unified_diff_sides` would merge (review 1 of fix batch 1): `tests/test_a.py<ch>` and
# `tests/test_a.py`, where Python's strip() removes U+0085, U+001C and U+001F and the port's trim() keeps them, and the
# port's trim() removes U+FEFF and Python's strip() keeps it; U+1C89 and U+1C8A, which Node 24 lower-cases together and
# CPython 3.12 does not; an ASCII case pair, which both lower-case together; and one path named twice. Read per key,
# one section's open `"""` put the next section's `def test_x` after an odd count in the port that merged them only
# (of 43 inputs review 1 built around such keys, 18 parted the two ports' records, 3 their gates without --strict and
# 15 with it); read per section, both ports read each alike.
SECTIONS_OPEN = [_sec("tests/test_a.py{ch}", ['S = """']), _sec("tests/test_a.py", ["def test_x(): pass"])]
SECTIONS_CLOSE = [_sec("tests/test_a.py{ch}", ['S = """', "x"]), _sec("tests/test_a.py", ['"""', "def test_x(): pass"])]
SECTION_CHARS = {"U+0085": "\x85", "U+001F": "\x1f", "U+001C": "\x1c", "U+FEFF": "\ufeff"}
CASES_SECTIONS = [
    _case(f"sections-open-{tag}", "Added 2 tests. Added function test_x.", "".join(SECTIONS_OPEN).replace("{ch}", ch),
          None, [("tests", 2), ("symbol", "test_x")], [_tests(1, 2), _symbol("function", "test_x", True)])
    for tag, ch in SECTION_CHARS.items()
] + [
    # the string one section opens is closed in the next, and the `def` after it is read after an odd count
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
# (beside a plain .py file, so #110's abstention, which reads no tail rule, is not what reads it)
RESIDUAL_TAIL_ABOVE_7F = _case("residual-tail-above-7f", "Added 2 tests.",
                               _new_file("tests/test_n.py", ["def test_n(): pass"])
                               + _new_file("tests/test_a.py\u00e9", ["def test_e():"]), None, [("tests", 2)],
                               [_tests(2, 2)])
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
# Named residual of #178's parity (review 2 of fix batch 1), pinned as it reads, at both doors, both ports alike: the
# parity is taken over a section's added lines, so an added line holding one end of a string whose other end is a
# context line turns it for the rest of the section. An edited docstring closing line (`    Longer text, edited."""`,
# its opening line a context line) and then an added `def test_b():` puts the `def` after an odd count, so a true count
# and a true symbol read UNCHECKABLE (#178; FAIL under --strict), where main read VERIFIED. The converse, a string
# opened out of view, is RESIDUAL_STRING above.
EDIT_BASE = ["def test_a():", '    """Check a.', "", '    Longer text."""', "    assert True"]
EDIT_HEAD = ["def test_a():", '    """Check a.', "", '    Longer text, edited."""', "    assert True", "", "",
             "def test_b():", "    assert True"]
RESIDUAL_DOCSTRING_EDIT = _case(
    "residual-docstring-edit-turns-the-parity", "Added 1 test. Added function test_b.",
    "diff --git a/tests/test_m.py b/tests/test_m.py\nindex 92a06e4..c130441 100644\n--- a/tests/test_m.py\n"
    "+++ b/tests/test_m.py\n@@ -1,5 +1,9 @@\n def test_a():\n     \"\"\"Check a.\n \n-    Longer text.\"\"\"\n"
    "+    Longer text, edited.\"\"\"\n+    assert True\n+\n+\n+def test_b():\n     assert True\n",
    {"tests/test_m.py": (_text(EDIT_BASE), _text(EDIT_HEAD))}, [("tests", 1), ("symbol", "test_b")],
    [_t178(1, 1, 1), _s178("function", "test_b")])
RESIDUALS_178 = [RESIDUAL_MIXED_QUOTES, RESIDUAL_TAIL_ABOVE_7F, RESIDUAL_QUOTED_PATH, RESIDUAL_SVN_TIMESTAMP,
                 RESIDUAL_DOCSTRING_EDIT]

# #205's seams under #178's symbol abstention (review 2 of fix batch 1, pinned per port then): a `def foo():` led by
# U+FEFF (line 1 of a file saved with a byte order mark; CPython defines foo) or a `def foo<U+00E9>():` (CPython defines
# another name) in a .py section, beside a plain `def foo():` in a .md file. main's search finds the plain line in both
# ports, and over the Python-source lines only the seam line is left, which the port read as defining foo and the
# Python did not: the Python abstained (#178) and the port kept main's VERIFIED, and the gates parted under --strict.
# The seam abstention reads the Python-source lines too, and no line there is one both ports read as defining foo, so
# both ports abstain (#205) before #178 (review 3 of fix batch 1). main read VERIFIED in both ports on both.
SEAM_MD = _new_file("docs/a.md", ["def foo():"])

# The `+++ /dev/null` header (review 2 of fix batch 1). git writes it bare, but the raw door reads any text. On
# `+++ /dev/null<ch>` or `+++ <ch>/dev/null`, where Python's strip() removes ch and the port's trim() keeps it (U+001C to
# U+001F, U+0085) or the reverse (U+FEFF), #178 read after strip() and trim() took the `---` path (`tests/test_a.py`,
# Python source) in one port and `dev/null<ch>` (not .py) in the other, and fired in that one only: on a false count
# one gate FAILed (CONTRADICTED) and the other PASSed (UNCHECKABLE) without --strict, where origin/main read each alike.
# Read on `_edge_cut` / `_edgeCut`, both ports take the `---` path and read each as main does. The deleted file's
# section adds a `def test_x` only because the raw door reads any text; src/o.py keeps #110 out of it.
def _deleted_adding(path: str, header: str, added: list) -> str:
    return (f"diff --git a/{path} b/{path}\ndeleted file mode 100644\nindex 1111111..0000000\n--- a/{path}\n"
            f"+++ {header}\n@@ -1,1 +0,{len(added)} @@\n-x = 0\n" + "".join("+" + x + "\n" for x in added))


D_OTHER = _edit("src/o.py", ["x = 1"], ["x = 2"])
DEVNULL_CHARS = {"U+001C": "\x1c", "U+001D": "\x1d", "U+001E": "\x1e", "U+001F": "\x1f", "U+0085": "\x85",
                 "U+FEFF": "\ufeff"}


# ---- #205's seams, and the name defined twice (review 3 of fix batch 1) -------------------------------------------------
# The reasons, the same strings in both ports. Each says what was measured: which code point, where it sits, and that
# the template cannot tell (whether the line heads the file, which file it sits in, or whether CPython defines the name).
WHY_SEAM_TESTS = ("an added `def test_` line holds {ch} before its `def`, which {what}; this template counts `def` "
                  "lines and cannot tell where it sits; claim says {n} (#205)")
WHY_SEAM_BEFORE = ("an added line that would define {kind} {name} holds {ch} before the name, which {what}; this "
                   "template cannot tell whether CPython defines it there (#205)")
WHY_SEAM_AFTER = ("an added line that would define {kind} {name} runs the name on into {ch}, where the two ports' "
                  "readers end a name apart; this template cannot tell whether CPython defines it there (#205)")
WHY_SEAM_HEADER = ("a `---` or `+++` header holds {ch} at an edge, which Python's strip() and the port's trim() remove "
                   "apart, and no header both read alike names a Python file; this template counts `def` lines and "
                   "cannot tell whether the diff touches one (#205)")
WHY_TWICE = ("{k} of the `def test_` lines added in one `+++` section define {name}, one of them after `async`; CPython "
             "binds a name once in a scope, and this template cannot tell the scopes apart; claim says {n} (#179)")
BOM = "CPython reads only as the byte order mark heading a file"
REFUSED = "CPython refuses in source outside a string or a comment"


def _tseam(ch: str, n: int) -> list:
    return ["tests_added", "UNCHECKABLE", WHY_SEAM_TESTS.format(ch=ch, n=n, what=BOM if ch == "U+FEFF" else REFUSED)]


def _sseam(kind: str, name: str, ch: str, after: bool = False) -> list:
    why = WHY_SEAM_AFTER if after else WHY_SEAM_BEFORE
    return ["symbol_added", "UNCHECKABLE",
            why.format(kind=kind, name=repr(name), ch=ch, what=BOM if ch == "U+FEFF" else REFUSED)]


def _hseam(kind: str, ch: str) -> list:
    return [kind, "UNCHECKABLE", WHY_SEAM_HEADER.format(ch=ch)]


def _twice(k: int, name: str, n: int) -> list:
    return ["tests_added", "UNCHECKABLE", WHY_TWICE.format(k=k, name=repr(name), n=n)]


# The seam before a `def` or a name, and after a name. On 35226d84 the two ports printed different records on every
# case but the control, and their gates parted without --strict on all but the two beside a Markdown file (with it, on
# those); main read the `async def` cases and the two beside a Markdown file alike, and the U+FEFF-led count, the run-on
# `def` and the unmasked run-on apart (#205's residual), parting the gates without --strict on the run-on `def` only.
# The BOM-led line 1 is a definition by CPython (a file's leading byte order mark); the other U+FEFF lines CPython
# refuses, and no run-on line defines the claimed name. The control: beside the run-on line, a plain `def foo():` both
# ports read keeps main's VERIFIED.
BOM_TEST = ["\ufeffasync def test_x():", "    pass"]
BOM_FETCH = ["\ufeffasync def fetch():", "    return 1"]
D_BOM_TEST = _new_file("tests/test_bom.py", BOM_TEST)
PY_BOM_TEST = {"tests/test_bom.py": (None, _text(BOM_TEST))}
SEAM_STANDS = _case("seam-a-line-both-ports-read-stands", "Added function foo.",
                    _new_file("src/c.py", ["def foo\u00e9():", "    pass", "def foo():", "    pass"]),
                    {"src/c.py": (None, _text(["def foo\u00e9():", "    pass", "def foo():", "    pass"]))},
                    [("symbol", "foo")], [_symbol("function", "foo", True)])
CASES_SEAM_TESTS = [
    _case("seam-bom-line-one-test", "Added 1 test.", D_BOM_TEST, PY_BOM_TEST, [("tests", 1)], [_tseam("U+FEFF", 1)]),
    _case("seam-bom-line-one-false-count", "Added 0 tests.", D_BOM_TEST, PY_BOM_TEST, [("tests", 0)],
          [_tseam("U+FEFF", 0)]),
    _case("seam-feff-lead-line-two", "Added 1 test.", _new_file("tests/test_g.py", FEFF_LEAD),
          {"tests/test_g.py": (None, _text(FEFF_LEAD))}, [("tests", 1)], [_tseam("U+FEFF", 1)]),
    _case("seam-feff-lead-shifts-the-count", "Added 2 tests.",
          _new_file("tests/test_s.py", ["\ufeffdef test_a():", "    pass", "async def test_b():", "    pass"]), None,
          [("tests", 2)], [_tseam("U+FEFF", 2)]),
]
CASES_SEAM_SYMBOL = [
    _case("seam-bom-line-one-fetch", "Added function fetch.", _new_file("src/bom.py", BOM_FETCH),
          {"src/bom.py": (None, _text(BOM_FETCH))}, [("symbol", "fetch")], [_sseam("function", "fetch", "U+FEFF")]),
    _case("seam-feff-gap", "Added function fetch.", _new_file("src/g.py", ["async def\ufefffetch():", "    pass"]),
          None, [("symbol", "fetch")], [_sseam("function", "fetch", "U+FEFF")]),
    _case("seam-feff-gap-across-lines", "Added function helper.",
          _new_file("src/g.py", ["async def", "\ufeff\xa0", "helper(x):"]), None, [("symbol", "helper")],
          [_sseam("function", "helper", "U+FEFF")]),
    _case("seam-run-on-async", "Added function fetch.", _new_file("src/r.py", ["async def fetch\u00e9():", "    pass"]),
          {"src/r.py": (None, _text(["async def fetch\u00e9():", "    pass"]))}, [("symbol", "fetch")],
          [_sseam("function", "fetch", "U+00E9", after=True)]),
    _case("seam-run-on-def", "Added function foo.", _new_file("src/r.py", ["def foo\u00e9():", "    pass"]),
          {"src/r.py": (None, _text(["def foo\u00e9():", "    pass"]))}, [("symbol", "foo")],
          [_sseam("function", "foo", "U+00E9", after=True)]),
    _case("seam-run-on-unmasked", "Added 1 test. Added function foo.",
          _new_file("tests/test_u.py", ["async def test_x():", "    pass", "def foo\u00e9():", "    pass"]),
          {"tests/test_u.py": (None, _text(["async def test_x():", "    pass", "def foo\u00e9():", "    pass"]))},
          [("tests", 1), ("symbol", "foo")], [_tests(1, 1), _sseam("function", "foo", "U+00E9", after=True)]),
    _case("seam-bom-led-def-beside-markdown", "Added function foo.",
          _new_file("src/b.py", ["\ufeffdef foo():", "    pass"]) + SEAM_MD, None, [("symbol", "foo")],
          [_sseam("function", "foo", "U+FEFF")]),
    _case("seam-run-on-beside-markdown", "Added function foo.",
          _new_file("src/b.py", ["def foo\u00e9():", "    pass"]) + SEAM_MD, None, [("symbol", "foo")],
          [_sseam("function", "foo", "U+00E9", after=True)]),
    SEAM_STANDS,
]
CASES_SEAM = CASES_SEAM_TESTS + CASES_SEAM_SYMBOL

# The seam at a header's edge: a `---` or `+++` header holding U+FEFF, U+0085 or U+001F at an edge of a path whose
# tail reads .py or .pyi, with no other header naming a Python file, where one port read a Python file and the other
# did not (#110 in one port only). On 35226d84 each parted the two ports' gates, without --strict on the three test
# counts and with it on the symbol; on main each parted their records (#110's header residual). The shapes are review
# 3's fuzz inputs.
CASES_HEADER = [
    _case("header-feff-tail", "Added 0 tests.",
          "--- \xa0a/tests/test_b.py\n+++ b/tests/test_b.py\ufeff\n@@ -1,1 +1,1 @@\n-def foo():\n+async def test_y():\n",
          None, [("tests", 0)], [_hseam("tests_added", "U+FEFF")]),
    _case("header-0085-both-ends", "Added 0 tests.",
          "diff --git a/A.PY b/A.PY\nnew file mode 100644\nindex 1111111..2222222 100644\n--- /dev/null\n"
          "+++ \x85A.PY\x85\n@@ -0,0 +1,1 @@\n+async  def test_n():\n", None, [("tests", 0)],
          [_hseam("tests_added", "U+0085")]),
    _case("header-001f-old-path", "Added 0 tests.",
          "diff --git a/tests/t.py b/tests/t.py\ndeleted file mode 100644\nindex 1111111..2222222 100644\n"
          "--- a/tests/t.py\x1f\n+++ /dev/null\n@@ -1,1 +1,1 @@\n+async\tdef test_z():\n", None, [("tests", 0)],
          [_hseam("tests_added", "U+001F")]),
    _case("header-feff-symbol", "Added function fetch.",
          "--- a/src/a.py\n+++ b/src/a.py\ufeff\n@@ -1,1 +1,1 @@\n-x = 0\n+async def fetch():\n", None,
          [("symbol", "fetch")], [_hseam("symbol_added", "U+FEFF")]),
]

# The name defined twice (#179's read reaching main's duplicate-definition limit). The two-hunk case is the truth world's shape
# (`p2a-101:20260930:217`, `:379`): a test made async and its new name added again in a later hunk, "Added 2 tests." read
# VERIFIED on 35226d84 (main: CONTRADICTED, right). Controls: one async name in each of two files is two tests; two
# `def` lines of one name keep main's count (a residual: ast binds one test, the count reads two); and two async
# methods of one name in two classes abstain, where CPython defines two tests (a right verdict the abstention costs).
TWICE_217 = ("diff --git a/tests/test_y.py b/tests/test_y.py\nindex 1111111..2222222 100644\n--- a/tests/test_y.py\n"
             "+++ b/tests/test_y.py\n@@ -3,1 +3,1 @@\n-def test_d():\n+async def test_d2():\n@@ -5,0 +6,3 @@\n"
             "+async def test_d2():\n+    assert os\n+\n")
TWICE_ASYNC = ["async def test_d():", "    pass", "", "async def test_d():", "    pass"]
TWICE_MIXED = ["def test_d():", "    pass", "", "async def test_d():", "    pass"]
TWICE_DEF = ["def test_d():", "    pass", "", "def test_d():", "    pass"]
TWICE_CLASSES = ["class TestA:", "    async def test_d(self):", "        pass", "", "", "class TestB:",
                 "    async def test_d(self):", "        pass"]
CASES_TWICE = [
    _case("twice-made-async-and-added-again", "Added 2 tests.", TWICE_217, None, [("tests", 2)],
          [_twice(2, "test_d2", 2)]),
    _case("twice-async", "Added 2 tests.", _new_file("tests/test_t.py", TWICE_ASYNC),
          {"tests/test_t.py": (None, _text(TWICE_ASYNC))}, [("tests", 2)], [_twice(2, "test_d", 2)]),
    _case("twice-async-true-count", "Added 1 test.", _new_file("tests/test_t.py", TWICE_ASYNC),
          {"tests/test_t.py": (None, _text(TWICE_ASYNC))}, [("tests", 1)], [_twice(2, "test_d", 1)]),
    _case("twice-def-then-async", "Added 2 tests.", _new_file("tests/test_t.py", TWICE_MIXED),
          {"tests/test_t.py": (None, _text(TWICE_MIXED))}, [("tests", 2)], [_twice(2, "test_d", 2)]),
    _case("twice-async-in-two-classes", "Added 2 tests.", _new_file("tests/test_t.py", TWICE_CLASSES),
          {"tests/test_t.py": (None, _text(TWICE_CLASSES))}, [("tests", 2)], [_twice(2, "test_d", 2)]),
    _case("twice-control-two-files", "Added 2 tests.",
          _new_file("tests/test_p.py", ["async def test_d():", "    pass"])
          + _new_file("tests/test_q.py", ["async def test_d():", "    pass"]),
          {"tests/test_p.py": (None, _text(["async def test_d():", "    pass"])),
           "tests/test_q.py": (None, _text(["async def test_d():", "    pass"]))}, [("tests", 2)], [_tests(2, 2)]),
]
RESIDUAL_TWICE_DEF = _case("residual-twice-def", "Added 2 tests.", _new_file("tests/test_t.py", TWICE_DEF),
                           {"tests/test_t.py": (None, _text(TWICE_DEF))}, [("tests", 2)], [_tests(2, 2)])

# #205's seam at a header's edge (review 3 of fix batch 1) does not fire on these: src/o.py is a header both ports read
# alike that names a Python file, so #110 reads them alike and the `_edge_cut` read of `/dev/null` decides. It decides,
# too, a header whose edge code point both strip() and trim() keep and the edge set removes (U+007F, U+200B, the shape
# of review 2's fuzz inputs f2024:7579 and f7:6141): CASES_DEVNULL_EDGE, read as main reads them.
CASES_DEVNULL = [
    _case(f"devnull-{end}-{tag}", "Added 1 test.",
          _deleted_adding("tests/test_a.py", "/dev/null" + ch if end == "tail" else ch + "/dev/null",
                          ["def test_x(): pass"]) + D_OTHER, None, [("tests", 1)], [_tests(1, 1)])
    for tag, ch in DEVNULL_CHARS.items() for end in ("tail", "lead")
] + [
    _case(f"devnull-false-count-{tag}", "Added 2 tests.",
          _deleted_adding("tests/test_a.py", "/dev/null" + ch, ["def test_x(): pass"]) + D_OTHER, None,
          [("tests", 2)], [_tests(1, 2)])
    for tag, ch in (("U+0085", "\x85"), ("U+FEFF", "\ufeff"))
]
CASES_DEVNULL_EDGE = [
    _case(f"devnull-edge-{end}-{tag}", "Added 1 test.",
          _deleted_adding("tests/test_a.py", "/dev/null" + ch if end == "tail" else ch + "/dev/null",
                          ["def test_x(): pass"]) + D_OTHER, None, [("tests", 1)], [_tests(1, 1)])
    for tag, ch in (("U+007F", "\x7f"), ("U+200B", "\u200b")) for end in ("tail", "lead")
] + [
    _case(f"devnull-edge-false-count-{tag}", "Added 2 tests.",
          _deleted_adding("tests/test_a.py", "/dev/null" + ch, ["def test_x(): pass"]) + D_OTHER, None,
          [("tests", 2)], [_tests(1, 2)])
    for tag, ch in (("U+007F", "\x7f"), ("U+200B", "\u200b"))
]
ALL_ALIKE = CASES_179 + CASES_GAP + [RESIDUAL_BACKSLASH, CASE_CHANGED] + CASES_178 + CASES_178_SIDES + \
    [RESIDUAL_UNDERCOUNT_ABOVE] + CASES_CONTROL + CASES_TAIL + \
    [RESIDUAL_STRING] + CASES_SECTIONS + RESIDUALS_178 + CASES_DEVNULL + CASES_DEVNULL_EDGE + CASES_SEAM + \
    CASES_HEADER + CASES_TWICE + [RESIDUAL_TWICE_DEF] + RESIDUALS_STRING_ASYNC + RESIDUALS_PARITY
RESIDUALS = {RESIDUAL_STRING["id"], RESIDUAL_MIXED_QUOTES["id"], RESIDUAL_UNDERCOUNT_ABOVE["id"]} | \
    {c["id"] for c in RESIDUALS_STRING_ASYNC + RESIDUALS_PARITY}                       # what the ast grid names


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
    cases = CASES_GAP + [RESIDUAL_BACKSLASH]
    got = _port(cases, tmp_path)
    assert _port_problems(cases, got) == []
    for c in cases:
        assert got[c["id"]] == _gate(N, c["summary"], c["diff"]), c["id"]


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


def test_178_the_two_sides_python():
    """Above every count main's verdict stands; at or below the larger count, and below the Python-source count,
    #178 abstains."""
    assert _problems(N, CASES_178_SIDES) == []


def test_178_the_two_sides_port(tmp_path):
    got = _port(CASES_178_SIDES, tmp_path)
    assert _port_problems(CASES_178_SIDES, got) == []


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


def test_178_named_residual_above_every_count(tmp_path):
    """Above every count main's verdict stands, on main's counts: `def  test_a` (two spaces) defines a test CPython
    reads and the template does not count, so a true "Added 2 tests." beside a fenced example reads CONTRADICTED in
    both ports, as main reads it."""
    assert [n.name for n in ast.parse(_text(UNDER)).body if isinstance(n, ast.FunctionDef)] == ["test_a", "test_b"]
    assert _problems(N, [RESIDUAL_UNDERCOUNT_ABOVE]) == []
    got = _port([RESIDUAL_UNDERCOUNT_ABOVE], tmp_path)
    assert got[RESIDUAL_UNDERCOUNT_ABOVE["id"]] == RESIDUAL_UNDERCOUNT_ABOVE["want"]


def test_178_named_residual_reached_through_async(tmp_path):
    """The same residual through `async def`: an `async def` inside a string opened in a context line is counted, where
    main counted no `async def` line, so a true count reads CONTRADICTED and a false one VERIFIED, at both doors, both
    ports alike; what CPython reads of the files."""
    heads = ast.parse(_text(PYTESTER_HEAD))
    assert [n.name for n in heads.body if isinstance(n, ast.FunctionDef)] == ["test_runs", "test_new"]
    assert [n.name for n in ast.parse(_text(SNIPPET_HEAD)).body if isinstance(n, ast.FunctionDef)] == ["run"]
    assert _problems(N, RESIDUALS_STRING_ASYNC) == []
    got = _port(RESIDUALS_STRING_ASYNC, tmp_path)
    assert _port_problems(RESIDUALS_STRING_ASYNC, got) == []
    for c in RESIDUALS_STRING_ASYNC:
        d = _repo_edit(tmp_path / c["id"], {p: b for p, (b, _h) in c["py"].items()},
                       {p: h for p, (_b, h) in c["py"].items()})
        rec = {"off": _rec(N.gate_diff(c["summary"], d, "HEAD~1", "HEAD")),
               "on": _rec(N.gate_diff(c["summary"], d, "HEAD~1", "HEAD", strict=True))}
        assert rec == c["want"], c["id"]


def test_178_named_residuals_of_the_parity_as_substrings(tmp_path):
    """A `\"\"\"` in a comment or in a one-quote string, and a one-quote string continued by a backslash: pinned as read."""
    assert _problems(N, RESIDUALS_PARITY) == []
    got = _port(RESIDUALS_PARITY, tmp_path)
    assert _port_problems(RESIDUALS_PARITY, got) == []


def test_178_the_sections_a_key_would_merge(tmp_path):
    """The parity and the path test are a `+++` section's: both ports print the same record where their keys of
    `parse_unified_diff_sides` merge two sections apart."""
    assert _problems(N, CASES_SECTIONS) == []
    got = _port(CASES_SECTIONS, tmp_path)
    assert _port_problems(CASES_SECTIONS, got) == []


def test_178_named_residuals_of_the_parity_and_the_path(tmp_path):
    """Mixed triple quotes, a tail from U+007F up, a quoted path (#169), an svn timestamp (#168) and an edited
    docstring line that turns the parity: pinned as read."""
    assert _problems(N, RESIDUALS_178) == []
    got = _port(RESIDUALS_178, tmp_path)
    assert _port_problems(RESIDUALS_178, got) == []


def _repo_edit(tmp_path: Path, base: dict, head: dict) -> Path:
    """A two-commit repository: a base holding `base`, and a head writing `head` over it."""
    if GIT is None:
        _missing("git", "the git door half of this check")
    d = tmp_path / "repo_edit"
    d.mkdir(parents=True)
    _git(d, "init", "-q")
    _git(d, "config", "core.autocrlf", "false")
    _git(d, "config", "core.longpaths", "true")
    for files, msg in ((base, "base"), (head, "head")):
        for rel, text in files.items():
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_bytes(text.encode("utf-8"))
        _git(d, "add", "-A")
        _git(d, "commit", "-q", "-m", msg)
    return d


def test_178_named_residual_the_docstring_edit_at_the_git_door(tmp_path):
    """The edited docstring line that turns the parity, through git: the same record as at the raw door."""
    c = RESIDUAL_DOCSTRING_EDIT
    d = _repo_edit(tmp_path, {p: b for p, (b, _h) in c["py"].items()}, {p: h for p, (_b, h) in c["py"].items()})
    got = {"off": _rec(N.gate_diff(c["summary"], d, "HEAD~1", "HEAD")),
           "on": _rec(N.gate_diff(c["summary"], d, "HEAD~1", "HEAD", strict=True))}
    assert got == c["want"]


def test_205_the_seams_before_a_def_and_after_a_name(tmp_path):
    """#205's seams where the counts read a line: both ports abstain (#205), and a line both read keeps main's verdict;
    what CPython reads of the BOM-led and run-on lines."""
    bom = [n.name for n in ast.walk(ast.parse("\ufeffdef foo():\n    pass\n".encode("utf-8")))
           if isinstance(n, ast.FunctionDef)]
    run_on = [n.name for n in ast.walk(ast.parse("def foo\u00e9():\n    pass\n")) if isinstance(n, ast.FunctionDef)]
    assert bom == ["foo"] and run_on == ["foo\u00e9"]
    assert _problems(N, CASES_SEAM) == []
    got = _port(CASES_SEAM, tmp_path)
    assert _port_problems(CASES_SEAM, got) == []


def test_205_the_bom_led_async_test_at_the_git_door(tmp_path):
    """A new file saved with a byte order mark whose line 1 is `async def test_x():`: CPython defines test_x, git writes
    the mark into the added line, and both counts abstain at the git door as at the raw door."""
    data = _text(BOM_TEST).encode("utf-8")
    assert [n.name for n in ast.parse(data).body] == ["test_x"]
    d = _repo(tmp_path, {"tests/test_bom.py": _text(BOM_TEST)})
    assert "+\ufeffasync def test_x():" in _git(d, "diff", "HEAD~1", "HEAD")
    for c in CASES_SEAM_TESTS[:2]:
        got = {"off": _rec(N.gate_diff(c["summary"], d, "HEAD~1", "HEAD")),
               "on": _rec(N.gate_diff(c["summary"], d, "HEAD~1", "HEAD", strict=True))}
        assert got == c["want"], c["id"]


def test_205_the_seam_at_a_header_edge(tmp_path):
    """A `---` or `+++` header holding a code point strip() and trim() remove apart, at an edge of a Python path, with
    no other header naming a Python file: both ports abstain. Beside one (the `/dev/null` cases): main's records."""
    cases = CASES_HEADER + CASES_DEVNULL
    assert _problems(N, cases) == []
    got = _port(cases, tmp_path)
    assert _port_problems(cases, got) == []


def test_179_a_name_defined_twice(tmp_path):
    """Two lines of one section the count reads define one name, one of them after `async`: both ports abstain (#179);
    two files are two tests; two `def` lines of one name keep main's count (a residual)."""
    cases = CASES_TWICE + [RESIDUAL_TWICE_DEF]
    # the module's ast grid counts definitions, not bound names, so it does not refute the residual: CPython binds one
    tree = ast.parse(_text(TWICE_DEF))
    assert [n.name for n in tree.body] == ["test_d", "test_d"]
    assert _problems(N, cases) == []
    got = _port(cases, tmp_path)
    assert _port_problems(cases, got) == []


def test_178_the_dev_null_header(tmp_path):
    """`+++ /dev/null` with a code point at either end, where strip() and trim() remove it apart or both keep it and
    the edge set removes it: both ports read the `---` path, and print main's records."""
    cases = CASES_DEVNULL + CASES_DEVNULL_EDGE
    assert _problems(N, cases) == []
    got = _port(cases, tmp_path)
    assert _port_problems(cases, got) == []
    for c in cases:
        assert got[c["id"]] == _gate(N, c["summary"], c["diff"]), c["id"]


EDGE_RUNNER = r"""
"use strict";
const fs = require("fs");
const vm = require("vm");
const [port, out] = process.argv.slice(2);
const P = vm.runInContext(fs.readFileSync(port, "utf8") + "\n;({_edgeCut})", vm.createContext({}), { filename: port });
const trimmed = [];
let cut = true;
for (let u = 0; u < 0x10000; u++) {
  const ch = String.fromCharCode(u);
  if (ch.trim() === "") trimmed.push(u);
}
for (const u of trimmed) {
  const ch = String.fromCharCode(u);
  if (P._edgeCut(ch + "/dev/null" + ch) !== "/dev/null") cut = false;
}
const astral = P._edgeCut("\u{1F600}/dev/null\u{1F600}") === "/dev/null" && P._edgeCut("a\u{1F600}b") === "a\u{1F600}b";
fs.writeFileSync(out, JSON.stringify({ trimmed, cut, astral }));
"""


def test_178_the_edge_set_holds_every_code_point_strip_or_trim_removes(tmp_path):
    """_edge_cut's set (at or below U+0020, or from U+007F up) holds every code point Python's strip() removes and
    every unit the port's trim() removes; the two differ by U+001C to U+001F and U+0085 (strip() only) and U+FEFF
    (trim() only), and each port's cut removes every one of them from either end of a header."""
    stripped = [c for c in range(0x110000) if chr(c).strip() == ""]
    assert all(c <= 32 or c >= 127 for c in stripped)
    assert all(N._edge_cut(chr(c) + "/dev/null" + chr(c)) == "/dev/null" for c in stripped)
    assert N._edge_cut("\U0001F600/dev/null\U0001F600") == "/dev/null" and N._edge_cut("a\U0001F600b") == "a\U0001F600b"
    if NODE is None:
        _missing("node", "the port half of this check")
    runner, out = tmp_path / "edge.js", tmp_path / "edge_out.json"
    runner.write_text(EDGE_RUNNER, encoding="utf-8")
    r = subprocess.run([NODE, str(runner), str(R.PORT), str(out)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    got = json.loads(out.read_text(encoding="utf-8"))
    assert all(u <= 32 or u >= 127 for u in got["trimmed"]) and got["cut"] and got["astral"]
    assert sorted(set(stripped) - set(got["trimmed"])) == [0x1C, 0x1D, 0x1E, 0x1F, 0x85]
    assert sorted(set(got["trimmed"]) - set(stripped)) == [0xFEFF]


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
    cases = [c for c in ALL_ALIKE if c["py"] is not None]
    assert len(cases) >= 40
    got = _port(cases, tmp_path)
    refuted = []
    for c in cases:
        refuted += _refuted(c, _gate(N, c["summary"], c["diff"])) + _refuted(c, got[c["id"]])
    assert {r[0] for r in refuted} <= RESIDUALS, refuted
    assert RESIDUAL_STRING["id"] in {r[0] for r in refuted}          # the residual is a residual: ast refutes it
    assert RESIDUAL_MIXED_QUOTES["id"] in {r[0] for r in refuted}
    assert RESIDUALS - {RESIDUAL_STRING["id"], RESIDUAL_MIXED_QUOTES["id"]} <= {r[0] for r in refuted}


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
    "#178: dropped from the count": ("if inside != placed and n <= max(placed, got):", "if False:", "def _gate(",
                                     CASES_178),
    "#178: a claim above every count abstains": ("if inside != placed and n <= max(placed, got):",
                                                 "if inside != placed:", "def _gate(", CASES_178_SIDES),
    "#178: a claim below the Python-source count decided": (
        "n <= max(placed, got):", "min(inside, got - (placed - inside)) <= n <= max(placed, got):", "def _gate(",
        CASES_178_SIDES),
    "#178: dropped from the symbol": ("if re.search(pat, sides_blob, re.M) and not re.search(pat, inside_blob, re.M):",
                                      "if False:", "def _gate(", CASES_178),
    "#178: the path's tail read as is": ("while e and (ord(p[e - 1]) <= 32 or ord(p[e - 1]) >= 127):", "while False:",
                                         "def _python_source_lines(", CASES_TAIL),
    "#178: the parity per key, not per section": (
        'sections=_diff_sections(diff_text or "")',
        'sections=[(p, a) for p, (a, _r) in parse_unified_diff_sides(diff_text or "").items()]',
        "def gate_diff_text(", CASES_SECTIONS),
    "#178: /dev/null read after strip()": ('if _edge_cut(line[4:]) == "/dev/null":', 'if new == "/dev/null":',
                                           "def _diff_sections(", CASES_DEVNULL + CASES_DEVNULL_EDGE),
    "#205: the seam dropped from the count": ("elif (seam := _seam_tests((added_blob, sides_blob))):", "elif False:",
                                              "def _gate(", CASES_SEAM_TESTS),
    "#205: the seam dropped from the symbol": (
        'elif (seam := _seam_symbol((added_blob, sides_blob, inside_blob), d["name"])):', "elif False:", "def _gate(",
        CASES_SEAM_SYMBOL),
    "#205: a line both ports read no longer stands": ("                    both = True\n",
                                                      "                    pass\n", "def _seam_symbol(", [SEAM_STANDS]),
    "#205: the seam at a header's edge dropped": ('seam_header=_seam_header(diff_text or "")', "seam_header=None",
                                                  "def gate_diff_text(", CASES_HEADER),
    "#205: a header both read alike no longer counts": ("                alike = True\n", "                pass\n",
                                                        "def _seam_header(", CASES_DEVNULL),
    "#179: the name defined twice dropped": ("elif (twice := _tests_twice(sections or [])):", "elif False:",
                                             "def _gate(", CASES_TWICE),
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
    "#178: dropped from the count": ("if (inside !== placed && n <= Math.max(placed, got)) {", "if (false) {",
                                     CASES_178),
    "#178: a claim above every count abstains": ("if (inside !== placed && n <= Math.max(placed, got)) {",
                                                 "if (inside !== placed) {", CASES_178_SIDES),
    "#178: a claim below the Python-source count decided": (
        "&& n <= Math.max(placed, got)) {",
        "&& Math.min(inside, got - (placed - inside)) <= n && n <= Math.max(placed, got)) {", CASES_178_SIDES),
    "#178: dropped from the symbol": ("if (pat.test(sidesBlob) && !pat.test(insideBlob)) {", "if (false) {", CASES_178),
    "#178: the path's tail read as is": (
        "    while (e > 0 && (p.charCodeAt(e - 1) <= 32 || p.charCodeAt(e - 1) >= 127)) e--;\n", "", CASES_TAIL),
    "#178: the parity per key, not per section": ("const sections = _diffSections(diffText);",
                                                  "const sections = [...sides].map(([p, [a]]) => [p, a]);",
                                                  CASES_SECTIONS),
    "#178: /dev/null read after trim()": ('if (_edgeCut(line.slice(4)) === "/dev/null")', 'if (nw === "/dev/null")',
                                          CASES_DEVNULL + CASES_DEVNULL_EDGE),
    "#205: the seam dropped from the count": ("} else if (_seamTests([addedBlob, sidesBlob]) !== null) {",
                                              "} else if (false) {", CASES_SEAM_TESTS),
    "#205: the seam dropped from the symbol": (
        "} else if (_seamSymbol([addedBlob, sidesBlob, insideBlob], d.name) !== null) {", "} else if (false) {",
        CASES_SEAM_SYMBOL),
    "#205: a line both ports read no longer stands": ("          both = true;\n", "", [SEAM_STANDS]),
    "#205: the seam at a header's edge dropped": ("const seamHeader = _seamHeader(diffText);", "const seamHeader = null;",
                                                  CASES_HEADER),
    "#205: a header both read alike no longer counts": ("        alike = true;\n", "", CASES_DEVNULL),
    "#179: the name defined twice dropped": ("} else if (_testsTwice(sections) !== null) {", "} else if (false) {",
                                             CASES_TWICE),
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
