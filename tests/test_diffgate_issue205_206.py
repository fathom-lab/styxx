"""#205 and #206 (fix/diffgate-205-206): the line cut, the abstention it needs, and the declared count.

#205. The Python cut a diff into lines with str.splitlines(), which also breaks at U+000B, U+000C, U+001C to U+001E,
U+0085, U+2028 and U+2029; git, CPython's tokenizer and the port do not. A line `+<U+000C>def test_a():` was cut in
two, its definition dropped, and the true claims "Added 1 test." and "Added function test_a." read CONTRADICTED. The
Python now cuts every piece of git output where the port does (`_git_lines`: CRLF, a lone CR, LF), and in both ports
the `def` counts and the compatibility reading abstain over a line holding a character the two readers read apart
once the line is whole (`_READ_APART`: U+000B, U+001C to U+001F, U+0085, U+2028, U+2029). The port reads the two
header patterns' `.` as the Python does, and the PATH-2a overlay models the new cut.

#206. A declared `files_changed` / `tests_added` count is 1-9 ASCII digits in both ports; any other value is
MALFORMED, and one holding a character from U+0080 up names the leftmost by its code point, so the verdict and the
reason are the same on every runtime, whatever its Unicode tables hold.

What is held here: the issues' reproductions at the raw door, at the git door (a real two-commit repository; skipped
without git) and in the port (skipped without node), in both strict modes; a grid of every str.splitlines() break,
every str.isspace() character and U+FEFF in seven positions on a definition line, judged by ast.parse; the cut against
str.splitlines() and against the port's, over every break, seeded; the sites that cut; the abstention's set and its
reasons, the same in both ports; every code point from U+0080 up in a declared count; and a plant of each change,
each told by a check here.
"""
from __future__ import annotations

import ast
import json
import random
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path

import pytest

import styxx.diffgate as N
from styxx import declare as D
from tests import _p2a_ref as R

NODE = shutil.which("node")
GIT = shutil.which("git")
FF = "\x0c"

# The port, loaded as a page loads it (one script), with the names these checks read.
RUNNER = r"""
"use strict";
const fs = require("fs");
const vm = require("vm");
const [port, inp, out] = process.argv.slice(2);
const ctx = vm.createContext({});
const P = vm.runInContext(fs.readFileSync(port, "utf8") +
  "\n;({gateDiffText, parseUnifiedDiff, canonicalSentence, _splitlines, _READ_APART})", ctx, { filename: port });
const rec = g => ({ verdict: g.verdict, claims: g.claims.map(c => [c.kind, c.verdict, c.why]) });
const res = JSON.parse(fs.readFileSync(inp, "utf8")).map(it => {
  const o = { id: it.id };
  try {
    if (it.op === "gate") {
      o.off = rec(P.gateDiffText(it.summary, it.diff));
      o.on = rec(P.gateDiffText(it.summary, it.diff, { strict: true }));
    } else if (it.op === "split") {
      o.lines = P._splitlines(it.text);
    } else if (it.op === "parse") {
      const p = P.parseUnifiedDiff(it.diff);
      o.status = [...p.status];
      o.blob = p.addedBlob;
    } else if (it.op === "canon") {
      o.canon = P.canonicalSentence(it.key, it.value);
    } else if (it.op === "apart") {
      o.source = P._READ_APART.source;
      o.flags = P._READ_APART.flags;
    } else if (it.op === "enum206") {
      // every code point from U+0080 up, after an ASCII digit: MALFORMED, naming that code point
      const bad = [];
      let n = 0;
      for (let cp = 0x80; cp <= 0x10ffff; cp++) {
        const [sent, why] = P.canonicalSentence("files_changed", "1" + String.fromCodePoint(cp));
        const want = "MALFORMED: a count is written in the ASCII digits 0-9, and the value holds U+" +
          cp.toString(16).toUpperCase().padStart(4, "0");
        n++;
        if (sent !== null || why !== want) { if (bad.length < 5) bad.push(cp); }
      }
      o.n = n;
      o.bad = bad;
    }
  } catch (e) {
    o.error = String((e && e.message) || e);
  }
  return o;
});
fs.writeFileSync(out, JSON.stringify(res));
"""


def _port(items: list, tmp_path: Path, port: Path = R.PORT, tag: str = "run") -> dict:
    """The port's answers, by id; skips where node is absent (the Python half has run by then)."""
    if NODE is None:
        pytest.skip("node is not on PATH; the port half of this check cannot run here")
    runner = tmp_path / "runner_205_206.js"
    runner.write_text(RUNNER, encoding="utf-8")
    src, out = tmp_path / f"{tag}_in.json", tmp_path / f"{tag}_out.json"
    src.write_text(json.dumps(items, ensure_ascii=True), encoding="utf-8")
    r = subprocess.run([NODE, str(runner), str(port), str(src), str(out)], capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    got = {x["id"]: x for x in json.loads(out.read_text(encoding="utf-8"))}
    assert not [x for x in got.values() if "error" in x], [x for x in got.values() if "error" in x][:3]
    return got


def _rec(g) -> dict:
    return {"verdict": g.verdict, "claims": [[c.kind, c.verdict, c.why] for c in g.claims]}


def _gate(mod, summary: str, diff: str) -> dict:
    return {"off": _rec(mod.gate_diff_text(summary, diff)), "on": _rec(mod.gate_diff_text(summary, diff, strict=True))}


def _new_file(path: str, lines: list) -> str:
    return (f"diff --git a/{path} b/{path}\nnew file mode 100644\nindex 0000000..27dd614\n--- /dev/null\n"
            f"+++ b/{path}\n@@ -0,0 +1,{len(lines)} @@\n" + "".join("+" + x + "\n" for x in lines))


# ---- 1. the issues' reproductions ------------------------------------------------------------------------------------

FF_DEF = [FF + "def test_a():", "    assert True"]                         # #205's shape
FF_METHOD = ["class TestA:", FF + "    def test_a(self):", "        assert True"]   # the same inside a class
DIFF_FF_DEF = _new_file("tests/test_a.py", FF_DEF)
DIFF_FF_METHOD = _new_file("tests/test_a.py", FF_METHOD)
TRUE_BOTH = "Added 1 test. Added function test_a."
FALSE_COUNT = "Added 0 tests."
DECLARED = "Adds a test.\n\n```styxx\ntests_added: 1\nadds_symbol: test_a\n```\n"
VERIFIED_BOTH = [["tests_added", "VERIFIED", "diff adds 1 test functions, claim says 1"],
                 ["symbol_added", "VERIFIED", "added lines do define function 'test_a'"]]
WANT_205 = {
    TRUE_BOTH: {"off": {"verdict": "PASS", "claims": VERIFIED_BOTH}, "on": {"verdict": "PASS", "claims": VERIFIED_BOTH}},
    FALSE_COUNT: {k: {"verdict": "FAIL", "claims": [["tests_added", "CONTRADICTED",
                                                     "diff adds 1 test functions, claim says 0"]]} for k in ("off", "on")},
    DECLARED: {k: {"verdict": "PASS", "claims": [["symbol_added", "VERIFIED", "added lines do define function 'test_a'"],
                                                 ["tests_added", "VERIFIED", "diff adds 1 test functions, claim says 1"]]}
               for k in ("off", "on")},                     # a declaration is read in its keys' order
}


def _ff_problems(mod, door: str = "raw") -> list:
    """What `mod` reads apart from WANT_205 on the issue's two shapes, at the raw door or (through a recorded
    listing) the git door; [] when it reads them as CPython does."""
    out = []
    for diff in (DIFF_FF_DEF, DIFF_FF_METHOD):
        for summary, want in WANT_205.items():
            if door == "raw":
                got = _gate(mod, summary, diff)
            else:
                saved = mod._git
                mod._git = R.fake_git("A\ttests/test_a.py\n", diff)
                try:
                    got = {"off": _rec(mod.gate_diff(summary, "(repo)", "base", "head")),
                           "on": _rec(mod.gate_diff(summary, "(repo)", "base", "head", strict=True))}
                finally:
                    mod._git = saved
            if got != want:
                out.append((door, summary[:20], diff.count("class"), got))
    return out


def test_the_truth_of_the_reproductions():
    """CPython defines test_a in both shapes: the form feed is white space in its indentation."""
    for lines in (FF_DEF, FF_METHOD):
        names = [n.name for n in ast.walk(ast.parse("\n".join(lines) + "\n")) if isinstance(n, ast.FunctionDef)]
        assert names == ["test_a"], lines


def test_205_raw_door_python():
    assert _ff_problems(N, "raw") == []


def test_205_git_door_python_with_a_recorded_listing():
    assert _ff_problems(N, "git") == []


def test_205_the_port(tmp_path):
    items = [{"id": f"{i}:{j}", "op": "gate", "summary": s, "diff": d}
             for i, d in enumerate((DIFF_FF_DEF, DIFF_FF_METHOD)) for j, s in enumerate(WANT_205)]
    got = _port(items, tmp_path)
    for i, d in enumerate((DIFF_FF_DEF, DIFF_FF_METHOD)):
        for j, (s, want) in enumerate(WANT_205.items()):
            g = got[f"{i}:{j}"]
            assert {"off": g["off"], "on": g["on"]} == want, (i, s)
            assert {"off": g["off"], "on": g["on"]} == _gate(N, s, d), (i, s)


def _git(d: Path, *a) -> str:
    return subprocess.run([GIT, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false",
                           "-c", "core.longpaths=true", *a], cwd=d, capture_output=True,
                          check=True).stdout.decode("utf-8")


@pytest.mark.skipif(GIT is None, reason="git is not on PATH; the git door cannot be exercised here")
@pytest.mark.parametrize("lines", [FF_DEF, FF_METHOD], ids=["def", "method"])
def test_205_git_door_on_a_real_repository(lines, tmp_path):
    """The issue's git-door reproduction: a commit adding the file. Git writes the form feed inside one diff line, and
    main's _git (which passes no -c) reads the repository's own configuration."""
    d = tmp_path / "repo"
    d.mkdir()
    _git(d, "init", "-q")
    _git(d, "config", "core.autocrlf", "false")
    _git(d, "config", "core.longpaths", "true")
    _git(d, "commit", "-q", "--allow-empty", "-m", "base")
    (d / "tests").mkdir()
    (d / "tests" / "test_a.py").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    _git(d, "add", "-A")
    _git(d, "commit", "-q", "-m", "head")
    diff = _git(d, "diff", "HEAD~1", "HEAD")
    assert "\n+" + lines[-2] + "\n" in diff                    # one git line, the form feed inside it
    for summary, want in WANT_205.items():
        got = {"off": _rec(N.gate_diff(summary, d, "HEAD~1", "HEAD")),
               "on": _rec(N.gate_diff(summary, d, "HEAD~1", "HEAD", strict=True))}
        assert got == want, summary
    declared = "Small change.\n\n```styxx\ntests_added: \uff11\n```\n"     # #206 at the same door
    g = N.gate_diff(declared, d, "HEAD~1", "HEAD")
    assert _rec(g) == {"verdict": "PASS", "claims": [["tests_added", "UNCHECKABLE", WIDE.format("FF11")]]}


# #206: the issue's table, over a diff that changes three files.
WIDE = "MALFORMED: a count is written in the ASCII digits 0-9, and the value holds U+{}"
THREE = "".join(f"diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n@@ -1 +1 @@\n-x\n+y\n" for p in ("a.md", "b.md", "c.md"))
TABLE_206 = [
    ("3", ["files_changed_count", "VERIFIED", "diff changes 3 files, claim says 3"], "PASS", "PASS"),
    ("\uff13", ["files_changed", "UNCHECKABLE", WIDE.format("FF13")], "PASS", "FAIL"),
    ("\uff15", ["files_changed", "UNCHECKABLE", WIDE.format("FF15")], "PASS", "FAIL"),
    ("\u0663", ["files_changed", "UNCHECKABLE", WIDE.format("0663")], "PASS", "FAIL"),
    ("\U00010d43", ["files_changed", "UNCHECKABLE", WIDE.format("10D43")], "PASS", "FAIL"),   # Garay, Unicode 16
    ("\xb3", ["files_changed", "UNCHECKABLE", WIDE.format("00B3")], "PASS", "FAIL"),
    ("0\uff13", ["files_changed", "UNCHECKABLE", WIDE.format("FF13")], "PASS", "FAIL"),
    ("\U0001d7d1", ["files_changed", "UNCHECKABLE", WIDE.format("1D7D1")], "PASS", "FAIL"),
    ("many", ["files_changed", "UNCHECKABLE", "MALFORMED: 'many' is not a count"], "PASS", "FAIL"),
]


def _declared_count(v: str) -> str:
    return "Small cleanup.\n\n```styxx\nfiles_changed: " + v + "\n```\n"


def test_206_the_issues_table_python():
    """The same row on every runtime: no row reads the interpreter's digit table (U+10D43 is a digit on Unicode 16 and
    unassigned on 15, and reads the same on both)."""
    for v, claim, off, on in TABLE_206:
        g = _gate(N, _declared_count(v), THREE)
        assert g == {"off": {"verdict": off, "claims": [claim]}, "on": {"verdict": on, "claims": [claim]}}, ascii(v)


def test_206_the_issues_table_port(tmp_path):
    items = [{"id": str(i), "op": "gate", "summary": _declared_count(v), "diff": THREE} for i, (v, *_r) in enumerate(TABLE_206)]
    got = _port(items, tmp_path)
    for i, (v, claim, off, on) in enumerate(TABLE_206):
        assert {"off": got[str(i)]["off"], "on": got[str(i)]["on"]} == \
            {"off": {"verdict": off, "claims": [claim]}, "on": {"verdict": on, "claims": [claim]}}, ascii(v)


def test_206_every_code_point_from_0x80_up_is_malformed_and_named_python():
    """For every code point from U+0080 to U+10FFFF, a count holding it is MALFORMED, and the reason names it; so no
    runtime's digit table, of any Unicode version, can reach the reader. The digits the runtime's own tables know are
    among them, whatever version they are."""
    digits = 0
    for cp in range(0x80, 0x110000):
        ch = chr(cp)
        for key in ("files_changed", "tests_added"):
            assert D.canonical_sentence(key, "1" + ch) == (None, WIDE.format(f"{cp:04X}")), (key, cp)
        digits += unicodedata.decimal(ch, None) is not None
    assert digits > 600                                           # the runtime's Nd digits, all MALFORMED above
    assert D.canonical_sentence("files_changed", "123456789") == ("123456789 files changed.", None)
    assert D.canonical_sentence("files_changed", "1234567890")[1] == "MALFORMED: '1234567890' is not a count"


def test_206_every_code_point_from_0x80_up_is_malformed_and_named_port(tmp_path):
    got = _port([{"id": "e", "op": "enum206"}], tmp_path)["e"]
    assert got["n"] == 0x110000 - 0x80 and got["bad"] == []


def test_206_the_two_ports_print_the_same_reason(tmp_path):
    values = ["3", "many", "", "-1", "1e3", "\uff13", "\u0663", "\U00010d43", "\xb3", "\ud800", "3\U0001f600", "x\uff13"]
    items = [{"id": f"{k}:{i}", "op": "canon", "key": k, "value": v}
             for k in ("files_changed", "tests_added") for i, v in enumerate(values)]
    got = _port(items, tmp_path)
    for it in items:
        assert list(D.canonical_sentence(it["key"], it["value"])) == got[it["id"]]["canon"], ascii(it["value"])


# ---- 2. the grid: every break and white space, in each position a definition can take --------------------------------

GRID_CHARS = sorted({c for c in range(0x110000) if len(("a" + chr(c) + "b").splitlines()) == 2}
                    | {c for c in range(0x110000) if chr(c).isspace()} | {0xFEFF} - {0x0A})
GRID_CHARS = [c for c in GRID_CHARS if c != 0x0A]               # LF is git's own line end
POSITIONS = ("t-lead-new", "t-lead-reindent", "t-mid", "t-class-indent", "s-lead-new", "s-sep-new", "s-end-new")
GRID_HEAD = "diff --git a/tests/test_x.py b/tests/test_x.py\n--- a/tests/test_x.py\n+++ b/tests/test_x.py\n"


def _cell(ch: str, pos: str) -> tuple:
    return {
        "t-lead-new": ("x = 0\n", "x = 0\n" + ch + "def test_n():\n    pass\n"),
        "t-lead-reindent": ("def test_n():\n    pass\n", ch + "def test_n():\n    pass\n"),
        "t-mid": ("x = 0\n", "x = 0\ny = 1" + ch + "def test_n():\n    pass\n"),
        "t-class-indent": ("class T:\n    pass\n", "class T:\n    pass\n" + ch + "    def test_n(self):\n        pass\n"),
        "s-lead-new": ("x = 0\n", "x = 0\n" + ch + "def foo():\n    pass\n"),
        "s-sep-new": ("x = 0\n", "x = 0\ndef" + ch + "foo():\n    pass\n"),
        "s-end-new": ("x = 0\n", "x = 0\ndef foo" + ch + "():\n    pass\n"),
    }[pos]


def _defs(src: str):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    return [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]


def _whole_file_hunk(base: str, head: str) -> str:
    b, h = base.split("\n")[:-1], head.split("\n")[:-1]           # git's own lines: cut at LF only
    return (GRID_HEAD + f"@@ -1,{len(b)} +1,{len(h)} @@\n" + "".join("-" + x + "\n" for x in b)
            + "".join("+" + x + "\n" for x in h))


def grid() -> list:
    """{id, summary, diff, true}: `true` is what ast.parse of the two files says of the claim (None where a file does
    not parse, so the claim has no truth there)."""
    out = []
    for c in GRID_CHARS:
        for pos in POSITIONS:
            base, head = _cell(chr(c), pos)
            bd, hd = _defs(base), _defs(head)
            diff = _whole_file_hunk(base, head)
            if pos.startswith("t-"):
                added = None if bd is None or hd is None else \
                    sum(n.startswith("test_") for n in hd) - sum(n.startswith("test_") for n in bd)
                for s, n in (("Added 0 tests.", 0), ("Added 1 test.", 1)):
                    out.append({"id": f"U+{c:04X}|{pos}|{s}", "summary": s, "diff": diff,
                                "true": None if added is None else added == n})
            else:
                out.append({"id": f"U+{c:04X}|{pos}|foo", "summary": "Adds function foo.", "diff": diff,
                            "true": None if bd is None or hd is None else ("foo" in hd and "foo" not in bd)})
    return out


# Measured on this change, CPython 3.12.10 and 3.14.2, Node 24.13.0: the cells where a claim ast reads as true is
# CONTRADICTED (the same four in both ports: a lone CR, #176, which both ports cut at and git does not), and the cells
# whose two final claim lists differ (all eight a U+FEFF lead or separator: the port's \s reads it, the Python's does
# not; unchanged by #205 and named in its residuals). On main before #205 these were 8 and 4 cells, and 52 apart.
RESIDUAL_CONTRADICTED = {"U+000D|t-lead-new|Added 1 test.", "U+000D|t-class-indent|Added 1 test.",
                         "U+000D|t-mid|Added 1 test.", "U+000D|s-lead-new|foo"}
RESIDUAL_APART_CHAR = "U+FEFF|"
RESIDUAL_APART = 8


def _contradicted_true(cells: list, finals: dict) -> set:
    return {c["id"] for c in cells if c["true"] is True
            and any(x[0] in ("tests_added", "symbol_added") and x[1] == "CONTRADICTED" for x in finals[c["id"]])}


def test_the_grid_python():
    cells = grid()
    assert len(GRID_CHARS) == 29 and len(cells) == 319
    finals = {c["id"]: _rec(N.gate_diff_text(c["summary"], c["diff"]))["claims"] for c in cells}
    assert _contradicted_true(cells, finals) == RESIDUAL_CONTRADICTED


def test_the_grid_port_and_across_ports(tmp_path):
    cells = grid()
    got = _port([{"id": c["id"], "op": "gate", "summary": c["summary"], "diff": c["diff"]} for c in cells], tmp_path)
    port = {k: v["off"]["claims"] for k, v in got.items()}
    assert _contradicted_true(cells, port) == RESIDUAL_CONTRADICTED
    apart = [c["id"] for c in cells if port[c["id"]] != _rec(N.gate_diff_text(c["summary"], c["diff"]))["claims"]]
    assert all(x.startswith(RESIDUAL_APART_CHAR) for x in apart) and len(apart) == RESIDUAL_APART, apart


# ---- 3. the cut ------------------------------------------------------------------------------------------------------

BREAKS = {c for c in range(0x110000) if len(("a" + chr(c) + "b").splitlines()) == 2}
CUT_APART = {0x0B, 0x0C, 0x1C, 0x1D, 0x1E, 0x85, 0x2028, 0x2029}      # str.splitlines() breaks git does not make


def test_the_breaks_str_splitlines_makes():
    assert BREAKS == CUT_APART | {0x0A, 0x0D}
    assert N._GIT_LINE_END.pattern == "\r\n|\r|\n" == N._P2A_COARSE.pattern      # the overlay models main's cut


def _texts(seed: int, n: int, alphabet: list) -> list:
    rng = random.Random(seed)
    out = ["", "\n", "\r\n", "\r", "a", "a\n", "a\r\n\r\n", "\n\n", "a\rb\r\n"]
    for _ in range(n):
        out.append("".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 24))))
    return out


ALPHABET = ["a", "+", "-", " ", "d", "\t", "\n", "\r", "\r\n", "\ufeff", "\x1f"] + [chr(c) for c in sorted(CUT_APART)]


def test_git_lines_is_str_splitlines_where_no_break_parts_them():
    """Seeded: on a text holding none of the eight breaks git does not make, the cut is str.splitlines() exactly (a
    trailing break and "" included); on a text holding one, it never breaks there."""
    plain = [x for x in ALPHABET if not (len(x) == 1 and ord(x) in CUT_APART)]
    for t in _texts(205, 4000, plain):
        assert N._git_lines(t) == t.splitlines(), ascii(t)
    for t in _texts(206, 4000, ALPHABET):
        lines = N._git_lines(t)
        assert "".join(lines).count("\r") == 0 and "\n" not in "".join(lines)
        assert sum(ln.count(chr(c)) for ln in lines for c in CUT_APART) == sum(t.count(chr(c)) for c in CUT_APART)


def test_git_lines_is_the_ports_cut_on_every_break(tmp_path):
    """Seeded texts over every str.splitlines() break, and each break alone in each place a line can hold it: the
    Python's cut and the port's _splitlines give the same lines, and parse_unified_diff the same statuses and added
    lines in both ports."""
    texts = _texts(207, 3000, ALPHABET)
    texts += [s.replace("X", chr(c)) for c in sorted(BREAKS) for s in ("X", "aX", "Xa", "aXb", "aX\n", "\nX\n", "+X+")]
    diffs = [_new_file("tests/test_x.py", [chr(c) + "def test_a():", "x" + chr(c) + "y", chr(c)]) for c in sorted(BREAKS)]
    diffs += ["diff --git a/x" + chr(c) + ".py b/x" + chr(c) + ".py\n--- a/x" + chr(c) + ".py\n+++ b/x" + chr(c)
              + ".py\n@@ -1 +1 @@\n-a\n+b" + chr(c) + "c\n" for c in sorted(BREAKS)]
    items = [{"id": f"s{i}", "op": "split", "text": t} for i, t in enumerate(texts)]
    items += [{"id": f"p{i}", "op": "parse", "diff": d} for i, d in enumerate(diffs)]
    got = _port(items, tmp_path)
    for i, t in enumerate(texts):
        assert got[f"s{i}"]["lines"] == N._git_lines(t), ascii(t)
    for i, d in enumerate(diffs):
        status, blob = N.parse_unified_diff(d)
        assert [got[f"p{i}"]["status"], got[f"p{i}"]["blob"]] == [[list(x) for x in status.items()], blob], ascii(d)


def _block_lines(text: str) -> range:
    lines = text.split("\n")
    a = next(i for i, x in enumerate(lines) if x == R.PY_BEGIN)
    b = next(i for i, x in enumerate(lines) if x == R.PY_END)
    return range(a + 1, b + 2)


def cut_sites(text: str) -> tuple:
    """(every `.splitlines()` call outside the PATH-2a block, every `_git_lines(...)` call by enclosing function)."""
    tree = ast.parse(text)
    block = _block_lines(text)
    split, cut = [], []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and node.lineno not in block:
                f = node.func
                if isinstance(f, ast.Attribute) and f.attr == "splitlines":
                    split.append((fn.name, node.lineno))
                if isinstance(f, ast.Name) and f.id == "_git_lines":
                    cut.append(fn.name)
    return split, sorted(cut)


WANT_SITES = sorted(["parse_unified_diff_sides", "parse_unified_diff", "gate_diff", "gate_diff"])


def test_every_site_that_cuts_git_output_uses_the_cut():
    """No reader outside the block cuts with str.splitlines() any more, and the four sites cut with _git_lines:
    parse_unified_diff_sides, parse_unified_diff, and gate_diff's --name-status and its added lines."""
    split, cut = cut_sites(R.lf(R.INSTRUMENT))
    assert split == [] and cut == WANT_SITES, (split, cut)


# ---- 4. the abstention ------------------------------------------------------------------------------------------------

APART = {0x0B, 0x1C, 0x1D, 0x1E, 0x1F, 0x85, 0x2028, 0x2029}
WHY_ADDED = ("an added line holds U+{}, which CPython refuses in source and the Python and JavaScript readers read "
             "apart; this template counts `def` lines (#205)")
WHY_COMPAT = ("compatibility claimed; a changed line holds U+{}, which the Python and JavaScript readers read apart, "
              "so no definition was read (#205)")


def _members(rx) -> set:
    return {c for c in range(0x110000) if rx.search(chr(c))}


def _class_of_port_source(source: str) -> set:
    m = re.fullmatch(r"\[((?:\\u[0-9a-fA-F]{4}(?:-\\u[0-9a-fA-F]{4})?)+)\]", source)
    assert m, source
    out = set()
    for a, b in re.findall(r"\\u([0-9a-fA-F]{4})(?:-\\u([0-9a-fA-F]{4}))?", m.group(1)):
        out |= set(range(int(a, 16), int(b or a, 16) + 1))
    return out


def test_the_abstention_set_python():
    assert _members(N._READ_APART) == APART
    assert APART == (CUT_APART - {0x0C}) | {0x1F}
    assert all(not chr(c).isspace() or c in (0x0B, 0x1C, 0x1D, 0x1E, 0x1F, 0x85, 0x2028, 0x2029) for c in APART)


def _abstain_cases() -> list:
    out = []
    for c in sorted(APART | {0x0C, 0xFEFF}):
        ch = chr(c)
        for tag, lines in (("lead", [ch + "def test_a():", "    pass"]), ("mid", ["x = 1" + ch + "y = 2"])):
            out.append({"id": f"{c:04X}|{tag}|tests", "op": "gate", "summary": "Added 1 test. Added function test_a.",
                        "diff": _new_file("tests/test_a.py", lines)})
        out.append({"id": f"{c:04X}|compat", "op": "gate", "summary": "No breaking changes.",
                    "diff": ("diff --git a/src/m.py b/src/m.py\n--- a/src/m.py\n+++ b/src/m.py\n@@ -1,2 +1,1 @@\n"
                             "-def keep():\n-x = 1" + ch + "\n+def keep():\n")})
    return out


def _abstain_problems(mod) -> list:
    """The abstention's reasons, as the Python must write them: each member of APART on an added line, at its start or
    inside it, gives both `def` claims the #205 reason naming it, and the compatibility reading its #205 reason; a form
    feed and U+FEFF give neither."""
    bad = []
    for it in _abstain_cases():
        c = int(it["id"].split("|")[0], 16)
        claims = _rec(mod.gate_diff_text(it["summary"], it["diff"]))["claims"]
        name = f"{c:04X}"
        if it["id"].endswith("|compat"):
            want = WHY_COMPAT.format(name) if c in APART else None
            ok = (claims[0][2] == want) if want else (claims[0][2] != WHY_COMPAT.format(name))
        else:
            want = WHY_ADDED.format(name) if c in APART else None
            ok = all(x[2] == want and x[1] == "UNCHECKABLE" for x in claims) if want else \
                all("(#205)" not in x[2] for x in claims)
        if not ok:
            bad.append((it["id"], claims))
    return bad


def test_the_abstention_reasons_python():
    assert _abstain_problems(N) == []


def _alike_cases() -> list:
    """The abstention's cases both ports must read byte for byte alike: all but a U+FEFF lead, the residual #205 names
    and leaves (the port's \\s reads it, the Python's does not; the grid above pins it)."""
    return [it for it in _abstain_cases() if it["id"] != "FEFF|lead|tests"]


def test_the_abstention_is_the_same_in_both_ports(tmp_path):
    """The port's _READ_APART holds the same code points, and on every case each port prints the same claims, byte
    for byte."""
    items = _abstain_cases() + [{"id": "apart", "op": "apart"}]
    got = _port(items, tmp_path)
    assert got["apart"]["flags"] == "" and _class_of_port_source(got["apart"]["source"]) == APART
    for it in _alike_cases():
        assert got[it["id"]]["off"] == _rec(N.gate_diff_text(it["summary"], it["diff"])), it["id"]
    feff = next(it for it in _abstain_cases() if it["id"] == "FEFF|lead|tests")       # the residual, as it stands
    assert [x[1] for x in got[feff["id"]]["off"]["claims"]] == ["VERIFIED", "VERIFIED"]
    assert [x[1] for x in _rec(N.gate_diff_text(feff["summary"], feff["diff"]))["claims"]] == ["CONTRADICTED"] * 2


# R3: a header whose old name holds U+2028 registers the same file in both ports
D_R3 = "diff --git a/x\u2028.py b/y.py\nBinary files a/x\u2028.py and b/y.py differ\n"
R3_WANT = {"verdict": "FAIL", "claims": [["tests_added", "CONTRADICTED", "diff adds 0 test functions, claim says 1"]]}


def test_a_header_holding_a_line_separator_is_read_alike(tmp_path):
    assert _rec(N.gate_diff_text("Added 1 test.", D_R3)) == R3_WANT
    got = _port([{"id": "r3", "op": "gate", "summary": "Added 1 test.", "diff": D_R3}], tmp_path)
    assert got["r3"]["off"] == R3_WANT


# R4: the overlay's count of `def test_` sites follows main's in each view, and its shortcut reads both views alike
D_R4 = _new_file("tests/test_x.py", ["x = 1\u2028def test_a():", "    pass"])


def _lockstep_problems(mod) -> list:
    """Where the overlay's model of the two mains' `def test_` counts parts from them: view 0 against the Python's
    `^\\s*` count, view 1 against the port's (its `^` also after U+2028 and U+2029, its \\s holding U+FEFF; these
    inputs hold none of the white space only CPython's \\s reads), and the facts object's shortcut against both views
    read apart."""
    bad = []
    for d in (D_R4, DIFF_FF_DEF, DIFF_FF_METHOD, D_R3):
        lines = mod._p2a_lines(d, mod._P2A_COARSE)
        views = mod._p2a_views(lines)
        counts = mod._p2a_pairing(views)
        status, blob = mod.parse_unified_diff(d)
        if counts[0][0] != len(re.findall(r"^\s*def test_", blob, re.M)):
            bad.append(("view 0 is not the Python's count", d[-40:]))
        if counts[1][0] != len(re.findall(r"^[\s\ufeff]*def test_", blob.replace("\u2028", "\n").replace("\u2029", "\n"),
                                          re.M)):
            bad.append(("view 1 is not the port's count", d[-40:]))
        unchanged = mod._p2a_context(lines) + mod._p2a_joined(d, " ")
        if mod._P2aFacts(d).pairing() != mod._p2a_pairing(views, False, mod._p2a_joined(d), unchanged):
            bad.append(("the shortcut reads the views otherwise", d[-40:]))
    return bad


def test_the_overlay_counts_as_each_main_does():
    assert _lockstep_problems(N) == []


# ---- 5. plants: each change undone, each told by a check above ---------------------------------------------------------

def _plant(old: str, new: str, after: str = "", name: str = "plant"):
    text = R.lf(R.INSTRUMENT)
    at = text.index(after) if after else 0
    i = text.index(old, at)
    return R.module_from(text[:i] + new + text[i + len(old):], "_p205_" + name), text[:i] + new + text[i + len(old):]


PY_PLANTS = {
    "R1 undone in parse_unified_diff": ("_git_lines(diff_text)", "diff_text.splitlines()", "def parse_unified_diff(", "ff"),
    "R1 undone at the git door's added lines": ("_git_lines(diff_text)", "diff_text.splitlines()", "def gate_diff(", "git"),
    "R1 undone at the git door's listing": ("_git_lines(name_status)", "name_status.splitlines()", "def gate_diff(", "sites"),
    "U+001F dropped from the abstention": ("\\x1c-\\x1f", "\\x1c-\\x1e", "_READ_APART = re.compile(", "apart"),
    "the abstention dropped from the counts": ("    elif apart_why:\n", "    elif False:\n", "def _gate(", "apart"),
    "the abstention dropped from the compatibility reading": ("            if apart:\n", "            if False:\n",
                                                              "def _compat_reading(", "apart"),
    "R4's alike undone": ("                                                         and not _P2A_SEG.search(self.diff_text)",
                          "", "def pairing(self)", "lockstep"),
    "R4's starts read in view 0": ("_p2a_counted(line, lead, v == 1)", "_p2a_counted(line, lead, True)", "", "lockstep"),
    # the slip test_diffgate_path2a_truth.py lists as EQUIVALENT (it moves no record since #205): told here
    "the starts after U+2028 and U+2029 dropped from view 1": (
        "        r = lead.match(line, b).end()\n", "        if b:\n            continue\n        r = lead.match(line, b).end()\n",
        "def _p2a_counted(", "lockstep"),
    "#206 undone": ('_INT_RE = re.compile(r"^[0-9]{1,9}$")', '_INT_RE = re.compile(r"^\\d{1,9}$")', "", "declare"),
}


def _told(kind: str, mod, text: str) -> bool:
    if kind == "ff":
        return _ff_problems(mod, "raw") != []
    if kind == "git":
        return _ff_problems(mod, "git") != []
    if kind == "sites":
        return cut_sites(text) != ([], WANT_SITES)
    if kind == "apart":
        return _abstain_problems(mod) != []
    if kind == "lockstep":
        return _lockstep_problems(mod) != []
    raise AssertionError(kind)


@pytest.mark.parametrize("name", sorted(PY_PLANTS))
def test_a_planted_python_slip_is_told(name):
    old, new, after, kind = PY_PLANTS[name]
    if kind == "declare":
        src = R.lf(R.ROOT / "styxx" / "declare.py")
        assert src.count(old) == 1
        mod = R.module_from(src.replace(old, new), "_p206_declare")
        assert mod.canonical_sentence("files_changed", "\uff13") != (None, WIDE.format("FF13"))
        return
    mod, text = _plant(old, new, after, "x" + str(sorted(PY_PLANTS).index(name)))
    assert _told(kind, mod, text), name


def _r3_undone(text: str) -> str:
    """R3 undone: the port's two header patterns read `.` again where they read `[^\\n]`."""
    lines = text.split("\n")
    out = [x.replace("[^\\n]", ".") if x.startswith(("const _DIFF_GIT = ", "const _BINARY_LINE = ")) else x
           for x in lines]
    assert sum(a != b for a, b in zip(lines, out)) == 2
    return "\n".join(out)


JS_PLANTS = {
    "R3 undone": (None, _r3_undone, "r3"),
    "U+001F dropped from the port's abstention": ("\\u001c-\\u001f", "\\u001c-\\u001e", "apart"),
    "#206's reason undone in the port": ("  if (wide === undefined) return", "  if (true) return", "declare"),
    "the port's starts read in view 0": ("_p2aCounted(line, _P2A_LEADS[v], v === 1)", "_p2aCounted(line, _P2A_LEADS[v], true)",
                                         "lockstep"),
}


@pytest.mark.parametrize("name", sorted(JS_PLANTS))
def test_a_planted_port_slip_is_told(name, tmp_path):
    old, new, kind = JS_PLANTS[name]
    text = R.lf(R.PORT)
    if old is None:
        planted_text = new(text)
    else:
        assert text.count(old) == 1, old
        planted_text = text.replace(old, new)
    planted = tmp_path / "diffgate_planted.js"
    planted.write_bytes(planted_text.encode("utf-8"))
    if kind == "r3":
        got = _port([{"id": "r3", "op": "gate", "summary": "Added 1 test.", "diff": D_R3}], tmp_path, planted)
        assert got["r3"]["off"] != R3_WANT
    elif kind == "apart":
        got = _port(_alike_cases(), tmp_path, planted)
        assert any(got[it["id"]]["off"] != _rec(N.gate_diff_text(it["summary"], it["diff"])) for it in _alike_cases())
    elif kind == "declare":
        got = _port([{"id": "c", "op": "canon", "key": "files_changed", "value": "\uff13"}], tmp_path, planted)
        assert got["c"]["canon"] != list(D.canonical_sentence("files_changed", "\uff13"))
    else:
        if NODE is None:
            pytest.skip("node is not on PATH; the port half of this check cannot run here")
        src = tmp_path / "lock_in.json"
        src.write_text(json.dumps([{"id": "r4", "summary": "", "diff": D_R4}], ensure_ascii=True), encoding="utf-8")
        for port, tag in ((R.PORT, "real"), (planted, "planted")):
            r = subprocess.run([NODE, str(R.DIFFERENTIAL / "check_path2a.js"), "--lockstep", str(src),
                                str(tmp_path / f"lock_{tag}.json"), str(port)], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-2000:]
        real, mine = (json.loads((tmp_path / f"lock_{t}.json").read_text(encoding="utf-8"))["counts"]["r4"]
                      for t in ("real", "planted"))
        assert real["views"] == [0, 1] and real["main"] == 1        # the port's `^` matches after U+2028; CPython's not
        assert mine["views"] != real["views"]

