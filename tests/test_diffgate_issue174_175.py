"""#174 and #175 (fix/diffgate-174-175): what `only_touches` reads.

#174. A rename's old path was registered by neither door: `parse_unified_diff` keys a rename by its new path (status M
at the raw door, R at the git door), so "Only touches src/." over `R080 lib/old.py -> src/new.py` read VERIFIED while
git's own pathspec reading (`git diff --name-status -- . ':(exclude)src'` prints `D lib/old.py`) says the change
leaves `src`. Both ports now read the old path (`rename from` at the raw door, the R row's first path at the git door)
beside the status keys, in `only_touches` only, and where every status key lies under the prefixes and an old path
does not, the claim abstains (UNCHECKABLE, #174). No CONTRADICTED is added: prose `only_touches` accusations measured
2/11 and 2/8 precision on BENCH-2, and a C-quoted old path (#169) would read outside a prefix that holds it. The count,
the status letters and the path claims read as before. The abstention's reason is one fixed string that names neither
the prefixes nor the old paths: printed, they part the two ports where `repr` and `pyRepr` escape a code point apart
(U+0085) and where `lower()` and `toLowerCase()` fold one apart (U+A7CB on CPython 3.12), which main's own VERIFIED
reason, naming no path, never did.

#175. The prefix was compared as text after `_norm` and `rstrip("/.")`: a `.` or `..` segment inside it was kept and
accused the paths it holds, and a trailing `/..` was cut off. Its segments now resolve as git resolves a pathspec's,
after `_norm` (whose reading of the leading run is #121's and is not changed); a last segment `..` reads both as the
parent and as the directory before it with its last dot a sentence's period, and the claim takes the verdict both give;
a last segment of three or more dots, a `..` that climbs, the root, a file name the resolution made bare, and a changed
path that holds a dot segment abstain. No CONTRADICTED is added either: a resolved prefix holds every path the text
prefix held unless a changed path holds a dot segment, and then the claim abstains.

The PATH-2a overlay's model of main's scope reading (`_p2a_scope` / `_p2aScope`) reads both, in its key spaces A and K,
so `unreproduced` does not fire on the repaired verdicts and #121's withholding still reads.
"""
from __future__ import annotations

import itertools
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


def _missing(tool: str, half: str):
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        pytest.fail(f"{tool} is not on PATH under CI; {half} did not run")
    pytest.skip(f"{tool} is not on PATH; {half} cannot run here")


RUNNER = r"""
"use strict";
const fs = require("fs");
const vm = require("vm");
const [port, inp, out] = process.argv.slice(2);
const P = vm.runInContext(fs.readFileSync(port, "utf8") + "\n;({gateDiffText, _gateDiffTextMain})", vm.createContext({}), { filename: port });
const rec = g => ({ verdict: g.verdict, claims: g.claims.map(c => [c.kind, c.verdict, c.why]) });
const res = JSON.parse(fs.readFileSync(inp, "utf8")).map(it => {
  const o = { id: it.id };
  try {
    o.off = rec(P.gateDiffText(it.summary, it.diff));
    o.on = rec(P.gateDiffText(it.summary, it.diff, { strict: true }));
    o.main = rec(P._gateDiffTextMain(it.summary, it.diff));
  } catch (e) {
    o.error = String((e && e.message) || e);
  }
  return o;
});
fs.writeFileSync(out, JSON.stringify(res));
"""


def _port(cases, tmp_path, port: Path = R.PORT, tag: str = "run") -> dict:
    if NODE is None:
        _missing("node", "the port half of this check")
    runner = tmp_path / "runner_174_175.js"
    runner.write_text(RUNNER, encoding="utf-8")
    src, out = tmp_path / f"{tag}_in.json", tmp_path / f"{tag}_out.json"
    src.write_text(json.dumps([{k: c[k] for k in ("id", "summary", "diff")} for c in cases], ensure_ascii=True),
                   encoding="utf-8")
    r = subprocess.run([NODE, str(runner), str(port), str(src), str(out)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    got = {x["id"]: x for x in json.loads(out.read_text(encoding="utf-8"))}
    assert not [x for x in got.values() if "error" in x], [x for x in got.values() if "error" in x][:3]
    return got


def _rec(g) -> dict:
    return {"verdict": g.verdict, "claims": [[c.kind, c.verdict, c.why] for c in g.claims]}


def _py(mod, case) -> dict:
    return {"off": _rec(mod.gate_diff_text(case["summary"], case["diff"])),
            "on": _rec(mod.gate_diff_text(case["summary"], case["diff"], strict=True))}


def _edit(path: str, old: str = "a", new: str = "b") -> str:
    return (f"diff --git a/{path} b/{path}\nindex 1111111..2222222 100644\n--- a/{path}\n+++ b/{path}\n"
            f"@@ -1 +1 @@\n-{old}\n+{new}\n")


def _rename(old: str, new: str, edit: bool = True) -> str:
    head = f"diff --git a/{old} b/{new}\nsimilarity index {80 if edit else 100}%\nrename from {old}\nrename to {new}\n"
    if not edit:
        return head
    return head + f"index 1111111..2222222 100644\n--- a/{old}\n+++ b/{new}\n@@ -1,2 +1,2 @@\n x\n-a\n+b\n"


def _scope(case_id: str, prefix: str, diff: str) -> dict:
    return {"id": case_id, "summary": f"Only touches {prefix}", "diff": diff}


def _only(rec) -> list:
    return [c for c in rec["claims"] if c[0] == "only_touches"]


# ---- #174 -------------------------------------------------------------------------------------------------------------

WHY_174 = ("every changed path lies under the prefix, but not the old path of a rename; git lists it as deleted where it "
           "reads no rename, and this template does not decide whether the claim covers it (#174)")
C174 = [
    # the reporter's case: R080 lib/old.py -> src/new.py with one line edited, git's own bytes
    (_scope("174-reporter", "src/.", _rename("lib/old.py", "src/new.py")), "UNCHECKABLE",
     WHY_174),
    # a pure rename: a header with no ---/+++ pair (BIN-1's flush)
    (_scope("174-pure", "src/.", _rename("lib/old.py", "src/new.py", edit=False)), "UNCHECKABLE",
     WHY_174),
    # both sides under the prefix: as main
    (_scope("174-inside", "src/.", _rename("src/a/old.py", "src/b/new.py")), "VERIFIED", "all changed paths under prefix"),
    # the old side outside the narrower prefix
    (_scope("174-inside-narrow", "src/b/.", _rename("src/a/old.py", "src/b/new.py")), "UNCHECKABLE",
     WHY_174),
    # a status key outside too: main's CONTRADICTED and its reason, byte for byte
    (_scope("174-also", "src/.", _edit("docs/x.md") + _rename("lib/old.py", "src/new.py")), "CONTRADICTED",
     "paths outside 'src': ['docs/x.md']"),
    # a copy leaves its source in place: `copy from` is not read
    (_scope("174-copy", "src/.", "diff --git a/lib/a.py b/src/b.py\nsimilarity index 90%\ncopy from lib/a.py\n"
            "copy to src/b.py\nindex 1111111..2222222 100644\n--- a/lib/a.py\n+++ b/src/b.py\n@@ -1 +1 @@\n-a\n+b\n"),
     "VERIFIED", "all changed paths under prefix"),
]


@pytest.mark.parametrize("case,verdict,why", C174, ids=[c[0]["id"] for c in C174])
def test_174_the_cases_in_python(case, verdict, why):
    got = _py(N, case)
    for strict in ("off", "on"):
        assert _only(got[strict]) == [["only_touches", verdict, why]], (case["id"], strict, got[strict])


def test_174_the_cases_in_the_port_and_the_two_alike(tmp_path):
    js = _port([c for c, _v, _w in C174], tmp_path)
    for case, verdict, why in C174:
        assert {"off": js[case["id"]]["off"], "on": js[case["id"]]["on"]} == _py(N, case), case["id"]
        assert _only(js[case["id"]]["off"]) == [["only_touches", verdict, why]], case["id"]


def _git(repo, *args, cfg=()):
    c = [x for k, v in cfg for x in ("-c", f"{k}={v}")]
    r = subprocess.run([GIT, *c, *args], cwd=repo, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    return r.stdout


def _repo(tmp, name, base: dict, head: dict, moves=()):
    repo = tmp / name
    repo.mkdir()
    _git(repo, "init", "-q")
    for k, v in (("user.email", "t@t"), ("user.name", "t"), ("core.autocrlf", "false")):
        _git(repo, "config", k, v)
    for p, t in base.items():
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        (repo / p).write_bytes(t.encode("utf-8"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    for old, new in moves:
        (repo / new).parent.mkdir(parents=True, exist_ok=True)
        _git(repo, "mv", old, new)
    for p, t in head.items():
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        (repo / p).write_bytes(t.encode("utf-8"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "head")
    return repo


BODY = "def f():\n    return 1\n\ndef g():\n    return 2\n\ndef h():\n    return 3\n"


def _rn(old: str, new: str) -> str:
    return (f"diff --git a/{old} b/{new}\nsimilarity index 90%\nrename from {old}\nrename to {new}\n--- a/{old}\n"
            f"+++ b/{new}\n@@ -1 +1 @@\n-a\n+b\n")


def test_174_the_reason_names_no_path_so_the_two_ports_print_it_alike(tmp_path):
    """An old path holding U+0085 (`repr` escapes it, `pyRepr` does not) or U+A7CB (CPython 3.12's `lower()` leaves it,
    Node's `toLowerCase()` folds it): the reason is the same string in both ports, at main's level and in the final
    record, so the abstention parts the two readers on neither."""
    cases = [{"id": "174-u0085", "summary": "Only touches docs/.", "diff": _rn("src/x\u0085y.py", "docs/y.md")},
             {"id": "174-ua7cb", "summary": "Only touches docs/.", "diff": _rn("\ua7cba/x.py", "docs/y.md")}]
    old_py = R.module_from(R.reconstruct_py(R.lf(R.INSTRUMENT)), "_b2_main_reason")
    js = _port(cases, tmp_path)
    for c in cases:
        main = _rec(old_py.gate_diff_text(c["summary"], c["diff"]))
        assert _only(main) == [["only_touches", "UNCHECKABLE", WHY_174]], (c["id"], main)
        assert js[c["id"]]["main"] == main, c["id"]
        assert {"off": js[c["id"]]["off"], "on": js[c["id"]]["on"]} == _py(N, c), c["id"]


def test_174_the_git_door_and_the_text_door(tmp_path):
    """The reporter's repository: under git's default rename reading both doors abstain on the scope claim with one
    reason; under diff.renames=false the git door reads `D lib/old.py` and keeps main's CONTRADICTED, as git's
    --no-renames text does at the raw door (the config dependence the batch leaves, named)."""
    if GIT is None:
        _missing("git", "the git door")
    repo = _repo(tmp_path, "r", {"lib/old.py": BODY}, {"src/new.py": BODY.replace("3", "4")},
                 [("lib/old.py", "src/new.py")])
    s = "Only touches src/."
    want = ["only_touches", "UNCHECKABLE", WHY_174]
    assert _only(_rec(N.gate_diff(s, str(repo), "HEAD~1", "HEAD"))) == [want]
    assert _only(_rec(N.gate_diff_text(s, _git(repo, "diff", "HEAD~1..HEAD")))) == [want]
    _git(repo, "config", "diff.renames", "false")
    off = _only(_rec(N.gate_diff(s, str(repo), "HEAD~1", "HEAD")))
    assert off == [["only_touches", "CONTRADICTED", "paths outside 'src': ['lib/old.py']"]], off
    assert _only(_rec(N.gate_diff_text(s, _git(repo, "diff", "--no-renames", "HEAD~1..HEAD")))) == off


def test_174_a_quoted_old_path_abstains_and_does_not_accuse(tmp_path):
    """core.quotePath (git's default) C-quotes a non-ASCII old path; main reads it as written (#169), outside `src`
    though the rename is inside it. The claim abstains; an accusation here would be false."""
    if GIT is None:
        _missing("git", "the git door")
    repo = _repo(tmp_path, "q", {"src/café.py": BODY}, {"src/cafe.py": BODY.replace("3", "4")},
                 [("src/café.py", "src/cafe.py")])
    for rec in (_rec(N.gate_diff("Only touches src/.", str(repo), "HEAD~1", "HEAD")),
                _rec(N.gate_diff_text("Only touches src/.", _git(repo, "diff", "HEAD~1..HEAD")))):
        assert [c[1] for c in _only(rec)] == ["UNCHECKABLE"], rec


def test_174_the_overlay_reads_the_old_path_in_both_key_spaces(tmp_path):
    """path2:121-a-pure-rename-to-a-dotted-name: eslintrc.json -> .eslintrc.json. main's key drops the dot, so main
    reads the old path under ".eslintrc.json" and keeps VERIFIED, which git's reading refutes (the change deletes
    eslintrc.json); with leading dots kept the old path is outside, and the overlay withholds it (`only`, #121)."""
    case = next(c for c in R.repro_cases() if c["id"] == "path2:121-a-pure-rename-to-a-dotted-name")
    got = _py(N, case)["off"]["claims"][1]
    assert got[1] == "UNCHECKABLE" and R.phrase_key(got[2], N._P2A_PHRASES) == "only", got
    js = _port([case], tmp_path)[case["id"]]
    assert js["off"]["claims"][1] == got


# ---- #175 -------------------------------------------------------------------------------------------------------------

DOCS = _edit("docs/guide/a.md")
SRC = _edit("src/x.py", "x = 1", "x = 2")
C175 = [
    # the reporter's three
    (_scope("175-a", "src/../docs/.", DOCS), "VERIFIED", "all changed paths under prefix"),
    (_scope("175-b", "./docs/./guide/.", DOCS), "VERIFIED", "all changed paths under prefix"),
    (_scope("175-c", "`docs/..`.", SRC), "UNCHECKABLE",
     "prefix 'docs/..' ends in '..', which reads as the directory before it (its last dot a sentence's period) and as "
     "that directory's parent (git's reading), and the claim holds under one and not the other (#175)"),
    # the two readings of a last `..` agree
    (_scope("175-c-both-hold", "`docs/..`.", DOCS), "VERIFIED", "all changed paths under prefix"),
    (_scope("175-c-both-fail", "`a/b/..`.", SRC), "CONTRADICTED", "paths outside 'a/b': ['src/x.py']"),
    # the triage's probes
    (_scope("175-file", "src/./x.py.", SRC), "VERIFIED", "all changed paths under prefix"),
    (_scope("175-back-and-forth", "docs/guide/../guide/.", DOCS), "VERIFIED", "all changed paths under prefix"),
    (_scope("175-root", "./.", SRC), "UNCHECKABLE",
     "prefix './.' names the repository root, which holds every changed path (#175)"),
    (_scope("175-root-tick", "`./`.", SRC), "UNCHECKABLE",
     "prefix './' names the repository root, which holds every changed path (#175)"),
    (_scope("175-climb", "src/../../docs/.", DOCS), "UNCHECKABLE",
     "prefix 'src/../../docs/.' has a '..' segment with no directory before it, so it names a directory outside the "
     "one it is read from (#175)"),
    (_scope("175-ellipsis", "docs/guide/...", DOCS), "UNCHECKABLE",
     "prefix 'docs/guide/...' ends in '...', which git reads as a directory of that name and prose as a sentence's "
     "period or an ellipsis (#175)"),
    (_scope("175-other-dir", "docs/../src/.", SRC), "VERIFIED", "all changed paths under prefix"),
    (_scope("175-bare", "x/../setup.py.", _edit("pkg/setup.py")), "UNCHECKABLE",
     "prefix 'x/../setup.py.' resolves to the file name 'setup.py' at the repository root, which this template would "
     "match in any directory (#175)"),
    (_scope("175-dot-key", "docs/./guide/.", _edit("docs/./guide/a.md")), "UNCHECKABLE",
     "prefix 'docs/./guide/.' and the changed path 'docs/./guide/a.md' each hold a '.' or '..' segment; git writes no "
     "such path, and the two are not compared (#175)"),
    # controls: main's reading, byte for byte
    (_scope("175-literal-dots", "src/.../tests/.", SRC), "CONTRADICTED", "paths outside 'src/.../tests': ['src/x.py']"),
    (_scope("175-leading-run", "../docs/.", SRC), "CONTRADICTED", "paths outside 'docs': ['src/x.py']"),
    (_scope("175-plain", "docs/.", DOCS), "VERIFIED", "all changed paths under prefix"),
]


@pytest.mark.parametrize("case,verdict,why", C175, ids=[c[0]["id"] for c in C175])
def test_175_the_cases_in_python(case, verdict, why):
    got = _py(N, case)
    for strict in ("off", "on"):
        assert _only(got[strict]) == [["only_touches", verdict, why]], (case["id"], strict, got[strict])


def test_175_the_cases_in_the_port_and_the_two_alike(tmp_path):
    js = _port([c for c, _v, _w in C175], tmp_path)
    for case, verdict, why in C175:
        assert {"off": js[case["id"]]["off"], "on": js[case["id"]]["on"]} == _py(N, case), case["id"]
        assert _only(js[case["id"]]["off"]) == [["only_touches", verdict, why]], case["id"]


def test_175_git_reads_each_prefix_as_the_gate_does(tmp_path):
    """For each decided case whose diff is a git diff of one repository, git's own pathspec decides: VERIFIED only where
    `git diff --name-status -- . ':(exclude)P'` lists nothing, CONTRADICTED only where it lists a path."""
    if GIT is None:
        _missing("git", "the pathspec oracle")
    repo = _repo(tmp_path, "t", {"docs/guide/a.md": "a\n", "src/x.py": "x = 1\n"}, {"docs/guide/a.md": "b\n"})
    for prefix, verdict in (("src/../docs", "VERIFIED"), ("./docs/./guide", "VERIFIED"), ("docs/..", "VERIFIED"),
                            ("docs/guide/../guide", "VERIFIED"), ("docs/../src", "CONTRADICTED")):
        rest = _git(repo, "diff", "--name-status", "HEAD~1..HEAD", "--", ".", f":(exclude){prefix}").strip()
        assert (rest == "") == (verdict == "VERIFIED"), (prefix, rest)
        got = _only(_rec(N.gate_diff(f"Only touches `{prefix}`.", str(repo), "HEAD~1", "HEAD")))
        assert got[0][1] == verdict, (prefix, got)


# ---- neither adds an accusation ---------------------------------------------------------------------------------------

PREFIXES = ["src/", "src/.", "src/b/", "./src/./b/", "src/../src/", "src/b/..", "src/b/...", "../src/", "src/../../x/",
            "lib/../src/", "./", "docs/.."]
DIFFS = [_rename("lib/old.py", "src/new.py"), _rename("src/a/old.py", "src/b/new.py"),
         _rename("lib/old.py", "src/new.py", edit=False), _edit("src/b/x.py"), _edit("src/x.py") + _edit("docs/y.md"),
         _rename(".lib/old.py", "src/new.py"), _rename("src/old.py", "lib/new.py")]


def test_neither_fix_adds_a_contradicted_verdict(tmp_path):
    """Over a grid of prefixes and rename shapes, both ports, main's level and the final record: a claim reads
    CONTRADICTED only where a status key lies outside every reading of its prefix (`_baseline`), the one shape in
    which main read it CONTRADICTED too (the measured comparison against main is the PR's)."""
    cases = [{"id": f"g{i}-{j}", "summary": f"Only touches {p}", "diff": d}
             for (i, p), (j, d) in itertools.product(enumerate(PREFIXES), enumerate(DIFFS))]
    old_py = R.module_from(R.reconstruct_py(R.lf(R.INSTRUMENT)), "_b2_main_now")
    js = _port(cases, tmp_path)
    for c in cases:
        for rec in (_py(N, c)["off"], _rec(old_py.gate_diff_text(c["summary"], c["diff"])), js[c["id"]]["off"],
                    js[c["id"]]["main"]):
            assert rec["claims"][0][1] != "CONTRADICTED" or _baseline(c), (c, rec)


def _baseline(case) -> bool:
    """Whether a reading of every changed path, the old side of a rename included, as status keys (git's --no-renames
    reading) puts some path outside the prefix under every reading of it: the only shape a CONTRADICTED may take."""
    status, _ = N.parse_unified_diff(case["diff"])
    paths = list(status) + N.parse_unified_diff_renames(case["diff"])
    prefix = case["summary"][len("Only touches "):]
    read, why = N._scope_readings(prefix, paths)
    if why is not None:
        return False
    return all(any(not (x == "" or N._path_inside(p, x)) for p in status) for x in read)
