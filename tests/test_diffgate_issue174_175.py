"""#174 and #175 (fix/diffgate-174-175): what `only_touches` reads.

#174. A rename's old path was registered by neither door: `parse_unified_diff` keys a rename by its new path (status M
at the raw door, R at the git door), so "Only touches src/." over `R080 lib/old.py -> src/new.py` read VERIFIED while
git's own pathspec reading (`git diff --name-status -- . ':(exclude)src'` prints `D lib/old.py`) says the change
leaves `src`. Both ports now read the old path (`rename from` at the raw door, the R row's old path at the git door)
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

The resolution cuts from a prefix only what main's `rstrip("/.")` cut (its trailing run of dots and slashes). A name a
later `..` lays bare keeps its dots, as git reads it: `a/b./c/../.` reads `a/b.`, not its sibling `a/b`, which held
neither `a/b./c/x.py` (a new, false CONTRADICTED) nor kept out `a/b/y.py` (a new, false VERIFIED). So every reading of
a prefix with no inner dot segment is main's reading or a directory above it. Where the resolved reading is not main's
and that cut run takes part of a name git reads whole (`...`, or the dots of the name written last beyond one final
period), git's reading, every name whole, is read beside it, and the claim abstains where a changed path lies under one
and not the other; where none does, the two give one verdict and the cut reading decides.

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
DOTNAME = _edit("a/b./c/x.py")      # a directory whose name ends in a dot, which git writes and reads as it is
CUT_TAIL = ", and a changed path lies under one of the two readings and not the other (#175)"
SIBLING = _edit("a/b/y.py")
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
    # a name a `..` lays bare keeps its dots, as git reads it (the review of this batch, F1)
    (_scope("175-dotname-pop", "a/b./c/../.", DOTNAME), "VERIFIED", "all changed paths under prefix"),
    (_scope("175-dotname-pop-tick", "`a/b./c/../`", DOTNAME), "VERIFIED", "all changed paths under prefix"),
    (_scope("175-dotname-two-pops", "a/b./c/d/../../.", _edit("a/b./c/d/x.py")), "VERIFIED",
     "all changed paths under prefix"),
    (_scope("175-dotname-two-dots", "a/v1../c/../.", _edit("a/v1../c/x.py")), "VERIFIED", "all changed paths under prefix"),
    (_scope("175-dotname-second-prefix", "src/ and a/b./c/../.", DOTNAME), "VERIFIED", "all changed paths under prefix"),
    (_scope("175-dotname-last-dotdot", "a/b./c/..", DOTNAME), "VERIFIED", "all changed paths under prefix"),
    (_scope("175-dotname-sibling", "a/b./c/../.", SIBLING), "CONTRADICTED", "paths outside 'a/b.': ['a/b/y.py']"),
    (_scope("175-dotname-sibling-v1", "releases/v1./notes/../.", _edit("releases/v1/x.md")), "CONTRADICTED",
     "paths outside 'releases/v1.': ['releases/v1/x.md']"),
    # main's cut run holds the dots of the name written last: main's reading, `b`, is kept
    (_scope("175-dotname-in-the-cut-run", "`b./.../../`", _edit("b/src/x.py")), "VERIFIED",
     "all changed paths under prefix"),
    # the resolved reading would rest on a cut git does not make, and a changed path lies under one reading only
    (_scope("175-cut-run-name", "`docs/../a/.../`", _edit("a/x.py")), "UNCHECKABLE",
     "prefix 'docs/../a/.../' ends in a run of dots and slashes that this template cuts as a sentence's period or an "
     "ellipsis and git reads as part of the name '...'" + CUT_TAIL),
    (_scope("175-cut-run-last-name", "src/../b./.", _edit("b/x.py")), "UNCHECKABLE",
     "prefix 'src/../b./.' ends in a run of dots and slashes that this template cuts as a sentence's period or an "
     "ellipsis and git reads as part of the name 'b.'" + CUT_TAIL),
    (_scope("175-cut-run-two-dots", "src/../b..", _edit("b/x.py")), "UNCHECKABLE",
     "prefix 'src/../b..' ends in a run of dots and slashes that this template cuts as a sentence's period or an "
     "ellipsis and git reads as part of the name 'b..'" + CUT_TAIL),
    # git's reading keeps a leading run of dots, which `_norm` drops (#121): `...` here is git's name, not the root
    (_scope("175-cut-run-leading-dots", "c/../.../..", _edit(".../setup.py")), "UNCHECKABLE",
     "prefix 'c/../.../..' ends in a run of dots and slashes that this template cuts as a sentence's period or an "
     "ellipsis and git reads as part of the name '...'" + CUT_TAIL),
    # the two readings hold the same changed paths: one verdict, the cut reading's (main printed it)
    (_scope("175-cut-run-readings-agree", "src/./x.py..", _edit("docs/y.md")), "CONTRADICTED",
     "paths outside 'src/x.py': ['docs/y.md']"),
    (_scope("175-cut-run-readings-agree-name", "`docs/../a/.../`", _edit("lib/x.py")), "CONTRADICTED",
     "paths outside 'a': ['lib/x.py']"),
    # a bare file name is read on the cut reading, before the two readings, as before the review
    (_scope("175-bare-before-the-cut-run", "x/../setup.py..", _edit("pkg/setup.py")), "UNCHECKABLE",
     "prefix 'x/../setup.py..' resolves to the file name 'setup.py' at the repository root, which this template would "
     "match in any directory (#175)"),
    # one final dot is a sentence's period, as main reads it
    (_scope("175-one-final-dot", "src/../b.", _edit("b/x.py")), "VERIFIED", "all changed paths under prefix"),
    # the reasons say what was read (M1, M2 of the review)
    (_scope("175-empty-segment-pop", "docs//../src/.", SRC), "UNCHECKABLE",
     "prefix 'docs//../src/.' has a '..' segment after an empty segment, which this template does not resolve (#175)"),
    (_scope("175-two-prefixes-the-second-parts", "`a/b/..` and `src/..` here.", _edit("a/b/x.py") + _edit("lib/y.py")),
     "UNCHECKABLE",
     "prefix 'src/..' ends in '..', which reads as the directory before it (its last dot a sentence's period) and as "
     "that directory's parent (git's reading), and the claim holds under one and not the other (#175)"),
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


def _plumbed(tmp, name: str, trees: list) -> tuple:
    """A repository built with git plumbing (hash-object, mktree, commit-tree; no checkout, so a name a Windows file
    system refuses, `b.`, is written as git stores it): one commit per {path: content} tree, each the next one's
    parent. Returns the repository and the commits."""
    repo = tmp / name
    repo.mkdir()
    _git(repo, "init", "-q")

    def put(d) -> str:
        rows = []
        for k, v in sorted(d.items()):
            sha = put(v) if isinstance(v, dict) else subprocess.run(
                [GIT, "hash-object", "-w", "--stdin"], cwd=repo, input=v.encode("utf-8"), capture_output=True,
                check=True).stdout.decode().strip()
            rows.append(f"{'040000 tree' if isinstance(v, dict) else '100644 blob'} {sha}\t{k}")
        return subprocess.run([GIT, "mktree"], cwd=repo, input=("\n".join(rows) + "\n").encode("utf-8"),
                              capture_output=True, check=True).stdout.decode().strip()

    commits, parent = [], []
    for files in trees:
        root: dict = {}
        for p, t in files.items():
            cur = root
            for seg in p.split("/")[:-1]:
                cur = cur.setdefault(seg, {})
            cur[p.split("/")[-1]] = t
        c = subprocess.run([GIT, "-c", "user.name=t", "-c", "user.email=t@t", "commit-tree", "-m", "c", put(root),
                            *parent], cwd=repo, capture_output=True, check=True).stdout.decode().strip()
        commits.append(c)
        parent = ["-p", c]
    return repo, commits


def test_175_a_name_a_dotdot_lays_bare_reads_as_git_reads_it(tmp_path):
    """A directory named `b.` or `v1.` under a later `..`: git's own pathspec reading decides each claim (empty
    `git diff --name-status -- . ':(exclude)P'`: every changed path lies under P), and the gate's decided verdict at
    both doors, main's level and the final record, in both ports, agrees with it; the raw door's records equal the git
    door's and the port's equal the Python's."""
    if GIT is None:
        _missing("git", "the pathspec oracle")
    trees = [{"a/b./c/x.py": "a\n", "a/b/y.py": "a\n", "releases/v1./notes/x.md": "a\n",
              "releases/v1/x.md": "a\n"}]
    for p in ("a/b./c/x.py", "a/b/y.py", "releases/v1./notes/x.md", "releases/v1/x.md"):   # one changed path a commit
        trees.append(dict(trees[-1], **{p: "b\n"}))
    repo, commits = _plumbed(tmp_path, "dotname", trees)
    old_py = R.module_from(R.reconstruct_py(R.lf(R.INSTRUMENT)), "_b2_main_dotname")
    cases = []
    for lo, hi in zip(commits, commits[1:]):
        text = _git(repo, "diff", lo, hi)
        for prefix in ("a/b./c/../.", "`a/b./c/../`", "a/b./c/d/../../.", "releases/v1./notes/../.", "a/b./c/.."):
            claim = f"Only touches {prefix}"
            want = None
            for p in [x for c in N.gate_diff_text(claim, "").claims if c.kind == "only_touches"
                      for x in (c.detail.get("prefix"), c.detail.get("prefix2")) if x]:
                rest = _git(repo, "diff", "--name-status", lo, hi, "--", ".", f":(exclude){p}").strip()
                want = "VERIFIED" if rest == "" else "CONTRADICTED"
            git_door = _only(_rec(N.gate_diff(claim, str(repo), lo, hi)))
            raw = _py(N, {"summary": claim, "diff": text})["off"]
            main = _rec(old_py.gate_diff_text(claim, text))
            for rec in (git_door, _only(raw), _only(main)):
                assert rec[0][1] in (want, "UNCHECKABLE"), (prefix, lo[:7], want, rec)
            assert git_door == _only(raw), (prefix, git_door, raw)
            cases.append({"id": f"{prefix}@{lo[:7]}", "summary": claim, "diff": text, "py": raw, "main": main})
    js = _port(cases, tmp_path, tag="dotname")
    for c in cases:
        assert js[c["id"]]["off"] == c["py"] and js[c["id"]]["main"] == c["main"], c["id"]
    assert all(_only(c["py"])[0][1] != "UNCHECKABLE" for c in cases), [(c["id"], _only(c["py"])) for c in cases]


# ---- neither adds an accusation ---------------------------------------------------------------------------------------

PREFIXES = ["src/", "src/.", "src/b/", "./src/./b/", "src/../src/", "src/b/..", "src/b/...", "../src/", "src/../../x/",
            "lib/../src/", "./", "docs/..", "a/b./c/../.", "a/b./c/..", "a/b./c/d/../../.", "src/../a/b./."]
DIFFS = [_rename("lib/old.py", "src/new.py"), _rename("src/a/old.py", "src/b/new.py"),
         _rename("lib/old.py", "src/new.py", edit=False), _edit("src/b/x.py"), _edit("src/x.py") + _edit("docs/y.md"),
         _rename(".lib/old.py", "src/new.py"), _rename("src/old.py", "lib/new.py"), _edit("a/b./c/x.py"),
         _edit("a/b/y.py"), _rename("a/b/old.py", "a/b./c/new.py")]


def test_neither_fix_adds_a_contradicted_verdict(tmp_path):
    """Over a grid of prefixes, rename shapes and directories whose names end in a dot, both ports, main's level and the
    final record: a claim reads CONTRADICTED only where a status key lies outside every reading of its prefix
    (`_baseline`) and outside the prefix as main read it before #175, as text (`_main_held`, which reads no reading of
    this batch): every CONTRADICTED here, main printed. And no claim reads VERIFIED where main's text reading and git's
    own reading of the prefix (`_git_reading`) both put a changed path outside it."""
    cases = [{"id": f"g{i}-{j}", "summary": f"Only touches {p}", "diff": d}
             for (i, p), (j, d) in itertools.product(enumerate(PREFIXES), enumerate(DIFFS))]
    old_py = R.module_from(R.reconstruct_py(R.lf(R.INSTRUMENT)), "_b2_main_now")
    js = _port(cases, tmp_path)
    for c in cases:
        for rec in (_py(N, c)["off"], _rec(old_py.gate_diff_text(c["summary"], c["diff"])), js[c["id"]]["off"],
                    js[c["id"]]["main"]):
            assert rec["claims"][0][1] != "CONTRADICTED" or (_baseline(c) and _main_held(c)), (c, rec)
            assert rec["claims"][0][1] != "VERIFIED" or not (_main_held(c) and _git_outside(c)), (c, rec)


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


def _main_held(case) -> bool:
    """Whether the prefix as main read it before #175 (after `_norm`, `rstrip("/.")`, compared as text) leaves a status
    key outside it: where main read the claim CONTRADICTED."""
    status, _ = N.parse_unified_diff(case["diff"])
    prefix = N._norm(case["summary"][len("Only touches "):]).rstrip("/.")
    return any(not N._path_inside(p, prefix) for p in status)


def _git_reading(prefix: str):
    """git's lexical reading of a pathspec at the top of the tree: `.` and empty segments dropped, `..` pops; None
    where it climbs out (git refuses it)."""
    out: list = []
    for s in prefix.replace("\\", "/").split("/"):
        if s in ("", "."):
            continue
        if s == "..":
            if not out:
                return None
            out.pop()
            continue
        out.append(s)
    return "/".join(out)


def _git_outside(case) -> bool:
    """Whether git's reading of the prefix, over git's listing without rename detection (every status key and every
    old path), leaves a changed path outside it."""
    status, _ = N.parse_unified_diff(case["diff"])
    g = _git_reading(case["summary"][len("Only touches "):].lower())
    if g is None:
        return False
    return any(not (g == "" or p == g or p.startswith(g + "/"))
               for p in list(status) + N.parse_unified_diff_renames(case["diff"]))


MIRROR_RUNNER = r"""
"use strict";
const fs = require("fs");
const vm = require("vm");
const [port, inp, out] = process.argv.slice(2);
const P = vm.runInContext(fs.readFileSync(port, "utf8") + "\n;({_scopeReadings, _p2aReadings, _p2aA})", vm.createContext({}), { filename: port });
fs.writeFileSync(out, JSON.stringify(JSON.parse(fs.readFileSync(inp, "utf8")).map(([p, ch]) =>
  [P._scopeReadings(p, ch)[0], P._p2aReadings(p, P._p2aA, ch)])));
"""


def test_the_overlay_reads_a_prefix_as_main_does(tmp_path):
    """The PATH-2a overlay's model of main's scope readings in main's key space (`_p2a_readings` with `_p2a_A`, the
    port's `_p2aReadings` with `_p2aA`) against main's own (`_scope_readings`, `_scopeReadings`), on the grid's prefixes
    and on seeded prefixes holding dots in every way a segment can, over changed paths with and without a dot segment:
    main's abstention is the overlay's None and main's readings are its readings, in each port, and the two ports' main
    readings agree. The overlay's own abstentions where main abstains change no record while the mirror is exact, so
    no claim above can tell them; this does."""
    import random
    rng = random.Random(1751742)
    segs = ["a", "b", "b.", "v1..", "...", "src", "docs", ".", "..", "", "c", "x.py"]
    cases = [(p, ["src/b/x.py", "a/b./c/x.py"]) for p in PREFIXES]
    for _ in range(3000):
        p = rng.choice(["/", "\\"]).join(rng.choice(segs) for _ in range(rng.randint(1, 6)))
        cases.append((p + rng.choice(["", ".", "/.", "/", "..", "/.."]),
                      ["a/b/x.py", "b./c/y.md"] + (["a/./b/z.py"] if rng.random() < 0.2 else [])))
    py = [[N._scope_readings(p, ch)[0], N._p2a_readings(p, N._p2a_A, ch)] for p, ch in cases]
    assert [c for c, (m, o) in zip(cases, py) if m != o] == []
    if NODE is None:
        _missing("node", "the port half of this check")
    runner, src, out = tmp_path / "mirror.js", tmp_path / "mirror_in.json", tmp_path / "mirror_out.json"
    runner.write_text(MIRROR_RUNNER, encoding="utf-8")
    src.write_text(json.dumps(cases, ensure_ascii=True), encoding="utf-8")
    r = subprocess.run([NODE, str(runner), str(R.PORT), str(src), str(out)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    js = json.loads(out.read_text(encoding="utf-8"))
    assert [c for c, (m, o) in zip(cases, js) if m != o] == []
    assert [c for c, a, b in zip(cases, py, js) if a[0] != b[0]] == []
