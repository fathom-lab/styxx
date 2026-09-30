"""PATH-2a: main's diff gate, unchanged, plus an overlay that only abstains (NOTE_path2a_abstain_overlay_2026_09_30,
NOTE_path2a_second_pass_2026_09_30, NOTE_path2a_third_pass_2026_09_30, NOTE_path2a_fourth_pass_2026_09_30).

The reference in every test here is `main` itself: this checkout's `styxx/diffgate.py` and `web/gate/diffgate.js` with
the PATH-2a block cut out and the door hooks reverted, asserted to hash to the files on `origin/main` 1cde8b82
(tests/_p2a_ref.py). No copy of main is committed.

(A) by construction -- on every committed input, both strict modes, both ports and the git door, each record equals
    main's except that a decided verdict in REACH may become UNCHECKABLE with the overlay's reason, and the gate
    verdict is main's formula over the claims; each claim reads the same under --strict as without it; where main
    raises, the branch raises the same exception type.
(C) cross-port -- a decision reads the claim's kind, verdict and detail and the door's bytes, never the claim's text;
    so wherever main's two ports give claims the same kind, verdict and detail -- by position, matched across the two
    lists, or anywhere in either list -- the overlay's verdict and phrase are the same (by construction, asserted).
    Measured on the committed inputs and asserted there only: the same for claims with the same kind, verdict and
    text, and for claims left over whose details differ but nest and lie in each other's text (one match read two
    ways, the `extract` guards' work); where every claim pairs so, the gate verdicts agree. The constant tables the overlay
    leans on are pinned by enumeration on the running engines.
And the static facts: the reconstruction, the self-checks over each block's source, the error fallback, the
reproductions, the cost per call, and what must not move (the demo, the committed capsules, charon's lines, the
bookmarklet source). Coverage (B) is in tests/test_diffgate_path2a_truth.py.
"""
from __future__ import annotations

import collections
import importlib.util
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from contextlib import redirect_stdout

import pytest

import styxx.diffgate as N
from tests import _p2a_ref as R

NODE = shutil.which("node")
GIT = shutil.which("git")
CHECK = R.DIFFERENTIAL / "check_path2a.js"
PHRASES = N._P2A_PHRASES
DECIDED = ("VERIFIED", "CONTRADICTED")


def no_node():
    """A missing `node` skips the port half locally, and fails it under CI, where a skip would read green
    (NOTE_path2a_third_pass_2026_09_30, I-7)."""
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        pytest.fail("node is not on PATH under CI; the port half of the PATH-2a checks did not run")
    pytest.skip("node is not on PATH; the port half of this check cannot run here")


def node(*args, timeout=900):
    if NODE is None:
        no_node()
    r = subprocess.run([NODE, str(CHECK), *map(str, args)], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout)
    assert r.returncode == 0, f"check_path2a.js {args[0]} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"


@pytest.fixture(scope="module")
def M():
    return R.main_module()


@pytest.fixture(scope="module")
def inputs():
    return R.inputs()


@pytest.fixture(scope="module")
def work(tmp_path_factory, inputs):
    """The inputs as the port reads them, and the reconstructed port, in one temporary directory."""
    d = tmp_path_factory.mktemp("p2a")
    (d / "in.json").write_text(json.dumps([{"id": R.uid(i, row), "summary": row[2], "diff": row[3]}
                                           for i, row in enumerate(inputs)], ensure_ascii=False), encoding="utf-8")
    R.main_port_path(d)
    return d


# ---- the reconstruction ----------------------------------------------------------------------------------------------

def test_reconstruction_python():
    text = R.lf(R.INSTRUMENT)
    assert text.count(R.PY_BEGIN) == 1 and text.count(R.PY_END) == 1
    assert text.count("    g = _gate(summary_text,") == 2
    assert sum(1 for line in text.split("\n") if line.endswith(R.PY_HOOK_SUFFIX)) == 2
    assert R.sha(R.reconstruct_py(text)) == R.MAIN_PY_SHA
    assert all(ord(ch) < 128 for ch in R.py_block(text)), "the Python block must be ASCII"


def test_reconstruction_port():
    text = R.lf(R.PORT)
    assert text.count(R.JS_BEGIN) == 1 and text.count(R.JS_END) == 1
    assert text.replace(R.js_block(text), "").count("_gateDiffTextMain(") == 2
    assert R.sha(R.reconstruct_js(text)) == R.MAIN_JS_SHA
    block = R.js_block(text)
    assert all(ord(ch) < 128 and (ord(ch) >= 32 or ch == "\n") for ch in block), "the port block must be printable ASCII"
    assert R.sha(R.lf(R.INSTRUMENT)) in block, "the port block names the Python file it mirrors"


def test_the_hooks_are_the_only_edits_at_the_doors():
    text = R.lf(R.INSTRUMENT)
    hooks = [line for line in text.split("\n") if line.endswith(R.PY_HOOK_SUFFIX)]
    assert hooks == ['    return _p2a_abstain(g, strict, lambda: _P2aFacts(diff_text or "", None, summary_text))  # PATH-2a',
                     "    return _p2a_abstain(g, strict, lambda: _P2aFacts(diff_text, name_status, summary_text))  # PATH-2a"]


def test_main_reconstructed_matches_every_pinned_expect_of_main(M):
    """The pinned files of main are main's records: the reconstruction reads them exactly, moves not applied."""
    n = 0
    for name in R.pinned_files():
        if name == "path2a_pairs.json":
            continue
        for p in json.loads((R.DIFFERENTIAL / name).read_text(encoding="utf-8")):
            g = M.gate_diff_text(p["summary"], p["diff"])
            width = len(p["expect"]["claims"][0]) if p["expect"]["claims"] else 3
            assert [[c.kind, c.verdict, c.why][:width] for c in g.claims] == p["expect"]["claims"], p["id"]
            assert (g.verdict, g.uncovered_sentences) == (p["expect"]["verdict"], p["expect"]["uncovered_sentences"])
            n += 1
    assert n == 54


# ---- (A) the abstain-only relation ------------------------------------------------------------------------------------

FLAVOURS = {"windows": pathlib.PureWindowsPath, "posix": pathlib.PurePosixPath}


def flavour(mod) -> str:
    """The path flavour `main`'s find_path reads base names with: `Path(p).name` of a drive-like name differs."""
    return "windows" if mod.Path("c:x.py").name == "x.py" else "posix"


# Abstentions by (kind, phrase key) over every committed input, strict off, under each path flavour (main decides
# three drive-like claims only under the Windows one). Pinned after review: every figure here was read against the
# NOTEs' rules before it was written down. A change to the overlay or to the inputs moves it. Pass 3
# (NOTE_path2a_third_pass_2026_09_30) moved `symbol_added:extract` by +5 on the inputs pass 2 pinned (#161's x1
# probes, whose name also occurs beside a letter outside ASCII); the text-seam set and the new pairs add the rest.
# Pass 4 (NOTE_path2a_fourth_pass_2026_09_30), on the inputs pass 3 pinned: `only_touches:shape` 25 (24 of them kept
# before, 1 `only`), and the wider neutral set keeps four claims `extract` withheld (one of them now `dir`); its 29 new
# pairs add the rest.
ABSTENTIONS = {
    "windows": {
        "decided": 12675, "main raises": 8,
        "file_created:case": 3, "file_created:dir": 65, "file_created:divergent": 19, "file_created:dot": 20,
        "file_created:dot_earliest": 4, "file_created:dot_tier": 11, "file_created:extract": 48, "file_created:odd": 1,
        "file_created:tier": 9,
        "file_deleted:case": 1, "file_deleted:dir": 52, "file_deleted:divergent": 20, "file_deleted:dot": 21,
        "file_deleted:dot_earliest": 1, "file_deleted:dot_tier": 7, "file_deleted:extract": 55, "file_deleted:odd": 1,
        "file_deleted:tier": 4,
        "file_touched:dir": 243, "file_touched:divergent": 102, "file_touched:dot": 82, "file_touched:dot_tier": 71,
        "file_touched:extract": 144, "file_touched:odd": 5,
        "files_changed_count:count": 382, "files_changed_count:divergent": 226, "files_changed_count:extract": 4,
        "only_touches:divergent": 108, "only_touches:extract": 158, "only_touches:only": 32, "only_touches:shape": 29,
        "symbol_added:extract": 24, "symbol_added:symbol": 173, "tests_added:split": 4, "tests_added:tests": 403,
    },
}
# Under the POSIX flavour main reads `c:x.py` as a bare name not in the diff, so three decided drive-like claims are
# UNCHECKABLE on main to begin with: path2a:guard-drive-like-path claim 0, and fuzz 20260930:1033 and :1590.
ABSTENTIONS["posix"] = {**ABSTENTIONS["windows"], "decided": 12672, "file_touched:odd": 4}
del ABSTENTIONS["posix"]["file_created:odd"], ABSTENTIONS["posix"]["file_deleted:odd"]
# main's own reading depends on the interpreter's Unicode tables where the inputs probe letters added in Unicode 14 and
# 16 (#161's k2, x1 and y3 cases). Measured on CPython 3.12 (Unicode 15.0) and 3.14 (16.0); for Unicode 13.0 and 14.0
# (CI's 3.9 to 3.11) by re-reading the inputs with the five Unicode 14 letters present mapped to code points no
# version assigns (NOTE_path2a_second_pass_2026_09_30, I-10): no figure moved. On 16.0 main reads the y3 name
# `fo` + U+105C0 whole and CONTRADICTS it, so it is not a VERIFIED symbol the overlay withholds. Unicode 15.1 (CPython
# 3.13) was never measured, so it fails here and names itself (NOTE_path2a_third_pass_2026_09_30, I-3).
UNICODE_SAME = ("13.0.0", "14.0.0", "15.0.0")


def pinned_abstentions(fl: str) -> dict:
    v = unicodedata.unidata_version
    pin = dict(ABSTENTIONS[fl])
    if v == "16.0.0":
        pin["symbol_added:extract"] -= 1
    else:
        assert v in UNICODE_SAME, f"Unicode {v}: measure the PATH-2a abstention figures on it and pin them here"
    return pin


def _abstentions(M, inputs):
    counts = collections.Counter()
    broken = []
    for sname, iid, summary, diff in inputs:
        mine = {}
        for strict in (False, True):
            try:
                a = M.gate_diff_text(summary, diff, strict=strict).to_dict()
            except Exception as e:
                with pytest.raises(type(e)):
                    N.gate_diff_text(summary, diff, strict=strict)
                counts["main raises"] += 1
                continue
            b = N.gate_diff_text(summary, diff, strict=strict).to_dict()
            bad = R.relation(a, b, strict, PHRASES)
            if bad:
                broken.append((sname, iid, strict, bad))
            mine[strict] = (a, b)
        if len(mine) == 2:
            bad = R.strict_alike(mine[False][1], mine[True][1])
            if bad:
                broken.append((sname, iid, "strict", bad))
        if False in mine:
            a, b = mine[False]
            for x, y in zip(a["claims"], b["claims"]):
                if x["verdict"] in DECIDED:
                    counts["decided"] += 1
                    if y["verdict"] == "UNCHECKABLE":
                        counts[x["kind"] + ":" + R.phrase_key(y["why"], PHRASES)] += 1
    return dict(counts), broken


def test_abstain_only_python(M, inputs):
    counts, broken = _abstentions(M, inputs)
    print(f"PATH-2a abstentions over the committed inputs ({flavour(M)}):",
          json.dumps(dict(sorted(counts.items())), indent=1))
    assert not broken, broken[:10]
    assert counts == pinned_abstentions(flavour(M))


def test_abstain_only_python_under_the_other_path_flavour(M, inputs, monkeypatch):
    """The same count with main's and the branch's `Path` read as the other pure flavour, so both pins run on every
    runner (pass 1 pinned the Windows reading only, and CI's ubuntu runners read the other)."""
    other = "posix" if flavour(M) == "windows" else "windows"
    monkeypatch.setattr(M, "Path", FLAVOURS[other])
    monkeypatch.setattr(N, "Path", FLAVOURS[other])
    assert flavour(M) == other
    counts, broken = _abstentions(M, inputs)
    print(f"PATH-2a abstentions over the committed inputs ({other}):", json.dumps(dict(sorted(counts.items())), indent=1))
    assert not broken, broken[:10]
    assert counts == pinned_abstentions(other)


def test_abstain_only_port(work):
    node("--relation", work / "diffgate_main_reference.js", work / "in.json", work / "rel.json")
    rep = json.loads((work / "rel.json").read_text(encoding="utf-8"))
    print("port relation:", rep["counts"])
    assert rep["broken"] == [] and rep["counts"]["broken"] == 0 and rep["counts"]["raise_differs"] == 0
    assert rep["counts"]["runs"] > 10000


def test_lockstep_python(M, inputs):
    """The overlay's status-map builder, keyed by main's _norm, is main's parse_unified_diff map, order included; and
    its count of `def test_` sites in line view 0 is main's count of `^\\s*def test_` over main's added lines."""
    checked = 0
    for _s, iid, _summary, diff in inputs:
        try:
            status, blob = M.parse_unified_diff(diff)
        except Exception:
            continue
        fine = N._p2a_lines(diff or "", N._P2A_FINE)
        assert list(N._p2a_build(N._p2a_regs_raw(fine), N._norm).items()) == list(status.items()), iid
        views = N._p2a_views(diff or "", fine)
        assert N._p2a_pairing(views)[0][0] == len(re.findall(r"^\s*def test_", blob, re.M)), iid
        # pass 4 (A-2): the facts object's shortcut (one view, counted once) gives what both views counted apart give
        assert N._P2aFacts(diff or "").pairing() == N._p2a_pairing(views), iid
        checked += 1
    assert checked > 5000


def test_lockstep_port(work, inputs):
    """The same in the port, and the two ports' overlays count the same sites in each line view."""
    node("--lockstep", work / "in.json", work / "lock.json")
    rep = json.loads((work / "lock.json").read_text(encoding="utf-8"))
    assert rep["differ"] == [] and rep["checked"] > 5000
    bad = []
    for i, row in enumerate(inputs):
        got = rep["counts"].get(R.uid(i, row))
        if got is None:
            continue
        mine = [g for g, _p in N._p2a_pairing(N._p2a_views(row[3] or "", N._p2a_lines(row[3] or "", N._P2A_FINE)))]
        if got["views"][1] != got["main"] or got["views"] != mine:
            bad.append((row[1], got, mine))
    assert bad == [], bad[:5]


# ---- the git door -----------------------------------------------------------------------------------------------------

def _repo(tmp_path, name, base, head, quotepath=None):
    d = tmp_path / name
    d.mkdir()

    def git(*a):
        return subprocess.run([GIT, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false", *a],
                              cwd=d, capture_output=True, check=True).stdout

    git("init", "-q")
    if quotepath is not None:
        git("config", "core.quotepath", quotepath)
    for side in (base, head):
        for p in [p for p in d.rglob("*") if p.is_file() and ".git" not in p.relative_to(d).parts]:
            if p.relative_to(d).as_posix() not in side:
                p.unlink()
        for p, t in side.items():
            (d / p).parent.mkdir(parents=True, exist_ok=True)
            (d / p).write_bytes(t.encode("utf-8"))
        git("add", "-A")
        git("commit", "-q", "--allow-empty", "-m", "c")
    return d


def _tree_repo(tmp_path, name, base, head, config=()):
    """Two commits built with plumbing, so a symlink can stand in a tree on any OS: an entry is a text, or
    ("link", target)."""
    d = tmp_path / name
    d.mkdir()

    def git(*a, data=None):
        return subprocess.run([GIT, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false", *a],
                              cwd=d, input=data, capture_output=True, check=True).stdout.decode().strip()

    git("init", "-q")
    for k, v in config:
        git("config", k, v)
    parent = None
    for side in (base, head):
        git("read-tree", "--empty")
        for p, entry in side.items():
            mode, text = ("120000", entry[1]) if isinstance(entry, tuple) else ("100644", entry)
            blob = git("hash-object", "-w", "--stdin", data=text.encode("utf-8"))
            git("update-index", "--add", "--cacheinfo", f"{mode},{blob},{p}")
        tree = git("write-tree")
        parent = git("commit-tree", tree, "-m", "c", *(["-p", parent] if parent else []))
    git("update-ref", "HEAD", parent)
    return d


R101_BASE = {"src/retry.py": "def backoff(n):\n    return n * 2\n",
             "tests/test_retry.py": "def test_a():\n    assert True\n\n\ndef test_b():\n    assert True\n"}
R101_HEAD = {"src/retry.py": "def backoff(n, jitter=0):\n    return n * 2\n",
             "tests/test_retry.py": "def test_a():  # comment added\n    assert True\n\n\n"
                                    "def test_b():  # comment added\n    assert True\n"}
GIT_CASES = [
    ("r101", "Adds function backoff with jitter. Added 2 tests.", R101_BASE, R101_HEAD, None),
    ("twins", "3 files changed. Created .pr_agent.toml. Deleted pr_agent.toml. Only touches src/.",
     {"pr_agent.toml": "x = 1\n", ".github/workflows/ci.yml": "o\n"},
     {".pr_agent.toml": "y = 2\n", ".github/workflows/ci.yml": "x\n"}, None),
    ("readmes", "Created integrations/git/README.md.", {"README.md": "o\n"},
     {"README.md": "x\n", "integrations/git/README.md": "x\n"}, None),
    ("readmes-reversed", "Modified readme.md.", {"readme.md": "o\n"},
     {"readme.md": "x\n", "integrations/git/README.md": "x\n"}, None),
    ("directory-claim", "Created integrations/git/README.md.", {}, {"README.md": "x\n"}, None),
    ("count-twin", "2 files changed.", {"env.json": "1\n"}, {"env.json": "2\n", ".env.json": "3\n"}, None),
    ("non-ascii", "Modified café/x.py. 2 files changed.", {"café/x.py": "1\n", "docs/é.py": "1\n"},
     {"café/x.py": "2\n", "docs/é.py": "2\n"}, "false"),
    # B-1 (NOTE_path2a_fourth_pass_2026_09_30): the fourth review's reproduction 3, a rename into a dotted directory,
    # read at the git door (`--name-status` lists the new path only) and on the same bytes at the raw door
    ("scope-shape-rename", "Only touches github and .github/workflows/.",
     {".github/workflows/ci.yml": "a\n", "docs/x.md": "one\ntwo\nthree\nfour\nfive\n"},
     {".github/workflows/ci.yml": "b\n", ".github/workflows/x.md": "one\ntwo\nthree\nfour\nfive\n"}, None),
]
# Renames, a copy and a type change: `--name-status` then carries R, C and T entries, whose last field is the new path.
SAME = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
GIT_TREE_CASES = [
    ("rename", "Created src/new.py. Deleted src/old.py. Modified cfg/.env.json. 2 files changed.",
     {"src/old.py": SAME, "cfg/env.json": '{"k": 1}\n'}, {"src/new.py": SAME, "cfg/.env.json": '{"k": 1}\n'}, ()),
    ("copy", "Created src/b.py. Modified src/a.py. 2 files changed.",
     {"src/a.py": SAME}, {"src/a.py": SAME + "\n\ndef c():\n    return 3\n", "src/b.py": SAME},
     (("diff.renames", "copies"),)),
    ("typechange", "Modified tools/run.py. 1 file changed.", {"tools/run.py": "print(1)\n"},
     {"tools/run.py": ("link", "../bin/run.py")}, ()),
    # Pass 3 (A-3): where the two doors part and each still relates to its own main. With diff.noprefix the deleted
    # empty c/x.py has a `diff --git c/x.py c/x.py` header main's raw reader cannot parse, so it is in --name-status
    # only; and a path holding U+2028 under core.quotepath=false is a header the raw door's divergent guard reads.
    ("noprefix", "Modified c/x.py.", {"c/x.py": ""}, {"a/x.py": "y = 1\n"}, (("diff.noprefix", "true"),)),
    ("line-separator", "Modified src/x.py. 2 files changed.",
     {"src/x.py": "1\n", "docs/a" + chr(0x2028) + "b.md": "1\n"},
     {"src/x.py": "2\n", "docs/a" + chr(0x2028) + "b.md": "2\n"}, (("core.quotepath", "false"),)),
]


def _git_repos(tmp_path):
    for name, summary, base, head, quote in GIT_CASES:
        yield name, summary, _repo(tmp_path, name, base, head, quote)
    for name, summary, base, head, config in GIT_TREE_CASES:
        yield name, summary, _tree_repo(tmp_path, name, base, head, config)


@pytest.mark.skipif(GIT is None, reason="git is not on PATH; the git door cannot be exercised here")
def test_git_door(M, tmp_path):
    seen = collections.Counter()
    compared = []                             # the repositories where the door agreement is asserted
    for name, summary, repo in _git_repos(tmp_path):
        mine = {}
        for strict in (False, True):
            a = M.gate_diff(summary, repo, "HEAD~1", "HEAD", strict=strict).to_dict()
            b = N.gate_diff(summary, repo, "HEAD~1", "HEAD", strict=strict).to_dict()
            assert R.relation(a, b, strict, PHRASES) == [], name
            mine[strict] = b
        assert R.strict_alike(mine[False], mine[True]) == [], name
        # the lockstep at the git door: the overlay's builder over --name-status is main's four-line loop
        name_status = N._git(repo, "diff", "--name-status", "HEAD~1..HEAD")
        seen.update(line[:1] for line in name_status.splitlines())
        want = {}
        for line in name_status.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                want[M._norm(parts[-1])] = parts[0][:1]
        assert list(N._p2a_build(N._p2a_regs_git(name_status), N._norm).items()) == list(want.items()), name
        # G-P3's analogue, where it holds (NOTE_path2a_third_pass_2026_09_30, A-3): where main's two doors agree on
        # the claims, main's two status maps are equal, and the raw door's divergent guard is off, the overlay decides
        # alike at both doors. Elsewhere each door relates to its own main (above), and the two may part.
        text = N._git(repo, "diff", "HEAD~1..HEAD")
        rows = lambda g: [(c.kind, c.verdict, c.why) for c in g.claims]  # noqa: E731
        same_main = rows(M.gate_diff(summary, repo, "HEAD~1", "HEAD")) == rows(M.gate_diff_text(summary, text))
        same_maps = list(want.items()) == list(M.parse_unified_diff(text)[0].items())
        if same_main and same_maps and not N._p2a_divergent(text):
            assert rows(N.gate_diff(summary, repo, "HEAD~1", "HEAD")) == rows(N.gate_diff_text(summary, text)), name
            compared.append(name)
    assert {"R", "C", "T"} <= set(seen), seen
    assert compared == ["r101", "twins", "readmes", "readmes-reversed", "directory-claim", "count-twin", "non-ascii"], \
        compared


@pytest.mark.skipif(GIT is None, reason="git is not on PATH; the git door cannot be exercised here")
def test_git_door_reproductions(tmp_path):
    repos = {name: (summary, repo) for name, summary, repo in _git_repos(tmp_path)}

    def rows(name):
        summary, repo = repos[name]
        g = N.gate_diff(summary, repo, "HEAD~1", "HEAD")
        return g.verdict, [(c.kind, c.verdict, R.phrase_key(c.why, PHRASES)) for c in g.claims]

    assert rows("r101") == ("PASS", [("symbol_added", "UNCHECKABLE", "symbol"), ("tests_added", "UNCHECKABLE", "tests")])
    assert rows("directory-claim") == ("PASS", [("file_created", "UNCHECKABLE", "dir")])
    assert rows("readmes-reversed") == ("PASS", [("file_touched", "VERIFIED", None)])
    assert rows("twins") == ("FAIL", [("files_changed_count", "UNCHECKABLE", "count"), ("file_created", "UNCHECKABLE", None),
                                      ("file_deleted", "VERIFIED", None), ("only_touches", "CONTRADICTED", None)])
    assert rows("count-twin") == ("PASS", [("files_changed_count", "UNCHECKABLE", "count")])
    assert rows("rename") == ("PASS", [("file_created", "UNCHECKABLE", None), ("file_deleted", "UNCHECKABLE", None),
                                       ("file_touched", "VERIFIED", None), ("files_changed_count", "VERIFIED", None)])
    assert rows("copy") == ("PASS", [("file_created", "UNCHECKABLE", None), ("file_touched", "VERIFIED", None),
                                     ("files_changed_count", "VERIFIED", None)])
    assert rows("typechange") == ("PASS", [("file_touched", "VERIFIED", None), ("files_changed_count", "VERIFIED", None)])
    # B-1: main's VERIFIED is false (docs/x.md changed) and V121 does not read `github` as a path; ea677740 kept it
    assert rows("scope-shape-rename") == ("PASS", [("only_touches", "UNCHECKABLE", "shape")])

    def raw(name):
        summary, repo = repos[name]
        g = N.gate_diff_text(summary, N._git(repo, "diff", "HEAD~1..HEAD"))
        return g.verdict, [(c.kind, c.verdict, R.phrase_key(c.why, PHRASES)) for c in g.claims]

    # A-3: the doors part here, each against its own main. main verifies "Modified c/x.py." at both doors by base name
    # (a/x.py); the git door also lists the deleted c/x.py, so V97 finds the claim itself there and keeps the verdict,
    # while the raw door never registered c/x.py and withholds it (dir).
    assert rows("noprefix") == ("PASS", [("file_touched", "VERIFIED", None)])
    assert raw("noprefix") == ("PASS", [("file_touched", "UNCHECKABLE", "dir")])
    assert rows("line-separator") == ("PASS", [("file_touched", "VERIFIED", None),
                                               ("files_changed_count", "VERIFIED", None)])
    assert raw("line-separator") == ("PASS", [("file_touched", "UNCHECKABLE", "divergent"),
                                              ("files_changed_count", "UNCHECKABLE", "divergent")])
    assert raw("scope-shape-rename") == ("PASS", [("only_touches", "UNCHECKABLE", "shape")])


@pytest.mark.skipif(GIT is None, reason="git is not on PATH; the git door cannot be exercised here")
def test_a_planted_git_door_mirror_is_refused(M, tmp_path):
    """Integration-2: the reviewer's plant, the `--name-status` mirror reading the old side of a rename (`parts[1]`),
    passed every check of pass 1. The lockstep over a real rename and a real copy refuses it."""
    text = R.lf(R.INSTRUMENT)
    block = R.py_block(text)
    old = '            regs.append((parts[-1], parts[0][:1], "+"))\n'
    assert block.count(old) == 1
    mod = R.module_from(text.replace(block, block.replace(old, old.replace("parts[-1]", "parts[1]"))), "_p2a_plant_git")
    caught = []
    for name, _summary, repo in _git_repos(tmp_path):
        if name not in ("rename", "copy"):
            continue
        name_status = N._git(repo, "diff", "--name-status", "HEAD~1..HEAD")
        want = {}
        for line in name_status.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                want[M._norm(parts[-1])] = parts[0][:1]
        if list(mod._p2a_build(mod._p2a_regs_git(name_status), mod._norm).items()) != list(want.items()):
            caught.append(name)
    assert caught == ["rename", "copy"], caught


# ---- the self-checks --------------------------------------------------------------------------------------------------

def test_selfcheck_p2a_only_abstains():
    rep = N.selfcheck_p2a_only_abstains()
    assert rep["ok"] is True, rep["problems"]


CLAIMED = "    ca, ck = _p2a_A(claimed), _p2a_K(claimed)\n"
SELFCHECK_PLANTS = [
    ('            c.verdict = "UNCHECKABLE"\n', '            c.verdict = "VERIFIED"\n', "verdict literal"),
    ('            c.verdict = "UNCHECKABLE"\n', '            c.verdict = "UNCHECKABLE"\n            c.detail["x"] = 1\n',
     "item store into .detail"),
    ('    todo = [c for c in g.claims', '    g.claims.append(None)\n    todo = [c for c in g.claims', ".claims.append"),
    ('    claimed = c.detail["path"]\n', '    claimed = c.detail["path"]\n    c.text = claimed\n', "outside _p2a_abstain"),
    ("def _p2a_fold(s: str) -> str:", "def _p2a_fold(s: str) -> str:\n    s = s.lower()", "attribute .lower"),
    ('_P2A_COARSE = re.compile("\\r\\n|\\r|\\n")', '_P2A_COARSE = re.compile("\\\\s+")', "class escape"),
    ("    return s.replace(\"\\\\\", \"/\")", "    return s.replace(\"\\\\\", \"/\").strip()", "without an explicit argument"),
    ("    d = c.detail\n", "    d = c.detail\n    d[\"prefix\"] = \"\"\n", "record field"),
    ("    return s.translate(_P2A_FOLD)\n", "    return s.translate(_P2A_FOLD).casefold()\n", "attribute .casefold"),
    # C-4: calls that read a Unicode table indirectly
    (CLAIMED, "    claimed = repr(claimed)\n" + CLAIMED, "name repr"),
    (CLAIMED, '    claimed = f"{claimed!r}"\n' + CLAIMED, "!r or !a conversion"),
    (CLAIMED, '    claimed = f"{claimed!a}"\n' + CLAIMED, "!r or !a conversion"),
    (CLAIMED, '    claimed = "%r" % (claimed,)\n' + CLAIMED, "%r or %a formatting"),
    (CLAIMED, '    claimed = getattr(claimed, "lo" + "wer")()\n' + CLAIMED, "name getattr"),
    (CLAIMED, '    claimed = claimed.encode("idna").decode("ascii")\n' + CLAIMED, "attribute .encode"),
    (CLAIMED, '    claimed = "{!r}".format(claimed)\n' + CLAIMED, "attribute .format"),
    (CLAIMED, '    claimed = eval("claimed.lower()")\n' + CLAIMED, "name eval"),
    ("    return _p2a_int(m.group(1)), _p2a_int(k.group(0))\n", '    return int(m.group(1)), int(detail["n"])\n',
     "name int"),
    # C-3 (NOTE_path2a_fourth_pass_2026_09_30): int() reads CPython's table of decimal digits, so it is refused
    # whatever its argument; the block reads digits from a fixed table instead
    ("    return _p2a_int(m.group(1)), _p2a_int(k.group(0))\n", "    return int(m.group(1)), int(k.group(0))\n",
     "name int"),
    ("    return _p2a_int(m.group(1)), _p2a_int(k.group(0))\n",
     "    return _p2a_int(m.group(1)), (lambda x: x)(int)(k.group(0))\n", "name int"),
    # C-3 (NOTE_path2a_third_pass_2026_09_30): the nine plants the pass-2 self-check accepted, and a few more
    (CLAIMED, "    import unicodedata as ud\n    claimed = ''.join(ch for ch in claimed if ud.category(ch) != 'Cf')\n"
     + CLAIMED, "an import"),
    (CLAIMED, "    import re as r2\n    claimed = r2.sub(chr(92) + 'W', '', claimed)\n" + CLAIMED, "an import"),
    (CLAIMED, "    from re import sub as rs\n    claimed = rs('(?i)x', '', claimed)\n" + CLAIMED, "an import"),
    (CLAIMED, "    claimed = claimed.split(None)[0]\n" + CLAIMED, ".split() without an explicit argument"),
    (CLAIMED, "    claimed = claimed.strip(None)\n" + CLAIMED, ".strip() without an explicit argument"),
    (CLAIMED, "    claimed = claimed.split(sep=None)[0]\n" + CLAIMED, ".split() without an explicit argument"),
    (CLAIMED, "    import builtins\n    claimed = builtins.repr(claimed)[1:-1]\n" + CLAIMED, "an import"),
    (CLAIMED, "    claimed = builtins.repr(claimed)[1:-1]\n" + CLAIMED, "name builtins"),
    (CLAIMED, "    claimed = claimed.__getattribute__('lo' + 'wer')()\n" + CLAIMED,
     "attribute .__getattribute__, not one the block may read"),
    (CLAIMED, "    claimed = operator.methodcaller('casefold')(claimed)\n" + CLAIMED, "name operator"),
    (CLAIMED, "    claimed = claimed[:0] + claimed[0:].join(map(str.__str__, claimed))\n" + CLAIMED,
     "attribute read off the type str"),
    (CLAIMED, "    claimed = str.split(claimed)[0]\n" + CLAIMED, "attribute read off the type str"),
    (CLAIMED, "    rc = re.compile\n    claimed = rc('x').sub('', claimed)\n" + CLAIMED,
     "re used other than as re.<function>(...)"),
    (CLAIMED, "    claimed = claimed.expandtabs(4)\n" + CLAIMED, "attribute .expandtabs, not one the block may read"),
    (CLAIMED, "    claimed = str([claimed])\n" + CLAIMED, "str() of a value"),
    (CLAIMED, "    claimed = unicodedata.normalize('NFKC', claimed)\n" + CLAIMED, "name unicodedata"),
    (CLAIMED, "    claimed = json.dumps(claimed)\n" + CLAIMED, "name json, not one the block binds or may read"),
]


@pytest.mark.parametrize("old,new,what", SELFCHECK_PLANTS)
def test_the_selfcheck_refuses_a_planted_write(old, new, what):
    text = R.lf(R.INSTRUMENT)
    block = R.py_block(text)
    assert block.count(old) == 1, old
    rep = N.selfcheck_p2a_only_abstains(text.replace(block, block.replace(old, new)))
    assert rep["ok"] is False and any(what in p for p in rep["problems"]), rep["problems"]


JS_BANNED = ("toLowerCase", "toUpperCase", "toLocale", "localeCompare", "normalize", ".trim", "trimStart", "trimEnd",
             "\\p{", "Intl", ".sort(", "String.raw", "eval(", "Function(", "prototype", ".call(", ".apply(",
             ".bind(", "Reflect", "globalThis", "require(", "import(", ".compile(", "__proto__", "constructor",
             # C-3 (NOTE_path2a_fourth_pass_2026_09_30): a string method that builds a RegExp from its argument at run
             # time, and a String object carrying a method off its prototype; the block uses .test and .exec on
             # constant RegExps only
             "new String(", ".match(", ".search(", ".matchAll(")
# Identifiers a computed member access may index with: counters, positions and the block's own constant keys. A name
# built from strings (claimed[kk]) is refused, and so is any call on a computed member (x[k](), (x[k])()).
JS_INDEXES = {"0", "1", "2", "i", "k", "k+1", "k-1", "v", "u", "space", "c.kind", "_P2A_OWN", "out.length-1"}
JS_KEYWORDS = {"return", "of", "in", "const", "let", "var", "case", "typeof", "void", "delete", "throw", "yield",
               "await", "else", "do", "new"}
JS_ESC = {"n": "\n", "r": "\r", "t": "\t", "v": "\v", "f": "\f", "b": "\b", "0": "\0"}


def js_string(body: str) -> str:
    """The value of a JavaScript string literal's body (the escapes the block writes)."""
    out, i = [], 0
    while i < len(body):
        ch = body[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        nx = body[i + 1]
        if nx == "u" and body[i + 2] == "{":
            j = body.index("}", i)
            out.append(chr(int(body[i + 3:j], 16)))
            i = j + 1
        elif nx == "u":
            out.append(chr(int(body[i + 2:i + 6], 16)))
            i += 6
        elif nx == "x":
            out.append(chr(int(body[i + 2:i + 4], 16)))
            i += 4
        else:
            out.append(JS_ESC.get(nx, nx))
            i += 2
    return "".join(out)


def js_regex_problems(pattern: str) -> list:
    """Class escapes, property escapes and an unescaped '.' in one static pattern, as the Python block's
    _p2a_regex_problems reads them."""
    out = []
    i, in_class = 0, False
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\":
            if pattern[i + 1:i + 2] in ("w", "W", "s", "S", "b", "B", "d", "D", "p", "P"):
                out.append("class escape " + pattern[i:i + 2])
            i += 2
            continue
        if ch == "[" and not in_class:
            in_class = True
        elif ch == "]" and in_class:
            in_class = False
        elif ch == "." and not in_class:
            out.append("unescaped '.'")
        i += 1
    return out


def js_problems(block: str) -> list:
    """The port block's token scan (NOTE_path2a_third_pass_2026_09_30, C-3). Code is read outside strings and comments
    with white space removed except between two identifier characters, so `a . b (` reads `a.b(`: banned calls
    anywhere; no regex literal (no '/' in code); `RegExp` only as `new RegExp(<one static string>)` -- string literals
    and top-level constants joined by `+` -- whose decoded value holds no class escape, no '.' outside a class, and
    no flags; a computed member access only with a listed index expression and no string in the brackets, and never
    called; no call on a parenthesised expression."""
    out = [f"banned token {t!r}" for t in JS_BANNED if t in block]
    code, strings, i = [], [], 0
    while i < len(block):
        ch = block[i]
        if block.startswith("//", i):
            i = block.index("\n", i)
            continue
        if block.startswith("/*", i):
            i = block.index("*/", i) + 2
            continue
        if ch in "\"'`":
            j = i + 1
            while block[j] != ch:
                j += 2 if block[j] == "\\" else 1
            strings.append(block[i + 1:j])
            code.append("S%d" % (len(strings) - 1) if ch != "`" else "T%d" % (len(strings) - 1))
            i = j + 1
            continue
        code.append(ch)
        i += 1
    raw = "".join(code)
    word = re.compile(r"[A-Za-z0-9_$]")
    dense = re.sub(r"\s+", lambda m: " " if (0 < m.start() and m.end() < len(raw) and word.match(raw[m.start() - 1])
                                                and word.match(raw[m.end()])) else "", raw)
    out += [f"banned token {t!r} in code" for t in JS_BANNED if t in dense]
    if "/" in dense:
        out.append("a '/' in code: a regex literal or a division")
    for s in strings:
        if re.search(r"(?<!\\)(?:\\\\)*\\\\[wWsSbBdDpP]", s):
            out.append(f"class escape in a string: {s[:40]!r}")
    # static strings: `const NAME = <strings and static names joined by +>;` at the block's top level
    static = {}
    for m in re.finditer(r"const ([A-Za-z_$][\w$]*)=((?:S\d+|[A-Za-z_$][\w$]*)(?:\+(?:S\d+|[A-Za-z_$][\w$]*))*);", dense):
        parts = m.group(2).split("+")
        vals = [js_string(strings[int(p[1:])]) if re.fullmatch(r"S\d+", p) else static.get(p) for p in parts]
        if all(v is not None for v in vals):
            static[m.group(1)] = "".join(vals)
    for m in re.finditer(r"RegExp", dense):
        if dense[max(0, m.start() - 4):m.start()] != "new " or dense[m.end():m.end() + 1] != "(":
            out.append("RegExp other than as new RegExp(...)")
            continue
        depth, j = 1, m.end() + 1
        while depth:
            depth += {"(": 1, ")": -1}.get(dense[j], 0)
            j += 1
        arg = dense[m.end() + 1:j - 1]
        parts = arg.split("+")
        vals = [js_string(strings[int(p[1:])]) if re.fullmatch(r"S\d+", p) else static.get(p) for p in parts]
        if "," in arg or any(v is None for v in vals):
            out.append(f"new RegExp with an argument that is not one static string: {arg[:40]}")
            continue
        out += [f"new RegExp({arg[:30]}): {p}" for p in js_regex_problems("".join(vals))]
    for m in re.finditer(r"([A-Za-z0-9_$]+|[)\]])(\?\.)?\[", dense):
        if m.group(1) in JS_KEYWORDS:           # an array literal or a destructuring pattern, not a member access
            continue
        depth, j = 1, m.end()
        while depth:
            depth += {"[": 1, "]": -1}.get(dense[j], 0)
            j += 1
        inside = dense[m.end():j - 1]
        if re.search(r"[ST]\d+", inside):
            out.append("computed member access with a string in the brackets")
        elif inside not in JS_INDEXES:
            out.append(f"computed member access indexed by {inside[:30]!r}")
        if dense[j:j + 1] == "(" or dense[j:j + 3] == "?.(" or dense[j:j + 2] == ")(":
            out.append("a call on a computed member")
    if re.search(r"\)\(|\]\)\(", dense):
        out.append("a call on a parenthesised expression")
    return out


def test_the_port_block_passes_its_token_scan():
    assert js_problems(R.js_block()) == []


@pytest.mark.parametrize("old,new", [
    ("const o = ch.codePointAt(0);\n    out +=", "const o = ch.toLowerCase().codePointAt(0);\n    out +="),
    ('new RegExp("\\r\\n|\\r|\\n")', 'new RegExp("\\r\\n|\\r|\\n", "u")'),
    ('new RegExp("\\r\\n|\\r|\\n")', 'new RegExp("\\\\s")'),
    ("const _p2aBs = s => s.split(", "const _p2aBs = s => s.replace(/x/, \"\").split("),
    ("const out = text.split(rx);", "const out = text.split(rx).sort();"),
    # C-4: a table read through a computed name
    ("  const ca = _p2aA(claimed), ck = _p2aK(claimed);",
     "  const ca = _p2aA(claimed[\"toLower\" + \"Case\"]()), ck = _p2aK(claimed);"),
    ("  const ca = _p2aA(claimed), ck = _p2aK(claimed);",
     "  const ca = _p2aA(String.prototype[\"norm\" + \"alize\"].call(claimed)), ck = _p2aK(claimed);"),
    ("  const ca = _p2aA(claimed), ck = _p2aK(claimed);",
     "  const ca = _p2aA(claimed[`to${\"Lower\"}Case`]()), ck = _p2aK(claimed);"),
])
def test_the_token_scan_refuses_a_planted_call(old, new):
    block = R.js_block()
    assert block.count(old) == 1, old
    assert js_problems(block.replace(old, new)) != []


CA = "  const ca = _p2aA(claimed), ck = _p2aK(claimed);"
BS = chr(92)


@pytest.mark.parametrize("new,what", [
    # C-3 (NOTE_path2a_third_pass_2026_09_30): the five plants the pass-2 scan accepted, and a few more
    ('  const ca = _p2aA(claimed.replace(RegExp("' + BS * 2 + '" + "s", "g"), "")), ck = _p2aK(claimed);',
     "RegExp other than as new RegExp(...)"),
    ('  const ca = _p2aA(claimed.split(new RegExp("' + BS * 2 + '" + "s")).join("")), ck = _p2aK(claimed);',
     "class escape " + BS + "s"),
    ('  const kk = "toLower" + "Case"; const ca = _p2aA(claimed[kk]()), ck = _p2aK(claimed);',
     "computed member access indexed by 'kk'"),
    ('  const ca = _p2aA(claimed.normalize ("NFKC")), ck = _p2aK(claimed);', "banned token 'normalize'"),
    ("  const ca = _p2aA(claimed. trim()), ck = _p2aK(claimed);", "banned token '.trim' in code"),
    ('  const i = "toLower" + "Case"; const ca = _p2aA((claimed[i])()), ck = _p2aK(claimed);',
     "a call on a parenthesised expression"),
    ('  const i = "toLower" + "Case"; const ca = _p2aA(claimed[i]?.()), ck = _p2aK(claimed);',
     "a call on a computed member"),
    ('  const i = "toLower" + "Case"; const ca = _p2aA(claimed[i]()), ck = _p2aK(claimed);',
     "a call on a computed member"),
    ("  const g2 = claimed.at.bind(claimed); " + CA.strip(), "banned token '.bind('"),
    ('  const ca = _p2aA(claimed.replace(new RegExp("x", "u"), "")), ck = _p2aK(claimed);',
     "new RegExp with an argument that is not one static string"),
    ('  const ca = _p2aA(claimed.replace(new RegExp("a.b"), "")), ck = _p2aK(claimed);', "unescaped '.'"),
    ('  const ca = _p2aA(claimed.replace(new RegExp(claimed), "")), ck = _p2aK(claimed);',
     "new RegExp with an argument that is not one static string"),
    # C-3 (NOTE_path2a_fourth_pass_2026_09_30): the fourth review's three plants, each of which the pass-3 scan accepted
    ('  const ca = _p2aA(claimed.match("' + BS * 2 + '" + "s") ? "" : claimed), ck = _p2aK(claimed);',
     "banned token '.match('"),
    ('  const ca = _p2aA(claimed.search("' + BS * 2 + '" + "s") > 0 ? "" : claimed), ck = _p2aK(claimed);',
     "banned token '.search('"),
    ('  const w = new String(claimed); w.q = w.normalize; const ca = _p2aA(w.q("NFKC")), ck = _p2aK(claimed);',
     "banned token 'normalize'"),
    ('  const w = new String(claimed); w.q = w.normalize; const ca = _p2aA(w.q("NFKC")), ck = _p2aK(claimed);',
     "banned token 'new String('"),
    ('  const ca = _p2aA([...claimed.matchAll(new RegExp("x"))].length ? claimed : ""), ck = _p2aK(claimed);',
     "banned token '.matchAll('"),
])
def test_the_token_scan_refuses_the_reviews_plants(new, what):
    block = R.js_block()
    assert block.count(CA) == 1
    problems = js_problems(block.replace(CA, new))
    assert any(what in p for p in problems), problems


# ---- the error fallback -----------------------------------------------------------------------------------------------

def test_error_fallback(M, monkeypatch):
    def boom(c, f):
        raise RuntimeError("planted")

    monkeypatch.setattr(N, "_p2a_decide", boom)
    seen = 0
    for c in R.repro_cases():
        try:
            a = M.gate_diff_text(c["summary"], c["diff"]).to_dict()
        except Exception:
            continue
        b = N.gate_diff_text(c["summary"], c["diff"]).to_dict()
        assert R.relation(a, b, False, PHRASES, allow_error=True) == [], c["id"]
        for x, y in zip(a["claims"], b["claims"]):
            if (x["kind"], x["verdict"]) in R.REACH:
                seen += 1
                assert y["verdict"] == "UNCHECKABLE" and R.phrase_key(y["why"], PHRASES) == "error"
    assert seen > 100


def test_error_fallback_port(work):
    node("--error-fallback", work / "diffgate_main_reference.js", work / "in.json", work / "err.json")
    seen = 0
    for d in json.loads((work / "err.json").read_text(encoding="utf-8")):
        if "error" in d["main"]:
            continue
        assert R.relation(d["main"], d["new"], False, PHRASES, allow_error=True) == [], d["id"]
        for x, y in zip(d["main"]["claims"], d["new"]["claims"]):
            if (x["kind"], x["verdict"]) in R.REACH:
                seen += 1
                assert y["verdict"] == "UNCHECKABLE" and R.phrase_key(y["why"], PHRASES) == "error", d["id"]
    assert seen > 1000


# ---- the checks refuse what they exist to refuse ----------------------------------------------------------------------

def _twins(M):
    c = next(x for x in R.repro_cases() if x["id"] == "prereg:121-dotfile-twins")
    return M.gate_diff_text(c["summary"], c["diff"]).to_dict(), N.gate_diff_text(c["summary"], c["diff"]).to_dict()


def _mutants(b):
    """Planted branch records, each outside the relation in one way."""
    def m(f):
        x = json.loads(json.dumps(b))
        f(x)
        return x
    err = R.reason("VERIFIED", "#97, #121", PHRASES["error"], "diff status 'A' for 'pr_agent.toml'")
    return {
        "detail moved": m(lambda x: x["claims"][1]["detail"].update(path="x")),
        "text moved": m(lambda x: x["claims"][1].update(text="x")),
        "reason moved alone": m(lambda x: x["claims"][1].update(why="x")),
        "verdict moved to an accusation": m(lambda x: x["claims"][1].update(verdict="CONTRADICTED")),
        "UNCHECKABLE decided": m(lambda x: x["claims"][2].update(verdict="VERIFIED")),
        "abstention without main's reason": m(lambda x: x["claims"][1].update(
            verdict="UNCHECKABLE", why=R.reason("VERIFIED", "#97", PHRASES["dir"], "x"))),
        "abstention naming another verdict": m(lambda x: x["claims"][1].update(
            verdict="UNCHECKABLE", why=R.reason("CONTRADICTED", "#97", PHRASES["dir"],
                                                "diff status 'A' for 'pr_agent.toml'"))),
        "error phrase": m(lambda x: x["claims"][1].update(verdict="UNCHECKABLE", why=err)),
        "gate verdict not recomputed": m(lambda x: x.update(verdict="PASS")),
        "a field moved": m(lambda x: x.update(uncovered_sentences=x["uncovered_sentences"] + 1)),
        "a claim dropped": m(lambda x: x["claims"].pop()),
    }


def test_the_relation_refuses_planted_records(M):
    a, b = _twins(M)
    assert R.relation(a, b, False, PHRASES) == []
    for name, x in _mutants(b).items():
        assert R.relation(a, x, False, PHRASES) != [], name
    assert R.strict_alike(b, b) == []
    assert R.strict_alike(b, _mutants(b)["verdict moved to an accusation"]) != []


def test_a_planted_strict_skip_is_refused(M, inputs):
    """Integration-1: an overlay that skips itself under --strict keeps every relation of pass 1; the strict check
    refuses it."""
    text = R.lf(R.INSTRUMENT)
    block = R.py_block(text)
    old = "    todo = [c for c in g.claims if (c.kind, c.verdict) in _P2A_REACH]\n"
    assert block.count(old) == 1
    mod = R.module_from(text.replace(block, block.replace(old, old[:-2] + " and not strict]\n")), "_p2a_plant_strict")
    caught = []
    for sname, iid, summary, diff in inputs[:600]:
        try:
            off, on = mod.gate_diff_text(summary, diff).to_dict(), mod.gate_diff_text(summary, diff, strict=True).to_dict()
        except Exception:
            continue
        if R.strict_alike(off, on):
            caught.append(iid)
    assert caught, "the strict check did not refuse an overlay that skips itself under --strict"


PORT_PLANTS = [
    ('      c.verdict = "UNCHECKABLE";', '      c.verdict = "CONTRADICTED";'),
    ("      c.why = _p2aReason(c.verdict, hit[1], hit[0], c.why);", "      c.why = _p2aReason(c.verdict, hit[1], hit[0], \"\");"),
    ('  g.verdict = (contradicted || (strict && uncheckable)) ? "FAIL" : "PASS";', ""),
    ("    if (hit !== null) {", "    c.detail = {};\n    if (hit !== null) {"),
    # Integration-1: the overlay skipped under --strict
    ('  const todo = g.claims.filter(c => _P2A_REACH.has(c.kind + "|" + c.verdict));',
     '  const todo = g.claims.filter(c => _P2A_REACH.has(c.kind + "|" + c.verdict) && !strict);'),
    # I-1 (NOTE_path2a_third_pass_2026_09_30): the review's two mutants, which pass 2's --relation could not see
    ('  g.verdict = (contradicted || (strict && uncheckable)) ? "FAIL" : "PASS";',
     '  g.verdict = (contradicted || (strict && uncheckable)) ? "FAIL" : "PASS";\n  g.unparsed_claims = g.claims.map(c => c.text);'),
    ("      c.verdict = \"UNCHECKABLE\";", "      c.verdict = \"UNCHECKABLE\";\n      c.main_verdict = c.verdict;"),
]


@pytest.mark.parametrize("old,new", PORT_PLANTS)
def test_the_port_relation_refuses_a_planted_port(old, new, work, tmp_path):
    text = R.lf(R.PORT)
    block = R.js_block(text)
    assert block.count(old) == 1, old
    planted = tmp_path / "diffgate_planted.js"
    planted.write_bytes(text.replace(block, block.replace(old, new)).encode("utf-8"))
    pairs = json.loads((R.DIFFERENTIAL / "path2a_pairs.json").read_text(encoding="utf-8"))
    (tmp_path / "in.json").write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf-8")
    node("--relation", work / "diffgate_main_reference.js", tmp_path / "in.json", tmp_path / "rel.json", planted)
    assert json.loads((tmp_path / "rel.json").read_text(encoding="utf-8"))["counts"]["broken"] > 0


# ---- the reproductions and what must not move -------------------------------------------------------------------------

def reason(v, d, k, why):
    return R.reason(v, d, PHRASES[k], why)


def test_reproductions(M):
    cases = {c["id"]: c for c in R.repro_cases()}

    def rows(cid):
        c = cases[cid]
        g = N.gate_diff_text(c["summary"], c["diff"])
        return g.verdict, [(x.kind, x.verdict, x.why) for x in g.claims]

    w = "accusation WITHHELD pending the EXTERNAL-1 repair"
    assert rows("prereg:101-the-issue") == ("PASS", [
        ("symbol_added", "UNCHECKABLE", reason("VERIFIED", "#101", "symbol", "added lines do define function 'backoff'")),
        ("tests_added", "UNCHECKABLE", reason("VERIFIED", "#101", "tests", "diff adds 2 test functions, claim says 2"))])
    assert rows("prereg:121-dotfile-twins") == ("FAIL", [
        ("files_changed_count", "UNCHECKABLE",
         reason("CONTRADICTED", "#121", "count", "diff changes 2 files, claim says 3")),
        ("file_created", "VERIFIED", "diff status 'A' for 'pr_agent.toml'"),
        ("file_deleted", "UNCHECKABLE", f"'pr_agent.toml' is status 'A', claim wants 'D' — {w}"),
        ("only_touches", "CONTRADICTED", "paths outside 'src': ['pr_agent.toml', 'github/workflows/ci.yml']")])
    assert rows("prereg:97-canonical") == ("PASS", [
        ("file_created", "UNCHECKABLE", f"'integrations/git/README.md' is status 'M', claim wants 'A' — {w}")])
    assert rows("prereg:97-reversed") == ("PASS", [
        ("file_touched", "VERIFIED", "diff status 'A' for 'integrations/git/readme.md'")])
    assert rows("prereg:97-directory-claim-by-base-name") == ("PASS", [
        ("file_created", "UNCHECKABLE", reason("VERIFIED", "#97", "dir", "diff status 'A' for 'readme.md'"))])


def test_the_path_accusation_is_still_withheld():
    assert N.WITHHOLD_PATH_ACCUSATION is True, (
        "a path claim can now be CONTRADICTED, and a path CONTRADICTED is outside the PATH-2a overlay's REACH: add "
        "(file_created | file_deleted | file_touched, CONTRADICTED) to _P2A_REACH in both ports, with the #97 and #121 "
        "rules for it, before this licence returns")
    assert N.P2A_DIRECTORY_BASENAME_ABSTAINS is True


def test_demo_is_byte_identical(M):
    out_main, out_new = io.StringIO(), io.StringIO()
    with redirect_stdout(out_main):
        M._demo()
    with redirect_stdout(out_new):
        N._demo()
    assert out_new.getvalue() == out_main.getvalue()


def test_committed_capsules_and_charon_lines_are_unaffected(M):
    from styxx import charon
    root = R.ROOT
    log = [json.loads(line) for line in (root / "papers" / "charon" / "charon.log.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    lines = [e for e in log[1:] if e.get("kind") == "capsule-diffgate"]
    assert {e["subject"]["name"] for e in lines} == {"DOGFOOD_session_2026_08_31.capsule.html",
                                                    "HANDOFF_capsule_v02_2026_08_31.capsule.html"}
    import base64
    for e in lines:
        path = root / e["subject"]["path"]
        payload = charon._capsule_payload(path)
        summary = base64.b64decode(payload["summary"]["b64"]).decode("utf-8")
        diff = base64.b64decode(payload["diff"]["b64"]).decode("utf-8")
        assert N.gate_diff_text(summary, diff).to_dict() == M.gate_diff_text(summary, diff).to_dict(), path.name
        fresh = charon.derive(path, root)
        assert [k for k in charon._COMPARED_KEYS if charon.jcs(fresh[k]) != charon.jcs(e[k])] == [], path.name


def test_the_bookmarklet_source_is_the_current_port():
    """bookmarklet_src.js is what build_bookmarklet.py assembles from diffgate.js; --check (terser) is below."""
    sys.path.insert(0, str(R.ROOT / "web" / "gate"))
    try:
        import build_bookmarklet
    finally:
        sys.path.pop(0)
    committed = (R.ROOT / "web" / "gate" / "bookmarklet_src.js").read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
    assert committed == build_bookmarklet.source().replace("\r\n", "\n")


def test_build_bookmarklet_check(tmp_path):
    """build_bookmarklet.py --check rewrites bookmarklet_src.js before it compares, so it runs on a copy of the files
    it reads, never on the checkout (NOTE_path2a_third_pass_2026_09_30, I-8)."""
    if shutil.which("terser") is None:
        pytest.skip("terser is not on PATH; build_bookmarklet.py --check cannot rebuild the minified bytes here")
    for name in ("build_bookmarklet.py", "diffgate.js", "bookmarklet_ui.js", "bookmarklet.min.js", "bookmarklet.href.txt"):
        (tmp_path / name).write_bytes((R.ROOT / "web" / "gate" / name).read_bytes().replace(b"\r\n", b"\n"))
    r = subprocess.run([sys.executable, str(tmp_path / "build_bookmarklet.py"), "--check"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "matches" in r.stdout, r.stdout + r.stderr


def test_the_action_shows_an_overlay_reason_whole(tmp_path, monkeypatch):
    """A-5: the Action's job-summary table cut every reason at 100 characters, which never reached main's reading
    in a reason the overlay wrote."""
    spec = importlib.util.spec_from_file_location("diffgate_action_p2a", R.ROOT / "diffgate_action.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    c = next(x for x in R.repro_cases() if x["id"] == "prereg:97-directory-claim-by-base-name")
    ev = tmp_path / "event.json"
    ev.write_text(json.dumps({"pull_request": {"number": 1, "body": c["summary"], "url": "https://api.github.test/pr/1"}}),
                  encoding="utf-8")
    for k, v in {"GITHUB_EVENT_PATH": str(ev), "GITHUB_EVENT_NAME": "pull_request", "GH_TOKEN": "x",
                 "GITHUB_STEP_SUMMARY": str(tmp_path / "sum.md"), "STYXX_STRICT": "false",
                 "STYXX_SOFT_FAIL": "false"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(mod, "api", lambda url, accept: c["diff"])
    assert mod.main() == 0
    table = (tmp_path / "sum.md").read_text(encoding="utf-8")
    assert "withheld by PATH-2a (#97)" in table and "main's reading: diff status 'A' for 'readme.md' |" in table
    # A-2 (NOTE_path2a_third_pass_2026_09_30): a reason that only contains the overlay's words -- a DECLARE-1
    # MALFORMED value a PR author writes -- is cut at 100 characters, as main cuts it.
    body = "```styxx\nfiles_changed: 3 withheld by PATH-2a (" + "A" * 300 + "\n```"
    diff = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
    ev.write_text(json.dumps({"pull_request": {"number": 1, "body": body, "url": "https://api.github.test/pr/1"}}),
                  encoding="utf-8")
    (tmp_path / "sum.md").write_text("", encoding="utf-8")
    monkeypatch.setattr(mod, "api", lambda url, accept: diff)
    assert mod.main() == 0
    table = (tmp_path / "sum.md").read_text(encoding="utf-8")
    row = next(line for line in table.splitlines() if "MALFORMED" in line)
    why = row[:-2].rsplit(" | ", 1)[1]
    assert why.startswith("MALFORMED") and len(why) == 100, row
    assert mod._OVERLAY_KINDS == {kind for kind, _v in N._P2A_REACH}
    # Pass 4 (NOTE_path2a_fourth_pass_2026_09_30, the integration lens): each of the three conditions is needed. A
    # reason of a kind the overlay may move that holds the form only mid-string, and a decided claim whose reason
    # starts with the form, are cut at 100 as main cuts them; the overlay's own reason is shown whole.
    form = "VERIFIED withheld by PATH-2a (#97): "
    rows = [("mid", N.DiffClaim(kind="file_touched", text="mid", detail={}, verdict="UNCHECKABLE",
                                why="x " + form + "A" * 300)),
            ("decided", N.DiffClaim(kind="file_touched", text="decided", detail={}, verdict="CONTRADICTED",
                                    why=form + "B" * 300)),
            ("ours", N.DiffClaim(kind="file_touched", text="ours", detail={}, verdict="UNCHECKABLE",
                                 why=form + "C" * 300))]
    fake = N.DiffGate(verdict="FAIL", base="(diff-text)", head="(diff-text)", claims=[c for _k, c in rows],
                      measured=True)
    monkeypatch.setattr(mod, "gate_diff_text", lambda *a, **k: fake)
    (tmp_path / "sum.md").write_text("", encoding="utf-8")
    mod.main()
    table = (tmp_path / "sum.md").read_text(encoding="utf-8")
    shown = {line.split(" | ")[1]: line[:-2].rsplit(" | ", 1)[1] for line in table.splitlines()
             if line.startswith("| ") and " | " in line and line.split(" | ")[1] in ("mid", "decided", "ours")}
    assert len(shown["mid"]) == 100 and len(shown["decided"]) == 100, shown
    assert shown["ours"] == rows[2][1].why


def test_the_port_differential_reads_mains_corpora_only():
    """Integration-3 and -4: path2a_pairs.json (the overlay's own pins, two of them inputs main's own ports read
    differently) is not one of the port differential's corpora, whose documented run reads 0 disagreements; and the
    recall script prints main's committed corpora apart from the overlay's own pins."""
    line = next(x for x in R.lf(R.DIFFERENTIAL / "py_side.py").split("\n") if x.startswith("CORPORA = ("))
    assert "declare1_pairs.json" in line and "path2a_pairs.json" not in line
    js = next(x for x in R.lf(R.DIFFERENTIAL / "js_side.js").split("\n") if x.startswith("for (const name of ["))
    assert "declare1_pairs.json" in js and "path2a_pairs.json" not in js
    recall = R.lf(R.DIFFERENTIAL / "path2a_recall.py")
    assert "TOTAL main" in recall and 'report["own"]' in recall and "--path-flavour" in recall


# ---- the cost per call ------------------------------------------------------------------------------------------------

def _timing_cases() -> list[dict]:
    e, zh, nbsp = chr(0xE9), chr(0x4E2D), chr(0xA0)

    def wide_def(n, ch):
        return {"id": f"wide-def-{n}", "summary": "Added function foo. Added 1 test.",
                "diff": ("--- a/src/app.py\n+++ b/src/app.py\n@@ -1,3 +1,2 @@\n-def " + ch * n + "\n-def test_" + ch * n
                         + "\n+def test_a():\n+def foo():\n")}

    def files(k, claims):
        parts = [f"--- a/d{i % 50}/f{i}.py\n+++ b/d{i % 50}/f{i}.py\n@@ -1 +1 @@\n-x\n+y\n" for i in range(k)]
        summ = [f"Modified d{i % 50}/f{(i * 37) % k}.py." if i % 2 else f"- d{i % 50}/f{(i * 37) % k}.py: tweak"
                for i in range(claims)]
        return {"id": f"files-{k}-{claims}", "summary": " ".join(summ), "diff": "".join(parts)}

    def same_base(k, claims):
        # A-1 (NOTE_path2a_third_pass_2026_09_30): every file has one base name, so the base-name group is the diff
        parts = [f"--- a/pkgs/p{i}/__init__.py\n+++ b/pkgs/p{i}/__init__.py\n@@ -1 +1 @@\n-x\n+y\n" for i in range(k)]
        summ = [f"Modified pkgs/p{(i * 7) % k}/__init__.py." for i in range(claims)]
        return {"id": f"same-base-{k}-{claims}", "summary": " ".join(summ), "diff": "".join(parts)}

    def symbols(claims, removed):
        # A-1: many symbol claims over a large removed side full of `def` sites
        body = "".join(f"-def g{j}(): pass\n" for j in range(removed)) + "".join(f"+def f{j}():\n" for j in range(claims))
        return {"id": f"symbols-{claims}-{removed}", "summary": " ".join(f"Added function f{j}." for j in range(claims)),
                "diff": "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n" + body}

    def class_nbsp(claims, removed):
        # A-1: `class` claims whose names sit beside NBSP runs in the removed lines
        line = "-class" + nbsp * 50 + "Foo" + nbsp * 50 + "class" + " " * 5 + "\n"
        body = line * removed + "+class Foo:\n" + "".join(f"+class Foo{j}:\n" for j in range(claims))
        return {"id": f"class-nbsp-{claims}-{removed}",
                "summary": " ".join(f"Added class Foo{j}." for j in range(claims)) + " Added class Foo.",
                "diff": "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n" + body}

    long_line = {"id": "long-line-20000", "summary": "Added 1 test.",
                 "diff": ("--- a/tests/test_x.py\n+++ b/tests/test_x.py\n@@ -1 +1 @@\n-def test_a():\n+x "
                          + "def test_a " * 20000 + "\n")}

    # A-1 (NOTE_path2a_fourth_pass_2026_09_30): the fourth review's Q1 to Q3, a summary of about 64 KB (GitHub's
    # pull-request body limit) whose claims each met every run or zone of the summary; ea677740 took 3.8 s, 2.6 s and
    # 0.6 s on them in Python. Q4 is the same for the count guard this pass adds.
    one = ("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1,2 @@\n-def foo(): pass\n"
           "+def foo(): return 1\n+x = 1\n")
    twins = one + "".join(f"diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n@@ -1 +1 @@\n-a\n+b\n" for p in (".env", "env"))

    def fill(head, n, tail):
        body = head * n + "\n"
        return body + tail * ((65536 - len(body)) // len(tail))

    q = [{"id": "q1-paths-by-wordish-runs", "summary": fill("Modified a.py. ", 2184, e + " "), "diff": one},
         {"id": "q2-symbols-by-wordish-runs", "summary": fill("Added function foo. ", 1638, e + " "), "diff": one},
         {"id": "q3-scopes-by-zones", "summary": fill("Only touches a.py. ", 1724, "only " + e + ".\n"), "diff": one},
         {"id": "q4-counts-by-digit-runs", "summary": fill("3 files changed. ", 1900, e + "3 "), "diff": twins}]
    return [wide_def(5000, e), wide_def(50000, e), wide_def(50000, zh), files(2000, 200), same_base(2000, 200),
            symbols(500, 50000), class_nbsp(300, 20000), long_line] + q


# The overlay's own time, the least of three runs on main's record for the same input (NOTE_path2a_third_pass_2026_09_30,
# I-2: pass 2 subtracted two single wall-clock samples of whole calls). The shapes: a `def` beside a 5,000- or
# 50,000-code-point run, one 220 KB added line of `def test_a`, 2,000 files with 200 path claims under 50 directories and
# under one base name, 500 symbol claims over 50,000 removed `def` lines, 300 class claims beside NBSP runs, and 64 KB
# summaries whose path, symbol, scope and count claims each meet thousands of runs or zones. Bound: 0.5 s in Python,
# 0.3 s in the port. On 40bba05b the overlay alone took 0.80 s on the one-base-name case, 38.6 s on the symbols and
# 16.8 s on the NBSP classes in Python (CPython 3.12.10); on ea677740, 3.8 s on q1. The figures at this head are in
# NOTE_path2a_fourth_pass_2026_09_30 and the README, measured the way this test measures them.
OVERLAY_S = (0.5, 0.3)


def test_cost_per_call_python(M):
    for it in _timing_cases():
        best = float("inf")
        for _ in range(3):
            g = M.gate_diff_text(it["summary"], it["diff"])
            t0 = time.perf_counter()
            N._p2a_abstain(g, False, lambda: N._P2aFacts(it["diff"], None, it["summary"]))
            best = min(best, time.perf_counter() - t0)
            assert not any(PHRASES["error"] in c.why for c in g.claims), it["id"]   # a failing overlay is fast too
        assert best < OVERLAY_S[0], (it["id"], best)


def test_cost_per_call_port(work, tmp_path):
    (tmp_path / "in.json").write_text(json.dumps(_timing_cases(), ensure_ascii=False), encoding="utf-8")
    node("--overlay-timing", work / "diffgate_main_reference.js", tmp_path / "in.json", tmp_path / "t.json")
    for d in json.loads((tmp_path / "t.json").read_text(encoding="utf-8")):
        assert d["overlay"] < 1000 * OVERLAY_S[1], d


# ---- (C) cross-port ---------------------------------------------------------------------------------------------------

def _seen(c):
    """A claim's verdict as the overlay left it, and its phrase key when the overlay wrote the reason."""
    return c["verdict"], R.phrase_key(c["why"], PHRASES)


def _kvd(c):
    return c["kind"], c["verdict"], json.dumps(c["detail"], sort_keys=True)


_READ = ("path", "name", "prefix", "prefix2", "n")


def _one_match(x, u):
    """Whether two claims with the same kind and verdict and different details may be one match read two ways: they
    read the same fields, each field's two values nest (one template read on past where the other stopped), and each
    claim's values lie in the other's text."""
    dx, du = x["detail"] or {}, u["detail"] or {}
    keys = [k for k in _READ if k in dx or k in du]
    return all(isinstance(dx.get(k), str) and isinstance(du.get(k), str) and (dx[k] in du[k] or du[k] in dx[k])
               and dx[k] in u["text"] and du[k] in x["text"] for k in keys)


def cross_port(a, b, ja, jb):
    """(C) for one input (NOTE_path2a_third_pass_2026_09_30, C-1; NOTE_path2a_fourth_pass_2026_09_30, C-2). a, b:
    main's and the branch's Python records; ja, jb: the same in the port. Returns the claim counts per key, the splits,
    and whether every claim pairs. By construction, since a decision reads nothing of a claim but its kind, verdict and
    detail (and the door's bytes): claims main's ports give the same (kind, verdict, detail) are decided alike, at the
    same position, matched across the lists in order, or anywhere in either list. Measured on named sets: the same
    position with the same (kind, verdict, text); and, in lists of equal length, the claims left over with the same kind
    and verdict whose details differ but nest and lie in each other's text (`_one_match`: one match the two templates
    read apart, the `extract` guards' work). Left-over claims that are not one match (two ports reading different sentences)
    are not paired, and an input with such a claim is not one whose claims all pair."""
    c = collections.Counter()
    splits = []
    A, B, X, Y = a["claims"], b["claims"], ja["claims"], jb["claims"]
    same_len = len(A) == len(X)
    if same_len:
        for k, (x, y, u, v) in enumerate(zip(A, X, B, Y)):
            for key, f in (("position: kind, verdict, detail", _kvd),
                           ("position: kind, verdict, text", lambda z: (z["kind"], z["verdict"], z["text"]))):
                if f(x) == f(y):
                    c[key] += 1
                    if _seen(u) != _seen(v):
                        splits.append((key, k, _seen(u), _seen(v)))
    decided = collections.defaultdict(set)
    for side, mains, news in (("python", A, B), ("port", X, Y)):
        for x, u in zip(mains, news):
            decided[_kvd(x)].add((side, _seen(u)))
    for k, got in decided.items():
        c["any: kind, verdict, detail"] += len(got)
        if len({s for _side, s in got}) > 1:
            splits.append(("any: kind, verdict, detail", k, sorted(got)))
    free = list(range(len(X)))
    pairs, rest = [], []
    for n, x in enumerate(A):
        m = next((k for k in free if _kvd(X[k]) == _kvd(x)), None)
        if m is None:
            rest.append(n)
        else:
            free.remove(m)
            pairs.append((n, m, "matched: kind, verdict, detail"))
    every = same_len
    if same_len:
        for n, m in zip(rest, free):
            if (A[n]["kind"], A[n]["verdict"]) == (X[m]["kind"], X[m]["verdict"]) and _one_match(A[n], X[m]):
                pairs.append((n, m, "left over, one match: kind and verdict"))
            else:
                every = False
    for n, m, key in pairs:
        c[key] += 1
        if _seen(B[n]) != _seen(Y[m]):
            splits.append((key, n, m, _seen(B[n]), _seen(Y[m])))
    if every and b["verdict"] != jb["verdict"]:
        splits.append(("gate verdict where every claim pairs",))
    return c, splits, every


# Inputs where main's two gate verdicts agree and the overlay's differ (C-5, disclosed): some claim rows differ between
# main's two ports (a line break or white space CPython reads and the port does not, which BC-1 then reads as a claim
# with no Python file), and each port withholds a claim its own main decides. A sixth fails here.
GATE_SPLITS_MAIN_AGREES = {
    "path2:f2-a-vertical-tab-is-not-a-line-break", "path2:f2-a-form-feed-is-not-a-line-break",
    "path2:f2-a-line-separator-is-not-a-line-break", "path2:f2-a-paragraph-separator-is-not-a-line-break",
    "path2:y2-a-changed-test-beside-a-created-bom-test-under-bare-hunks"}


def test_cross_port_decisions(M, inputs, work):
    node("--decisions", work / "diffgate_main_reference.js", work / "in.json", work / "dec.json")
    js = {d["id"]: d for d in json.loads((work / "dec.json").read_text(encoding="utf-8"))}
    c = collections.Counter()
    splits, gate = [], set()
    for i, row in enumerate(inputs):
        j = js[R.uid(i, row)]
        try:
            a = M.gate_diff_text(row[2], row[3]).to_dict()
            b = N.gate_diff_text(row[2], row[3]).to_dict()
        except Exception:
            continue
        if "error" in j["main"] or "error" in j["new"]:
            assert "error" in j["main"] and "error" in j["new"], row[1]
            continue
        counts, found, every = cross_port(a, b, j["main"], j["new"])
        c.update(counts)
        c["inputs whose claims all pair"] += every
        splits += [(row[0], row[1]) + s for s in found]
        if a["verdict"] == j["main"]["verdict"] and b["verdict"] != j["new"]["verdict"]:
            gate.add(row[1])
    print("cross-port:", dict(c), "gate splits where main's gates agree:", sorted(gate))
    assert not splits, splits[:10]
    assert c["position: kind, verdict, detail"] > 18000 and c["left over, one match: kind and verdict"] > 20
    assert gate == GATE_SPLITS_MAIN_AGREES


D_MOD = "diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-x = 0\n+x = 1\n"
D_GUIDE = ("diff --git a/docs/guide.md b/docs/guide.md\n--- a/docs/guide.md\n+++ b/docs/guide.md\n@@ -1 +1 @@\n"
           "-old\n+new\n") + D_MOD
D_TWINS = D_MOD + "".join(f"diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n@@ -1 +1 @@\n-a\n+b\n"
                          for p in (".env", "env"))
D_CAFE = ("diff --git a/docs/r\u00e9sum\u00e9/index.md b/docs/r\u00e9sum\u00e9/index.md\n--- a/docs/r\u00e9sum\u00e9/index.md\n"
          "+++ b/docs/r\u00e9sum\u00e9/index.md\n@@ -1 +1 @@\n-a\n+b\n"
          "diff --git a/docs/caf\u00e9.md b/docs/caf\u00e9.md\nnew file mode 100644\n--- /dev/null\n+++ b/docs/caf\u00e9.md\n"
          "@@ -0,0 +1 @@\n+x\n")
# The reviews' cross-port reproductions that main's own ports read with different claim texts, details or sentence
# counts, so that one pinned expect cannot hold both: each port's decisions, pinned in each (want for Python, want for
# the port), and the gate verdict both must reach.
XPORT_CASES = [
    ("R1-bom-joined-sentences", "Modified src/app.py." + chr(0xFEFF) + "Tidied up.", D_MOD,
     [("VERIFIED", None)], [("VERIFIED", None)], "PASS"),
    ("R4-nel-after-only-prefix", "Only touches docs/." + chr(0x85) + "Thanks.", D_MOD,
     [("UNCHECKABLE", "extract")], [("UNCHECKABLE", "extract")], "PASS"),
    ("E1-emoji-release-note", chr(0x1F680) + chr(0x1F389) + " Release prep " + chr(0x1F9F9) + chr(0x1F527)
     + ": bumped the pinned dependencies, regenerated the lockfile, fixed two flaky network timeouts in the nightly "
       "CI job, and updated docs/guide.md for the next release.", D_GUIDE, [("VERIFIED", None)], [("VERIFIED", None)],
     "PASS"),
    # C-1 (NOTE_path2a_fourth_pass_2026_09_30): one count match that CPython reads as 33 and the port as 3 (X5, X5b);
    # ea677740 kept the Python's CONTRADICTED and withheld the port's, and the gates split FAIL / PASS
    ("X5-fullwidth-digit-before-the-count", chr(0xFF13) + "3 files changed.", D_TWINS,
     [("UNCHECKABLE", "extract")], [("UNCHECKABLE", "extract")], "PASS"),
    ("X5b-arabic-digit-before-the-count", chr(0x663) + "3 files changed.", D_TWINS,
     [("UNCHECKABLE", "extract")], [("UNCHECKABLE", "extract")], "PASS"),
    # each port reads a count the other does not (X2, X3): on ea677740 the gates split PASS / FAIL
    ("X2-nel-and-cjk-counts", "3 files" + chr(0x85) + "changed. " + chr(0x5171) + "5 files changed.", D_TWINS,
     [("UNCHECKABLE", "count")], [("UNCHECKABLE", "extract")], "PASS"),
    ("X3-fullwidth-and-cjk-counts", chr(0xFF13) + " files changed. " + chr(0x5171) + "5 files changed.", D_TWINS,
     [("UNCHECKABLE", "extract")], [("UNCHECKABLE", "extract")], "PASS"),
    # C-2 (fourth review): the realistic X6b, two different claims each port reads from a different sentence; the
    # decisions differ, as the two claims do, and neither is paired with the other
    ("X6b-accented-directory-and-a-created-file",
     "Changed the parser in `docs/r\u00e9sum\u00e9/index.md`. Added `docs/caf\u00e9.md`.", D_CAFE,
     [("VERIFIED", None)], [("UNCHECKABLE", "extract")], "PASS"),
]


def test_cross_port_reproductions(M, tmp_path):
    """Pass 3's C-1 and pass 4's C-1: the reviews' inputs whose claim text or detail differs between main's two ports.
    On 40bba05b R1, R4 and E1 split the ports; on ea677740 X5, X5b, X2 and X3 split the gate verdicts."""
    items = [{"id": cid, "summary": s, "diff": d} for cid, s, d, _py, _js, _gate in XPORT_CASES]
    (tmp_path / "in.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    ref = R.main_port_path(tmp_path)
    node("--decisions", ref, tmp_path / "in.json", tmp_path / "out.json")
    js = {d["id"]: d for d in json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))}
    for cid, s, d, want_py, want_js, gate in XPORT_CASES:
        a, b = M.gate_diff_text(s, d).to_dict(), N.gate_diff_text(s, d).to_dict()
        assert [_seen(x) for x in b["claims"]] == want_py, cid
        assert [_seen(x) for x in js[cid]["new"]["claims"]] == want_js, cid
        assert cross_port(a, b, js[cid]["main"], js[cid]["new"])[1] == [], cid
        assert b["verdict"] == js[cid]["new"]["verdict"] == gate, cid


# ---- tables pinned by enumeration -------------------------------------------------------------------------------------

def _set(s):
    return {ord(ch) for ch in s}


def _neutral():
    """The neutral code points, read off the Python block's regex class."""
    rx = re.compile("[" + N._P2A_NEUTRAL + "]")
    return {cp for cp in range(0x80, 0x10000) if rx.match(chr(cp))}


def test_python_tables():
    breaks = {cp for cp in range(0x110000) if len(("a" + chr(cp) + "b").splitlines()) == 2}
    space = {cp for cp in range(0x110000) if chr(cp).isspace()}
    rx_space = {cp for cp in range(0x110000) if re.match(r"\s", chr(cp))}
    holds_ascii = {cp for cp in range(0x80, 0x110000) if any(ord(x) < 128 for x in chr(cp).lower())}
    longer = {cp for cp in range(0x110000) if len(chr(cp).lower()) > 1}
    assert breaks == _set(N._P2A_PY_BREAKS)
    assert space == rx_space == _set(N._P2A_PY_SPACE), "CPython's white space (isspace and the regex class) moved"
    assert holds_ascii == {0x130, 0x212A} and longer == {0x130}, "the fold/wild lemma no longer holds here"
    assert {cp for cp in range(0x110000) if chr(cp).translate(N._P2A_FOLD) != chr(cp)} == \
        set(range(65, 91)) | {0x130, 0x212A}
    assert N._P2A_OWN == 0
    # The summary's classes (NOTE_path2a_third_pass_2026_09_30, C-1; NOTE_path2a_fourth_pass_2026_09_30, B-2): the
    # neutral code points and the four divergent ones are never a word character here, nor fold to an ASCII letter
    # under re.I; the neutral ones have no case, lie in the Basic Multilingual Plane (one UTF-16 unit each), and are
    # white space in both ports or in neither; every other code point from 0x80 up is wordish, and so is every one
    # CPython's templates read as a word character or fold to a letter.
    neutral = _neutral()
    never_word = neutral | {0x85, 0x2028, 0x2029, 0xFEFF}
    assert len(neutral) == 1828 and max(neutral) < 0x10000 and not neutral & _set(N._P2A_DIVERGENT)
    assert all(re.match(r"\w", chr(cp)) is None and not chr(cp).isalnum() for cp in never_word)
    assert all(re.match("[A-Za-z_]", chr(cp), re.I) is None for cp in never_word)
    assert all(chr(cp).lower() == chr(cp).upper() == chr(cp).casefold() == chr(cp) for cp in neutral)
    assert all((cp in _set(N._P2A_PY_SPACE)) == (cp in _set(N._P2A_JS_SPACE)) for cp in neutral)
    # every neutral code point assigned in Unicode 3.2 was punctuation, a symbol or a space there too, so none became
    # a letter, a mark or a number between the versions CI's interpreters carry (the two variation selectors aside)
    old = unicodedata.ucd_3_2_0
    assert all(old.category(chr(cp)) == "Cn" or old.category(chr(cp))[0] in "PSZ" or cp in (0xFE0E, 0xFE0F)
               for cp in neutral)
    wordish = {cp for cp in range(0x80, 0x110000) if N._P2A_WORDISH_RX.match(chr(cp))}
    assert wordish == set(range(0x80, 0x110000)) - never_word
    assert {cp for cp in range(0x110000) if N._P2A_BAD_RX.match(chr(cp))} == wordish | _set(N._P2A_DIVERGENT)
    assert {cp for cp in range(0x80, 0x110000)
            if re.match(r"\w", chr(cp)) or re.match("[A-Za-z_]", chr(cp), re.I)} <= wordish


def test_port_tables_and_constants(tmp_path):
    node("--tables", tmp_path / "t.json")
    t = json.loads((tmp_path / "t.json").read_text(encoding="utf-8"))
    print("node", t["node"], "unicode", t["unicode"])
    assert set(t["whitespace"]) == set(t["trim"]) == set(t["js_space"]), "the port's trim table moved"
    assert set(t["js_space"]) == _set(N._P2A_JS_SPACE) and set(t["py_space"]) == _set(N._P2A_PY_SPACE)
    assert set(t["lower_holds_ascii"]) == {0x130, 0x212A} and set(t["lower_longer"]) == {0x130}
    py_space, js_space = _set(N._P2A_PY_SPACE), set(t["js_space"])
    need = (_set(N._P2A_PY_BREAKS) - {0x0A, 0x0D}) | (py_space ^ js_space) | {0x2028, 0x2029}
    assert need <= _set(N._P2A_DIVERGENT), "a character the two ports read differently is not in the guard"
    assert set(t["divergent"]) == _set(N._P2A_DIVERGENT) and set(t["py_breaks"]) == _set(N._P2A_PY_BREAKS)
    assert t["headers"] == list(N._P2A_HEADERS)
    assert {tuple(x) for x in t["reach"]} == set(N._P2A_REACH) == set(R.REACH)
    assert t["phrases"] == N._P2A_PHRASES and t["kind_defect"] == N._P2A_KIND_DEFECT
    assert t["directory_rule"] is True and t["own"] == 1
    # the summary's classes in UTF-16 units (C-1): every unit of a surrogate pair is wordish, as its code point is
    never_word = _neutral() | {0x85, 0x2028, 0x2029, 0xFEFF}
    assert set(t["neutral"]) == _neutral() and t["neutral_cased"] == []
    assert set(t["not_wordish"]) == set(range(0x80)) | never_word
    assert set(t["not_bad"]) == (set(range(0x80)) - _set(N._P2A_DIVERGENT)) | _neutral()
    assert set(t["word_class"]) <= set(range(0x80)), "the port's word class reads outside ASCII"
