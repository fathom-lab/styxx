"""PATH-2a: main's diff gate, unchanged, plus an overlay that only abstains (NOTE_path2a_abstain_overlay_2026_09_30,
NOTE_path2a_second_pass_2026_09_30, NOTE_path2a_third_pass_2026_09_30, NOTE_path2a_fourth_pass_2026_09_30,
NOTE_path2a_fifth_pass_2026_09_30, NOTE_path2a_sixth_pass_2026_09_30, NOTE_path2a_seventh_pass_2026_09_30,
NOTE_path2a_eighth_pass_2026_10_01, NOTE_path2a_ninth_pass_2026_10_04).

The reference in every test here is `main` itself: this checkout's `styxx/diffgate.py` and `web/gate/diffgate.js` with
the PATH-2a block cut out and the door hooks reverted, asserted to hash to the files on `origin/main` 1cde8b82
(tests/_p2a_ref.py). No copy of main is committed.

(A) by construction -- on every committed input, both strict modes, both ports and the git door, each record equals
    main's except that a decided verdict in REACH may become UNCHECKABLE with the overlay's reason, and the gate
    verdict is main's formula over the claims; each claim reads the same under --strict as without it; where main
    raises, the branch raises the same exception type. The same with a run leg, a test report and a commit handed to
    both doors, where main decides `tests_pass` claims, which are outside REACH.
(C) cross-port, as the ninth pass restates it -- a decision reads the claim's kind, verdict and detail, main's counts
    and the door's bytes, never the claim's text, through code that asks no runtime a Unicode question. So
    (i) wherever main's two ports give claims the same kind, verdict and detail -- by position, matched across the two
    lists, or anywhere in either list -- the overlay's verdict and phrase are the same (asserted on every set read
    here); and (ii) on every input where main's two ports read the same claim list (one length; the same kind, verdict
    and detail at each position) the two final lists and the two gate verdicts are the same, in both strict modes
    (asserted on the committed inputs, the cross-port cases and three seeded sets, with a guard that the overlay
    withheld claims there). (iii) Where main's two lists differ nothing is promised: the tests count how often that
    happens, on which side (the description read apart; the same claims decided apart on the diff), and how often the
    two gate verdicts then differ under main and under the overlay, and pin the counts per runtime.
    Measured on the committed inputs and asserted there only: the same decision for claims with the same kind, verdict
    and text, and for claims left over whose details differ but nest and lie in each other's text (both heuristics for
    one match read two ways, the `extract` guards' work; each also pairs two different matches: L1 and OM1 are pinned,
    NOTE_path2a_fifth_pass_2026_09_30, C-3, and NOTE_path2a_sixth_pass_2026_09_30, C-3). The constant tables the
    overlay leans on are pinned by enumeration on the running engines.
And the static facts: the reconstruction, the self-checks over each block's source, the error fallback, the
reproductions, the cost per call, and what must not move (the demo, the committed capsules, charon's lines, the
bookmarklet source). Coverage (B) is in tests/test_diffgate_path2a_truth.py.
"""
from __future__ import annotations

import collections
import functools
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
import types
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
# pairs add the rest. Pass 5 (NOTE_path2a_fifth_pass_2026_09_30 and its corrections), on the inputs pass 4 pinned:
# `seam` 1 (the X2 pair, `count` before), and names read through NFKC 25 (24 on the seeded PATH-2a fuzz, whose removed
# lines write U+00A0, U+3000 or U+FEFF between `def` and the name, and #161's y5); no path claim moves. Its 20 new pairs
# add 5 `seam`, 6 `tests`, 5 `symbol` and 4 kept declared claims. Pass 6 (NOTE_path2a_sixth_pass_2026_09_30), on the
# inputs pass 5 pinned: C-1 keeps main's CONTRADICTED where the two ports' mains may read the claims apart (`divergent`
# counts -156 and scopes -105, `extract` scopes -154 and counts -3, `seam` -5, `tests` -83, `split` -3; almost all on
# the seeded PATH-2a fuzz and the text-seam set), and B-1 withholds a test or a name an unchanged line defines
# (`redefined` 15, `again` 5). Its 7 new pairs add `redefined` 2, `again` 1, `dir` 1, `dot_tier` 1 and `split` 1, a
# CONTRADICTED kept (C-1's diff part) and a VERIFIED scope kept. Pass 7 (NOTE_path2a_seventh_pass_2026_09_30), on the
# inputs pass 6 pinned: the trigger read in its template's order withholds 11 more `tests` CONTRADICTEDs (seeded PATH-2a
# fuzz), O-11 reads pictograph emoji as neutral, so 13 path VERIFIEDs are no longer `extract` (the text-seam set and one
# fuzz input), and the case-pair clause keeps 11 count CONTRADICTEDs (`count`; seeded fuzz whose paths differ in one
# letter outside ASCII). Its 5 new pairs add `tests` 4 and keep a CONTRADICTED tests claim (the case pair).
# Pass 8 (NOTE_path2a_eighth_pass_2026_10_01), on the inputs pass 7 pinned: B-1 withholds a count claim in [#CA, #A]
# wherever two changed paths differ only in case outside ASCII, whatever its verdict (`case_count` 42: 22 CONTRADICTEDs
# the case-pair clause kept and 13 VERIFIEDs, one of them `count` before, on the seeded PATH-2a fuzz; #161's six k2
# reproductions, Unicode 14 and 16 case pairs; the pass-7 case-pair pair), and no longer keeps a CONTRADICTED for them,
# so 5 more counts are withheld by the dot-twin rule (`count`) and the pair's tests claim (`tests`); B-2 reads the
# window a match can cover, so 5 scope CONTRADICTEDs of the text-seam set whose sentence holds such a character only
# outside it are decided, and withheld (`extract`: one follows `only`). Its 5 new pairs add 8 decided claims: `count`
# 1 and `tests` 2, and two CONTRADICTED tests claims kept (a `def` ending its line; a name running into an accent).
# Pass 9 (NOTE_path2a_ninth_pass_2026_10_04), on the inputs pass 8 pinned: the C-1 switch is removed, so every
# CONTRADICTED it kept is decided by its kind's rule: 498 more are withheld (`divergent` counts +156 and scopes +105,
# `extract` scopes +145 and counts +3, `seam` +5, `tests` +74, `redefined` +6, `split` +4), nearly all on the seeded
# PATH-2a fuzz and the text-seam set, whose summaries and diffs are built from the characters the switch read; no
# VERIFIED moves, and nothing withheld before is kept now.
ABSTENTIONS = {
    "windows": {
        "decided": 12718, "main raises": 8,
        "file_created:case": 3, "file_created:dir": 65, "file_created:divergent": 19, "file_created:dot": 20,
        "file_created:dot_earliest": 4, "file_created:dot_tier": 11, "file_created:extract": 43, "file_created:odd": 1,
        "file_created:tier": 9,
        "file_deleted:case": 1, "file_deleted:dir": 52, "file_deleted:divergent": 20, "file_deleted:dot": 21,
        "file_deleted:dot_earliest": 1, "file_deleted:dot_tier": 7, "file_deleted:extract": 52, "file_deleted:odd": 1,
        "file_deleted:tier": 4,
        "file_touched:dir": 244, "file_touched:divergent": 102, "file_touched:dot": 82, "file_touched:dot_tier": 72,
        "file_touched:extract": 139, "file_touched:odd": 5,
        "files_changed_count:case_count": 42,
        "files_changed_count:count": 375, "files_changed_count:divergent": 226, "files_changed_count:extract": 4,
        "files_changed_count:seam": 6,
        "only_touches:divergent": 108, "only_touches:extract": 154, "only_touches:only": 32, "only_touches:shape": 29,
        "symbol_added:again": 6, "symbol_added:extract": 24, "symbol_added:symbol": 184,
        "tests_added:redefined": 23, "tests_added:split": 5, "tests_added:tests": 437,
    },
}
# Under the POSIX flavour main reads `c:x.py` as a bare name not in the diff, so three decided drive-like claims are
# UNCHECKABLE on main to begin with: path2a:guard-drive-like-path claim 0, and fuzz 20260930:1033 and :1590.
ABSTENTIONS["posix"] = {**ABSTENTIONS["windows"], "decided": 12715, "file_touched:odd": 4}
del ABSTENTIONS["posix"]["file_created:odd"], ABSTENTIONS["posix"]["file_deleted:odd"]
# main's own reading depends on the interpreter's Unicode tables where the inputs probe letters added in Unicode 14 and
# 16 (#161's k2, x1 and y3 cases). Measured on CPython 3.12 (Unicode 15.0) and 3.14 (16.0); for Unicode 13.0 and 14.0
# (CI's 3.9 to 3.11) by re-reading the inputs with the five Unicode 14 letters present mapped to code points no
# version assigns (NOTE_path2a_second_pass_2026_09_30, I-10): no figure moved. On 16.0 main reads the y3 name
# `fo` + U+105C0 whole and CONTRADICTS it, so it is not a VERIFIED symbol the overlay withholds. Unicode 15.1 (CPython
# 3.13; NOTE_path2a_seventh_pass_2026_09_30, I-5): the committed inputs hold no code point assigned in 15.0 or 15.1
# (the sixth integration review's inventory: 99 distinct code points from 0x80 up, whose only letters added after 13.0
# are the five of Unicode 14, and whose unassigned ones are Unicode 16's), so its figures are 15.0's.
UNICODE_SAME = ("13.0.0", "14.0.0", "15.0.0", "15.1.0")


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
    if engine_unicode() == "16":                  # measured on Node 24.13.0: what this port withholds of what its main decides
        assert (rep["counts"]["decided"], rep["counts"]["abstained"]) == (12526, 2590), rep["counts"]


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
        # pass 4 (A-2): the facts object's shortcut (one view, counted once) gives what both views counted apart give;
        # pass 5 (B-1): with the removed lines no view reads; pass 6 (B-1): with the unchanged lines too
        unchanged = N._p2a_context(diff or "", fine) + N._p2a_joined(diff or "", " ")
        assert N._P2aFacts(diff or "").pairing() == \
            N._p2a_pairing(views, False, N._p2a_joined(diff or ""), unchanged), iid
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
        mine = [x[0] for x in N._p2a_pairing(N._p2a_views(row[3] or "", N._p2a_lines(row[3] or "", N._P2A_FINE)))]
        if got["views"][1] != got["main"] or got["views"] != mine:
            bad.append((row[1], got, mine))
    assert bad == [], bad[:5]


# ---- the git door -----------------------------------------------------------------------------------------------------

def _repo(tmp_path, name, base, head, quotepath=None):
    d = tmp_path / name
    d.mkdir()

    def git(*a):
        # I-6 (NOTE_path2a_fifth_pass_2026_09_30): core.longpaths, so a long pytest temp path works on Windows too
        return subprocess.run([GIT, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false",
                               "-c", "core.longpaths=true", *a], cwd=d, capture_output=True, check=True).stdout

    git("init", "-q")
    # I-6 (NOTE_path2a_eighth_pass_2026_10_01): kept in the repository, so main's own _git, which passes no -c, reads it
    git("config", "core.longpaths", "true")
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
        return subprocess.run([GIT, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false",
                               "-c", "core.longpaths=true", *a], cwd=d, input=data, capture_output=True,
                              check=True).stdout.decode().strip()

    git("init", "-q")
    git("config", "core.longpaths", "true")       # I-6 of the eighth pass, as in _repo
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
MOVED = "            moved = True\n"
HITS = "        hits = [(c, _p2a_decide(c, f)) for c in todo]\n"
TODO = "    todo = [c for c in g.claims if (c.kind, c.verdict) in _P2A_REACH]\n"
BEFORE_LOOP = "    moved = False\n"
REACH_END = '    ("tests_added", "VERIFIED"), ("tests_added", "CONTRADICTED"), ("symbol_added", "VERIFIED")})\n'
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
    # C-2 (NOTE_path2a_fifth_pass_2026_09_30): flag groups whose case or Unicode flag is not the leading letter, which
    # the pass-4 check accepted; (?si) reads the case table (K matches U+212A)
    ('_P2A_COARSE = re.compile("\\r\\n|\\r|\\n")', '_P2A_COARSE = re.compile("(?si)\\r\\n|\\r|\\n")', "inline flag (?si)"),
    ('_P2A_DIGITS = re.compile("[0-9]+")', '_P2A_DIGITS = re.compile("(?mi:[0-9]+)")', "inline flag (?mi:"),
    ('_P2A_WORDISH_RX = re.compile("[" + _P2A_WORDISH + "]")', '_P2A_WORDISH_RX = re.compile("(?xi)[" + _P2A_WORDISH + "]")',
     "inline flag (?xi)"),
    ('_P2A_DIGITS = re.compile("[0-9]+")', '_P2A_DIGITS = re.compile("(?-i:[0-9]+)")', "inline flag (?-i:"),
    # C-2 (NOTE_path2a_sixth_pass_2026_09_30): a named-character escape reads the Unicode name table when it compiles
    ('_P2A_CR = re.compile("\\r")', '_P2A_CR = re.compile("\\\\N{LATIN SMALL LETTER A}|\\r")',
     "named-character escape"),
    # A-1 (NOTE_path2a_eighth_pass_2026_10_01): the seventh review's three plants, each of which the pass-7 check
    # accepted (a record field reached through a `for` target), and the same reached through `:=`, a call's result, a
    # name bound from another, an augmented store and a helper's parameter
    (MOVED, MOVED + "            for cl in (g.claims,):\n                cl.pop()\n", ".pop() on what may hold"),
    (MOVED, MOVED + '            for dd in (c.detail,):\n                dd["n"] = "9"\n', "item store into what may hold"),
    (MOVED, MOVED + '            for dd in (c.detail,):\n                dd.setdefault("zz", 1)\n',
     ".setdefault() on what may hold"),
    (MOVED, MOVED + '            (dd := c.detail)["n"] = "9"\n', "item store into what may hold"),
    (MOVED, MOVED + "            [x.clear() for x in [c.detail]]\n", ".clear() on what may hold"),
    (MOVED, MOVED + '            x = [c.detail]\n            y = x[0]\n            y["n"] = "9"\n',
     "item store into what may hold"),
    (MOVED, MOVED + "            cl = g.claims\n            cl += [c]\n", "augmented store into cl"),
    (MOVED, MOVED + '            _p2a_reason(c.detail, "", "dir", "")["n"] = "9"\n', "item store into what may hold"),
    (MOVED, MOVED + '            def _poke(d):\n                d["n"] = "9"\n            _poke(c.detail)\n',
     "item store into what may hold"),
    # I-1 (NOTE_path2a_ninth_pass_2026_10_04): the eighth integration review's plant, which withheld a `tests_pass`
    # VERIFIED beside a count claim and passed the pass-8 check and every committed test, and the other ways to write a
    # claim outside the overlay's own set (`todo`, the claims of g.claims in reach)
    (HITS, HITS + '        hits = hits + [(c, ("tests", "#101")) for c in g.claims if c.kind == "tests_pass" and '
     'c.verdict == "VERIFIED"\n                       and any(x.kind == "files_changed_count" for x in g.claims)]\n',
     "hits bound other than by a comprehension over todo"),
    (HITS, HITS + '        hits.extend((c, ("tests", "#101")) for c in g.claims if c.kind == "tests_pass")\n',
     "hits read other than by the abstain loop"),
    (HITS, HITS + '        hits.append((g.claims[0], ("tests", "#101")))\n', "hits read other than by the abstain loop"),
    (HITS, HITS + "        more = hits\n", "hits read other than by the abstain loop"),
    (HITS, "        hits = [(c, _p2a_decide(c, f)) for c in g.claims]\n",
     "hits bound other than by a comprehension over todo"),
    (HITS, "        hits = [(g.claims[0], _p2a_decide(c, f)) for c in todo]\n",
     "hits bound other than by a comprehension over todo"),
    (TODO, TODO + '    todo = todo + [c for c in g.claims if c.kind == "tests_pass"]\n',
     "todo bound other than as the claims of g.claims in reach"),
    (TODO, '    todo = [c for c in g.claims if c.verdict in ("VERIFIED", "CONTRADICTED")]\n',
     "todo bound other than as the claims of g.claims in reach"),
    (TODO, TODO + "    todo.extend(c for c in g.claims)\n", "todo read other than as what a comprehension iterates"),
    (TODO, TODO + "    g = g\n", "g bound again in _p2a_abstain"),
    (TODO, '    global _P2A_REACH\n    _P2A_REACH = _P2A_REACH | {("tests_pass", "VERIFIED")}\n' + TODO,
     "_P2A_REACH is not bound once"),
    (REACH_END, REACH_END.replace("})\n", ', ("tests_pass", "VERIFIED")})\n'), "_P2A_REACH is not bound once"),
    (BEFORE_LOOP, BEFORE_LOOP + '    for c in g.claims:\n        c.verdict = "UNCHECKABLE"\n',
     "store into .verdict of a claim the abstain loop does not hold"),
    (BEFORE_LOOP, BEFORE_LOOP + '    g.claims[0].why = ""\n', "store into .why of a claim the abstain loop does not hold"),
    (MOVED, MOVED + '            c = g.claims[0]\n            c.why = ""\n',
     "the claim variable of the abstain loop bound again"),
    (MOVED, MOVED + "    for c, hit in hits:\n        c.why = c.why\n", "does not hold exactly one"),
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
# C-2 (NOTE_path2a_fifth_pass_2026_09_30): identifiers that turn a string into a number through the engine's own tables
# (StringToNumber skips the engine's white space: parseInt("\u30003", 10) is 3), refused as whole words in code, so
# `_p2aNumbers` and a comment naming them are not; the block reads digits through `_p2aInt`
JS_BANNED_WORDS = ("Number", "parseInt", "parseFloat", "isNaN", "isFinite",
                   # C-2 (NOTE_path2a_sixth_pass_2026_09_30): Math and Date convert their arguments with ToNumber, and a
                   # typed array converts what is stored into it; the block uses a Set and conditionals instead
                   "Math", "Date", "BigInt", "DataView", "Int8Array", "Uint8Array", "Uint8ClampedArray", "Int16Array",
                   "Uint16Array", "Int32Array", "Uint32Array", "Float16Array", "Float32Array", "Float64Array",
                   "BigInt64Array", "BigUint64Array",
                   # A-2 (NOTE_path2a_eighth_pass_2026_10_01): Object.assign, Object.defineProperty and a Proxy write a
                   # record's fields without a store the store scan reads
                   "Object", "Proxy")
JS_UNARY_AFTER = "(,=[:?!&|;{}<>*%~^"
JS_UNARY_KEYWORDS = ("return", "typeof", "void", "in", "of", "case", "throw", "yield", "await", "new", "delete")


def js_numeric_problems(dense: str) -> list:
    """C-2 (NOTE_path2a_sixth_pass_2026_09_30): loose equality, and a unary + or - before a name, a call, a bracket or a
    string, each of which runs StringToNumber (the engine's white space) on a string. `++` and `--` are passed over.
    Binary arithmetic and relational operators, and the numeric parameters of built-in methods, convert a string too;
    the scan cannot type their operands, and the block gives them numbers only (disclosed)."""
    out = []
    if re.search(r"(?<![=!<>])(?:==|!=)(?!=)", dense):
        out.append("loose equality (== or !=)")
    for m in re.finditer(r"(?<![+-])[+-](?![+=-])", dense):
        k = m.start()
        prev = dense[k - 1] if k else ""
        unary = (not prev or prev in JS_UNARY_AFTER or re.search(
            r"(?:^|[^A-Za-z0-9_$])(?:" + "|".join(JS_UNARY_KEYWORDS) + r")$", dense[:k]) is not None)
        if unary and (dense[k + 1:k + 2] == "" or re.match(r"[A-Za-z_$(\[]", dense[k + 1:k + 2])):
            out.append(f"unary {m.group()} before {dense[k + 1:k + 12]!r}")
    return out
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
        elif ch == "(" and not in_class and pattern[i + 1:i + 2] == "?" and re.match(r"[A-Za-z-]", pattern[i + 2:i + 3]):
            # C-2 (NOTE_path2a_fifth_pass_2026_09_30): regex modifiers, `(?i:...)`, live in V8 13.6
            out.append("regex modifiers " + pattern[i:i + 3])
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
    out += [f"banned word {t!r} in code" for t in JS_BANNED_WORDS
            if re.search(r"(?<![A-Za-z0-9_$])" + t + r"(?![A-Za-z0-9_$])", dense)]
    out += js_numeric_problems(dense)
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
    return out + js_store_problems(dense, strings)


# A-2 (NOTE_path2a_eighth_pass_2026_10_01): the port block's store scan. What it reads, on the same dense code:
# - every store into a member by name (`x.y =`, `x.y += 1`, `x.y++`) is one of `c.why = ...`, `c.verdict =
#   "UNCHECKABLE"` and `g.verdict = (...) ? "FAIL" : "PASS"`, inside _p2aAbstain; no `delete`;
# - every store into a computed member (`x[i] = v`, `x[i]++`) and every call of a mutating method (`push`, `pop`,
#   `splice`, `shift`, `unshift`, `set`, `add`, `fill`, `delete`, `clear`, `reverse`, `copyWithin`) is on a chain that
#   starts at a name the block made itself and then reads only `[...]` and `.get(...)` (no `.claims`, `.detail` or other
#   field): a name every binding of which, in its top-level function (or at the block's top level), is an array literal,
#   `new Map(...)`, `new Set(...)`, `_p2aFlags(...)`, or such a chain from such a name, that is no parameter there, and
#   none of whose bindings mentions `.claims` or `.detail`;
# - no value stored that way holds `.claims` or a detail itself (`push(c.detail)`), nor `g` in _p2aAbstain.
# What it does not do: follow a value through a parameter, or type an expression. The Python block's self-check is the
# precise one (_p2a_holder); this scan keeps the port's stores to the shapes the Python block's check allows.
JS_MUTATORS = ("push", "pop", "splice", "shift", "unshift", "set", "add", "fill", "delete", "clear", "reverse",
               "copyWithin")
JS_STORE = r"(?:=(?![=>])|[-+*/%&|^]=|\*\*=|<<=|>>>?=|&&=|\|\|=|\?\?=|\+\+|--)"
JS_NAME = r"[A-Za-z_$][\w$]*"
JS_HOLDS = re.compile(r"\.(?:claims|detail)(?![\w$.\[])")


def _js_close(dense: str, i: int) -> int:
    """The index just past the bracket that closes the one at dense[i]."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    depth, j = 0, i
    while True:
        if dense[j] in pairs:
            depth += 1
        elif dense[j] in ")]}":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1


def _js_open(dense: str, j: int) -> int:
    """The index of the bracket that opens the one closing just before j."""
    depth, i = 0, j - 1
    while True:
        if dense[i] in ")]}":
            depth += 1
        elif dense[i] in "([{":
            depth -= 1
            if depth == 0:
                return i
        i -= 1


def _js_chain_start(dense: str, k: int) -> int:
    """Where the member chain that ends just before k (at a '.' or '[') starts, read backward: a name continues it
    only through a '.' before it, and a bracketed group only where a '.', '[' or '(' follows it."""
    j = k
    while j > 0:
        if re.match(r"[\w$]", dense[j - 1]):
            while j > 0 and re.match(r"[\w$]", dense[j - 1]):
                j -= 1
            if j > 0 and dense[j - 1] == ".":
                j -= 1
                continue
            break
        if dense[j - 1] in ")]" and dense[j:j + 1] in (".", "[", "("):
            j = _js_open(dense, j)
            continue
        break
    return j


def _js_scopes(dense: str) -> list:
    """(start, end) of each top-level function of the block: `function NAME(...) {...}` and `const NAME = ...;` whose
    value holds a function (`=>`)."""
    out, i, depth = [], 0, 0
    while i < len(dense):
        if depth == 0 and (i == 0 or not re.match(r"[\w$]", dense[i - 1])):
            m = re.match(r"function " + JS_NAME + r"\(", dense[i:])
            if m:
                p = _js_close(dense, i + m.end() - 1)
                e = _js_close(dense, p)
                out.append((i, e))
                i = e
                continue
            m = re.match(r"const " + JS_NAME + "=", dense[i:])
            if m:
                e = i + m.end()
                d = 0
                while e < len(dense) and not (d == 0 and dense[e] == ";"):
                    d += {"(": 1, "[": 1, "{": 1, ")": -1, "]": -1, "}": -1}.get(dense[e], 0)
                    e += 1
                if "=>" in dense[i:e]:
                    out.append((i, e + 1))
                    i = e + 1
                    continue
        depth += {"(": 1, "[": 1, "{": 1, ")": -1, "]": -1, "}": -1}.get(dense[i], 0)
        i += 1
    return out


def _js_expr_end(dense: str, i: int) -> int:
    """Where the expression starting at i ends: a ',', ';' or closing bracket at its own depth."""
    d = 0
    while i < len(dense):
        ch = dense[i]
        if d == 0 and (ch in ",;" or ch in ")]}"):
            return i
        d += {"(": 1, "[": 1, "{": 1, ")": -1, "]": -1, "}": -1}.get(ch, 0)
        i += 1
    return i


def _js_bindings(text: str) -> tuple:
    """({name: [init text or None for a binding that is not a plain one]}, {parameter names}) of one scope's text."""
    binds: dict = {}
    params: set = set()
    for m in re.finditer(r"(?<![\w$])(?:const|let|var)(?= |\[|\{)", text):
        i = m.end()
        while True:
            if text[i] == " ":
                i += 1
            if text[i] in "[{":
                j = _js_close(text, i)
                for n in re.findall(JS_NAME, text[i:j]):
                    binds.setdefault(n, []).append(None)
            else:
                n = re.match(JS_NAME, text[i:]).group()
                j = i + len(n)
                if text[j:j + 1] != "=":
                    binds.setdefault(n, []).append(None)       # for (const x of y), let x;
            if text[j:j + 1] == "=":
                e = _js_expr_end(text, j + 1)
                if text[i] not in "[{":
                    binds.setdefault(n, []).append(text[j + 1:e])
                j = e
            if text[j:j + 1] != ",":
                break
            i = j + 1
    for m in re.finditer(r"(?<![\w$.])(" + JS_NAME + r")=(?![=>])", text):
        if not re.search(r"(?<![\w$])(?:const|let|var) $", text[:m.start()]):
            binds.setdefault(m.group(1), []).append(text[m.end():_js_expr_end(text, m.end())])
    for m in re.finditer(r"function " + JS_NAME + r"\(", text):
        params.update(re.findall(JS_NAME, text[m.end():_js_close(text, m.end() - 1) - 1]))
    for m in re.finditer(r"\)=>", text):
        params.update(re.findall(JS_NAME, text[_js_open(text, m.start() + 1) + 1:m.start()]))
    for m in re.finditer(r"(?<![\w$.])(" + JS_NAME + r")=>", text):
        params.add(m.group(1))
    return binds, params


def js_abstain_problems(dense: str, abstain: tuple, strings: list) -> list:
    """I-1 (NOTE_path2a_ninth_pass_2026_10_04): the claims _p2aAbstain may write, as the Python block's
    _p2a_shape_problems reads them, on the same dense code. `todo` is bound once, as `g.claims.filter(c =>
    _P2A_REACH.has(c.kind + "|" + c.verdict))` (a further `&&` condition may narrow it), and is read only as
    `!todo.length`, `todo.filter(` and `todo.map(`; `hits` is declared once, bound only as `todo.map(c => [c, ...])`,
    and read only by the one `for (const [c, hit] of hits)`; the stores into `c.why` and `c.verdict` lie inside that
    loop, whose body binds no other `c`; and `_P2A_REACH` is built once from `_P2A_REACH_PAIRS` and read only through
    `.has(`. Before this an arrow function's own `c` could carry the two stores to any claim of `g.claims`."""
    out = []
    fn = dense[abstain[0]:abstain[1]]

    def word(n):
        return r"(?<![\w$.])" + re.escape(n) + r"(?![\w$])"

    def count(rx, text=fn):
        return len(re.findall(rx, text))

    m = re.search(r"(?<![\w$])const todo=", fn)
    init = fn[m.end():_js_expr_end(fn, m.end())] if m else ""
    t = re.fullmatch(r"g\.claims\.filter\(c=>_P2A_REACH\.has\(c\.kind\+S(\d+)\+c\.verdict\)(?:&&[^|?]*)?\)", init)
    if not (t and js_string(strings[int(t.group(1))]) == "|" and count(word("todo") + r"=(?![=>])") == 1):
        out.append("todo is not bound once, as the claims of g.claims in reach")
    if count(word("todo")) != 1 + count(r"!todo\.length(?![\w$])") + count(word("todo") + r"\.(?:filter|map)\("):
        out.append("todo is read other than as !todo.length, todo.filter( or todo.map(")
    binds = 0
    for m in re.finditer(word("hits") + r"=(?![=>])", fn):
        binds += 1
        init = fn[m.end():_js_expr_end(fn, m.end())]
        arg = init[len("todo.map("):-1]
        if not (init.startswith("todo.map(") and _js_close(init, len("todo.map")) == len(init)
                and arg.startswith("c=>[c,") and _js_close(arg, 3) == len(arg)):
            out.append("hits is bound other than as todo.map(c => [c, ...])")
    loop = "for(const[c,hit]of hits){"
    if count(r"(?<![\w$])let hits;") != 1 or fn.count(loop) != 1 or binds < 1:
        return out + ["hits is not declared once, bound, and read by exactly one for (const [c, hit] of hits)"]
    if count(word("hits")) != 2 + binds:
        out.append("hits is read other than by the abstain loop")
    s = fn.index(loop) + len(loop) - 1
    e = _js_close(fn, s)
    for m in re.finditer(r"(?<![\w$.])c\.(?:why|verdict)" + JS_STORE, fn):
        if not s < m.start() < e:
            out.append("a store into c.why or c.verdict outside the abstain loop")
    body_binds, body_params = _js_bindings(fn[s:e])
    if "c" in body_binds or "c" in body_params:
        out.append("the abstain loop's body binds c again")
    if count(r"const _P2A_REACH=new Set\(_P2A_REACH_PAIRS\.map\(p=>p\[0\]\+S\d+\+p\[1\]\)\);", dense) != 1 \
            or count(word("_P2A_REACH"), dense) != 1 + count(word("_P2A_REACH") + r"\.has\(", dense) \
            or count(word("_P2A_REACH_PAIRS"), dense) != 2:
        out.append("_P2A_REACH is read other than through .has(, or built other than once from _P2A_REACH_PAIRS")
    return out


def js_store_problems(dense: str, strings: list) -> list:
    """The store scan (A-2 of NOTE_path2a_eighth_pass_2026_10_01); see the comment above JS_MUTATORS."""
    out = []
    scopes = _js_scopes(dense)
    top = dense
    for a, e in reversed(scopes):
        top = top[:a] + " " * (e - a) + top[e:]
    facts = {None: _js_bindings(top)}
    for a, e in scopes:
        facts[a] = _js_bindings(dense[a:e])
    abstain = next((a, e) for a, e in scopes if dense.startswith("function _p2aAbstain(", a))
    out += js_abstain_problems(dense, abstain, strings)

    def scope_of(k):
        return next((a for a, e in scopes if a <= k < e), None)

    def fresh(name, at, seen=()):
        """Every binding of `name` a value the block made; a chain back to a name already being read adds none."""
        if name in seen:
            return True
        for sc in (scope_of(at), None):
            binds, params = facts[sc]
            if name in binds or name in params:
                if name in params:
                    return False
                for init in binds[name]:
                    if init is None or JS_HOLDS.search(init) or ".claims" in init or ".detail" in init:
                        return False
                    if re.match(r"(?:\[|new Map\(|new Set\(|_p2aFlags\()", init):
                        continue
                    m = re.match(JS_NAME, init)
                    if not (m and steps_ok(init, m.end(), len(init)) and fresh(m.group(), at, seen + (name,))):
                        return False
                return True
        return False

    def steps_ok(text, i, end):
        while i < end:
            if text[i] == "[":
                i = _js_close(text, i)
            elif text.startswith(".get(", i):
                i = _js_close(text, i + 4)
            else:
                return False
        return True

    def string_is(tok, value):
        m = re.fullmatch(r"S(\d+)", tok)
        return m is not None and js_string(strings[int(m.group(1))]) == value

    # stores into a member by name
    for m in re.finditer(r"\.(" + JS_NAME + r")(" + JS_STORE + r")", dense):
        start = _js_chain_start(dense, m.start())
        target, op = dense[start:m.start() + 1 + len(m.group(1))], m.group(2)
        rhs = dense[m.end():_js_expr_end(dense, m.end())]
        inside = abstain[0] <= m.start() < abstain[1]
        ok = inside and op == "=" and (
            target == "c.why"
            or (target == "c.verdict" and string_is(rhs, "UNCHECKABLE"))
            or (target == "g.verdict" and re.fullmatch(r"\(.*\)\?(S\d+):(S\d+)", rhs) is not None
                and {js_string(strings[int(x[1:])]) for x in re.fullmatch(r"\(.*\)\?(S\d+):(S\d+)", rhs).groups()}
                == {"FAIL", "PASS"}))
        if not ok:
            out.append(f"store into a member: {target}{op}")
    for m in re.finditer(r"(?:\+\+|--)(" + JS_NAME + r"(?:\.[\w$]+|\[[^\]]*\])*\.[\w$]+)(?![\w$.(\[])", dense):
        out.append(f"store into a member: {m.group()}")
    if re.search(r"(?<![\w$.])delete(?![\w$])", dense):
        out.append("a delete")
    # stores into a computed member, and mutating calls
    sites = []
    for m in re.finditer(r"\](" + JS_STORE + r")", dense):
        i = _js_open(dense, m.start() + 1)
        before = re.search(JS_NAME + "$", dense[:i])
        if not (before and before.group() not in JS_KEYWORDS) and dense[i - 1:i] not in ("]", ")"):
            continue                              # an array literal or a destructuring pattern, not a member
        start = _js_chain_start(dense, i)
        sites.append(("store", start, i, m.start() + 1, dense[m.end():_js_expr_end(dense, m.end())]))
    for m in re.finditer(r"(?:\+\+|--)(" + JS_NAME + r")\[", dense):
        sites.append(("store", m.start() + 2, m.end() - 1, _js_close(dense, m.end() - 1), ""))
    for m in re.finditer(r"\.(" + "|".join(JS_MUTATORS) + r")\(", dense):
        start = _js_chain_start(dense, m.start())
        args = dense[m.end():_js_close(dense, m.end() - 1) - 1]
        sites.append(("call ." + m.group(1), start, m.start(), m.start(), args))
    for what, start, root_end, end, value in sites:
        root = re.match(JS_NAME, dense[start:])
        chain = dense[start:end]
        if root is None:
            out.append(f"{what} on {chain[:40]}, which does not start at a name")
            continue
        if what == "store" and start + root.end() != root_end:
            out.append(f"store into {chain[:40]}, a member of a member")
            continue
        if what != "store" and not steps_ok(dense, start + root.end(), end):
            out.append(f"{what} on {chain[:40]}, which reads a field")
            continue
        if not fresh(root.group(), start):
            out.append(f"{what} on {chain[:40]}, whose name the block did not make")
            continue
        if JS_HOLDS.search(value) or (abstain[0] <= start < abstain[1]
                                      and re.search(r"(?<![\w$.])g(?![\w$])", value)):
            out.append(f"{what} on {chain[:40]} stores a record field")
    return out


def test_the_port_block_passes_its_token_scan():
    assert js_problems(R.js_block()) == []


@pytest.mark.parametrize("old,new", [
    ("const o = ch.codePointAt(0);\n    out.push(", "const o = ch.toLowerCase().codePointAt(0);\n    out.push("),
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
    # C-2 (NOTE_path2a_fifth_pass_2026_09_30): the fifth review's plants, each of which the pass-4 scan accepted
    ('  const ca = _p2aA(claimed.split(new RegExp("(?i:k)")).join("k")), ck = _p2aK(claimed);', "regex modifiers (?i"),
    ('  const ca = _p2aA(claimed), ck = _p2aK(claimed); const q = parseInt(claimed, 10);', "banned word 'parseInt'"),
    ('  const ca = _p2aA(claimed), ck = _p2aK(claimed); const q = Number(claimed);', "banned word 'Number'"),
    ('  const pf = parseFloat; const ca = _p2aA(claimed), ck = _p2aK(claimed);', "banned word 'parseFloat'"),
    ('  const ca = _p2aA(claimed), ck = _p2aK(claimed); const q = isNaN(claimed);', "banned word 'isNaN'"),
    # C-2 (NOTE_path2a_sixth_pass_2026_09_30): the sixth review's plants, each accepted by the pass-5 scan
    (CA.rstrip(";") + "; const q = +claimed;", "unary + before"),
    (CA.rstrip(";") + "; const q = -claimed;", "unary - before"),
    (CA.rstrip(";") + "; const q = (+claimed);", "unary + before"),
    (CA.rstrip(";") + "; const q = claimed == 3;", "loose equality"),
    (CA.rstrip(";") + "; const q = claimed != 3;", "loose equality"),
    (CA.rstrip(";") + "; const q = Math.max(claimed, 0);", "banned word 'Math'"),
    (CA.rstrip(";") + "; const q = new Uint8Array([claimed]);", "banned word 'Uint8Array'"),
    (CA.rstrip(";") + "; const q = new Date(claimed);", "banned word 'Date'"),
])
def test_the_token_scan_refuses_the_reviews_plants(new, what):
    block = R.js_block()
    assert block.count(CA) == 1
    problems = js_problems(block.replace(CA, new))
    assert any(what in p for p in problems), problems


@pytest.mark.parametrize("op", ["claimed * 1", "claimed - 0", "claimed < 3"])
def test_the_token_scan_cannot_type_binary_operators(op):
    """C-2 (NOTE_path2a_sixth_pass_2026_09_30), the disclosed limit: a binary arithmetic or relational operator on a
    string runs StringToNumber too, and the scan cannot tell a string operand from a number. It accepts these; the
    block gives such operators numbers only. Pinned so the README's sentence stays true."""
    block = R.js_block()
    assert block.count(CA) == 1
    assert js_problems(block.replace(CA, CA.rstrip(";") + "; const q = " + op + ";")) == []


@pytest.mark.parametrize("old,new,what", [
    # C-2 (NOTE_path2a_fifth_pass_2026_09_30): the fifth review's plants in place
    ('new RegExp("\\r\\n|\\r|\\n")', 'new RegExp("(?i:\\r\\n|\\r|\\n)")', "regex modifiers (?i"),
    ("  return [_p2aInt(m[1]), _p2aInt(k[0])];", "  return [parseInt(m[1], 10), parseInt(k[0], 10)];",
     "banned word 'parseInt'"),
    ("  return [_p2aInt(m[1]), _p2aInt(k[0])];", "  return [Number(m[1]), Number(k[0])];", "banned word 'Number'"),
])
def test_the_token_scan_refuses_the_fifth_reviews_plants_in_place(old, new, what):
    block = R.js_block()
    assert block.count(old) == 1, old
    problems = js_problems(block.replace(old, new))
    assert any(what in p for p in problems), problems


JS_MOVED = '      c.verdict = "UNCHECKABLE";\n      moved = true;\n'


@pytest.mark.parametrize("plant,what", [
    # A-2 (NOTE_path2a_eighth_pass_2026_10_01): the seventh review's plant, which the pass-7 scan and the committed
    # --relation inputs both passed, and the other ways to write the record the store scan reads
    ('if (todo.length > 400) c.text = c.text + " ";', "store into a member: c.text="),
    ("g.claims.push(c);", "call .push on g.claims, which reads a field"),
    ('c.detail.n = "9";', "store into a member: c.detail.n="),
    ("delete c.detail.n;", "a delete"),
    ('const d = c.detail; d.n = "9";', "store into a member: d.n="),
    ("const d = c.detail; const k = 0; d[k] = 1;", "whose name the block did not make"),
    ("const a = [g.claims]; a[0].push(c);", "whose name the block did not make"),
    ("[g.claims][0].push(c);", "which does not start at a name"),
    ("Object.assign(c, {});", "banned word 'Object'"),
    ('c.why += "";', "store into a member: c.why+="),
    ("g.claims.length = 0;", "store into a member: g.claims.length="),
    ("const q = g.claims; q.splice(0);", "whose name the block did not make"),
    ("hits.pop();", "whose name the block did not make"),
    ('todo[0].verdict = "PASS";', "store into a member: todo[0].verdict="),
    ("c.detail.n++;", "store into a member: c.detail.n++"),
    ("++c.detail.n;", "store into a member: ++c.detail.n"),
    ("const out = []; out.push(c.detail);", "stores a record field"),
    ('c.verdict = "PASS";', "store into a member: c.verdict="),
    ('g.verdict = "PASS";', "store into a member: g.verdict="),
    ("const m2 = new Map(); m2.set(0, g.claims); m2.get(0).push(c);", "stores a record field"),
])
def test_the_store_scan_refuses_planted_stores(plant, what):
    block = R.js_block()
    assert block.count(JS_MOVED) == 1
    problems = js_problems(block.replace(JS_MOVED, JS_MOVED + "      " + plant + "\n"))
    assert any(what in p for p in problems), problems


JS_HITS = "    hits = todo.map(c => [c, _p2aDecide(c, f)]);\n"
JS_TODO = '  const todo = g.claims.filter(c => _P2A_REACH.has(c.kind + "|" + c.verdict));\n'
JS_BEFORE_LOOP = "  let moved = false;\n"


@pytest.mark.parametrize("old,new,what", [
    # I-1 (NOTE_path2a_ninth_pass_2026_10_04): the port's side of the eighth integration review's plant. The pass-8 scan
    # read `c.verdict = "UNCHECKABLE"` anywhere in _p2aAbstain as the overlay's own store, whatever `c` was
    (JS_MOVED, JS_MOVED + '      g.claims.forEach(c => { c.verdict = "UNCHECKABLE"; });\n',
     "the abstain loop's body binds c again"),
    (JS_BEFORE_LOOP, JS_BEFORE_LOOP + '  for (const c of g.claims) c.verdict = "UNCHECKABLE";\n',
     "a store into c.why or c.verdict outside the abstain loop"),
    (JS_HITS, JS_HITS + '    hits = hits.concat(g.claims.filter(c => c.kind === "tests_pass").map(c => [c, ["tests", '
     '"#101"]]));\n', "hits is bound other than as todo.map"),
    (JS_HITS, JS_HITS.replace(";\n", '.concat(g.claims.map(c => [c, ["tests", "#101"]]));\n'),
     "hits is bound other than as todo.map"),
    (JS_HITS, "    hits = g.claims.map(c => [c, _p2aDecide(c, f)]);\n", "hits is bound other than as todo.map"),
    (JS_HITS, "    hits = todo.map(c => [g.claims[0], _p2aDecide(c, f)]);\n", "hits is bound other than as todo.map"),
    (JS_HITS, JS_HITS + '    hits.push([g.claims[0], ["tests", "#101"]]);\n', "hits is read other than by the abstain loop"),
    (JS_TODO, JS_TODO.replace("));\n", ') || c.kind === "tests_pass");\n'), "todo is not bound once"),
    (JS_TODO, JS_TODO + "  todo.push(g.claims[0]);\n", "todo is read other than as"),
    (JS_TODO, '  _P2A_REACH.add("tests_pass|VERIFIED");\n' + JS_TODO, "_P2A_REACH is read other than through .has("),
])
def test_the_store_scan_refuses_a_write_outside_the_abstain_loop(old, new, what):
    block = R.js_block()
    assert block.count(old) == 1, old
    problems = js_problems(block.replace(old, new))
    assert any(what in p for p in problems), problems


D_TWINS_HEAD = ("diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-x = 0\n+x = 1\n"
                + "".join(f"diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n@@ -1 +1 @@\n-a\n+b\n" for p in (".env", "env")))


def test_a_widened_count_head_reads_no_digit_table(M, tmp_path):
    """C-2 (NOTE_path2a_fifth_pass_2026_09_30): the review widened the count head to `([^,]+)` and the port's parseInt
    read "\u30003" as 3. The digits are read through fixed tables in both ports now: a widened head meets a digit
    outside the table and raises, and the error fallback withholds (a reason main never writes, given by hand)."""
    def fresh():
        g = M.gate_diff_text("3 files changed.", D_TWINS_HEAD)
        assert [(c.kind, c.verdict, c.why) for c in g.claims] == [
            ("files_changed_count", "CONTRADICTED", "diff changes 2 files, claim says 3")]
        g.claims[0].why = "diff changes " + chr(0x3000) + "2 files, claim says 3"
        return g

    rec = fresh().to_dict()
    text = R.lf(R.INSTRUMENT)
    block = R.py_block(text)
    old = '_P2A_COUNT_HEAD = re.compile("diff changes ([0-9]+) files, claim says ")'
    assert block.count(old) == 1
    mod = R.module_from(text.replace(block, block.replace(old, old.replace("[0-9]+", "[^,]+"))), "_p2a_plant_head")
    for m, key in ((N, "unparsed"), (mod, "error")):
        x = fresh()
        m._p2a_abstain(x, False, lambda m=m: m._P2aFacts(D_TWINS_HEAD, None, "3 files changed."))
        assert x.claims[0].verdict == "UNCHECKABLE" and R.phrase_key(x.claims[0].why, PHRASES) == key, (key, x.claims[0].why)
    js = R.lf(R.PORT)
    jblock = R.js_block(js)
    jold = 'new RegExp("^diff changes ([0-9]+) files, claim says ")'
    assert jblock.count(jold) == 1
    planted = tmp_path / "diffgate_widened.js"
    planted.write_bytes(js.replace(jblock, jblock.replace(jold, jold.replace("[0-9]+", "[^,]+"))).encode("utf-8"))
    (tmp_path / "in.json").write_text(json.dumps([{"id": "w", "summary": "3 files changed.", "diff": D_TWINS_HEAD,
                                                   "gate": rec}], ensure_ascii=False), encoding="utf-8")
    for port, key in ((R.PORT, "unparsed"), (planted, "error")):
        node("--abstain", port, tmp_path / "in.json", tmp_path / "out.json")
        c = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))[0]["rec"]["claims"][0]
        assert c["verdict"] == "UNCHECKABLE" and R.phrase_key(c["why"], PHRASES) == key, (port, c["why"])


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


def test_the_shipped_bookmarklet_is_the_build_the_readme_names():
    """C-5 = I-4 (NOTE_path2a_fifth_pass_2026_09_30): no terser needed. The minified file hashes to the README's line, and
    the href is `javascript:` + that file, so whatever a browser holds either hashes to the line or is not this build."""
    import hashlib
    gate = R.ROOT / "web" / "gate"
    mini = (gate / "bookmarklet.min.js").read_bytes().decode("utf-8")
    href = (gate / "bookmarklet.href.txt").read_bytes().decode("utf-8")
    assert href == "javascript:" + mini
    readme = R.lf(gate / "README.md")
    m = re.search(r"^    bookmarklet\.min\.js    sha256 ([0-9a-f]{64})   ([0-9,]+) chars$", readme, re.M)
    h = re.search(r"^    bookmarklet\.href\.txt  sha256 ([0-9a-f]{64})   ([0-9,]+) chars$", readme, re.M)
    assert m and h, "the README no longer names the shipped build in the form this test reads"
    assert hashlib.sha256(mini.encode("utf-8")).hexdigest() == m.group(1) and len(mini) == int(m.group(2).replace(",", ""))
    assert hashlib.sha256(href.encode("utf-8")).hexdigest() == h.group(1) and len(href) == int(h.group(2).replace(",", ""))


def test_the_shipped_bookmarklet_runs_the_port(tmp_path):
    """C-5 = I-4: the minified bytes users install, loaded in a stub page as a browser runs them, give the port's record
    on every pinned pair, both strict modes, and the overlay's reasons appear. Under CI a missing node fails."""
    pairs = json.loads((R.DIFFERENTIAL / "path2a_pairs.json").read_text(encoding="utf-8"))
    (tmp_path / "in.json").write_text(json.dumps([{"id": p["id"], "summary": p["summary"], "diff": p["diff"]}
                                                  for p in pairs], ensure_ascii=False), encoding="utf-8")
    node("--bookmarklet", R.ROOT / "web" / "gate" / "bookmarklet.min.js", tmp_path / "in.json", tmp_path / "out.json")
    rep_ = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert rep_["differ"] == [] and rep_["runs"] == 2 * len(pairs), rep_
    assert rep_["runs_with_overlay_reason"] > 100, rep_


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
    # I-3 (NOTE_path2a_sixth_pass_2026_09_30): an overlay reason with main's reading after its words shows the words
    # whole and main's reading cut at 100 characters, as main cuts a reason; 300 files with 4 KB paths passed GitHub's
    # 1 MiB step-summary limit on 495d2204.
    long_path = ("z" * 200 + "/") * 20 + "f.py"
    ours = N.DiffClaim(kind="file_touched", text="long", detail={}, verdict="UNCHECKABLE",
                       why=R.reason("VERIFIED", "#97", PHRASES["dir"], f"diff status 'M' for '{long_path}'"))
    monkeypatch.setattr(mod, "gate_diff_text", lambda *a, **k: N.DiffGate(
        verdict="PASS", base="(diff-text)", head="(diff-text)", claims=[ours], measured=True))
    (tmp_path / "sum.md").write_text("", encoding="utf-8")
    mod.main()
    table = (tmp_path / "sum.md").read_text(encoding="utf-8")
    row = next(line for line in table.splitlines() if line.startswith("| ") and " | long | " in line)
    shown_why = row[:-2].rsplit(" | ", 1)[1]
    head = "VERIFIED withheld by PATH-2a (#97): " + PHRASES["dir"] + ". main's reading: "
    assert shown_why == head + f"diff status 'M' for '{long_path}'"[:100], shown_why
    # I-3 (NOTE_path2a_fifth_pass_2026_09_30): the kind condition. A claim of a kind the overlay never moves, whose
    # reason starts with the overlay's form, is cut at 100 as main cuts it.
    other = N.DiffClaim(kind="declaration_problem", text="other", detail={}, verdict="UNCHECKABLE", why=form + "D" * 300)
    monkeypatch.setattr(mod, "gate_diff_text", lambda *a, **k: N.DiffGate(
        verdict="PASS", base="(diff-text)", head="(diff-text)", claims=[other], measured=True))
    (tmp_path / "sum.md").write_text("", encoding="utf-8")
    mod.main()
    table = (tmp_path / "sum.md").read_text(encoding="utf-8")
    row = next(line for line in table.splitlines() if line.startswith("| ") and " | other | " in line)
    assert len(row[:-2].rsplit(" | ", 1)[1]) == 100, row


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

    # NOTE_path2a_fifth_pass_2026_09_30, A-1: deep paths. 5ebe0b6b stored every suffix of every key: 0.50 s and 766 MB
    # for one 40 KB path, 0.69 s and 799 MB for 100 paths of 4 KB (the overlay alone, CPython 3.12.10).
    def deep(depth, k, summary):
        paths = ["a/" * depth + f"f{i}.py" for i in range(k)] + ["README.md"]
        return {"id": f"deep-{depth}-{k}", "summary": summary,
                "diff": "".join(f"--- a/{p}\n+++ b/{p}\n@@ -1 +1 @@\n-x\n+y\n" for p in paths)}

    # A-2: claims against a wide base-name group. On 5ebe0b6b, 6.1 s (4,369 identical claims over 3,000 directories
    # outside ASCII), 19.3 s (3,000 distinct claims whose base name is outside ASCII) and 0.22 s.
    def wide(k, claim, path):
        return {"id": f"wide-{claim(0)}-{k}", "summary": " ".join(claim(i) for i in range(k)),
                "diff": "".join(f"--- a/{path(j)}\n+++ b/{path(j)}\n@@ -1 +1 @@\n-x\n+y\n" for j in range(3000))}

    a = [deep(20000, 1, "Modified f0.py. Modified README.md. Created README.md."),
         deep(2000, 100, "Modified f0.py. Modified README.md. Created README.md."),
         wide(4369, lambda i: "Modified x.py.", lambda j: f"{e}{j}/x.py"),
         wide(3000, lambda i: f"Modified y{i}/d{e}.py.", lambda j: f"x{j}/d{e}.py"),
         wide(3000, lambda i: f"Modified q{i}{e}/x.py.", lambda j: f"{e}{j}/x.py"),
         wide(200, lambda i: f"Modified __init__.py for p{i}.", lambda j: f"pkgs/p{j}{e}/__init__.py"),
         wide(4000, lambda i: f"Modified e{i}/x.py.", lambda j: f"d{j}/x.py"),
         {"id": "dots-40000", "summary": "Modified f0.py. 5 files changed. Only touches f0.py.",
          "diff": "--- a/" + "./" * 40000 + "f0.py\n+++ b/" + "./" * 40000 + "f0.py\n@@ -1 +1 @@\n-x\n+y\n"}]
    return [wide_def(5000, e), wide_def(50000, e), wide_def(50000, zh), files(2000, 200), same_base(2000, 200),
            symbols(500, 50000), class_nbsp(300, 20000), long_line] + q + a


# The overlay's own time, the least of three runs on main's record for the same input (NOTE_path2a_third_pass_2026_09_30,
# I-2: pass 2 subtracted two single wall-clock samples of whole calls). The shapes: a `def` beside a 5,000- or
# 50,000-code-point run, one 220 KB added line of `def test_a`, 2,000 files with 200 path claims under 50 directories and
# under one base name, 500 symbol claims over 50,000 removed `def` lines, 300 class claims beside NBSP runs, and 64 KB
# summaries whose path, symbol, scope and count claims each meet thousands of runs or zones. Bound: 0.5 s in Python,
# 0.3 s in the port. On 40bba05b the overlay alone took 0.80 s on the one-base-name case, 38.6 s on the symbols and
# 16.8 s on the NBSP classes in Python (CPython 3.12.10); on ea677740, 3.8 s on q1. The figures at this head are in
# NOTE_path2a_fourth_pass_2026_09_30 and the README, measured the way this test measures them.
OVERLAY_S = (0.5, 0.3)
# I-1 (NOTE_path2a_eighth_pass_2026_10_01): the bound is relative to main's own call on the same input too, so a slower
# runner (CI's CPython 3.9 and 3.10, a shared Linux machine) moves both sides: the overlay alone within the larger of the
# absolute figure above and OVERLAY_TIMES times main's call (the least of three runs each).
OVERLAY_TIMES = 5
# I-8 (NOTE_path2a_ninth_pass_2026_10_04): on the case closest to its limit (500 symbol claims over 50,000 removed `def`
# lines) five times main's call is below the absolute figure, so that case is bounded by wall-clock time alone, and the
# overlay's loops are Python where main's are C regex: an interpreter without the specialising interpreter (CPython
# below 3.11) slows the overlay more than main. There the absolute figure is 1.0 s. Not measured on 3.9 or 3.10, which
# are not on the machine this was written on; on 3.12.10 and 3.14.2 the slowest case sits at about a third of 0.5 s.
OVERLAY_S_PY = OVERLAY_S[0] if sys.version_info >= (3, 11) else 1.0


def test_cost_per_call_python(M):
    """The overlay alone on each timing case, within max(0.5 s, 5 times main's call), 1.0 s below CPython 3.11 (I-8 of
    the ninth pass). The README gives the figures measured this way at this head."""
    for it in _timing_cases():
        best = whole = float("inf")
        for _ in range(3):
            t = time.perf_counter()
            g = M.gate_diff_text(it["summary"], it["diff"])
            t0 = time.perf_counter()
            N._p2a_abstain(g, False, lambda: N._P2aFacts(it["diff"], None, it["summary"]))
            best, whole = min(best, time.perf_counter() - t0), min(whole, t0 - t)
            assert not any(PHRASES["error"] in c.why for c in g.claims), it["id"]   # a failing overlay is fast too
        assert best < max(OVERLAY_S_PY, OVERLAY_TIMES * whole), (it["id"], best, whole)


OVERLAY_MB = 64


def test_peak_memory_python(M):
    """A-1 (NOTE_path2a_fifth_pass_2026_09_30): the overlay's peak memory on every timing case, by tracemalloc. On
    5ebe0b6b the deep-path cases peaked at 766 MB and 799 MB (every suffix of every key), and under a 1,024 MB cap the
    overlay's own MemoryError withheld a right VERIFIED with the error phrase."""
    import tracemalloc
    for it in _timing_cases():
        g = M.gate_diff_text(it["summary"], it["diff"])
        tracemalloc.start()
        try:
            N._p2a_abstain(g, False, lambda: N._P2aFacts(it["diff"], None, it["summary"]))
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert not any(PHRASES["error"] in c.why for c in g.claims), it["id"]
        assert peak < OVERLAY_MB * 2 ** 20, (it["id"], peak)


def test_cost_per_call_port(work, tmp_path):
    (tmp_path / "in.json").write_text(json.dumps(_timing_cases(), ensure_ascii=False), encoding="utf-8")
    node("--overlay-timing", work / "diffgate_main_reference.js", tmp_path / "in.json", tmp_path / "t.json")
    for d in json.loads((tmp_path / "t.json").read_text(encoding="utf-8")):
        assert d["overlay"] < max(1000 * OVERLAY_S[1], OVERLAY_TIMES * d["main"]), d


def _large_cases(scale: int = 1) -> list[dict]:
    """A-1 (NOTE_path2a_sixth_pass_2026_09_30): summaries of about 1.2 MB whose 10,000 distinct count, path and scope
    claims each look a token up in the summary's runs or zones, over a one-file diff (the sixth review's S1, S2 and S3,
    with scopes main verifies). 495d2204 scanned the summary once per token: the overlay alone
    took about 1.0, 3.0 and 4.5 s in Python, where main's whole call takes 0.7 to 0.8 s. The port reads them at three
    times the size, where 495d2204's cost grows further past main's."""
    wide = chr(0xAD) * (1_000_000 * scale)
    n = 10_000 * scale

    def files(ps):
        return "".join(f"diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n@@ -0,0 +1 @@\n+x\n" for p in ps)

    return [
        {"id": "large-counts", "summary": wide + ". " + " ".join(f"{i} files changed." for i in range(n)),
         "diff": files([".a.py", "a.py"])},
        {"id": "large-paths", "summary": wide + ". " + " ".join(f"Modified d{i}/x.py." for i in range(n)),
         "diff": files(["x.py"])},
        {"id": "large-scopes",
         "summary": "only " + wide + ". " + " ".join(f"Only touches s{i}/ and x.py." for i in range(n)),
         "diff": files(["x.py"])},
    ]


def test_cost_on_large_summaries_python(M):
    """The overlay alone, on the large cases, within 1.5 times main's own call on the same input (the least of two runs
    each): a bound relative to main, so a slower runner moves both sides. The overlay's loops are Python where main's
    are mostly C regex, so a slower interpreter moves the overlay more (NOTE_path2a_seventh_pass_2026_09_30, I-1):
    measured at the eighth pass's head 0.21 to 0.47 of main's call on CPython 3.12.10 and 0.19 to 0.32 on 3.14.2
    (Windows, one run each; such ratios move by about a tenth between runs: NOTE_path2a_eighth_pass_2026_10_01, I-4),
    0.35 to 0.65 on 3.12.3 (Linux, the sixth review, at its head); 1.5 leaves room for CI's 3.9 to 3.11, which were not
    available here. 495d2204 took 2.5 to 7.5 times main's call."""
    for it in _large_cases():
        over = whole = float("inf")
        for _ in range(2):
            t = time.perf_counter()
            g = M.gate_diff_text(it["summary"], it["diff"])
            t0 = time.perf_counter()
            N._p2a_abstain(g, False, lambda: N._P2aFacts(it["diff"], None, it["summary"]))
            over, whole = min(over, time.perf_counter() - t0), min(whole, t0 - t)
            assert not any(PHRASES["error"] in c.why for c in g.claims), it["id"]
        assert over < 1.5 * whole, (it["id"], over, whole)


def _large_relation_python(mod, M) -> list:
    bad = []
    for it in _large_cases():
        recs = {}
        for strict in (False, True):
            a = M.gate_diff_text(it["summary"], it["diff"], strict=strict).to_dict()
            b = mod.gate_diff_text(it["summary"], it["diff"], strict=strict).to_dict()
            bad += [(it["id"], strict, x) for x in R.relation(a, b, strict, PHRASES)]
            recs[strict] = b
        bad += [(it["id"], "strict", x) for x in R.strict_alike(recs[False], recs[True])]
    return bad


def test_the_relation_on_large_summaries_python(M):
    """A-1 (NOTE_path2a_eighth_pass_2026_10_01): the large cases push 10,000 claims each through the overlay, and the
    timing tests never compared their records with main's: the seventh review planted `g.claims.pop()` behind
    `len(todo) > 400` and every committed test passed. The relation, both strict modes, on them; and that plant (and
    one that rewrites claim text) is refused here as well as by the self-check."""
    assert _large_relation_python(N, M) == []
    text = R.lf(R.INSTRUMENT)
    block = R.py_block(text)
    anchor = '            c.verdict = "UNCHECKABLE"\n            moved = True\n'
    assert block.count(anchor) == 1
    for name, plant in (("drop", "            if len(todo) > 400:\n                g.claims.pop()\n"),):
        mod = R.module_from(text.replace(block, block.replace(anchor, anchor + plant)), "_p2a_plant_large_" + name)
        assert _large_relation_python(mod, M) != [], name


def test_the_relation_on_large_summaries_port(work, tmp_path):
    """The same in the port, at three times the size (30,000 claims a case), with the seventh review's port plant (each
    claim's text gains a trailing space behind `todo.length > 400`), which --relation on the committed inputs passed."""
    (tmp_path / "in.json").write_text(json.dumps(_large_cases(3), ensure_ascii=False), encoding="utf-8")
    node("--relation", work / "diffgate_main_reference.js", tmp_path / "in.json", tmp_path / "rel.json")
    rep = json.loads((tmp_path / "rel.json").read_text(encoding="utf-8"))
    assert rep["broken"] == [] and rep["counts"]["broken"] == 0 and rep["counts"]["runs"] == 6, rep["counts"]
    text = R.lf(R.PORT)
    block = R.js_block(text)
    anchor = '      c.verdict = "UNCHECKABLE";\n      moved = true;\n'
    assert block.count(anchor) == 1
    planted = tmp_path / "diffgate_planted.js"
    planted.write_bytes(text.replace(block, block.replace(
        anchor, anchor + '      if (todo.length > 400) c.text = c.text + " ";\n')).encode("utf-8"))
    node("--relation", work / "diffgate_main_reference.js", tmp_path / "in.json", tmp_path / "rel2.json", planted)
    assert json.loads((tmp_path / "rel2.json").read_text(encoding="utf-8"))["counts"]["broken"] > 0


def test_cost_on_large_summaries_port(work, tmp_path):
    """The same in the port, at three times the size, within three times main's own call: the port's main is about
    seven times faster than CPython's, so the overlay's fixed per-claim work weighs more beside it, and a shared runner's
    noise more. Measured: 0.8 to 1.4 times main's call here; 495d2204's overlay took 8.6 to 9.6 times."""
    (tmp_path / "in.json").write_text(json.dumps(_large_cases(3), ensure_ascii=False), encoding="utf-8")
    node("--overlay-timing", work / "diffgate_main_reference.js", tmp_path / "in.json", tmp_path / "t.json")
    for d in json.loads((tmp_path / "t.json").read_text(encoding="utf-8")):
        assert d["overlay"] < 3 * d["main"], d


def test_found_reads_what_one_scan_per_word_reads(tmp_path):
    """A-1 (NOTE_path2a_sixth_pass_2026_09_30): `_p2a_found` reads more than _P2A_MANY words through one automaton; it
    must give the set one scan per word gives, in both ports, so no decision moves with the number of claims. Pass 7
    (NOTE_path2a_seventh_pass_2026_09_30, A-1): it reads the text as its pieces between characters a named word never
    holds (_P2A_SEP and those of _P2A_BAD_RX), and the words through automata of a bounded size, so the words here are
    the ones `tokens` names (no such character), the texts hold those characters, and some cases hold more than one
    automaton's worth of words."""
    import random
    rng = random.Random(6)
    words_abc = "ab./-0_" + chr(0x3000) + chr(0xB7)
    text_abc = words_abc + chr(0xE9) + "\x00" + chr(0x1F600) + chr(0x85) + chr(0xFEFF)
    cases = []
    for _ in range(300):
        text = "".join(rng.choice(text_abc) for _ in range(rng.randint(0, 400)))
        words = sorted({"".join(rng.choice(words_abc) for _ in range(rng.randint(1, 6)))
                        for _ in range(rng.randint(1, 90))})
        cases.append([words, text])
    for k in range(4):                    # more than _P2A_BUDGET characters of words: several automata
        tails = ["".join(rng.choice("ab") for _ in range(300)) for _ in range(400)]
        words = sorted({str(i) + t for i, t in enumerate(tails)})
        text = chr(0xE9).join(("x" + w if i % 3 else w[:-1]) for i, w in enumerate(words[: 300 + 30 * k]))
        cases.append([words, text])
    assert sum(len(w) for w in cases[-1][0]) > N._P2A_BUDGET
    assert all(N._P2A_BAD_RX.search(w) is None and "\x00" not in w for words, _t in cases for w in words)
    want = [sorted(w for w in words if w in text) for words, text in cases]
    assert [sorted(N._p2a_found(frozenset(words), text)) for words, text in cases] == want
    (tmp_path / "in.json").write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    node("--found", tmp_path / "in.json", tmp_path / "out.json")
    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8")) == want


def _long_token_cases(n: int = 3000, w: int = 110) -> list[dict]:
    """A-1 (NOTE_path2a_seventh_pass_2026_09_30): the seventh review's memory shape, distinct long tokens. n count claims,
    each a distinct w-digit number, over a diff with a dot twin, so each reaches the count's extract guard. In
    `long-counts-empty-runs` no run of the summary holds a character the two ports read apart, so the guard's text is
    empty; in `long-counts-in-runs` each number also sits, after a 9, in a run that holds U+00AD, a proper part of a
    piece of the guard's text. fcd3ce6a built one automaton over every number either way: 68.5 MB at its peak in
    Python on both, and an abort under a 64 MB heap in Node."""
    soft = chr(0xAD)
    tok = [(str(i + 1) + "7" * w)[:w] for i in range(n)]
    diff = "".join(f"diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n@@ -0,0 +1 @@\n+x\n" for p in (".a.py", "a.py"))
    claims = " ".join(f"{t} files changed." for t in tok)
    return [{"id": "long-counts-empty-runs", "summary": claims, "diff": diff},
            {"id": "long-counts-in-runs", "summary": claims + " " + " ".join(f"9{t}{soft}" for t in tok), "diff": diff}]


OVERLAY_LONG_MB = 32


def test_peak_memory_on_long_distinct_tokens_python(M):
    """A-1 (NOTE_path2a_seventh_pass_2026_09_30): the overlay's peak memory on distinct long tokens, by tracemalloc,
    within 32 MB: an empty text builds no automaton, and one automaton holds at most _P2A_BUDGET characters of words
    (or a sixteenth of the text). Measured here: 0.4 and 16.5 MB, where main's own call peaks at 2.6 and 2.8 MB;
    fcd3ce6a took 68.1 and 68.5 MB. Its time stays within three times main's call."""
    import tracemalloc
    for it in _long_token_cases():
        t = time.perf_counter()
        g = M.gate_diff_text(it["summary"], it["diff"])
        whole = time.perf_counter() - t
        t0 = time.perf_counter()
        N._p2a_abstain(g, False, lambda: N._P2aFacts(it["diff"], None, it["summary"]))
        over = time.perf_counter() - t0
        assert not any(PHRASES["error"] in c.why for c in g.claims), it["id"]
        assert over < 3 * whole, (it["id"], over, whole)
        g = M.gate_diff_text(it["summary"], it["diff"])
        tracemalloc.start()
        try:
            N._p2a_abstain(g, False, lambda: N._P2aFacts(it["diff"], None, it["summary"]))
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < OVERLAY_LONG_MB * 2 ** 20, (it["id"], peak)


def test_peak_memory_on_long_distinct_tokens_port(work, tmp_path):
    """The same in the port, under a 64 MB heap, where main's port needs less than 16 MB: the long counts at twice the
    count, and the seventh review's own input, 10,000 path claims of 60-character directories over a one-file diff
    (an empty text). fcd3ce6a's port aborted on each (heap out of memory); this one completes under 32 MB and 24 MB.
    A flat string per key form keeps the forms the overlay stores small (A-1 of the seventh pass)."""
    if NODE is None:
        no_node()
    big = _long_token_cases(6000) + [{
        "id": "long-paths-empty-runs",
        "summary": " ".join(f"Modified d{i}/{'q' * 60}/f.py." for i in range(10000)),
        "diff": "diff --git a/x/f.py b/x/f.py\n--- a/x/f.py\n+++ b/x/f.py\n@@ -1 +1 @@\n-a\n+b\n"}]
    (tmp_path / "in.json").write_text(json.dumps(big, ensure_ascii=True), encoding="utf-8")
    for port, tag in ((work / "diffgate_main_reference.js", "main"), (R.PORT, "new")):
        r = subprocess.run([NODE, "--max-old-space-size=64", str(CHECK), "--records", str(port), str(tmp_path / "in.json"),
                            str(tmp_path / f"{tag}.json")], capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, (tag, r.stderr[-1500:])
    recs = json.loads((tmp_path / "new.json").read_text(encoding="utf-8"))
    assert all("error" not in x["rec"] and not any(PHRASES["error"] in c["why"] for c in x["rec"]["claims"])
               for x in recs)
    assert [len(x["rec"]["claims"]) for x in recs] == [6000, 6000, 10000]


# ---- A-2 (NOTE_path2a_sixth_pass_2026_09_30): the options, read once -------------------------------------------------

def test_the_port_reads_its_options_once(work, tmp_path):
    """The port read `opts.strict` once for main and once more for the gate verdict: a getter that answers true, then
    false, gave main FAIL and the overlay PASS with the same claims on 495d2204. Now it reads `strict` and `_declared`
    once, in main's order, and main reads that snapshot: every odd `opts` gives main's record, and the reads main
    makes of a Proxy are the same."""
    node("--opts", work / "diffgate_main_reference.js", tmp_path / "opts.json")
    rows = json.loads((tmp_path / "opts.json").read_text(encoding="utf-8"))
    assert len(rows) >= 9 and all(r["same"] and r["reads_same"] for r in rows), [r for r in rows if not r["same"]
                                                                                 or not r["reads_same"]]


def test_no_claim_moved_leaves_mains_gate_verdict(M):
    """Where no claim moves, the record is main's object: the gate verdict is not computed again, so a `strict` whose
    truth changes between reads reads as main read it. On 495d2204 the overlay recomputed it whenever a claim was in
    reach (Python: main FAIL, the overlay PASS, the same claims)."""
    class Flip:
        def __init__(self):
            self.n = 0

        def __bool__(self):
            self.n += 1
            return self.n == 1

    s, d = "Modified x.py. All tests pass.", "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
    a = M.gate_diff_text(s, d, strict=Flip()).to_dict()
    b = N.gate_diff_text(s, d, strict=Flip()).to_dict()
    assert a["verdict"] == "FAIL" and b == a


# ---- I-1 (NOTE_path2a_ninth_pass_2026_10_04): the relation with a run leg, a test report and a commit ---------------

# a green JUnit report, the bytes tests/test_evidence.py and tests/test_diffgate_evidence.py pin
GREEN_REPORT = ('<?xml version="1.0" encoding="utf-8"?>\n<testsuites name="pytest tests"><testsuite name="pytest" '
                'errors="0" failures="0" skipped="0" tests="2" time="0.012">\n'
                '<testcase classname="tests.test_app" name="test_one" time="0.001" />\n'
                '<testcase classname="tests.test_app" name="test_two" time="0.001" />\n</testsuite></testsuites>\n')


class _RunStub:
    """Stands in for subprocess.run, so the run leg's command is never executed (as tests/test_diffgate_evidence.py's
    stub): `--run` passes shell=True with the tree under test as cwd."""

    def __init__(self, returncode=0):
        self.returncode, self.calls = returncode, 0

    def __call__(self, *a, **kw):
        self.calls += 1
        return types.SimpleNamespace(returncode=self.returncode, stdout="", stderr="")


def _tests_pass_relation(mod, M, green, monkeypatch, until_broken=False) -> tuple:
    """The relation and strict_alike for `mod` against main, both strict modes, over #161's reproductions with a
    sentence main reads as a `tests_pass` claim appended, with what only the Python doors take: a run leg that exits 0, a
    green report, a green report under a commit it does not name, and both legs; at the raw door, and at the git door for
    the cases that carry their own --name-status. Returns (what broke, what was seen)."""
    stub = _RunStub()
    monkeypatch.setattr(N.subprocess, "run", stub)
    configs = [("a run leg", {"run": "exit 0"}), ("a report", {"evidence": [green]}),
               ("a report under another commit", {"evidence": [green], "commit": "1" * 40}),
               ("both legs", {"run": "exit 0", "evidence": [green]})]
    broken, seen = [], collections.Counter()
    for c in R.repro_cases():
        summary = c["summary"] + " All tests pass."
        doors = [("raw", lambda m, kw, c=c, s=summary: m.gate_diff_text(s, c["diff"], repo=".", **kw))]
        if c.get("name_status"):
            fake = R.fake_git(c["name_status"], c["diff"])
            for m in (M, mod):
                monkeypatch.setattr(m, "_git", fake)
            doors.append(("git", lambda m, kw, s=summary: m.gate_diff(s, "(repo)", "base", "head", **kw)))
        for door, call in doors:
            for name, kw in configs:
                recs = {}
                for strict in (False, True):
                    try:
                        a = call(M, dict(kw, strict=strict)).to_dict()
                    except Exception:
                        break
                    b = call(mod, dict(kw, strict=strict)).to_dict()
                    bad = R.relation(a, b, strict, PHRASES)
                    if bad:
                        broken.append((c["id"], door, name, strict, bad))
                    recs[strict] = b
                    if not strict:
                        for x in a["claims"]:
                            if x["kind"] == "tests_pass":
                                seen[f"{door}, {name}: tests_pass {x['verdict']}"] += 1
                        seen[f"{door}: a claim withheld beside a tests_pass claim"] += any(
                            R.phrase_key(y["why"], PHRASES) for y in b["claims"])
                if len(recs) == 2 and R.strict_alike(recs[False], recs[True]):
                    broken.append((c["id"], door, name, "strict"))
                if broken and until_broken:
                    return broken, seen
    assert stub.calls > 0
    return broken, seen


def test_the_relation_holds_with_a_run_leg_and_a_test_report_at_both_doors(M, tmp_path, monkeypatch):
    """I-1 (NOTE_path2a_ninth_pass_2026_10_04). No committed test ran the relation with `run=`, `evidence=` or
    `commit=`, the only arguments under which main decides a `tests_pass` claim, a kind outside REACH: the eighth
    integration review planted an abstention on a `tests_pass` VERIFIED beside a count claim, and every test passed.
    Here main verifies `tests_pass` through the run leg and through a report at both doors, leaves it UNCHECKABLE where
    the report does not name the commit, and the branch's record is main's but for abstentions in reach; the review's
    plant is refused here as well as by the self-check."""
    green = tmp_path / "green.xml"
    green.write_bytes(GREEN_REPORT.encode("utf-8"))
    broken, seen = _tests_pass_relation(N, M, str(green), monkeypatch)
    print("the relation with a run leg and a report:", json.dumps(dict(sorted(seen.items())), indent=1))
    assert broken == [], broken[:5]
    for door, n, beside in (("raw", 400, 400), ("git", 80, 30)):
        for name in ("a run leg", "a report", "both legs"):
            assert seen[f"{door}, {name}: tests_pass VERIFIED"] >= n, (door, name, seen)
        assert seen[f"{door}, a report under another commit: tests_pass UNCHECKABLE"] >= n, (door, seen)
        assert seen[f"{door}: a claim withheld beside a tests_pass claim"] >= beside, (door, seen)
    text = R.lf(R.INSTRUMENT)
    block = R.py_block(text)
    assert block.count(HITS) == 1
    plant = HITS + ('        hits = hits + [(c, ("tests", "#101")) for c in g.claims if c.kind == "tests_pass" and '
                    'c.verdict == "VERIFIED"\n                       and any(x.kind == "files_changed_count" for x in g.claims)]\n')
    mod = R.module_from(text.replace(block, block.replace(HITS, plant)), "_p2a_plant_tests_pass")
    planted, _seen = _tests_pass_relation(mod, M, str(green), monkeypatch, until_broken=True)
    assert planted and "tests_pass VERIFIED -> UNCHECKABLE is not an abstention in reach" in str(planted[0][-1]), planted[:2]


# ---- (C) cross-port ---------------------------------------------------------------------------------------------------

def _seen(c):
    """A claim's verdict as the overlay left it, and its phrase key when the overlay wrote the reason."""
    return c["verdict"], R.phrase_key(c["why"], PHRASES)


def _kvd(c):
    return c["kind"], c["verdict"], json.dumps(c["detail"], sort_keys=True)


_READ = ("path", "name", "prefix", "prefix2", "n")
BY_CONSTRUCTION = ("position: kind, verdict, detail", "any: kind, verdict, detail", "matched: kind, verdict, detail",
                   "gate verdict where the two lists hold the same claims")


def _one_match(x, u):
    """Whether two claims with the same kind and verdict and different details may be one match read two ways: they
    read the same fields, each field's two values nest (one template read on past where the other stopped), and each
    claim's values lie in the other's text."""
    dx, du = x["detail"] or {}, u["detail"] or {}
    keys = [k for k in _READ if k in dx or k in du]
    return all(isinstance(dx.get(k), str) and isinstance(du.get(k), str) and (dx[k] in du[k] or du[k] in dx[k])
               and dx[k] in u["text"] and du[k] in x["text"] for k in keys)


def cross_port(a, b, ja, jb):
    """(C)(i) for one input (NOTE_path2a_third_pass_2026_09_30, C-1; NOTE_path2a_fourth_pass_2026_09_30, C-2;
    NOTE_path2a_ninth_pass_2026_10_04). a, b: main's and the branch's Python records; ja, jb: the same in the port.
    Returns the claim counts per key, the splits, and whether every claim pairs. By construction (the keys of
    BY_CONSTRUCTION), since a decision reads nothing of a claim but its kind, verdict and detail (and the door's bytes):
    claims main's ports give the same (kind, verdict, detail) are decided alike, at the same position, matched across
    the lists in order, or anywhere in either list; and where the two lists hold the same claims, in whatever order,
    the gate verdicts are the same. Measured on named sets: the same position with the same (kind, verdict, text);
    in lists of equal length, the claims left over with the same kind and verdict whose details differ but nest and lie
    in each other's text (`_one_match`: one match the two templates read apart, the `extract` guards' work); and the
    gate verdict where every claim pairs one of those ways. Left-over claims that are not one match (two ports reading
    different sentences) are not paired, and an input with such a claim is not one whose claims all pair."""
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
    if same_len and not rest and b["verdict"] != jb["verdict"]:
        splits.append(("gate verdict where the two lists hold the same claims",))
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


def _decisions(tmp, items, mode="--decisions"):
    """The port's --decisions output for items {id, summary, diff}, keyed by id."""
    # ASCII JSON: a summary may hold a lone surrogate, which UTF-8 cannot write (NOTE_path2a_seventh_pass_2026_09_30)
    (tmp / "c_in.json").write_text(json.dumps(items, ensure_ascii=True), encoding="utf-8")
    node(mode, R.main_port_path(tmp), tmp / "c_in.json", tmp / "c_out.json")
    return {d["id"]: d for d in json.loads((tmp / "c_out.json").read_text(encoding="utf-8"))}


@pytest.fixture(scope="module")
def decided(M, inputs, work):
    """The committed inputs as bar C reads them: per input, main's and the branch's records in Python and in the port
    and the four gate verdicts under --strict (`R.bar_c_rows`); and the port's own output, for a second Python reading."""
    node("--decisions", work / "diffgate_main_reference.js", work / "in.json", work / "dec.json")
    js = {d["id"]: d for d in json.loads((work / "dec.json").read_text(encoding="utf-8"))}
    items = [{"id": R.uid(i, row), "summary": row[2], "diff": row[3]} for i, row in enumerate(inputs)]
    return items, js, R.bar_c_rows(M, N, items, js)


def test_cross_port_decisions(decided):
    """C(i) on the committed inputs: 0 splits under the by-construction keys and under the measured ones."""
    _items, _js, rows = decided
    c = collections.Counter()
    splits = []
    for r in rows:
        if r.get("raises"):
            continue
        counts, found, every = cross_port(r["a"], r["b"], r["ja"], r["jb"])
        c.update(counts)
        c["inputs whose claims all pair"] += every
        splits += [(r["id"],) + s for s in found]
    print("cross-port:", dict(c))
    assert not splits, splits[:10]
    assert c["position: kind, verdict, detail"] > 18000 and c["left over, one match: kind and verdict"] > 20


@functools.lru_cache(maxsize=None)
def engine_unicode() -> str:
    """The Unicode version of the engine the port runs on here (`process.versions.unicode`), its major number."""
    if NODE is None:
        no_node()
    r = subprocess.run([NODE, "-p", "process.versions.unicode"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip().split(".")[0]


# Bar C(iii) (NOTE_path2a_ninth_pass_2026_10_04), measured and pinned: how often main's two ports give one input
# different claim lists, on which side, and how often the two gate verdicts then differ under main and under the
# overlay. The order of the figures is BAR_C_KEYS. The figures move with the runtimes, since main's own two readings do:
# a pin is keyed by the interpreter's Unicode version, the path flavour main reads base names with, and the Unicode
# version of the port's engine; each was measured on the runtime it names (CPython 3.12.10 and 3.14.2, Node 24.13.0, the
# other path flavour by `Path` read as that pure flavour). On a runtime not measured here the test checks that the
# figures lie near the measured ones, and prints them.
BAR_C_KEYS = ("inputs", "a main raises", "lists equal", "lists equal, a claim withheld",
              "lists differ, description side", "lists differ, diff side",
              "no --strict: gates differ under main", "no --strict: gates differ under the overlay",
              "no --strict: main agrees, the overlay differs", "no --strict: main differs, the overlay agrees",
              "--strict: gates differ under main", "--strict: gates differ under the overlay",
              "--strict: main agrees, the overlay differs", "--strict: main differs, the overlay agrees")
BAR_C = {
    # the Windows flavour holds three more inputs on the diff side: main's Python decides a drive-like claim (`c:x.py`)
    # there that its port, and its Python under the POSIX flavour, leave UNCHECKABLE
    "committed": {
        ("15.0.0", "windows", "16"): (6672, 5, 6067, 1127, 480, 120, 19, 23, 6, 2, 20, 13, 2, 9),
        ("15.0.0", "posix", "16"): (6672, 5, 6070, 1127, 480, 117, 19, 23, 6, 2, 19, 13, 2, 8),
    },
    # every input of this shape where main's two ports decide the symbol claim apart parts the gate verdicts under the
    # overlay (149), since the tests or count claim that made both of main's gates FAIL is withheld in both ports
    "line break": {("15.0.0", "any", "16"): (600, 0, 451, 451, 0, 149, 0, 149, 149, 0, 0, 0, 0, 0)},
    # on the patched engine main's two gate verdicts differ on 421 inputs and the overlay's on none: the counts the two
    # mains decide apart are withheld in both ports (`case_count`)
    "newer engine": {("15.0.0", "any", "16"): (1500, 0, 669, 294, 0, 831, 421, 0, 0, 421, 358, 0, 0, 358)},
    "decorated world": {("15.0.0", "any", "16"): (1000, 0, 752, 355, 248, 0, 0, 0, 0, 0, 0, 0, 0, 0)},
}
# The committed inputs on which main's two gate verdicts agree and the overlay's do not. Without --strict: #161's four
# f2 separator inputs and y2, where the port's main alone counts a `def test_` after a vertical tab, a form feed, U+2028,
# U+2029 or U+FEFF and its two false verdicts are withheld (PASS) while the Python's right CONTRADICTED stands (FAIL);
# and f2's context line, the other way round. Under --strict: two text-seam inputs where only the port's main reads a
# path claim, which `extract` withholds there.
GATES_APART_UNDER_THE_OVERLAY_ONLY = {
    "no --strict": {"path2:f2-a-vertical-tab-is-not-a-line-break", "path2:f2-a-form-feed-is-not-a-line-break",
                    "path2:f2-a-line-separator-is-not-a-line-break",
                    "path2:f2-a-paragraph-separator-is-not-a-line-break",
                    "path2:f2-a-context-line-holding-a-separator-adds-nothing",
                    "path2:y2-a-changed-test-beside-a-created-bom-test-under-bare-hunks"},
    "--strict": {"p2a-seam:4242:335", "p2a-seam:4242:487"},
}


def _bar_c(name, rows, fl, floor):
    """Assert C(ii) on the rows, that the equal side is not vacuous, and the pinned C(iii) figures of the set `name`."""
    counts, broken, ids = R.bar_c(rows, PHRASES)
    got = tuple(counts[k] for k in BAR_C_KEYS)
    runtime = (unicodedata.unidata_version, fl, engine_unicode())
    print(f"bar C on {name} at {runtime}:", json.dumps(counts), {k: v[:8] for k, v in ids.items()})
    assert broken == [], broken[:10]                                            # C(ii)
    assert counts["lists equal, a claim withheld"] >= floor, counts               # ... and not vacuously
    pins = BAR_C[name]
    if runtime in pins:
        assert got == pins[runtime], dict(zip(BAR_C_KEYS, got))
    else:
        ref = next((pins[k] for k in sorted(pins) if k[1] == fl), None)
        assert all(abs(g - p) <= max(8, p // 20) for g, p in zip(got, ref)), (
            f"bar C(iii) on {name}, on a runtime not measured ({runtime}), is far from the measured figures: "
            f"{dict(zip(BAR_C_KEYS, got))}")
    return counts, ids


def test_equal_lists_give_equal_lists_and_gates_and_the_rest_is_measured(M, decided):
    """C(ii) and C(iii) on the committed inputs. Where main's two ports read the same claim list, the two final lists
    and the two gate verdicts are the same, in both strict modes: asserted, with more than a thousand such inputs on
    which the overlay withheld a claim. Where they do not, the counts are pinned, and so are the inputs on which only
    the overlay's two gate verdicts differ."""
    _items, _js, rows = decided
    _counts, ids = _bar_c("committed", rows, flavour(M), 1000)
    if (unicodedata.unidata_version, engine_unicode()) in {(k[0], k[2]) for k in BAR_C["committed"]}:
        assert {k: {x.split("::")[1] for x in v} for k, v in ids.items()} == GATES_APART_UNDER_THE_OVERLAY_ONLY


def test_bar_c_under_the_other_path_flavour(M, decided, monkeypatch):
    """The same with main's and the branch's `Path` read as the other pure flavour, so both pins run on every runner;
    the port has no path flavour, and its output is read again."""
    other = "posix" if flavour(M) == "windows" else "windows"
    monkeypatch.setattr(M, "Path", FLAVOURS[other])
    monkeypatch.setattr(N, "Path", FLAVOURS[other])
    items, js, _rows = decided
    _bar_c("committed", R.bar_c_rows(M, N, items, js), other, 1000)


D_MOD = "diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-x = 0\n+x = 1\n"
D_GUIDE = ("diff --git a/docs/guide.md b/docs/guide.md\n--- a/docs/guide.md\n+++ b/docs/guide.md\n@@ -1 +1 @@\n"
           "-old\n+new\n") + D_MOD
D_TWINS = D_MOD + "".join(f"diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n@@ -1 +1 @@\n-a\n+b\n"
                          for p in (".env", "env"))
def _mod(p):
    return f"diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n@@ -1 +1 @@\n-a\n+b\n"


D_FOO = ("diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1,3 @@\n-x = 0\n+x = 1\n"
         "+def foo():\n+    return 1\n")
D_DOTC = "diff --git a/.c.py b/.c.py\ndeleted file mode 100644\n--- a/.c.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
D_TEST = ("diff --git a/tests/test_x.py b/tests/test_x.py\n--- a/tests/test_x.py\n+++ b/tests/test_x.py\n"
          "@@ -1,2 +1,2 @@\n-def test_a():\n-    assert 0\n+def test_a():\n+    assert 1\n")
D_TEST_ADDED = ("diff --git a/tests/test_x.py b/tests/test_x.py\n--- a/tests/test_x.py\n+++ b/tests/test_x.py\n"
                "@@ -1,2 +1,4 @@\n-def test_a():\n-    assert 0\n+def test_a():\n+    assert 1\n+def test_b():\n+    assert 2\n")
D_HELPER = ("diff --git a/src/m.py b/src/m.py\n--- a/src/m.py\n+++ b/src/m.py\n@@ -1 +1,2 @@\n-x = 0\n+x = 1\n"
            "+def helper():\n")
D_ZH = ("diff --git a/docs/zh.md b/docs/zh.md\n--- a/docs/zh.md\n+++ b/docs/zh.md\n@@ -1 +1,2 @@\n-a\n+b\n+"
        + "".join(chr(x) for x in (0x4F7F, 0x7528)) + " def " + "".join(chr(x) for x in (0x5B9A, 0x4E49, 0x51FD, 0x6570))
        + "\n")
JOSE = "\n\nThanks to Jos" + chr(0xE9) + " for the review; only a typo fix otherwise."
# Pass 8 (NOTE_path2a_eighth_pass_2026_10_01): a changed test beside a new one (truth: 1 test added), and src/m.py with
# the given added lines
D_TEST_ONE = ("diff --git a/tests/test_x.py b/tests/test_x.py\n--- a/tests/test_x.py\n+++ b/tests/test_x.py\n"
              "@@ -1,2 +1,5 @@\n-def test_a(x):\n-    assert x\n+def test_a(x, y):\n+    assert x\n+\n+def test_b():\n"
              "+    assert 1\n")


def D_LINES(lines):
    return ("diff --git a/src/m.py b/src/m.py\n--- a/src/m.py\n+++ b/src/m.py\n@@ -1 +1,%d @@\n-x = 0\n" % len(lines)
            + "".join("+" + x + "\n" for x in lines))

D_CAFE = ("diff --git a/docs/résumé/index.md b/docs/résumé/index.md\n--- a/docs/résumé/index.md\n"
          "+++ b/docs/résumé/index.md\n@@ -1 +1 @@\n-a\n+b\n"
          "diff --git a/docs/café.md b/docs/café.md\nnew file mode 100644\n--- /dev/null\n+++ b/docs/café.md\n"
          "@@ -0,0 +1 @@\n+x\n")
# The reviews' cross-port reproductions: inputs main's own two ports read with different claim texts, details, sentence
# counts or verdicts, so that one pinned expect cannot hold both. Each row pins the overlay's decisions in each port
# (want for Python, want for the port) and its gate verdict (one word where both ports reach it, "python/port" where
# they part); XPORT_MAIN below pins main's own reading of each beside it. Up to the eighth pass a switch (C-1, `apart`)
# kept every CONTRADICTED on most of these, so that the two gate verdicts agreed wherever main's did; the ninth pass
# removed it (NOTE_path2a_ninth_pass_2026_10_04), and a CONTRADICTED is decided by its kind's rule like any other
# claim: where main's two lists are the same the ports still agree (C(ii)); where they differ the gate verdicts may.
XPORT_CASES = [
    ("R1-bom-joined-sentences", "Modified src/app.py." + chr(0xFEFF) + "Tidied up.", D_MOD,
     [("VERIFIED", None)], [("VERIFIED", None)], "PASS"),
    # the sentence holds `only` and U+0085, where the two templates may read the prefix apart: withheld in both ports
    # (`extract`). main's CONTRADICTED is right here (src/app.py is outside docs/): a right verdict the removal costs
    ("R4-nel-after-only-prefix", "Only touches docs/." + chr(0x85) + "Thanks.", D_MOD,
     [("UNCHECKABLE", "extract")], [("UNCHECKABLE", "extract")], "PASS"),
    ("E1-emoji-release-note", chr(0x1F680) + chr(0x1F389) + " Release prep " + chr(0x1F9F9) + chr(0x1F527)
     + ": bumped the pinned dependencies, regenerated the lockfile, fixed two flaky network timeouts in the nightly "
       "CI job, and updated docs/guide.md for the next release.", D_GUIDE, [("VERIFIED", None)], [("VERIFIED", None)],
     "PASS"),
    # C-1 (NOTE_path2a_fourth_pass_2026_09_30): one count match that CPython reads as 33 and the port as 3 (X5, X5b).
    # Withheld in both (`extract`): the port's CONTRADICTED is false (3 files did change, which only #121 hides), the
    # Python's is right
    ("X5-fullwidth-digit-before-the-count", chr(0xFF13) + "3 files changed.", D_TWINS,
     [("UNCHECKABLE", "extract")], [("UNCHECKABLE", "extract")], "PASS"),
    ("X5b-arabic-digit-before-the-count", chr(0x663) + "3 files changed.", D_TWINS,
     [("UNCHECKABLE", "extract")], [("UNCHECKABLE", "extract")], "PASS"),
    # each port reads a count the other does not (X2, X3); X2's summary holds U+0085, a count seam (`seam`). One port's
    # count is false through #121 (3 files changed), the other's is right
    ("X2-nel-and-cjk-counts", "3 files" + chr(0x85) + "changed. " + chr(0x5171) + "5 files changed.", D_TWINS,
     [("UNCHECKABLE", "seam")], [("UNCHECKABLE", "seam")], "PASS"),
    ("X3-fullwidth-and-cjk-counts", chr(0xFF13) + " files changed. " + chr(0x5171) + "5 files changed.", D_TWINS,
     [("UNCHECKABLE", "extract")], [("UNCHECKABLE", "extract")], "PASS"),
    # C-2 (fourth review): the realistic X6b, two different claims each port reads from a different sentence; the
    # decisions differ, as the two claims do, and neither is paired with the other
    ("X6b-accented-directory-and-a-created-file",
     "Changed the parser in `docs/résumé/index.md`. Added `docs/café.md`.", D_CAFE,
     [("VERIFIED", None)], [("UNCHECKABLE", "extract")], "PASS"),
    # C-1 (NOTE_path2a_fifth_pass_2026_09_30): each port reads a count the other does not, one of them with a clean
    # number, across a white space only one port reads or through a letter only CPython folds: withheld in both (`seam`)
    ("X2-13-nel-and-cjk-counts", "13 files\x85changed. 共5 files changed.", D_TWINS,
     [("UNCHECKABLE", "seam")], [("UNCHECKABLE", "seam")], "PASS"),
    ("G1-unit-separator-then-e-acute", "13\x1ffiles changed, é3 files changed.", D_TWINS,
     [("UNCHECKABLE", "seam")], [("UNCHECKABLE", "seam")], "PASS"),
    ("G2-bom-then-long-s", "13﻿files changed and 3 fileſ changed.", D_TWINS,
     [("UNCHECKABLE", "seam")], [("UNCHECKABLE", "seam")], "PASS"),
    ("G3-nel-then-e-acute", "13\x85files changed, é3 files changed.", D_TWINS,
     [("UNCHECKABLE", "seam")], [("UNCHECKABLE", "seam")], "PASS"),
    # I-2 (NOTE_path2a_fifth_pass_2026_09_30): a declared count, path, name and prefix whose value also occurs beside a
    # letter outside ASCII keep main's verdict in both ports (DECLARE-1 writes their sentences); the same sentences
    # undeclared are withheld (`extract`). A port-only removal of a declared skip splits the gates.
    ("D-declared-count", "```styxx\nfiles_changed: 7\n```\nSee ticket é7.", D_TWINS,
     [("CONTRADICTED", None)], [("CONTRADICTED", None)], "FAIL"),
    ("D-declared-path", "```styxx\nfile_touched: src/app.py\n```\nSee src/app.pyé too.", D_MOD,
     [("VERIFIED", None)], [("VERIFIED", None)], "PASS"),
    ("D-declared-name", "```styxx\nadds_symbol: foo\n```\nSee fooé too.", D_FOO,
     [("VERIFIED", None)], [("VERIFIED", None)], "PASS"),
    ("D-declared-prefix", "```styxx\nonly_touches: src\n```\nWe only édited src.", D_MOD,
     [("VERIFIED", None)], [("VERIFIED", None)], "PASS"),
    ("U-undeclared-count", "7 files changed. See ticket é7.", D_TWINS,
     [("UNCHECKABLE", "extract")], [("UNCHECKABLE", "extract")], "PASS"),
    ("U-undeclared-path", "Modified src/app.py. See src/app.pyé too.", D_MOD,
     [("UNCHECKABLE", "extract")], [("UNCHECKABLE", "extract")], "PASS"),
    ("U-undeclared-name", "Adds function foo. See fooé too.", D_FOO,
     [("UNCHECKABLE", "extract")], [("UNCHECKABLE", "extract")], "PASS"),
    ("U-undeclared-prefix", "Only touches src. We only édited src.", D_MOD,
     [("UNCHECKABLE", "extract")], [("UNCHECKABLE", "extract")], "PASS"),
    # C-3 (NOTE_path2a_fifth_pass_2026_09_30): the one known false pairing of the measured one-match key. main's Python
    # reads ['..c.py', '.c.py'] and its port ['..c.py', '..c.py'] (the port's `^` after U+2028); the left-over claims
    # nest and each lies in the other's text, so the key pairs two different matches, whose decisions differ as the
    # claims do. The split is expected here and nowhere else.
    ("L1-a-false-one-match-pairing", 'Tidied.\x85- ".c.py" — updated - "..c.py" -- updated.', D_DOTC,
     [("UNCHECKABLE", "dot"), ("VERIFIED", None)], [("UNCHECKABLE", "dot"), ("UNCHECKABLE", "dot")], "PASS",
     [("left over, one match: kind and verdict", 1, 1, ("VERIFIED", None), ("UNCHECKABLE", "dot"))]),
    # A-2 (NOTE_path2a_fifth_pass_2026_09_30): the case doubt's base-name fact alone decides, then its suffix fact alone.
    # main's Python verifies each by base name or suffix after lower(); its port reads no claim (a base name outside
    # ASCII), so these pin the Python's decision, and the plants test reads them too.
    ("P5-case-base-name-only", "Modified y/dé.md.", _mod("x/dÉ.md"),
     [("UNCHECKABLE", "case")], [], "PASS"),
    ("P5-case-suffix-only", "Modified é/dé.md.", _mod("z/É/dé.md"),
     [("UNCHECKABLE", "case")], [], "PASS"),
    # C-1 (NOTE_path2a_sixth_pass_2026_09_30): each port reads a claim the other does not, a count or a tests claim in
    # either place, and main's gates are FAIL / FAIL. The Python's claim (a count 13 beside a seam, `seam`; "Added 0
    # tests" over a changed test, false through #101, `tests`) is withheld; the port's (3 or 9 against the diff's count)
    # is right and stands. So the two gate verdicts part where main's agree: C(iii), main's lists differ here
    ("C1-count-vs-tests", "13\x1cfiles changed. Added 3﻿tests.", D_TWINS + D_TEST,
     [("UNCHECKABLE", "seam")], [("CONTRADICTED", None)], "PASS/FAIL"),
    ("C1-tests-vs-count", "Added 0\x1ctests. 9﻿files changed.", D_TEST,
     [("UNCHECKABLE", "tests")], [("CONTRADICTED", None)], "PASS/FAIL"),
    ("C1-tests-vs-tests", "Added 0\x1ctests. Added 3﻿tests.", D_TEST,
     [("UNCHECKABLE", "tests")], [("CONTRADICTED", None)], "PASS/FAIL"),
    ("C1-tests-vs-tests-long-s", "Added 0 teſts. Added 3﻿tests.", D_TEST,
     [("UNCHECKABLE", "tests")], [("CONTRADICTED", None)], "PASS/FAIL"),
    ("C1-tests-vs-tests-accents", "Addéd 0 tests. éAdded 3 tests.", D_TEST,
     [("UNCHECKABLE", "tests")], [("CONTRADICTED", None)], "PASS/FAIL"),
    # ... and the review's inherent case: a tests claim both ports read (false through #101, withheld in both), beside
    # a count only the port reads, which is right and stands there
    ("C1-inherent", "Added 0 tests. 9﻿files changed.", D_TEST,
     [("UNCHECKABLE", "tests")], [("UNCHECKABLE", "tests"), ("CONTRADICTED", None)], "PASS/FAIL"),
    # ... and the DECLARE-1 shape: the port's `^` opens a fence after U+2028, CPython's does not, so only the port reads
    # a declared count, which is right and stands
    ("C1-a-fence-only-the-port-opens", "Added 0 tests.\nx ```styxx\nfiles_changed: 9\n```\n", D_TEST,
     [("UNCHECKABLE", "tests")], [("UNCHECKABLE", "tests"), ("CONTRADICTED", None)], "PASS/FAIL"),
    # C-3 (NOTE_path2a_sixth_pass_2026_09_30): OM1, a false pairing of the one-match key. main's Python reads b/c.py
    # (exact, kept), its port a/b/c.py (by base name, withheld as dir); under --strict the gates split although every
    # claim pairs (the strict gate is C-4's case).
    ("OM1-a-false-one-match-pairing", "Modıfied b/c.py.﻿`a/b/c.py` — updated",
     _mod("b/c.py").replace("-a\n+b\n", "-x = 0\n+x = 1\n"), [("VERIFIED", None)], [("UNCHECKABLE", "dir")], "PASS",
     [("left over, one match: kind and verdict", 0, 0, ("VERIFIED", None), ("UNCHECKABLE", "dir"))]),
    # Pass 7 (NOTE_path2a_seventh_pass_2026_09_30), B-1: a sentence elsewhere in the summary holding an accented letter
    # or a pictograph emoji beside a word of a template; main's false tests or count verdict is withheld in both ports
    ("B1-jose", "Added 1 test." + JOSE, D_TEST_ADDED, [("UNCHECKABLE", "tests")], [("UNCHECKABLE", "tests")], "PASS"),
    ("B1-emoji-heading", "Added 1 test.\n\n## " + chr(0x1F9EA) + " Tests added", D_TEST_ADDED,
     [("UNCHECKABLE", "tests")], [("UNCHECKABLE", "tests")], "PASS"),
    ("B1-naive", "Added 1 test.\n\nBehaviour is unchanged for na" + chr(0xEF) + "ve callers.", D_TEST_ADDED,
     [("UNCHECKABLE", "tests")], [("UNCHECKABLE", "tests")], "PASS"),
    ("B1-bug-emoji", "Added 1 test.\n\n" + chr(0x1F41B) + " What changed: see above.", D_TEST_ADDED,
     [("UNCHECKABLE", "tests")], [("UNCHECKABLE", "tests")], "PASS"),
    ("B1-count-jose", "2 files changed." + JOSE, _mod(".env.example") + _mod("env.example"),
     [("UNCHECKABLE", "count")], [("UNCHECKABLE", "count")], "PASS"),
    # O-11: a pictograph emoji in the claim's own sentence is neutral, as one code point or as the two surrogates a JSON
    # reader hands the port, held so in the Python too
    ("O11-emoji-in-the-sentence", "Added 1 test " + chr(0x1F9EA) + ".", D_TEST_ADDED,
     [("UNCHECKABLE", "tests")], [("UNCHECKABLE", "tests")], "PASS"),
    ("O11-split-surrogates", "Added 1 test " + chr(0xD83E) + chr(0xDDEA) + ".", D_TEST_ADDED,
     [("UNCHECKABLE", "tests")], [("UNCHECKABLE", "tests")], "PASS"),
    # ... and the words in the template's order beside an accented letter, or beside an emoji outside the five blocks
    ("B1-ordered-words-outside-the-window", "Added 1 test. Jos" + chr(0xE9) + " added 1 test too.", D_TEST_ADDED,
     [("UNCHECKABLE", "tests"), ("UNCHECKABLE", "tests")], [("UNCHECKABLE", "tests"), ("UNCHECKABLE", "tests")], "PASS"),
    ("O11-another-block-outside-the-window", "Added 1 test " + chr(0x1F7E0) + ".", D_TEST_ADDED,
     [("UNCHECKABLE", "tests")], [("UNCHECKABLE", "tests")], "PASS"),
    # ... and O-11 where it decides a claim (the ninth pass's pins for the plants that drop O-11 from one port): a
    # pictograph emoji right after a claimed name or right before a count is neutral, so the name's VERIFIED and the
    # count's CONTRADICTED are decided by their own rules. With the emoji read as wordish, the name and the number
    # would lie in a run the two templates may read apart, and both would be withheld (`extract`)
    ("O11-emoji-after-a-name", "Adds function foo" + chr(0x1F9EA) + ". Added 1 test.", D_TEST_ONE,
     [("CONTRADICTED", None), ("UNCHECKABLE", "tests")], [("CONTRADICTED", None), ("UNCHECKABLE", "tests")], "FAIL"),
    ("O11-emoji-after-a-verified-name", "Adds function foo" + chr(0x1F9EA) + ".", D_FOO,
     [("VERIFIED", None)], [("VERIFIED", None)], "PASS"),
    ("O11-emoji-before-a-count", chr(0x1F680) + "5 files changed.", _mod(".env") + _mod("env"),
     [("CONTRADICTED", None)], [("CONTRADICTED", None)], "FAIL"),
    # ... a name or a prefix that runs into a letter outside ASCII, a test count only the port reads: the symbol claim
    # main CONTRADICTS in both ports stands (no removed line defines foo), the scope is withheld (`extract`), and the
    # false tests and count verdicts beside them are withheld
    ("B2-in-window-name-runs-on", "Adds function foo\xe9. Added 1 test.", D_TEST_ONE,
     [("CONTRADICTED", None), ("UNCHECKABLE", "tests")], [("CONTRADICTED", None), ("UNCHECKABLE", "tests")], "FAIL"),
    ("B2-in-window-prefix", "Only touches docs/\xe9 and src/. Added 1 test.", D_TEST_ONE,
     [("UNCHECKABLE", "extract"), ("UNCHECKABLE", "tests")], [("UNCHECKABLE", "extract"), ("UNCHECKABLE", "tests")],
     "PASS"),
    ("B2-in-window-test-count", "Added 1 test\xe9. 3 files changed.", _mod(".env") + _mod("env") + D_TEST_ONE,
     [("UNCHECKABLE", "count")], [("UNCHECKABLE", "tests"), ("UNCHECKABLE", "count")], "PASS"),
    # ... the character just before the verb (its `\b`: CPython reads no claim, the port reads "Added 3 tests", which is
    # right and stands), and one inside the optional noun after `tests` (CPython reads no noun, the port reads `cases`,
    # a case not being a function, and main itself leaves that claim UNCHECKABLE)
    ("B2-wordish-before-the-verb", "Added 1 test. " + chr(0xE9) + "Added 3 tests.", D_TEST_ONE,
     [("UNCHECKABLE", "tests")], [("UNCHECKABLE", "tests"), ("CONTRADICTED", None)], "PASS/FAIL"),
    ("B2-noun-in-the-window", "Added 1 tests cases" + chr(0xE9) + ". 3 files changed.",
     _mod(".env") + _mod("env") + D_TEST_ONE,
     [("UNCHECKABLE", "tests"), ("UNCHECKABLE", "count")], [("UNCHECKABLE", None), ("UNCHECKABLE", "count")], "PASS"),
    # B-3: an added def whose name runs into an accented letter, or a CJK line holding `def`, that no claim names: the
    # false tests verdict is withheld. Where the claimed name is that run, CPython's \b reads the accented letter as a
    # word character and the port's does not, so the two mains decide the symbol claim apart (the diff side of C(iii));
    # the tests claim is withheld in both, and the gates part on the symbol claim alone
    ("B3-an-accented-def-elsewhere", "Added function helper. Added 1 test.",
     D_TEST_ADDED + D_HELPER.replace("+def helper():\n", "+def helper():\n+def caf" + chr(0xE9) + "():\n"),
     [("VERIFIED", None), ("UNCHECKABLE", "tests")], [("VERIFIED", None), ("UNCHECKABLE", "tests")], "PASS"),
    ("B3-a-cjk-line-holding-def", "Added function helper. Added 1 test.", D_TEST_ADDED + D_HELPER + D_ZH,
     [("VERIFIED", None), ("UNCHECKABLE", "tests")], [("VERIFIED", None), ("UNCHECKABLE", "tests")], "PASS"),
    ("B3-the-claimed-name-runs-on", "Added function caf. Added 1 test.",
     D_TEST_ADDED + D_HELPER.replace("+def helper():\n", "+def caf" + chr(0xE9) + "():\n"),
     [("CONTRADICTED", None), ("UNCHECKABLE", "tests")], [("VERIFIED", None), ("UNCHECKABLE", "tests")], "FAIL/PASS"),
    # a Unicode 16 case pair, U+A7DC and U+019B. CPython 3.9 to 3.12 (Unicode 15 or older) key the two paths apart and
    # Node 24 (Unicode 16) merges them, so main's two ports may count 3 and 2 files: the count is withheld in both
    # whatever each main reads (`case_count`), and the changed test's false verdict too
    ("C1-a-unicode-16-case-pair", "2 files changed. Added 0 tests.",
     _mod("src/" + chr(0xA7DC) + ".py") + _mod("src/" + chr(0x19B) + ".py") + D_TEST,
     [("UNCHECKABLE", "case_count"), ("UNCHECKABLE", "tests")], [("UNCHECKABLE", "case_count"), ("UNCHECKABLE", "tests")],
     "PASS"),
    # Pass 8 (NOTE_path2a_eighth_pass_2026_10_01), C-1: main's symbol regex reads `\s+` across the joined added lines,
    # so a `def` that ends its line is read with the name on the next one, where the two ports' \s and \b part: main's
    # Python and port decide the symbol claim apart (the diff side of C(iii)) and agree on FAIL through the tests or
    # count claim, which is false (#101, #121) and withheld in both. The gates then part on the symbol claim alone
    ("C1-xl-unit-separator", "Adds function foo. Added 0 tests.", D_TEST + D_LINES(["x = 1", "def", "\x1ffoo():"]),
     [("VERIFIED", None), ("UNCHECKABLE", "tests")], [("CONTRADICTED", None), ("UNCHECKABLE", "tests")], "PASS/FAIL"),
    ("C1-xl-bom", "Adds function foo. Added 0 tests.", D_TEST + D_LINES(["x = 1", "def", "﻿foo():"]),
     [("CONTRADICTED", None), ("UNCHECKABLE", "tests")], [("VERIFIED", None), ("UNCHECKABLE", "tests")], "FAIL/PASS"),
    ("C1-xl-e-acute", "Adds function foo. Added 0 tests.", D_TEST + D_LINES(["def", "foo\xe9():"]),
     [("CONTRADICTED", None), ("UNCHECKABLE", "tests")], [("VERIFIED", None), ("UNCHECKABLE", "tests")], "FAIL/PASS"),
    ("C1-xl-count", "Adds function foo. 3 files changed.", _mod(".env") + _mod("env") + D_LINES(["def", "\x1ffoo():"]),
     [("VERIFIED", None), ("UNCHECKABLE", "count")], [("CONTRADICTED", None), ("UNCHECKABLE", "count")], "PASS/FAIL"),
    # B-1: two changed paths that differ outside ASCII but not in case (CJK, Cyrillic, two accented letters): the count
    # is decided as before and the changed test's false verdict (#101) is withheld in both ports
    ("B1-cjk-names", "3 files changed. Added 1 test.", _mod("docs/zh/安装.md") + _mod("docs/zh/配置.md")
     + D_TEST_ONE, [("VERIFIED", None), ("UNCHECKABLE", "tests")], [("VERIFIED", None), ("UNCHECKABLE", "tests")], "PASS"),
    ("B1-cyrillic-names", "3 files changed. Added 1 test.",
     _mod("x/ФабЖе.os") + _mod("x/ФабЗа.os") + D_TEST_ONE,
     [("VERIFIED", None), ("UNCHECKABLE", "tests")], [("VERIFIED", None), ("UNCHECKABLE", "tests")], "PASS"),
    ("B1-two-accents", "3 files changed. Added 1 test.", _mod("i18n/caf\xe9.txt") + _mod("i18n/caf\xe8.txt")
     + D_TEST_ONE, [("VERIFIED", None), ("UNCHECKABLE", "tests")], [("VERIFIED", None), ("UNCHECKABLE", "tests")], "PASS"),
    # ... and a case pair (E acute, both cases): the count claim in [#CA, #A] is withheld in both ports, whatever its
    # verdict
    ("B1-a-latin-case-pair", "3 files changed. Added 1 test.", _mod("src/\xc9.py") + _mod("src/\xe9.py") + D_TEST_ONE,
     [("UNCHECKABLE", "case_count"), ("UNCHECKABLE", "tests")],
     [("UNCHECKABLE", "case_count"), ("UNCHECKABLE", "tests")], "PASS"),
    # B-2: a letter outside ASCII in the claim's own sentence, away from where its number is read
    ("B2-naive-inputs", "Added 1 test for na\xefve inputs.", D_TEST_ONE, [("UNCHECKABLE", "tests")],
     [("UNCHECKABLE", "tests")], "PASS"),
    ("B2-joses-parser", "Added 1 test for Jos\xe9's parser.", D_TEST_ONE, [("UNCHECKABLE", "tests")],
     [("UNCHECKABLE", "tests")], "PASS"),
    ("B2-cjk-word", "Added 1 test for 用户 login.", D_TEST_ONE, [("UNCHECKABLE", "tests")],
     [("UNCHECKABLE", "tests")], "PASS"),
    ("B2-count-cafe", "2 files changed (caf\xe9 config).", _mod(".env") + _mod("env"), [("UNCHECKABLE", "count")],
     [("UNCHECKABLE", "count")], "PASS"),
    ("B2-count-thanks", "2 files changed, thanks to Jos\xe9.", _mod(".env") + _mod("env"), [("UNCHECKABLE", "count")],
     [("UNCHECKABLE", "count")], "PASS"),
    ("B2-new-button", chr(0x1F195) + " Added 1 test.", D_TEST_ONE, [("UNCHECKABLE", "tests")], [("UNCHECKABLE", "tests")],
     "PASS"),
    ("B2-keycap", "1️⃣ Added 1 test.", D_TEST_ONE, [("UNCHECKABLE", "tests")], [("UNCHECKABLE", "tests")],
     "PASS"),
    # Pass 9 (NOTE_path2a_ninth_pass_2026_10_04), C-1 of the eighth review: the scope template's optional `files in`
    # group, which main's regex backtracks out of to read `files` as the prefix. CPython reads no `\b` before `only`
    # after the accented letter, so only the port's main reads the scope claim, which is right (other/b.py is outside
    # files/) and stands; the count, false through #121, is withheld in both. The gates part as they did at e820f291,
    # whose window reader missed the backtrack
    ("P9-only-after-an-accent-reads-files-as-the-prefix", "Caf\xe9only touches files in (docs). 4 files changed.",
     _mod("files/a.py") + _mod("other/b.py") + _mod(".env") + _mod("env"),
     [("UNCHECKABLE", "count")], [("CONTRADICTED", None), ("UNCHECKABLE", "count")], "PASS/FAIL"),
    # ... and the decorations under which the removed switch kept every CONTRADICTED though main's two ports read the
    # summary alike (P-4 of the ninth note): U+FEFF before the summary, and a sentence naming styxx beside U+2028
    ("P9-a-bom-before-the-summary", "﻿3 files changed. Added 1 test.", _mod(".env") + _mod("env") + D_TEST_ONE,
     [("UNCHECKABLE", "count"), ("UNCHECKABLE", "tests")], [("UNCHECKABLE", "count"), ("UNCHECKABLE", "tests")], "PASS"),
    ("P9-styxx-beside-a-line-separator", "3 files changed. Added 1 test.\nChecked with styxx. Thanks.",
     _mod(".env") + _mod("env") + D_TEST_ONE,
     [("UNCHECKABLE", "count"), ("UNCHECKABLE", "tests")], [("UNCHECKABLE", "count"), ("UNCHECKABLE", "tests")], "PASS"),
]
# main's own reading of each cross-port case, pinned beside the overlay's: one letter per claim (V, C, U; `?` where the
# verdict moves with the runtime's Unicode version) for the Python and for the port, then the two gate verdicts.
XPORT_MAIN = {
    "R1-bom-joined-sentences": "V/V PASS/PASS", "R4-nel-after-only-prefix": "C/C FAIL/FAIL",
    "E1-emoji-release-note": "V/V PASS/PASS", "X5-fullwidth-digit-before-the-count": "C/C FAIL/FAIL",
    "X5b-arabic-digit-before-the-count": "C/C FAIL/FAIL", "X2-nel-and-cjk-counts": "C/C FAIL/FAIL",
    "X3-fullwidth-and-cjk-counts": "C/C FAIL/FAIL", "X6b-accented-directory-and-a-created-file": "V/V PASS/PASS",
    "X2-13-nel-and-cjk-counts": "C/C FAIL/FAIL", "G1-unit-separator-then-e-acute": "C/C FAIL/FAIL",
    "G2-bom-then-long-s": "C/C FAIL/FAIL", "G3-nel-then-e-acute": "C/C FAIL/FAIL", "D-declared-count": "C/C FAIL/FAIL",
    "D-declared-path": "V/V PASS/PASS", "D-declared-name": "V/V PASS/PASS", "D-declared-prefix": "V/V PASS/PASS",
    "U-undeclared-count": "C/C FAIL/FAIL", "U-undeclared-path": "V/V PASS/PASS", "U-undeclared-name": "V/V PASS/PASS",
    "U-undeclared-prefix": "V/V PASS/PASS", "L1-a-false-one-match-pairing": "VV/VV PASS/PASS",
    "P5-case-base-name-only": "V/ PASS/PASS", "P5-case-suffix-only": "V/ PASS/PASS",
    "C1-count-vs-tests": "C/C FAIL/FAIL", "C1-tests-vs-count": "C/C FAIL/FAIL", "C1-tests-vs-tests": "C/C FAIL/FAIL",
    "C1-tests-vs-tests-long-s": "C/C FAIL/FAIL", "C1-tests-vs-tests-accents": "C/C FAIL/FAIL",
    "C1-inherent": "C/CC FAIL/FAIL", "C1-a-fence-only-the-port-opens": "C/CC FAIL/FAIL",
    "OM1-a-false-one-match-pairing": "V/V PASS/PASS", "B1-jose": "C/C FAIL/FAIL", "B1-emoji-heading": "C/C FAIL/FAIL",
    "B1-naive": "C/C FAIL/FAIL", "B1-bug-emoji": "C/C FAIL/FAIL", "B1-count-jose": "C/C FAIL/FAIL",
    "O11-emoji-in-the-sentence": "C/C FAIL/FAIL", "O11-split-surrogates": "C/C FAIL/FAIL",
    "B1-ordered-words-outside-the-window": "CC/CC FAIL/FAIL", "O11-another-block-outside-the-window": "C/C FAIL/FAIL",
    "O11-emoji-after-a-name": "CC/CC FAIL/FAIL", "O11-emoji-after-a-verified-name": "V/V PASS/PASS",
    "O11-emoji-before-a-count": "C/C FAIL/FAIL", "B2-in-window-name-runs-on": "CC/CC FAIL/FAIL",
    "B2-in-window-prefix": "CC/CC FAIL/FAIL", "B2-in-window-test-count": "C/CC FAIL/FAIL",
    "B2-wordish-before-the-verb": "C/CC FAIL/FAIL", "B2-noun-in-the-window": "CC/UC FAIL/FAIL",
    "B3-an-accented-def-elsewhere": "VC/VC FAIL/FAIL", "B3-a-cjk-line-holding-def": "VC/VC FAIL/FAIL",
    "B3-the-claimed-name-runs-on": "CC/VC FAIL/FAIL", "C1-a-unicode-16-case-pair": "?C/?C FAIL/FAIL",
    "C1-xl-unit-separator": "VC/CC FAIL/FAIL", "C1-xl-bom": "CC/VC FAIL/FAIL", "C1-xl-e-acute": "CC/VC FAIL/FAIL",
    "C1-xl-count": "VC/CC FAIL/FAIL", "B1-cjk-names": "VC/VC FAIL/FAIL", "B1-cyrillic-names": "VC/VC FAIL/FAIL",
    "B1-two-accents": "VC/VC FAIL/FAIL", "B1-a-latin-case-pair": "CC/CC FAIL/FAIL", "B2-naive-inputs": "C/C FAIL/FAIL",
    "B2-joses-parser": "C/C FAIL/FAIL", "B2-cjk-word": "C/C FAIL/FAIL", "B2-count-cafe": "C/C FAIL/FAIL",
    "B2-count-thanks": "C/C FAIL/FAIL", "B2-new-button": "C/C FAIL/FAIL", "B2-keycap": "C/C FAIL/FAIL",
    "P9-only-after-an-accent-reads-files-as-the-prefix": "C/CC FAIL/FAIL",
    "P9-a-bom-before-the-summary": "CC/CC FAIL/FAIL", "P9-styxx-beside-a-line-separator": "CC/CC FAIL/FAIL",
}



def test_cross_port_reproductions(M, tmp_path):
    """The reviews' inputs whose claim text, detail or verdict differs between main's two ports, the declared values
    beside a letter outside ASCII, the decorated sentences, and the ninth pass's: for each, main's reading in both
    ports (XPORT_MAIN), the overlay's decisions in both, and the gate verdict each port reaches. C(i) holds on every
    one (a case may name the splits of the measured keys it expects: L1 and OM1, the known false pairings); C(ii) holds
    on those whose main lists are the same."""
    items = [{"id": x[0], "summary": x[1], "diff": x[2]} for x in XPORT_CASES]
    js = _decisions(tmp_path, items)
    letter = {"VERIFIED": "V", "CONTRADICTED": "C", "UNCHECKABLE": "U"}
    assert set(XPORT_MAIN) == {x[0] for x in XPORT_CASES}
    for cid, s, d, want_py, want_js, gate, *expect in XPORT_CASES:
        a, b = M.gate_diff_text(s, d).to_dict(), N.gate_diff_text(s, d).to_dict()
        mains = "%s/%s %s/%s" % ("".join(letter[c["verdict"]] for c in a["claims"]),
                                 "".join(letter[c["verdict"]] for c in js[cid]["main"]["claims"]),
                                 a["verdict"], js[cid]["main"]["verdict"])
        pin = XPORT_MAIN[cid]
        assert len(mains) == len(pin) and all(p in ("?", m) for p, m in zip(pin, mains)), (cid, mains)
        assert [_seen(x) for x in b["claims"]] == want_py, cid
        assert [_seen(x) for x in js[cid]["new"]["claims"]] == want_js, cid
        assert cross_port(a, b, js[cid]["main"], js[cid]["new"])[1] == (expect[0] if expect else []), cid
        assert [b["verdict"], js[cid]["new"]["verdict"]] == (gate.split("/") if "/" in gate else [gate, gate]), cid
    counts, broken, _ids = R.bar_c(R.bar_c_rows(M, N, items, js), PHRASES)
    print("bar C on the cross-port cases:", json.dumps(counts))
    assert broken == [] and counts["lists equal, a claim withheld"] >= 25, (broken[:5], counts)


def _newer_engine_items(seed: int = 8, n: int = 1500) -> list[dict]:
    """C-1 (NOTE_path2a_seventh_pass_2026_09_30): the sixth review's gen8 shape. Diffs holding both halves of a case pair
    beside claims the overlay withholds (a changed test, a changed function, a dotted scope, a dot-twin count), under
    ASCII summaries: U+A7CE and U+A7CF, which the patched port folds and no CPython yet does, and the Unicode 16 pair
    U+A7DC and U+019B, which CPython 3.14 and Node 24 fold and 3.12 does not."""
    import random
    r = random.Random(seed)
    pairs = [(chr(0xA7CE), chr(0xA7CF)), (chr(0xA7DC), chr(0x19B))]
    plain = ["src/app.py", "docs/guide.md", ".env", "env", "src/.app.py", "README.md", "pkg/mod.py"]
    changed_fn = ("diff --git a/src/lib.py b/src/lib.py\n--- a/src/lib.py\n+++ b/src/lib.py\n"
                  "@@ -1,2 +1,2 @@\n-def helper():\n-    return 0\n+def helper():\n+    return 1\n")
    out = []
    for i in range(n):
        a, b = pairs[0] if r.random() < 0.7 else pairs[1]
        d = r.choice(["src/", "x/", "", "docs/"])
        ext = r.choice([".py", ".md"])
        files = [d + a + ext, d + b + ext] + r.sample(plain, r.randint(0, 3))
        r.shuffle(files)
        diff = "".join(_mod(p) for p in files)
        extra = r.choice(["tests", "symbol", "scope", "none"])
        diff += D_TEST if extra == "tests" else changed_fn if extra == "symbol" else ""
        n_files = len(set(files)) + (extra in ("tests", "symbol"))
        k = r.choice([n_files - 1, n_files, n_files + 1, r.randint(0, 8)])
        sents = ["%d files changed" % k]
        if extra == "tests":
            sents.append(r.choice(["Added 0 tests", "Added 1 test", "Added 2 tests"]))
        elif extra == "symbol":
            sents.append("Adds function helper")
        elif extra == "scope":
            sents.append(r.choice(["Only touches src", "Only touches env", "Only touches .env and src/",
                                   "Only touches docs and github/"]))
        if r.random() < 0.3:
            sents.append("Only touches " + r.choice(["src/", "docs/", "x/"]))
        r.shuffle(sents)
        out.append({"id": f"newer:{seed}:{i}", "summary": ". ".join(sents) + ".", "diff": diff})
    return out


def _line_break_items(seed: int = 8, n: int = 600) -> list[dict]:
    """C-1 (NOTE_path2a_eighth_pass_2026_10_01): the seventh review's `genxl` shape. An added `def`, `class` or `async
    def` line followed only by white space of either port, the name on a later added line after a run of such white
    space (U+001F and U+0085 are CPython's alone, U+FEFF the port's), the name running on into a letter outside ASCII or
    not, beside a claim the overlay withholds: a changed test under "Added 0 tests", or a dot twin under "3 files
    changed"."""
    import random
    r = random.Random(seed)
    seps = ["", " ", "\t", chr(0x1F), chr(0xFEFF), chr(0xA0), chr(0x3000), chr(0x0B), chr(0x0C), chr(0x1C), chr(0x85)]
    tails = ["():", "(x):", chr(0xE9) + "():", chr(0xE9), ":", " = 1", chr(0x105C0) + "():", chr(0x301) + "():", "_x():",
             ""]
    out = []
    for i in range(n):
        name = r.choice(["foo", "Foo", "helper"])
        lines = [r.choice(["", "  "]) + r.choice(["def", "class", "async def"])
                 + "".join(r.choice(seps) for _ in range(r.randint(0, 2)))]
        lines += ["".join(r.choice(seps) for _ in range(r.randint(1, 2))) for _ in range(r.randint(0, 2))]
        lines.append("".join(r.choice(seps) for _ in range(r.randint(0, 2))) + name + r.choice(tails))
        tests = r.random() < 0.5
        sents = [r.choice(["Adds function %s", "Added class %s", "Introduces method %s"]) % name,
                 "Added 0 tests" if tests else "3 files changed"]
        r.shuffle(sents)
        out.append({"id": f"xl:{seed}:{i}", "summary": ". ".join(sents) + ".",
                    "diff": (D_TEST if tests else _mod(".env") + _mod("env")) + D_LINES(lines)})
    return out


def _decorated_world_items(seed: int = 20261004, per_family: int = 250) -> list[dict]:
    """A seeded world for C(ii) (NOTE_path2a_ninth_pass_2026_10_04): the truth generator's four families (#97, #121,
    #121 with case outside ASCII, #101) at a seed of its own, each summary as the generator wrote it or under one
    decoration that leaves main's two claim lists the same: U+FEFF before it; a sentence naming styxx beside U+2028;
    "(naive)" with an i-diaeresis after each tests or count word; a lone CR and an accented name after it; a form feed
    and a pictograph emoji. The switch the ninth pass removed read the U+FEFF, the U+2028 and the lone CR as reasons
    to keep every CONTRADICTED (P-4 of that note), in both ports."""
    from tests import _p2a_cases as C
    decorations = [
        lambda s: s,
        lambda s: chr(0xFEFF) + s,
        lambda s: s + "\nChecked with styxx." + chr(0x2028) + "Thanks.",
        lambda s: re.sub(r"(\b(?:tests?|files? changed)\b)", r"\1 (na" + chr(0xEF) + "ve)", s),
        lambda s: s + "\rThanks to Jos" + chr(0xE9) + ", who ran styxx on it.",
        lambda s: s + "\n\x0c" + chr(0x1F9EA) + " Notes",
    ]
    return [{"id": f"world:{seed}:{k}", "summary": decorations[k % len(decorations)](c["summary"]), "diff": c["diff"]}
            for k, c in enumerate(C.families(seed, per_family))]


SETS = {"newer engine": (_newer_engine_items, "--decisions-newer-engine", 250),
        "line break": (_line_break_items, "--decisions", 400),
        "decorated world": (_decorated_world_items, "--decisions", 300)}


@pytest.mark.parametrize("name", sorted(SETS))
def test_bar_c_on_the_seeded_sets(name, M, tmp_path):
    """C(i), C(ii) and C(iii) on three seeded sets (NOTE_path2a_ninth_pass_2026_10_04). `newer engine`: the port runs on
    an engine patched to fold U+A7CE and U+A7CF (unassigned through Unicode 16, standing in for a later version's
    pair), so main's two ports key such a pair of paths apart and decide count claims apart; `case_count` withholds
    those counts in both ports, and the test asserts that main's two ports do count apart there. `line break`: a `def`
    that ends its line, where main's two ports decide the symbol claim apart on more than 100 inputs, and the gate
    verdicts then part on that claim (the eighth pass kept every CONTRADICTED there to hide it). `decorated world`: the
    truth generator's families under decorations that leave main's two lists the same. On each: no split under the
    by-construction keys; where main's lists are the same, the same final lists and gate verdicts in both strict modes,
    on more inputs with a withheld claim than the floor; and the pinned counts of the rest."""
    make, mode, floor = SETS[name]
    items = make()
    js = _decisions(tmp_path, items, mode)
    rows = R.bar_c_rows(M, N, items, js)
    splits = [(r["id"],) + s for r in rows if not r.get("raises")
              for s in cross_port(r["a"], r["b"], r["ja"], r["jb"])[1] if s[0] in BY_CONSTRUCTION]
    assert splits == [], splits[:10]
    counts, _ids = _bar_c(name, rows, "any", floor)   # no path here reads otherwise under the other flavour
    if name == "newer engine":                    # the patch is live: main's two ports count apart
        apart = sum(1 for r in rows if [(x["verdict"], x["why"]) for x in r["a"]["claims"] if x["kind"] == "files_changed_count"]
                    != [(x["verdict"], x["why"]) for x in r["ja"]["claims"] if x["kind"] == "files_changed_count"])
        assert apart > 300, apart
    if name == "line break":                      # the shape is live: main's two ports decide the symbol claim apart
        assert counts["lists differ, diff side"] > 100, counts


# The port's decisions on #161's reproductions where its main alone counts a `def test_` after U+FEFF, a vertical tab, a
# form feed, U+2028 or U+2029 (P-2 of NOTE_path2a_ninth_pass_2026_10_04). Up to the eighth pass the switch kept the
# port's false tests CONTRADICTEDs here (#101: a changed test counted as added), and no committed test read the port on
# these ids. Per id: the Python's gate verdict and decisions, then the port's.
PORT_161 = {
    "path2:r1-a-bom-on-a-changed-test-does-not-hide-a-new-one":
        ("FAIL", [("CONTRADICTED", None), ("VERIFIED", None)], "FAIL", [("CONTRADICTED", None), ("UNCHECKABLE", "tests")]),
    "path2:r1-a-bom-on-a-changed-test-beside-two-new-ones":
        ("FAIL", [("CONTRADICTED", None), ("VERIFIED", None)], "FAIL", [("CONTRADICTED", None), ("UNCHECKABLE", "tests")]),
    "path2:f2-a-vertical-tab-is-not-a-line-break":
        ("FAIL", [("VERIFIED", None), ("CONTRADICTED", None)], "PASS", [("UNCHECKABLE", "tests"), ("UNCHECKABLE", "tests")]),
    "path2:f2-a-form-feed-is-not-a-line-break":
        ("FAIL", [("VERIFIED", None), ("CONTRADICTED", None)], "PASS", [("UNCHECKABLE", "tests"), ("UNCHECKABLE", "tests")]),
    "path2:f2-a-line-separator-is-not-a-line-break":
        ("FAIL", [("VERIFIED", None), ("CONTRADICTED", None)], "PASS", [("UNCHECKABLE", "tests"), ("UNCHECKABLE", "tests")]),
    "path2:f2-a-paragraph-separator-is-not-a-line-break":
        ("FAIL", [("VERIFIED", None), ("CONTRADICTED", None)], "PASS", [("UNCHECKABLE", "tests"), ("UNCHECKABLE", "tests")]),
    "path2:y2-a-changed-test-beside-a-created-bom-test-under-bare-hunks":
        ("FAIL", [("UNCHECKABLE", "split"), ("UNCHECKABLE", "tests"), ("CONTRADICTED", None)],
         "PASS", [("UNCHECKABLE", "split"), ("UNCHECKABLE", "tests"), ("UNCHECKABLE", "tests")]),
}


def test_the_port_withholds_its_own_false_tests_verdicts_on_161s_reproductions(tmp_path):
    cases = {c["id"]: c for c in R.repro_cases()}
    items = [{"id": k, "summary": cases[k]["summary"], "diff": cases[k]["diff"]} for k in PORT_161]
    js = _decisions(tmp_path, items)
    for k, (gate, want, js_gate, js_want) in PORT_161.items():
        b = N.gate_diff_text(cases[k]["summary"], cases[k]["diff"]).to_dict()
        assert (b["verdict"], [_seen(x) for x in b["claims"]]) == (gate, want), k
        assert (js[k]["new"]["verdict"], [_seen(x) for x in js[k]["new"]["claims"]]) == (js_gate, js_want), k
        assert js[k]["main"]["verdict"] == "FAIL", k


# B-2 (NOTE_path2a_sixth_pass_2026_09_30): #161's joint #121 reproductions. V121's count is false too (a submodule
# line, an hg binary notice or a no-prefix directory main registers no file for), so under the but-for attribution
# they are not misses; the false CONTRADICTED is kept and the right one withheld, in both ports. Operator option O-7
# would withhold them, at the cost measured in the NOTE.
JOINT_121 = {
    "path2:m-121-a-submodule-line-licenses-nothing": ("FAIL", [("UNCHECKABLE", "count"), ("CONTRADICTED", None)]),
    "path2:k4-a-dotted-twin-beside-a-submodule-line":
        ("FAIL", [("CONTRADICTED", None), ("UNCHECKABLE", "count"), ("UNCHECKABLE", "only")]),
    "path2:k4-a-dotted-twin-beside-an-hg-binary-notice": ("FAIL", [("CONTRADICTED", None), ("UNCHECKABLE", "count")]),
    "path2:l-r10-np1-a-no-prefix-b-directory-beside-dotted-twins-abstains":
        ("FAIL", [("UNCHECKABLE", "count"), ("UNCHECKABLE", "count"), ("CONTRADICTED", None)]),
    "path2:l-r10-np2-a-no-prefix-whitespace-twin-beside-dotted-twins-abstains":
        ("FAIL", [("UNCHECKABLE", "count"), ("UNCHECKABLE", "count"), ("CONTRADICTED", None)]),
}


def test_the_joint_121_reproductions_keep_mains_false_contradicted(tmp_path):
    cases = {c["id"]: c for c in R.repro_cases()}
    items = [{"id": k, "summary": cases[k]["summary"], "diff": cases[k]["diff"]} for k in JOINT_121]
    (tmp_path / "in.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    node("--decisions", R.main_port_path(tmp_path), tmp_path / "in.json", tmp_path / "out.json")
    js = {d["id"]: d for d in json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))}
    for k, (gate, want) in JOINT_121.items():
        b = N.gate_diff_text(cases[k]["summary"], cases[k]["diff"]).to_dict()
        assert (b["verdict"], [_seen(x) for x in b["claims"]]) == (gate, want), k
        assert (js[k]["new"]["verdict"], [_seen(x) for x in js[k]["new"]["claims"]]) == (gate, want), k


# ---- tables pinned by enumeration -------------------------------------------------------------------------------------

def _set(s):
    return {ord(ch) for ch in s}


def _neutral():
    """The neutral code points, read off the Python block's regex class."""
    rx = re.compile("[" + N._P2A_NEUTRAL + "]")
    return {cp for cp in range(0x80, 0x10000) if rx.match(chr(cp))}


def _case_table(runs) -> dict:
    """{code point: its lowercase} from (start, end, step, delta) runs."""
    return {cp: cp + d for lo, hi, step, d in runs for cp in range(lo, hi + 1, step)}


def _assert_case_tables(table: dict, never: set, here: dict, version: str):
    """B-1 (NOTE_path2a_eighth_pass_2026_10_01): a runtime's lowercase mappings `here` against the static tables."""
    targets = set(table.values())
    assert not never & (set(table) | targets), "a code point of a block with no case is in the case table"
    assert not set(table) & targets, "a lowercase of the table is also mapped"
    for cp, d in here.items():
        assert cp not in never and d not in never, (hex(cp), hex(d))
        if cp in table:
            assert table[cp] == d, (hex(cp), hex(d), hex(table[cp]))
    if version.startswith("16."):
        assert here == table, sorted(set(here) ^ set(table))[:10]
    elif int(version.split(".")[0]) < 16:
        assert set(here) <= set(table), sorted(set(here) - set(table))[:10]


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
    # C-1 (NOTE_path2a_fifth_pass_2026_09_30 and its corrections): the count seam reads the white space only one port
    # reads, in a run of either port's, and 'file' spelled with a code point CPython's IGNORECASE folds to i or s. The
    # four such code points, the letters each folds to, and that no word of the count holds k, the fourth's letter
    # (the port's IGNORECASE folds none: test_port_tables_and_constants).
    folds = {cp: {x for x in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" if re.match(x, chr(cp), re.I)}
             for cp in range(0x80, 0x110000) if re.match("[A-Za-z]", chr(cp), re.I)}
    assert folds == {0x130: {"i", "I"}, 0x131: {"i", "I"}, 0x17F: {"s", "S"}, 0x212A: {"k", "K"}}
    words = re.search(r"\\s\+(files\?\\s\+\(\?:were\\s\+\)\?changed)", R.lf(R.INSTRUMENT)).group(1)
    assert set(words.lower()) & set("iks") == {"i", "s"}, words
    assert _set(N._P2A_ONE_SPACE) == _set(N._P2A_PY_SPACE) ^ _set(N._P2A_JS_SPACE)
    assert _set(N._P2A_ANY_SPACE) == _set(N._P2A_PY_SPACE) | _set(N._P2A_JS_SPACE)
    for s, want in (("13\x1ffiles changed", True), ("\ufeffCreated a.py.", False), ("3 files\x85changed.", True),
                    ("files\u2028\x1fwere changed", True), ("fixed\x1fwith care", False), ("3\x1f\x1f", False),
                    ("f\u0131les", True),
                    ("File\u017f", True), ("\u212a files changed", False), ("x\x85y", False)):
        assert N._p2a_seam(s) is want, (s, want)
    # O-11 (NOTE_path2a_seventh_pass_2026_09_30, B-1): the five pictograph blocks the summary reads as U+2190. None is a
    # word character, white space in either port, cased, alphanumeric, or folded to an ASCII letter under re.I here,
    # and each is a symbol or unassigned; U+2190 is neutral; the regex reads each as one code point or as its two
    # surrogates, and reads nothing else.
    emoji = set(range(0x1F300, 0x1F650)) | set(range(0x1F680, 0x1F700)) | set(range(0x1F900, 0x1FA00)) | \
        set(range(0x1FA70, 0x1FB00))
    assert all(re.match(r"\w", chr(cp)) is None and not chr(cp).isalnum() and not chr(cp).isspace()
               and chr(cp).lower() == chr(cp).upper() == chr(cp).casefold() == chr(cp)
               and re.match("[A-Za-z_]", chr(cp), re.I) is None
               and unicodedata.category(chr(cp)) in ("So", "Sk", "Cn") for cp in emoji)
    assert not emoji & (_set(N._P2A_PY_SPACE) | _set(N._P2A_JS_SPACE))
    assert ord(N._P2A_EMOJI_AS) in _neutral() and len(N._P2A_EMOJI_AS) == 1

    def halves(cp):
        x = cp - 0x10000
        return chr(0xD800 + (x >> 10)) + chr(0xDC00 + (x & 0x3FF))
    assert {cp for cp in range(0x10000, 0x110000) if N._P2A_EMOJI_RX.fullmatch(chr(cp))} == emoji
    assert {cp for cp in range(0x10000, 0x110000) if N._P2A_EMOJI_RX.fullmatch(halves(cp))} == emoji
    assert not any(N._P2A_EMOJI_RX.search(chr(cp)) for cp in range(0x10000))
    # B-1 (NOTE_path2a_eighth_pass_2026_10_01): the static case tables against this runtime's lowercase. Every mapping
    # from 0x80 up (U+0130 and U+212A aside, which the fold form reads) is to one code point, joins two code points the
    # table gives one class (or one the table does not hold, which a later version may add), and touches no code point of
    # the blocks with no case; on Unicode 16 the table is exactly this runtime's mappings, on an older one it holds them.
    table = _case_table(N._P2A_LOWER_RUNS)
    never = {cp for cp in range(0x80, 0x110000) if N._P2A_NEVER_RX.match(chr(cp))}
    here = {}
    for cp in range(0x80, 0x110000):
        low = chr(cp).lower()
        if cp not in (0x130, 0x212A) and low != chr(cp):
            assert len(low) == 1, hex(cp)
            here[cp] = ord(low)
    _assert_case_tables(table, never, here, unicodedata.unidata_version)
    assert {ord(k): ord(v) for k, v in N._P2A_CASE.items()} == {**table, **{d: d for d in table.values()}}
    # C-2: the digit table reads ASCII digits only
    for x in ("\u30003", "3\u3000", "\uff13", " 3", "3 ", "+3", "0x3", "3e1", "\u0663"):
        with pytest.raises(KeyError):
            N._p2a_int(x)
    assert N._p2a_int("0123456789") == 123456789


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
    # C-1 and C-2 (NOTE_path2a_fifth_pass_2026_09_30)
    assert set(t["one_space"]) == _set(N._P2A_ONE_SPACE)
    # O-11 (NOTE_path2a_seventh_pass_2026_09_30): the pictograph emoji as this engine's strings hold them, two units
    # each: the port's regex reads exactly those of the five blocks, no single unit, and none of them is \w or \s for
    # this engine or has a case here
    emoji = set(range(0x1F300, 0x1F650)) | set(range(0x1F680, 0x1F700)) | set(range(0x1F900, 0x1FA00)) | \
        set(range(0x1FA70, 0x1FB00))
    assert set(t["emoji_matched"]) == emoji and t["emoji_units_matched"] == [] and t["emoji_flagged"] == []
    assert t["emoji_as"] == N._P2A_EMOJI_AS
    assert t["ascii_folds"] == [], "this engine's non-Unicode IGNORECASE folds a code point from 0x80 up to ASCII"
    # B-1 (NOTE_path2a_eighth_pass_2026_10_01): the port's case tables are the Python's, and hold this engine's lowercase
    assert [tuple(x) for x in t["lower_runs"]] == list(N._P2A_LOWER_RUNS)
    never = {cp for a, b in t["never_ranges"] for cp in range(a, b + 1)}
    assert never == {cp for cp in range(0x80, 0x110000) if N._P2A_NEVER_RX.match(chr(cp))}
    assert all(d >= 0 for _cp, d in t["lower_map"]), "this engine lowers a code point from 0x80 up to more than one"
    _assert_case_tables(_case_table(N._P2A_LOWER_RUNS), never, {cp: d for cp, d in t["lower_map"]}, t["unicode"])
    assert all(t["int_rejects"]), "the port's digit table read a string that is not ASCII digits"
