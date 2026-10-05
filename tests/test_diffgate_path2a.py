"""PATH-2a: main's diff gate, unchanged, plus an overlay that only abstains (NOTE_path2a_abstain_overlay_2026_09_30,
NOTE_path2a_second_pass_2026_09_30, NOTE_path2a_third_pass_2026_09_30, NOTE_path2a_fourth_pass_2026_09_30,
NOTE_path2a_fifth_pass_2026_09_30, NOTE_path2a_sixth_pass_2026_09_30, NOTE_path2a_seventh_pass_2026_09_30,
NOTE_path2a_eighth_pass_2026_10_01, NOTE_path2a_ninth_pass_2026_10_04, NOTE_path2a_tenth_pass_2026_10_05).

The reference in every test here is `main` itself: this checkout's `styxx/diffgate.py` and `web/gate/diffgate.js` with
the PATH-2a block cut out and the door hooks reverted, asserted to hash to the files on `origin/main` 1cde8b82
(tests/_p2a_ref.py). No copy of main is committed.

(A) by construction -- on every committed input, both strict modes, both ports and the git door, each record equals
    main's except that a decided verdict in REACH may become UNCHECKABLE with the overlay's reason, and the gate
    verdict is main's formula over the claims; each claim reads the same under --strict as without it; where main
    raises, the branch raises the same exception type. The same with a run leg, a test report and a commit handed to
    both doors, where main decides `tests_pass` claims, which are outside REACH. Since the tenth pass the relation
    holds at run time whatever the rules do: they run on a copy (DECIDE) and one short function writes the record
    (APPLY). Functions written to do harm are handed to APPLY in both ports, the ninth review's plants are planted
    where the decisions are computed, and the text of the lines that touch the record is pinned.
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
And the static facts: the reconstruction, the lints over each block's source (against an honest edit that would ask
the runtime a Unicode question; not proofs, and nothing about what the block may write), the error fallback, the
reproductions, the cost per call, and what must not move (the demo, the committed capsules, charon's lines, the
bookmarklet source). Coverage (B) is in tests/test_diffgate_path2a_truth.py.
"""
from __future__ import annotations

import collections
import copy
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
from contextlib import contextmanager, redirect_stdout

import pytest

import styxx.diffgate as N
from tests import _p2a_ref as R

NODE = shutil.which("node")
GIT = shutil.which("git")
CHECK = R.DIFFERENTIAL / "check_path2a.js"
# A copy of the phrase table taken before any test runs a DECIDE, so that a DECIDE that writes the module's own table
# cannot make the relation accept what it wrote (A-3 of the tenth construction review, NOTE_path2a_eleventh_pass_2026_10_05)
PHRASES = dict(N._P2A_PHRASES)
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


# ---- the lints over each block's source ------------------------------------------------------------------------------
#
# NOTE_path2a_tenth_pass_2026_10_05: these are lints against an honest future edit that would ask the runtime a Unicode
# question, which bar C(i) leans on the block not doing. They are not proofs against a hostile edit, and they say
# nothing about what the block may write: that the overlay only abstains holds at run time, in APPLY (the section
# after the next). The record-write checks of passes one to nine, and their plants, are gone.

def test_the_python_block_passes_its_lints():
    rep = N.selfcheck_p2a_asks_no_runtime()
    assert rep["ok"] is True, rep["problems"]
    assert any("only abstains" in x for x in rep["not_checked"])


CLAIMED = "    ca, ck = _p2a_A(claimed), _p2a_K(claimed)\n"
LINT_PLANTS = [
    ("def _p2a_fold(s: str) -> str:", "def _p2a_fold(s: str) -> str:\n    s = s.lower()", "attribute .lower"),
    ('_P2A_COARSE = re.compile("\\r\\n|\\r|\\n")', '_P2A_COARSE = re.compile("\\\\s+")', "class escape"),
    ("    return s.replace(\"\\\\\", \"/\")", "    return s.replace(\"\\\\\", \"/\").strip()", "without an explicit argument"),
    ("    return s.translate(_P2A_FOLD)\n", "    return s.translate(_P2A_FOLD).casefold()\n", "attribute .casefold"),
    # C-4: calls that read a Unicode table indirectly
    (CLAIMED, "    claimed = repr(claimed)\n" + CLAIMED, "name repr"),
    (CLAIMED, '    claimed = f"{claimed!r}"\n' + CLAIMED, "!r or !a conversion"),
    (CLAIMED, '    claimed = f"{claimed!a}"\n' + CLAIMED, "!r or !a conversion"),
    (CLAIMED, '    claimed = "%r" % (claimed,)\n' + CLAIMED, "%-formatting of a string literal"),
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
    # NOTE_path2a_eleventh_pass_2026_10_05: APPLY tests exact types, so `type` may be called only as type(x) is T or
    # type(x) is not T, T a builtin type, and `int` named only as that T (at the tenth pass: as the type isinstance tests)
    (CLAIMED, "    q = type(claimed) is int and int(claimed)\n" + CLAIMED, "name int"),
    (CLAIMED, "    q = isinstance(claimed, int)\n" + CLAIMED, "name int"),
    (CLAIMED, "    q = type(claimed) is not [int][0]\n" + CLAIMED, "name int"),
    (CLAIMED, "    q = type(claimed)\n" + CLAIMED, "name type"),
    (CLAIMED, "    q = type(claimed) == str\n" + CLAIMED, "name type"),
    (CLAIMED, "    q = type(claimed, (), {})\n" + CLAIMED, "name type"),
    (CLAIMED, "    q = type(claimed) is type(ca)\n" + CLAIMED, "name type"),
    # A-6 of the tenth construction review: '%s' of a container reads repr as %r does
    (CLAIMED, '    claimed = "%s" % ([claimed],)\n' + CLAIMED, "%-formatting of a string literal"),
    (CLAIMED, '    claimed = "%s/%s" % (claimed, claimed)\n' + CLAIMED, "%-formatting of a string literal"),
]


@pytest.mark.parametrize("old,new,what", LINT_PLANTS)
def test_the_python_lints_refuse_a_planted_read(old, new, what):
    text = R.lf(R.INSTRUMENT)
    block = R.py_block(text)
    assert block.count(old) == 1, old
    rep = N.selfcheck_p2a_asks_no_runtime(text.replace(block, block.replace(old, new)))
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
                   # Object and Proxy reach a method or a field by a computed name
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
JS_INDEXES = {"0", "1", "2", "i", "k", "k+1", "k-1", "v", "u", "space", "c.kind", "_P2A_OWN", "out.length-1",
              "key", "fallback"}      # the last two: APPLY's lookups in the phrase table
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


JS_CODE_ESCAPE = re.compile(r"\\u\{([0-9A-Fa-f]{1,6})\}|\\u([0-9A-Fa-f]{4})")


def js_bitwise_problems(dense: str) -> list:
    """C-6 of the ninth cross-port review: `~`, `|`, `&`, `^`, `<<` and `>>` convert a string operand with the engine's
    white space, as a unary `+` does. The block uses none of them but `>>` on a `.length`, a number."""
    out = []
    for m in re.finditer(r"~|<<|>>>?|\^|(?<![|&])[|&](?![|&])", dense):
        if m.group().startswith(">>") and dense[:m.start()].endswith(".length"):
            continue
        out.append(f"bitwise operator {m.group()!r}")
    return out


def _js_skip_string(src: str, i: int) -> int:
    """The index after the string literal that opens at src[i] (a quote or a backtick, whose `${...}` it steps over)."""
    q, j = src[i], i + 1
    while src[j] != q:
        if src[j] == BS:
            j += 2
        elif q == "`" and src.startswith("${", j):
            j = _js_close(src, j + 2) + 1
        else:
            j += 1
    return j + 1


def _js_close(src: str, i: int) -> int:
    """The index of the `}` that closes a template substitution whose code starts at src[i]."""
    depth = 1
    while True:
        ch = src[i]
        if ch in "\"'`":
            i = _js_skip_string(src, i)
            continue
        depth += {"{": 1, "}": -1}.get(ch, 0)
        if depth == 0:
            return i
        i += 1


def js_lex(src: str, strings: list) -> str:
    """The block's code outside comments, each string literal written S<n> with its body in `strings`. A template
    literal is written T<n> for each run of its text, and each of its `${...}` substitutions is read as code, in
    brackets after a `+` (NOTE_path2a_eleventh_pass_2026_10_05, C-3 and I-4 of the tenth reviews: up to the tenth pass
    a template was read whole as a string, so code inside `${...}` met no rule)."""
    code, i = [], 0
    while i < len(src):
        ch = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = len(src) if j < 0 else j
            continue
        if src.startswith("/*", i):
            i = src.index("*/", i) + 2
            continue
        if ch in "\"'":
            j = _js_skip_string(src, i)
            strings.append(src[i + 1:j - 1])
            code.append("S%d" % (len(strings) - 1))
            i = j
            continue
        if ch == "`":
            j = start = i + 1
            while src[j] != "`":
                if src[j] == BS:
                    j += 2
                elif src.startswith("${", j):
                    strings.append(src[start:j])
                    end = _js_close(src, j + 2)
                    code.append("T%d+(" % (len(strings) - 1) + js_lex(src[j + 2:end], strings) + ")+")
                    j = start = end + 1
                else:
                    j += 1
            strings.append(src[start:j])
            code.append("T%d" % (len(strings) - 1))
            i = j + 1
            continue
        code.append(ch)
        i += 1
    return "".join(code)


def js_problems(block: str) -> list:
    """The port block's token scan (NOTE_path2a_third_pass_2026_09_30, C-3), a lint. Code is read outside strings and
    comments, with a Unicode escape read as the character it spells (an escape in code can only be part of a name:
    NOTE_path2a_tenth_pass_2026_10_05) and white space removed except between two identifier characters, so
    `a . b (` reads `a.b(`, and a template's `${...}` read as code (js_lex): banned calls anywhere; no regex literal (no '/' in code); `RegExp` only as
    `new RegExp(<one static string>)` -- string literals and top-level constants joined by `+` -- whose decoded value
    holds no class escape, no '.' outside a class, and no flags; a computed member access only with a listed index
    expression and no string in the brackets, and never called; no call on a parenthesised expression; no bitwise
    operator. It reads names, so it does not see a name built at run time, and it types nothing."""
    out = [f"banned token {t!r}" for t in JS_BANNED if t in block]
    strings = []
    raw = JS_CODE_ESCAPE.sub(lambda m: chr(min(int(m.group(1) or m.group(2), 16), 0x10FFFF)), js_lex(block, strings))
    if "\\" in raw:
        out.append("a backslash in code outside a string")
    word = re.compile(r"[A-Za-z0-9_$]")
    dense = re.sub(r"\s+", lambda m: " " if (0 < m.start() and m.end() < len(raw) and word.match(raw[m.start() - 1])
                                                and word.match(raw[m.end()])) else "", raw)
    out += [f"banned token {t!r} in code" for t in JS_BANNED if t in dense]
    out += [f"banned word {t!r} in code" for t in JS_BANNED_WORDS
            if re.search(r"(?<![A-Za-z0-9_$])" + t + r"(?![A-Za-z0-9_$])", dense)]
    out += js_numeric_problems(dense)
    out += js_bitwise_problems(dense)
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


@pytest.mark.parametrize("new,what", [
    # A-2 of the ninth construction review (NOTE_path2a_tenth_pass_2026_10_05): a Unicode escape inside an identifier
    # hid the name from every rule of the scan at c69b161b; the scan reads the name the escape spells
    ("  const ca = _p2aA(claimed.toLowerC" + BS + "u0061se()), ck = _p2aK(claimed);", "banned token 'toLowerCase' in code"),
    ("  const ca = _p2aA(claimed.n" + BS + 'u{6f}rmalize("NFKC")), ck = _p2aK(claimed);', "banned token 'normalize' in code"),
    ("  const ca = _p2aA(claimed." + BS + "u0074rim()), ck = _p2aK(claimed);", "banned token '.trim' in code"),
    (CA.rstrip(";") + "; const q = " + BS + "u004eumber(claimed);", "banned word 'Number'"),
    (CA.rstrip(";") + "; const q = " + BS + "u{4f}bject.keys(claimed);", "banned word 'Object'"),
    ("  const ca = _p2aA(claimed.to" + BS + "x4cowerCase()), ck = _p2aK(claimed);", "a backslash in code outside a string"),
    # the name rules read a banned name whatever the form of the call
    ("  const ca = _p2aA(claimed.toLowerCase?.()), ck = _p2aK(claimed);", "banned token 'toLowerCase' in code"),
    ("  const ca = _p2aA(claimed?.trim()), ck = _p2aK(claimed);", "banned token '.trim' in code"),
    # C-6 of the ninth cross-port review: operators that convert a string as a unary + does
    (CA.rstrip(";") + '; const q = ~~claimed ? claimed : "";', "bitwise operator '~'"),
    (CA.rstrip(";") + "; const q = (claimed | 0);", "bitwise operator '|'"),
    (CA.rstrip(";") + "; const q = claimed & 1;", "bitwise operator '&'"),
    (CA.rstrip(";") + "; const q = claimed ^ 1;", "bitwise operator '^'"),
    (CA.rstrip(";") + "; const q = claimed << 1;", "bitwise operator '<<'"),
    (CA.rstrip(";") + "; const q = claimed >> 1;", "bitwise operator '>>'"),
    (CA.rstrip(";") + "; const q = claimed >>> 1;", "bitwise operator '>>>'"),
])
def test_the_token_scan_reads_escaped_names_and_refuses_bitwise_operators(new, what):
    block = R.js_block()
    assert block.count(CA) == 1
    problems = js_problems(block.replace(CA, new))
    assert any(what in p for p in problems), problems


@pytest.mark.parametrize("new,what", [
    # C-3 and I-4 of the tenth reviews (NOTE_path2a_eleventh_pass_2026_10_05): code inside a template's `${...}`, each
    # of which the tenth pass's scan passed and refused outside a template
    (CA.rstrip(";") + "; const q = `${Number(claimed)}`;", "banned word 'Number'"),
    (CA.rstrip(";") + "; const q = `n=${parseInt(claimed, 10)}`;", "banned word 'parseInt'"),
    (CA.rstrip(";") + "; const q = `${+claimed}`;", "unary + before"),
    (CA.rstrip(";") + "; const q = `${claimed == 3}`;", "loose equality"),
    (CA.rstrip(";") + "; const q = `${Math.max(claimed, 0)}`;", "banned word 'Math'"),
    (CA.rstrip(";") + "; const q = `${claimed | 0}`;", "bitwise operator '|'"),
    (CA.rstrip(";") + '; const k2 = "toLower" + "Case"; const q = `${claimed[k2]()}`;',
     "computed member access indexed by 'k2'"),
    (CA.rstrip(";") + "; const q = `${Object.keys(claimed)}`;", "banned word 'Object'"),
    (CA.rstrip(";") + "; const q = `${claimed.split(/x/).length}`;", "a '/' in code"),
    (CA.rstrip(";") + "; const q = `a${`b${Number(claimed)}`}c`;", "banned word 'Number'"),
    (CA.rstrip(";") + '; const q = `a${claimed ? "}" : `${-claimed}`}`;', "unary - before"),
])
def test_the_token_scan_reads_code_inside_a_template(new, what):
    block = R.js_block()
    assert block.count(CA) == 1
    problems = js_problems(block.replace(CA, new))
    assert any(what in p for p in problems), problems


def test_the_token_scan_reads_the_blocks_own_template_as_code():
    """APPLY's reason is the block's one template: its four substitutions are read as code, and pass."""
    strings = []
    code = js_lex("x = `${c.verdict} withheld (${tag}): ${phrase}. main's: ${c.why}`;", strings)
    assert code == "x = T0+(c.verdict)+T1+(tag)+T2+(phrase)+T3+(c.why)+T4;", code
    assert strings == ["", " withheld (", "): ", ". main's: ", ""]


def test_the_token_scan_passes_an_increment_of_a_string():
    """The disclosed limit, beside the binary operators above: `++` and `--` convert a string too, and the scan cannot
    tell a string from a counter. Pinned so the README's sentence stays true."""
    block = R.js_block()
    assert block.count(CA) == 1
    assert js_problems(block.replace(CA, CA.rstrip(";") + "; let t = claimed; t++;")) == []


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


# ---- bar A at run time: DECIDE and APPLY (NOTE_path2a_tenth_pass_2026_10_05) ---------------------------------------
#
# The rules run on a copy (DECIDE) and one short function writes the record (APPLY), so the abstain-only relation does
# not rest on what the rules do: it holds for ANY function handed to APPLY as `decide`. The tests below hand it
# functions written to do harm, in both ports; plant the ninth construction review's edits where the decisions are
# computed; pin the text of the few lines that do touch the record; and check that APPLY takes every decision the
# block's own DECIDE returns, since a decision it refused would be a silent miss.

APPLY_SHA = {
    "python": "803c83e2f3a386265d5b203ea36ef3b0b7b8a58ed8fec8596fa4a0c43ac2d892",
    "port": "2ee400057779b16bf84996fca6f19a26c42c9452ddf2037f8c281833eaec2143",
}


def _apply_text(port: str) -> str:
    """The lines of a block that build what DECIDE is handed or hold main's live record: in Python the copy's two
    classes, APPLY and the doors' call of it; in the port APPLY's tools taken at load, APPLY, and down to the block's
    end (the `gateDiffText` that calls main and then the overlay). NOTE_path2a_eleventh_pass_2026_10_05 widened both
    to take in the copy's classes and APPLY's tools."""
    if port == "python":
        block = R.py_block()
        return block[block.index("class _P2aClaim:"):block.index("_P2A_MARK = ")]
    block = R.js_block()
    return block[block.index("const _p2aIsArray = "):]


@pytest.mark.parametrize("port", ["python", "port"])
def test_the_code_that_touches_the_record_is_the_pinned_text(port):
    """APPLY is not covered by the construction: it is the construction. An edit of it, of `_p2a_abstain` or of the
    port's `gateDiffText` fails here by name (the byte pins of the whole file fail on a comment anywhere, so they do
    not single such an edit out). To move the pin, read the edit as an edit of the one function that writes main's
    record, then set the hash this failure prints."""
    text = _apply_text(port)
    assert R.sha(text) == APPLY_SHA[port], f"the code that touches main's record moved ({port}): {R.sha(text)}"
    if port == "python":
        src = R.lf(R.INSTRUMENT)
        block = R.py_block(src)
        # the record reaches the block through the two hooks only, and APPLY through _p2a_abstain only
        assert src.count("_p2a_abstain(") == 3 and block.count("_p2a_abstain(") == 1
        assert src.count("_p2a_apply(") == 2 and block.count("_p2a_apply(") == 2
    else:
        block = R.js_block()
        assert block.count("_p2aApply(") == 2 and block.count("_p2aAbstain(") == 2
        assert block.count("_gateDiffTextMain(") == 2          # the door's two calls of main's port
    # 80 lines since NOTE_path2a_eleventh_pass_2026_10_05 (75 before): the port's APPLY reads by index and fills its
    # arrays before DECIDE runs, so that nothing after DECIDE calls an inherited method
    assert text.count("\n") <= 80, "the code that touches the record no longer reads on one screen"


ALL_TAGS = ("#97", "#121", "#97, #121", "#101")


def _wreck(seen):
    """Change, empty and grow everything DECIDE is given (a copy has no `text` slot; trying to add one is part of it)."""
    for c in seen.claims:
        c.verdict, c.kind, c.why = "CONTRADICTED", "tests_pass", "rewritten"
        try:
            c.text = "rewritten"
        except AttributeError:
            pass
        c.detail.clear()
        c.detail["path"] = "x"
        c.detail = None
    seen.claims.clear()
    seen.claims.append(N.DiffClaim("files_changed_count", "", {}, "VERIFIED", ""))
    seen.claims = None
    return []


def _honest(summary, diff):
    return lambda seen: N._p2a_decisions(seen, lambda: N._P2aFacts(diff or "", None, summary))


def _each(seen, pick, make):
    return [d for i, c in enumerate(seen.claims) if pick(c) for d in make(i, c)]


def _in_reach(c):
    return (c.kind, c.verdict) in R.REACH


def _flipped(summary, diff):
    def decide(seen):
        for c in seen.claims:
            c.verdict = "CONTRADICTED" if c.verdict == "VERIFIED" else "VERIFIED"
        return _honest(summary, diff)(seen)
    return decide


def _raiser(exc, wreck=False):
    def decide(seen):
        if wreck:
            _wreck(seen)
        raise exc
    return decide


class _ListThatRaises(list):
    def __iter__(self):
        raise RuntimeError("planted")


class _TupleThatLies(tuple):
    def __len__(self):
        return 3

    def __iter__(self):
        return iter((0, "dir"))                 # two values where three are unpacked


class _KeyWhoseHashRaises(str):
    def __hash__(self):
        raise RuntimeError("planted")


class _IndexEqualToAll(int):
    def __hash__(self):
        return 0

    def __eq__(self, other):
        return True


def _generator():
    yield (0, "unreproduced", "#97")


# NOTE_path2a_eleventh_pass_2026_10_05: APPLY takes exact types, so a subclass of list, tuple, int or str is not taken,
# whatever it holds
class _ListOfDecisions(list):
    pass


class _DecisionTuple(tuple):
    pass


class _Index(int):
    pass


class _Text(str):
    pass


def _subclassed(s, d):
    """The block's own decisions, each given four times, with its tuple, its index, its key or its tag a subclass."""
    return lambda seen: [x for i, k, t in _honest(s, d)(seen)
                         for x in (_DecisionTuple((i, k, t)), (_Index(i), k, t), (i, _Text(k), t), (i, k, _Text(t)))]


# The tenth reviews' blocker (NOTE_path2a_eleventh_pass_2026_10_05, A-1, B-1, C-1, I-1): a DECIDE that uses only its
# argument and patches the class of its copy. At 16daa725 that class was main's DiffClaim, the class of every claim in
# the record, so APPLY's own stores ran the patch on the record; now it is the block's own _P2aClaim. The patches stay
# for the rest of the test (later copies are built under them) and _classes_restored undoes them.
def _set_verdict_property(seen):
    type(seen.claims[0]).verdict = property(lambda c: "VERIFIED", lambda c, v: None)


def _set_setattr(seen):
    type(seen.claims[0]).__setattr__ = lambda c, k, v: object.__setattr__(
        c, k, "VERIFIED" if (k == "verdict" and v == "UNCHECKABLE") else ("any text DECIDE likes" if k == "why" else v))


def _set_swallowed_reason(seen):
    type(seen.claims[0]).why = property(lambda c: "", lambda c, v: None)


def _set_container_setattr(seen):
    type(seen).__setattr__ = lambda o, k, v: object.__setattr__(o, k, v[:1] if k == "claims" else v)


CLASS_ATTACKS = {
    "a property on verdict of its claims' class": _set_verdict_property,
    "a __setattr__ on its claims' class": _set_setattr,
    "a property on its claims' class that swallows the reason": _set_swallowed_reason,
    "a __setattr__ on its container's class": _set_container_setattr,
}


def _class_attack(attack, every=False):
    """A DECIDE that runs one of the attacks above on its copy, then returns a decision for every claim (`every`) or
    the block's own decisions."""
    def make(s, d):
        def decide(seen):
            attack(seen)
            if every:
                return [(i, "unreproduced", N._P2A_KIND_DEFECT.get(c.kind, "#97")) for i, c in enumerate(seen.claims)]
            return _honest(s, d)(seen)
        return decide
    return make


# name -> (what the record must then be, the phrase where every claim in reach is withheld, a maker of `decide`).
# "same": main's record, untouched (APPLY ignored everything); "all": every claim in reach withheld, with the phrase;
# "relation": inside the abstain-only relation, and no more is said.
HOSTILE = {
    "the block's own DECIDE": ("relation", None, _honest),
    "changes, empties and grows its copy, decides nothing": ("same", None, lambda s, d: _wreck),
    "decides, then changes its copy": ("relation", None, lambda s, d: lambda seen: (_honest(s, d)(seen), _wreck(seen))[0]),
    "flips every verdict of its copy, then decides": ("relation", None, _flipped),
    "returns nothing": ("all", "malformed", lambda s, d: lambda seen: None),
    "returns a number": ("all", "malformed", lambda s, d: lambda seen: 7),
    "returns a string": ("all", "malformed", lambda s, d: lambda seen: "dir"),
    "returns a mapping": ("all", "malformed", lambda s, d: lambda seen: {0: (0, "dir", "#97")}),
    "returns a tuple, not a list": ("all", "malformed", lambda s, d: lambda seen: ((0, "unreproduced", "#97"),)),
    "returns a generator": ("all", "malformed", lambda s, d: lambda seen: _generator()),
    "returns its own argument": ("all", "malformed", lambda s, d: lambda seen: seen),
    "returns a list of junk": ("same", None, lambda s, d: lambda seen: [
        None, 7, "x", {}, [], (), (0,), (0, "dir"), (0, "dir", "#97", 1), [0, "dir", "#97"], ((0,), "dir", "#97"),
        ("0", "dir", "#97"), (0, 1, 2), (0, "dir", 97), (0, ["dir"], "#97"), (0, "dir", ["#97"]), seen, seen.claims,
        (seen.claims[0], "dir", "#97")]),
    "returns the claims of its copy": ("same", None, lambda s, d: lambda seen: seen.claims),
    "indices out of range, negative, fractional, boolean and as strings": ("same", None, lambda s, d: lambda seen: [
        (i, "unreproduced", t) for i in (-1, len(seen.claims), len(seen.claims) + 1, 10 ** 9, 0.5, 0.0, 1.0,
                                         float("nan"), float("inf"), True, False, "0", "1", None, (0,))
        for t in ALL_TAGS]),
    # NOTE_path2a_eleventh_pass_2026_10_05: `error` is APPLY's own and ignored, so the second phrase stands
    "one index twice, with two phrases": ("all", "unparsed", lambda s, d: lambda seen: _each(
        seen, _in_reach, lambda i, c: [(i, k, N._P2A_KIND_DEFECT[c.kind]) for k in ("error", "unparsed", "dir")])),
    "phrases outside the fixed set": ("same", None, lambda s, d: lambda seen: _each(
        seen, lambda c: True, lambda i, c: [(i, k, N._P2A_KIND_DEFECT.get(c.kind, "#97"))
                                            for k in ("nope", "", "DIR", " dir", "error ", "__class__", "get")])),
    "tags outside the fixed sets, and another kind's tag": ("same", None, lambda s, d: lambda seen: _each(
        seen, lambda c: True, lambda i, c: [(i, "unreproduced", x) for x in ("#1", "", "#97,#121", "#121, #97", " #97", "97")
                                            + tuple(x for x in ALL_TAGS if x not in R.DEFECTS.get(c.kind, ()))])),
    "decisions for every claim outside reach": ("same", None, lambda s, d: lambda seen: _each(
        seen, lambda c: not _in_reach(c), lambda i, c: [(i, "unreproduced", x) for x in ALL_TAGS])),
    "a decision for every index, with every tag": ("all", "unreproduced", lambda s, d: lambda seen: _each(
        seen, lambda c: True, lambda i, c: [(i, "unreproduced", x) for x in ALL_TAGS])),
    "raises": ("all", "error", lambda s, d: _raiser(RuntimeError("planted"))),
    "raises a KeyError": ("all", "error", lambda s, d: _raiser(KeyError("dir"))),
    "changes its copy, then raises": ("all", "error", lambda s, d: _raiser(TypeError("planted"), wreck=True)),
    # NOTE_path2a_eleventh_pass_2026_10_05: exact types. A list subclass is not a list (`malformed`); a tuple, index or
    # key of a subclass is ignored, so no method of it runs in APPLY. At the tenth pass these four were read through
    # isinstance and fell back with `error`, or (the last) were taken.
    "a list that raises when it is read": ("all", "malformed", lambda s, d: lambda seen: _ListThatRaises(
        [(0, "unreproduced", "#97")])),
    "a tuple that lies about its length": ("same", None, lambda s, d: lambda seen: [_TupleThatLies((0, "dir", "#97", 4))]),
    "a phrase key whose hash raises": ("same", None, lambda s, d: lambda seen: _each(
        seen, _in_reach, lambda i, c: [(i, _KeyWhoseHashRaises("dir"), N._P2A_KIND_DEFECT[c.kind])])),
    "an index that says it equals every index": ("same", None, lambda s, d: lambda seen: [
        (_IndexEqualToAll(5), "unreproduced", x) for x in ALL_TAGS]),
    "a list subclass of the block's own decisions": ("all", "malformed", lambda s, d: lambda seen: _ListOfDecisions(
        _honest(s, d)(seen))),
    "the block's own decisions with a subclassed tuple, index, key or tag": ("same", None, _subclassed),
    # A-4 of the tenth construction review: `error` and `malformed` are APPLY's own and never taken from DECIDE
    "only the phrases APPLY keeps for its own fallbacks": ("same", None, lambda s, d: lambda seen: _each(
        seen, _in_reach, lambda i, c: [(i, k, N._P2A_KIND_DEFECT[c.kind]) for k in ("error", "malformed")])),
    # the tenth reviews' blocker: the class of the copy patched through the copy itself
    "a property on verdict of its claims' class, then a decision for every index": (
        "all", "unreproduced", _class_attack(_set_verdict_property, every=True)),
    "a __setattr__ on its claims' class, then the block's own decisions": (
        "relation", None, _class_attack(_set_setattr)),
    "a property on its claims' class that swallows the reason, then the block's own decisions": (
        "relation", None, _class_attack(_set_swallowed_reason)),
    "a __setattr__ on its container's class, then a decision for every index": (
        "relation", None, _class_attack(_set_container_setattr, every=True)),
}



def _hostile_inputs() -> list:
    """#161's reproductions; every other one with a sentence main reads as a `tests_pass` claim, a kind the overlay has
    no tag for, so that an APPLY that looked a decision up for a claim outside reach would be seen (a mutation check
    of this pass found that, without such a claim, the test could not tell that edit from the head)."""
    return [(c["id"], c["summary"] + (" All tests pass." if n % 2 else ""), c["diff"])
            for n, c in enumerate(R.repro_cases())]


def _own(g0):
    """main's record built of this module's own classes, as the doors hand it to APPLY. The tenth pass's hostile test
    handed APPLY records of the reference module's classes, so a DECIDE that patched the class of its copy (then this
    module's DiffClaim) could not reach them there, while it moved the record at both doors (the tenth reviews)."""
    g = N.DiffGate(**vars(copy.deepcopy(g0)))
    g.claims = [N.DiffClaim(**vars(c)) for c in g.claims]
    return g


@contextmanager
def _classes_restored():
    """Undo whatever a DECIDE of a test did to the copy's classes and to main's two record classes."""
    classes = (N._P2aClaim, N._P2aSeen, N.DiffClaim, N.DiffGate)
    saved = [dict(vars(cls)) for cls in classes]
    try:
        yield
    finally:
        for cls, was in zip(classes, saved):
            for k in [k for k in vars(cls) if k not in was]:
                delattr(cls, k)
            for k, v in was.items():
                if vars(cls).get(k) is not v and k not in ("__dict__", "__weakref__"):
                    setattr(cls, k, v)
        assert [dict(vars(cls)) for cls in classes] == saved


@pytest.fixture(scope="module")
def mains(M):
    """main's live record for each of those inputs, in both strict modes, built of this module's own classes, and the
    same as a dict."""
    out = []
    for _id, summary, diff in _hostile_inputs():
        for strict in (False, True):
            try:
                g = M.gate_diff_text(summary, diff, strict=strict)
            except Exception:
                continue
            out.append((summary, diff, strict, _own(g), g.to_dict()))
    assert sum(any(c.kind == "tests_pass" for c in x[3].claims) for x in out) > 400
    return out


@pytest.mark.parametrize("name", sorted(HOSTILE))
def test_a_hostile_decide_cannot_leave_the_relation(name, mains):
    """Bar A by construction at run time, in Python: APPLY against a DECIDE written to do harm. Whatever it does to its
    copy, returns or raises, the record is main's but for abstentions in reach with a reason of the fixed form, the
    gate verdict is main's formula, and APPLY returns the record it was given."""
    want, phrase, make = HOSTILE[name]
    runs = same = in_reach = withheld = 0
    with _classes_restored():
        for summary, diff, strict, g0, a in mains:
            g = copy.deepcopy(g0)
            ret = N._p2a_apply(g, strict, make(summary, diff))
            b = g.to_dict()
            assert ret is g, "APPLY returns the record it was given"
            assert R.relation(a, b, strict, PHRASES, allow_error=True) == [], (name, summary[:60])
            runs += 1
            same += a == b
            for x, y in zip(a["claims"], b["claims"]):
                if (x["kind"], x["verdict"]) in R.REACH:
                    in_reach += 1
                    if y["verdict"] != x["verdict"]:
                        withheld += 1
                        assert phrase is None or R.phrase_key(y["why"], PHRASES) == phrase, (name, y["why"])
            if name == "the block's own DECIDE":
                assert b == N.gate_diff_text(summary, diff, strict=strict).to_dict()
    assert runs > 900 and in_reach > 1000, (runs, in_reach)
    assert {"same": same == runs, "all": withheld == in_reach, "relation": withheld > 0}[want], (same, withheld, in_reach)


def test_what_apply_does_not_catch_leaves_the_record_as_main_made_it(mains):
    """APPLY's `try` is `except Exception`, as the overlay's always was: a KeyboardInterrupt in DECIDE goes up, and
    nothing has been written when it does."""
    def interrupt(seen):
        _wreck(seen)
        raise KeyboardInterrupt

    seen_reach = 0
    for _summary, _diff, strict, g0, a in mains[:200]:
        g = copy.deepcopy(g0)
        if not any((c.kind, c.verdict) in R.REACH for c in g.claims):
            assert N._p2a_apply(g, strict, interrupt) is g      # nothing in reach: DECIDE is not called at all
        else:
            seen_reach += 1
            with pytest.raises(KeyboardInterrupt):
                N._p2a_apply(g, strict, interrupt)
        assert g.to_dict() == a
    assert seen_reach > 50


ORIGINAL_DECIDE = N._p2a_decisions


def _git_door_inputs() -> list:
    """The reproductions that carry their own `--name-status`, every other one with a `tests_pass` sentence."""
    cases = [c for c in R.repro_cases() if c.get("name_status")]
    return [(c["summary"] + (" All tests pass." if n % 2 else ""), c["diff"], c["name_status"]) for n, c in enumerate(cases)]


def _door_runs(M, monkeypatch, decide):
    """Both doors with this module's DECIDE replaced by `decide(seen, facts)`, both strict modes, against main: the
    records outside the relation, the runs, and the claims withheld."""
    monkeypatch.setattr(N, "_p2a_decisions", decide)
    broken, runs, withheld = [], 0, 0
    doors = [("raw", s, d, None) for _i, s, d in _hostile_inputs()] + [("git", s, d, ns) for s, d, ns in _git_door_inputs()]
    for door, summary, diff, ns in doors:
        if door == "git":
            fake = R.fake_git(ns, diff)
            monkeypatch.setattr(M, "_git", fake)
            monkeypatch.setattr(N, "_git", fake)
        for strict in (False, True):
            run = ((lambda m: m.gate_diff_text(summary, diff, strict=strict)) if door == "raw" else
                   (lambda m: m.gate_diff(summary, "(repo)", "base", "head", strict=strict)))
            try:
                a = run(M).to_dict()
            except Exception:
                continue
            b = run(N).to_dict()
            runs += 1
            bad = R.relation(a, b, strict, PHRASES, allow_error=True)
            if bad:
                broken.append((door, strict, summary[:50], bad[:2]))
            withheld += sum(x["verdict"] != y["verdict"] for x, y in zip(a["claims"], b["claims"]))
    monkeypatch.setattr(N, "_p2a_decisions", ORIGINAL_DECIDE)
    return broken, runs, withheld


@pytest.mark.parametrize("name", sorted(CLASS_ATTACKS))
def test_a_decide_that_patches_its_copys_class_cannot_move_the_record_at_either_door(name, M, monkeypatch):
    """The tenth reviews' blocker at the doors, on records of this module's own classes: the module's DECIDE replaced
    by one that patches the class of its copy (reached by type()), then decides as the block does. At 16daa725 the
    copy's class was main's DiffClaim, and the reviewers counted 580 of 958 raw-door runs and 42 of 192 git-door runs
    outside the relation for a __setattr__ like the one here. The patch is left in place for the next call, as a
    DECIDE that did this would leave it: that call must give main's record where nothing is in reach, and stay inside
    the relation where something is."""
    attack = CLASS_ATTACKS[name]
    with _classes_restored():
        broken, runs, withheld = _door_runs(M, monkeypatch, lambda seen, facts: (attack(seen), ORIGINAL_DECIDE(seen, facts))[1])
        assert broken == [], (name, len(broken), broken[:3])
        assert runs > 1100 and withheld > 0, (runs, withheld)
        # the next call, the class still patched and the module's own DECIDE back
        summary = "Updated src/app.py. All tests pass."          # main reads it unmeasured: nothing in reach
        assert N.gate_diff_text(summary, "").to_dict() == M.gate_diff_text(summary, "").to_dict()
        one = D_TWINS_HEAD.split("diff --git a/.env")[0]
        for summary, diff in [("Updated src/app.py. 1 files changed.", one)] + [
                (c["summary"], c["diff"]) for c in R.repro_cases()[:60]]:
            a = M.gate_diff_text(summary, diff).to_dict()
            assert R.relation(a, N.gate_diff_text(summary, diff).to_dict(), False, PHRASES, allow_error=True) == [], summary


def test_a_decide_that_reaches_the_module_through_facts_is_the_stated_limit(M, monkeypatch):
    """The limit README and NOTE_path2a_eleventh_pass_2026_10_05 state: bar A by construction covers a DECIDE that
    uses what it is handed as data and calls the `facts` it is given, not one that reaches around that by reflection.
    `facts.__globals__` is this module; a DECIDE that takes main's DiffClaim from there and patches it moves the record,
    and nothing APPLY does can stop it. Asserted to leave the relation, so that a change that closes this route is
    noticed: then this test fails, and the README's sentence on what bar A does not cover moves with it."""
    def decide(seen, facts):
        facts.__globals__["DiffClaim"].__setattr__ = lambda c, k, v: object.__setattr__(
            c, k, "VERIFIED" if (k == "verdict" and v == "UNCHECKABLE") else v)
        return ORIGINAL_DECIDE(seen, facts)

    with _classes_restored():
        broken, runs, _withheld = _door_runs(M, monkeypatch, decide)
    assert runs > 1100 and len(broken) > 100, (runs, len(broken))
    assert N.gate_diff_text("Updated src/app.py. All tests pass.", "").to_dict() == \
        M.gate_diff_text("Updated src/app.py. All tests pass.", "").to_dict()


def test_a_hostile_decide_cannot_leave_the_relation_port(work, tmp_path):
    """The same in the port: check_path2a.js --hostile hands _p2aApply main's record and each DECIDE of its own list
    (ones that change their copy, return junk, a Proxy that answers differently on each read, or throw; and, with the
    record built by the port's own main in the realm DECIDE runs in, ones that patch what every object or array
    inherits: NOTE_path2a_eleventh_pass_2026_10_05)."""
    items = [{"id": i, "summary": s, "diff": d} for i, s, d in _hostile_inputs()]
    (tmp_path / "in.json").write_text(json.dumps(items, ensure_ascii=True), encoding="utf-8")
    node("--hostile", work / "diffgate_main_reference.js", tmp_path / "in.json", tmp_path / "hostile.json")
    out = json.loads((tmp_path / "hostile.json").read_text(encoding="utf-8"))
    assert len(out) >= 30 and {v["want"] for v in out.values()} == {"same", "all", "relation"}
    assert sum(bool(v.get("realm")) for v in out.values()) >= 5
    for name, v in out.items():
        c = v["counts"]
        assert c["broken"] == 0 and c["not_returned"] == 0 and c["threw"] == 0, (name, c, v["examples"])
        assert c["runs"] > 900 and c["in_reach"] > 1000, (name, c)
        assert {"same": c["same"] == c["runs"], "all": c["withheld"] == c["in_reach"],
                "relation": c["withheld"] > 0}[v["want"]], (name, c)
        if v["phrase"]:
            assert v["phrases"] == {v["phrase"]: c["in_reach"]}, (name, v["phrases"])
    own = out["the block's own DECIDE"]
    assert own["counts"]["unlike_the_port"] == 0 and own["counts"]["decisions"] == own["counts"]["withheld"]
    assert not set(own["phrases"]) & set(R.FALLBACKS), own["phrases"]


def test_apply_takes_every_decision_the_blocks_decide_returns(M, inputs):
    """APPLY ignores a decision it does not take, which would leave main's verdict standing in silence. On the
    committed inputs the block's own DECIDE returns one decision per claim it withholds, each for a claim in reach,
    and APPLY takes them all."""
    n = 0
    for _sname, iid, summary, diff in inputs:
        try:
            g = M.gate_diff_text(summary, diff)
        except Exception:
            continue
        log = []

        def spy(seen, log=log, s=summary, d=diff):
            log.append(_honest(s, d)(seen))
            return log[-1]

        before = [c.verdict for c in g.claims]
        N._p2a_apply(g, False, spy)
        for out in log:
            assert all(type(x) is tuple and len(x) == 3 for x in out), iid
            assert [i for i, (v, c) in enumerate(zip(before, g.claims)) if c.verdict != v] == [x[0] for x in out], iid
            n += len(out)
    assert n > 2500, n


def _decision_site(block: str, python: bool) -> tuple:
    """Where a block computes its decisions: the one line that builds the facts, and the claims the function it lies
    in is given, as that function's own parameter spells them. At c69b161b that function was `_p2a_abstain` and those
    claims were main's record; here it is DECIDE and they are a copy."""
    found = list(re.finditer(r"^( +)f = facts\(\)\n" if python else r"^( +)const f = facts\(\);\n", block, re.M))
    assert len(found) == 1, "the block builds its facts on one line"
    heads = list(re.finditer(r"^def (\w+)\((\w+)" if python else r"^function (\w+)\((\w+)", block[:found[0].start()], re.M))
    return found[0], heads[-1].group(2)


# A-1 of the ninth construction review: five ways to carry the record's claims past the pass-9 self-check's alias
# reader. At c69b161b each, planted where the facts are built, dropped or added a claim of main's record while the
# self-check passed.
NINTH_REVIEW_PLANTS_PY = {
    "a default argument": "def _d(x=G):\n    x.append(x[0])\n_d()\n",
    "a lambda default": "(lambda x=G: x.pop())()\n",
    "*args": "def _k(*a):\n    a[0].pop()\n_k(G)\n",
    "**kwargs": "def _kw(**kw):\n    kw['x'].append(kw['x'][0])\n_kw(x=G)\n",
    "a class attribute": "class _K:\n    b = G\n_K.b.pop()\n",
}


@pytest.mark.parametrize("name", sorted(NINTH_REVIEW_PLANTS_PY))
def test_the_ninth_reviews_python_plants_cannot_move_the_record(name, M):
    text = R.lf(R.INSTRUMENT)
    block = R.py_block(text)
    site, given = _decision_site(block, python=True)
    plant = "".join(site.group(1) + line + "\n" for line in
                    NINTH_REVIEW_PLANTS_PY[name].replace("G", given + ".claims").rstrip("\n").split("\n"))
    mod = R.module_from(text.replace(block, block[:site.start()] + plant + block[site.start():]), "_p2a_plant_ninth")
    runs = 0
    for c in R.repro_cases():
        for strict in (False, True):
            try:
                a = M.gate_diff_text(c["summary"], c["diff"], strict=strict).to_dict()
            except Exception:
                continue
            b = mod.gate_diff_text(c["summary"], c["diff"], strict=strict).to_dict()
            assert R.relation(a, b, strict, PHRASES, allow_error=True) == [], (name, c["id"])
            runs += 1
    assert runs > 900


# A-2 of the same review: nine ways past the pass-9 store scan of the port.
NINTH_REVIEW_PLANTS_JS = {
    "a member as a for-of target": 'for (G[0].verdict of ["CONTRADICTED"]) {}',
    "a member as a for-in target": "for (G[0].text in { rewritten: 1 }) {}",
    "an optional call of pop": "G.pop?.();",
    "an optional call of push": "G.push?.(G[0]);",
    "an optional call of reverse": "G.reverse?.();",
    "an escaped member store": "G[0]." + chr(92) + 'u0076erdict = "CONTRADICTED";',
    "an escaped mutator": "G.p" + chr(92) + "u006fp();",
    "an escaped Object.assign": chr(92) + 'u004fbject.assign(G[0], { verdict: "CONTRADICTED", why: "rewritten" });',
    "a helper that hides .claims": "const pick = x => x.claims; const cl = [pick(P)][0]; cl.pop();",
}


@pytest.mark.parametrize("name", sorted(NINTH_REVIEW_PLANTS_JS))
def test_the_ninth_reviews_port_plants_cannot_move_the_record(name, work, tmp_path):
    text = R.lf(R.PORT)
    block = R.js_block(text)
    site, given = _decision_site(block, python=False)
    plant = site.group(1) + NINTH_REVIEW_PLANTS_JS[name].replace("G", given + ".claims").replace("P", given) + "\n"
    planted = tmp_path / "diffgate_planted.js"
    planted.write_bytes(text.replace(block, block[:site.start()] + plant + block[site.start():]).encode("utf-8"))
    items = [{"id": c["id"], "summary": c["summary"], "diff": c["diff"]} for c in R.repro_cases()]
    (tmp_path / "in.json").write_text(json.dumps(items, ensure_ascii=True), encoding="utf-8")
    node("--relation", work / "diffgate_main_reference.js", tmp_path / "in.json", tmp_path / "rel.json", planted)
    rel = json.loads((tmp_path / "rel.json").read_text(encoding="utf-8"))
    assert rel["counts"]["broken"] == 0 and rel["counts"]["raise_differs"] == 0, rel["broken"][:3]
    assert rel["counts"]["runs"] > 900 and rel["counts"]["abstained"] > 100, rel["counts"]


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
    old = "    pending = {i: c for i, c in enumerate(g.claims) if (c.kind, c.verdict) in _P2A_REACH}\n"
    assert block.count(old) == 1
    mod = R.module_from(text.replace(block, block.replace(old, old[:-2] + " and not strict}\n")), "_p2a_plant_strict")
    caught = []
    for sname, iid, summary, diff in inputs[:600]:
        try:
            off, on = mod.gate_diff_text(summary, diff).to_dict(), mod.gate_diff_text(summary, diff, strict=True).to_dict()
        except Exception:
            continue
        if R.strict_alike(off, on):
            caught.append(iid)
    assert caught, "the strict check did not refuse an overlay that skips itself under --strict"


JS_WHY = "    c.why = `${c.verdict} withheld by PATH-2a (${tag}): ${phrase}. main's reading: ${c.why}`;"
JS_REACH = '    reach.push(_P2A_REACH.has(claims[k].kind + "|" + claims[k].verdict));'
# Edits of APPLY itself, which the relation must refuse (the pin of APPLY's text names each of them too)
PORT_PLANTS = [
    ('    c.verdict = "UNCHECKABLE";', '    c.verdict = "CONTRADICTED";'),
    (JS_WHY, JS_WHY.replace("${c.why}`;", "`;")),
    ('    g.verdict = (contradicted || (strict && uncheckable)) ? "FAIL" : "PASS";', ""),
    ('    c.verdict = "UNCHECKABLE";', '    c.verdict = "UNCHECKABLE";\n    c.detail = {};'),
    # Integration-1: the overlay skipped under --strict
    (JS_REACH, JS_REACH.replace("verdict));", "verdict) && !strict);")),
    # I-1 (NOTE_path2a_third_pass_2026_09_30): the review's two mutants, which pass 2's --relation could not see
    ('    g.verdict = (contradicted || (strict && uncheckable)) ? "FAIL" : "PASS";',
     '    g.verdict = (contradicted || (strict && uncheckable)) ? "FAIL" : "PASS";\n    g.unparsed_claims = claims.map(c => c.text);'),
    ("    c.verdict = \"UNCHECKABLE\";", "    c.verdict = \"UNCHECKABLE\";\n    c.main_verdict = c.verdict;"),
    # NOTE_path2a_tenth_pass_2026_10_05: a reason of another form
    (JS_WHY, JS_WHY.replace("withheld by PATH-2a", "withheld by PATH-2a,")),
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
    measured at the ninth pass's head 0.10 to 0.51 of main's call on CPython 3.12.10 and 0.10 to 0.33 on 3.14.2
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
    `len(todo) > 400` and every committed test passed. The relation, both strict modes, on them. Since the tenth pass
    that plant, set where the decisions are computed, drops a claim of DECIDE's copy and cannot move the record; set in
    APPLY, the one function that holds the record, it is refused here (and by the pin of APPLY's text)."""
    assert _large_relation_python(N, M) == []
    text = R.lf(R.INSTRUMENT)
    block = R.py_block(text)
    for anchor, plant, moves in (
            ("    out = []\n    for i, c in todo:\n", "    if len(todo) > 400:\n        seen.claims.pop()\n", False),
            ("    moved = False\n", "    if len(pending) > 400:\n        g.claims.pop()\n", True)):
        assert block.count(anchor) == 1
        mod = R.module_from(text.replace(block, block.replace(anchor, plant + anchor)), "_p2a_plant_large")
        assert (_large_relation_python(mod, M) != []) is moves, anchor


def test_the_relation_on_large_summaries_port(work, tmp_path):
    """The same in the port, at three times the size (30,000 claims a case), with the seventh review's port plant (each
    claim's text gains a trailing space behind `todo.length > 400`), which --relation on the committed inputs passed:
    set where the decisions are computed it rewrites DECIDE's copy and moves nothing; set in APPLY it is refused."""
    (tmp_path / "in.json").write_text(json.dumps(_large_cases(3), ensure_ascii=False), encoding="utf-8")
    node("--relation", work / "diffgate_main_reference.js", tmp_path / "in.json", tmp_path / "rel.json")
    rep = json.loads((tmp_path / "rel.json").read_text(encoding="utf-8"))
    assert rep["broken"] == [] and rep["counts"]["broken"] == 0 and rep["counts"]["runs"] == 6, rep["counts"]
    text = R.lf(R.PORT)
    block = R.js_block(text)
    planted = tmp_path / "diffgate_planted.js"
    for anchor, plant, moves in (
            ("  const out = [];\n  for (const [i, c] of todo) {\n",
             '  if (todo.length > 400) for (const x of seen.claims) { x.text = x.text + " "; x.why = x.why + " "; }\n', False),
            ("  let moved = false;\n", '  if (pending.size > 400) for (const x of claims) x.text = x.text + " ";\n', True)):
        assert block.count(anchor) == 1
        planted.write_bytes(text.replace(block, block.replace(anchor, plant + anchor)).encode("utf-8"))
        node("--relation", work / "diffgate_main_reference.js", tmp_path / "in.json", tmp_path / "rel2.json", planted)
        assert (json.loads((tmp_path / "rel2.json").read_text(encoding="utf-8"))["counts"]["broken"] > 0) is moves, anchor


def test_cost_on_large_summaries_port(work, tmp_path):
    """The same in the port, at three times the size, within three times main's own call: the port's main is about
    seven times faster than CPython's, so the overlay's fixed per-claim work weighs more beside it, and a shared runner's
    noise more. Measured at the ninth pass's head: 0.42 to 1.38 times main's call; 495d2204's overlay took 8.6 to 9.6."""
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
    the report does not name the commit, and the branch's record is main's but for abstentions in reach. The review's
    plant, moved to where the decisions are computed, asks for an abstention on every `tests_pass` VERIFIED: APPLY takes
    no decision for a claim outside reach, so the record is what it was (NOTE_path2a_tenth_pass_2026_10_05)."""
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
    old = "    out = []\n    for i, c in todo:\n"
    assert block.count(old) == 1
    plant = ('    out = [(i, "tests", "#101") for i, c in enumerate(seen.claims) if c.kind == "tests_pass" and '
             'c.verdict == "VERIFIED"]\n    _P2A_PLANTED.append(len(out))\n    for i, c in todo:\n')
    mod = R.module_from(text.replace(block, block.replace(old, plant)), "_p2a_plant_tests_pass")
    mod._P2A_PLANTED = []
    planted, seen_planted = _tests_pass_relation(mod, M, str(green), monkeypatch)
    assert planted == [] and seen_planted == seen, planted[:2]
    assert sum(mod._P2A_PLANTED) > 1500, "the plant did not ask for an abstention on the tests_pass claims"


# ---- (C) cross-port ---------------------------------------------------------------------------------------------------

def _seen(c):
    """A claim's verdict as the overlay left it and, when the overlay wrote the reason, its phrase key and its defect
    tag: the decision bar C(i) compares (I-3 of the ninth integration review: the tag was not read)."""
    key = R.phrase_key(c["why"], PHRASES)
    return c["verdict"], key, (c["why"].split(" withheld by PATH-2a (", 1)[1].split("): ", 1)[0] if key else None)


def _said(c):
    """The verdict and the phrase key, as the pinned decisions of the cross-port cases are written."""
    return _seen(c)[:2]


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
# figures lie near the pin nearest to it (`_nearest_pin`), and prints them.
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
        ("16.0.0", "windows", "16"): (6672, 5, 6068, 1128, 482, 117, 20, 24, 6, 2, 21, 13, 2, 10),
        ("16.0.0", "posix", "16"): (6672, 5, 6071, 1128, 482, 114, 20, 24, 6, 2, 20, 13, 2, 9),
    },
    # every input of this shape where main's two ports decide the symbol claim apart parts the gate verdicts under the
    # overlay (149; 147 where the interpreter knows U+105C0 as a letter), since the tests or count claim that made both
    # of main's gates FAIL is withheld in both ports
    "line break": {("15.0.0", "any", "16"): (600, 0, 451, 451, 0, 149, 0, 149, 149, 0, 0, 0, 0, 0),
                   ("16.0.0", "any", "16"): (600, 0, 453, 453, 0, 147, 0, 147, 147, 0, 0, 0, 0, 0)},
    # on the patched engine main's two gate verdicts differ on 421 inputs (294 where the interpreter folds the Unicode
    # 16 pair too) and the overlay's on none: the counts the two mains decide apart are withheld in both ports
    # (`case_count`)
    "newer engine": {("15.0.0", "any", "16"): (1500, 0, 669, 294, 0, 831, 421, 0, 0, 421, 358, 0, 0, 358),
                     ("16.0.0", "any", "16"): (1500, 0, 913, 538, 0, 587, 294, 0, 0, 294, 255, 0, 0, 255)},
    "decorated world": {("15.0.0", "any", "16"): (1000, 0, 752, 355, 248, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        ("16.0.0", "any", "16"): (1000, 0, 752, 355, 248, 0, 0, 0, 0, 0, 0, 0, 0, 0)},
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


def _nearest_pin(name: str, pins: dict, runtime: tuple) -> tuple:
    """The measured figures a runtime without a pin is held near (C-3 and I-2 of the ninth reviews: at c69b161b this
    was whichever pin sorted ahead, so the newer-engine set failed on an engine below Unicode 16 for no fault of the
    code). The pin for the running interpreter's Unicode version, where there is one. The newer-engine set turns on
    something else: whether the two runtimes fold U+A7DC alike (Unicode 16 gave it a lowercase), so it takes the
    16.0.0 row where both do or neither does, and the 15.0.0 row where only the engine does. Exercised under emulation
    with an engine reporting Unicode 15.1 and one reporting 17, beside CPython 3.12.10 and 3.14.2."""
    py, fl, engine = runtime
    same = [k for k in sorted(pins) if k[1] == fl]
    if name == "newer engine":
        alike = (int(py.split(".")[0]) >= 16) == (int(engine) >= 16)
        return pins[next(k for k in same if k[0] == ("16.0.0" if alike else "15.0.0"))]
    return pins[next((k for k in same if k[0] == py), same[0])]


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
        ref = _nearest_pin(name, pins, runtime)
        assert all(abs(g - p) <= max(10, p // 10) for g, p in zip(got, ref)), (
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


def _new(p):
    return f"diff --git a/{p} b/{p}\nnew file mode 100644\n--- /dev/null\n+++ b/{p}\n@@ -0,0 +1 @@\n+x\n"


D_DOT_TWINS = _mod(".env") + _mod("env") + _mod("src/app.py")


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
     [("left over, one match: kind and verdict", 1, 1, ("VERIFIED", None, None), ("UNCHECKABLE", "dot", "#121"))]),
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
     [("left over, one match: kind and verdict", 0, 0, ("VERIFIED", None, None), ("UNCHECKABLE", "dir", "#97"))]),
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
    # ... and the U+2028 and U+2029 starts of the definition readers, which no committed input held (I-2 of the ninth
    # note: with either dropped from one port every test passed). The port's `^` also matches after them, so its V101
    # reads a definition there: a removed line `x = 1` U+2028 `def foo():` defines foo (`symbol`), an unchanged one
    # after U+2029 does too (`again`), and an added line `x = 1` U+2028 `def test_new():` holds a test only the port's
    # main counts, so "Added 0 tests" is CONTRADICTED on two different counts and withheld as `split` in both ports
    ("P9-a-removed-def-after-a-line-separator", "Adds function foo.",
     D_LINES(["x = 1", "def foo():"]).replace("-x = 0\n", "-x = 1 def foo():\n"),
     [("UNCHECKABLE", "symbol")], [("UNCHECKABLE", "symbol")], "PASS"),
    ("P9-an-unchanged-def-after-a-paragraph-separator", "Adds function foo.",
     D_LINES(["y = 1", "def foo():"]).replace("-x = 0\n", " x = 1 def foo():\n-y = 0\n"),
     [("UNCHECKABLE", "again")], [("UNCHECKABLE", "again")], "PASS"),
    ("P9-a-test-after-a-line-separator", "Added 0 tests.", D_TEST + D_LINES(["x = 1 def test_new():"]),
     [("UNCHECKABLE", "split")], [("UNCHECKABLE", "split")], "PASS"),
    # C-2 of the ninth cross-port review (NOTE_path2a_tenth_pass_2026_10_05): ten two-line inputs, each of which tells
    # one single-character slip in ONE port from this head (XPORT_PLANTS below; at c69b161b eleven such edits passed
    # every committed behaviour test). What each rests on: the placeholder form read by code point, not by UTF-16
    # unit (T1); a drive-like name whose letter is two units (T2); the backslash in a path run (T3); the digit 0 in a
    # count run (T4); a sentence ended by `!` and by `.` then a tab, so that a later accent is outside the claim's zone
    # (T5, T6); the count seam's `w`, its `e` and `file` spelled with a long s (T7 to T9, where main's Python reads a
    # second count the port does not); U+007F between `def` and a name (T10, one edit in each port).
    ("T1-astral-and-bmp-directories-share-a-placeholder", "Created app.py.",
     _new(chr(0xE9) + "/app.py") + _mod(chr(0x1F600) + "/app.py"),
     [("UNCHECKABLE", "case")], [("UNCHECKABLE", "case")], "PASS"),
    ("T2-a-drive-like-name-whose-letter-is-astral", "Modified x.py.", _mod("src/x.py") + _mod(chr(0x10400) + ":y.py"),
     [("UNCHECKABLE", "odd")], [("UNCHECKABLE", "odd")], "PASS"),
    ("T3-a-backslash-joins-a-path-run", "Modified app.py.\nNotes: " + chr(0xE9) + chr(92) + "app.py", _mod("app.py"),
     [("UNCHECKABLE", "extract")], [("UNCHECKABLE", "extract")], "PASS"),
    ("T4-a-zero-in-a-count-run", "30 files changed.\nNotes: " + chr(0xE9) + "30", D_DOT_TWINS,
     [("UNCHECKABLE", "extract")], [("UNCHECKABLE", "extract")], "PASS"),
    ("T5-a-sentence-ends-at-a-bang", "Only touches docs! Thanks to Jos" + chr(0xE9) + " for docs.", _mod("docs/a.md"),
     [("VERIFIED", None)], [("VERIFIED", None)], "PASS"),
    ("T6-a-sentence-ends-at-a-dot-and-a-tab", "Only touches docs.\tThanks to Jos" + chr(0xE9) + " for docs.",
     _mod("docs/a.md"), [("VERIFIED", None)], [("VERIFIED", None)], "PASS"),
    ("T7-a-count-seam-before-were", "3 files changed.\nAlso 3 files\x1fwere changed.", D_DOT_TWINS,
     [("UNCHECKABLE", "seam"), ("UNCHECKABLE", "seam")], [("UNCHECKABLE", "seam")], "PASS"),
    ("T8-a-count-seam-after-file", "3 files changed.\nAlso 1 file\x1fchanged.", D_DOT_TWINS,
     [("UNCHECKABLE", "seam"), ("UNCHECKABLE", "seam")], [("UNCHECKABLE", "seam")], "PASS"),
    ("T9-files-spelled-with-a-long-s", "3 files changed.\nAlso 2 file" + chr(0x17F) + " changed.", D_DOT_TWINS,
     [("UNCHECKABLE", "seam"), ("UNCHECKABLE", "seam")], [("UNCHECKABLE", "seam")], "PASS"),
    ("T10-del-between-def-and-a-name", "Adds function helper.",
     "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1 +1 @@\n-def\x7fhelper():\n+def helper():\n",
     [("UNCHECKABLE", "symbol")], [("UNCHECKABLE", "symbol")], "PASS"),
    # I-3 of the ninth integration review: a directory claim both ports withhold as `dir`, so that a port naming
    # another of the kind's tags for it is told (the tag is compared where the two ports' pinned decisions are equal)
    ("T11-a-directory-claim-resolved-by-base-name", "Modified docs/README.md.", _mod("README.md"),
     [("UNCHECKABLE", "dir")], [("UNCHECKABLE", "dir")], "PASS"),
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
    "P9-a-removed-def-after-a-line-separator": "V/V PASS/PASS",
    "P9-an-unchanged-def-after-a-paragraph-separator": "V/V PASS/PASS",
    "P9-a-test-after-a-line-separator": "C/C FAIL/FAIL",
    "T1-astral-and-bmp-directories-share-a-placeholder": "V/V PASS/PASS",
    "T2-a-drive-like-name-whose-letter-is-astral": "V/V PASS/PASS", "T3-a-backslash-joins-a-path-run": "V/V PASS/PASS",
    "T4-a-zero-in-a-count-run": "C/C FAIL/FAIL", "T5-a-sentence-ends-at-a-bang": "V/V PASS/PASS",
    "T6-a-sentence-ends-at-a-dot-and-a-tab": "V/V PASS/PASS", "T7-a-count-seam-before-were": "CC/C FAIL/FAIL",
    "T8-a-count-seam-after-file": "CC/C FAIL/FAIL", "T9-files-spelled-with-a-long-s": "CV/C FAIL/FAIL",
    "T10-del-between-def-and-a-name": "V/V PASS/PASS", "T11-a-directory-claim-resolved-by-base-name": "V/V PASS/PASS",
}
# One-character slips in ONE port, each named with the cross-port case that tells it from this head: (the port, the
# case, the text, its replacement). C-2 of the ninth cross-port review planted the eleven of T1 to T10 at c69b161b and
# every committed behaviour test passed; the last is I-3's, a tag only the port names otherwise.
XPORT_PLANTS = [
    ("port", "T1-astral-and-bmp-directories-share-a-placeholder",
     '  for (const ch of f.slice(k)) out.push(ch.codePointAt(0) < 128 ? ch : "' + chr(92) + 'ufffd");\n',
     '  for (const ch of f.slice(k).split("")) out.push(ch.codePointAt(0) < 128 ? ch : "' + chr(92) + 'ufffd");\n'),
    ("port", "T2-a-drive-like-name-whose-letter-is-astral",
     "  const cps = Array.from(q.slice(0, 6)).slice(0, 3);\n", '  const cps = q.slice(0, 3).split("");\n'),
    ("port", "T3-a-backslash-joins-a-path-run", "0123456789_./-" + chr(92) * 2 + '";', '0123456789_./-";'),
    ("port", "T4-a-zero-in-a-count-run", "const _p2aCountUnit = u => (u >= 48 && u <= 57) || _p2aWordishUnit(u);",
     "const _p2aCountUnit = u => (u >= 49 && u <= 57) || _p2aWordishUnit(u);"),
    ("port", "T5-a-sentence-ends-at-a-bang", '(ch === "." || ch === "!" || ch === "?")', '(ch === "." || ch === "?")'),
    ("port", "T6-a-sentence-ends-at-a-dot-and-a-tab",
     '(s[k + 1] === " " || s[k + 1] === "' + chr(92) + 't" || s[k + 1] === "' + chr(92) + 'r")',
     '(s[k + 1] === " " || s[k + 1] === "' + chr(92) + 'r")'),
    ("port", "T7-a-count-seam-before-were", '"CcFfWw".includes(s.charAt(k))', '"CcFf".includes(s.charAt(k))'),
    ("port", "T8-a-count-seam-after-file", '"0123456789EeSs' + chr(92) + 'u017f".includes(s.charAt(a - 1))',
     '"0123456789Ss' + chr(92) + 'u017f".includes(s.charAt(a - 1))'),
    ("port", "T9-files-spelled-with-a-long-s",
     "[Ll][Ee]|[Ff][Ii" + chr(92) + "u0130" + chr(92) + "u0131][Ll][Ee]" + chr(92) + 'u017f");', '[Ll][Ee]");'),
    ("port", "T10-del-between-def-and-a-name", "const _p2aCoarseUnit = u => u <= 0x20 || u === 0x7f || u >= 0x80;",
     "const _p2aCoarseUnit = u => u <= 0x20 || u >= 0x80;"),
    ("python", "T10-del-between-def-and-a-name",
     '_P2A_COARSE_RUN = re.compile("[' + chr(92) + "x00-" + chr(92) + "x20" + chr(92) + "x7f-" + chr(92) + 'U0010ffff]*")',
     '_P2A_COARSE_RUN = re.compile("[' + chr(92) + "x00-" + chr(92) + "x20" + chr(92) + "x80-" + chr(92) + 'U0010ffff]*")'),
    ("port", "T11-a-directory-claim-resolved-by-base-name",
     '  if (v121 && !v97) return [r97 === null ? "dir" : "tier", "#97"];',
     '  if (v121 && !v97) return [r97 === null ? "dir" : "tier", "#121"];'),
]


@pytest.mark.parametrize("which,cid,old,new", XPORT_PLANTS, ids=[f"{p[0]}: {p[1]}" for p in XPORT_PLANTS])
def test_a_one_port_slip_is_told_by_its_cross_port_case(which, cid, old, new, tmp_path):
    """Bar C rests on a transliteration held by tests. Each edit here changes one character class or one rule in one
    port alone; planted, it changes what that port decides (verdict, phrase or tag) on the case it is named with, whose
    pin in XPORT_CASES therefore fails."""
    _cid, summary, diff, want_py, want_js = next(x for x in XPORT_CASES if x[0] == cid)[:5]
    if which == "python":
        text = R.lf(R.INSTRUMENT)
        block = R.py_block(text)
        assert block.count(old) == 1, old
        mod = R.module_from(text.replace(block, block.replace(old, new)), "_p2a_plant_xport")
        head = [_seen(x) for x in N.gate_diff_text(summary, diff).to_dict()["claims"]]
        got = [_seen(x) for x in mod.gate_diff_text(summary, diff).to_dict()["claims"]]
        want = want_py
    else:
        text = R.lf(R.PORT)
        block = R.js_block(text)
        assert block.count(old) == 1, old
        planted = tmp_path / "diffgate_planted.js"
        planted.write_bytes(text.replace(block, block.replace(old, new)).encode("utf-8"))
        (tmp_path / "in.json").write_text(json.dumps([{"id": cid, "summary": summary, "diff": diff}], ensure_ascii=True),
                                          encoding="utf-8")
        seen = {}
        for name, port in (("head", R.PORT), ("planted", planted)):
            node("--records", port, tmp_path / "in.json", tmp_path / (name + ".json"))
            rec = json.loads((tmp_path / (name + ".json")).read_text(encoding="utf-8"))[0]["rec"]
            seen[name] = [_seen(x) for x in rec["claims"]]
        head, got, want = seen["head"], seen["planted"], want_js
    assert [x[:2] for x in head] == want, "the case's pin is this head's decision"
    assert got != head, f"the planted {which} decides {cid} as the head does"





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
        assert [_said(x) for x in b["claims"]] == want_py, cid
        assert [_said(x) for x in js[cid]["new"]["claims"]] == want_js, cid
        if want_py == want_js:                  # ... and where the two ports decide alike they name one defect
            assert [_seen(x) for x in b["claims"]] == [_seen(x) for x in js[cid]["new"]["claims"]], cid
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
        assert (b["verdict"], [_said(x) for x in b["claims"]]) == (gate, want), k
        assert (js[k]["new"]["verdict"], [_said(x) for x in js[k]["new"]["claims"]]) == (js_gate, js_want), k
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
        assert (b["verdict"], [_said(x) for x in b["claims"]]) == (gate, want), k
        assert (js[k]["new"]["verdict"], [_said(x) for x in js[k]["new"]["claims"]]) == (gate, want), k


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
    # APPLY's fixed sets (NOTE_path2a_tenth_pass_2026_10_05): the tags a kind's decision may carry, each mapped to
    # itself in the Python, and the detail fields DECIDE's copy carries; the same in both ports and in the reference
    assert {k: tuple(v) for k, v in t["tags"].items()} == R.DEFECTS == {k: tuple(v) for k, v in N._P2A_TAGS.items()}
    assert all(k == v for tags in N._P2A_TAGS.values() for k, v in tags.items())
    assert all(N._P2A_KIND_DEFECT[k] in v for k, v in N._P2A_TAGS.items()) and set(N._P2A_TAGS) == set(N._P2A_KIND_DEFECT)
    assert sorted(t["fields"]) == sorted(N._P2A_FIELDS) == ["declared", "n", "name", "path", "prefix", "prefix2"]
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
