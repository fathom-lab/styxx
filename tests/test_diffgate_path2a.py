"""PATH-2a: main's diff gate, unchanged, plus an overlay that only abstains (NOTE_path2a_abstain_overlay_2026_09_30,
NOTE_path2a_second_pass_2026_09_30).

The reference in every test here is `main` itself: this checkout's `styxx/diffgate.py` and `web/gate/diffgate.js` with
the PATH-2a block cut out and the door hooks reverted, asserted to hash to the files on `origin/main` 1cde8b82
(tests/_p2a_ref.py). No copy of main is committed.

(A) by construction -- on every committed input, both strict modes, both ports and the git door, each record equals
    main's except that a decided verdict in REACH may become UNCHECKABLE with the overlay's reason, and the gate
    verdict is main's formula over the claims; each claim reads the same under --strict as without it; where main
    raises, the branch raises the same exception type.
(C) cross-port -- wherever main's two ports give a claim the same kind, verdict and text, the overlay's verdict and
    phrase are the same in both, and wherever they give it the same record, so is the overlay's; the constant tables
    the overlay leans on are pinned by enumeration on the running engines.
And the static facts: the reconstruction, the self-checks over each block's source, the error fallback, the
reproductions, the cost per call, and what must not move (the demo, the committed capsules, charon's lines, the
bookmarklet source). Coverage (B) is in tests/test_diffgate_path2a_truth.py.
"""
from __future__ import annotations

import collections
import importlib.util
import io
import json
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


def node(*args, timeout=900):
    if NODE is None:
        pytest.skip("node is not on PATH; the port half of this check cannot run here")
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
    assert hooks == ['    return _p2a_abstain(g, strict, lambda: _P2aFacts(diff_text or ""))  # PATH-2a',
                     "    return _p2a_abstain(g, strict, lambda: _P2aFacts(diff_text, name_status))  # PATH-2a"]


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
# NOTEs' rules before it was written down. A change to the overlay or to the inputs moves it.
ABSTENTIONS = {
    "windows": {
        "decided": 11115, "main raises": 8,
        "file_created:case": 3, "file_created:dir": 65, "file_created:divergent": 19, "file_created:dot": 20,
        "file_created:dot_earliest": 4, "file_created:dot_tier": 11, "file_created:extract": 24, "file_created:odd": 1,
        "file_created:tier": 9,
        "file_deleted:case": 1, "file_deleted:dir": 52, "file_deleted:divergent": 20, "file_deleted:dot": 21,
        "file_deleted:dot_earliest": 1, "file_deleted:dot_tier": 7, "file_deleted:extract": 31, "file_deleted:odd": 1,
        "file_deleted:tier": 4,
        "file_touched:dir": 242, "file_touched:divergent": 102, "file_touched:dot": 82, "file_touched:dot_tier": 71,
        "file_touched:extract": 92, "file_touched:odd": 5,
        "files_changed_count:count": 381, "files_changed_count:divergent": 226,
        "only_touches:divergent": 108, "only_touches:extract": 1, "only_touches:only": 33,
        "symbol_added:extract": 17, "symbol_added:symbol": 172, "tests_added:split": 4, "tests_added:tests": 402,
    },
}
# Under the POSIX flavour main reads `c:x.py` as a bare name not in the diff, so three decided drive-like claims are
# UNCHECKABLE on main to begin with: path2a:guard-drive-like-path claim 0, and fuzz 20260930:1033 and :1590.
ABSTENTIONS["posix"] = {**ABSTENTIONS["windows"], "decided": 11112, "file_touched:odd": 4}
del ABSTENTIONS["posix"]["file_created:odd"], ABSTENTIONS["posix"]["file_deleted:odd"]
# main's own reading depends on the interpreter's Unicode tables where the inputs probe letters added in Unicode 14 and
# 16 (#161's k2, x1 and y3 cases). Measured on CPython 3.12 (Unicode 15.0) and 3.14 (16.0); for Unicode 13.0 and 14.0
# (CI's 3.9 to 3.11) by re-reading the inputs with the five Unicode 14 letters present mapped to code points no
# version assigns (NOTE_path2a_second_pass_2026_09_30, I-10): no figure moved. On 16.0 main reads the y3 name
# `fo` + U+105C0 whole and CONTRADICTS it, so it is not a VERIFIED symbol the overlay withholds.
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


def test_lockstep_python(M, inputs):
    """The overlay's status-map builder, keyed by main's _norm, is main's parse_unified_diff map, order included; and
    its count of `def test_` sites in line view 0 is main's count of `^\\s*def test_` over main's added lines."""
    checked = 0
    for _s, iid, _summary, diff in inputs:
        try:
            status, blob = M.parse_unified_diff(diff)
        except Exception:
            continue
        assert list(N._p2a_build(N._p2a_regs_raw(diff or ""), N._norm).items()) == list(status.items()), iid
        assert N._p2a_pairing(N._p2a_views(diff or ""))[0][0] == len(re.findall(r"^\s*def test_", blob, re.M)), iid
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
        mine = [g for g, _p in N._p2a_pairing(N._p2a_views(row[3] or ""))]
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
]


def _git_repos(tmp_path):
    for name, summary, base, head, quote in GIT_CASES:
        yield name, summary, _repo(tmp_path, name, base, head, quote)
    for name, summary, base, head, config in GIT_TREE_CASES:
        yield name, summary, _tree_repo(tmp_path, name, base, head, config)


@pytest.mark.skipif(GIT is None, reason="git is not on PATH; the git door cannot be exercised here")
def test_git_door(M, tmp_path):
    seen = collections.Counter()
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
        # G-P3's analogue: where main's two doors agree on the claims, the overlay decides alike at both
        text = N._git(repo, "diff", "HEAD~1..HEAD")
        rows = lambda g: [(c.kind, c.verdict, c.why) for c in g.claims]  # noqa: E731
        if rows(M.gate_diff(summary, repo, "HEAD~1", "HEAD")) == rows(M.gate_diff_text(summary, text)):
            assert rows(N.gate_diff(summary, repo, "HEAD~1", "HEAD")) == rows(N.gate_diff_text(summary, text)), name
    assert {"R", "C", "T"} <= set(seen), seen


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
    ("    g, n = int(m.group(1)), int(m.group(2))\n", '    g, n = int(m.group(1)), int(c.detail["n"])\n',
     "int() of anything but a regex group"),
]


@pytest.mark.parametrize("old,new,what", SELFCHECK_PLANTS)
def test_the_selfcheck_refuses_a_planted_write(old, new, what):
    text = R.lf(R.INSTRUMENT)
    block = R.py_block(text)
    assert block.count(old) == 1, old
    rep = N.selfcheck_p2a_only_abstains(text.replace(block, block.replace(old, new)))
    assert rep["ok"] is False and any(what in p for p in rep["problems"]), rep["problems"]


JS_BANNED = ("toLowerCase", "toUpperCase", "toLocale", "localeCompare", "normalize(", ".trim", "trimStart", "trimEnd",
             "\\p{", "Intl", ".sort(", "String.raw", "eval(", "Function(", "prototype", ".call(", ".apply(",
             "Reflect", "globalThis")


JS_KEYWORDS = {"return", "of", "in", "const", "let", "var", "case", "typeof", "void", "delete", "throw", "yield",
               "await", "else", "do", "new"}


def js_problems(block: str) -> list[str]:
    """The port block's token scan: banned calls anywhere; no regex literal (no '/' in code outside strings and
    comments); `new RegExp` with one static argument, no class escape, no '.' and no flags; no computed member access
    whose brackets hold a string (a name built from pieces)."""
    out = [f"banned token {t!r}" for t in JS_BANNED if t in block]
    code, i, strings = [], 0, []
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
            code.append(ch + ch)
            i = j + 1
            continue
        code.append(ch)
        i += 1
    code = "".join(code)
    if "/" in code:
        out.append("a '/' in code: a regex literal or a division")
    for s in strings:
        if re.search(r"(?<!\\)(?:\\\\)*\\\\[wWsSbBdD]", s):
            out.append(f"class escape in a string: {s[:40]!r}")
    for m in re.finditer(r"new RegExp\(", code):
        depth, j, commas = 1, m.end(), 0
        while depth:
            depth += {"(": 1, ")": -1}.get(code[j], 0)
            commas += code[j] == "," and depth == 1
            j += 1
        if commas:
            out.append("new RegExp with a second argument")
    for m in re.finditer(r"([A-Za-z0-9_$]+|[)\]])\s*\[", code):
        if m.group(1) in JS_KEYWORDS:           # an array literal or a destructuring pattern, not a member access
            continue
        depth, j = 1, m.end()
        while depth:
            depth += {"[": 1, "]": -1}.get(code[j], 0)
            j += 1
        if re.search("\"\"|''|``", code[m.end():j - 1]):
            out.append("computed member access with a string in the brackets")
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


def test_build_bookmarklet_check():
    if shutil.which("terser") is None:
        pytest.skip("terser is not on PATH; build_bookmarklet.py --check cannot rebuild the minified bytes here")
    r = subprocess.run([sys.executable, str(R.ROOT / "web" / "gate" / "build_bookmarklet.py"), "--check"],
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
    e, zh = chr(0xE9), chr(0x4E2D)

    def wide_def(n, ch):
        return {"id": f"wide-def-{n}", "summary": "Added function foo. Added 1 test.",
                "diff": ("--- a/src/app.py\n+++ b/src/app.py\n@@ -1,3 +1,2 @@\n-def " + ch * n + "\n-def test_" + ch * n
                         + "\n+def test_a():\n+def foo():\n")}

    def files(k, claims):
        parts = [f"--- a/d{i % 50}/f{i}.py\n+++ b/d{i % 50}/f{i}.py\n@@ -1 +1 @@\n-x\n+y\n" for i in range(k)]
        summ = [f"Modified d{i % 50}/f{(i * 37) % k}.py." if i % 2 else f"- d{i % 50}/f{(i * 37) % k}.py: tweak"
                for i in range(claims)]
        return {"id": f"files-{k}-{claims}", "summary": " ".join(summ), "diff": "".join(parts)}

    long_line = {"id": "long-line-20000", "summary": "Added 1 test.",
                 "diff": ("--- a/tests/test_x.py\n+++ b/tests/test_x.py\n@@ -1 +1 @@\n-def test_a():\n+x "
                          + "def test_a " * 20000 + "\n")}
    return [wide_def(5000, e), wide_def(50000, e), wide_def(50000, zh), files(2000, 200), long_line]


# The bounds the second-pass NOTE states: a `def` beside a 5,000- or 50,000-code-point run, and one 280 KB line, under
# 1 s a call in both ports; 2,000 files with 200 path claims, the overlay's own cost at most 0.5 s in Python and 0.3 s
# in the port beyond main's time on the same input. On e12214e5 the first takes seconds and the fourth 9 s in Python.
BOUND_S = {"wide-def-5000": 1.0, "wide-def-50000": 1.0, "long-line-20000": 1.0}
OVERLAY_S = {"files-2000-200": (0.5, 0.3)}


def test_cost_per_call_python(M):
    for it in _timing_cases():
        t0 = time.perf_counter()
        M.gate_diff_text(it["summary"], it["diff"])
        t1 = time.perf_counter()
        N.gate_diff_text(it["summary"], it["diff"])
        t2 = time.perf_counter()
        if it["id"] in BOUND_S:
            assert t2 - t1 < BOUND_S[it["id"]], (it["id"], t2 - t1)
        else:
            assert (t2 - t1) - (t1 - t0) < OVERLAY_S[it["id"]][0], (it["id"], t1 - t0, t2 - t1)


def test_cost_per_call_port(work, tmp_path):
    (tmp_path / "in.json").write_text(json.dumps(_timing_cases(), ensure_ascii=False), encoding="utf-8")
    node("--timing", work / "diffgate_main_reference.js", tmp_path / "in.json", tmp_path / "t.json")
    for d in json.loads((tmp_path / "t.json").read_text(encoding="utf-8")):
        if d["id"] in BOUND_S:
            assert d["new"] < 1000 * BOUND_S[d["id"]], d
        else:
            assert d["new"] - d["main"] < 1000 * OVERLAY_S[d["id"]][1], d


# ---- (C) cross-port ---------------------------------------------------------------------------------------------------

def _rows(d):
    return [(c["kind"], c["verdict"], c["why"], c["text"], json.dumps(c["detail"], sort_keys=True)) for c in d["claims"]]


def _seen(c):
    """A claim's verdict as the overlay left it, and its phrase key when the overlay wrote the reason."""
    return c["verdict"], R.phrase_key(c["why"], PHRASES)


def test_cross_port_decisions(M, inputs, work):
    node("--decisions", work / "diffgate_main_reference.js", work / "in.json", work / "dec.json")
    js = {d["id"]: d for d in json.loads((work / "dec.json").read_text(encoding="utf-8"))}
    c = collections.Counter()
    differ = []
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
        ra, rb, ja, jb = _rows(a), _rows(b), _rows(j["main"]), _rows(j["new"])
        if ra == ja and a["verdict"] == j["main"]["verdict"]:
            c["inputs main agrees on"] += 1
            if rb != jb or b["verdict"] != j["new"]["verdict"]:
                differ.append(("whole record", row[0], row[1]))
        if len(ra) != len(ja):
            c["inputs whose claim lists differ in main"] += 1
            continue
        key = lambda r: r[:2] + r[3:4]  # noqa: E731  (kind, verdict, text)
        if [key(r) for r in ra] == [key(r) for r in ja] and b["verdict"] != j["new"]["verdict"]:
            differ.append(("gate verdict where main's claims agree on kind, verdict and text", row[0], row[1]))
        for k in range(len(ra)):
            if key(ra[k]) == key(ja[k]):
                c["claims main gives the same kind, verdict and text"] += 1
                c["... the overlay's verdict and phrase agree"] += _seen(b["claims"][k]) == _seen(j["new"]["claims"][k])
                if _seen(b["claims"][k]) != _seen(j["new"]["claims"][k]):
                    differ.append((row[0], row[1], k, _seen(b["claims"][k]), _seen(j["new"]["claims"][k])))
            if ra[k] == ja[k]:
                c["claims main agrees on"] += 1
                c["... overlay agrees"] += rb[k] == jb[k]
                if rb[k] != jb[k]:
                    differ.append((row[0], row[1], k, rb[k][:3], jb[k][:3]))
            else:
                c["claims main disagrees on"] += 1
                moved = lambda r, s: s[1] == "UNCHECKABLE" and r[1] != "UNCHECKABLE"  # noqa: E731
                c["... overlay decision differs (reported)"] += moved(ra[k], rb[k]) != moved(ja[k], jb[k])
    print("cross-port:", dict(c))
    assert not differ, differ[:10]
    assert c["claims main agrees on"] == c["... overlay agrees"] > 15000
    assert c["claims main gives the same kind, verdict and text"] == c["... the overlay's verdict and phrase agree"]


# ---- tables pinned by enumeration -------------------------------------------------------------------------------------

def _set(s):
    return {ord(ch) for ch in s}


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
