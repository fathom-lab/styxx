"""PATH-2a: main's diff gate, unchanged, plus an overlay that only abstains (NOTE_path2a_abstain_overlay_2026_09_30).

The reference in every test here is `main` itself: this checkout's `styxx/diffgate.py` and `web/gate/diffgate.js` with
the PATH-2a block cut out and the door hooks reverted, asserted to hash to the files on `origin/main` 1cde8b82
(tests/_p2a_ref.py). No copy of main is committed.

(A) by construction -- on every committed input, both strict modes, both ports and the git door, each record equals
    main's except that a decided verdict in REACH may become UNCHECKABLE with the overlay's reason, and the gate
    verdict is main's formula over the claims; where main raises, the branch raises the same exception type.
(C) cross-port -- wherever main's two ports give a claim the same record, the overlay's decision and reason are the
    same in both; the constant tables the overlay leans on are pinned by enumeration on the running engines.
And the static facts: the reconstruction, the self-checks over each block's source, the error fallback, the
reproductions, and what must not move (the demo, the committed capsules, charon's lines, the bookmarklet source).
Coverage (B) is in tests/test_diffgate_path2a_truth.py.
"""
from __future__ import annotations

import collections
import io
import json
import re
import shutil
import subprocess
import sys
from contextlib import redirect_stdout

import pytest

import styxx.diffgate as N
from tests import _p2a_ref as R

NODE = shutil.which("node")
GIT = shutil.which("git")
CHECK = R.DIFFERENTIAL / "check_path2a.js"
PHRASES = N._P2A_PHRASES


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

# Abstentions by (kind, phrase key) over every committed input, strict off. Pinned after review: every figure here
# was read against the NOTE's rules before it was written down. A change to the overlay or to the inputs moves it.
ABSTENTIONS = {
    "decided": 11106, "main raises": 8,
    "file_created:case": 6, "file_created:dir": 71, "file_created:divergent": 19, "file_created:dot": 26,
    "file_created:dot_tier": 10, "file_created:odd": 3, "file_created:tier": 11,
    "file_deleted:case": 9, "file_deleted:dir": 62, "file_deleted:divergent": 20, "file_deleted:dot": 25,
    "file_deleted:dot_tier": 7, "file_deleted:odd": 2, "file_deleted:tier": 5,
    "file_touched:case": 28, "file_touched:dir": 270, "file_touched:divergent": 102, "file_touched:dot": 87,
    "file_touched:dot_tier": 71, "file_touched:odd": 9,
    "files_changed_count:count": 381, "files_changed_count:divergent": 226,
    "only_touches:divergent": 108, "only_touches:only": 33,
    "symbol_added:symbol": 172, "tests_added:tests": 420,
}


def test_abstain_only_python(M, inputs):
    counts = collections.Counter()
    broken = []
    for i, (sname, iid, summary, diff) in enumerate(inputs):
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
            if not strict:
                for x, y in zip(a["claims"], b["claims"]):
                    if x["verdict"] in ("VERIFIED", "CONTRADICTED"):
                        counts["decided"] += 1
                        if y["verdict"] == "UNCHECKABLE":
                            counts[x["kind"] + ":" + R.phrase_key(y["why"], PHRASES)] += 1
    print("PATH-2a abstentions over the committed inputs:", json.dumps(dict(sorted(counts.items())), indent=1))
    assert not broken, broken[:10]
    assert dict(counts) == ABSTENTIONS


def test_abstain_only_port(work):
    node("--relation", work / "diffgate_main_reference.js", work / "in.json", work / "rel.json")
    rep = json.loads((work / "rel.json").read_text(encoding="utf-8"))
    print("port relation:", rep["counts"])
    assert rep["broken"] == [] and rep["counts"]["broken"] == 0 and rep["counts"]["raise_differs"] == 0
    assert rep["counts"]["runs"] > 10000


def test_lockstep_python(M, inputs):
    """The overlay's status-map builder, keyed by main's _norm, is main's parse_unified_diff map, order included."""
    checked = 0
    for _s, iid, _summary, diff in inputs:
        try:
            want = list(M.parse_unified_diff(diff)[0].items())
        except Exception:
            continue
        assert list(N._p2a_build(N._p2a_regs_raw(diff or ""), N._norm).items()) == want, iid
        checked += 1
    assert checked > 5000


def test_lockstep_port(work):
    node("--lockstep", work / "in.json", work / "lock.json")
    rep = json.loads((work / "lock.json").read_text(encoding="utf-8"))
    assert rep["differ"] == [] and rep["checked"] > 5000


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


@pytest.mark.skipif(GIT is None, reason="git is not on PATH; the git door cannot be exercised here")
def test_git_door(M, tmp_path):
    for name, summary, base, head, quote in GIT_CASES:
        repo = _repo(tmp_path, name, base, head, quote)
        for strict in (False, True):
            a = M.gate_diff(summary, repo, "HEAD~1", "HEAD", strict=strict).to_dict()
            b = N.gate_diff(summary, repo, "HEAD~1", "HEAD", strict=strict).to_dict()
            assert R.relation(a, b, strict, PHRASES) == [], name
        # the lockstep at the git door: the overlay's builder over --name-status is main's four-line loop
        name_status = N._git(repo, "diff", "--name-status", "HEAD~1..HEAD")
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


@pytest.mark.skipif(GIT is None, reason="git is not on PATH; the git door cannot be exercised here")
def test_git_door_reproductions(tmp_path):
    def rows(name):
        _n, summary, base, head, quote = next(c for c in GIT_CASES if c[0] == name)
        g = N.gate_diff(summary, _repo(tmp_path, name, base, head, quote), "HEAD~1", "HEAD")
        return g.verdict, [(c.kind, c.verdict, R.phrase_key(c.why, PHRASES)) for c in g.claims]

    assert rows("r101") == ("PASS", [("symbol_added", "UNCHECKABLE", "symbol"), ("tests_added", "UNCHECKABLE", "tests")])
    assert rows("directory-claim") == ("PASS", [("file_created", "UNCHECKABLE", "dir")])
    assert rows("readmes-reversed") == ("PASS", [("file_touched", "VERIFIED", None)])
    assert rows("twins") == ("FAIL", [("files_changed_count", "UNCHECKABLE", "count"), ("file_created", "UNCHECKABLE", None),
                                      ("file_deleted", "VERIFIED", None), ("only_touches", "CONTRADICTED", None)])
    assert rows("count-twin") == ("PASS", [("files_changed_count", "UNCHECKABLE", "count")])


# ---- the self-checks --------------------------------------------------------------------------------------------------

def test_selfcheck_p2a_only_abstains():
    rep = N.selfcheck_p2a_only_abstains()
    assert rep["ok"] is True, rep["problems"]


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
    ("        out.append(chr(o + 32)", "        out.append(chr(o + 32) if o else ch.casefold())\n        out.append(chr(o + 32)",
     "attribute .casefold"),
]


@pytest.mark.parametrize("old,new,what", SELFCHECK_PLANTS)
def test_the_selfcheck_refuses_a_planted_write(old, new, what):
    text = R.lf(R.INSTRUMENT)
    block = R.py_block(text)
    assert block.count(old) == 1, old
    rep = N.selfcheck_p2a_only_abstains(text.replace(block, block.replace(old, new)))
    assert rep["ok"] is False and any(what in p for p in rep["problems"]), rep["problems"]


JS_BANNED = ("toLowerCase", "toUpperCase", "toLocale", "localeCompare", "normalize(", ".trim", "trimStart", "trimEnd",
             "\\p{", "Intl", ".sort(", "String.raw", "eval(", "Function(")


def js_problems(block: str) -> list[str]:
    """The port block's token scan: banned calls anywhere; no regex literal (no '/' in code outside strings and
    comments); `new RegExp` with one static argument, no class escape, no '.' and no flags."""
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
    return out


def test_the_port_block_passes_its_token_scan():
    assert js_problems(R.js_block()) == []


@pytest.mark.parametrize("old,new", [
    ("const o = ch.codePointAt(0);\n    out +=", "const o = ch.toLowerCase().codePointAt(0);\n    out +="),
    ('new RegExp("\\r\\n|\\r|\\n")', 'new RegExp("\\r\\n|\\r|\\n", "u")'),
    ('new RegExp("\\r\\n|\\r|\\n")', 'new RegExp("\\\\s")'),
    ("const _p2aBs = s => s.split(", "const _p2aBs = s => s.replace(/x/, \"\").split("),
    ("const out = text.split(rx);", "const out = text.split(rx).sort();"),
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


PORT_PLANTS = [
    ('      c.verdict = "UNCHECKABLE";', '      c.verdict = "CONTRADICTED";'),
    ("      c.why = _p2aReason(c.verdict, hit[1], hit[0], c.why);", "      c.why = _p2aReason(c.verdict, hit[1], hit[0], \"\");"),
    ('  g.verdict = (contradicted || (strict && uncheckable)) ? "FAIL" : "PASS";', ""),
    ("    if (hit !== null) {", "    c.detail = {};\n    if (hit !== null) {"),
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


# ---- (C) cross-port ---------------------------------------------------------------------------------------------------

def _rows(d):
    return [(c["kind"], c["verdict"], c["why"], c["text"], json.dumps(c["detail"], sort_keys=True)) for c in d["claims"]]


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
        for k in range(len(ra)):
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


# ---- tables pinned by enumeration -------------------------------------------------------------------------------------

def _set(s):
    return {ord(ch) for ch in s}


def test_python_tables():
    breaks = {cp for cp in range(0x110000) if len(("a" + chr(cp) + "b").splitlines()) == 2}
    space = {cp for cp in range(0x110000) if chr(cp).isspace()}
    holds_ascii = {cp for cp in range(0x80, 0x110000) if any(ord(x) < 128 for x in chr(cp).lower())}
    longer = {cp for cp in range(0x110000) if len(chr(cp).lower()) > 1}
    assert breaks == _set(N._P2A_PY_BREAKS)
    assert space == _set(N._P2A_PY_SPACE)
    assert holds_ascii == {0x130, 0x212A} and longer == {0x130}, "the fold/wild lemma no longer holds here"


def test_port_tables_and_constants(tmp_path):
    node("--tables", tmp_path / "t.json")
    t = json.loads((tmp_path / "t.json").read_text(encoding="utf-8"))
    print("node", t["node"], "unicode", t["unicode"])
    assert set(t["whitespace"]) == set(t["trim"]) == set(t["js_space"]), "the port's trim table moved"
    assert set(t["lower_holds_ascii"]) == {0x130, 0x212A} and set(t["lower_longer"]) == {0x130}
    py_space, js_space = _set(N._P2A_PY_SPACE), set(t["js_space"])
    need = (_set(N._P2A_PY_BREAKS) - {0x0A, 0x0D}) | (py_space ^ js_space) | {0x2028, 0x2029}
    assert need <= _set(N._P2A_DIVERGENT), "a character the two ports read differently is not in the guard"
    assert set(t["divergent"]) == _set(N._P2A_DIVERGENT) and set(t["py_breaks"]) == _set(N._P2A_PY_BREAKS)
    assert t["headers"] == list(N._P2A_HEADERS)
    assert {tuple(x) for x in t["reach"]} == set(N._P2A_REACH) == set(R.REACH)
    assert t["phrases"] == N._P2A_PHRASES and t["kind_defect"] == N._P2A_KIND_DEFECT
    assert t["directory_rule"] is True
