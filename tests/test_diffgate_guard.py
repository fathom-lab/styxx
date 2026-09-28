# -*- coding: utf-8 -*-
"""NOTE_path2_eleventh_pass_2026_09_28: the licensed-difference rule at the verdict.

(a) THE REFERENCE. `main`'s reader is vendored unchanged: `styxx/_diffgate_ref.py` is `origin/main`'s
    `styxx/diffgate.py` and `web/gate/diffgate_ref.js` is `origin/main`'s `web/gate/diffgate.js`, byte for byte.
    The pins below are the sha256s of those two blobs (the file 7.48.0 ships, and its port).
"""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REF_PY = ROOT / "styxx" / "_diffgate_ref.py"
REF_JS = ROOT / "web" / "gate" / "diffgate_ref.js"
INSTRUMENT = ROOT / "styxx" / "diffgate.py"

# origin/main 2a6ce0a3: styxx/diffgate.py (the file styxx 7.48.0 ships) and web/gate/diffgate.js, LF.
MAIN_DIFFGATE_PY_SHA256 = "9b620e00a19464589308a987819894ae7cc3c111c66a5f8a457a84b8a6c604eb"
MAIN_DIFFGATE_JS_SHA256 = "06688702999cdabe763265722a0ac14d4b9ffb40d0efcbb32339eba89f00c141"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- (a) the reference is main's reader, byte for byte ----------------------------------------------------------

def test_the_python_reference_is_mains_diffgate_byte_for_byte():
    """No normalisation: .gitattributes marks the file -text, so every checkout holds the blob's LF bytes."""
    assert b"\r\n" not in REF_PY.read_bytes()
    assert _sha(REF_PY) == MAIN_DIFFGATE_PY_SHA256


def test_the_javascript_reference_is_mains_port_byte_for_byte():
    assert b"\r\n" not in REF_JS.read_bytes()
    assert _sha(REF_JS) == MAIN_DIFFGATE_JS_SHA256


def test_both_references_are_pinned_against_a_checkout_rewriting_them():
    lines = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert "styxx/_diffgate_ref.py -text" in lines
    assert "web/gate/diffgate_ref.js -text" in lines


def test_the_python_reference_imports_as_its_own_module():
    """Its one relative import (DECLARE-1's reader) resolves inside the package, as it does on main."""
    import styxx._diffgate_ref as ref
    import styxx.declare as declare

    assert ref.__name__ == "styxx._diffgate_ref"
    assert ref._declaration_pass is declare.declaration_pass
    g = ref.gate_diff_text("Modified src/a.py.", "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-a\n+b\n")
    assert [(c.kind, c.verdict) for c in g.claims] == [("file_touched", "VERIFIED")]


def _defs(path: Path) -> dict:
    src = path.read_text(encoding="utf-8")
    return {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def test_the_tests_pass_leg_is_mains_function_unchanged():
    """The guard evaluates the reference without --run and --evidence (a command is never run twice) and reads the
    tests_pass verdict of a measured reference as this function's result: that is sound only while the function is
    main's, byte for byte."""
    ref, new = _defs(REF_PY), _defs(INSTRUMENT)
    for name in ("_tests_pass_verdict", "_evidence_leg", "_run_leg", "DiffClaim", "DiffGate"):
        assert ref[name] == new[name], name


# ---- (b) the switches ---------------------------------------------------------------------------------------------

import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import pytest  # noqa: E402

from styxx import _diffgate_ref as ref  # noqa: E402
from styxx import diffgate as dg  # noqa: E402

PAIRS = {p["id"]: p for p in json.loads((ROOT / "web" / "gate" / "differential" / "path2_pairs.json")
                                        .read_text(encoding="utf-8"))}
REPRO = {"#97": "path2:97-two-readmes", "#121": "path2:121-dotfile-twins", "#101": "path2:101-the-issue"}
PORT = ROOT / "web" / "gate" / "diffgate.js"


def _verdicts(g):
    return [(c.kind, c.verdict) for c in g.claims]


def test_the_switch_set_is_the_three_repairs_and_nothing_else():
    assert dg.REPAIRS == ("#97", "#121", "#101")
    with pytest.raises(ValueError):
        dg._Repairs({"W-1"})
    assert dg._ALL_ON.off == frozenset()


@pytest.mark.parametrize("repair", ["#97", "#121", "#101"])
def test_each_switch_gives_back_mains_verdicts_on_its_own_reproduction(repair):
    """Switched off, a repair's reproduction reads as main reads it; switched on, it is repaired."""
    p = PAIRS[REPRO[repair]]
    main = _verdicts(ref.gate_diff_text(p["summary"], p["diff"]))
    off = _verdicts(dg._evaluate_text(p["summary"], p["diff"], dg._Repairs({repair})))
    on = _verdicts(dg._evaluate_text(p["summary"], p["diff"], dg._ALL_ON))
    assert off == main
    assert on != main


@pytest.mark.parametrize("repair", ["#97", "#121", "#101"])
def test_a_switch_turns_off_its_own_repair_only(repair):
    """The two other reproductions keep their repaired readings with this switch off."""
    for other, pid in REPRO.items():
        if other == repair:
            continue
        p = PAIRS[pid]
        on = _verdicts(dg._evaluate_text(p["summary"], p["diff"], dg._ALL_ON))
        off = _verdicts(dg._evaluate_text(p["summary"], p["diff"], dg._Repairs({repair})))
        assert off == on, (repair, pid)


def test_the_switches_are_a_parameter_not_state():
    """An evaluation with a repair off leaves no trace on the next evaluation with every repair on."""
    p = PAIRS[REPRO["#121"]]
    before = dg.gate_diff_text(p["summary"], p["diff"]).to_dict()
    dg._evaluate_text(p["summary"], p["diff"], dg._Repairs({"#121", "#97", "#101"}))
    assert dg.gate_diff_text(p["summary"], p["diff"]).to_dict() == before


def test_the_git_door_switch_keys_gits_list_by_mains_key(tmp_path):
    """#121 switched off at the git door: `.env` and `env` in git's --name-status are one key, as on main."""
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not on PATH")
    run = lambda *a: subprocess.run([git, *a], cwd=tmp_path, check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    run("config", "user.email", "t@example.invalid")
    run("config", "user.name", "t")
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    run("add", "-A")
    run("commit", "-qm", "base")
    (tmp_path / ".env").write_text("A=1\n", encoding="utf-8")
    (tmp_path / "env").write_text("B=1\n", encoding="utf-8")
    run("add", "-A")
    run("commit", "-qm", "head")
    ns = dg._git(tmp_path, "diff", "--name-status", "HEAD~1..HEAD")
    text = dg._git(tmp_path, "diff", "HEAD~1..HEAD")
    on = dg._evaluate_git("2 files changed.", ns, text, dg._ALL_ON, repo=tmp_path, base="HEAD~1", head="HEAD")
    off = dg._evaluate_git("2 files changed.", ns, text, dg._Repairs({"#121"}), repo=tmp_path, base="HEAD~1",
                           head="HEAD")
    main = ref.gate_diff("2 files changed.", tmp_path, "HEAD~1", "HEAD")
    assert _verdicts(on) == [("files_changed_count", "VERIFIED")]
    assert _verdicts(off) == _verdicts(main) == [("files_changed_count", "CONTRADICTED")]


def test_the_port_switches_read_as_the_python_switches_on_the_reproductions():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH")
    script = ("const B = require(process.argv[1]); const P = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
              "const out = {}; for (const [id, p] of Object.entries(P)) { out[id] = {};"
              " for (const r of B.REPAIRS) out[id][r] = B._evaluate(p.summary, p.diff, { rp: new B._Repairs([r]) })"
              ".claims.map(c => [c.kind, c.verdict, c.why]); }"
              " process.stdout.write(JSON.stringify(out));")
    pairs = {pid: {"summary": PAIRS[pid]["summary"], "diff": PAIRS[pid]["diff"]} for pid in REPRO.values()}
    r = subprocess.run([node, "-e", script, str(PORT)], input=json.dumps(pairs), capture_output=True, text=True,
                       encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    js = json.loads(r.stdout)
    for pid, p in pairs.items():
        for repair in dg.REPAIRS:
            py = [[c.kind, c.verdict, c.why] for c in
                  dg._evaluate_text(p["summary"], p["diff"], dg._Repairs({repair})).claims]
            assert js[pid][repair] == py, (pid, repair)


# ---- (c) the guard ------------------------------------------------------------------------------------------------

from styxx.diffgate import DiffClaim, DiffGate  # noqa: E402

# Round 10's regressions lens, R0.1 (rv10_repros.json): sentences the two ports' templates read apart.
K5_REPROS = {
    "R10-K5A": ("Updated lib/a.pyé/x.md.",
                "diff --git a/other/a.py b/other/a.py\nindex 1111111..2222222 100644\n--- a/other/a.py\n+++ b/other/a.py\n"
                "@@ -1 +1 @@\n-a\n+b\ndiff --git a/x.md b/x.md\nindex 1111111..2222222 100644\n--- a/x.md\n+++ b/x.md\n"
                "@@ -1 +1 @@\n-a\n+b\n"),
    "R10-K5B": ("Changed\u001csrc/util.py and refactored web/util.py.",
                "diff --git a/web/util.py b/web/util.py\nnew file mode 100644\nindex 0000000..2222222\n--- /dev/null\n"
                "+++ b/web/util.py\n@@ -0,0 +1 @@\n+b\n"),
    "R10-K5C": ("Only touches\u001c.docs/ and only touches docs/.",
                "diff --git a/.docs/a.md b/.docs/a.md\nindex 1111111..2222222 100644\n--- a/.docs/a.md\n+++ b/.docs/a.md\n"
                "@@ -1 +1 @@\n-a\n+b\n"),
    "R10-K5D": ("émodified lib/a.py and updated x.md.",
                "diff --git a/x.md b/x.md\nindex 1111111..2222222 100644\n--- a/x.md\n+++ b/x.md\n@@ -1 +1 @@\n-a\n+b\n"
                "diff --git a/other/a.py b/other/a.py\nindex 1111111..2222222 100644\n--- a/other/a.py\n+++ b/other/a.py\n"
                "@@ -1 +1 @@\n-a\n+b\n"),
    # (the same shape with the separators only one port's `\s` holds: U+0085 Python's, U+FEFF JavaScript's)
    "K5-NEL": ("Only touches\u0085.docs/ and only touches docs/.",
               "diff --git a/.docs/a.md b/.docs/a.md\nindex 1111111..2222222 100644\n--- a/.docs/a.md\n+++ b/.docs/a.md\n"
               "@@ -1 +1 @@\n-a\n+b\n"),
    "K5-BOM": ("Only touches﻿.docs/ and only touches docs/.",
               "diff --git a/.docs/a.md b/.docs/a.md\nindex 1111111..2222222 100644\n--- a/.docs/a.md\n+++ b/.docs/a.md\n"
               "@@ -1 +1 @@\n-a\n+b\n"),
}


def _full(g):
    return [(c.kind, c.text, c.verdict, c.why, c.detail) for c in g.claims]


def _stub_gate(*claims, measured=True):
    return DiffGate(verdict="PASS", base="b", head="h", claims=list(claims), measured=measured)


def _claim(kind, verdict, text="s.", **detail):
    return DiffClaim(kind=kind, text=text, detail=dict(detail), verdict=verdict, why=f"{verdict.lower()} here")


def _stub_evaluate(on, switched, status=None, sides=None, apart=None):
    """evaluate(rp, out) for `_guard`: `on` with every repair on, `switched[repair]` with one switched off."""
    def evaluate(rp, out):
        if out is not None:
            out.update(status=status or {}, sides=sides or {}, apart=apart or [False] * len(on.claims))
            return on
        (repair,) = rp.off
        return switched[repair]
    return evaluate


@pytest.mark.parametrize("repair", ["#97", "#121", "#101"])
def test_the_reproductions_stay_repaired_under_the_guard(repair):
    """Each difference from main on a reproduction is licensed by its own repair: the guarded gate is the reading."""
    p = PAIRS[REPRO[repair]]
    assert _full(dg.gate_diff_text(p["summary"], p["diff"])) == \
        _full(dg._evaluate_text(p["summary"], p["diff"], dg._ALL_ON))


def test_a_difference_no_named_repair_explains_abstains_and_names_mains_verdict():
    """A `diff --git` header holding U+2028: F-2's split reads the file main's str.splitlines() does not. No named
    repair explains the difference, so the claims main leaves UNCHECKABLE abstain here too."""
    p = PAIRS["path2:v2-a-binary-header-holding-a-line-separator-registers-its-file"]
    before = dg._evaluate_text(p["summary"], p["diff"], dg._ALL_ON)
    after = dg.gate_diff_text(p["summary"], p["diff"])
    moved = [(b.kind, b.verdict, a.verdict, a.why) for b, a in zip(before.claims, after.claims) if b.verdict != a.verdict]
    assert moved and all(v == "VERIFIED" and w == "UNCHECKABLE" for _k, v, w, _y in moved)
    assert all(y.startswith("main's reading gives UNCHECKABLE and this one VERIFIED; no named repair") for *_x, y in moved)


def test_a_licence_needs_the_switch_and_the_precondition():
    mine, theirs = _claim("files_changed_count", "VERIFIED", n="2"), _claim("files_changed_count", "CONTRADICTED", n="2")
    switched = {"#97": _stub_gate(mine), "#121": _stub_gate(theirs), "#101": _stub_gate(mine)}
    reference = lambda: _stub_gate(theirs)  # noqa: E731
    # #121 switched off gives main's verdict, and a dotted key is in the file list: licensed, kept
    g = dg._guard(_stub_evaluate(_stub_gate(mine), switched, status={".env": "A", "env": "A"}), reference,
                  strict=False, tp=[])
    assert [c.verdict for c in g.claims] == ["VERIFIED"]
    # the same switch, no dotted key anywhere: #121's precondition fails, so the difference abstains
    g = dg._guard(_stub_evaluate(_stub_gate(mine), switched, status={"env": "A", "x": "M"}), reference,
                  strict=False, tp=[])
    assert [(c.verdict, c.why) for c in g.claims] == [("UNCHECKABLE", dg._GUARD_DIFFERS.format(
        main="CONTRADICTED", this="VERIFIED"))]
    # the precondition holds but no switch gives main's verdict: abstains
    switched["#121"] = _stub_gate(mine)
    g = dg._guard(_stub_evaluate(_stub_gate(mine), switched, status={".env": "A"}), reference, strict=False, tp=[])
    assert [c.verdict for c in g.claims] == ["UNCHECKABLE"]


def test_where_main_raises_or_makes_no_such_claim_every_decided_claim_abstains():
    mine = _claim("files_changed_count", "CONTRADICTED", n="2")
    same = {"#97": _stub_gate(mine), "#121": _stub_gate(mine), "#101": _stub_gate(mine)}

    def raises():
        raise AttributeError("'NoneType' object has no attribute 'startswith'")

    g = dg._guard(_stub_evaluate(_stub_gate(mine), same, status={".env": "A"}), raises, strict=False, tp=[])
    assert [(c.verdict, c.why) for c in g.claims] == [("UNCHECKABLE", dg._GUARD_RAISES.format(this="CONTRADICTED"))]
    assert g.verdict == "PASS"
    other = _claim("files_changed_count", "CONTRADICTED", text="another sentence.", n="2")
    g = dg._guard(_stub_evaluate(_stub_gate(mine), same, status={".env": "A"}), lambda: _stub_gate(other),
                  strict=False, tp=[])
    assert [(c.verdict, c.why) for c in g.claims] == [("UNCHECKABLE", dg._GUARD_ABSENT.format(this="CONTRADICTED"))]


def test_an_abstention_and_an_equal_verdict_are_kept_with_their_own_reasons():
    a = _claim("only_touches", "UNCHECKABLE", prefix="x")
    b = _claim("only_touches", "VERIFIED", text="t.", prefix="x")
    main = [_claim("only_touches", "VERIFIED", prefix="x"), _claim("only_touches", "VERIFIED", text="t.", prefix="x")]
    g = dg._guard(_stub_evaluate(_stub_gate(a, b), {}), lambda: _stub_gate(*main), strict=True, tp=[])
    assert g.claims == [a, b]
    assert g.verdict == "FAIL"                       # --strict, recomputed from the final claims: one abstains


def test_the_verdict_is_recomputed_from_the_final_claims():
    mine = _claim("files_changed_count", "CONTRADICTED", n="2")
    same = {"#97": _stub_gate(mine), "#121": _stub_gate(mine), "#101": _stub_gate(mine)}
    on = _stub_gate(mine)
    on.verdict = "FAIL"
    g = dg._guard(_stub_evaluate(on, same), lambda: _stub_gate(_claim("files_changed_count", "VERIFIED", n="2")),
                  strict=False, tp=[])
    assert [c.verdict for c in g.claims] == ["UNCHECKABLE"] and g.verdict == "PASS"


def test_k5_a_claim_of_a_sentence_the_ports_read_apart_reads_as_main_read_it():
    """R10-K5A to R10-K5D: the Python reads each such claim as main's Python did, reason and detail included."""
    for rid, (summary, diff) in K5_REPROS.items():
        assert _full(dg.gate_diff_text(summary, diff)) == _full(ref.gate_diff_text(summary, diff)), rid


def test_k5_the_port_reads_such_a_claim_as_mains_port_read_it():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH")
    script = ("const B = require(process.argv[1]), M = require(process.argv[2]);"
              "const P = JSON.parse(require('fs').readFileSync(0, 'utf8')); const out = {};"
              "for (const [id, [s, d]] of Object.entries(P)) out[id] = [B.gateDiffText(s, d).claims,"
              " M.gateDiffText(s, d).claims];"
              "process.stdout.write(JSON.stringify(out));")
    r = subprocess.run([node, "-e", script, str(PORT), str(REF_JS)], input=json.dumps(K5_REPROS), capture_output=True,
                       text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    for rid, (mine, main) in json.loads(r.stdout).items():
        assert mine == main, rid


# This round's own differential (m10d, soup-101004-238): main's reading keys `b/<U+1C89>.md` and `b/<U+1C8A>.md` as two
# files on Python 3.12 (Unicode 15.0) and one on Node 24 (16.0), so where main's two line splits differ was another place
# in each port, and Z-3's reason named it.
FOLDS_DIFF = ("+++ b/Ᲊ.md\rdiff -Nu a/Src/config.toml b/Src/config.toml\n--- /dev/null\t1970-01-01 00:00:00.000000000 "
              "+0000\x1c+++ b/Src/config.toml\t2024-05-06 07:08:09.000000000 +0000\x1c@@ -0,0 +1,3 @@\n+k723 = 8\x85+++ "
              "/dev/null\t1970-01-01 00:00:00.000000000 +0000\rdiff --cc m.py +k271 = 1\n+k279 = 8\x0b\x0b+++ b/ᲊ.md\n")


def test_z3_asks_first_whether_mains_paths_fold_alike():
    assert dg._folds_apart(["Ᲊ.md", "ᲊ.md"]) and dg._folds_apart(["docs/Guide.md", "./docs/guide.md"])
    assert not dg._folds_apart(["docs/a.md", "docs/a.md", "./docs/a.md"])
    g = dg.gate_diff_text("3 files changed. Only touches assets/.", FOLDS_DIFF)
    assert [(c.kind, c.verdict) for c in g.claims] == [("files_changed_count", "UNCHECKABLE"),
                                                        ("only_touches", "UNCHECKABLE")]
    assert all(c.why.endswith(dg._Z3_FOLDS) or dg._Z3_FOLDS in c.why for c in g.claims)
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH")
    script = ("const B = require(process.argv[1]); const [s, d] = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
              "process.stdout.write(JSON.stringify(B.gateDiffText(s, d).claims.map(c => [c.kind, c.verdict, c.why])));")
    r = subprocess.run([node, "-e", script, str(PORT)], input=json.dumps(["3 files changed. Only touches assets/.",
                                                                         FOLDS_DIFF]),
                       capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert json.loads(r.stdout) == [[c.kind, c.verdict, c.why] for c in g.claims]


def test_k5_reads_the_sentence_not_every_non_ascii_character():
    """Punctuation, symbols and emoji read alike in both ports' templates: the em dash of `path -- created.` (U+2014, a
    literal in the template itself) leaves #97's repair in place; a word character or a split mark does not."""
    assert dg._apart_readings("integrations/git/README.md — created.") == (False, False)
    assert dg._apart_readings("Added 2 tests ✅ «» ’") == (False, False)
    assert dg._apart_readings("Updated lib/a.pyé/x.md.") == (True, True)
    assert dg._apart_readings("Changed\u001csrc/util.py") == (True, True)
    assert dg._apart_readings("a\rb.py: x") == (True, True)
    for mark in ("\u001f", "\u0085", "﻿", " ", " "):
        assert dg._apart_readings(f"Only touches{mark}docs/.") == (True, True), hex(ord(mark))
    assert dg._apart_readings("Updated a.py.\r") == (False, False)
    assert dg._apart_readings("Added function café.") == (True, False)          # the name is W-2's in both ports
    assert dg._apart_readings("éadded function foo.") == (True, True)          # a word character outside the name
    g = dg.gate_diff_text("integrations/git/README.md — created.", PAIRS["path2:97-two-readmes"]["diff"])
    m = ref.gate_diff_text("integrations/git/README.md — created.", PAIRS["path2:97-two-readmes"]["diff"])
    assert [(c.kind, c.verdict) for c in g.claims if c.kind == "file_created"] == [("file_created", "VERIFIED")]
    assert [(c.kind, c.verdict) for c in m.claims if c.kind == "file_created"] == [("file_created", "UNCHECKABLE")]


def test_tests_pass_runs_its_command_once_and_reads_as_main(tmp_path):
    """The reference is run without --run; its tests_pass verdict is the shared leg's answer where it is measured."""
    counter = tmp_path / "ran.txt"
    cmd = f'"{sys.executable}" -c "open(r\'{counter}\', \'a\').write(\'x\')"'
    diff = "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-a = 1\n+a = 2\n"
    g = dg.gate_diff_text("All tests pass. Modified src/a.py.", diff, run=cmd, repo=tmp_path)
    assert counter.read_text() == "x"
    assert sorted((c.kind, c.verdict) for c in g.claims) == [("file_touched", "VERIFIED"), ("tests_pass", "VERIFIED")]


def test_the_git_door_is_guarded_against_mains_git_door_on_the_same_range(tmp_path, monkeypatch):
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not on PATH")
    run = lambda *a: subprocess.run([git, *a], cwd=tmp_path, check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    run("config", "user.email", "t@example.invalid")
    run("config", "user.name", "t")
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    run("add", "-A")
    run("commit", "-qm", "base")
    (tmp_path / ".env").write_text("A=1\n", encoding="utf-8")
    (tmp_path / "env").write_text("B=1\n", encoding="utf-8")
    run("add", "-A")
    run("commit", "-qm", "head")
    seen = []
    real = ref.gate_diff

    def recorded(summary, repo, base, head, **kw):
        seen.append((summary, Path(repo), base, head, kw))
        return real(summary, repo, base, head, **kw)

    monkeypatch.setattr(dg._REF, "gate_diff", recorded)
    g = dg.gate_diff("2 files changed.", tmp_path, "HEAD~1", "HEAD")
    assert [(c.kind, c.verdict) for c in g.claims] == [("files_changed_count", "VERIFIED")]     # #121, licensed
    assert seen == [("2 files changed.", tmp_path, "HEAD~1", "HEAD",
                     {"run": None, "strict": False, "evidence": None, "commit": None})]


# ---- 3. THE GUARANTEE ------------------------------------------------------------------------------------------------
#
# Stated (NOTE_path2_eleventh_pass, B): for every claim, the final verdict is main's, or UNCHECKABLE, or this reading's own
# licensed by one named repair -- #97, #121 or #101 -- whose precondition holds on the claim and whose switch, alone, gives
# main's verdict back. Held here claim by claim over every committed corpus (the pinned pairs of every file, and the
# differential corpora where a checkout has built them) and a seeded randomised set; the scratch harness of this round
# runs the same check over every reviewer's harness set and more than 10,000 fresh cases.

import random  # noqa: E402

DIFFERENTIAL = ROOT / "web" / "gate" / "differential"
PAIR_FILES = ("bc1_pairs.json", "compat_pairs.json", "bin1_pairs.json", "compat2_pairs.json", "path1_pairs.json",
              "declare1_pairs.json", "path2_pairs.json")


def guarantee_violations(summary: str, diff: str, module=dg) -> tuple:
    """(claims read, [claims that break the guarantee]) for one record, through `module`'s raw door."""
    final = module.gate_diff_text(summary, diff)
    try:
        main = ref.gate_diff_text(summary, diff)
        theirs = dict(zip(dg._claim_keys(main.claims), main.claims))
    except Exception:
        theirs = None
    seen: dict = {}
    before = module._evaluate_text(summary, diff, module._ALL_ON, out=seen)
    switched = {r: dict(zip(dg._claim_keys(g.claims), g.claims))
                for r in dg.REPAIRS for g in [module._evaluate_text(summary, diff, module._Repairs({r}))]}
    bad = []
    for key, c, b in zip(dg._claim_keys(final.claims), final.claims, before.claims):
        m = None if theirs is None else theirs.get(key)
        mv = None if m is None else m.verdict
        if c.verdict in ("UNCHECKABLE", mv):
            continue
        if not (mv is not None and c.verdict == b.verdict
                and any(switched[r].get(key) is not None and switched[r][key].verdict == mv
                        and module._precondition(r, b, seen["status"], seen["sides"]) for r in dg.REPAIRS)):
            bad.append((key, c.verdict, mv))
    if final.verdict != ("FAIL" if any(c.verdict == "CONTRADICTED" for c in final.claims) else "PASS"):
        bad.append(("gate verdict", final.verdict, None))
    return len(final.claims), bad


def _pinned():
    for name in PAIR_FILES:
        for p in json.loads((DIFFERENTIAL / name).read_text(encoding="utf-8")):
            yield p["id"], p["summary"], p["diff"]


# The randomised set: the shapes every round's review built its regressions from, crossed at random.
_G_FILES = [".env", "env", ".github/x.yml", "github/x.yml", ".pr_agent.toml", "pr_agent.toml", "README.md",
            "integrations/git/README.md", "docs/README.md", "src/node/glob.ts", "glob.ts", "src/a.py", "lib/a.py",
            "tests/test_a.py", "b/x.py", "x.py", "docs/Guide.md", "docs/guide.md", "db/q.sql", "src/café.py",
            "docs/é.md", "a.py ", "sp ace.py", "tests/Ɤt.py", "docs/x\U00010d50.md", "docs/x\U00010d70.md"]
_G_LINES = ["x = 1", "def test_a():", "def test_b(x):", "async def test_c():", "def helper():", "class K:", "-- users",
            "++ x", "﻿def test_d():", "\x0cdef test_e():", " def test_f():", "a b", "c\rd", "    pass",
            "def café():", "def get_नमस्ते():", "SELECT 1;"]
_G_SEPS = [" ", " ", " ", "\n", "  ", "\x1c", " ", " — ", " ✅ ", "’ ", "\t"]


def _g_header(rng, path: str, st: str, style: str) -> str:
    a = "/dev/null" if st == "A" else path
    b = "/dev/null" if st == "D" else path
    if style == "git":
        q = '"' if " " in path else ""
        mode = {"A": "new file mode 100644\n", "D": "deleted file mode 100644\n"}.get(st, "")
        return (f"diff --git {q}a/{path}{q} {q}b/{path}{q}\n{mode}index 1111111..2222222\n"
                f"--- {'/dev/null' if st == 'A' else q + 'a/' + path + q}\n"
                f"+++ {'/dev/null' if st == 'D' else q + 'b/' + path + q}\n")
    if style == "noprefix":
        return f"diff --git {path} {path}\n--- {a}\n+++ {b}\n"
    if style == "gnu":
        ts = "\t2024-05-06 07:08:09.000000000 +0000"
        return f"--- {'/dev/null' if st == 'A' else 'a/' + path}{ts}\n+++ {'/dev/null' if st == 'D' else 'b/' + path}{ts}\n"
    return f"--- {a}\n+++ {b}\n"


def _g_hunk(rng, st: str) -> str:
    old = [] if st == "A" else [rng.choice(_G_LINES) for _ in range(rng.randint(0, 3))]
    new = [] if st == "D" else [rng.choice(_G_LINES) for _ in range(rng.randint(0, 3))]
    ctx = [] if st != "M" else [rng.choice(_G_LINES) for _ in range(rng.randint(0, 2))]
    if not (old or new or ctx):
        new = ["x = 2"] if st != "D" else []
        old = old or (["x = 1"] if st == "D" else [])
    b, d = len(old) + len(ctx), len(new) + len(ctx)
    if rng.random() < 0.2:                            # a hunk that declares otherwise than it carries
        b, d = max(0, b + rng.choice((-1, 1))), max(0, d + rng.choice((-1, 1)))
    body = [" " + x for x in ctx] + ["-" + x for x in old] + ["+" + x for x in new]
    return f"@@ -{0 if st == 'A' else 1},{b} +{0 if st == 'D' else 1},{d} @@\n" + "".join(x + "\n" for x in body)


def _g_diff(rng) -> str:
    out = []
    for path in rng.sample(_G_FILES, rng.randint(1, 5)):
        st = rng.choice("AMMD")
        style = rng.choice(("git", "git", "git", "noprefix", "gnu", "plain"))
        if rng.random() < 0.08:
            out.append(f"diff --git a/{path} b/{path}\nindex 1111111..2222222 100644\n"
                       f"Binary files a/{path} and b/{path} differ\n")
            continue
        out.append(_g_header(rng, path, st, style) + _g_hunk(rng, st))
        if rng.random() < 0.05:
            out.append("\\ No newline at end of file\n")
    extra = rng.random()
    if extra < 0.05:
        out.append("Submodule vendor/lib 1234567..89abcde:\n")
    elif extra < 0.08:
        out.append("+++ /dev/null\n")
    elif extra < 0.1:
        out.append("Index: img/logo.png\n===\nCannot display: file marked as a binary type.\n")
    text = "".join(out)
    if rng.random() < 0.08:
        text = text.replace("\n", "\r\n")
    return text


def _g_sentence(rng) -> str:
    p = rng.choice(_G_FILES).strip()
    r = rng.random()
    if r < 0.18:
        return f"{rng.choice(['Modified', 'Updated', 'Edited', 'émodified', 'Refactored'])} {p}."
    if r < 0.3:
        return rng.choice([f"Created {p}.", f"{p} — created.", f"New file {p}.", f"Created file {p}."])
    if r < 0.38:
        return f"{rng.choice(['Deleted', 'Removed'])} {p}."
    if r < 0.52:
        return f"{rng.randint(0, 6)} files changed."
    if r < 0.64:
        return f"Added {rng.randint(0, 4)} {rng.choice(['', 'new '])}tests."
    if r < 0.74:
        return f"Adds {rng.choice(['function', 'class'])} {rng.choice(['helper', 'K', 'test_a', 'café', 'zap'])}."
    if r < 0.88:
        pre = rng.choice(["src/", "docs/", "github/", ".github/", "env", ".env", "tests", "db/", "lib/"])
        two = rng.choice(["", f" and {rng.choice(['src/', 'docs/', 'tests/'])}"])
        return f"Only touches {pre}{two}."
    if r < 0.94:
        return rng.choice(["No breaking changes.", "All tests pass.", "Keeps backward compatibility."])
    return "```styxx\nfiles_changed: 2\ntests_added: 1\n```"


def guard_cases(seed: int, n: int):
    rng = random.Random(seed)
    for i in range(n):
        summary = "".join(_g_sentence(rng) + rng.choice(_G_SEPS) for _ in range(rng.randint(1, 5)))
        yield f"guard:{seed}:{i}", summary, _g_diff(rng)


def test_the_guarantee_holds_on_every_pinned_pair():
    claims = 0
    for pid, summary, diff in _pinned():
        n, bad = guarantee_violations(summary, diff)
        claims += n
        assert not bad, (pid, bad)
    assert claims > 500


@pytest.mark.parametrize("name", ["corpus_fuzz.json", "corpus_real.json"])
def test_the_guarantee_holds_on_the_differential_corpora(name):
    path = DIFFERENTIAL / name
    if not path.exists():
        pytest.skip(f"{name} is built by the differential's own scripts and not committed")
    for it in json.loads(path.read_text(encoding="utf-8")):
        _n, bad = guarantee_violations(it["summary"], it["diff"])
        assert not bad, (it["id"], bad)


def test_the_guarantee_holds_on_a_seeded_randomised_set():
    claims = moved = 0
    for pid, summary, diff in guard_cases(20260928, 1500):
        n, bad = guarantee_violations(summary, diff)
        claims += n
        assert not bad, (pid, bad)
    assert claims > 3500


# ---- 3. mutation: a defect planted in the reader outside the three repairs --------------------------------------------

def _mutant(good: str, bad: str, tag: str):
    """A copy of styxx/diffgate.py with one defect planted, loaded inside the package (its reference is the real one)."""
    import types
    src = INSTRUMENT.read_text(encoding="utf-8")
    assert src.count(good) == 1, good
    mod = types.ModuleType(f"styxx._diffgate_mutant_{tag}")
    mod.__package__, mod.__file__ = "styxx", f"<mutant {tag}>"
    sys.modules[mod.__name__] = mod
    exec(compile(src.replace(good, bad), mod.__file__, "exec"), mod.__dict__)  # noqa: S102
    return mod


MUTANTS = {
    # label: (good, bad) -- each outside #97's tiers, #121's key and #101's pairing
    "W-1: every hunk exact": ("def _hunk_is_exact(lines: list, k: int) -> bool:\n",
                              "def _hunk_is_exact(lines: list, k: int) -> bool:\n    return True\n"),
    "Y-4: /dev/null read with anything after it": ('    return path == "/dev/null" or (path or "").startswith("/dev/null\\t")',
                                                    '    return (path or "").startswith("/dev/null")'),
    "the test count reads async tests": ('    return sum(1 for line in added_blob.split("\\n") if _test_name(line))',
                                         '    return sum(1 for line in added_blob.split("\\n") if _test_name(line, True))'),
    "BC-1: every diff holds Python": ("    return any(_undotted(p).lower().endswith(_PY_SUFFIXES) for p in status)",
                                      "    return True"),
    "A-1: an added async test no longer abstains": ("                        if unread:", "                        if False:"),
    "the count off by one": ('                        c.verdict = "VERIFIED" if n == len(status) else "CONTRADICTED"',
                             '                        c.verdict = "VERIFIED" if n == len(status) + 1 else "CONTRADICTED"'),
    "only_touches containment by string prefix": ("    return path == pref or path.startswith(pref + \"/\")",
                                                  "    return path == pref or path.startswith(pref)"),
    "F-2 breaks a line at a form feed": ('_DIFF_LINE_BREAK = re.compile(r"\\r\\n|\\r|\\n")',
                                         '_DIFF_LINE_BREAK = re.compile(r"\\r\\n|\\r|\\n|\\x0c")'),
}


# Not among them: Z-3's and Y-1's doubts. They are #121's licence's own defence -- each abstains where #121's dotted split may
# have removed an error that balanced main's merge of dotfile twins -- so a defect there passes through the licence (the
# surface the guarantee names), and the reading oracles refuse it (tests/test_diffgate_path2.py, the Z-3 and Y-1 plants).
def _mutation_records():
    yield from _pinned()
    yield from guard_cases(20260929, 600)


@pytest.mark.parametrize("label", sorted(MUTANTS))
def test_a_defect_outside_the_three_repairs_can_only_abstain(label):
    """Every claim of the mutant's final gate reads main's verdict, UNCHECKABLE, or the clean branch's own final verdict:
    the defect can only take a verdict away. Without the guard the same defect gives verdicts none of those is."""
    mutant = _mutant(*MUTANTS[label], tag=label.split(":")[0].replace(" ", "_").replace("-", "_"))
    live = unguarded_new = 0
    for pid, summary, diff in _mutation_records():
        try:
            main = ref.gate_diff_text(summary, diff)
            theirs = dict(zip(dg._claim_keys(main.claims), main.claims))
        except Exception:
            theirs = {}
        clean = dg.gate_diff_text(summary, diff)
        mine = mutant.gate_diff_text(summary, diff)
        before = mutant._evaluate_text(summary, diff, mutant._ALL_ON)
        clean_by = dict(zip(dg._claim_keys(clean.claims), clean.claims))
        for key, c, b in zip(dg._claim_keys(mine.claims), mine.claims, before.claims):
            allowed = {"UNCHECKABLE", getattr(theirs.get(key), "verdict", None), getattr(clean_by.get(key), "verdict", None)}
            assert c.verdict in allowed, (label, pid, key, c.verdict, allowed)
            live += c.verdict != getattr(clean_by.get(key), "verdict", None)
            unguarded_new += b.verdict not in allowed
    assert live or unguarded_new, f"{label}: the mutant moved nothing on these records"
    assert unguarded_new, f"{label}: without the guard the mutant gives no new verdict here (an equivalent mutant)"
