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
