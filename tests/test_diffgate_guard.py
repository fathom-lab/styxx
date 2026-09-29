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
    # a count over #121's dotfile twins, in a sentence holding a word character: main's count, not the licensed one
    "K5-COUNT": ("3 files changed café.", PAIRS["path2:121-dotfile-twins"]["diff"]),
}


def _full(g):
    return [(c.kind, c.text, c.verdict, c.why, c.detail) for c in g.claims]


def _stub_gate(*claims, measured=True):
    return DiffGate(verdict="PASS", base="b", head="h", claims=list(claims), measured=measured)


def _claim(kind, verdict, text="s.", **detail):
    return DiffClaim(kind=kind, text=text, detail=dict(detail), verdict=verdict, why=f"{verdict.lower()} here")


def _git_licence(status):
    """What the licences read of a diff in git's own rendering, with no Z-3 doubt, each key read from itself, once."""
    return {"rendered": True, "soft": False, "forms": {k: [k] for k in (status or {})}, "multi": set()}


class _NotOneRepair(AssertionError):
    """The guard asked a switched-off reading with other than exactly one repair off."""


def _stub_evaluate(on, switched, status=None, sides=None, apart=None, licence=None):
    """evaluate(rp, out) for `_guard`: `on` with every repair on, `switched[repair]` with one switched off; `licence` what
    the licences read (NOTE_path2_twelfth_pass), by default git's own rendering with no doubt.
    NOTE_path2_thirteenth_pass (round 12, PA): the guard's rule is ONE repair switched off at a time -- a licence read off
    two or three repairs reverted together is not the operator's -- so a stub asked otherwise fails by that rule."""
    def evaluate(rp, out):
        if out is not None:
            out.update(status=status or {}, sides=sides or {}, apart=apart or [False] * len(on.claims),
                       licence=_git_licence(status) if licence is None else licence)
            return on
        if len(rp.off) != 1:
            raise _NotOneRepair(f"the guard asked a reading with {sorted(rp.off)} switched off, not one repair")
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


def test_the_tightened_licences_on_stubs():
    """NOTE_path2_twelfth_pass (A.1, A.2): #121 licenses only in git's own rendering, and on a path claim only where the
    resolved entry matches the claim case kept by its tier; #97 only on a case-kept exact or suffix match and never beside
    a Z-3 doubt. Each stub has the switch giving main's verdict back, so only the precondition decides."""
    twins = {".env": "A", "env": "A"}
    mine, theirs = _claim("files_changed_count", "VERIFIED", n="2"), _claim("files_changed_count", "CONTRADICTED", n="2")
    switched = {"#97": _stub_gate(mine), "#121": _stub_gate(theirs), "#101": _stub_gate(mine)}

    def final(on, status, licence, reference):
        return [(c.verdict, c.why) for c in dg._guard(_stub_evaluate(_stub_gate(on), switched, status=status,
                                                                     licence=licence), reference, strict=False,
                                                      tp=[]).claims]
    plain = {"rendered": False, "soft": False, "forms": {k: [k] for k in twins}}
    assert final(mine, twins, None, lambda: _stub_gate(theirs)) == [("VERIFIED", "verified here")]
    assert final(mine, twins, plain, lambda: _stub_gate(theirs)) == [
        ("UNCHECKABLE", dg._GUARD_DIFFERS.format(main="CONTRADICTED", this="VERIFIED"))]
    # #121 on a path claim resolved only in case
    path_mine = _claim("file_created", "VERIFIED", text="Created X.toml.", path="X.toml")
    path_main = _claim("file_created", "UNCHECKABLE", text="Created X.toml.", path="X.toml")
    switched.update({"#121": _stub_gate(path_main), "#97": _stub_gate(path_mine)})
    listing = {".config/x.toml": "A", "config/x.toml": "M"}
    assert final(path_mine, listing, None, lambda: _stub_gate(path_main))[0][0] == "UNCHECKABLE"
    # #97: a case-only suffix match, and a Z-3 doubt
    created = _claim("file_created", "VERIFIED", text="Created src/README.md.", path="src/README.md")
    withheld = _claim("file_created", "UNCHECKABLE", text="Created src/README.md.", path="src/README.md")
    switched.update({"#97": _stub_gate(withheld), "#121": _stub_gate(created)})
    listing = {"docs/readme.md": "M", "lib/src/readme.md": "A"}
    only_case = {"rendered": True, "soft": False, "forms": {"docs/readme.md": ["docs/README.md"],
                                                            "lib/src/readme.md": ["lib/src/readme.md"]}}
    kept = {"rendered": True, "soft": False, "forms": {"docs/readme.md": ["docs/README.md"],
                                                       "lib/src/readme.md": ["lib/src/README.md"]}}
    assert final(created, listing, only_case, lambda: _stub_gate(withheld))[0][0] == "UNCHECKABLE"
    assert final(created, listing, kept, lambda: _stub_gate(withheld))[0][0] == "VERIFIED"
    assert final(created, listing, dict(kept, soft=True), lambda: _stub_gate(withheld))[0][0] == "UNCHECKABLE"
    # NOTE_path2_thirteenth_pass (A.1): the same licence, where the resolved entry is a key two file sections registered
    assert final(created, listing, dict(kept, multi={"lib/src/readme.md"}), lambda: _stub_gate(withheld))[0][0] == \
        "UNCHECKABLE"
    assert final(created, listing, dict(kept, multi={"docs/readme.md"}), lambda: _stub_gate(withheld))[0][0] == \
        "VERIFIED"


def test_the_guard_asks_one_switched_off_repair_at_a_time():
    """NOTE_path2_thirteenth_pass (round 12, PA): a licence is one repair switched off giving main's verdict back on its
    own precondition. The stub refuses any other question, so a guard that asked two or three repairs reverted together
    fails here by that rule, not by an unpacking error."""
    mine, theirs = _claim("files_changed_count", "VERIFIED", n="2"), _claim("files_changed_count", "CONTRADICTED", n="2")
    switched = {"#97": _stub_gate(mine), "#121": _stub_gate(theirs), "#101": _stub_gate(mine)}
    g = dg._guard(_stub_evaluate(_stub_gate(mine), switched, status={".env": "A", "env": "A"}),
                  lambda: _stub_gate(theirs), strict=False, tp=[])
    assert [c.verdict for c in g.claims] == ["VERIFIED"]
    with pytest.raises(_NotOneRepair):
        _stub_evaluate(_stub_gate(mine), switched)(dg._Repairs(dg.REPAIRS), None)


def test_git_own_rendering_is_read_alike_in_both_ports():
    """The facts #121's licence reads, on the renderings round 11 and this pass name, the same in the Python and the port."""
    diffs = {
        "git": "diff --git a/.env b/.env\nindex 1..2 100644\n--- a/.env\n+++ b/.env\n@@ -1 +1 @@\n-a\n+b\n",
        "an empty created file": "diff --git a/p/__init__.py b/p/__init__.py\nnew file mode 100644\nindex 0000000..e69de29\n",
        "a quoted path": 'diff --git "a/sp\\tx.py" "b/sp\\tx.py"\n--- "a/sp\\tx.py"\n+++ "b/sp\\tx.py"\n@@ -1 +1 @@\n-a\n+b\n',
        "a name ending in a space": "diff --git a/x.py  b/x.py \nnew file mode 100644\n--- /dev/null\n+++ b/x.py \t\n@@ -0,0 +1 @@\n+a\n",
        "difflib": "--- a/.env\n+++ b/.env\n@@ -1 +1 @@\n-a\n+b\n",
        "GNU per file": "--- a/.env\t2024-05-06 07:08:09.000000000 +0000\n+++ b/.env\t2024-05-06 07:08:09.000000000 +0000\n"
                        "@@ -1 +1 @@\n-a\n+b\n",
        "--no-prefix": "diff --git .env .env\n--- .env\n+++ .env\n@@ -1 +1 @@\n-a\n+b\n",
        "mnemonic prefixes": "diff --git c/.env w/.env\n--- c/.env\n+++ w/.env\n@@ -1 +1 @@\n-a\n+b\n",
        "a pair naming another file": "diff --git a/x b/x\n--- a/y\n+++ b/y\n@@ -1 +1 @@\n-a\n+b\n",
        "a +++ line naming another file": "diff --git a/x b/x\n--- a/x\n+++ b/y\n@@ -1 +1 @@\n-a\n+b\n",
        "a rename under --no-prefix": "diff --git a/x.py b/x.py\nsimilarity index 100%\nrename from a/x.py\nrename to b/x.py\n",
        "a Submodule line": "diff --git a/.env b/.env\n--- a/.env\n+++ b/.env\n@@ -1 +1 @@\n-a\n+b\nSubmodule v 1234567..89abcde:\n",
        "no --- line": "diff --git a/x b/x\n+++ /dev/null\n",
    }
    want = {"git": True, "an empty created file": True, "a quoted path": True, "a name ending in a space": True,
            "difflib": False, "GNU per file": False, "--no-prefix": False, "mnemonic prefixes": False,
            "a pair naming another file": False, "a +++ line naming another file": False,
            "a rename under --no-prefix": False, "a Submodule line": False,
            "no --- line": False}
    py = {}
    for name, diff in diffs.items():
        facts: dict = {}
        dg._diff_notes(diff, facts)
        py[name] = facts["rendered"]
    assert py == want
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH")
    script = ("const B = require(process.argv[1]); const D = JSON.parse(require('fs').readFileSync(0, 'utf8')); const out = {};"
              "for (const [k, d] of Object.entries(D)) { const o = {}; B._evaluate('3 files changed.', d, { out: o });"
              " out[k] = o.licence.rendered; } process.stdout.write(JSON.stringify(out));")
    r = subprocess.run([node, "-e", script, str(PORT)], input=json.dumps(diffs), capture_output=True, text=True,
                       encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads(r.stdout) == want


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


PORT_GUARD_SCRIPT = r"""
const B = require(process.argv[1]);
const claim = (kind, verdict, text = "s.", detail = {}) => ({ kind, text, detail, verdict, why: verdict.toLowerCase() + " here" });
const gate = (...claims) => ({ diffgate: "v0", verdict: "PASS", base: "b", head: "h", claims, uncovered_sentences: 0,
                               sentences_total: 1, uncovered_texts: [], unparsed_claims: [], measured: true, why_unmeasured: "" });
const stub = (on, switched, keys) => (rp, out) => {
  if (out !== null) { Object.assign(out, { status: new Map(keys.map(k => [k, "A"])), sides: new Map(), apart: on.claims.map(() => false),
                                           licence: { rendered: true, soft: false, forms: new Map(keys.map(k => [k, [k]])) } }); return on; }
  return switched[[...rp.off][0]];
};
const mine = claim("files_changed_count", "VERIFIED", "s.", { n: "2" }), theirs = claim("files_changed_count", "CONTRADICTED", "s.", { n: "2" });
const out = {};
const sw = { "#97": gate(mine), "#121": gate(theirs), "#101": gate(mine) };
out.licensed = B._guard(stub(gate(mine), sw, [".env", "env"]), () => gate(theirs), false).claims.map(c => c.verdict);
out.no_precondition = B._guard(stub(gate(mine), sw, ["env", "x"]), () => gate(theirs), false).claims.map(c => [c.verdict, c.why]);
const sw2 = { "#97": gate(mine), "#121": gate(mine), "#101": gate(mine) };
out.no_switch = B._guard(stub(gate(mine), sw2, [".env"]), () => gate(theirs), false).claims.map(c => c.verdict);
const acc = claim("files_changed_count", "CONTRADICTED", "s.", { n: "2" }), ver = claim("files_changed_count", "VERIFIED", "s.", { n: "2" });
const on = gate(acc); on.verdict = "FAIL";
const g = B._guard(stub(on, { "#97": gate(acc), "#121": gate(acc), "#101": gate(acc) }, ["x"]), () => gate(ver), false);
out.recomputed = [g.verdict, g.claims.map(c => c.verdict)];
out.raises = B._guard(stub(gate(acc), { "#97": gate(acc), "#121": gate(acc), "#101": gate(acc) }, ["x"]),
                      () => { throw new TypeError("x"); }, false).claims.map(c => [c.verdict, c.why]);
const u = claim("only_touches", "UNCHECKABLE", "s.", { prefix: "x" }), v = claim("only_touches", "VERIFIED", "t.", { prefix: "x" });
const s = B._guard(stub(gate(u, v), {}, ["x"]), () => gate(claim("only_touches", "VERIFIED", "s."), claim("only_touches", "VERIFIED", "t.")), true);
out.strict = [s.verdict, s.claims.map(c => c.why)];
process.stdout.write(JSON.stringify(out));
"""


def test_the_ports_guard_reads_as_the_pythons_on_the_same_stubs():
    """The port's `_guard` against the Python's own stub cases: a licence needs the switch and the precondition, main
    raising or absent abstains, the verdict and --strict are recomputed from the final claims."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH")
    r = subprocess.run([node, "-e", PORT_GUARD_SCRIPT, str(PORT)], capture_output=True, text=True, encoding="utf-8",
                       timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout)
    assert out["licensed"] == ["VERIFIED"]
    assert out["no_precondition"] == [["UNCHECKABLE", dg._GUARD_DIFFERS.format(main="CONTRADICTED", this="VERIFIED")]]
    assert out["no_switch"] == ["UNCHECKABLE"]
    assert out["recomputed"] == ["PASS", ["UNCHECKABLE"]]
    assert out["raises"] == [["UNCHECKABLE", dg._GUARD_RAISES.format(this="CONTRADICTED")]]
    assert out["strict"] == ["FAIL", ["uncheckable here", "verified here"]]


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
        # and beside a character both templates read alike (the sentence is not ASCII, so not the fast path)
        assert dg._apart_readings(f"Only touches{mark}docs/ — done.") == (True, True), hex(ord(mark))
    assert dg._apart_readings("a\rb.py: x ✅") == (True, True)
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

import base64  # noqa: E402
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
                        and module._precondition(r, b, seen["status"], seen["sides"], seen["licence"])
                        for r in dg.REPAIRS)):
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
def test_a_defect_outside_the_three_repairs_can_only_abstain(label, request):
    """Every claim of the mutant's final gate reads main's verdict, UNCHECKABLE, or the clean branch's own final verdict:
    the defect can only take a verdict away. Without the guard the same defect gives verdicts none of those is.

    NOTE_path2_twelfth_pass: except where the defect passes through a licence -- the eleventh pass's section B.3, stated
    and now met: over git-rendered dotfile twins, "4 files changed." beside a count off by one reads VERIFIED, and with
    #121 switched off the same defect reads 3 files as not 4, CONTRADICTED, main's verdict by coincidence, so #121's
    licence carries the defect's verdict. The guard cannot see that; the scorer's reading oracle (G-C7) does, and each
    such record is held to it here."""
    mutant = _mutant(*MUTANTS[label], tag=label.split(":")[0].replace(" ", "_").replace("-", "_"))
    live = unguarded_new = 0
    through: list = []
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
        seen: dict = {}
        mutant._evaluate_text(summary, diff, mutant._ALL_ON, out=seen)
        for key, c, b in zip(dg._claim_keys(mine.claims), mine.claims, before.claims):
            allowed = {"UNCHECKABLE", getattr(theirs.get(key), "verdict", None), getattr(clean_by.get(key), "verdict", None)}
            if c.verdict not in allowed:
                # only through a licence: some repair's precondition holds and the mutant's own switch gives main's verdict
                switched = {r: dict(zip(dg._claim_keys(g.claims), g.claims))
                            for r in dg.REPAIRS for g in [mutant._evaluate_text(summary, diff, mutant._Repairs({r}))]}
                assert any(mutant._precondition(r, b, seen["status"], seen["sides"], seen["licence"])
                           and getattr(switched[r].get(key), "verdict", None) == theirs[key].verdict
                           for r in dg.REPAIRS), (label, pid, key, c.verdict, allowed)
                through.append((pid, summary, diff))
            live += c.verdict != getattr(clean_by.get(key), "verdict", None)
            unguarded_new += b.verdict not in allowed
    assert live or unguarded_new, f"{label}: the mutant moved nothing on these records"
    assert unguarded_new, f"{label}: without the guard the mutant gives no new verdict here (an equivalent mutant)"
    if through:
        pg = request.getfixturevalue("scorer")
        for pid, summary, diff in dict.fromkeys(through):
            reading = mutant._evaluate_text(summary, diff, mutant._ALL_ON)
            assert pg.oracle_violations(summary, diff, pg.raw_paths(diff), reading), (label, pid)


# ---- 4. THE GUARANTEE JUDGED AGAINST TRUTH, AND AGAINST THE SCORER'S OWN GUARD ------------------------------------------
#
# NOTE_path2_twelfth_pass_2026_09_29, B. Everything above in section 3 asks the module's own `_precondition` and switches
# whether a difference is licensed: that is self-consistency, and round 11 found licensed differences that were false
# (R11.0 to R11.4) while it reported 0 violations. Here the round-11 reproductions are judged by a truth model that reads
# no line of the instrument -- the file lists and statuses from the case's own base and head trees (and, at the git door,
# git's `--name-status` on a repository built from them), path claims by the claim as written against those paths, case
# kept -- and the guarantee is held to the scorer's own guard (`path2_gates.expected_guard`: main's verdict from the
# baseline, the switched readings from the scorer's reverts, the preconditions from the scorer's code) on both doors.

R11 = json.loads((ROOT / "tests" / "fixtures" / "path2_round11_repros.json").read_text(encoding="utf-8"))["cases"]
# and the records this pass's own truth-judged differential found against its licences as committed at e4586637 (#121
# licensing a path claim whose entry matched it only once lower-cased)
R12 = json.loads((ROOT / "tests" / "fixtures" / "path2_twelfth_pass_repros.json").read_text(encoding="utf-8"))["cases"]
# NOTE_path2_thirteenth_pass_2026_09_29 (E): round 12's reproductions -- git's typechange through #97 and #121, and a name
# ending in whitespace in difflib's rendering through #97 -- each with a model whose entries carry their modes
R13 = json.loads((ROOT / "tests" / "fixtures" / "path2_round12_repros.json").read_text(encoding="utf-8"))["cases"]
TRUTH_CASES = R11 + R12 + R13
# the twelfth pass's instrument (0f559a87) reads 30 of the 34 worse than main on the raw door: every typechange under git's
# default diff.submodule (under diff.submodule=log the Submodule line's doubt already stopped #97) and every whitespace name;
# not the two controls
R13_WORSE_AT_0F559A87 = 30


def _entry(value):
    """(type, mode, content) of one model entry, or None where the path is absent: a str is a regular file (100644),
    {"x": text} an executable (100755), {"b64": data} a binary regular file (100644), {"link": target} a symlink
    (120000), {"gitlink": commit} a submodule (160000). NOTE_path2_thirteenth_pass (E): round 12 found the content-only
    model blind to git's `T` (typechange), which two licences had turned into false verdicts."""
    if value is None:
        return None
    if isinstance(value, str):
        return ("file", "100644", value)
    if "x" in value:
        return ("file", "100755", value["x"])
    if "b64" in value:
        return ("file", "100644", "b64:" + value["b64"])
    if "link" in value:
        return ("symlink", "120000", value["link"])
    return ("gitlink", "160000", value["gitlink"])


def _truth_status(model) -> dict:
    """{path: A | M | D | T}, as git's --name-status letters them: A absent in the base, D absent in the head, T the
    path's type changed (a regular file, a symlink, a submodule), M its bytes or its executable bit changed."""
    base, head = model["base"], model["head"]
    out = {}
    for p in sorted(set(base) | set(head)):
        b, h = _entry(base.get(p)), _entry(head.get(p))
        if b == h:
            continue
        out[p] = "A" if b is None else "D" if h is None else "T" if b[0] != h[0] else "M"
    return out


def _judge(values: set):
    return "T" if values == {True} else "F" if values == {False} else "?"


def _written(path: str) -> str:
    """A path as a claim writes it: backslashes as slashes, a leading run of `./` and `/` dropped, case kept."""
    path = path.replace("\\", "/")
    while path.startswith(("./", "/")):
        path = path[2:] if path.startswith("./") else path[1:]
    return path


def _truth(kind: str, detail: dict, status: dict):
    """T, F or ? (the readings a writer may mean disagree) for one claim, or None for a kind this model does not judge."""
    if kind == "files_changed_count":
        return _judge({len(status) == int(detail["n"])})
    if kind in ("file_created", "file_deleted", "file_touched"):
        claimed = _written(detail["path"])
        want = {"file_created": "A", "file_deleted": "D"}.get(kind)
        readings = [[p for p in status if p == claimed], [p for p in status if p == claimed or p.endswith("/" + claimed)]]
        if "/" not in claimed:
            readings.append([p for p in status if p.rsplit("/", 1)[-1] == claimed])
        return _judge({bool(hits) and (want is None or any(status[p] == want for p in hits)) for hits in readings})
    return None


def _worse(final: list, main: list, status: dict) -> list:
    """Claims (kind, text, occurrence) whose final verdict is false by truth where main's was not."""
    def wrong(verdict, t):
        return (verdict == "VERIFIED" and t == "F") or (verdict == "CONTRADICTED" and t == "T")
    theirs = dict(zip(_claim_keys_of(main), main))
    out = []
    for key, (kind, verdict, detail) in zip(_claim_keys_of(final), final):
        t = _truth(kind, detail, status)
        m = theirs.get(key)
        if t is not None and wrong(verdict, t) and not (m is not None and wrong(m[1], t)):
            out.append((key, verdict, None if m is None else m[1], t))
    return out


def _claim_keys_of(claims: list) -> list:
    seen: dict = {}
    out = []
    for kind, _verdict, detail in claims:
        text = detail.get("_text", "")
        out.append((kind, text, seen.get((kind, text), 0)))
        seen[(kind, text)] = seen.get((kind, text), 0) + 1
    return out


def _as_rows(g) -> list:
    return [(c.kind, c.verdict, dict(c.detail or {}, _text=c.text)) for c in g.claims]


def test_the_truth_model_reads_the_reproductions_as_round_11_recorded():
    """The model agrees with the reviewer's own recorded truth on every claim it recorded (their model is theirs; this one
    is written out here), so a test below is judged by the same truth round 11 measured with."""
    checked = 0
    for case in R11:
        status = _truth_status(case["model"])
        for key, recorded in case["reviewer_truth"].items():
            kind, text, _occ = key.split("\x1f")
            claim = next(c for c in ref.gate_diff_text(case["summary"], case["diff"]).claims if c.text == text)
            assert _truth(kind, claim.detail, status) == recorded, (case["id"], key)
            checked += 1
    assert checked == 16


@pytest.mark.parametrize("case", TRUTH_CASES, ids=[c["id"] for c in TRUTH_CASES])
def test_the_reproductions_read_no_worse_than_main_against_truth_on_the_raw_door(case):
    status = _truth_status(case["model"])
    final = _as_rows(dg.gate_diff_text(case["summary"], case["diff"]))
    main = _as_rows(ref.gate_diff_text(case["summary"], case["diff"]))
    assert not _worse(final, main, status), (case["id"], _worse(final, main, status))


def test_the_truth_judged_check_refuses_the_eleventh_pass_instrument():
    """The check above is not vacuous: the eleventh pass's instrument (15878ab5, before the licences were tightened) reads
    12 of the 17 reproductions worse than main by the same truth -- the 12 raw-door cells round 11 measured."""
    import types
    r = subprocess.run(["git", "-C", str(ROOT), "show", "15878ab5:styxx/diffgate.py"], capture_output=True)
    if r.returncode:
        pytest.skip("the eleventh pass's commit is not in this clone")
    mod = types.ModuleType("styxx._diffgate_eleventh_pass")
    mod.__package__, mod.__file__ = "styxx", "<15878ab5:styxx/diffgate.py>"
    sys.modules[mod.__name__] = mod
    exec(compile(r.stdout.decode("utf-8"), mod.__file__, "exec"), mod.__dict__)  # noqa: S102
    worse = [case["id"] for case in R11
             if _worse(_as_rows(mod.gate_diff_text(case["summary"], case["diff"])),
                       _as_rows(ref.gate_diff_text(case["summary"], case["diff"])), _truth_status(case["model"]))]
    assert len(worse) == 12 and not any(x.endswith("control") or x.startswith("C121-git") for x in worse), worse


def test_the_truth_judged_check_refuses_the_licences_before_the_tier_kept_case():
    """And this pass's own records: the licences as committed at e4586637, before #121's path licence asked for the case
    kept by the tier, read four of the six worse than main on the raw door (the other two only at the git door)."""
    import types
    r = subprocess.run(["git", "-C", str(ROOT), "show", "e4586637:styxx/diffgate.py"], capture_output=True)
    if r.returncode:
        pytest.skip("commit e4586637 is not in this clone")
    mod = types.ModuleType("styxx._diffgate_e4586637")
    mod.__package__, mod.__file__ = "styxx", "<e4586637:styxx/diffgate.py>"
    sys.modules[mod.__name__] = mod
    exec(compile(r.stdout.decode("utf-8"), mod.__file__, "exec"), mod.__dict__)  # noqa: S102
    worse = [case["id"] for case in R12
             if _worse(_as_rows(mod.gate_diff_text(case["summary"], case["diff"])),
                       _as_rows(ref.gate_diff_text(case["summary"], case["diff"])), _truth_status(case["model"]))]
    assert len(worse) == 4, worse


def _instrument_at(sha: str, tag: str):
    import types
    r = subprocess.run(["git", "-C", str(ROOT), "show", f"{sha}:styxx/diffgate.py"], capture_output=True)
    if r.returncode:
        pytest.skip(f"commit {sha} is not in this clone")
    mod = types.ModuleType(f"styxx._diffgate_{tag}")
    mod.__package__, mod.__file__ = "styxx", f"<{sha}:styxx/diffgate.py>"
    sys.modules[mod.__name__] = mod
    exec(compile(r.stdout.decode("utf-8"), mod.__file__, "exec"), mod.__dict__)  # noqa: S102
    return mod


def _worse_at(mod, cases) -> list:
    return [case["id"] for case in cases
            if _worse(_as_rows(mod.gate_diff_text(case["summary"], case["diff"])),
                      _as_rows(ref.gate_diff_text(case["summary"], case["diff"])), _truth_status(case["model"]))]


def test_the_truth_model_reads_gits_letters_for_every_round_12_model():
    """The model's statuses are git's own --name-status for the same trees, T included (recorded when the fixture was
    built from repositories with those modes); the two controls, a mode change and a symlink retarget, are M."""
    for case in R13:
        listed = {}
        for line in case["name_status"].splitlines():
            parts = line.split("\t")
            path = parts[-1]
            if path.startswith('"'):
                path = path[1:-1].encode("latin-1", "backslashreplace").decode("unicode_escape").encode(
                    "latin-1").decode("utf-8")
            listed[path] = parts[0][:1]
        assert listed == _truth_status(case["model"]), case["id"]
    assert {"T", "M", "A", "D"} <= {st for case in R13 for st in _truth_status(case["model"]).values()}


def test_the_truth_judged_check_refuses_the_twelfth_pass_instrument():
    """NOTE_path2_thirteenth_pass (E), calibration: the twelfth pass's instrument (0f559a87) reads round 12's
    reproductions worse than main by this truth on the raw door -- the typechange through #97 and #121, and the name
    ending in whitespace in difflib's rendering -- and this pass's head reads none of them worse (the test above)."""
    mod = _instrument_at("0f559a87", "twelfth_pass")
    worse = _worse_at(mod, R13)
    assert len(worse) == R13_WORSE_AT_0F559A87, worse
    assert not any(x.startswith("M1") or x.startswith("M2") for x in worse), worse


def test_the_reproductions_read_no_worse_than_main_against_truth_in_the_port():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH")
    script = ("const B = require(process.argv[1]), M = require(process.argv[2]);"
              "const P = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
              "const rows = g => g.claims.map(c => [c.kind, c.verdict, Object.assign({}, c.detail || {}, {_text: c.text})]);"
              "process.stdout.write(JSON.stringify(P.map(([s, d]) => [rows(B.gateDiffText(s, d)), rows(M.gateDiffText(s, d))])));")
    r = subprocess.run([node, "-e", script, str(PORT), str(REF_JS)],
                       input=json.dumps([[c["summary"], c["diff"]] for c in TRUTH_CASES]),
                       capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    for case, (final, main) in zip(TRUTH_CASES, json.loads(r.stdout)):
        status = _truth_status(case["model"])
        final, main = [tuple(x) for x in final], [tuple(x) for x in main]
        assert not _worse(final, main, status), (case["id"], _worse(final, main, status))


def _fast_import_path(path: str) -> bytes:
    """A path as `git fast-import` reads it, C-quoted so that a trailing space, a quote or a backslash survive."""
    out = bytearray(b'"')
    for byte in path.encode("utf-8"):
        out += (b"\\" + bytes([byte]) if byte in (0x22, 0x5C) else
                b"\\%03o" % byte if byte < 0x20 or byte == 0x7F else bytes([byte]))
    return bytes(out + b'"')


def _repository(tmp_path: Path, cases: list) -> Path:
    """One bare repository holding each case's base and head trees as branches `bN` and `hN`, written by fast-import
    (no working tree, so paths that differ only in case, end in a space or open with `a/` all survive).
    NOTE_path2_thirteenth_pass (E): each entry is written with its mode -- 100644, 100755, a symlink (120000) and a
    submodule (160000, a commit id) -- so git's --name-status reads its `T`."""
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not on PATH")
    repo = tmp_path / "repo.git"
    subprocess.run([git, "init", "-q", "--bare", str(repo)], check=True, capture_output=True)
    subprocess.run([git, "-C", str(repo), "config", "core.ignorecase", "false"], check=True)
    # (git for Windows refuses a path ending in a space unless NTFS protection is off; no file is written to disk here)
    subprocess.run([git, "-C", str(repo), "config", "core.protectNTFS", "false"], check=True)
    stream = bytearray()
    for i, case in enumerate(cases):
        for side, ref_name in (("base", f"b{i}"), ("head", f"h{i}")):
            stream += b"commit refs/heads/" + ref_name.encode() + b"\ncommitter t <t@t> 1700000000 +0000\ndata 0\n"
            if side == "head":
                stream += b"from refs/heads/b%d\n" % i
            stream += b"deleteall\n"
            for path, value in sorted(case["model"][side].items()):
                entry = _entry(value)
                if entry is None:
                    continue
                kind, mode, content = entry
                if kind == "gitlink":
                    stream += b"M 160000 " + content.encode("ascii") + b" " + _fast_import_path(path) + b"\n"
                    continue
                data = (base64.b64decode(content[4:]) if isinstance(value, dict) and "b64" in value
                        else content.encode("utf-8"))
                stream += (b"M " + mode.encode("ascii") + b" inline " + _fast_import_path(path) + b"\ndata %d\n" % len(data)
                           + data + b"\n")
            stream += b"\n"
    subprocess.run([git, "-C", str(repo), "fast-import", "--quiet"], input=bytes(stream), check=True, capture_output=True)
    return repo


def test_the_reproductions_read_no_worse_than_main_against_truth_at_the_git_door(tmp_path):
    """Every reproduction, rebuilt as a repository from its model: git's own --name-status is the model's file list (so
    truth is git's), and the git door reads no claim worse than main's git door."""
    repo = _repository(tmp_path, TRUTH_CASES)
    for i, case in enumerate(TRUTH_CASES):
        listed = dg._git(repo, "diff", "--name-status", f"b{i}..h{i}")
        status = _truth_status(case["model"])
        assert sorted(line.split("\t")[0][:1] + line.split("\t")[-1] for line in listed.splitlines()) == sorted(
            st + path for path, st in status.items()) or '"' in listed, case["id"]
        final = _as_rows(dg.gate_diff(case["summary"], repo, f"b{i}", f"h{i}"))
        main = _as_rows(ref.gate_diff(case["summary"], repo, f"b{i}", f"h{i}"))
        assert not _worse(final, main, status), (case["id"], _worse(final, main, status))


# The property, against the scorer's reverts: the instrument's final claims are the scorer's own guard's, claim for claim.
@pytest.fixture(scope="module")
def scorer():
    import importlib.util
    base = "98a5c368ba9ffa242c6862e021df7f8bad2ed8e6"
    if subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", base + "^{commit}"], capture_output=True).returncode:
        pytest.skip("the scorer's baseline commit is not in this clone (a shallow checkout)")
    saved = sys.modules.get("styxx.claimdetect", None)
    had = "styxx.claimdetect" in sys.modules
    spec = importlib.util.spec_from_file_location(
        "path2_gates_for_the_guard", ROOT / "papers" / "closed-model-frontier" / "path2_gates.py")
    pg = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(pg)
    except SystemExit as e:
        pytest.skip(f"path2_gates refused to load: {e}")
    finally:
        if had:
            sys.modules["styxx.claimdetect"] = saved
        else:
            sys.modules.pop("styxx.claimdetect", None)
    return pg


def _scorer_guard_violations(pg, summary: str, diff: str) -> list:
    before = pg.new._evaluate_text(summary, diff, pg.new._ALL_ON)
    final = pg.new.gate_diff_text(summary, diff, run=None, strict=False)
    try:
        main = pg.BASE.gate_diff_text(summary, diff, run=None, strict=False).claims
    except Exception:
        main = None
    facts: dict = {}
    own = pg.own_read(diff, facts)
    cf = pg.Counterfactual(summary, diff)
    return pg.guard_violations(summary, main, before, final, lambda r: cf.reading({r}),
                               lambda r: pg.new._evaluate_text(summary, diff, pg.new._Repairs({r})).claims,
                               own[0], own[2], None, facts)


def test_the_guarantee_holds_against_the_scorers_own_guard_on_the_raw_door(scorer):
    pg = scorer
    records = list(_pinned()) + list(guard_cases(20260930, 500)) + [(c["id"], c["summary"], c["diff"]) for c in TRUTH_CASES]
    for pid, summary, diff in records:
        bad = _scorer_guard_violations(pg, summary, diff)
        assert not bad, (pid, bad)
    assert len(records) > 800


def test_the_guarantee_holds_against_the_scorers_own_guard_at_the_git_door(scorer):
    """The git-buildable part of the randomised set and of the pinned pairs, rebuilt as two-commit repositories: the git
    door's final claims are the scorer's own guard's over git's --name-status and git's own diff text."""
    import itertools
    pg = scorer
    scored = 0
    records = itertools.chain(guard_cases(20260930, 500), _pinned())
    for pid, summary, diff in records:
        try:
            files = pg.rebuild(diff)
        except (UnicodeError, ValueError):
            files = None
        if files is None:
            continue
        with pg.GitDoor(*files) as door:
            if not door.diff.strip():
                continue
            gb, gn, gu = door.gate(pg.BASE, summary), door.gate(pg.new, summary), door.reading(pg.new, summary)
            cf = pg.Counterfactual(summary, diff, door)
            facts = pg.own_git_licence(door.text, door.status_paths, door.name_status)
            bad = pg.guard_violations(summary, gb.claims, gu, gn, lambda r: cf.reading({r}),
                                      lambda r: door.reading(pg.new, summary, pg.new._Repairs({r})).claims,
                                      door.status, pg.own_read(door.text)[2], None, facts)
            assert not bad, (pid, bad)
            scored += 1
        if scored >= 60:
            break
    assert scored >= 60, scored


def test_the_git_door_licence_facts_read_gits_typechange_and_the_scorers_own(tmp_path, scorer):
    """NOTE_path2_thirteenth_pass (A.1): at the git door an entry whose --name-status letter is T, one a mode change names,
    or a key the diff text registers twice, licenses nothing (inert for a verdict: T and a mode-changed M never equal A or
    D). What `_evaluate_git` hands its guard is held to the scorer's own reading of the same bytes, T included."""
    pg = scorer
    cases = [c for c in R13 if c["id"] in ("T1", "T2", "T8", "M1-mode-change", "T4")]
    repo = _repository(tmp_path, cases)
    multi = {}
    for i, case in enumerate(cases):
        ns = dg._git(repo, "diff", "--name-status", f"b{i}..h{i}")
        text = dg._git(repo, "diff", f"b{i}..h{i}")
        out: dict = {}
        dg._evaluate_git(case["summary"], ns, text, dg._ALL_ON, repo=repo, base=f"b{i}", head=f"h{i}", out=out)
        paths = [x.split("\t")[-1] for x in ns.splitlines() if len(x.split("\t")) >= 2]
        assert out["licence"] == pg.own_git_licence(text, paths, ns), case["id"]
        typechanged = {dg._norm(p) for p, st in _truth_status(case["model"]).items() if st == "T"}
        assert typechanged <= out["licence"]["multi"], case["id"]
        multi[case["id"]] = out["licence"]["multi"]
    assert all(multi[c] for c in ("T1", "T2", "T8", "T4"))
    assert multi["M1-mode-change"] == {"bin/run.sh"}          # git's `old mode`/`new mode`, an M in --name-status


def test_the_licence_facts_read_alike_in_both_ports_on_round_12s_renderings():
    """NOTE_path2_thirteenth_pass (A.1, A.2): `multi`, `moded` and the forms as written, the same in the Python and the port
    on every round-12 rendering (typechange sections, names ending in whitespace, a TAB that ends a name)."""
    py = {}
    for case in R13:
        facts: dict = {}
        dg._diff_notes(case["diff"], facts)
        py[case["id"]] = [sorted(facts["multi"]), sorted(facts["moded"]), sorted(facts["forms"].items())]
    assert any(v[0] for v in py.values())
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH")
    script = ("const B = require(process.argv[1]); const D = JSON.parse(require('fs').readFileSync(0, 'utf8')); const out = {};"
              "for (const [k, d] of Object.entries(D)) { const o = {}; B._evaluate('1 file changed.', d, { out: o });"
              " out[k] = [[...o.licence.multi].sort(), [...o.licence.moded].sort(), [...o.licence.forms].sort()]; }"
              " process.stdout.write(JSON.stringify(out));")
    r = subprocess.run([node, "-e", script, str(PORT)], input=json.dumps({c["id"]: c["diff"] for c in R13}),
                       capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    js = json.loads(r.stdout)
    for cid, (multi, moded, forms) in py.items():
        assert js[cid][0] == multi and js[cid][1] == moded, cid
        assert [[k, v] for k, v in forms] == js[cid][2], cid


# ---- 5. the port's reference ---------------------------------------------------------------------------------------------

def test_the_port_without_its_reference_throws_rather_than_reading_main_as_raising():
    """Round-11 guard lens: a page that loads diffgate.js without diffgate_ref.js (or the bookmarklet's bundle) read every
    decided claim as UNCHECKABLE with "main's reading raises on this diff". The port now finds its reference before the
    guard runs, so the missing reference throws; only what main's own gateDiffText throws reads as main raising."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH")
    script = ("const vm = require('vm'); const src = require('fs').readFileSync(process.argv[1], 'utf8');"
              "const ctx = { module: { exports: {} }, console }; vm.createContext(ctx); vm.runInContext(src, ctx);"
              "let out; try { ctx.module.exports.gateDiffText('Modified src/a.py. 1 file changed.',"
              " '--- a/src/a.py\\n+++ b/src/a.py\\n@@ -1 +1 @@\\n-a\\n+b\\n'); out = 'no error'; }"
              " catch (e) { out = 'threw: ' + e.message; } process.stdout.write(out);")
    r = subprocess.run([node, "-e", script, str(PORT)], capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.startswith("threw: diffgate.js needs its reference"), r.stdout
