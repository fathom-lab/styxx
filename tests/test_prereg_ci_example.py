# -*- coding: utf-8 -*-
"""prereg-ci B9: DESIGN 4.B, the five-minute repository, run as its README runs it.

``examples/prereg-ci-demo/`` ships a plan (``PREREG_digits.md``), the run (``eval.py``: scikit-learn's
bundled digits set, a logistic regression, held-out accuracy and expected calibration error), the README
sentence the plan's HOLD row withdraws (``README_demo.md``), the config naming that README as the
shipped surface (``prereg-ci.toml``), a RESULT with a result fence, an amendment with its own gates
block, and pinned versions. These tests copy the files into a repository made with ``git init`` in
tmp_path and walk ``examples/prereg-ci-demo/README.md``'s commands: the honest PASS, the four
refusals a byte comparison of the plan does not give (B.1 a RESULT that types another rung, B.2 a
receipt scored under a bar moved outside the sealed file, B.3 a committed receipt rewritten, B.4 a
sentence the landed row withdraws, here under the amendment), the overlap (a bar moved in the sealed
file) and the amendment route that stays on the record (PASS with AMENDED_AFTER_DATA beside it).

``eval.py`` runs as a child process with this checkout on PYTHONPATH, as a stranger runs it with
styxx installed; ``check`` runs in this process. The FAIL codes of B.1 to B.3 do not depend on the
numbers, since each branch is built to trip one check; which row the run lands on does, and the
numbers are pinned only under the versions ``requirements.txt`` names. No remote, no network: the
digits set ships inside scikit-learn.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx.prereg_ci import cli  # noqa: E402
from styxx.prereg_ci import core  # noqa: E402
from tests.test_prereg_ci_git import Git  # noqa: E402
from tests.test_prereg_ci_package import HOUSE_RX, PRIORITY_RX  # noqa: E402

DEMO = ROOT / "examples" / "prereg-ci-demo"
FILES = ("AMENDMENT_digits.md", "PREREG_digits.md", "README.md", "README_demo.md", "RESULT_digits.md",
         "eval.py", "prereg-ci.toml", "requirements.txt")
VERDICTS = ("SHIP", "HOLD__calibration", "INVALID__accuracy_floor")
SENTENCE = "The model is calibrated well enough to ship."
# what the example author's run read under the pinned versions (eval.py; README.md "where the bars come from")
SEED0 = {"accuracy": 0.963293, "ece": 0.022243}
BAR_RUN = {"accuracy": 0.964405, "ece": 0.021373}
SPREAD_RANGE = {"accuracy": (0.953281, 0.979978), "ece": (0.01533, 0.027582)}

sklearn = pytest.importorskip("sklearn")
np = pytest.importorskip("numpy")
scipy = pytest.importorskip("scipy")


def _bytes(name: str) -> bytes:
    """The example file as git stores it: a checkout on this box may write CRLF; the blob is LF."""
    return (DEMO / name).read_bytes().replace(b"\r\n", b"\n")


def _pins() -> dict:
    out = {}
    for line in _bytes("requirements.txt").decode("utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, _, version = line.partition("==")
            out[name.strip()] = version.strip()
    return out


PINNED = _pins() == {"numpy": np.__version__, "scikit-learn": sklearn.__version__, "scipy": scipy.__version__}
# the six-decimal numbers were read on Windows (README.md); another platform's BLAS may move a last digit, so the
# tests that print them exactly run where they were read, and Linux parity is owed to the pull request's CI
READ_HERE = PINNED and sys.platform == "win32"


def _eval(g: Git, *argv: str) -> str:
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-B", "eval.py", *argv], cwd=str(g.root), env=env, capture_output=True)
    out = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
    assert r.returncode == 0, out
    return out


def _check(g: Git, capsys, *argv: str):
    capsys.readouterr()
    cwd = os.getcwd()
    try:
        os.chdir(g.root)
        code = cli.main(list(argv))
    finally:
        os.chdir(cwd)
    return code, capsys.readouterr().out


def _codes(out: str, kind: str = "FAIL") -> list:
    return sorted(set(re.findall(rf"^\s*{kind} ([A-Z_]+)\s", out, re.M)))


@pytest.fixture(scope="module")
def flow(tmp_path_factory):
    """The README's repository: main holds the sealed plan and the result; each refusal is a branch off
    main; the amendment commits land on main before B.4 and the record route branch off it."""
    g = Git(tmp_path_factory.mktemp("example") / "prereg-demo")
    for name in ("PREREG_digits.md", "README_demo.md", "prereg-ci.toml"):
        g.write(name, _bytes(name))
    g.commit("seal the plan")
    for name in ("eval.py", "RESULT_digits.md"):
        g.write(name, _bytes(name))
    run = {"result": _eval(g)}
    g.commit("result")
    landed = json.loads((g.root / "receipt.json").read_text(encoding="utf-8"))["verdict"]

    g("switch", "-q", "-c", "type-the-verdict", "main")                         # B.1
    other = next(v for v in VERDICTS if v != landed)
    p = g.root / "RESULT_digits.md"
    p.write_bytes(p.read_bytes().replace(landed.encode(), other.encode()))
    g.commit("type the verdict")

    plan_edit = g.root.parent / "plan_edit.md"                                   # B.2 and B.3
    plan_edit.write_bytes(_bytes("PREREG_digits.md").replace(b'"value": 0.95', b'"value": 0.90'))
    g("switch", "-q", "-c", "working-copy-bar", "main")
    run["rerun"] = _eval(g, "--plan", str(plan_edit), "--out", "receipt_rerun.json")
    g.commit("re-run")
    g("switch", "-q", "-c", "rewrite-receipt", "main")
    run["rewrite"] = _eval(g, "--plan", str(plan_edit))
    g.commit("rewrite")

    g("switch", "-q", "-c", "lower-the-bar", "main")                             # the overlap
    g.write("PREREG_digits.md", _bytes("PREREG_digits.md").replace(b'"value": 0.95', b'"value": 0.90'))
    g.commit("lower the bar")

    g("switch", "-q", "main")                                                    # the amendment, on main
    g.write("AMENDMENT_digits.md", _bytes("AMENDMENT_digits.md"))
    g.commit("amendment")
    cwd = os.getcwd()
    try:
        os.chdir(g.root)
        seal_code = cli.main(["seal", "AMENDMENT_digits.md", "--amends", "PREREG_digits.md", "--seen", "results seen",
                              "--append"])
    finally:
        os.chdir(cwd)
    assert seal_code == 0
    g.commit("seal the amendment")

    g("switch", "-q", "-c", "keep-the-sentence", "main")                        # B.4
    run["amended"] = _eval(g, "--plan", "AMENDMENT_digits.md", "--out", "receipt_amended.json")
    g.commit("re-run under the amendment")

    g("switch", "-q", "-c", "withdraw-the-sentence", "main")                    # the record route
    _eval(g, "--plan", "AMENDMENT_digits.md", "--out", "receipt_amended.json")
    readme = g.root / "README_demo.md"
    text = readme.read_text(encoding="utf-8")
    assert text.count(SENTENCE) == 1
    readme.write_bytes(text.replace(SENTENCE, f'Withdrawn: "{SENTENCE}" The amended table\'s HOLD__calibration row '
                                              f"withdraws it (receipt_amended.json).").encode("utf-8"))
    g.commit("re-run under the amendment, and withdraw the sentence it withdraws")
    g("switch", "-q", "main")
    return g, landed, run


# ---- the files ---------------------------------------------------------------------------------------------

def test_the_example_ships_the_files_the_design_names_lf_and_no_bom():
    assert sorted(p.name for p in DEMO.iterdir() if p.is_file()) == sorted(FILES)
    for name in FILES:
        raw = (DEMO / name).read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), name
        r = subprocess.run(["git", "-C", str(ROOT), "show", f":examples/prereg-ci-demo/{name}"], capture_output=True)
        if r.returncode == 0:                                  # once committed: the blob is LF
            assert b"\r" not in r.stdout, name


def test_requirements_pin_exact_versions():
    pins = _pins()
    assert set(pins) == {"numpy", "scikit-learn", "scipy"}
    assert all(re.fullmatch(r"[0-9]+(\.[0-9]+)+", v) for v in pins.values()), pins


def test_the_plan_is_the_shape_4b_names_and_its_bar_is_one_sed_target():
    plan = _bytes("PREREG_digits.md").decode("utf-8")
    assert plan.count('"value": 0.95') == 1                  # B.2's and the overlap's sed edit one bar
    d = core._decision(plan.encode("utf-8"))
    spec = json.loads(d["block"])
    assert [r["verdict"] for r in spec["outcomes"]] == ["INVALID__accuracy_floor", "SHIP", "HOLD__calibration"]
    (lic,) = spec["outcomes"][2]["withdraws"]
    assert lic["quote"] == SENTENCE and re.search(lic["pattern"], SENTENCE, re.I)
    assert SENTENCE in _bytes("README_demo.md").decode("utf-8")
    assert "withdr" not in _bytes("README_demo.md").decode("utf-8").lower()   # the sentence stands until a row lands
    amend = json.loads(core._decision(_bytes("AMENDMENT_digits.md"))["block"])
    assert amend["gates"]["G_ece"]["value"] == 0.01 and amend["gates"]["G_acc"] == spec["gates"]["G_acc"]
    assert amend["outcomes"] == spec["outcomes"]
    assert cli._toml(_bytes("prereg-ci.toml").decode("utf-8")) == {"surfaces": ["README*.md"]}
    result = core._result_fence(_bytes("RESULT_digits.md").decode("utf-8"))
    assert (result["prereg"], result["receipt"], result["verdict"]) == ("PREREG_digits.md", "receipt.json", "SHIP")


def test_lint_reads_the_plan_and_the_amendment_total_and_each_licence_finds_the_readme(flow, capsys):
    g, _, _ = flow
    for name in ("PREREG_digits.md", "AMENDMENT_digits.md"):
        code, out = _check(g, capsys, "lint", name, "--rev", "main")
        assert code == 0, out
        assert "total over 4 assignments; 3 rows, each reached" in out
        assert "licence calibration: 1 surface (README_demo.md)" in out


def test_the_word_law_over_the_example():
    for name in FILES:
        text = _bytes(name).decode("utf-8")
        assert not HOUSE_RX.search(text) and not PRIORITY_RX.search(text), name


# ---- the run -----------------------------------------------------------------------------------------------

def test_the_receipt_names_the_plan_and_records_the_sealed_tables_reading(flow):
    g, landed, run = flow
    rec = json.loads(subprocess.run(["git", "show", "main:receipt.json"], cwd=str(g.root), capture_output=True,
                                    check=True).stdout)
    assert rec["prereg"] == "PREREG_digits.md" and rec["smoke"] is False and rec["seed"] == 0
    rs = core.rescore(_bytes("PREREG_digits.md"), rec)
    assert (rec["verdict"], rec["gates"], rec["gates_sha256"]) == (rs["verdict"], rs["gates"], rs["gates_sha256"])
    assert landed in VERDICTS
    assert "G_acc result.accuracy" in run["result"] and "G_ece result.ece" in run["result"]


@pytest.mark.skipif(not READ_HERE, reason="the numbers are pinned under requirements.txt's versions, as read on Windows")
def test_under_the_pinned_versions_the_run_lands_on_ship_with_both_margins_inside_the_spread(flow):
    g, landed, run = flow
    rec = json.loads((g.root / "receipt.json").read_text(encoding="utf-8"))
    assert landed == "SHIP" and {k: rec["result"][k] for k in SEED0} == SEED0
    assert run["result"].count("MARGIN_INSIDE_SPREAD") == 2
    assert "G_acc result.accuracy 0.963293 >= 0.95 fires, margin +0.013293" in run["result"]
    assert "G_ece result.ece 0.022243 <= 0.03 fires, margin +0.007757" in run["result"]
    assert "receipt_amended.json: HOLD__calibration under AMENDMENT_digits.md" in run["amended"]
    assert "G_ece result.ece 0.022243 <= 0.01 does not fire, margin -0.012243" in run["amended"]


@pytest.mark.skipif(not READ_HERE, reason="the numbers are pinned under requirements.txt's versions, as read on Windows")
def test_the_spread_and_the_bar_run_are_the_numbers_the_plan_and_readme_print(flow):
    g, _, _ = flow
    out = _eval(g, "--spread")
    assert f"seed   7: accuracy {BAR_RUN['accuracy']:.6f}  ece {BAR_RUN['ece']:.6f}" in out
    assert "its ECE rounded up to the next 0.01 is 0.03" in out
    for key, (lo, hi) in SPREAD_RANGE.items():
        assert f"{key}: {lo:.6f} to {hi:.6f} over seeds 101-110" in out
    assert out.count("the same to four decimals") == 2
    plan = _bytes("PREREG_digits.md").decode("utf-8")
    readme = _bytes("README.md").decode("utf-8")
    for text in (plan, readme):
        for number in ("0.964405", "0.021373", "0.953281", "0.979978", "0.015330", "0.027582", "0.0267", "0.0123"):
            assert number in text, number
    for number in ("0.963293", "0.022243", "0.013293", "0.007757", "-0.012243"):
        assert number in readme, number


# ---- check: the README's branches -------------------------------------------------------------------------

@pytest.mark.skipif(not PINNED, reason="RESULT_digits.md states the row the run lands on under the pinned versions")
def test_the_honest_result_passes_with_its_ladder_flip_table_and_author_order(flow, capsys):
    g, landed, _ = flow
    # main reads: "seal the amendment", ~1 "amendment", ~2 "result", ~3 "seal the plan"; README: --base HEAD~1
    code, out = _check(g, capsys, "check", "--base", "main~3", "--head", "main~2")
    assert code == 0, out
    assert "PASS  RESULT_digits.md" in out and f"the RESULT RESULT_digits.md prints {landed}" in out
    assert "time AUTHOR_ORDER (T2 not read, T3 none)" in out and "flips that move the verdict" in out
    assert "not examined: the run itself" in out and out.rstrip().endswith("PASS  exit 0")


def test_b1_a_result_that_types_another_rung_is_verdict_differs(flow, capsys):
    g, _, _ = flow
    code, out = _check(g, capsys, "check", "--base", "main~2", "--head", "type-the-verdict")
    assert code == 1 and _codes(out) == ["VERDICT_DIFFERS"], out


def test_b2_a_rerun_under_a_bar_moved_outside_the_sealed_file_is_digest_unsealed(flow, capsys):
    g, _, _ = flow
    code, out = _check(g, capsys, "check", "--base", "main~2", "--head", "working-copy-bar")
    assert code == 1 and "DECISION_DIGEST_UNSEALED" in _codes(out), out
    assert "C0 sealed decisions: 0 moved" in out                       # the plan's bytes did not move
    if PINNED:                                                         # 0.90 flips no gate at accuracy 0.963293
        assert _codes(out) == ["DECISION_DIGEST_UNSEALED"], out


def test_b3_a_committed_receipt_rewritten_in_place_is_refused(flow, capsys):
    g, _, _ = flow
    code, out = _check(g, capsys, "check", "--base", "main~2", "--head", "rewrite-receipt")
    assert code == 1 and {"RECEIPT_REWRITTEN", "DECISION_DIGEST_UNSEALED"} <= set(_codes(out)), out


def test_the_overlap_a_bar_moved_in_the_sealed_file_is_decision_edited_after_seal(flow, capsys):
    g, _, _ = flow
    code, out = _check(g, capsys, "check", "--base", "main~2", "--head", "lower-the-bar")
    assert code == 1 and _codes(out) == ["DECISION_EDITED_AFTER_SEAL"], out
    assert "the refusal path protection gives too" in out


def test_the_amendment_is_sealed_with_what_had_been_seen(flow):
    g, _, _ = flow
    lines = [json.loads(x) for x in (g.root / "prereg-ci.seals.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(x["kind"], x["path"], x["amends"], x["decision"], x["seen"]) for x in lines] == [
        ("prereg", "PREREG_digits.md", None, "own", None),
        ("amendment", "AMENDMENT_digits.md", lines[0]["id"], "own", "results seen")]


@pytest.mark.skipif(not PINNED, reason="the amended run lands on HOLD__calibration under the pinned versions")
def test_b4_under_the_amendment_the_readme_sentence_the_landed_row_withdraws_is_refused(flow, capsys):
    g, _, _ = flow
    code, out = _check(g, capsys, "check", "--base", "main", "--head", "keep-the-sentence")
    assert code == 1 and _codes(out) == ["LICENCE_WITHDRAWN"], out
    assert "FAIL LICENCE_WITHDRAWN README_demo.md:6:14" in out
    assert "NOTE AMENDED_AFTER_DATA  receipt.json@" in out and "amendment seen: results seen" in out


@pytest.mark.skipif(not PINNED, reason="the amended run lands on HOLD__calibration under the pinned versions")
def test_the_record_route_passes_under_the_amended_table_with_amended_after_data_beside_it(flow, capsys):
    g, _, _ = flow
    code, out = _check(g, capsys, "check", "--base", "main", "--head", "withdraw-the-sentence")
    assert code == 0, out
    assert "PASS  receipt_amended.json" in out and "landed row 3 of 3: HOLD__calibration" in out
    assert "seen: results seen): in effect" in out
    assert "NOTE AMENDED_AFTER_DATA  receipt.json@" in out and "read SHIP under the sealed table" in out


def test_the_digests_the_readme_quotes_are_the_files(flow):
    """The README quotes the plan's seal bytes and the three gates digests the branches print; a change to
    either file moves them, and this test names the stale quote."""
    readme = _bytes("README.md").decode("utf-8")
    plan = _bytes("PREREG_digits.md")
    moved = plan.replace(b'"value": 0.95', b'"value": 0.90')
    quoted = {"seal bytes": hashlib.sha256(plan).hexdigest()[:8],
              "sealed gates": core._decision(plan)["gates_sha256"][:12],
              "moved gates": core._decision(moved)["gates_sha256"][:8],
              "amendment gates": core._decision(_bytes("AMENDMENT_digits.md"))["gates_sha256"][:8]}
    for what, prefix in quoted.items():
        assert prefix in readme, (what, prefix)
