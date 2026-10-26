# -*- coding: utf-8 -*-
"""prereg-ci B7 done-when: DESIGN 4.A on styxx's own history, through the commands.

4.A replays the lab's bytes. The PASS on the #224 merge prints the seal, the order commit, the time
rung, the landed row and the flips that move the verdict (R2, R9); A.1 types a rung the receipt did
not land on (VERDICT_DIFFERS); A.2 writes a consistent re-run under a bar moved only in the working
copy to a new receipt path (DECISION_DIGEST_UNSEALED and VERDICT_DIFFERS, R2's moved digest); A.3
reads EXTERNAL-1's receipt under the RECONSTRUCTED table at its RESULT's commit (RV4) and its licence
over the surfaces there (RV3); A.4 moves the bar in the sealed PREREG (DECISION_EDITED_AFTER_SEAL,
the refusal path protection gives too).

The branches A.1, A.2 and A.4 are built in ``git clone --shared --no-checkout`` of this repository's
own path in tmp_path: filesystem only, no remote, the styxx objects borrowed through alternates, and
every new object and ref written into the clone, never into the styxx store. They are built with
plumbing (hash-object, update-index on a private index, write-tree, commit-tree), which writes the
same blobs the design's ``sed`` and ``git commit`` would on a checkout. The receipts' numbers are R2,
R9, RV3 and RV4 (the design's census and revision probes, copied under ``conformance/prereg_ci/``).
The tests skip, saying why, where the commits are not in the object store or the clone is shallow.
Linux parity is left to the pull request's CI.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx import protocol  # noqa: E402
from styxx.prereg_ci import cli  # noqa: E402
from styxx.research_capsule import score_frozen  # noqa: E402

MERGE_224 = "20187746"                      # the #224 merge (R3b)
EXTERNAL1_RESULT = "414db45a"               # the commit that added EXTERNAL-1's RESULT (RV3, RV4)
P = "papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md"
R = "papers/frequency-resonance/pmnist_robustness_result.json"
RESULT = "papers/frequency-resonance/RESULT_pmnist_robustness_2026_10_23.md"
RERUN = "papers/frequency-resonance/pmnist_robustness_result_rerun.json"
TABLE = ROOT / "conformance" / "prereg_ci" / "reconstructed" / "external1_reconstructed.md"
LANDED = "BUDGET_CONFOUND__gap_does_not_survive_3x_steps"
ROBUST = "ROBUST__ROTATION_LOAD_BEARING__beyond_diversity"
SEALED_SHA = "578924f8e776573cc82db8fbbd8bb9cf84106b81171a5ba6d5783e422c96e854"     # R2
MOVED_SHA = "e513a1da1f8c068ea6ba60138392a031bd13edd2b2539a0f4f0fbe4ab9c61c82"      # R2
EXTERNAL1_SHA = "bd6f4256dcd01ccbfdcfa9864c312d1c008e0a5fd8aebed29486cb3321311841"  # RV4
BAR = '"metric": "result.gap_long", "op": ">=", "value": 0.15'                       # DESIGN 4.A's sed
# EXTERNAL-1's sentence, which its PREREG bound the lab to withdraw, and the RV3 pattern
QUOTE = "an agent's PR summary cannot lie about its diff"
FIVE = "cannot lie about its diff"
# R9: each gate's metric, op and bar, its fired value, and the row its single flip lands on
R9 = [("G_P_plumbing", "==", 1.0, 1.0, True, 0, "INVALID__plumbing"),
      ("G_D_stable", "<=", 0.1, 0.0, True, 1, "INVALID__training_unstable"),
      ("G_F_floor", ">=", 0.85, 0.9274, True, 2, "ABSTAIN__free_arm_below_floor"),
      ("G_C_anchor", ">=", 0.15, 0.3182, True, 3, "HEADLINE_NOT_REPRODUCED__anchor_permutation_three_new_seeds"),
      ("G_G_exists", ">", 0.0, 0.2066, True, 4, "NOT_ROBUST__gap_confined_to_anchor_permutation"),
      ("G_G_mean", ">=", 0.15, 0.2676, True, 12, "ATTENUATED__gap_below_half_receipt_on_permutation_average"),
      ("G_G_ci", ">=", 0.05, 0.2066, True, 10,
       "ATTENUATED_VARIABLE__ROTATION_LOAD_BEARING__interval_reaches_below_widens_bar"),
      ("G_B_long", ">=", 0.15, 0.1097, False, 6, ROBUST),
      ("G_R_diversity", "<=", 0.03, 0.275, False, 8, LANDED),
      ("G_R_rotation", "<=", 0.03, -0.0074, True, 8, LANDED)]
# RV3 at 414db45a, with the line and column B5's review read for each match
WITHDRAWN = ["README.md:55:33", "action.yml:3:27", "styxx/__init__.py:429:56", "styxx/diffgate.py:2:63",
             "styxx/diffgate.py:430:66"]


def _git(cwd, *args, env=None, data=None) -> str:
    r = subprocess.run(["git", "--no-replace-objects", *args], cwd=str(cwd), capture_output=True, input=data,
                       env=env)
    if r.returncode != 0:
        raise AssertionError(f"git {' '.join(args)}: {r.stderr.decode('utf-8', 'replace')}")
    return r.stdout.decode("utf-8", "replace").strip()


def _have(*revs) -> bool:
    for rev in revs:
        r = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", rev + "^{commit}"], capture_output=True)
        if r.returncode != 0:
            return False
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--is-shallow-repository"], capture_output=True)
    return r.stdout.strip() == b"false"


pytestmark = pytest.mark.skipif(not _have(MERGE_224, EXTERNAL1_RESULT),
                                reason=f"{MERGE_224} or {EXTERNAL1_RESULT} is not in the object store, or the clone "
                                       f"is shallow; 4.A replays them")


class Clone:
    """``git clone --shared --no-checkout`` of this repository, and commits made with plumbing."""

    def __init__(self, where: Path):
        self.root = where
        _git(where.parent, "clone", "-q", "--shared", "--no-checkout", str(ROOT), str(where))
        self.index = where.parent / "plumbing.index"

    def show(self, rev: str, path: str) -> bytes:
        r = subprocess.run(["git", "--no-replace-objects", "show", f"{rev}:{path}"], cwd=str(self.root),
                           capture_output=True, check=True)
        return r.stdout

    def commit(self, parent: str, changes: dict, msg: str, branch: str) -> str:
        env = dict(os.environ, GIT_INDEX_FILE=str(self.index), GIT_AUTHOR_NAME="demo", GIT_AUTHOR_EMAIL="d@d",
                   GIT_COMMITTER_NAME="demo", GIT_COMMITTER_EMAIL="d@d",
                   GIT_AUTHOR_DATE="1761400000 +0000", GIT_COMMITTER_DATE="1761400000 +0000")
        _git(self.root, "read-tree", parent, env=env)
        for path, data in changes.items():
            blob = _git(self.root, "hash-object", "-w", "--no-filters", "--stdin", env=env, data=data)
            _git(self.root, "update-index", "--add", "--cacheinfo", f"100644,{blob},{path}", env=env)
        tree = _git(self.root, "write-tree", env=env)
        commit = _git(self.root, "commit-tree", tree, "-p", _git(self.root, "rev-parse", parent), "-m", msg, env=env)
        _git(self.root, "update-ref", f"refs/heads/{branch}", commit)
        return commit


@pytest.fixture(scope="module")
def clone(tmp_path_factory):
    c = Clone(tmp_path_factory.mktemp("demo224") / "styxx")
    result = c.show(MERGE_224, RESULT)
    assert result.count(LANDED.encode()) >= 1
    c.typed = c.commit(MERGE_224, {RESULT: result.replace(LANDED.encode(), ROBUST.encode())}, "type the rung",
                       "demo/typed-rung")
    plan = c.show(MERGE_224, P).decode("utf-8")
    assert plan.count(BAR) == 1
    moved = plan.replace(BAR, BAR[:-4] + "0.10")
    rec = json.loads(c.show(MERGE_224, R))
    s = score_frozen(protocol._select_gates_block(moved), rec)
    rec.update(verdict=s["verdict"], gates=s["gates"], gates_sha256=s["gates_sha256"])
    c.rerun_verdict, c.rerun_sha = s["verdict"], s["gates_sha256"]
    c.rerun = c.commit(MERGE_224, {RERUN: (json.dumps(rec, indent=1) + "\n").encode("utf-8")}, "re-run",
                       "demo/working-copy-bar")
    c.bar = c.commit(MERGE_224, {P: moved.encode("utf-8")}, "move one bar", "demo/move-one-bar")
    return c


def run(where: Path, *argv, capsys):
    cwd = os.getcwd()
    try:
        os.chdir(where)
        code = cli.main(list(argv))
    finally:
        os.chdir(cwd)
    return code, capsys.readouterr().out


@pytest.fixture(scope="module")
def passed(clone, tmp_path_factory):
    out_json = tmp_path_factory.mktemp("report224") / "report.json"
    cwd = os.getcwd()
    import io
    import contextlib
    buf = io.StringIO()
    try:
        os.chdir(clone.root)
        with contextlib.redirect_stdout(buf):
            code = cli.main(["check", "--base", MERGE_224 + "^1", "--head", MERGE_224, "--json", str(out_json)])
    finally:
        os.chdir(cwd)
    return code, buf.getvalue(), json.loads(out_json.read_text(encoding="ascii"))


def test_the_224_merge_passes_with_the_seal_order_time_and_landed_row_of_4a(passed):
    code, out, rep = passed
    assert code == 0, out
    lines = [ln.strip() for ln in out.splitlines()]
    assert "PASS  RESULT_pmnist_robustness_2026_10_23.md" in lines
    assert ("seal bytes 2068e673... landed at 976b097f (#215); order commit 3598e59d; "
            "time AUTHOR_ORDER (T2 not read, T3 none)") in lines
    assert f"landed row 9 of 13: {LANDED}" in lines
    assert f"G_B_long 0.1097 >= 0.15 (margin -0.0403) -> {ROBUST}" in lines
    assert any(ln.startswith("not examined: the run itself") for ln in lines)
    assert "nothing in this run is ordered after the seal" in out
    assert lines[-1] == "PASS  exit 0"


def test_the_224_rescore_is_r2(passed):
    code, out, rep = passed
    (pair,) = rep["pairs"]
    assert pair["receipt"] == R and pair["prereg"] == P and pair["receipt_commit"].startswith("237e7784")
    assert pair["rescore"]["verdict"] == LANDED and pair["rescore"]["gates_sha256"] == SEALED_SHA
    assert pair["rescore"]["total"] is True and pair["rescore"]["row"] == 8
    assert pair["recorded"]["verdict"] == LANDED and pair["recorded"]["gates_sha256"] == SEALED_SHA
    assert pair["seal"]["blob_sha256"].startswith("2068e673") and pair["seal"]["landing"].startswith("976b097f")
    assert pair["seal"]["order"].startswith("3598e59d") and pair["seal"]["add"].startswith("3598e59d")
    assert pair["declared"][0]["result"] == RESULT and pair["declared"][0]["verdict"] == LANDED
    assert [f["code"] for f in pair["findings"] if f["kind"] != "NOTE"] == []


def test_the_224_flip_table_is_r9(passed):
    code, out, rep = passed
    lad = rep["pairs"][0]["ladder"]
    assert lad["landed"] == 8 and len(lad["rows"]) == 13 and [r["landed"] for r in lad["rows"]].count(True) == 1
    gates = {g["name"]: g for g in lad["gates"]}
    flips = {f["gate"]: f for f in lad["flips"]}
    assert [g["name"] for g in lad["gates"]] == [r[0] for r in R9]
    for name, op, bar, observed, fired, row, verdict in R9:
        assert (gates[name]["op"], gates[name]["bar"], gates[name]["observed"], gates[name]["fired"]) == \
            (op, bar, observed, fired), name
        assert (flips[name]["row"], flips[name]["verdict"]) == (row, verdict), name
        assert flips[name]["moves"] is (verdict != LANDED), name
    assert sum(f["moves"] for f in lad["flips"]) == 8
    assert gates["G_P_plumbing"]["margin"] == 0
    assert round(gates["G_B_long"]["margin"], 4) == -0.0403
    for name, op, bar, observed, fired, row, verdict in R9:
        if verdict != LANDED:
            assert f"{name} " in out and f"-> {verdict}" in out
    assert "flips that leave it: G_R_diversity, G_R_rotation" in out


def test_a1_a_result_that_types_a_rung_the_receipt_did_not_land_on_is_verdict_differs(clone, capsys):
    code, out = run(clone.root, "check", "--base", MERGE_224, "--head", "demo/typed-rung", capsys=capsys)
    assert code == 1, out
    assert "FAIL VERDICT_DIFFERS" in out
    assert f"the sealed decision scores this receipt {LANDED}; the RESULT {RESULT} prints {ROBUST}" in out
    assert out.rstrip().splitlines()[-1].startswith("FAIL  exit 1")


def test_a2_a_consistent_rerun_under_a_working_copy_bar_is_digest_unsealed_and_verdict_differs(clone, capsys):
    assert clone.rerun_verdict == ROBUST and clone.rerun_sha == MOVED_SHA          # R2
    code, out = run(clone.root, "check", "--base", MERGE_224, "--head", "demo/working-copy-bar", capsys=capsys)
    assert code == 1, out
    assert f"FAIL DECISION_DIGEST_UNSEALED  the receipt records gates {MOVED_SHA}; the sealed PREREG holds {SEALED_SHA}" in out
    assert f"FAIL VERDICT_DIFFERS  the sealed decision scores this receipt {LANDED}; the receipt records {ROBUST}" in out
    assert "C0 sealed decisions: 0 moved" in out and f"FAIL  {RERUN}" in out


def test_a3_external1_under_the_reconstructed_table_is_rv4_and_its_licence_is_rv3(clone, capsys, tmp_path):
    out_json = tmp_path / "ladder.json"
    code, out = run(clone.root, "ladder", "papers/closed-model-frontier/external1_adjudication.json", "--rev",
                    EXTERNAL1_RESULT, "--prereg", str(TABLE), "--json", str(out_json), capsys=capsys)
    assert code == 1, out
    rep = json.loads(out_json.read_text(encoding="ascii"))
    rs = rep["rescore"]
    assert rs["verdict"] == "FAILURE__gate_claim_not_yet_true" and rs["gates"] == {"G_rel": True, "G_E1": False}
    assert rs["gates_sha256"] == EXTERNAL1_SHA and rs["row"] == 1
    assert rs["withdraws"] == [{"id": "gate-claim", "quote": QUOTE, "pattern": FIVE}]
    assert "landed row 2 of 3: FAILURE__gate_claim_not_yet_true" in out
    assert "G_E1 0.23 >= 0.95 (margin -0.72)" in out
    assert "surfaces read 328" in out
    for loc in WITHDRAWN:
        assert f"FAIL LICENCE_WITHDRAWN {loc}" in out, loc
    assert (f'withdrawn by the landed row, still printed: pattern "{FIVE}" in README.md, '
            f'action.yml, styxx/__init__.py, styxx/diffgate.py') in out
    assert "the licence's verbatim quote matches 0 surfaces" in out and "lint would have refused it before data" in out
    lic = rep["licences"]
    assert lic["surfaces"] == 328 and sorted(lic["licences"][0]["surfaces"]) == \
        ["README.md", "action.yml", "styxx/__init__.py", "styxx/diffgate.py"]
    assert rep["quote_coverage"] == [{"id": "gate-claim", "quote": QUOTE, "surfaces": []}]


def test_a4_a_bar_moved_in_the_sealed_prereg_is_decision_edited_after_seal(clone, capsys):
    code, out = run(clone.root, "check", "--base", MERGE_224, "--head", "demo/move-one-bar", capsys=capsys)
    assert code == 1, out
    assert f"FAIL DECISION_EDITED_AFTER_SEAL  {P} held the sealed fence {SEALED_SHA}" in out
    assert "path protection" in out


def test_the_demo_wrote_nothing_into_the_styxx_store_or_its_refs(clone):
    assert _git(ROOT, "for-each-ref", "refs/heads/demo") == ""
    for commit in (clone.typed, clone.rerun, clone.bar):
        r = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", commit], capture_output=True)
        assert r.returncode != 0, commit
    alternates = (clone.root / ".git" / "objects" / "info" / "alternates").read_text(encoding="utf-8")
    assert alternates.strip()
