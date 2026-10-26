# -*- coding: utf-8 -*-
"""prereg-ci B7: the outcome ladder, the flip table, licences by landed row, and the report.

DESIGN 3.4 C7: every outcome row in order, the landed row marked; each gate with metric, op, bar,
margin, and whether the landed row reads it; the single-gate flip table, which prints every gate whose
flip moves the verdict with its margin (RED_TEAM m3: no "decisive bar"). DESIGN 3.2(e): the report is
``styxx.prereg-ci/report/v0``, JSON plus a markdown job summary, with the scorer's identity (module
digests by charon's rule, the runtime's three fields) and no absolute path, host name or
``sys.version`` string. The B5 review: ``licences`` names the landed row by its index where verdicts
repeat, and the job summary aggregates matches per licence and file (its N3).

Everything here is pure or reads a ``git init`` repository in tmp_path; nothing reaches a network.
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx.prereg_ci import core  # noqa: E402
from styxx.prereg_ci import report as R  # noqa: E402

F = "`" * 3


def prereg(spec: dict) -> str:
    return f"# PREREG\n\n{F}gates\n{json.dumps(spec, indent=1)}\n{F}\n"


ALL_OPS = {
    "gates": {"G_ge": {"metric": "a", "op": ">=", "value": 0.5},
              "G_gt": {"metric": "b", "op": ">", "value": 0.5},
              "G_le": {"metric": "c", "op": "<=", "value": 0.5},
              "G_lt": {"metric": "d", "op": "<", "value": 0.5},
              "G_eq": {"metric": "e", "op": "==", "value": 1.0}},
    "outcomes": [{"when": {"G_eq": False}, "verdict": "INVALID__plumbing"},
                 {"when": {"G_ge": True, "G_gt": True}, "verdict": "SHIP"},
                 {"when": {"G_ge": True}, "verdict": "HOLD"},
                 {"when": {"G_ge": False}, "verdict": "NULL"}],
    "smoke_verdict": "SMOKE"}


def test_the_ladder_reads_each_gate_with_its_metric_op_bar_and_margin():
    receipt = {"a": 0.7, "b": 0.5, "c": 0.4, "d": 0.6, "e": 1.0}
    rs = core.rescore(prereg(ALL_OPS), receipt)
    assert rs["code"] is None and rs["verdict"] == "HOLD" and rs["row"] == 2
    lad = core._ladder(rs, prereg(ALL_OPS), receipt)
    by = {g["name"]: g for g in lad["gates"]}
    # the margin is positive where the gate fired with room, negative where it did not fire
    assert by["G_ge"]["observed"] == 0.7 and by["G_ge"]["fired"] is True and by["G_ge"]["margin"] == pytest.approx(0.2)
    assert by["G_gt"]["fired"] is False and by["G_gt"]["margin"] == pytest.approx(0.0)
    assert by["G_le"]["fired"] is True and by["G_le"]["margin"] == pytest.approx(0.1)
    assert by["G_lt"]["fired"] is False and by["G_lt"]["margin"] == pytest.approx(-0.1)
    assert by["G_eq"]["fired"] is True and by["G_eq"]["margin"] == 0.0
    assert [g["name"] for g in lad["gates"]] == list(ALL_OPS["gates"])
    for name in ("G_ge", "G_gt", "G_le", "G_lt", "G_eq"):
        assert by[name]["metric"] == ALL_OPS["gates"][name]["metric"]
        assert by[name]["op"] == ALL_OPS["gates"][name]["op"]
        assert by[name]["bar"] == ALL_OPS["gates"][name]["value"]
    # the landed row (HOLD, index 2) reads G_ge alone; rows are every row in order, the landed one marked
    assert by["G_ge"]["read_by_landed"] is True and by["G_le"]["read_by_landed"] is False
    assert [r["verdict"] for r in lad["rows"]] == ["INVALID__plumbing", "SHIP", "HOLD", "NULL"]
    assert [r["landed"] for r in lad["rows"]] == [False, False, True, False]
    assert lad["landed"] == 2 and lad["verdict"] == "HOLD"


def test_the_flip_table_names_every_gate_whose_flip_moves_the_verdict():
    receipt = {"a": 0.7, "b": 0.5, "c": 0.4, "d": 0.6, "e": 1.0}
    rs = core.rescore(prereg(ALL_OPS), receipt)
    flips = {f["gate"]: f for f in core._ladder(rs, prereg(ALL_OPS), receipt)["flips"]}
    assert flips["G_ge"]["verdict"] == "NULL" and flips["G_ge"]["moves"] is True
    assert flips["G_gt"]["verdict"] == "SHIP" and flips["G_gt"]["moves"] is True
    assert flips["G_eq"]["verdict"] == "INVALID__plumbing" and flips["G_eq"]["moves"] is True
    for g in ("G_le", "G_lt"):
        assert flips[g]["moves"] is False and flips[g]["verdict"] == "HOLD" and flips[g]["row"] == 2
    assert flips["G_ge"]["margin"] == pytest.approx(0.2)


def test_a_flip_onto_no_row_is_a_hole_and_moves_the_verdict():
    spec = {"gates": {"A": {"metric": "a", "op": ">=", "value": 1}, "B": {"metric": "b", "op": ">=", "value": 1}},
            "outcomes": [{"when": {"A": True, "B": True}, "verdict": "YES"},
                         {"when": {"A": False, "B": False}, "verdict": "NO"}],
            "smoke_verdict": "SMOKE"}
    rs = core.rescore(prereg(spec), {"a": 2, "b": 2})
    flips = {f["gate"]: f for f in core._ladder(rs, prereg(spec), {"a": 2, "b": 2})["flips"]}
    assert flips["A"]["row"] is None and flips["A"]["verdict"] is None and flips["A"]["moves"] is True


def test_the_ladder_reads_the_metrics_at_the_receipt_pointer():
    spec = dict(ALL_OPS, receipt_pointer="/metrics")
    receipt = {"metrics": {"a": 0.4, "b": 0.5, "c": 0.4, "d": 0.6, "e": 1.0}}
    rs = core.rescore(prereg(spec), receipt)
    assert rs["verdict"] == "NULL"
    lad = core._ladder(rs, prereg(spec), receipt)
    assert {g["name"]: g["observed"] for g in lad["gates"]}["G_ge"] == 0.4


def test_no_ladder_for_a_smoke_receipt_a_refusal_or_a_locks_decision():
    assert core._ladder(core.rescore(prereg(ALL_OPS), {"smoke": True}), prereg(ALL_OPS), {"smoke": True}) is None
    refused = core.rescore(prereg(ALL_OPS), {"a": 1})
    assert refused["code"] == "SCORER_REFUSES" and core._ladder(refused, prereg(ALL_OPS), {"a": 1}) is None
    locks = f"# L\n\n{F}locks\n" + json.dumps({"scorer": "s.py", "reads": [], "outcomes": ["A"]}) + f"\n{F}\n"
    assert core._ladder(core.rescore(locks, {}), locks, {}) is None


def test_the_number_format_rounds_to_four_places_and_keeps_ints():
    assert R._num(0.1097) == "0.1097"
    assert R._num(0.1097 - 0.15) == "-0.0403"
    assert R._num(1.0) == "1.0" and R._num(0.0) == "0.0" and R._num(-0.0) == "0.0"
    assert R._num(27) == "27" and R._num(0.23) == "0.23"
    assert R._num(True) == "true" and R._num(None) == "null"


# ---- licences by the landed row (the B5 review's carried finding) ----------------------------------------

REPEATED = {"gates": {"G": {"metric": "m", "op": ">=", "value": 1}, "H": {"metric": "n", "op": ">=", "value": 1}},
            "outcomes": [{"when": {"G": True}, "verdict": "HOLD",
                          "withdraws": [{"id": "one", "quote": "the alpha claim", "pattern": "alpha claim"}]},
                         {"when": {"G": False}, "verdict": "HOLD",
                          "withdraws": [{"id": "two", "quote": "the beta claim", "pattern": "beta claim"}]}],
            "smoke_verdict": "SMOKE"}
SURFACES = [("README.md", "the alpha claim\nthe beta claim\n")]


def test_licences_by_row_refuse_only_the_landed_rows_licences_where_verdicts_repeat():
    by_verdict = core.licences(prereg(REPEATED), SURFACES, landed="HOLD")
    assert sorted(f["licence"] for f in by_verdict["findings"]) == ["one", "two"]
    by_row = core.licences(prereg(REPEATED), SURFACES, landed="HOLD", row=0)
    assert [(f["code"], f["licence"]) for f in by_row["findings"]] == [("LICENCE_WITHDRAWN", "one")]
    assert [(f["code"], f["licence"]) for f in
            core.licences(prereg(REPEATED), SURFACES, landed="HOLD", row=1)["findings"]] == [("LICENCE_WITHDRAWN", "two")]


def test_licences_by_row_refuse_a_row_that_is_not_the_landed_verdicts():
    with pytest.raises(ValueError):
        core.licences(prereg(REPEATED), SURFACES, landed="SMOKE", row=0)
    with pytest.raises(ValueError):
        core.licences(prereg(REPEATED), SURFACES, landed=None, row=0)
    with pytest.raises(ValueError):
        core.licences(prereg(REPEATED), SURFACES, landed="HOLD", row=7)


def test_the_quote_coverage_reads_each_licences_quote_as_a_literal_pattern():
    spec = {"gates": {"G": {"metric": "m", "op": ">=", "value": 1}},
            "outcomes": [{"when": {"G": False}, "verdict": "FAILURE",
                          "withdraws": [{"id": "c", "quote": "an agent's PR summary cannot (really) lie",
                                         "pattern": "cannot \\(really\\) lie"}]},
                         {"when": {"G": True}, "verdict": "SHIP"}],
            "smoke_verdict": "SMOKE"}
    surfaces = [("a.md", "An agent’s PR   summary\ncannot (really) lie."), ("b.md", "cannot (really) lie"),
                ("c.md", None)]
    cov = core._quote_coverage(prereg(spec), surfaces)
    assert cov == [{"id": "c", "quote": "an agent's PR summary cannot (really) lie", "surfaces": ["a.md"]}]


# ---- the report -------------------------------------------------------------------------------------------

def test_the_schema_names_the_report_and_its_version():
    assert R._SCHEMA == "styxx.prereg-ci/report/v0"


def test_the_exit_code_reads_a_refusal_ahead_of_an_input_error_and_an_unchecked_reading():
    f = lambda kind: {"code": "X", "kind": kind, "detail": ""}  # noqa: E731
    assert R._exit([]) == 0 and R._exit([f("NOTE")]) == 0
    assert R._exit([f("UNCHECKABLE"), f("NOTE")]) == 3
    assert R._exit([f("INPUT"), f("UNCHECKABLE")]) == 2
    assert R._exit([f("FAIL"), f("INPUT"), f("UNCHECKABLE")]) == 1
    assert [R._VERDICT[c] for c in (0, 1, 2, 3)] == ["PASS", "FAIL", "INPUT_ERROR", "UNCHECKABLE"]


def test_every_code_the_check_can_print_has_one_kind():
    kinds = R._kinds()
    assert kinds["SEAL_NOT_BEFORE_RECEIPT"] == "FAIL"
    assert kinds["LOCKED_FILE_MOVED"] == "NOTE"
    assert kinds["CONFIG_INVALID"] == "INPUT"
    for code in ("VERDICT_DIFFERS", "DECISION_DIGEST_UNSEALED", "RECEIPT_REWRITTEN", "DECISION_EDITED_AFTER_SEAL",
                 "LICENCE_WITHDRAWN", "TIME_BELOW_POLICY", "SCORER_READS_UNLOCKED", "RESULT_REFUSES"):
        assert kinds[code] == "FAIL", code
    assert set(kinds.values()) <= {"FAIL", "INPUT", "NOTE", "UNCHECKABLE"}


def test_the_module_digests_follow_charons_rule():
    from styxx import charon
    for name, sha in R._module_pairs():
        assert [name, sha] == list(charon._module_pair(name)), name
    names = [n for n, _ in R._module_pairs()]
    for name in ("styxx.prereg_ci.core", "styxx.protocol", "styxx.research_capsule", "styxx.sworn"):
        assert name in names


def test_the_runtime_is_the_three_compared_fields_and_never_sys_version():
    from styxx import instrument
    rt = R._runtime()
    assert set(rt) == set(instrument.RUNTIME_COMPARED)
    assert rt == {k: instrument.runtime()[k] for k in instrument.RUNTIME_COMPARED}
    assert sys.version not in json.dumps(rt)


def _report_of_a_tmp_repo(tmp_path):
    from tests.test_prereg_ci_git import Git
    from styxx.prereg_ci import cli
    g = Git(tmp_path / "home" / "someone" / "repo")
    g.write("p/PREREG_a.md", prereg(ALL_OPS))
    base = g.commit("seal")
    receipt = {"prereg": "PREREG_a.md", "a": 0.7, "b": 0.5, "c": 0.4, "d": 0.6, "e": 1.0}
    rs = core.rescore(prereg(ALL_OPS), receipt)
    receipt.update(verdict=rs["verdict"], gates=rs["gates"], gates_sha256=rs["gates_sha256"])
    g.write("p/receipt.json", json.dumps(receipt))
    head = g.commit("result")
    out = tmp_path / "report.json"
    md = tmp_path / "summary.md"
    import os
    cwd = os.getcwd()
    try:
        os.chdir(g.root)
        code = cli.main(["check", "--base", base, "--head", head, "--json", str(out), "--summary", str(md)])
    finally:
        os.chdir(cwd)
    return code, out, md, g


def test_the_report_holds_no_absolute_path_no_host_name_and_no_sys_version(tmp_path):
    code, out, md, g = _report_of_a_tmp_repo(tmp_path)
    assert code == 0
    raw = out.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw and raw.endswith(b"\n")
    text = raw.decode("ascii")
    rep = json.loads(text)
    assert rep["schema"] == "styxx.prereg-ci/report/v0" and rep["verdict"] == "PASS" and rep["exit"] == 0
    for needle in (str(tmp_path), str(tmp_path).replace("\\", "/"), str(Path.home()),
                   str(Path.home()).replace("\\", "/"), sys.version, socket.gethostname(), platform.node()):
        assert needle and needle not in text, needle
        assert needle not in md.read_text(encoding="utf-8"), needle
    # no drive letter and no leading slash anywhere a path is printed
    assert not re.search(r'"[A-Za-z]:[\\/]', text) and '"/' not in text
    assert rep["tool"]["name"] == "prereg-ci" and rep["tool"]["version"] == "v0"
    assert rep["tool"]["runtime"] == R._runtime() and rep["tool"]["modules"] == [list(p) for p in R._module_pairs()]


def test_the_report_carries_every_reading_the_design_names_per_pair(tmp_path):
    code, out, md, g = _report_of_a_tmp_repo(tmp_path)
    rep = json.loads(out.read_text(encoding="ascii"))
    (pair,) = rep["pairs"]
    for key in ("receipt", "receipt_commit", "prereg", "seal", "effect", "time", "blob_history", "decision",
                "rescore", "recorded", "declared", "ladder", "licences", "findings", "reading", "not_examined"):
        assert key in pair, key
    seal = pair["seal"]
    for key in ("blob_sha256", "gates_sha256", "landing", "add", "order", "manifest_id", "form"):
        assert key in seal, key
    assert "bytes" not in seal and "bytes" not in pair["effect"]
    assert pair["rescore"]["verdict"] == "HOLD" and pair["reading"] == "PASS"
    assert pair["ladder"]["landed"] == 2 and len(pair["ladder"]["flips"]) == 5
    assert pair["time"]["name"] == "AUTHOR_ORDER" and pair["time"]["line"] == "time AUTHOR_ORDER (T2 not read, T3 none)"
    assert any("the run itself" in n for n in pair["not_examined"])
    assert rep["policies"]["receipt_rewrite"] == "fail" and rep["policies"]["require_time"] == "T1"


def test_the_summary_is_markdown_with_the_verdict_and_each_pair(tmp_path):
    code, out, md, g = _report_of_a_tmp_repo(tmp_path)
    s = md.read_text(encoding="utf-8")
    assert s.startswith("## prereg-ci check: PASS")
    assert "| `p/receipt.json` |" in s and "HOLD" in s and "AUTHOR_ORDER" in s
    assert "not examined" in s


def test_the_summary_aggregates_licence_matches_per_licence_and_file():
    rep = {"command": "check", "verdict": "FAIL", "exit": 1, "base": "a" * 40, "head": "b" * 40, "merge_base": "a" * 40,
           "policies": {}, "findings": [], "pairs": [], "not_examined": [],
           "tree_licences": [{"prereg": "p/PREREG.md", "receipt": None, "landed": None, "licences": [
               {"id": "x", "quote": "q", "pattern": "p", "verdict": "HOLD", "row": 0,
                "matches": [{"path": "README.md", "line": i, "col": 1, "withdrawn": False} for i in range(1, 51)]
                + [{"path": "docs/a.md", "line": 3, "col": 1, "withdrawn": True}]}]}]}
    s = R._markdown(rep)
    assert "| `x` | `README.md` | 50 | 0 |" in s and "| `x` | `docs/a.md` | 1 | 1 |" in s
    assert s.count("README.md") == 1


def test_json_drops_bytes_and_reads_tuples_and_sets_as_lists():
    assert R._jsonable({"bytes": b"x", "out_bytes": b"y", "a": (1, 2), "b": {3, 1}, "c": [{"bytes": b"z", "d": 1}]}) == \
        {"a": [1, 2], "b": [1, 3], "c": [{"d": 1}]}


def test_a_digest_of_the_report_bytes_is_stable_for_one_input(tmp_path):
    code, out, md, g = _report_of_a_tmp_repo(tmp_path)
    one = hashlib.sha256(out.read_bytes()).hexdigest()
    import os
    from styxx.prereg_ci import cli
    again = tmp_path / "again.json"
    cwd = os.getcwd()
    try:
        os.chdir(g.root)
        cli.main(["check", "--base", "HEAD~1", "--head", "HEAD", "--json", str(again)])
    finally:
        os.chdir(cwd)
    assert hashlib.sha256(again.read_bytes()).hexdigest() == one
