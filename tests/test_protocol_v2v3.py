"""protocol v2/v3 — every defect the 2026-08-07 pre-release red team found, as a regression.

The red team's verdict was DO NOT SHIP on six defects. Each one below is that defect, written so
it fails against the pre-fix code. Nothing here is hypothetical: every case reproduced.
"""
import json
import math
import subprocess
import pickle
from dataclasses import asdict
from pathlib import Path

import pytest

from styxx.protocol import Experiment, Verdict, GateSpecError, undeclared_power_gates


def _prereg(tmp_path, gates, name="PREREG_t.md"):
    """A committed prereg — protocol refuses to score an uncommitted one."""
    d = tmp_path / "r"
    d.mkdir(exist_ok=True)
    f = d / name
    spec = {"gates": gates, "outcomes": [{"when": {}, "verdict": "V_DEFAULT"}],
            "smoke_verdict": "S_SMOKE"}
    f.write_text("t\n\n```gates\n" + json.dumps(spec) + "\n```\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(d)], capture_output=True)
    subprocess.run(["git", "-C", str(d), "add", "-A"], capture_output=True)
    subprocess.run(["git", "-C", str(d), "-c", "user.email=a@b", "-c", "user.name=t",
                    "commit", "-qm", "x"], capture_output=True)
    return f


G = {"metric": "m", "op": ">=", "value": 1}


# ---- defect 1: blank / non-string power_basis counted as a declaration -------------------
@pytest.mark.parametrize("pb", [" ", "\n", "\t  ", True, 1, ["x"], {"a": 1}])
def test_strict_mode_refuses_a_blank_or_nonstring_power_basis(tmp_path, pb):
    """`if not v` accepted ' ' and True. A whitespace declaration carries zero information and
    made the refusal that exists to stop bar-without-power errors decorative."""
    f = _prereg(tmp_path, {"G1": {**G, "power_basis": pb}})
    with pytest.raises(GateSpecError, match="power basis"):
        Experiment(f, require_power_basis=True)


def test_strict_mode_accepts_a_real_declaration(tmp_path):
    f = _prereg(tmp_path, {"G1": {**G, "power_basis": "derived from the null at n=200"}})
    assert Experiment(f, require_power_basis=True).undeclared_power_gates == []


def test_undeclared_power_gates_module_function_exists(tmp_path):
    """Frozen as a deliverable in the v2 prereg, silently dropped from the implementation, and
    caught by the red team rather than by the exam that was supposed to cover it."""
    f = _prereg(tmp_path, {"G1": dict(G), "G2": {**G, "power_basis": "stated"}})
    assert undeclared_power_gates(f) == ["G1"]


# ---- defect 2: check_metrics false-alarmed on smoke results -----------------------------
def test_check_metrics_does_not_false_alarm_on_a_smoke_result(tmp_path):
    """Smoke runs score by TYPE and never read gate metrics, so absent paths are expected, not a
    finding. check_metrics reported every path absent on nine committed smoke results."""
    f = _prereg(tmp_path, {"G1": dict(G)})
    e = Experiment(f)
    cm = e.check_metrics({"smoke": True})
    assert cm["G1"]["present"] is False
    assert "smoke" in cm["G1"]["note"]
    assert e.score({"smoke": True}, smoke=True).verdict == "S_SMOKE"


# ---- defect 3: NaN sealed silently; non-comparable crashed score() ----------------------
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), None, "str", {"a": 1}, [1], True])
def test_a_non_comparable_metric_refuses_instead_of_scoring(tmp_path, bad):
    """NaN made every comparison False, so the frozen table returned its false branch as a
    legitimate SEALED verdict with no refusal anywhere. Others raised TypeError, which is not in
    seal()'s except clause, so a malformed result crashed the seal instead of refusing it."""
    f = _prereg(tmp_path, {"G1": dict(G)})
    with pytest.raises(GateSpecError, match="cannot be"):
        Experiment(f).score({"m": bad})


@pytest.mark.parametrize("bad", [float("nan"), None, "str", {"a": 1}])
def test_check_metrics_reports_non_comparable_as_present_but_unusable(tmp_path, bad):
    """present != usable. Reporting a NaN as simply 'present' green-lit a run that would seal a
    verdict computed from it."""
    f = _prereg(tmp_path, {"G1": dict(G)})
    cm = Experiment(f).check_metrics({"m": bad})
    assert cm["G1"]["present"] is True
    assert cm["G1"]["usable"] is False


def test_check_metrics_and_score_agree_on_a_good_result(tmp_path):
    f = _prereg(tmp_path, {"G1": dict(G)})
    e = Experiment(f)
    assert e.check_metrics({"m": 5})["G1"]["usable"] is True
    assert e.score({"m": 5}).verdict == "V_DEFAULT"


# ---- defect 4: missing / null metric crashed the pre-run checker -------------------------
@pytest.mark.parametrize("metric", [None, "", 5, ["m"]])
def test_a_missing_or_nonstring_metric_path_refuses_at_construction(tmp_path, metric):
    """None went into metric_paths and made check_metrics raise AttributeError — the safety tool
    crashing on the most mis-specified gate there is."""
    g = {"op": ">=", "value": 1} if metric is None else {"metric": metric, "op": ">=", "value": 1}
    f = _prereg(tmp_path, {"G1": g})
    with pytest.raises(GateSpecError, match="metric"):
        Experiment(f)


# ---- defect 5: Verdicts shared one mutable dict ------------------------------------------
def test_each_verdict_owns_its_metadata(tmp_path):
    """Mutating one receipt rewrote its siblings and the Experiment. A Verdict is the receipt;
    it must not be a live handle onto shared state."""
    f = _prereg(tmp_path, {"G1": {**G, "power_basis": "stated"}})
    e = Experiment(f)
    v1, v2 = e.score({"m": 5}), e.score({"m": 5})
    assert v1.power_basis is not e.power_basis
    assert v1.power_basis is not v2.power_basis
    assert v1.metric_paths is not v2.metric_paths
    v1.power_basis["G1"] = "MUTATED"
    v1.undeclared_power_gates.append("X")
    assert v2.power_basis["G1"] == "stated"
    assert e.score({"m": 5}).undeclared_power_gates == []


# ---- backward compatibility: the catastrophic question ------------------------------------
def test_verdict_still_constructs_positionally_and_pickles():
    """Three fields were appended. 169 sealed certificates depend on nothing breaking here."""
    v = Verdict("V", {"G": True}, "abc", "deadbeef")
    assert v.power_basis == {} and v.metric_paths == {} and v.undeclared_power_gates == []
    assert pickle.loads(pickle.dumps(v)).verdict == "V"
    assert json.dumps(asdict(v))


def test_every_committed_result_still_scores_to_its_stored_verdict():
    """The whole-repo integrity check, as a test rather than a one-off script: any diff here
    breaks a committed seal."""
    root = Path(__file__).resolve().parent.parent / "papers"
    if not root.exists():
        pytest.skip("papers/ not present")
    scored = diffs = 0
    for res in sorted(root.rglob("*_result*.json")):
        try:
            r = json.loads(res.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(r, dict):        # some receipts are top-level lists
            continue
        pr = res.parent / str(r.get("prereg", ""))
        if not r.get("prereg") or not pr.exists() or "verdict" not in r:
            continue
        if str(r["verdict"]).startswith("UNSCORED__"):
            continue
        try:
            v = Experiment(pr).score(r, smoke=bool(r.get("smoke")))
        except Exception:
            continue
        scored += 1
        diffs += (v.verdict != r["verdict"])
    assert scored > 20, f"only {scored} results exercised — the sweep is not covering the corpus"
    assert diffs == 0, f"{diffs} committed verdict strings changed"
