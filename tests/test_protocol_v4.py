"""Protocol v4 (declared gate composition) regression tests.

One test per defect class from the v4 exam, plus the E1 retro-case pinned against the committed
receipt. If _check_composition ever regresses, the failure names the exact mutant that got
through.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from styxx.protocol import Experiment, GateSpecError  # noqa: E402

BASE = {"gates": {"G": {"metric": "m", "op": "<=", "value": 0.2,
                        "agg": "min", "over": "pool", "excluding": "dq"}},
        "outcomes": [{"when": {"G": True}, "verdict": "PASS"},
                     {"when": {"G": False}, "verdict": "FAIL"}],
        "smoke_verdict": "SMOKE"}
POOL = {"a": 0.30, "b": 0.15, "c": 0.25}


@pytest.fixture()
def mk(tmp_path):
    def _mk(gates=None):
        spec = json.loads(json.dumps(BASE))
        if gates:
            spec["gates"]["G"].update(gates)
            for k, v in list(spec["gates"]["G"].items()):
                if v is None:
                    del spec["gates"]["G"][k]
        p = tmp_path / "PREREG_case.md"
        p.write_text("# case\n\n```gates\n" + json.dumps(spec) + "\n```\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                        "commit", "-qm", "case"], cwd=tmp_path, check=True)
        return p
    return _mk


def test_e1_shape_refuses(mk):
    """The defining case: metric quotes the unrestricted min while exclusions exist."""
    with pytest.raises(GateSpecError, match="COMPOSITION VIOLATION"):
        Experiment(mk()).score({"m": 0.15, "pool": POOL, "dq": ["b"]})


def test_correct_declaration_scores(mk):
    v = Experiment(mk()).score({"m": 0.25, "pool": POOL, "dq": ["b"]})
    assert v.verdict == "FAIL"          # 0.25 > 0.2 — scored, honestly, not refused


def test_no_declaration_unchanged(mk):
    v = Experiment(mk({"agg": None, "over": None, "excluding": None})).score({"m": 0.15})
    assert v.verdict == "PASS"


def test_half_declaration_refuses_at_init(mk):
    with pytest.raises(GateSpecError, match="both 'agg' and 'over'"):
        Experiment(mk({"over": None, "excluding": None}))


def test_bad_agg_refuses_at_init(mk):
    with pytest.raises(GateSpecError, match="'agg' must be"):
        Experiment(mk({"agg": "mean"}))


def test_over_missing_refuses(mk):
    with pytest.raises(GateSpecError):
        Experiment(mk()).score({"m": 0.25, "dq": ["b"]})


def test_over_not_dict_refuses(mk):
    with pytest.raises(GateSpecError, match="non-empty dict"):
        Experiment(mk()).score({"m": 0.25, "pool": [0.3, 0.15], "dq": ["b"]})


def test_unknown_exclusion_refuses(mk):
    with pytest.raises(GateSpecError, match="absent from 'over'"):
        Experiment(mk()).score({"m": 0.15, "pool": POOL, "dq": ["zz"]})


def test_all_excluded_refuses(mk):
    with pytest.raises(GateSpecError, match="empty population"):
        Experiment(mk()).score({"m": 0.25, "pool": POOL, "dq": ["a", "b", "c"]})


def test_nan_member_refuses(mk):
    with pytest.raises(GateSpecError, match="cannot be aggregated"):
        Experiment(mk()).score({"m": 0.25, "pool": {"a": 0.3, "b": float("nan")}, "dq": []})


def test_bool_member_refuses(mk):
    with pytest.raises(GateSpecError, match="cannot be aggregated"):
        Experiment(mk()).score({"m": 0.25, "pool": {"a": 0.3, "b": True}, "dq": []})


def test_max_agg_checks_too(mk):
    with pytest.raises(GateSpecError, match="COMPOSITION VIOLATION"):
        Experiment(mk({"agg": "max", "op": ">=", "value": 0.1})).score(
            {"m": 0.30, "pool": POOL, "dq": ["a"]})


def test_near_miss_within_rounding_refuses(mk):
    """A value off by 1e-6 is a mismatch, not a rounding forgiveness."""
    with pytest.raises(GateSpecError, match="COMPOSITION VIOLATION"):
        Experiment(mk()).score({"m": 0.250001, "pool": POOL, "dq": ["b"]})


def _mk_raw(tmp_path, md_text):
    p = tmp_path / "PREREG_raw.md"
    p.write_text(md_text, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "raw"], cwd=tmp_path, check=True)
    return p


def test_multi_fence_refuses(tmp_path):
    """Red team D1: a hidden or display fence must not shadow the visible block."""
    F = "```"
    honest = json.dumps(BASE)
    decoy = honest.replace("0.2", "99")
    p = _mk_raw(tmp_path,
                f"# d\n\n<!--\n{F}gates\n{decoy}\n{F}\n-->\n\n{F}gates\n{honest}\n{F}\n")
    with pytest.raises(GateSpecError, match="2 .*fences"):
        Experiment(p)


def test_duplicate_json_key_refuses(tmp_path):
    """Red team D2: json.loads keeping the last duplicate key was a shadowing channel."""
    F = "```"
    dup = json.dumps(BASE).replace('"excluding": "dq"',
                                   '"excluding": "dq", "excluding": "dq_decoy"')
    p = _mk_raw(tmp_path, f"# d\n\n{F}gates\n{dup}\n{F}\n")
    with pytest.raises(GateSpecError, match="duplicate key"):
        Experiment(p)


def test_mixed_type_excluding_refuses_typed(mk):
    """Red team D8: previously a raw TypeError, i.e. a crash instead of a refusal."""
    with pytest.raises(GateSpecError, match="must be strings"):
        Experiment(mk()).score({"m": 0.25, "pool": POOL, "dq": [1, "b"]})


def test_numpy_members_aggregate(mk):
    """Red team D9: finite numpy scalars are numbers, not refusals."""
    np = pytest.importorskip("numpy")
    pool = {"a": np.float32(0.30), "b": np.float64(0.15), "c": np.int64(1)}
    # The quoted metric must come from the same stored values — float32(0.30) widens to
    # 0.30000001..., and quoting the python literal 0.30 against it is a genuine mismatch the
    # exact comparison correctly refuses (same-precision-both-sides convention, red team D5).
    v = Experiment(mk()).score({"m": float(pool["a"]), "pool": pool, "dq": ["b", "c"]})
    assert v.verdict == "FAIL"          # scored (0.3000... > 0.2), not refused


def test_check_metrics_sees_composition_paths(mk):
    """Red team D4: the pre-run tool must catch an absent over-path before the compute."""
    e = Experiment(mk())
    out = e.check_metrics({"m": 0.25, "dq": ["b"]})     # pool missing
    assert out["G:over"]["present"] is False and out["G:over"]["usable"] is False
    ok = e.check_metrics({"m": 0.25, "pool": POOL, "dq": ["b"]})
    assert ok["G:over"]["usable"] is True and ok["G:excluding"]["usable"] is True


def test_e1_retro_case_refuses_against_committed_receipt():
    """The real defect, pinned forever: E1's G1 with a v4 declaration against e1_result.json."""
    e1 = json.loads((ROOT / "papers" / "first-afference" / "e1_result.json")
                    .read_text(encoding="utf-8"))
    import tempfile
    spec = {"gates": {"G1": {"metric": "best_median_abs_rel_error", "op": "<=", "value": 0.20,
                             "agg": "min", "over": "pooled_median_abs_rel_error",
                             "excluding": "disqualified_by_silent_probe"}},
            "outcomes": [{"when": {"G1": True}, "verdict": "USABLE"},
                         {"when": {"G1": False}, "verdict": "NOT_ESTIMABLE"}],
            "smoke_verdict": "SMOKE"}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        p = tmp / "PREREG_retro.md"
        p.write_text("# retro\n\n```gates\n" + json.dumps(spec) + "\n```\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
        subprocess.run(["git", "add", "-A"], cwd=tmp, check=True)
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                        "commit", "-qm", "retro"], cwd=tmp, check=True)
        with pytest.raises(GateSpecError, match="COMPOSITION VIOLATION"):
            Experiment(p).score(e1)
