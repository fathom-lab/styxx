# -*- coding: utf-8 -*-
"""Unit tests for styxx.gauntlet — the public-challenge runner (7.7.5).

Covers:
  - benchmark loading (default path resolves)
  - method spec resolution ("module:attr")
  - classification gauntlet on a known-failing baseline (majority class)
  - detection gauntlet on a known-failing baseline (constant zero → AUC 0.5)
  - the "perfect oracle" upper bound: an oracle that reads true labels passes everything
  - error handling: bad spec, non-callable, bad return shape
  - the F1 and AUC math primitives in isolation
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from styxx.gauntlet import (
    BASELINE_ENTRY,
    GauntletResult,
    Submission,
    compute_auc,
    compute_f1,
    load_benchmark,
    resolve_method,
    run_classification_gauntlet,
    run_detection_gauntlet,
    _majority_baseline_predict,
    _zero_baseline_detect,
)


# ──────────────────────────────────────────────────────────────────────
# Metric math
# ──────────────────────────────────────────────────────────────────────

def test_f1_perfect():
    assert compute_f1([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0


def test_f1_zero_when_no_positives_predicted():
    assert compute_f1([1, 1, 0, 0], [0, 0, 0, 0]) == 0.0


def test_f1_half():
    # 1 TP, 1 FP, 1 FN → P=0.5, R=0.5, F1=0.5
    f1 = compute_f1([1, 1, 0, 0], [1, 0, 1, 0])
    assert abs(f1 - 0.5) < 1e-9


def test_auc_perfect_separation():
    assert compute_auc([1.0, 2.0, 3.0], [0.0, 0.5]) == 1.0


def test_auc_constant_scores_returns_half():
    # AUC of identical constant scores = 0.5 (tie-broken equally)
    assert compute_auc([0.0, 0.0], [0.0, 0.0]) == 0.5


def test_auc_handles_empty_input():
    assert math.isnan(compute_auc([], [1.0, 2.0]))


# ──────────────────────────────────────────────────────────────────────
# Benchmark loading + method resolution
# ──────────────────────────────────────────────────────────────────────

def test_load_benchmark_default_path_resolves():
    b = load_benchmark()
    assert "records" in b and len(b["records"]) > 0
    assert "class_distribution" in b
    # at minimum the four classes are present in the distribution
    dist = b["class_distribution"]
    assert all(c in dist for c in ("folklore", "pseudoscience", "factual-error", "truth"))


def test_load_benchmark_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        load_benchmark(Path("/nonexistent/benchmark.json"))


def test_load_benchmark_package_data_exists():
    """Regression test for the 7.7.5 → 7.7.6 bundling fix.

    The benchmark must be available from the package's _data dir, not just
    from the source-tree papers/ path. Without this, `pip install styxx`
    users get FileNotFoundError when running `styxx gauntlet`."""
    pkg_data = Path(__file__).resolve().parent.parent / "styxx" / "_data" / "darkcore_benchmark_2026_05_27.json"
    assert pkg_data.exists(), (
        "package-data benchmark missing — clean pip install would fail. "
        "Run: cp papers/consensus-hallucination/darkcore_benchmark_2026_05_27.json "
        "styxx/_data/ to fix."
    )
    # And the loader must successfully resolve to it.
    b = load_benchmark()
    assert b.get("name") == "darkcore_benchmark"
    assert b.get("n_records") > 0


def test_resolve_method_loads_callable():
    fn = resolve_method("styxx.gauntlet:_majority_baseline_predict")
    assert fn is _majority_baseline_predict


def test_resolve_method_bad_spec_raises():
    with pytest.raises(ValueError):
        resolve_method("no_colon_here")


def test_resolve_method_missing_attr_raises():
    with pytest.raises(AttributeError):
        resolve_method("styxx.gauntlet:nope_doesnt_exist")


# ──────────────────────────────────────────────────────────────────────
# Classification gauntlet — failing baselines
# ──────────────────────────────────────────────────────────────────────

def _bench():
    return load_benchmark()


def test_majority_baseline_fails_all_classification_bars():
    sub = Submission(name="majority", method=_majority_baseline_predict, task="classification")
    r = run_classification_gauntlet(sub, _bench())
    assert r.error is None
    assert r.overall_pass is False
    assert r.n_passed == 0
    # All three bars must fail by construction (folklore F1 = 0 since "truth" only).
    assert r.bar_results["K1_folklore_F1"] is False
    assert r.bar_results["K2_accuracy"] is False
    assert r.bar_results["K3_crosscorpus_F1"] is False
    assert r.metrics["folklore_F1_indist"] == 0.0
    assert r.metrics["folklore_F1_crosscorpus"] == 0.0


def test_classification_bad_return_shape():
    def broken(question: str):
        return "not a dict"
    sub = Submission(name="broken", method=broken, task="classification")
    r = run_classification_gauntlet(sub, _bench())
    assert r.error is not None
    assert "dict" in r.error.lower()


def test_classification_method_raises():
    def boom(question: str):
        raise RuntimeError("oops")
    sub = Submission(name="boom", method=boom, task="classification")
    r = run_classification_gauntlet(sub, _bench())
    assert r.error is not None
    assert "RuntimeError" in r.error
    assert "oops" in r.error


# ──────────────────────────────────────────────────────────────────────
# Classification gauntlet — perfect oracle (the upper bound)
# ──────────────────────────────────────────────────────────────────────

def test_perfect_oracle_passes_all_classification_bars():
    """An oracle that reads the labels directly must pass every bar.
    This validates that the bars are actually beatable when given perfect signal."""
    bench = _bench()
    label_lookup = {r["id"]: r["class"] for r in bench["records"]}

    def oracle_predict(question: str) -> dict:
        # look up by question text since cmd_gauntlet only passes the question
        for rec in bench["records"]:
            if rec["question"] == question:
                return {"class": rec["class"]}
        return {"class": "truth"}

    sub = Submission(name="oracle", method=oracle_predict, task="classification")
    r = run_classification_gauntlet(sub, bench)
    assert r.error is None
    assert r.overall_pass is True
    assert r.n_passed == 3
    assert r.metrics["accuracy"] == 1.0


# ──────────────────────────────────────────────────────────────────────
# Detection gauntlet — failing + passing baselines
# ──────────────────────────────────────────────────────────────────────

def test_zero_detector_fails_all_detection_bars():
    """7.7.8: detection task now has 3 bars (D1, D2, D3 length-control)."""
    sub = Submission(name="zero-detector", method=_zero_baseline_detect, task="detection")
    r = run_detection_gauntlet(sub, _bench())
    assert r.error is None
    assert r.overall_pass is False
    assert r.n_passed == 0
    assert r.n_total_bars == 3
    # constant scores → AUC = 0.5
    assert abs(r.metrics["D1_misconception_AUC"] - 0.5) < 1e-9
    assert abs(r.metrics["D2_folklore_AUC"] - 0.5) < 1e-9


def test_length_oracle_passes_D1_D2_but_fails_D3():
    """The D3 length-control bar: a detector whose score IS length must fail D3
    (delta = 0 by construction). Regression test against re-introducing the
    Baseline-007 benchmark-artifact exploit."""
    from styxx.gauntlet import _length_oracle_detect
    sub = Submission(name="length-oracle", method=_length_oracle_detect, task="detection")
    r = run_detection_gauntlet(sub, _bench())
    assert r.error is None
    # D1 + D2 may pass because length correlates with class, but D3 must fail
    assert r.bar_results["D3_length_control_delta"] is False
    assert r.overall_pass is False
    # And the delta should be exactly 0 (or very close) because the detector IS the length oracle
    assert abs(r.metrics["D1_minus_length_AUC"]) < 1e-6
    assert abs(r.metrics["D2_minus_length_AUC"]) < 1e-6


def test_perfect_oracle_passes_all_detection_bars():
    """A perfect detector (reads true label) must pass all 3 bars, including
    D3 length-control (since it gets perfect AUC while the length oracle
    is below 1.0, the delta is comfortably above 0.10)."""
    bench = _bench()
    def oracle_detect(question: str, response: str) -> dict:
        for rec in bench["records"]:
            if rec["question"] == question:
                return {"score": 0.0 if rec["class"] == "truth" else 1.0}
        return {"score": 0.5}

    sub = Submission(name="oracle-det", method=oracle_detect, task="detection")
    r = run_detection_gauntlet(sub, bench)
    assert r.error is None
    assert r.overall_pass is True
    assert r.n_passed == 3  # 7.7.8: 3 bars now (D1, D2, D3)
    assert r.metrics["D1_misconception_AUC"] >= 0.99
    assert r.metrics["D2_folklore_AUC"] >= 0.99
    # D3 must pass: perfect AUC (1.0) minus length-oracle AUC (~0.79) >> 0.10
    assert r.bar_results["D3_length_control_delta"] is True


# ──────────────────────────────────────────────────────────────────────
# Result serialization + baseline entry
# ──────────────────────────────────────────────────────────────────────

def test_result_serializes_to_dict():
    sub = Submission(name="majority", method=_majority_baseline_predict, task="classification")
    r = run_classification_gauntlet(sub, _bench())
    d = r.as_dict()
    # round-trip via JSON to ensure serializability
    s = json.dumps(d, default=str)
    d2 = json.loads(s)
    assert d2["submission_name"] == "majority"
    assert d2["task"] == "classification"
    assert isinstance(d2["bar_results"], dict)


def test_baseline_entry_has_required_keys():
    for k in ("rank", "submitter", "method", "tasks", "n_bars_passed", "paper"):
        assert k in BASELINE_ENTRY


# ──────────────────────────────────────────────────────────────────────
# End-to-end via the CLI
# ──────────────────────────────────────────────────────────────────────

def test_cli_gauntlet_runs_via_module():
    proc = subprocess.run(
        [sys.executable, "-m", "styxx", "gauntlet",
         "--method", "styxx.gauntlet:_majority_baseline_predict",
         "--task", "classification",
         "--name", "test-majority",
         "--format", "json"],
        capture_output=True, text=True, timeout=60,
    )
    # Exit code 2 = ran successfully but didn't pass bars (the expected outcome for a failing baseline)
    assert proc.returncode in (0, 2), f"unexpected returncode {proc.returncode}; stderr: {proc.stderr}"
    data = json.loads(proc.stdout)
    assert data["submission_name"] == "test-majority"
    assert data["task"] == "classification"
    assert data["overall_pass"] is False
    assert data["n_passed"] == 0
