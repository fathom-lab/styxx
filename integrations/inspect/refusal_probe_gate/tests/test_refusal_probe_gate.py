# -*- coding: utf-8 -*-
"""Offline tests for the refusal-probe-gate Inspect eval (no GPU, no model)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from refusal_probe_gate import (  # noqa: E402
    auc_score,
    operating_point,
    record_to_sample,
    refusal_probe_gate,
)


def test_auc_perfect_separation():
    # positives all out-rank negatives -> AUC 1.0
    assert auc_score([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0


def test_auc_inverted():
    # positives all below negatives -> AUC 0.0
    assert auc_score([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0


def test_auc_chance_with_ties():
    # all identical scores -> 0.5 (Mann-Whitney with average ranks)
    assert abs(auc_score([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) - 0.5) < 1e-9


def test_auc_known_value():
    # one mis-ordered pair out of 4 (pos 0.4 < neg 0.5) -> AUC 0.75
    assert abs(auc_score([1, 1, 0, 0], [0.9, 0.4, 0.5, 0.2]) - 0.75) < 1e-9


def test_auc_undefined_single_class():
    import math
    assert math.isnan(auc_score([1, 1, 1], [0.9, 0.8, 0.7]))


def test_operating_point_counts():
    # thr 0.5: scores>=.5 predicted refuse
    op = operating_point([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1], 0.5)
    # tp: .9(y1) ; fn: .4(y1) ; fp: .6(y0) ; tn: .1(y0)
    assert abs(op["precision"] - 0.5) < 1e-9      # tp/(tp+fp)=1/2
    assert abs(op["recall"] - 0.5) < 1e-9         # tp/(tp+fn)=1/2
    assert abs(op["fpr"] - 0.5) < 1e-9            # fp/(fp+tn)=1/2
    assert abs(op["accuracy_at_thr"] - 0.5) < 1e-9


def test_record_to_sample():
    s = record_to_sample({"prompt": "x", "refused": "1", "probe_score": "0.7"})
    assert s.target == "refuse"
    assert s.metadata["probe_score"] == 0.7
    s0 = record_to_sample({"prompt": "y", "refused": "0", "probe_score": "0.1"})
    assert s0.target == "allow"


def test_task_constructs_from_sample_csv():
    t = refusal_probe_gate()                       # default bundled sample
    n = len(t.dataset)
    assert n == 24
    # the sample set is designed to be strongly but not perfectly separable
    ys = [1 if s.target == "refuse" else 0 for s in t.dataset]
    ps = [s.metadata["probe_score"] for s in t.dataset]
    a = auc_score(ys, ps)
    assert 0.85 < a < 1.0, a


if __name__ == "__main__":
    import math
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  ok  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")
