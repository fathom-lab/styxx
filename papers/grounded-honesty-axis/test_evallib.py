"""Unit tests for _evallib — the statistics every FINDING depends on must be PROVABLY correct.
A silent bug in spearman/auc would falsify conclusions invisibly, so this is a GATE, not polish.
pytest-discoverable (test_* functions) AND runnable standalone (`python test_evallib.py`).
"""
from _evallib import (spearman, perm_p, auc_pos_gt_neg, brier, reliability_bins,
                      normalize_answer, alias_match, params_for)


def approx(a, b, t=1e-9):
    return a is not None and abs(a - b) <= t


def test_spearman():
    assert approx(spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), 1.0)
    assert approx(spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]), -1.0)
    assert spearman([1, 2], [1, 2]) is None                  # n < 3
    assert spearman([1, 1, 1], [1, 2, 3]) is None            # constant side
    assert approx(spearman([1, 2, 3, 4], [1, 3, 2, 4]), 0.8)  # one swap, known value


def test_perm_p():
    assert approx(perm_p([1, 2, 3, 4], [1, 2, 3, 4]), 2.0 / 24.0)  # only identity+reverse hit |rho|>=1
    assert perm_p(list(range(11)), list(range(11))) is None        # n > 9 guard


def test_auc():
    assert approx(auc_pos_gt_neg([1, 2, 3], [0, 0, 0]), 1.0)
    assert approx(auc_pos_gt_neg([0, 0], [1, 1]), 0.0)
    assert approx(auc_pos_gt_neg([1, 2], [1, 2]), 0.5)       # ties -> 0.5 by symmetry
    assert auc_pos_gt_neg([], [1]) is None


def test_brier():
    assert approx(brier([1.0, 0.0], [True, False]), 0.0)
    assert approx(brier([0.0, 1.0], [True, False]), 1.0)
    assert approx(brier([0.5, 0.5], [True, False]), 0.25)


def test_reliability():
    rb = reliability_bins([0.9, 0.85, 0.4], [True, False, True])
    hi = [b for b in rb if b["band"].startswith("[0.8")][0]
    assert hi["n"] == 2 and approx(hi["accuracy"], 0.5)


def test_normalize():
    assert normalize_answer("The 48 Hrs.!") == "48 hrs"
    assert normalize_answer("A Boojum") == "boojum"
    assert normalize_answer("  Republic   of  Niger ") == "republic of niger"


def test_alias_match():
    assert alias_match("Pink Floyd", ["the pink floyd", "pink floyd"])
    assert alias_match("the answer is Paris", ["paris"])      # alias inside a verbose prediction
    assert alias_match("Niger", ["Republic of Niger", "Niger"])
    assert alias_match("Highway 61", ["61", "sixty-one"])     # alias is a token-run of the prediction
    assert not alias_match("India", ["in"])                   # NO char-substring false positive
    assert not alias_match("", ["x"])
    # documented HONEST limitation: do NOT invent abbreviation equivalences (hrs != hours).
    # self-audit #33: "48 Hrs." is semantically Eddie Murphy's debut but the dataset alias is
    # "48 Hours"; exact-match cannot credit it, and forcing it would create false positives.
    assert not alias_match("48 Hrs.", ["48 Hours", "48 time"])


def test_params():
    assert params_for("Qwen/Qwen2.5-7B-Instruct") == 7.0
    assert params_for("meta-llama/Llama-3.2-1B-Instruct") == 1.24
    assert params_for("mystery-model") is None


if __name__ == "__main__":
    for _name, _fn in sorted(list(globals().items())):
        if _name.startswith("test_") and callable(_fn):
            _fn()
    print("all _evallib tests passed")
