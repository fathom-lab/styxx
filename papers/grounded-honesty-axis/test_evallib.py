"""Unit tests for _evallib — the statistics every FINDING depends on must be PROVABLY correct.
A silent bug in spearman/auc would falsify conclusions invisibly, so this is not optional polish.
Run: python test_evallib.py   (raises AssertionError on any failure; prints OK otherwise).
"""
from _evallib import (spearman, perm_p, auc_pos_gt_neg, brier, reliability_bins,
                      normalize_answer, alias_match, params_for)


def approx(a, b, t=1e-9):
    return a is not None and abs(a - b) <= t


# ---- spearman ----
assert approx(spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), 1.0)
assert approx(spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]), -1.0)
assert spearman([1, 2], [1, 2]) is None                       # n < 3
assert spearman([1, 1, 1], [1, 2, 3]) is None                 # constant side -> None
assert approx(spearman([1, 2, 3, 4], [1, 3, 2, 4]), 0.8)      # one swap, known value

# ---- perm_p ----
p = perm_p([1, 2, 3, 4], [1, 2, 3, 4])                        # only identity+reverse hit |rho|>=1
assert approx(p, 2.0 / 24.0)
assert perm_p(list(range(11)), list(range(11))) is None      # n > 9 guard

# ---- auc ----
assert approx(auc_pos_gt_neg([1, 2, 3], [0, 0, 0]), 1.0)
assert approx(auc_pos_gt_neg([0, 0], [1, 1]), 0.0)
assert approx(auc_pos_gt_neg([1, 2], [1, 2]), 0.5)           # ties -> 0.5 by symmetry
assert auc_pos_gt_neg([], [1]) is None

# ---- brier ----
assert approx(brier([1.0, 0.0], [True, False]), 0.0)
assert approx(brier([0.0, 1.0], [True, False]), 1.0)
assert approx(brier([0.5, 0.5], [True, False]), 0.25)

# ---- reliability ----
rb = reliability_bins([0.9, 0.85, 0.4], [True, False, True])
hi = [b for b in rb if b["band"].startswith("[0.8")][0]
assert hi["n"] == 2 and approx(hi["accuracy"], 0.5)

# ---- normalize ----
assert normalize_answer("The 48 Hrs.!") == "48 hrs"
assert normalize_answer("A Boojum") == "boojum"
assert normalize_answer("  Republic   of  Niger ") == "republic of niger"

# ---- alias_match: principled token-level ----
assert alias_match("Pink Floyd", ["the pink floyd", "pink floyd"])
assert alias_match("the answer is Paris", ["paris"])          # alias inside a verbose prediction
assert alias_match("Niger", ["Republic of Niger", "Niger"])   # exact alias
assert alias_match("Highway 61", ["61", "sixty-one"])         # alias is a token-run of the prediction
assert not alias_match("India", ["in"])                       # NO char-substring false positive
assert not alias_match("", ["x"])
# documented HONEST limitation: we do NOT invent abbreviation equivalences (hrs != hours).
# #33 of the self-audit: "48 Hrs." is semantically Eddie Murphy's debut but the dataset alias is
# "48 Hours" — exact-match cannot credit it, and forcing it would create false positives elsewhere.
assert not alias_match("48 Hrs.", ["48 Hours", "48 time"])

# ---- params ----
assert params_for("Qwen/Qwen2.5-7B-Instruct") == 7.0
assert params_for("meta-llama/Llama-3.2-1B-Instruct") == 1.24
assert params_for("mystery-model") is None

print("all _evallib tests passed")
