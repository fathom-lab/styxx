"""Appendix B distance functions (styxx/v8/distances.py).

Every pinned number below is derived by hand in the comment beside it; the assertion is the
test, the comment is the derivation.  Nothing here skips.
"""
from __future__ import annotations

import math
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from styxx.v8 import distances as D
from styxx.v8.distances import exact, lens, resid, rounded, seqlp, topk

# ----------------------------------------------------------------------------- fixtures
# Three-item batteries.  Ids are chosen so that insertion order != item_id order, which is
# what makes the order-independence tests bite: "b" < "c" < "z" in byte order.


def _items(**per_id):
    """{item_id: fields} -> list of §3.1 item records, in the given (non-sorted) order."""
    return [dict(item_id=k, **v) for k, v in per_id.items()]


A_EXACT = _items(
    z={"token_ids": [1, 2, 3]},
    b={"token_ids": [5, 6]},
    c={"token_ids": [9]},
)
B_EXACT = _items(
    z={"token_ids": [1, 2, 3]},      # matches
    b={"token_ids": [5, 7]},         # differs
    c={"token_ids": [9, 9]},         # differs
)
ROLES = {"z": "item", "b": "canary", "c": "anchor"}

A_SEQLP = _items(z={"seq_logprob": -1.0}, b={"seq_logprob": -2.5}, c={"seq_logprob": -0.25})
B_SEQLP = _items(z={"seq_logprob": -1.5}, b={"seq_logprob": -2.0}, c={"seq_logprob": -0.25})

A_TOPK = _items(
    z={"topk": [{"pos": 0, "ids": [1, 2], "lps": [-0.5, -1.0]}]},
    b={"topk": [{"pos": 0, "ids": [7], "lps": [-0.1]}, {"pos": 1, "ids": [8], "lps": [-0.4]}]},
    c={"topk": [{"pos": 0, "ids": [3, 4], "lps": [-0.2, -0.3]}]},
)
B_TOPK = _items(
    z={"topk": [{"pos": 0, "ids": [1, 3], "lps": [-0.5, -2.0]}]},
    b={"topk": [{"pos": 0, "ids": [7], "lps": [-0.3]}]},              # pos 1 missing here
    c={"topk": [{"pos": 0, "ids": [4, 3], "lps": [-0.3, -0.2]}]},     # same vector, other order
)

A_RESID = {"z": [1.0, 0.0], "b": [0.5, 0.5], "c": [3.0, 1.0]}
B_RESID = {"z": [0.0, 1.0], "b": [0.5, 0.5], "c": [-1.0, 1.0]}

A_LENS = {"z": 10, "b": 14, "c": 20}
B_LENS = {"z": 12, "b": 14, "c": 15}
N_LAYERS = 26


def _shuffled(seq, seed):
    out = list(seq)
    random.Random(seed).shuffle(out)
    return out


# ----------------------------------------------------------------------------- hand-computed


def test_exact_hand_value():
    # scored items (role item|canary): z matches, b differs -> matching 1 of 2 -> 1 - 1/2 = 0.5
    # anchor c differs -> anchor_flips 1
    assert exact(A_EXACT, B_EXACT, ROLES) == (0.5, 1)


def test_exact_identical_is_zero():
    assert exact(A_EXACT, A_EXACT, ROLES) == (0.0, 0)


def test_exact_roles_from_battery_items_list():
    battery_items = [{"item_id": k, "role": v, "prompt_text": "x"} for k, v in ROLES.items()]
    assert exact(A_EXACT, B_EXACT, battery_items) == (0.5, 1)


def test_exact_by_digest_when_token_ids_absent():
    # token_ids_sha256 is sha256(JCS(ids)) (A.3); equal digests <=> equal ids
    a = _items(z={"token_ids_sha256": "a" * 64}, b={"token_ids_sha256": "b" * 64})
    b = _items(z={"token_ids_sha256": "a" * 64}, b={"token_ids_sha256": "c" * 64})
    assert exact(a, b, {"z": "item", "b": "item"}) == (0.5, 0)


def test_seqlp_hand_value():
    # |delta| = |-1.0 - -1.5| = 0.5, |-2.5 - -2.0| = 0.5, |-0.25 - -0.25| = 0
    # fsum(0.5, 0.5, 0) / 3 = 1.0 / 3 = 0.3333333333333333
    assert seqlp(A_SEQLP, B_SEQLP) == 0.3333333333333333


def test_topk_hand_value():
    # item b (earliest in id order): pos 0 both sides, ids {7}: |-0.1 - -0.3| = 0.2
    #                             pos 1 on a only -> 5 * 20 = 100, counted
    # item c: pos 0 both sides, same vector in a different order -> 0
    # item z: pos 0 both sides, union {1,2,3}:
    #          id 1: |-0.5 - -0.5| = 0
    #          id 2: |-1.0 - (-20)| = 19     (absent on b -> -20 fill)
    #          id 3: |(-20) - (-2.0)| = 18   (absent on a -> -20 fill)
    #          -> 37
    # pooled: fsum(0.2, 100, 0, 37) / 4 positions = 137.2 / 4 = 34.3
    assert topk(A_TOPK, B_TOPK) == 34.3


def test_resid_hand_value():
    # item b: p = q = [0.5, 0.5] -> JSD 0
    # item c: p = [3,1]/4 = [0.75, 0.25]; q = clamp([-1, 1]) = [0, 1] -> [0, 1]
    #          m = [0.375, 0.625]
    #          KL(p||m) = 0.75 ln(0.75/0.375) + 0.25 ln(0.25/0.625) = 0.75 ln 2 + 0.25 ln 0.4
    #          KL(q||m) = 0 (0 ln 0 = 0) + 1 ln(1/0.625) = ln 1.6
    #          JSD = 0.5 (0.75 ln 2 + 0.25 ln 0.4) + 0.5 ln 1.6 = 0.38039566584857787
    # item z: p = [1,0], q = [0,1], m = [0.5,0.5]: KL(p||m) = ln 2, KL(q||m) = ln 2 -> JSD = ln 2
    # mean = fsum(0, 0.38039566584857787, 0.6931471805599453) / 3 = 0.35784761546950766
    jsd_c = 0.5 * (0.75 * math.log(2) + 0.25 * math.log(0.4)) + 0.5 * math.log(1.6)
    assert jsd_c == 0.38039566584857787
    expected = math.fsum([0.0, jsd_c, math.log(2)]) / 3
    assert expected == 0.35784761546950766
    assert resid(A_RESID, B_RESID) == 0.35784761546950766


def test_lens_hand_value():
    # |delta| / L: b: 0/26 = 0; c: |20-15|/26 = 5/26; z: |10-12|/26 = 2/26
    # mean = fsum(0, 5/26, 2/26) / 3 = 0.08974358974358976
    # (the closed form 7/78 = 0.08974358974358974 differs in the last bit; the pinned value is
    #  the fsum over per-item terms that Appendix B prescribes, and both round to 0.08974359)
    got = lens(A_LENS, B_LENS, N_LAYERS)
    assert got == 0.08974358974358976
    assert abs(got - 7 / 78) < 1e-15
    assert rounded(got) == rounded(7 / 78) == 0.08974359


def test_rounded():
    assert D.ROUND_PLACES == 9
    assert rounded(0.1234567890123) == 0.123456789
    assert rounded(1 / 3) == 0.333333333
    assert rounded(2 / 3) == 0.666666667
    assert rounded(0.0) == 0.0
    assert rounded(5) == 5.0 and isinstance(rounded(5), float)


# ----------------------------------------------------------------------------- order independence


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_order_independence(seed):
    ra, rb = seed, seed + 100
    assert exact(_shuffled(A_EXACT, ra), _shuffled(B_EXACT, rb), ROLES) == (0.5, 1)
    assert seqlp(_shuffled(A_SEQLP, ra), _shuffled(B_SEQLP, rb)) == 0.3333333333333333
    assert topk(_shuffled(A_TOPK, ra), _shuffled(B_TOPK, rb)) == 34.3
    a_res = dict(_shuffled(list(A_RESID.items()), ra))
    b_res = dict(_shuffled(list(B_RESID.items()), rb))
    assert resid(a_res, b_res) == 0.35784761546950766
    a_l = dict(_shuffled(list(A_LENS.items()), ra))
    b_l = dict(_shuffled(list(B_LENS.items()), rb))
    assert lens(a_l, b_l, N_LAYERS) == 0.08974358974358976


def test_order_independence_topk_position_order():
    # positions listed in a different order on the two sides -> same number
    a = _items(z={"topk": [{"pos": 1, "ids": [1], "lps": [-1.0]}, {"pos": 0, "ids": [2], "lps": [-2.0]}]})
    b = _items(z={"topk": [{"pos": 0, "ids": [2], "lps": [-2.5]}, {"pos": 1, "ids": [1], "lps": [-1.25]}]})
    # pos 0: |-2.0 - -2.5| = 0.5 ; pos 1: |-1.0 - -1.25| = 0.25 ; mean = 0.75 / 2 = 0.375
    assert topk(a, b) == 0.375
    assert topk(b, a) == 0.375


# ----------------------------------------------------------------------------- topk rules


def test_topk_minus_20_fill():
    # one token on each side, different ids: |lp_a - (-20)| + |(-20) - lp_b|
    a = _items(z={"topk": [{"pos": 0, "ids": [1], "lps": [-0.5]}]})
    b = _items(z={"topk": [{"pos": 0, "ids": [2], "lps": [-0.25]}]})
    # 19.5 + 19.75 = 39.25 over 1 position
    assert topk(a, b) == 39.25
    assert D.ABSENT_LP == -20.0


def test_topk_fill_is_not_zero_fill():
    # the same case with a 0 fill would give 0.5 + 0.25 = 0.75; guard the constant's use
    a = _items(z={"topk": [{"pos": 0, "ids": [1], "lps": [-0.5]}]})
    b = _items(z={"topk": [{"pos": 0, "ids": [2], "lps": [-0.25]}]})
    assert topk(a, b) != 0.75


def test_topk_one_sided_position_counts_100():
    a = _items(z={"topk": [{"pos": 0, "ids": [1], "lps": [-0.5]}, {"pos": 1, "ids": [1], "lps": [-0.5]}]})
    b = _items(z={"topk": [{"pos": 0, "ids": [1], "lps": [-0.5]}]})
    # pos 0: 0 ; pos 1 one-sided: 100 ; mean over 2 positions = 50
    assert topk(a, b) == 50.0
    assert topk(b, a) == 50.0
    assert D.ONE_SIDED_POSITION == 100.0


def test_topk_one_sided_only_position_is_exactly_the_maximum():
    a = _items(z={"topk": [{"pos": 0, "ids": [1], "lps": [-0.5]}]})
    b = _items(z={"topk": []})
    assert topk(a, b) == 100.0


def test_topk_identical_is_zero():
    assert topk(A_TOPK, A_TOPK) == 0.0


# ----------------------------------------------------------------------------- JSD properties


def test_jsd_of_self_is_zero():
    assert resid(A_RESID, A_RESID) == 0.0
    assert resid([[0.1, 0.2, 0.7]], [[0.1, 0.2, 0.7]]) == 0.0
    # scale invariance: normalization makes [2, 4, 14] the same profile as [0.1, 0.2, 0.7]
    assert resid([[0.1, 0.2, 0.7]], [[2.0, 4.0, 14.0]]) == 0.0


def test_jsd_symmetric():
    assert resid(A_RESID, B_RESID) == resid(B_RESID, A_RESID)
    p, q = [[0.9, 0.05, 0.05]], [[0.2, 0.3, 0.5]]
    assert resid(p, q) == resid(q, p)


def test_jsd_disjoint_support_is_ln2():
    assert resid([[1.0, 0.0]], [[0.0, 1.0]]) == math.log(2)


def test_jsd_bounded_by_ln2_and_nonnegative():
    got = resid([[0.6, 0.4, 0.0]], [[0.0, 0.1, 0.9]])
    assert 0.0 <= got <= math.log(2)


def test_resid_clamps_negatives_before_normalizing():
    # [-5, 1, 1] -> [0, 1, 1] -> [0, .5, .5], same as [0, 3, 3]
    assert resid([[-5.0, 1.0, 1.0]], [[0.0, 3.0, 3.0]]) == 0.0


def test_resid_positional_and_keyed_agree():
    keyed = resid(A_RESID, B_RESID)
    order = sorted(A_RESID)   # A.3 order
    positional = resid([A_RESID[k] for k in order], [B_RESID[k] for k in order])
    assert keyed == positional


# ----------------------------------------------------------------------------- fsum vs naive


def test_fsum_not_naive_sum():
    # |delta| = 0.1, 0.2, 0.3 exactly (each is |-x - 0| for the double nearest x)
    # naive: (0.1 + 0.2) + 0.3 = 0.6000000000000001 ; fsum: 0.6
    a = _items(b={"seq_logprob": -0.1}, c={"seq_logprob": -0.2}, z={"seq_logprob": -0.3})
    b = _items(b={"seq_logprob": 0.0}, c={"seq_logprob": 0.0}, z={"seq_logprob": 0.0})
    naive = (0.1 + 0.2 + 0.3) / 3
    exact_sum = math.fsum([0.1, 0.2, 0.3]) / 3
    assert naive != exact_sum                       # the case really differs in the last bit
    assert naive == 0.20000000000000004
    assert exact_sum == 0.19999999999999998
    assert seqlp(a, b) == exact_sum
    assert seqlp(a, b) != naive


def test_fsum_not_naive_sum_topk():
    # three positions on one item, each with a one-token L1 of 0.1, 0.2, 0.3
    a = _items(z={"topk": [{"pos": p, "ids": [1], "lps": [-x]} for p, x in enumerate([0.1, 0.2, 0.3])]})
    b = _items(z={"topk": [{"pos": p, "ids": [1], "lps": [0.0]} for p in range(3)]})
    assert topk(a, b) == 0.19999999999999998
    assert topk(a, b) != (0.1 + 0.2 + 0.3) / 3


# ----------------------------------------------------------------------------- refusals


def test_missing_item_on_either_side_raises():
    short = A_SEQLP[:2]
    with pytest.raises(ValueError, match="not aligned"):
        seqlp(short, B_SEQLP)
    with pytest.raises(ValueError, match="not aligned"):
        seqlp(A_SEQLP, short)
    with pytest.raises(ValueError, match="not aligned"):
        exact(A_EXACT[:2], B_EXACT, ROLES)
    with pytest.raises(ValueError, match="not aligned"):
        topk(A_TOPK, B_TOPK[1:])
    with pytest.raises(ValueError, match="not aligned"):
        resid(A_RESID, {k: v for k, v in B_RESID.items() if k != "z"})
    with pytest.raises(ValueError, match="not aligned"):
        lens({k: v for k, v in A_LENS.items() if k != "b"}, B_LENS, N_LAYERS)


def test_exact_missing_role_or_battery_item_raises():
    with pytest.raises(ValueError, match="without a role"):
        exact(A_EXACT, B_EXACT, {"z": "item", "b": "canary"})
    with pytest.raises(ValueError, match="missing from both sides"):
        exact(A_EXACT, B_EXACT, {**ROLES, "q": "item"})
    with pytest.raises(ValueError, match="unknown role"):
        exact(A_EXACT, B_EXACT, {**ROLES, "c": "sentinel"})


def test_exact_anchors_only_raises():
    with pytest.raises(ValueError, match="undefined"):
        exact(A_EXACT, B_EXACT, {"z": "anchor", "b": "anchor", "c": "anchor"})


def test_duplicate_item_id_raises():
    with pytest.raises(ValueError, match="duplicate"):
        seqlp(A_SEQLP + [A_SEQLP[0]], B_SEQLP + [B_SEQLP[0]])


def test_absent_channel_raises_not_zero():
    a = _items(z={"seq_logprob": None})
    b = _items(z={"seq_logprob": -1.0})
    with pytest.raises(ValueError, match="absent"):
        seqlp(a, b)
    a = _items(z={"topk": None})
    b = _items(z={"topk": []})
    with pytest.raises(ValueError, match="absent"):
        topk(a, b)


def test_empty_battery_raises():
    with pytest.raises(ValueError, match="undefined"):
        seqlp([], [])
    with pytest.raises(ValueError, match="undefined"):
        resid([], [])
    with pytest.raises(ValueError, match="undefined"):
        lens([], [], 4)
    # a battery whose items generated no positions has no topk mean
    with pytest.raises(ValueError, match="undefined"):
        topk(_items(z={"topk": []}), _items(z={"topk": []}))


def test_resid_length_mismatch_raises():
    with pytest.raises(ValueError, match="length differs"):
        resid([[0.5, 0.5]], [[0.2, 0.3, 0.5]])
    with pytest.raises(ValueError, match="item count differs"):
        resid([[0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError, match="cannot normalize"):
        resid([[0.0, -1.0]], [[0.5, 0.5]])


def test_lens_bad_n_layers_raises():
    with pytest.raises(ValueError, match="n_layers"):
        lens(A_LENS, B_LENS, 0)
    with pytest.raises(ValueError, match="n_layers"):
        lens(A_LENS, B_LENS, -3)
    with pytest.raises(ValueError, match="item count differs"):
        lens([1, 2], [1], 4)


def test_non_finite_raises():
    with pytest.raises(ValueError, match="not finite"):
        seqlp(_items(z={"seq_logprob": float("nan")}), _items(z={"seq_logprob": -1.0}))
    with pytest.raises(ValueError, match="not finite"):
        resid([[float("inf"), 1.0]], [[0.5, 0.5]])


def test_topk_malformed_entries_raise():
    with pytest.raises(ValueError, match="lengths differ"):
        topk(_items(z={"topk": [{"pos": 0, "ids": [1, 2], "lps": [-1.0]}]}),
             _items(z={"topk": []}))
    with pytest.raises(ValueError, match="duplicate pos"):
        topk(_items(z={"topk": [{"pos": 0, "ids": [1], "lps": [-1.0]}, {"pos": 0, "ids": [1], "lps": [-1.0]}]}),
             _items(z={"topk": []}))


# ----------------------------------------------------------------------------- hypothesis

_ids = st.lists(st.text(min_size=1, max_size=8), min_size=1, max_size=6, unique=True)
_lp = st.floats(min_value=-1e3, max_value=0.0, allow_nan=False, allow_infinity=False)
_pos_entry = st.builds(
    lambda pos, pairs: {"pos": pos, "ids": [p[0] for p in pairs], "lps": [p[1] for p in pairs]},
    st.integers(min_value=0, max_value=7),
    st.lists(st.tuples(st.integers(min_value=0, max_value=50), _lp), max_size=5,
             unique_by=lambda p: p[0]),
)
_topk_list = st.lists(_pos_entry, max_size=4, unique_by=lambda e: e["pos"])
_profile_val = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)


@st.composite
def _seqlp_pair(draw):
    ids = draw(_ids)
    a = [{"item_id": i, "seq_logprob": draw(_lp)} for i in ids]
    b = [{"item_id": i, "seq_logprob": draw(_lp)} for i in ids]
    return a, b


@st.composite
def _topk_pair(draw):
    ids = draw(_ids)
    a = [{"item_id": i, "topk": draw(_topk_list)} for i in ids]
    b = [{"item_id": i, "topk": draw(_topk_list)} for i in ids]
    return a, b


@st.composite
def _resid_pair(draw):
    ids = draw(_ids)
    n_layers = draw(st.integers(min_value=1, max_value=6))
    vec = st.lists(_profile_val, min_size=n_layers, max_size=n_layers).filter(
        lambda v: any(x > 0 for x in v))
    a = {i: draw(vec) for i in ids}
    b = {i: draw(vec) for i in ids}
    return a, b


@st.composite
def _lens_pair(draw):
    ids = draw(_ids)
    n_layers = draw(st.integers(min_value=1, max_value=80))
    layer = st.integers(min_value=0, max_value=n_layers)
    a = {i: draw(layer) for i in ids}
    b = {i: draw(layer) for i in ids}
    return a, b, n_layers


@st.composite
def _exact_pair(draw):
    ids = draw(_ids)
    tok = st.lists(st.integers(min_value=0, max_value=9), max_size=4)
    a = [{"item_id": i, "token_ids": draw(tok)} for i in ids]
    b = [{"item_id": i, "token_ids": draw(tok)} for i in ids]
    roles = {i: draw(st.sampled_from(["item", "canary", "anchor"])) for i in ids}
    roles[ids[0]] = "item"   # at least one scored item
    return a, b, roles


@settings(max_examples=200, deadline=None)
@given(_exact_pair())
def test_hyp_exact_nonneg_symmetric(pair):
    a, b, roles = pair
    d, flips = exact(a, b, roles)
    assert 0.0 <= d <= 1.0 and flips >= 0
    assert exact(b, a, roles) == (d, flips)
    assert exact(a, a, roles) == (0.0, 0)


@settings(max_examples=200, deadline=None)
@given(_seqlp_pair())
def test_hyp_seqlp_nonneg_symmetric(pair):
    a, b = pair
    d = seqlp(a, b)
    assert d >= 0.0
    assert seqlp(b, a) == d
    assert seqlp(a, a) == 0.0


@settings(max_examples=200, deadline=None)
@given(_topk_pair())
def test_hyp_topk_nonneg_symmetric(pair):
    a, b = pair
    try:
        d = topk(a, b)
    except ValueError as e:
        # only the no-positions-at-all case is allowed to refuse
        assert "undefined" in str(e)
        assert all(not r["topk"] for r in a) and all(not r["topk"] for r in b)
        return
    assert d >= 0.0
    assert topk(b, a) == d
    if any(r["topk"] for r in a):
        assert topk(a, a) == 0.0


@settings(max_examples=200, deadline=None)
@given(_resid_pair())
def test_hyp_resid_nonneg_symmetric_bounded(pair):
    a, b = pair
    d = resid(a, b)
    assert 0.0 <= d <= math.log(2) + 1e-12
    assert resid(b, a) == d
    assert resid(a, a) == 0.0


@settings(max_examples=200, deadline=None)
@given(_lens_pair())
def test_hyp_lens_nonneg_symmetric_bounded(pair):
    a, b, n = pair
    d = lens(a, b, n)
    assert 0.0 <= d <= 1.0
    assert lens(b, a, n) == d
    assert lens(a, a, n) == 0.0


@settings(max_examples=100, deadline=None)
@given(_seqlp_pair(), st.integers(min_value=0, max_value=2**31))
def test_hyp_order_independence(pair, seed):
    a, b = pair
    assert seqlp(_shuffled(a, seed), _shuffled(b, seed + 1)) == seqlp(a, b)
