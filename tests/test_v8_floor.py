"""tests for styxx.v8.floor — the noise floor and the drift decision (spec v0.2 §5, §5.2, §6).

Contract: styxx/v8/INTERFACES_layer2.md section 4.

1. Every row of the §5.2 table, plus the GATED S5-02 row (`exceeds_floor`).
2. alpha_single for R=5 (10 pairs → 1/11) and R=8 (28 pairs → 1/29).
3. The overall precedence matrix, exhaustively over pairs and triples of channel verdicts.
4. Comparisons use `rounded` (9 places) on both sides.
5. pairwise/floors: hand-computed floors on exact and seqlp, run-order independence,
   absent channels → None, R < 2 → ValueError.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math

import pytest

from styxx.v8 import floor as F
from styxx.v8.floor import CHANNEL_VERDICTS, decide, floors, overall, pairwise

EXIT = F.EXIT
ROUND_PLACES = F.ROUND_PLACES
CHANNELS = F.CHANNELS


# --------------------------------------------------------------------------- helpers
def _sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest()


def _item(iid, ids, lp=None, topk=None):
    it = {"item_id": iid, "token_ids": list(ids), "token_ids_sha256": _sha(list(ids)), "n_generated": len(ids)}
    it["seq_logprob"] = lp
    it["topk"] = topk
    return it


def _body(items, *, seqlp=True, topk=False, resid=None, lens=None, run_index=0):
    channels = {"exact": {"hash": hashlib.sha256(b"".join(bytes.fromhex(i["token_ids_sha256"]) for i in sorted(items, key=lambda i: i["item_id"]))).hexdigest()}}
    channels["seqlp"] = {"present": bool(seqlp)}
    channels["topk"] = {"present": bool(topk)}
    if resid is not None:
        channels["resid"] = resid
    if lens is not None:
        channels["lens"] = lens
    return {"run_index": run_index, "nuisance": {"batch_size": 1}, "items": items, "channels": channels}


# Four items; every run below changes some subset of token ids and every seq_logprob.
IDS = {"a": [1, 2, 3], "b": [4, 5], "c": [6, 7, 8, 9], "d": [10]}


def _run(changed: set[str], lps: dict[str, float], **kw):
    items = []
    for iid in sorted(IDS):
        ids = IDS[iid] + ([99] if iid in changed else [])
        items.append(_item(iid, ids, lps[iid]))
    return _body(items, **kw)


def _same_model_runs(n):
    """n runs of one body (distance 0 everywhere) — the batch-1 finding of the probe."""
    base = {"a": -1.0, "b": -2.0, "c": -3.0, "d": -4.0}
    return [_run(set(), base, run_index=i) for i in range(n)]


# =========================================================================== 1. the §5.2 table
class TestDecideTable:
    def test_no_floor_is_inconclusive(self):
        assert decide(0.0, None, None) == "inconclusive"
        # floor None wins over everything else the caller says
        assert decide(0.9, None, 0.9, covered=False, skew=True) == "inconclusive"

    def test_outside_coverage(self):
        assert decide(0.0, 0.1, None, covered=False) == "beyond-floor-coverage"
        assert decide(0.5, 0.1, 0.5, covered=False, skew=True) == "beyond-floor-coverage"

    def test_same_at_or_below_floor(self):
        assert decide(0.05, 0.1, None) == "same"
        assert decide(0.1, 0.1, None) == "same"  # ≤, not <
        assert decide(0.0, 0.0, None) == "same"  # the batch-1 floor of the probe
        # skew and a confirmation are irrelevant when d ≤ floor
        assert decide(0.1, 0.1, 0.9, skew=True) == "same"

    def test_transient(self):
        assert decide(0.2, 0.1, 0.1) == "same (transient)"
        assert decide(0.2, 0.1, 0.0) == "same (transient)"

    def test_drift_needs_both_runs_over_the_floor(self):
        assert decide(0.2, 0.1, 0.2) == "drift"
        assert decide(0.2, 0.1, 0.100000001) == "drift"

    def test_skew_when_instrument_differs_and_d_exceeds(self):
        assert decide(0.2, 0.1, None, skew=True) == "skew"
        # skew is checked before drift: a confirmed exceedance under skew is still skew
        assert decide(0.2, 0.1, 0.2, skew=True) == "skew"
        assert decide(0.2, 0.1, 0.0, skew=True) == "skew"

    def test_exceeds_floor_without_confirmation_gated_s5_02(self):
        # GATED S5-02 recommendation A: --diff cannot confirm, so it never says drift
        assert decide(0.2, 0.1, None) == "exceeds_floor"
        assert decide(1.0, 0.0, None) == "exceeds_floor"

    def test_identity_is_a_subject_verdict_not_a_channel_verdict(self):
        v, code = overall({"exact": "same"}, identity_diff=["weights_sha256"], skipped=[], sensitivity_present=True)
        assert v == "identity (weights_sha256)"
        assert code == EXIT["identity"] == 1
        v, code = overall({}, identity_diff=["weights_sha256", "tokenizer_sha256"], skipped=[], sensitivity_present=False)
        assert v == "identity (weights_sha256, tokenizer_sha256)"
        assert code == 1

    def test_exit_codes_per_row(self):
        rows = {
            "inconclusive": 2,
            "beyond-floor-coverage": 2,
            "same": 0,
            "same (transient)": 0,
            "drift": 1,
            "skew": 2,
            "exceeds_floor": 2,
        }
        for v, code in rows.items():
            assert overall({"exact": v}, identity_diff=[], skipped=[], sensitivity_present=True)[1] == code, v

    def test_distance_none_is_inconclusive(self):
        # a channel that could not be measured (lens with differing n_layers) says nothing
        assert decide(None, 0.1, None) == "inconclusive"

    def test_refusals(self):
        with pytest.raises(ValueError):
            decide(float("nan"), 0.1, None)
        with pytest.raises(ValueError):
            decide(0.2, float("inf"), None)
        with pytest.raises(TypeError):
            decide("0.2", 0.1, None)  # type: ignore[arg-type]


# =========================================================================== 4. rounding
class TestRounding:
    def test_comparisons_use_rounded(self):
        assert ROUND_PLACES == 9
        floor = 0.1
        # 4e-10 above the floor rounds back onto it → same
        assert decide(floor + 4e-10, floor, None) == "same"
        # 6e-10 above the floor rounds to 0.100000001 → over
        assert decide(floor + 6e-10, floor, None) == "exceeds_floor"
        # the floor is rounded too: a floor 4e-10 above d does not hide a d that rounds equal
        assert decide(0.100000001, 0.100000001 + 4e-10, None) == "same"
        # the confirmation run is rounded the same way
        assert decide(0.2, floor, floor + 4e-10) == "same (transient)"
        assert decide(0.2, floor, floor + 6e-10) == "drift"

    def test_pairwise_distances_are_rounded(self):
        lps0 = {"a": -1.0, "b": -2.0, "c": -3.0, "d": -4.0}
        lps1 = {k: v + 1.23456789012345e-3 for k, v in lps0.items()}
        blk = pairwise([_run(set(), lps0), _run(set(), lps1)], "seqlp")
        assert blk is not None
        assert blk["distances"] == [round(1.23456789012345e-3, 9)]
        assert blk["floor"] == 0.001234568
        for d in blk["distances"]:
            assert d == round(d, ROUND_PLACES)


# =========================================================================== 2. alpha_single
class TestAlphaSingle:
    def test_r5_gives_10_pairs_and_1_over_11(self):
        blk = pairwise(_same_model_runs(5), "exact")
        assert blk is not None
        assert blk["runs"] == 5
        assert blk["pairs"] == 10
        assert blk["alpha_single"] == 1 / 11
        assert abs(blk["alpha_single"] - 0.0909) < 5e-5  # the §3.1 example prints 0.0909
        assert blk["floor"] == 0.0
        assert blk["distances"] == [0.0] * 10

    def test_r8_gives_28_pairs_and_1_over_29(self):
        blk = pairwise(_same_model_runs(8), "seqlp")
        assert blk is not None
        assert blk["runs"] == 8
        assert blk["pairs"] == 28
        assert blk["alpha_single"] == 1 / 29
        assert len(blk["distances"]) == 28

    def test_pairs_formula_for_every_r(self):
        for r in range(2, 12):
            blk = pairwise(_same_model_runs(r), "exact")
            assert blk["pairs"] == r * (r - 1) // 2
            assert blk["alpha_single"] == 1 / (blk["pairs"] + 1)

    def test_block_keys_are_exactly_the_contract(self):
        blk = pairwise(_same_model_runs(3), "exact")
        assert set(blk) == {"floor", "distances", "runs", "pairs", "alpha_single"}


# =========================================================================== 5. pairwise / floors
class TestPairwise:
    def test_fewer_than_two_runs_refused(self):
        with pytest.raises(ValueError):
            pairwise(_same_model_runs(1), "exact")
        with pytest.raises(ValueError):
            pairwise([], "exact")
        with pytest.raises(ValueError):
            floors(_same_model_runs(1))

    def test_unknown_channel_refused(self):
        with pytest.raises(ValueError):
            pairwise(_same_model_runs(2), "vibes")

    def test_hand_computed_exact_floor(self):
        # run0: nothing changed; run1: item a changed; run2: items a and c changed.
        # exact(run0, run1) = 1 − 3/4 = 0.25
        # exact(run0, run2) = 1 − 2/4 = 0.5
        # exact(run1, run2): a equal (both changed the same way), c differs → 1 − 3/4 = 0.25
        lps = {"a": -1.0, "b": -2.0, "c": -3.0, "d": -4.0}
        runs = [_run(set(), lps), _run({"a"}, lps), _run({"a", "c"}, lps)]
        blk = pairwise(runs, "exact")
        assert blk["distances"] == [0.25, 0.5, 0.25]
        assert blk["floor"] == 0.5
        assert blk["pairs"] == 3 and blk["alpha_single"] == 0.25

    def test_hand_computed_seqlp_floor(self):
        # |Δ| per item between run0 and run1: a 0.5, b 0.0, c 1.5, d 2.0 → mean 4.0/4 = 1.0
        # run0 vs run2: a 0.0, b 1.0, c 0.0, d 1.0 → 0.5
        # run1 vs run2: a 0.5, b 1.0, c 1.5, d 1.0 → 4.0/4 = 1.0
        lps0 = {"a": -1.0, "b": -2.0, "c": -3.0, "d": -4.0}
        lps1 = {"a": -1.5, "b": -2.0, "c": -4.5, "d": -6.0}
        lps2 = {"a": -1.0, "b": -3.0, "c": -3.0, "d": -5.0}
        blk = pairwise([_run(set(), lps0), _run(set(), lps1), _run(set(), lps2)], "seqlp")
        assert blk["distances"] == [1.0, 0.5, 1.0]
        assert blk["floor"] == 1.0

    def test_run_order_independence(self):
        lps0 = {"a": -1.0, "b": -2.0, "c": -3.0, "d": -4.0}
        lps1 = {"a": -1.5, "b": -2.0, "c": -4.5, "d": -6.0}
        lps2 = {"a": -1.0, "b": -3.0, "c": -3.0, "d": -5.0}
        runs = [_run(set(), lps0), _run({"a"}, lps1), _run({"a", "c"}, lps2)]
        ref = {ch: pairwise(runs, ch) for ch in ("exact", "seqlp")}
        for perm in itertools.permutations(runs):
            for ch in ("exact", "seqlp"):
                blk = pairwise(list(perm), ch)
                assert blk["floor"] == ref[ch]["floor"]
                assert sorted(blk["distances"]) == sorted(ref[ch]["distances"])
                assert blk["pairs"] == ref[ch]["pairs"]

    def test_item_order_within_a_run_does_not_matter(self):
        lps = {"a": -1.0, "b": -2.0, "c": -3.0, "d": -4.0}
        r0, r1 = _run(set(), lps), _run({"c"}, lps)
        r1_rev = dict(r1, items=list(reversed(r1["items"])))
        assert pairwise([r0, r1], "exact") == pairwise([r0, r1_rev], "exact")

    def test_absent_channel_on_any_run_is_none(self):
        lps = {"a": -1.0, "b": -2.0, "c": -3.0, "d": -4.0}
        runs = [_run(set(), lps), _run(set(), lps, seqlp=False), _run(set(), lps)]
        assert pairwise(runs, "seqlp") is None
        assert pairwise(runs, "topk") is None  # absent on every run
        assert pairwise(runs, "resid") is None  # no block at all
        assert pairwise(runs, "exact") is not None

    def test_floors_covers_every_channel_with_none_for_absent(self):
        runs = _same_model_runs(5)
        out = floors(runs)
        assert set(out) == set(CHANNELS)
        assert out["exact"]["floor"] == 0.0 and out["exact"]["pairs"] == 10
        assert out["seqlp"]["floor"] == 0.0
        assert out["topk"] is None and out["resid"] is None and out["lens"] is None

    def test_floors_with_white_box_channels(self):
        lps = {"a": -1.0, "b": -2.0, "c": -3.0, "d": -4.0}
        prof0 = [[1.0, 1.0, 2.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]]
        prof1 = [[1.0, 1.0, 2.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]]
        lens0 = {"n_layers": 3, "converge_layer": [1, 2, 2, 0], "mean": 1.25}
        lens1 = {"n_layers": 3, "converge_layer": [1, 2, 2, 3], "mean": 2.0}  # d: |0−3|/3 = 1 → mean 0.25
        r0 = _run(set(), lps, resid={"n_layers": 3, "profile": prof0}, lens=lens0)
        r1 = _run(set(), lps, resid={"n_layers": 3, "profile": prof1}, lens=lens1)
        out = floors([r0, r1])
        assert out["resid"]["floor"] == 0.0  # identical profiles: JSD(p,p) = 0
        assert out["lens"]["floor"] == 0.25
        assert out["exact"]["floor"] == 0.0

    def test_topk_floor_through_the_shared_distance_module(self):
        # one item, one position, both sides carry ids {1,2}: run0 lps (−0.1, −2.0),
        # run1 lps (−0.2, −2.5) → L1 = 0.1 + 0.5 = 0.6 over 1 position → 0.6
        lps = {"a": -1.0, "b": -2.0, "c": -3.0, "d": -4.0}
        r0, r1 = _run(set(), lps, topk=True), _run(set(), lps, topk=True)
        for it in r0["items"] + r1["items"]:
            it["topk"] = [{"pos": 0, "ids": [1, 2], "lps": [-0.1, -2.0]}]
        r1["items"][0]["topk"] = [{"pos": 0, "ids": [1, 2], "lps": [-0.2, -2.5]}]
        blk = pairwise([r0, r1], "topk")
        assert blk is not None
        # 4 items × 1 position: (0.6 + 0 + 0 + 0) / 4 = 0.15
        assert blk["distances"] == [0.15]
        assert blk["floor"] == 0.15

    def test_lens_layer_count_mismatch_refused(self):
        lps = {"a": -1.0, "b": -2.0, "c": -3.0, "d": -4.0}
        r0 = _run(set(), lps, lens={"n_layers": 3, "converge_layer": [1, 2, 2, 0], "mean": 1.25})
        r1 = _run(set(), lps, lens={"n_layers": 4, "converge_layer": [1, 2, 2, 0], "mean": 1.25})
        with pytest.raises(ValueError):
            pairwise([r0, r1], "lens")

    def test_misaligned_items_refused(self):
        lps = {"a": -1.0, "b": -2.0, "c": -3.0, "d": -4.0}
        r0, r1 = _run(set(), lps), _run(set(), lps)
        r1 = dict(r1, items=r1["items"][:-1])
        with pytest.raises(ValueError):
            pairwise([r0, r1], "exact")

    def test_floor_block_is_json_serializable(self):
        blk = pairwise(_same_model_runs(5), "seqlp")
        json.dumps(blk, allow_nan=False)


# =========================================================================== 3. precedence matrix
RANK = ["identity", "drift", "skew", "exceeds_floor", "inconclusive", "beyond-floor-coverage", "same"]


def _expected(verdicts, identity, sensitivity):
    if identity:
        return "identity", 1
    s = set(verdicts)
    if "drift" in s:
        return "drift", 1
    if "skew" in s:
        return "skew", 2
    if "exceeds_floor" in s:
        return "exceeds_floor", 2
    if "inconclusive" in s:
        return "inconclusive", 2
    if "beyond-floor-coverage" in s:
        return "beyond-floor-coverage", 2
    if not sensitivity:
        return "same (sensitivity unmeasured)", 2
    return "same", 0


class TestPrecedence:
    @pytest.mark.parametrize("sensitivity", [True, False])
    def test_every_pair_and_triple_of_channel_verdicts(self, sensitivity):
        channels = ("exact", "seqlp", "topk")
        for n in (1, 2, 3):
            for combo in itertools.product(CHANNEL_VERDICTS, repeat=n):
                per = dict(zip(channels, combo))
                got = overall(per, identity_diff=[], skipped=["lens"], sensitivity_present=sensitivity)
                want = _expected(combo, False, sensitivity)
                assert got == want, (combo, sensitivity, got, want)

    def test_identity_beats_everything(self):
        for combo in itertools.product(CHANNEL_VERDICTS, repeat=2):
            per = dict(zip(("exact", "seqlp"), combo))
            v, code = overall(per, identity_diff=["config_sha256"], skipped=[], sensitivity_present=False)
            assert v == "identity (config_sha256)" and code == 1

    def test_same_without_sensitivity_exits_2(self):
        assert overall({"exact": "same"}, identity_diff=[], skipped=[], sensitivity_present=False) == ("same (sensitivity unmeasured)", 2)
        assert overall({"exact": "same"}, identity_diff=[], skipped=[], sensitivity_present=True) == ("same", 0)
        assert overall({"exact": "same (transient)", "seqlp": "same"}, identity_diff=[], skipped=[], sensitivity_present=True) == ("same", 0)

    def test_skipped_never_changes_the_code(self):
        for combo in itertools.product(CHANNEL_VERDICTS, repeat=2):
            per = dict(zip(("exact", "seqlp"), combo))
            a = overall(per, identity_diff=[], skipped=[], sensitivity_present=True)
            b = overall(per, identity_diff=[], skipped=["topk", "resid", "lens"], sensitivity_present=True)
            assert a == b

    def test_no_shared_channel_is_inconclusive(self):
        assert overall({}, identity_diff=[], skipped=["exact", "seqlp"], sensitivity_present=True) == ("inconclusive", 2)

    def test_exit_codes_match_the_table(self):
        assert EXIT["same"] == 0
        assert EXIT["drift"] == EXIT["identity"] == 1
        for k in ("inconclusive", "skew", "beyond-floor-coverage", "sensitivity-unmeasured"):
            assert EXIT[k] == 2
        assert EXIT["mismatch"] == 3 and EXIT["invalid"] == 4 and EXIT["unavailable"] == 5

    def test_unknown_channel_verdict_refused(self):
        with pytest.raises(ValueError):
            overall({"exact": "unmeasured"}, identity_diff=[], skipped=[], sensitivity_present=True)
        with pytest.raises(ValueError):
            overall({"exact": "same"}, identity_diff=[""], skipped=[], sensitivity_present=True)


# =========================================================================== end to end on the floor
class TestFloorThenDecide:
    def test_batch1_floor_zero_then_a_single_flip_exceeds(self):
        runs = _same_model_runs(5)
        blk = pairwise(runs, "exact")
        assert blk["floor"] == 0.0
        lps = {"a": -1.0, "b": -2.0, "c": -3.0, "d": -4.0}
        new = _run({"d"}, lps)
        d = F._distance(runs[0], new, "exact")
        assert d == 0.25
        assert decide(d, blk["floor"], None) == "exceeds_floor"
        assert decide(d, blk["floor"], 0.25) == "drift"
        assert decide(d, blk["floor"], 0.0) == "same (transient)"
        assert decide(0.0, blk["floor"], None) == "same"

    def test_a_noisy_floor_absorbs_a_distance_inside_it(self):
        lps0 = {"a": -1.0, "b": -2.0, "c": -3.0, "d": -4.0}
        lps1 = {"a": -1.5, "b": -2.0, "c": -4.5, "d": -6.0}
        lps2 = {"a": -1.0, "b": -3.0, "c": -3.0, "d": -5.0}
        runs = [_run(set(), lps0), _run(set(), lps1), _run(set(), lps2)]
        blk = pairwise(runs, "seqlp")
        assert blk["floor"] == 1.0
        new = _run(set(), {"a": -1.25, "b": -2.25, "c": -3.25, "d": -4.25})  # mean |Δ| = 0.25
        assert decide(F._distance(runs[0], new, "seqlp"), blk["floor"], None) == "same"
        per = {"exact": decide(0.0, 0.0, None), "seqlp": decide(0.25, blk["floor"], None)}
        assert overall(per, identity_diff=[], skipped=["topk"], sensitivity_present=True) == ("same", 0)
