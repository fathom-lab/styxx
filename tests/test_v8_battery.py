"""tests/test_v8_battery.py -- Appendix C scoring and canary-v1 selection (styxx/v8/battery.py).

The Appendix C arithmetic is pinned on hand-built sweep records: every expected number is a
literal with the arithmetic that produced it in a comment beside it, so a change to the
formula fails here rather than quietly re-weighting every battery ever selected.

The selection tests then drive `select` from hand-built score blocks, which is the only way to
put an item at a chosen rank, family and margin at once.  The end of the file runs the whole
path on the MockRunner: sweep -> score -> select, including the probe's finding
(``papers/v8/probe_batch_invariance_2026_09_08``) that on that battery every precision-moved
item was also batch-moved, so the flip2 exclusion of section 4.4 step 1 left the battery with
no measured precision sensitivity at all.
"""
from __future__ import annotations

import copy
import math
import random

import pytest

from styxx.v8 import battery, cert, sweep
from styxx.v8.consts import FAMILIES
from styxx.v8.jcs import canonical_bytes, sha256_hex
from styxx.v8.runner import MockRunner
from tests import v8_fixtures as F


# --------------------------------------------------------------------------- hand-built records

def result(item_id: str, ids: list[int], *, margins: list[float], stay: float) -> dict:
    return {
        "item_id": item_id,
        "token_ids": list(ids),
        "output_text": " ".join(f"w{t}" for t in ids),
        "n_generated": len(ids),
        "seq_logprob": -1.0,
        "topk": None,
        "margin_by_position": list(margins),
        "stay": stay,
    }


LN2 = math.log(2.0)

# margin = min over positions; stay chosen so flip3 = 1 - exp(stay) is an exact literal.
REFERENCE = {
    "a": result("a", [1, 2, 3], margins=[0.5, 1.25, 2.0], stay=-LN2),   # margin 0.5, flip3 0.5
    "b": result("b", [4, 5], margins=[2.0, 1.0], stay=0.0),             # margin 1.0, flip3 0.0
    "c": result("c", [7], margins=[0.25], stay=0.0),                    # margin 0.25, flip3 0.0
    "d": result("d", [9], margins=[3.0], stay=0.0),                     # margin 3.0, flip3 0.0
}


def variant(flipped: set[str]) -> dict:
    """A copy of the reference pass with ``flipped``'s token ids changed and nothing else."""
    out = copy.deepcopy(REFERENCE)
    for item_id in flipped:
        out[item_id]["token_ids"] = [t + 1000 for t in out[item_id]["token_ids"]]
    return out


def handmade_record(*, delta4: bool = True) -> dict:
    record = {
        "reference": copy.deepcopy(REFERENCE),
        # flip1: a moves under fp16 only (1/2); c moves under both (2/2); b, d never.
        "delta1": {"fp16": variant({"a", "c"}), "int8-bnb": variant({"c"})},
        # flip2: c moves under one of the two nuisance configurations (1/2); nothing else moves.
        "delta2": [
            {"config": {"batch_size": 8, "order": "canonical"}, "results": variant({"c"})},
            {"config": {"batch_size": 32, "order": "canonical"}, "results": variant(set())},
        ],
        "delta4": [],
        "params": {"record": sweep.RECORD_TAG},
    }
    if delta4:
        record["delta4"] = [
            {"config": {"id": "trailing-space", "prompt_suffix": " "}, "results": variant(set())},
            {"config": {"id": "newline", "prompt_suffix": "\n"}, "results": variant(set())},
        ]
    return record


# --------------------------------------------------------------------------- Appendix C

def test_appendix_c_arithmetic_is_pinned():
    scores = battery.score(handmade_record())
    assert sorted(scores) == ["a", "b", "c", "d"]

    a = scores["a"]
    assert a["margin"] == 0.5                    # min(0.5, 1.25, 2.0)
    assert a["flip1"] == 0.5                     # 1 of 2 delta1 variants moved
    assert a["flip2"] == 0.0                     # 0 of 2 delta2 variants moved
    assert a["flip3"] == 0.5                     # 1 - exp(-ln 2) = 1 - 0.5
    assert a["flip4"] == 0.0                     # 0 of 2 delta4 variants moved
    # 0.6 * (0.5 + 0.5) / 2 + 0.4 * exp(-0.5) = 0.3 + 0.4 * 0.6065306597126334
    #                                         = 0.3 + 0.24261226388505336
    assert a["s"] == 0.542612264

    b = scores["b"]
    assert (b["margin"], b["flip1"], b["flip2"], b["flip3"], b["flip4"]) == (1.0, 0.0, 0.0, 0.0, 0.0)
    # 0.6 * (0 + 0) / 2 + 0.4 * exp(-1.0) = 0.4 * 0.36787944117144233
    assert b["s"] == 0.147151776

    c = scores["c"]
    assert (c["margin"], c["flip1"], c["flip2"], c["flip3"]) == (0.25, 1.0, 0.5, 0.0)
    # 0.6 * (1.0 + 0.0) / 2 + 0.4 * exp(-0.25) = 0.3 + 0.4 * 0.7788007830714049
    assert c["s"] == 0.611520313

    d = scores["d"]
    assert (d["margin"], d["flip1"], d["flip2"], d["flip3"]) == (3.0, 0.0, 0.0, 0.0)
    # 0.4 * exp(-3.0) = 0.4 * 0.049787068367863944
    assert d["s"] == 0.019914827


def test_tau_scales_only_the_margin_term():
    scores = battery.score(handmade_record(), tau=2.0)
    # 0.6 * (0.5 + 0.5) / 2 + 0.4 * exp(-0.5 / 2) = 0.3 + 0.4 * 0.7788007830714049
    assert scores["a"]["s"] == 0.611520313
    assert scores["a"]["flip1"] == 0.5 and scores["a"]["flip3"] == 0.5


def test_flip4_is_none_when_delta4_was_not_run():
    scores = battery.score(handmade_record(delta4=False))
    assert all(blk["flip4"] is None for blk in scores.values())
    assert scores["a"]["s"] == 0.542612264  # delta4 never enters s


def test_an_empty_delta_family_yields_a_zero_fraction():
    record = handmade_record(delta4=False)
    record["delta1"] = {}
    scores = battery.score(record)
    assert scores["c"]["flip1"] == 0.0
    # s collapses to the margin term: 0.6 * (0 + 0) / 2 + 0.4 * exp(-0.25)
    assert scores["c"]["s"] == 0.311520313


def test_margin_by_position_and_stay_are_carried_through():
    scores = battery.score(handmade_record())
    assert scores["a"]["margin_by_position"] == [0.5, 1.25, 2.0]
    assert scores["b"]["stay"] == 0.0


@pytest.mark.parametrize("mutate, match", [
    (lambda r: r["reference"]["a"].__setitem__("margin_by_position", None), "margin_by_position"),
    (lambda r: r["reference"]["a"].__setitem__("stay", None), "stay is required"),
    (lambda r: r["reference"]["a"].__setitem__("token_ids", "1,2,3"), "token_ids"),
    (lambda r: r["delta1"]["fp16"].pop("a"), "is missing"),
    (lambda r: r["delta2"][0]["results"].pop("b"), "is missing"),
    (lambda r: r.__setitem__("reference", {}), "non-empty object"),
    (lambda r: r.__setitem__("delta1", []), "keyed by precision"),
    (lambda r: r.__setitem__("delta2", [{"config": {}}]), "results"),
])
def test_score_refusals(mutate, match):
    record = handmade_record()
    mutate(record)
    with pytest.raises(ValueError, match=match):
        battery.score(record)


def test_score_refuses_a_non_positive_tau():
    with pytest.raises(ValueError, match="tau must be > 0"):
        battery.score(handmade_record(), tau=0.0)


def test_a_subject_without_logprobs_cannot_carry_a_canary_selection():
    """An alias without log-probs has no margin and no stay; Appendix C says so, loudly."""
    rec = sweep.run_sweep(
        lambda p: MockRunner(logprobs=False),
        [{"item_id": "i00", "prompt_text": "p", "family": "recall"}],
        F.recipe(), F.alias_subject(), delta1=[], delta2=[{"batch_size": 8}],
    )
    with pytest.raises(ValueError, match="margin_by_position"):
        battery.score(rec)


# --------------------------------------------------------------------------- selection helpers

def blk(*, s: float, margin: float, flip1: float = 0.0, flip2: float = 0.0,
        flip3: float = 0.0, flip4=None) -> dict:
    return {
        "margin": margin, "margin_by_position": [margin], "stay": 0.0,
        "flip1": flip1, "flip2": flip2, "flip3": flip3, "flip4": flip4, "s": s,
    }


def item(item_id: str, family: str = "recall") -> dict:
    return {"item_id": item_id, "prompt_text": f"prompt {item_id}", "family": family}


# --------------------------------------------------------------------------- exclusion

def test_the_exclusion_removes_exactly_flip2_greater_than_zero():
    pool = [item(f"i{k}") for k in range(5)]
    scores = {
        "i0": blk(s=0.9, margin=1.0, flip2=0.0),
        "i1": blk(s=0.8, margin=1.0, flip2=0.25),
        "i2": blk(s=0.7, margin=1.0, flip2=1.0),
        "i3": blk(s=0.6, margin=1.0, flip2=0.0),
        "i4": blk(s=0.5, margin=1.0, flip2=1e-12),   # rounds to 0.0 at ROUND_PLACES
    }
    body = battery.select(pool, scores, n=5, k=0, max_family_share=1.0, perm_seed=1)
    assert [e["item_id"] for e in body["excluded"]] == ["i1", "i2"]
    assert [e["flip2"] for e in body["excluded"]] == [0.25, 1.0]
    assert {it["item_id"] for it in body["items"]} == {"i0", "i3", "i4"}
    assert battery.validate_body(body) == []


# --------------------------------------------------------------------------- stratification

def test_the_family_cap_binds_even_against_higher_scores():
    """Six recall items outrank every format item; the cap admits only four of them."""
    pool = [item(f"r{k}", "recall") for k in range(6)] + [item(f"f{k}", "format") for k in range(4)]
    scores = {f"r{k}": blk(s=0.90 - 0.01 * k, margin=1.0) for k in range(6)}
    scores.update({f"f{k}": blk(s=0.50 - 0.01 * k, margin=1.0) for k in range(4)})

    body = battery.select(pool, scores, n=8, k=0, max_family_share=0.5, perm_seed=1)
    chosen = {it["item_id"] for it in body["items"]}
    assert chosen == {"r0", "r1", "r2", "r3", "f0", "f1", "f2", "f3"}
    assert body["params"]["family_cap"] == 4          # floor(8 * 0.5)
    assert body["params"]["family_counts"] == {"recall": 4, "format": 4}
    assert body["params"]["shortfall"] == 0
    assert battery.validate_body(body) == []


def test_a_short_family_is_filled_by_rank_and_the_shortfall_is_recorded():
    pool = [item(f"r{k}", "recall") for k in range(4)] + [item(f"f{k}", "format") for k in range(3)]
    scores = {f"r{k}": blk(s=0.90 - 0.10 * k, margin=1.0) for k in range(4)}
    scores.update({f"f{k}": blk(s=0.55 - 0.05 * k, margin=1.0) for k in range(3)})

    body = battery.select(pool, scores, n=4, k=0, max_family_share=0.25, perm_seed=1)
    assert body["params"]["family_cap"] == 1          # max(1, floor(4 * 0.25))
    # Stratified pass takes the best of each family (r0, f0); the fill takes r1, r2 by rank.
    assert {it["item_id"] for it in body["items"]} == {"r0", "f0", "r1", "r2"}
    assert body["params"]["shortfall"] == 2
    assert body["params"]["family_counts"] == {"recall": 3, "format": 1}
    assert battery.validate_body(body) == []


def test_the_cap_never_drops_below_one():
    pool = [item(f"i{k}", FAMILIES[k % len(FAMILIES)]) for k in range(5)]
    scores = {f"i{k}": blk(s=0.5 - 0.01 * k, margin=1.0) for k in range(5)}
    body = battery.select(pool, scores, n=2, k=0, max_family_share=0.25, perm_seed=1)
    assert body["params"]["family_cap"] == 1          # max(1, floor(2 * 0.25)) == max(1, 0)
    assert body["params"]["n_actual"] == 2


# --------------------------------------------------------------------------- anchors

ANCHOR_POOL = [item(i) for i in ("x0", "x1", "z0", "z1", "z2", "e0")]
ANCHOR_SCORES = {
    "x0": blk(s=0.90, margin=5.0, flip1=0.5),        # flipped: never an anchor
    "x1": blk(s=0.80, margin=4.0, flip3=0.2),        # flipped: never an anchor
    "z2": blk(s=0.30, margin=1.0),                   # zero(i), smallest margin
    "z1": blk(s=0.20, margin=2.0),                   # zero(i)
    "z0": blk(s=0.10, margin=3.0),                   # zero(i), largest margin
    "e0": blk(s=0.95, margin=6.0, flip2=0.5),        # excluded, so not zero(i) either
}


def test_anchors_are_the_largest_margin_zero_items_and_never_a_flipped_one():
    body = battery.select(ANCHOR_POOL, ANCHOR_SCORES, n=2, k=2, max_family_share=1.0, perm_seed=1)
    roles = {it["item_id"]: it["role"] for it in body["items"]}
    assert roles == {"x0": "canary", "x1": "canary", "z0": "anchor", "z1": "anchor"}
    assert body["params"]["k_anchors_actual"] == 2
    assert body["params"]["anchor_overlap"] == 0
    assert body["params"]["n_actual"] == 2
    for it in body["items"]:
        if it["role"] == "anchor":
            assert it["flip1"] == 0.0 and it["flip2"] == 0.0 and it["flip3"] == 0.0
    assert [e["item_id"] for e in body["excluded"]] == ["e0"]
    assert battery.validate_body(body) == []


def test_k_anchors_actual_is_capped_by_the_supply_of_zero_items():
    body = battery.select(ANCHOR_POOL, ANCHOR_SCORES, n=2, k=64, max_family_share=1.0, perm_seed=1)
    assert body["params"]["k_anchors"] == 64
    assert body["params"]["k_anchors_actual"] == 3     # only z0, z1, z2 are zero(i)
    assert {it["item_id"] for it in body["items"] if it["role"] == "anchor"} == {"z0", "z1", "z2"}


def test_the_anchor_role_wins_when_an_item_is_in_both_sets():
    body = battery.select(ANCHOR_POOL, ANCHOR_SCORES, n=5, k=2, max_family_share=1.0, perm_seed=1)
    roles = {it["item_id"]: it["role"] for it in body["items"]}
    assert roles == {"x0": "canary", "x1": "canary", "z2": "canary", "z1": "anchor", "z0": "anchor"}
    assert body["params"]["anchor_overlap"] == 2
    assert body["params"]["n_actual"] == 3
    assert battery.validate_body(body) == []


def test_no_anchors_when_k_is_zero():
    body = battery.select(ANCHOR_POOL, ANCHOR_SCORES, n=3, k=0, max_family_share=1.0, perm_seed=1)
    assert body["params"]["k_anchors_actual"] == 0
    assert all(it["role"] == "canary" for it in body["items"])


# --------------------------------------------------------------------------- tie-break

def test_ties_break_by_margin_ascending_then_item_id_ascending():
    pool = [item("m2"), item("m1"), item("m0"), item("m3")]
    scores = {
        "m0": blk(s=0.5, margin=2.0),
        "m1": blk(s=0.5, margin=1.0),      # same s, smaller margin -> ranks ahead of m0
        "m2": blk(s=0.5, margin=1.0),      # same s and margin as m1 -> item_id breaks it
        "m3": blk(s=0.5, margin=3.0),
    }
    body = battery.select(pool, scores, n=1, k=0, max_family_share=1.0, perm_seed=1)
    assert [it["item_id"] for it in body["items"]] == ["m1"]
    body2 = battery.select(pool, scores, n=2, k=0, max_family_share=1.0, perm_seed=1)
    assert [it["item_id"] for it in body2["items"]] == ["m1", "m2"]


def test_selection_is_deterministic_under_a_shuffled_pool():
    pool = [item(f"i{k:02d}", FAMILIES[k % len(FAMILIES)]) for k in range(20)]
    scores = {
        f"i{k:02d}": blk(s=round(0.5 + 0.001 * (k % 7), 9), margin=round(1.0 + 0.1 * (k % 5), 9),
                         flip2=0.5 if k % 9 == 0 else 0.0)
        for k in range(20)
    }
    one = battery.select(pool, scores, n=8, k=3, perm_seed=11)
    shuffled = list(pool)
    random.Random(1234).shuffle(shuffled)
    two = battery.select(shuffled, scores, n=8, k=3, perm_seed=11)
    assert canonical_bytes(one) == canonical_bytes(two)
    assert [it["item_id"] for it in one["items"]] == sorted(it["item_id"] for it in one["items"])


# --------------------------------------------------------------------------- select refusals

def test_select_refuses_a_score_gap_in_either_direction():
    pool = [item("i0"), item("i1")]
    with pytest.raises(ValueError, match="no score for pool items"):
        battery.select(pool, {"i0": blk(s=0.5, margin=1.0)}, n=1, k=0, perm_seed=1)
    scores = {"i0": blk(s=0.5, margin=1.0), "i1": blk(s=0.4, margin=1.0), "i9": blk(s=0.3, margin=1.0)}
    with pytest.raises(ValueError, match="not in the pool"):
        battery.select(pool, scores, n=1, k=0, perm_seed=1)


@pytest.mark.parametrize("kwargs, match", [
    ({"n": 0, "k": 0, "perm_seed": 1}, "n must be"),
    ({"n": 1, "k": -1, "perm_seed": 1}, "k must be"),
    ({"n": 1, "k": 0, "perm_seed": "x"}, "perm_seed"),
    ({"n": 1, "k": 0, "perm_seed": 1, "max_family_share": 0.0}, "max_family_share"),
    ({"n": 1, "k": 0, "perm_seed": 1, "max_family_share": 1.5}, "max_family_share"),
])
def test_select_parameter_refusals(kwargs, match):
    pool = [item("i0")]
    scores = {"i0": blk(s=0.5, margin=1.0)}
    with pytest.raises(ValueError, match=match):
        battery.select(pool, scores, **kwargs)


def test_select_refuses_a_pool_item_without_a_family():
    with pytest.raises(ValueError, match="family must be one of"):
        battery.select([{"item_id": "i0", "prompt_text": "p"}],
                       {"i0": blk(s=0.5, margin=1.0)}, n=1, k=0, perm_seed=1)


def test_select_refuses_a_prompt_sha256_that_does_not_match():
    bad = item("i0")
    bad["prompt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="prompt_sha256"):
        battery.select([bad], {"i0": blk(s=0.5, margin=1.0)}, n=1, k=0, perm_seed=1)


# --------------------------------------------------------------------------- sensitivity

def test_sensitivity_after_exclusion_is_the_mean_flip1_over_the_selected_canaries():
    body = battery.select(ANCHOR_POOL, ANCHOR_SCORES, n=2, k=2, max_family_share=1.0, perm_seed=1)
    # canaries are x0 (flip1 0.5) and x1 (flip1 0.0): mean 0.25.  Anchors are zero by
    # construction and are not counted; they would only dilute the number.
    assert body["params"]["sensitivity_after_exclusion"] == 0.25


# --------------------------------------------------------------------------- other bodies

def test_pool_v1_body():
    body = battery.pool_v1([item("i1", "recall"), item("i0", "format")])
    assert body["kind"] == "pool-v1"
    assert [it["item_id"] for it in body["items"]] == ["i0", "i1"]
    assert all(it["role"] == "item" for it in body["items"])
    assert body["families"] == ["recall", "format"]   # FAMILIES order, not item order
    assert body["pool_size"] == 2
    assert body["pool_sha256"] == sweep.pool_sha256(body["items"])
    assert "params" not in body and "excluded" not in body
    assert battery.validate_body(body) == []


def test_fixed_v1_body():
    body = battery.fixed_v1([item("i0"), item("i1")], source="styxx-bench@v8")
    assert body["kind"] == "fixed-v1"
    assert body["source"] == "styxx-bench@v8"
    assert all(it["role"] == "item" for it in body["items"])
    assert not any(f in it for it in body["items"] for f in battery.SELECTION_ITEM_FIELDS)
    assert battery.validate_body(body) == []


def test_fixed_v1_refuses_an_empty_source():
    with pytest.raises(ValueError, match="source"):
        battery.fixed_v1([item("i0")], source="")


def test_prompt_sha256_is_computed_from_the_prompt_bytes():
    body = battery.pool_v1([item("i0")])
    assert body["items"][0]["prompt_sha256"] == sha256_hex(b"prompt i0")


# --------------------------------------------------------------------------- validate_body

def canary_body() -> dict:
    return battery.select(ANCHOR_POOL, ANCHOR_SCORES, n=2, k=2, max_family_share=1.0, perm_seed=1)


def test_validate_body_accepts_what_this_module_builds():
    assert battery.validate_body(canary_body()) == []
    assert battery.validate_body(battery.pool_v1([item("i0")])) == []
    assert battery.validate_body(battery.fixed_v1([item("i0")], source="s")) == []


def _prefix(reasons: list[str]) -> set[str]:
    return {r.split(":", 1)[0].split("[", 1)[0] for r in reasons}


@pytest.mark.parametrize("mutate, prefix", [
    (lambda b: b.__setitem__("kind", "canary-v2"), "kind"),
    (lambda b: b.__setitem__("items", {}), "items"),
    (lambda b: b.__setitem__("items", []), "items"),
    (lambda b: b["items"].reverse(), "items"),
    (lambda b: b["items"][0].__setitem__("prompt_text", "tampered"), "item"),
    (lambda b: b["items"][0].__setitem__("family", "poetry"), "item"),
    (lambda b: b["items"][0].__setitem__("role", "item"), "item"),
    (lambda b: b["items"][0].pop("score"), "item"),
    (lambda b: b["items"][0].__setitem__("flip2", 0.5), "item"),
    (lambda b: b.pop("params"), "params"),
    (lambda b: b["params"].pop("tie_break"), "params"),
    (lambda b: b.pop("excluded"), "excluded"),
    (lambda b: b["excluded"][0].__setitem__("flip2", 0.0), "excluded"),
    (lambda b: b["excluded"].append({"item_id": b["items"][0]["item_id"], "flip2": 1.0}), "excluded"),
    (lambda b: b.__setitem__("families", ["poetry"]), "families"),
    (lambda b: b.__setitem__("families", []), "families"),
    (lambda b: b.pop("redacted"), "body"),
])
def test_validate_body_negatives_on_a_canary_body(mutate, prefix):
    body = canary_body()
    mutate(body)
    reasons = battery.validate_body(body)
    assert reasons, f"expected a {prefix} reason"
    assert prefix in _prefix(reasons), reasons


def test_validate_body_catches_an_anchor_that_moved():
    body = canary_body()
    for it in body["items"]:
        if it["role"] == "anchor":
            it["flip1"] = 0.5
            break
    reasons = battery.validate_body(body)
    assert any(r.startswith("item[") and "anchor needs flip1" in r for r in reasons), reasons


def test_validate_body_catches_a_miscounted_k_anchors_actual():
    body = canary_body()
    body["params"]["k_anchors_actual"] = 99
    reasons = battery.validate_body(body)
    assert any(r.startswith("anchors: params.k_anchors_actual") for r in reasons), reasons


def test_validate_body_catches_more_anchors_than_k():
    body = canary_body()
    body["params"]["k_anchors"] = 1
    reasons = battery.validate_body(body)
    assert any(r.startswith("anchors: 2 anchors exceed") for r in reasons), reasons


def test_validate_body_catches_a_family_over_the_cap_the_shortfall_cannot_explain():
    pool = [item(f"r{k}", "recall") for k in range(4)] + [item("f0", "format")]
    scores = {f"r{k}": blk(s=0.9 - 0.1 * k, margin=1.0) for k in range(4)}
    scores["f0"] = blk(s=0.1, margin=1.0)
    body = battery.select(pool, scores, n=4, k=0, max_family_share=0.25, perm_seed=1)
    assert battery.validate_body(body) == []
    body["params"]["shortfall"] = 0      # the same battery, with the fill no longer declared
    reasons = battery.validate_body(body)
    assert any(r.startswith("share: family 'recall'") for r in reasons), reasons


@pytest.mark.parametrize("mutate, prefix", [
    (lambda b: b["items"][0].__setitem__("role", "canary"), "item"),
    (lambda b: b["items"][0].__setitem__("score", 0.5), "item"),
    (lambda b: b.__setitem__("params", {"n": 1}), "params"),
    (lambda b: b.__setitem__("excluded", []), "excluded"),
])
def test_validate_body_negatives_on_a_pool_body(mutate, prefix):
    body = battery.pool_v1([item("i0"), item("i1")])
    mutate(body)
    reasons = battery.validate_body(body)
    assert reasons and prefix in _prefix(reasons), reasons


def test_validate_body_refuses_a_non_object():
    assert battery.validate_body(["not", "a", "body"]) == ["body: not a JSON object"]


# --------------------------------------------------------------------------- cert integration

def test_the_bodies_this_module_builds_pass_cert_check():
    canary = cert.check(F.make_cert(
        "battery",
        body=canary_body(),
        recipe=F.recipe(battery=F.POOL_ID),
        refs=F.canary_battery_refs(F.POOL_ID, F.FINGERPRINT_ID),
    ))
    assert canary.ok, canary.reasons

    # Section 4.5: a pool-v1 or fixed-v1 body is a root -- no recipe, no refs, and it still checks.
    pool_cert = F.make_cert("battery", body=battery.pool_v1([item("i0"), item("i1")]))
    assert pool_cert["recipe"] == {} and pool_cert["refs"] == []
    pool = cert.check(pool_cert)
    assert pool.ok, pool.reasons

    fixed_cert = F.make_cert(
        "battery", body=battery.fixed_v1([item("i0")], source="styxx-bench@v8"))
    assert fixed_cert["recipe"] == {} and fixed_cert["refs"] == []
    fixed = cert.check(fixed_cert)
    assert fixed.ok, fixed.reasons


def test_a_canary_body_without_selected_against_is_refused_by_the_schema():
    check = cert.check(F.make_cert(
        "battery",
        body=canary_body(),
        recipe=F.recipe(battery=F.POOL_ID),
        refs=[{"role": "battery", "id": F.POOL_ID}, {"role": "pool", "id": F.POOL_ID}],
    ))
    assert not check.ok
    assert any(r.startswith("schema[battery]") for r in check.reasons), check.reasons


def test_a_canary_body_without_a_recipe_battery_is_refused_by_the_schema():
    """Section 4.5: the recipe.battery requirement moved onto canary-v1; it did not evaporate."""
    check = cert.check(F.make_cert(
        "battery",
        body=canary_body(),
        refs=[{"role": "selected_against", "id": F.FINGERPRINT_ID}],
    ))
    assert not check.ok
    assert "schema[battery]: recipe: 'battery' is a required property" in check.reasons, check.reasons


# --------------------------------------------------------------------------- the probe scenario

def probe_pool(n: int = 12) -> list[dict]:
    return [
        {"item_id": f"i{k:02d}", "prompt_text": f"prompt {k}",
         "family": FAMILIES[k % len(FAMILIES)]}
        for k in range(n)
    ]


def test_the_probe_scenario_leaves_the_battery_with_no_measured_precision_sensitivity():
    """precision_items is a SUBSET of nuisance_items, as the batch-invariance probe found.

    Every precision-sensitive item is therefore also batch-sensitive, the flip2 exclusion of
    section 4.4 step 1 removes all of them, and what survives has flip1 == 0 by construction.
    That is the hypothesis of section 4.1 failing to get a receipt, reproduced in the mock.
    """
    nuisance = {"i00", "i01", "i02", "i03"}
    precision = {"i00", "i01"}
    assert precision < nuisance

    def make(p):
        return MockRunner(nuisance_items=nuisance,
                          precision_items={"fp16": precision, "int8-bnb": precision})

    record = sweep.run_sweep(
        make, probe_pool(), F.recipe(), F.weights_subject(),
        delta1=["fp16", "int8-bnb"],
        delta2=[{"batch_size": 8, "order": "canonical"},
                {"batch_size": 32, "order": "canonical"},
                {"batch_size": 8, "order": "perm", "perm_seed": 11}],
    )
    scores = battery.score(record)
    # The precision-sensitive items really did move under delta1 -- they were not silent.
    assert {i for i in scores if scores[i]["flip1"] > 0} == precision
    assert {i for i in scores if scores[i]["flip2"] > 0} == nuisance

    body = battery.select(probe_pool(), scores, n=6, k=64, perm_seed=11)
    assert {e["item_id"] for e in body["excluded"]} == nuisance
    assert all(it["flip1"] == 0.0 for it in body["items"])
    assert body["params"]["sensitivity_after_exclusion"] == 0.0
    assert battery.validate_body(body) == []


def test_the_mock_yields_no_anchors_because_its_flip3_is_never_exactly_zero():
    """A recorded property of MockRunner, not of the spec.

    ``flip3 = 1 - exp(stay)`` and the mock's log-prob gaps are small enough that ``stay`` is
    always a few tenths of a nat below zero, so ``zero(i)`` is unsatisfiable on mock data.  A
    real confident item has ``exp(stay) == 1.0`` in float64 and does qualify -- the hand-built
    anchor tests above are where that path is exercised.
    """
    record = sweep.run_sweep(
        lambda p: MockRunner(), probe_pool(6), F.recipe(), F.weights_subject(),
        delta1=["fp16"], delta2=[{"batch_size": 8, "order": "canonical"}],
    )
    scores = battery.score(record)
    assert all(blk["flip3"] > 0 for blk in scores.values())
    body = battery.select(probe_pool(6), scores, n=3, k=64, perm_seed=11)
    assert body["params"]["k_anchors_actual"] == 0
