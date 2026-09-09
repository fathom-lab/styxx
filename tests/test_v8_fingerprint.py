"""Tests for ``styxx.v8.fingerprint`` -- fingerprint bodies, the A.3 exact hash, floors attached.

Contract: ``styxx/v8/INTERFACES_layer2.md`` section 6.  Spec: v0.2 draft sections 3, 5,
Appendix A.3, Appendix B.

Every number pinned here was derived by hand or from bytes constructed in the test itself; the
arithmetic is in the comment beside it.  Nothing skips.
"""
from __future__ import annotations

import copy
import hashlib
import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from styxx.v8 import cert as certmod
from styxx.v8 import fingerprint as FP
from styxx.v8.jcs import canonical_bytes
from styxx.v8.runner import MockRunner
from tests import v8_fixtures as F

# --------------------------------------------------------------------------- helpers


def pool_cert(n: int = 3, *, redacted: bool = False, roles: dict[str, str] | None = None) -> dict:
    """A pool-v1 battery cert whose id is the one ``F.recipe()`` names under ``recipe.battery``."""
    body = F.battery_body("pool-v1", n=n)
    body["redacted"] = redacted
    if redacted:
        for item in body["items"]:
            item.pop("prompt_text", None)
    if roles:
        for item in body["items"]:
            item["role"] = roles.get(item["item_id"], item.get("role", "item"))
    return {"id": F.BATTERY_ID, "type": "battery", "body": body}


def result(item_id: str, token_ids: list[int], text: str, *, logprobs: bool = True) -> dict:
    """An ``ItemResult`` (section 3) built by hand."""
    out: dict = {
        "item_id": item_id,
        "token_ids": list(token_ids),
        "output_text": text,
        "n_generated": len(token_ids),
        "seq_logprob": None,
        "topk": None,
        "margin_by_position": None,
        "stay": None,
    }
    if logprobs:
        out["seq_logprob"] = -1.25
        out["topk"] = [{"pos": 0, "ids": [token_ids[0], token_ids[0] + 1], "lps": [-0.1, -2.3]}]
    return out


def three_results(*, logprobs: bool = True) -> list[dict]:
    return [
        result("i00", [100, 200, 300], "answer 0", logprobs=logprobs),
        result("i01", [101, 201, 301], "answer 1", logprobs=logprobs),
        result("i02", [102, 202, 302], "answer 2", logprobs=logprobs),
    ]


def a3_digest(token_ids: list[int]) -> bytes:
    """Appendix A.3, recomputed in the test from bytes the test constructs."""
    return hashlib.sha256(canonical_bytes(token_ids)).digest()


PERMUTED_ORDERS = [
    None,  # A.3 order
    ["i02", "i00", "i01", "i03", "i04", "i05", "i06", "i07"],
    ["i07", "i06", "i05", "i04", "i03", "i02", "i01", "i00"],
    ["i01", "i02", "i03", "i04", "i05", "i06", "i07", "i00"],
    ["i03", "i01", "i00", "i02", "i05", "i04", "i07", "i06"],
]


def five_runs(
    batch_size: int, *, logprobs: bool = True, nuisance_items=("i02", "i05"), drift=()
) -> list[dict]:
    """The section 5.1 plan on the mock: 5 runs varying item order (and batch size).

    ``drift`` moves those item ids on every run, which is how a test builds a run set whose
    floor is above 0 without changing anything else about the plan.
    """
    runner = MockRunner(
        nuisance_items=set(nuisance_items), logprobs=logprobs, drift_items=set(drift)
    )
    recipe = F.recipe(decoding=F.decoding(batch_size=batch_size))
    battery = pool_cert(8)
    return [
        FP.run_fingerprint(runner, F.weights_subject(), recipe, battery, run_index=k, nuisance={}, order=order)
        for k, order in enumerate(PERMUTED_ORDERS)
    ]


# --------------------------------------------------------------------------- item_record


def test_item_record_hashes_are_appendix_a3():
    r = result("i00", [100, 200, 300], "answer 0")
    rec = FP.item_record(r, False)
    # token_ids_sha256 = sha256(UTF-8(JCS([100,200,300]))) = sha256(b"[100,200,300]")
    assert canonical_bytes([100, 200, 300]) == b"[100,200,300]"
    assert rec["token_ids_sha256"] == hashlib.sha256(b"[100,200,300]").hexdigest()
    # output_sha256 = sha256(UTF-8("answer 0"))
    assert rec["output_sha256"] == hashlib.sha256(b"answer 0").hexdigest()
    assert rec["n_generated"] == 3
    assert rec["token_ids"] == [100, 200, 300]
    assert rec["output_text"] == "answer 0"
    assert rec["seq_logprob"] == -1.25
    assert rec["topk"] == [{"pos": 0, "ids": [100, 101], "lps": [-0.1, -2.3]}]


def test_item_record_redacted_omits_text_but_keeps_the_checksum_preimage():
    r = result("i00", [100, 200, 300], "answer 0")
    plain = FP.item_record(r, False)
    hidden = FP.item_record(r, True)
    assert "output_text" in plain
    assert "output_text" not in hidden
    # the checksum and its preimage both survive redaction
    assert hidden["token_ids_sha256"] == plain["token_ids_sha256"]
    assert hidden["token_ids"] == [100, 200, 300]
    assert hidden["output_sha256"] == plain["output_sha256"]
    # and the exact channel is unaffected by redaction
    assert FP.exact_hash([hidden]) == FP.exact_hash([plain])


def test_item_record_carries_absent_logprobs_as_none():
    rec = FP.item_record(result("i00", [1], "x", logprobs=False), False)
    assert rec["seq_logprob"] is None
    assert rec["topk"] is None


def test_item_record_carries_prefix_token_ids_when_the_runner_supplies_them():
    r = result("i00", [1, 2], "x")
    r["prefix_token_ids"] = [7, 8, 9]
    assert FP.item_record(r, False)["prefix_token_ids"] == [7, 8, 9]


def test_item_record_refuses_an_n_generated_that_does_not_match_the_ids():
    r = result("i00", [1, 2], "x")
    r["n_generated"] = 3
    with pytest.raises(ValueError, match="n_generated"):
        FP.item_record(r, False)


@pytest.mark.parametrize(
    "mutate, exc, match",
    [
        (lambda r: r.update(item_id=""), ValueError, "item_id"),
        (lambda r: r.update(item_id=7), ValueError, "item_id"),
        (lambda r: r.update(token_ids=[1, -2]), ValueError, "token_ids"),
        (lambda r: r.update(token_ids=[1, True]), TypeError, "token_ids"),
        (lambda r: r.update(token_ids="12"), TypeError, "token_ids"),
        (lambda r: r.update(output_text=None), TypeError, "output_text"),
        (lambda r: r.update(seq_logprob=float("nan")), ValueError, "seq_logprob"),
        (lambda r: r.update(seq_logprob=float("inf")), ValueError, "seq_logprob"),
        (lambda r: r.update(topk=[{"pos": 0, "ids": [1], "lps": [-0.1, -2.0]}]), ValueError, "ids against"),
        (lambda r: r.update(topk=[{"pos": 0, "ids": [], "lps": []}]), ValueError, "empty top-k"),
        (lambda r: r.update(topk=[{"pos": 0, "ids": [1], "lps": [-0.1]}, {"pos": 0, "ids": [2], "lps": [-0.2]}]), ValueError, "repeats position"),
        (lambda r: r.update(topk=[{"pos": -1, "ids": [1], "lps": [-0.1]}]), ValueError, "pos"),
        (lambda r: r.update(topk=[{"pos": 0, "ids": [1], "lps": [float("inf")]}]), ValueError, "lps"),
    ],
)
def test_item_record_refuses_malformed_results(mutate, exc, match):
    r = result("i00", [1, 2], "x")
    mutate(r)
    with pytest.raises(exc, match=match):
        FP.item_record(r, False)


def test_item_record_refuses_a_non_bool_redacted_flag():
    with pytest.raises(TypeError, match="redacted"):
        FP.item_record(result("i00", [1], "x"), "yes")


# --------------------------------------------------------------------------- exact_hash


def test_exact_hash_pins_the_a3_construction_by_hand():
    # Two items, given out of order: "b" ([1,2]) then "a" ([3]).
    # A.3 sorts by item_id, so the concatenation is digest("a") || digest("b"):
    #   digest(a) = sha256(b"[3]"), digest(b) = sha256(b"[1,2]")
    #   hash      = sha256(digest(a) || digest(b))
    recs = [
        FP.item_record(result("b", [1, 2], "x", logprobs=False), False),
        FP.item_record(result("a", [3], "y", logprobs=False), False),
    ]
    da = hashlib.sha256(b"[3]").digest()
    db = hashlib.sha256(b"[1,2]").digest()
    assert da.hex() == "06d033ece6645de592db973644cf7357255f24536ff7b03c3b2ace10736f7636"
    assert db.hex() == "49a64717d5d4cb19952e6eac2946415cf6879adacf9908e7d872332d32c6e684"
    assert FP.exact_hash(recs) == hashlib.sha256(da + db).hexdigest()
    assert FP.exact_hash(recs) == "32a16fe4045541ef38bd4cc9446aff6b7374984a86ea5eb7005ca8018bfc1fae"


def test_exact_hash_is_independent_of_the_order_the_items_are_given_in():
    recs = [FP.item_record(r, False) for r in three_results()]
    assert FP.exact_hash(recs) == FP.exact_hash(list(reversed(recs)))
    assert FP.exact_hash(recs) == FP.exact_hash([recs[1], recs[2], recs[0]])


def test_exact_hash_changes_when_one_token_id_changes():
    recs = [FP.item_record(r, False) for r in three_results()]
    before = FP.exact_hash(recs)
    moved = three_results()
    moved[1]["token_ids"][2] = 302  # was 301
    after = FP.exact_hash([FP.item_record(r, False) for r in moved])
    assert after != before


def test_exact_hash_reads_a_record_that_carries_only_the_digest():
    rec = FP.item_record(result("i00", [1, 2], "x", logprobs=False), False)
    digest_only = {"item_id": "i00", "token_ids_sha256": rec["token_ids_sha256"]}
    assert FP.exact_hash([digest_only]) == FP.exact_hash([rec])
    # the prefixed form of A.1 names the same digest
    prefixed = {"item_id": "i00", "token_ids_sha256": "sha256:" + rec["token_ids_sha256"]}
    assert FP.exact_hash([prefixed]) == FP.exact_hash([rec])


def test_exact_hash_refuses_a_record_that_disagrees_with_itself():
    rec = FP.item_record(result("i00", [1, 2], "x", logprobs=False), False)
    rec["token_ids_sha256"] = hashlib.sha256(b"[9]").hexdigest()
    with pytest.raises(ValueError, match="is not the digest of its own"):
        FP.exact_hash([rec])


@pytest.mark.parametrize(
    "items, match",
    [
        ([], "at least one item"),
        ([{"item_id": "a"}], "neither token_ids nor token_ids_sha256"),
        ([{"item_id": "a", "token_ids_sha256": "AB" * 32}], "64 lowercase hex"),
        ([{"item_id": "a", "token_ids_sha256": "ab" * 31}], "64 lowercase hex"),
        (
            [{"item_id": "a", "token_ids": [1]}, {"item_id": "a", "token_ids": [2]}],
            "repeats item_id",
        ),
    ],
)
def test_exact_hash_refuses_malformed_item_lists(items, match):
    with pytest.raises(ValueError, match=match):
        FP.exact_hash(items)


# --------------------------------------------------------------------------- order_sha256


def test_order_sha256_is_the_jcs_digest_of_the_id_array():
    assert FP.order_sha256(["i00", "i01"]) == hashlib.sha256(b'["i00","i01"]').hexdigest()
    assert FP.order_sha256(["i01", "i00"]) != FP.order_sha256(["i00", "i01"])


def test_order_sha256_refuses_a_repeated_id():
    with pytest.raises(ValueError, match="repeats an id"):
        FP.order_sha256(["i00", "i00"])


# --------------------------------------------------------------------------- build


def test_build_produces_the_section_3_1_body_and_a_cert_that_checks():
    battery = pool_cert(3)
    body = FP.build(
        F.weights_subject(),
        F.recipe(),
        battery,
        three_results(),
        run_index=0,
        nuisance={},
    )
    assert body["run_index"] == 0
    assert body["tier"] == "white-box"  # subject.kind == weights
    assert body["redacted"] is False
    assert body["nuisance"]["batch_size"] == 1
    assert body["nuisance"]["item_order_sha256"] == FP.order_sha256(["i00", "i01", "i02"])
    assert [i["item_id"] for i in body["items"]] == ["i00", "i01", "i02"]
    assert body["channels"]["exact"]["hash"] == FP.exact_hash(body["items"])
    assert body["channels"]["seqlp"] == {"present": True}
    assert body["channels"]["topk"] == {"present": True}
    assert "resid" not in body["channels"] and "lens" not in body["channels"]
    assert "redacted_battery" not in body

    cert = F.make_cert(
        "fingerprint",
        subject=F.weights_subject(),
        recipe=F.recipe(),
        body=body,
        refs=[{"role": "battery", "id": F.BATTERY_ID}],
    )
    outcome = certmod.check(cert)
    assert outcome.ok, outcome.reasons


def test_build_tier_follows_subject_kind_and_alias_bodies_check():
    battery = pool_cert(3)
    body = FP.build(
        F.alias_subject(),
        F.recipe(),
        battery,
        three_results(logprobs=False),
        run_index=1,
        nuisance={},
    )
    assert body["tier"] == "black-box"
    assert body["channels"]["seqlp"] == {"present": False}
    assert body["channels"]["topk"] == {"present": False}
    cert = F.make_cert(
        "fingerprint",
        subject=F.alias_subject(),
        recipe=F.recipe(),
        body=body,
        refs=[{"role": "battery", "id": F.BATTERY_ID}],
    )
    outcome = certmod.check(cert)
    assert outcome.ok, outcome.reasons


def test_build_marks_a_channel_absent_when_a_single_item_lacks_it():
    results = three_results()
    results[1]["seq_logprob"] = None  # one item out of three
    body = FP.build(F.weights_subject(), F.recipe(), pool_cert(3), results, run_index=0, nuisance={})
    assert body["channels"]["seqlp"] == {"present": False}
    assert body["channels"]["topk"] == {"present": True}


def test_build_copies_the_battery_role_onto_every_item():
    battery = pool_cert(3, roles={"i00": "anchor", "i01": "canary"})
    body = FP.build(F.weights_subject(), F.recipe(), battery, three_results(), run_index=0, nuisance={})
    assert [i["role"] for i in body["items"]] == ["anchor", "canary", "item"]


def test_build_attaches_the_white_box_channels_and_computes_their_means():
    battery = pool_cert(3)
    white_box = {
        "resid": {"n_layers": 2, "profile": [[0.25, 0.75], [0.75, 0.25], [0.5, 0.5]]},
        "lens": {"n_layers": 2, "converge_layer": [1, 2, 0]},
    }
    body = FP.build(
        F.weights_subject(), F.recipe(), battery, three_results(), run_index=0,
        nuisance={}, white_box=white_box,
    )
    # resid mean, element-wise: (0.25+0.75+0.5)/3 = 0.5 and (0.75+0.25+0.5)/3 = 0.5
    assert body["channels"]["resid"]["mean"] == [0.5, 0.5]
    assert body["channels"]["resid"]["present"] is True
    assert body["channels"]["resid"]["n_layers"] == 2
    # lens mean: (1 + 2 + 0)/3 = 1.0
    assert body["channels"]["lens"]["mean"] == 1.0
    assert body["channels"]["lens"]["converge_layer"] == [1, 2, 0]


def test_build_refuses_white_box_channels_on_an_alias_subject():
    with pytest.raises(ValueError, match="weights subject"):
        FP.build(
            F.alias_subject(), F.recipe(), pool_cert(3), three_results(), run_index=0,
            nuisance={}, white_box={"lens": {"n_layers": 2, "converge_layer": [1, 1, 1]}},
        )


@pytest.mark.parametrize(
    "white_box, match",
    [
        ({"resid": {"n_layers": 2, "profile": [[0.5, 0.5]]}}, "for 3 items"),
        ({"resid": {"n_layers": 2, "profile": [[0.5], [0.5], [0.5]]}}, "layers"),
        ({"lens": {"n_layers": 2, "converge_layer": [1, 2]}}, "for 3 items"),
        ({"lens": {"n_layers": 2, "converge_layer": [1, 2, 3]}}, "above n_layers"),
        ({"sae": {}}, "white_box may only carry"),
        ({}, "carries no channel"),
    ],
)
def test_build_refuses_malformed_white_box_blocks(white_box, match):
    with pytest.raises(ValueError, match=match):
        FP.build(
            F.weights_subject(), F.recipe(), pool_cert(3), three_results(), run_index=0,
            nuisance={}, white_box=white_box,
        )


def test_build_refuses_results_that_do_not_cover_the_battery():
    with pytest.raises(ValueError, match="missing"):
        FP.build(F.weights_subject(), F.recipe(), pool_cert(3), three_results()[:2], run_index=0, nuisance={})


def test_build_refuses_an_item_the_battery_does_not_carry():
    results = three_results()
    results[2]["item_id"] = "i99"
    with pytest.raises(ValueError, match="is not in the battery"):
        FP.build(F.weights_subject(), F.recipe(), pool_cert(3), results, run_index=0, nuisance={})


def test_build_refuses_a_repeated_result():
    results = three_results()
    results[2] = copy.deepcopy(results[1])
    with pytest.raises(ValueError, match="repeats an item_id"):
        FP.build(F.weights_subject(), F.recipe(), pool_cert(3), results, run_index=0, nuisance={})


def test_build_refuses_a_nuisance_block_that_contradicts_the_run():
    battery = pool_cert(3)
    with pytest.raises(ValueError, match="contradicts recipe.decoding.batch_size"):
        FP.build(
            F.weights_subject(), F.recipe(), battery, three_results(), run_index=0,
            nuisance={"batch_size": 8},
        )
    with pytest.raises(ValueError, match="item_order_sha256"):
        FP.build(
            F.weights_subject(), F.recipe(), battery, three_results(), run_index=0,
            nuisance={"item_order_sha256": "0" * 64},
        )


def test_build_keeps_a_nuisance_block_that_agrees_and_carries_the_extra_fields():
    body = FP.build(
        F.weights_subject(),
        F.recipe(decoding=F.decoding(batch_size=8)),
        pool_cert(3),
        three_results(),
        run_index=2,
        nuisance={"batch_size": 8, "gpu": "A100", "driver": "550.54"},
    )
    assert body["nuisance"]["batch_size"] == 8
    assert body["nuisance"]["gpu"] == "A100"
    assert body["nuisance"]["driver"] == "550.54"


def test_build_refuses_a_recipe_naming_another_battery():
    battery = pool_cert(3)
    with pytest.raises(ValueError, match="is not the battery cert's id"):
        FP.build(
            F.weights_subject(), F.recipe(battery=F.POOL_ID), battery, three_results(),
            run_index=0, nuisance={},
        )


def test_build_flags_a_redacted_battery():
    body = FP.build(
        F.weights_subject(), F.recipe(), pool_cert(3, redacted=True), three_results(),
        run_index=0, nuisance={},
    )
    assert body["redacted_battery"] is True


def test_build_refuses_an_unknown_subject_kind():
    with pytest.raises(ValueError, match="subject.kind"):
        FP.build({"kind": "vibes"}, F.recipe(), pool_cert(3), three_results(), run_index=0, nuisance={})


def test_build_is_deterministic_to_the_byte():
    args = (F.weights_subject(), F.recipe(), pool_cert(3), three_results())
    a = FP.build(*args, run_index=0, nuisance={})
    b = FP.build(*args, run_index=0, nuisance={})
    assert canonical_bytes(a) == canonical_bytes(b)


def test_build_redacted_body_omits_every_output_text():
    body = FP.build(
        F.weights_subject(), F.recipe(), pool_cert(3), three_results(),
        run_index=0, nuisance={}, redacted=True,
    )
    assert body["redacted"] is True
    assert all("output_text" not in item for item in body["items"])
    assert all(item["token_ids"] for item in body["items"])


# --------------------------------------------------------------------------- attach_floor


def test_attach_floor_batch_one_plan_yields_a_zero_exact_floor():
    """The probe's finding, reproduced in the mock: a batch-1 rerun moves nothing."""
    runs = five_runs(batch_size=1)
    body = FP.attach_floor(runs[0], runs, F.NOISE_PLAN_ID, F.RUN_IDS, ["order"], ["hardware.gpu"])
    per = body["noise_floor"]["per_channel"]
    assert per["exact"]["floor"] == 0.0
    assert per["seqlp"]["floor"] == 0.0
    assert per["topk"]["floor"] == 0.0
    # R = 5 -> 10 pairs -> alpha_single = 1/11 (section 5.1 step 3)
    assert per["exact"]["runs"] == 5
    assert per["exact"]["pairs"] == 10
    assert per["exact"]["alpha_single"] == 1 / 11
    assert per["exact"]["distances"] == [0.0] * 10
    assert body["noise_floor"]["plan"] == F.NOISE_PLAN_ID
    assert body["noise_floor"]["runs"] == list(F.RUN_IDS)
    assert body["noise_floor"]["covers"] == ["order"]
    assert body["noise_floor"]["not_covered"] == ["hardware.gpu"]


def test_attach_floor_batch_eight_plan_yields_a_positive_exact_floor():
    """The same battery under a plan that varies batch size: a floor on a different quantity."""
    runs = five_runs(batch_size=8)
    body = FP.attach_floor(
        runs[0], runs, F.NOISE_PLAN_ID, F.RUN_IDS, ["order", "batch_size"], ["hardware.gpu"]
    )
    per = body["noise_floor"]["per_channel"]
    # 2 of the 8 items are nuisance items and flip when the batch changes their parity:
    # exact = 1 - 6/8 = 0.25
    assert per["exact"]["floor"] == 0.25
    assert per["seqlp"]["floor"] > 0.0
    assert per["topk"]["floor"] > 0.0
    assert max(per["exact"]["distances"]) == 0.25


def test_attach_floor_channel_absent_on_one_run_gets_no_floor():
    runs = five_runs(batch_size=1)
    blind = five_runs(batch_size=1, logprobs=False)
    mixed = runs[:4] + [blind[4]]
    body = FP.attach_floor(runs[0], mixed, F.NOISE_PLAN_ID, F.RUN_IDS, ["order"], [])
    per = body["noise_floor"]["per_channel"]
    assert set(per) == {"exact"}


def test_attach_floor_does_not_touch_the_body_it_was_given():
    runs = five_runs(batch_size=1)
    before = canonical_bytes(runs[0])
    body = FP.attach_floor(runs[0], runs, F.NOISE_PLAN_ID, F.RUN_IDS, ["order"], [])
    assert canonical_bytes(runs[0]) == before
    assert "noise_floor" not in runs[0]
    assert "noise_floor" in body


def test_attach_floor_body_with_a_sensitivity_ref_checks_as_a_cert():
    runs = five_runs(batch_size=1)
    body = FP.attach_floor(runs[0], runs, F.NOISE_PLAN_ID, F.RUN_IDS, ["order"], ["hardware.gpu"])
    body["sensitivity"] = F.SENSITIVITY_ID
    refs = [
        {"role": "battery", "id": F.BATTERY_ID},
        {"role": "noise_plan", "id": F.NOISE_PLAN_ID},
        {"role": "sensitivity", "id": F.SENSITIVITY_ID},
    ] + [{"role": "run", "id": rid} for rid in F.RUN_IDS]
    cert = F.make_cert(
        "fingerprint",
        subject=F.weights_subject(),
        recipe=F.recipe(decoding=F.decoding(batch_size=1)),
        body=body,
        refs=refs,
    )
    outcome = certmod.check(cert)
    assert outcome.ok, outcome.reasons
    # every id inside the floor block is a ref, so a stranger can resolve the plan and the runs
    assert certmod.embedded_ids(cert) == {F.BATTERY_ID, F.NOISE_PLAN_ID, F.SENSITIVITY_ID, *F.RUN_IDS}


def test_attach_floor_takes_r_minus_one_ids_as_the_runs_after_the_canonical():
    """Section 5.1 step 5: the R-1 non-canonical runs are logged, then the canonical carries the
    floor.  At that moment the canonical run has no cert of its own -- it is the cert being
    built -- so the block names R-1 runs and the floor is still over all R bodies."""
    runs = five_runs(batch_size=1)
    later = list(F.RUN_IDS[1:])
    body = FP.attach_floor(runs[0], runs, F.NOISE_PLAN_ID, later, ["order"], ["hardware.gpu"])
    block = body["noise_floor"]
    assert block["runs"] == later and len(block["runs"]) == len(runs) - 1
    # the floor itself is unchanged: it is a function of the R bodies, not of how many were logged
    full = FP.attach_floor(runs[0], runs, F.NOISE_PLAN_ID, F.RUN_IDS, ["order"], ["hardware.gpu"])
    assert block["per_channel"] == full["noise_floor"]["per_channel"]
    assert block["per_channel"]["exact"]["runs"] == 5
    assert block["per_channel"]["exact"]["pairs"] == 10


def test_attach_floor_carries_the_section_5_7_overall_size_with_its_two_rules():
    runs = five_runs(batch_size=1)
    block = FP.attach_floor(
        runs[0], runs, F.NOISE_PLAN_ID, list(F.RUN_IDS[1:]), ["order"], ["hardware.gpu"]
    )["noise_floor"]
    assert block["standardization"] == FP.STANDARDIZATION
    assert block["alpha_overall_method"] == FP.ALPHA_OVERALL_METHOD
    assert [row["run"] for row in block["standardized_max"]] == [0, 1, 2, 3, 4]
    # At batch 1 the mock moves nothing under a reordering, so every floor is 0.0 and every
    # distance to the reference is 0.0.  Section 5.7 makes a zero-floor channel an exceedance
    # only when its distance is above 0, so no run exceeds and the measured fraction is 0 of 5.
    assert all(b["floor"] == 0.0 for b in block["per_channel"].values())
    assert all(row["exceeds"] is False for row in block["standardized_max"])
    assert block["alpha_overall"] == 0.0


def test_a_zero_floor_never_reaches_a_division_so_the_block_stays_appendable():
    """A floor of 0 on every channel is the ordinary outcome of greedy batch 1, and section 5.7
    handles it by an exceedance test on the distance, not by dividing by it.  JCS forbids
    Infinity and NaN (section 2.1), so a ratio taken against a zero floor would make the
    canonical fingerprint unappendable; these bytes canonicalize."""
    runs = five_runs(batch_size=1)
    block = FP.attach_floor(
        runs[0], runs, F.NOISE_PLAN_ID, list(F.RUN_IDS[1:]), ["order"], ["hardware.gpu"]
    )["noise_floor"]
    assert all(b["floor"] == 0.0 for b in block["per_channel"].values())
    text = canonical_bytes(block).decode("utf-8")
    assert "Infinity" not in text and "NaN" not in text
    assert all(row["standardized_max"] == 0.0 for row in block["standardized_max"])


def test_the_overall_size_is_counted_over_runs_when_a_channel_has_a_width():
    """One run moved: the ``exact`` floor is the max over the pairs, so it is above 0 and the
    standardized maxima are ordinary finite ratios.  No run exceeds a max taken over its own
    pairs, which is what makes the recorded fraction a measurement rather than an assertion."""
    runs = five_runs(batch_size=1)
    moved = five_runs(batch_size=1, drift={"i00"})
    mixed = runs[:4] + [moved[4]]
    block = FP.attach_floor(
        runs[0], mixed, F.NOISE_PLAN_ID, list(F.RUN_IDS[1:]), ["order"], ["hardware.gpu"]
    )["noise_floor"]
    assert block["per_channel"]["exact"]["floor"] == 0.125  # 1 of 8 items moved
    rows = block["standardized_max"]
    assert all(isinstance(row["standardized_max"], float) for row in rows)
    assert rows[4]["standardized_max"] == 1.0  # d(ref, run4) is exactly the floor
    assert all(row["exceeds"] is False for row in rows)
    assert block["alpha_overall"] == 0.0


def test_attach_floor_refuses_a_non_canonical_run():
    runs = five_runs(batch_size=1)
    with pytest.raises(ValueError, match="run_index 0"):
        FP.attach_floor(runs[1], runs, F.NOISE_PLAN_ID, F.RUN_IDS, ["order"], [])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        # 4 for 5 is now the section 5.1 step 5 shape (the canonical run's own cert is the one
        # being built); 3 for 5 is still an id list that does not describe the run set.
        ({"run_ids": F.RUN_IDS[:3]}, "3 run ids for 5 run bodies"),
        ({"run_ids": [F.RUN_IDS[0]] * 5}, "repeats a cert id"),
        ({"plan_id": "not-an-id"}, "plan_id"),
        ({"run_ids": ["nope"] * 5}, "run_ids"),
        ({"covers": ["order"], "not_covered": ["order"]}, "both covered and not covered"),
        ({"covers": ["order", "order"]}, "repeats a name"),
    ],
)
def test_attach_floor_refuses_a_malformed_floor_block(kwargs, match):
    runs = five_runs(batch_size=1)
    call = {
        "plan_id": F.NOISE_PLAN_ID,
        "run_ids": list(F.RUN_IDS),
        "covers": ["order"],
        "not_covered": ["hardware.gpu"],
    }
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        FP.attach_floor(runs[0], runs, call["plan_id"], call["run_ids"], call["covers"], call["not_covered"])


def test_attach_floor_refuses_a_run_set_with_no_shared_channel():
    runs = five_runs(batch_size=1)
    broken = copy.deepcopy(runs)
    for body in broken:
        body["channels"] = {}
    with pytest.raises(ValueError, match="no channel is present on every run"):
        FP.attach_floor(broken[0], broken, F.NOISE_PLAN_ID, F.RUN_IDS, ["order"], [])


def test_attach_floor_scores_the_exact_channel_over_item_and_canary_roles_only():
    """Appendix B: anchors are counted separately, so an anchor flip does not move the floor."""
    battery = pool_cert(8, roles={"i02": "anchor", "i05": "anchor"})
    runner = MockRunner(nuisance_items={"i02", "i05"})
    recipe = F.recipe(decoding=F.decoding(batch_size=8))
    runs = [
        FP.run_fingerprint(runner, F.weights_subject(), recipe, battery, run_index=k, nuisance={}, order=o)
        for k, o in enumerate(PERMUTED_ORDERS)
    ]
    body = FP.attach_floor(runs[0], runs, F.NOISE_PLAN_ID, F.RUN_IDS, ["order", "batch_size"], [])
    # the only items that move are the two anchors -> the exact floor stays 0.0
    assert body["noise_floor"]["per_channel"]["exact"]["floor"] == 0.0
    # while the log-prob channels, which do not read roles, still see the movement
    assert body["noise_floor"]["per_channel"]["seqlp"]["floor"] > 0.0


# --------------------------------------------------------------------------- the plan, applied
#
# The first real floor (`papers/v8/vacuous_floor_2026_09_09/`) declared batch_size 1|8|32 and
# item_order canonical|perm11|perm12, then ran five times at batch_size 1 -- where item order
# cannot move anything, because at batch 1 each item is its own forward pass. Every pairwise
# distance was 0.0, so the floor was 0.0 on exact, seqlp and topk, and every later difference
# exceeded it. These tests pin the half of the repair that makes the runs take the plan.

PLAN_BODY = {
    "kind": "noise-plan",
    "runs": 5,
    "nuisance": [
        {"factor": "batch_size", "values": ["1", "8", "32"]},
        {"factor": "item_order", "values": ["canonical", "perm11", "perm12"]},
    ],
}
EIGHT_IDS = [f"i{k:02d}" for k in range(8)]


def settings_for(plan_body=None, runs: int = 5, recipe=None, item_ids=None) -> list[dict]:
    return FP.plan_run_settings(
        PLAN_BODY if plan_body is None else plan_body,
        runs,
        F.recipe() if recipe is None else recipe,
        EIGHT_IDS if item_ids is None else item_ids,
    )


def test_the_five_runs_of_a_plan_take_five_different_assignments():
    """The defect, from the other side: R runs under a plan vary every factor the plan declared,
    and no two of the five take the same assignment (the plan's product is 3 x 3 = 9 >= 5)."""
    settings = settings_for()
    assignments = [s["assignment"] for s in settings]
    assert len(assignments) == 5
    assert len({tuple(sorted(a.items())) for a in assignments}) == 5
    assert len({a["batch_size"] for a in assignments}) > 1
    assert len({a["item_order"] for a in assignments}) > 1
    # the enumeration rule, spelled out: the diagonal head first, then the rest of the product
    assert assignments[:3] == [
        {"batch_size": "1", "item_order": "canonical"},
        {"batch_size": "8", "item_order": "perm11"},
        {"batch_size": "32", "item_order": "perm12"},
    ]
    assert assignments[3:] == [
        {"batch_size": "1", "item_order": "perm11"},
        {"batch_size": "1", "item_order": "perm12"},
    ]


def test_run_zero_is_the_recipe_as_written_in_a3_order():
    """Section 5.1 step 2: the reference run is batch 1 in A.3 order. Run 0 takes the first value
    of every factor, so it is the recipe the command was given, unpermuted."""
    first = settings_for()[0]
    assert first["order"] is None  # A.3 order
    assert first["recipe"]["decoding"]["batch_size"] == F.recipe()["decoding"]["batch_size"]
    assert first["nuisance"] == {"batch_size": 1, "item_order": "canonical"}


def test_each_run_carries_the_recipe_it_actually_ran_under():
    """A cert whose recipe says batch 1 while the run was batched is a cert that lies about its
    own execution, so the plan's batch size is written into the run's recipe."""
    settings = settings_for()
    assert [s["recipe"]["decoding"]["batch_size"] for s in settings] == [1, 8, 32, 1, 1]
    assert [s["nuisance"]["batch_size"] for s in settings] == [1, 8, 32, 1, 1]
    # and `build` refuses a nuisance block that contradicts the recipe, so the two cannot drift
    body = FP.build(
        F.weights_subject(), settings[1]["recipe"], pool_cert(3),
        three_results(), run_index=1, nuisance=dict(settings[1]["nuisance"]),
    )
    assert body["nuisance"]["batch_size"] == 8


def test_every_declared_factor_varies_by_the_second_run():
    """The diagonal head is what makes this true for every factor at once. A plain odometer over
    two three-valued factors would hold the first factor at its first value for three runs --
    the shape of the defect."""
    for runs in (2, 3, 4, 5, 9):
        assignments = [s["assignment"] for s in settings_for(runs=runs)]
        assert len({a["batch_size"] for a in assignments}) > 1, runs
        assert len({a["item_order"] for a in assignments}) > 1, runs


def test_the_enumeration_cycles_when_r_exceeds_what_the_plan_can_form():
    """Two values and five runs cannot be five different assignments; the enumeration says so by
    repeating rather than by inventing a value the plan did not declare."""
    plan = {"nuisance": [{"factor": "order", "values": ["a3", "permuted"]}]}
    assignments = [s["assignment"]["order"] for s in settings_for(plan)]
    assert assignments == ["a3", "permuted", "a3", "permuted", "a3"]


def test_a_declared_order_value_is_a_reproducible_permutation():
    settings = settings_for()
    orders = {s["assignment"]["item_order"]: s["order"] for s in settings}
    assert orders["canonical"] is None
    assert sorted(orders["perm11"]) == EIGHT_IDS and orders["perm11"] != EIGHT_IDS
    assert orders["perm11"] != orders["perm12"]
    # keyed on the value's text alone: the same name gives the same order, run after run
    assert orders["perm11"] == FP.order_for_value(EIGHT_IDS, "perm11")
    assert FP.order_for_value(EIGHT_IDS, "perm11") == FP.order_for_value(list(reversed(EIGHT_IDS)), "perm11")


def test_a_factor_no_runner_can_apply_is_refused_and_named():
    """Never a silent run at the default: the message names the factor the plan declared."""
    for factor in ("gpu", "driver", "region", "time_of_day"):
        plan = {"nuisance": [{"factor": factor, "values": ["a", "b"]}]}
        with pytest.raises(ValueError, match=f"cannot apply nuisance factor '{factor}'"):
            settings_for(plan)


def test_an_order_factor_whose_first_value_is_not_a3_order_is_refused():
    plan = {"nuisance": [{"factor": "item_order", "values": ["perm11", "canonical"]}]}
    with pytest.raises(ValueError, match="does not name A.3 order"):
        settings_for(plan)


def test_an_order_factor_that_names_a3_order_twice_is_refused():
    """`a3|canonical` is one order under two names: a factor that varies on paper only."""
    plan = {"nuisance": [{"factor": "order", "values": ["a3", "canonical"]}]}
    with pytest.raises(ValueError, match="names A.3 order a second time"):
        settings_for(plan)


def test_one_order_factor_under_two_names_is_refused():
    plan = {
        "nuisance": [
            {"factor": "item_order", "values": ["canonical", "p1"]},
            {"factor": "order", "values": ["a3", "p2"]},
        ]
    }
    with pytest.raises(ValueError, match="one item-order factor under two names"):
        settings_for(plan)


def test_an_order_factor_a_small_battery_cannot_realize_is_refused():
    """Two items have two orders. A plan asking for three is a plan the battery cannot honour,
    and two runs recording different value names while running the same order is exactly the
    thing this repair exists to stop."""
    plan = {"nuisance": [{"factor": "order", "values": ["a3", "p1", "p2"]}]}
    with pytest.raises(ValueError, match="realize the same order"):
        settings_for(plan, item_ids=["i00", "i01"])


def test_a_batch_size_whose_first_value_is_not_the_recipes_own_is_refused():
    plan = {"nuisance": [{"factor": "batch_size", "values": ["8", "1"]}]}
    with pytest.raises(ValueError, match="while the recipe runs at 1"):
        settings_for(plan)
    # and the same plan against a recipe that does run at 8 is fine
    settings = settings_for(plan, recipe=F.recipe(decoding=F.decoding(batch_size=8)))
    assert [s["nuisance"]["batch_size"] for s in settings] == [8, 1, 8, 1, 8]


@pytest.mark.parametrize(
    "values, match",
    [
        (["1", "eight"], "is not an integer"),
        (["1", "0"], "is not positive"),
    ],
)
def test_a_batch_size_value_that_is_not_a_batch_size_is_refused(values, match):
    with pytest.raises(ValueError, match=match):
        settings_for({"nuisance": [{"factor": "batch_size", "values": values}]})


def test_padding_side_is_applied_and_bounded():
    plan = {"nuisance": [{"factor": "padding_side", "values": ["left", "right"]}]}
    settings = settings_for(plan)
    assert [s["recipe"]["decoding"]["padding_side"] for s in settings] == [
        "left", "right", "left", "right", "left"
    ]
    assert settings[1]["nuisance"]["padding_side"] == "right"
    with pytest.raises(ValueError, match="is not left or right"):
        settings_for({"nuisance": [{"factor": "padding_side", "values": ["left", "middle"]}]})


def test_a_plan_that_declares_no_factors_leaves_every_run_at_the_recipe():
    settings = settings_for({"kind": "noise-plan", "runs": 5})
    assert [s["assignment"] for s in settings] == [{}] * 5
    assert all(s["order"] is None and s["nuisance"] == {} for s in settings)


def test_plan_factors_reads_the_declaration_and_refuses_a_malformed_one():
    assert FP.plan_factors(PLAN_BODY) == [
        ("batch_size", ["1", "8", "32"]),
        ("item_order", ["canonical", "perm11", "perm12"]),
    ]
    with pytest.raises(ValueError, match="has no values"):
        FP.plan_factors({"nuisance": [{"factor": "order", "values": []}]})
    with pytest.raises(ValueError, match="repeats a value"):
        FP.plan_factors({"nuisance": [{"factor": "order", "values": ["a3", "a3"]}]})
    with pytest.raises(ValueError, match="declared more than once"):
        FP.plan_factors(
            {"nuisance": [{"factor": "o", "values": ["a"]}, {"factor": "o", "values": ["b"]}]}
        )


def test_assignment_sequence_is_a_permutation_of_the_product():
    factors = [("a", ["0", "1", "2"]), ("b", ["x", "y"]), ("c", ["p", "q", "r", "s"])]
    total = 3 * 2 * 4
    seq = FP.assignment_sequence(factors, total)
    tuples = [tuple(a[name] for name, _ in factors) for a in seq]
    assert len(set(tuples)) == total  # every assignment exactly once, none invented
    assert set(tuples) == {(a, b, c) for a in "012" for b in "xy" for c in "pqrs"}
    assert tuples[0] == ("0", "x", "p")  # run 0 is the first value of every factor
    assert FP.assignment_sequence(factors, total + 3)[total:] == seq[:3]  # then it cycles


def test_the_floor_of_a_plan_that_was_applied_is_not_the_floor_of_one_that_was_not():
    """End to end on the mock, in one test: the same plan, applied and ignored.

    The mock moves ``nuisance_items`` only at batch_size != 1 (the probe's finding: a batch-1
    rerun moved nothing, a batch-size change alone moved outputs). So five runs that honour
    ``batch_size 1|8|32`` have a floor above 0, and five that pin batch 1 and permute the order
    have a floor of exactly 0.0 on every channel -- against which every later difference
    exceeds.
    """
    runner = MockRunner(nuisance_items={"i02", "i05"})
    battery = pool_cert(8)
    settings = settings_for(item_ids=sorted(FP.battery_item_map(battery)))
    applied = [
        FP.run_fingerprint(
            runner, F.weights_subject(), s["recipe"], battery,
            run_index=k, nuisance=dict(s["nuisance"]), order=s["order"],
        )
        for k, s in enumerate(settings)
    ]
    assert [b["nuisance"]["batch_size"] for b in applied] == [1, 8, 32, 1, 1]
    floor = FP.attach_floor(
        applied[0], applied, F.NOISE_PLAN_ID, F.RUN_IDS, ["batch_size", "item_order"], []
    )["noise_floor"]["per_channel"]
    assert floor["exact"]["floor"] > 0.0
    assert floor["seqlp"]["floor"] > 0.0

    ignored = five_runs(batch_size=1)  # the plan declared batch size; these runs did not take it
    assert [b["nuisance"]["batch_size"] for b in ignored] == [1] * 5
    zero = FP.attach_floor(
        ignored[0], ignored, F.NOISE_PLAN_ID, F.RUN_IDS, ["batch_size", "item_order"], []
    )["noise_floor"]["per_channel"]
    assert {c: b["floor"] for c, b in zero.items()} == {"exact": 0.0, "seqlp": 0.0, "topk": 0.0}


# --------------------------------------------------------------------------- run_fingerprint


def test_run_fingerprint_defaults_to_the_a3_order():
    body = FP.run_fingerprint(
        MockRunner(), F.weights_subject(), F.recipe(), pool_cert(3), run_index=0, nuisance={}
    )
    assert [i["item_id"] for i in body["items"]] == ["i00", "i01", "i02"]
    assert body["nuisance"]["item_order_sha256"] == FP.order_sha256(["i00", "i01", "i02"])


def test_run_fingerprint_permutation_moves_the_order_hash_and_not_the_battery_hash():
    runner = MockRunner(nuisance_items={"i01"})
    battery = pool_cert(3)
    canonical = FP.run_fingerprint(
        runner, F.weights_subject(), F.recipe(), battery, run_index=0, nuisance={}
    )
    permuted = FP.run_fingerprint(
        runner, F.weights_subject(), F.recipe(), battery, run_index=1, nuisance={},
        order=["i02", "i01", "i00"],
    )
    assert [i["item_id"] for i in permuted["items"]] == ["i02", "i01", "i00"]
    assert permuted["nuisance"]["item_order_sha256"] != canonical["nuisance"]["item_order_sha256"]
    # at batch 1 the order is not a nuisance the mock responds to, and the A.3 hash never was
    assert permuted["channels"]["exact"]["hash"] == canonical["channels"]["exact"]["hash"]


def test_run_fingerprint_sees_a_drifted_subject_as_a_different_battery_hash():
    battery = pool_cert(3)
    base = FP.run_fingerprint(
        MockRunner(), F.weights_subject(), F.recipe(), battery, run_index=0, nuisance={}
    )
    drifted = FP.run_fingerprint(
        MockRunner(drift_items={"i01"}), F.weights_subject(), F.recipe(), battery,
        run_index=0, nuisance={},
    )
    assert drifted["channels"]["exact"]["hash"] != base["channels"]["exact"]["hash"]
    assert drifted["items"][0]["token_ids_sha256"] == base["items"][0]["token_ids_sha256"]
    assert drifted["items"][1]["token_ids_sha256"] != base["items"][1]["token_ids_sha256"]


def test_run_fingerprint_fills_gpu_and_driver_from_the_runner_environment():
    body = FP.run_fingerprint(
        MockRunner(), F.weights_subject(), F.recipe(), pool_cert(3), run_index=0, nuisance={}
    )
    assert body["nuisance"]["gpu"] == "none"
    assert body["nuisance"]["driver"] == "none"


def test_run_fingerprint_keeps_the_hardware_the_caller_declared():
    body = FP.run_fingerprint(
        MockRunner(), F.weights_subject(), F.recipe(), pool_cert(3), run_index=0,
        nuisance={"gpu": "A100", "driver": "550.54"},
    )
    assert body["nuisance"]["gpu"] == "A100"


@pytest.mark.parametrize(
    "order, match",
    [
        (["i00", "i01"], "permutation"),
        (["i00", "i01", "i99"], "permutation"),
        (["i00", "i00", "i01"], "repeats an item_id"),
    ],
)
def test_run_fingerprint_refuses_an_order_that_is_not_a_permutation(order, match):
    with pytest.raises(ValueError, match=match):
        FP.run_fingerprint(
            MockRunner(), F.weights_subject(), F.recipe(), pool_cert(3), run_index=0,
            nuisance={}, order=order,
        )


def test_run_fingerprint_refuses_a_battery_whose_prompts_are_withheld():
    with pytest.raises(ValueError, match="cannot be run"):
        FP.run_fingerprint(
            MockRunner(), F.weights_subject(), F.recipe(), pool_cert(3, redacted=True),
            run_index=0, nuisance={},
        )


def test_run_fingerprint_refuses_a_runner_that_reorders_its_results():
    class Reversing:
        def run(self, items, recipe, subject):
            return list(reversed(MockRunner().run(items, recipe, subject)))

        def environment(self):
            return MockRunner().environment()

        def subject(self, requested):
            return MockRunner().subject(requested)

    with pytest.raises(ValueError, match="one result per item in the order given"):
        FP.run_fingerprint(
            Reversing(), F.weights_subject(), F.recipe(), pool_cert(3), run_index=0, nuisance={}
        )


def test_run_fingerprint_bodies_are_reproducible():
    args = (MockRunner(), F.weights_subject(), F.recipe(), pool_cert(3))
    a = FP.run_fingerprint(*args, run_index=0, nuisance={})
    b = FP.run_fingerprint(*args, run_index=0, nuisance={})
    assert canonical_bytes(a) == canonical_bytes(b)


# --------------------------------------------------------------------------- properties

_ids = st.lists(
    st.text(alphabet="abcdefghij0123456789", min_size=1, max_size=4),
    min_size=1,
    max_size=6,
    unique=True,
)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(ids=_ids, data=st.data())
def test_property_exact_hash_ignores_the_order_of_its_input(ids, data):
    records = [
        FP.item_record(result(iid, [k + 1, k + 2], f"t{k}", logprobs=False), False)
        for k, iid in enumerate(ids)
    ]
    shuffled = data.draw(st.permutations(records))
    assert FP.exact_hash(shuffled) == FP.exact_hash(records)


@settings(max_examples=60, deadline=None)
@given(
    token_ids=st.lists(st.integers(min_value=0, max_value=2**31 - 1), min_size=1, max_size=12),
    text=st.text(max_size=32),
)
def test_property_item_record_hashes_are_the_appendix_a3_preimages(token_ids, text):
    rec = FP.item_record(
        {
            "item_id": "i00",
            "token_ids": token_ids,
            "output_text": text,
            "n_generated": len(token_ids),
            "seq_logprob": None,
            "topk": None,
        },
        False,
    )
    assert rec["token_ids_sha256"] == hashlib.sha256(canonical_bytes(token_ids)).hexdigest()
    assert rec["output_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert len(rec["token_ids_sha256"]) == 64
    assert FP.exact_hash([rec]) == hashlib.sha256(bytes.fromhex(rec["token_ids_sha256"])).hexdigest()


@settings(max_examples=40, deadline=None)
@given(ids=_ids)
def test_property_order_sha256_separates_orders(ids):
    forward = FP.order_sha256(ids)
    assert len(forward) == 64
    if len(ids) > 1:
        assert FP.order_sha256(list(reversed(ids))) != forward
    assert forward == hashlib.sha256(canonical_bytes(list(ids))).hexdigest()


def test_no_channel_block_is_ever_zero_filled():
    """Section 3.2: an absent channel says so; it never carries a fabricated number."""
    body = FP.build(
        F.weights_subject(), F.recipe(), pool_cert(3), three_results(logprobs=False),
        run_index=0, nuisance={},
    )
    assert body["channels"]["seqlp"] == {"present": False}
    assert body["channels"]["topk"] == {"present": False}
    assert all(item["seq_logprob"] is None for item in body["items"])
    assert all(item["topk"] is None for item in body["items"])
    # the absent block carries the flag and nothing else -- no zero, no empty list
    assert list(body["channels"]["seqlp"]) == ["present"]
    assert list(body["channels"]["topk"]) == ["present"]


def test_the_floor_and_the_distance_are_in_the_same_units():
    """A floor computed by floor.pairwise is comparable to a distance from the same function."""
    from styxx.v8 import distances as D

    runs = five_runs(batch_size=8)
    body = FP.attach_floor(runs[0], runs, F.NOISE_PLAN_ID, F.RUN_IDS, ["order", "batch_size"], [])
    roles = {i["item_id"]: i["role"] for i in runs[0]["items"]}
    d, anchor_flips = D.exact(runs[0]["items"], runs[1]["items"], roles)
    assert anchor_flips == 0
    assert D.rounded(d) in body["noise_floor"]["per_channel"]["exact"]["distances"]
    assert D.rounded(d) <= body["noise_floor"]["per_channel"]["exact"]["floor"]
    assert math.isfinite(body["noise_floor"]["per_channel"]["seqlp"]["floor"])
