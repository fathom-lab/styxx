"""tests/test_v8_sweep.py -- the delta sweep record (styxx/v8/sweep.py, spec section 4.2).

What the sweep owes its consumer:

* one JSON-serializable record with the reference pass and every delta family in it;
* determinism -- the item order comes from the item ids alone, so two sweeps against the same
  runner factory produce byte-identical records;
* honest refusal on a malformed pool or a malformed delta configuration, never a silent
  best-effort run that scores a battery on a sweep nobody can reconstruct.

One property of the mock the tests pin because a reader will otherwise be puzzled by it:
``MockRunner`` derives its output from (subject identity, item_id, recipe_core) and NOT from
``prompt_text``.  A delta4 template jitter therefore cannot move it, and every ``flip4`` in
these tests is 0.0.  The delta4 coverage here is structural (shape, ordering, refusals); the
behavioural half of delta4 waits for a runner whose output depends on the prompt bytes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from styxx.v8 import sweep
from styxx.v8.jcs import canonical_bytes, sha256_hex
from styxx.v8.runner import MockRunner
from tests import v8_fixtures as F

FAMILY_CYCLE = ("recall", "format", "short-reasoning", "instruction-following", "refusal-boundary")


def pool(n: int = 8) -> list[dict]:
    return [
        {
            "item_id": f"i{k:02d}",
            "prompt_text": f"prompt {k}",
            "family": FAMILY_CYCLE[k % len(FAMILY_CYCLE)],
        }
        for k in range(n)
    ]


def factory(**mock_kwargs):
    """A runner factory that hands every precision the same mock configuration."""
    def make(precision):
        return MockRunner(**mock_kwargs)
    return make


DELTA2 = [
    {"batch_size": 1, "order": "canonical"},
    {"batch_size": 8, "order": "canonical"},
    {"batch_size": 8, "order": "perm", "perm_seed": 11},
]


# --------------------------------------------------------------------------- ordering

def test_a3_order_is_utf8_byte_order_and_ignores_the_given_order():
    items = [{"item_id": "b"}, {"item_id": "A"}, {"item_id": "a"}, {"item_id": "B"}]
    assert [it["item_id"] for it in sweep.a3_order(items)] == ["A", "B", "a", "b"]
    again = sweep.a3_order(list(reversed(items)))
    assert [it["item_id"] for it in again] == ["A", "B", "a", "b"]


def test_item_order_sha256_is_the_newline_join_of_the_ids():
    items = sweep.a3_order(pool(3))
    expected = sha256_hex("i00\ni01\ni02".encode("utf-8"))
    assert sweep.item_order_sha256(items) == expected
    assert len(expected) == 64


def test_pool_sha256_is_order_independent_and_moves_with_the_prompt():
    a = sweep.pool_sha256(pool(4))
    b = sweep.pool_sha256(list(reversed(pool(4))))
    assert a == b
    changed = pool(4)
    changed[2]["prompt_text"] = "prompt 2 "
    assert sweep.pool_sha256(changed) != a


# --------------------------------------------------------------------------- record shape

def test_record_shape_and_params():
    rec = sweep.run_sweep(
        factory(), pool(6), F.recipe(), F.weights_subject(),
        delta1=["fp16", "int8-bnb"], delta2=DELTA2,
        delta4=[{"id": "trailing-space", "prompt_suffix": " "}],
    )
    assert set(rec) == {"reference", "delta1", "delta2", "delta4", "params"}
    assert sorted(rec["reference"]) == [f"i{k:02d}" for k in range(6)]
    assert sorted(rec["delta1"]) == ["fp16", "int8-bnb"]
    for block in rec["delta1"].values():
        assert sorted(block) == sorted(rec["reference"])
    assert len(rec["delta2"]) == 3
    assert len(rec["delta4"]) == 1
    for entry in rec["delta2"] + rec["delta4"]:
        assert set(entry) >= {"config", "results"}
        assert sorted(entry["results"]) == sorted(rec["reference"])

    params = rec["params"]
    assert params["record"] == sweep.RECORD_TAG
    assert params["reference"] == {"batch_size": 1, "order": "canonical", "precision": "bf16"}
    assert params["delta1"] == ["fp16", "int8-bnb"]
    assert params["delta4_run"] is True
    assert params["pool_size"] == 6
    assert params["pool_sha256"] == sweep.pool_sha256(pool(6))
    assert params["item_order_sha256"] == sweep.item_order_sha256(sweep.a3_order(pool(6)))


def test_record_is_json_serializable_and_deterministic():
    args = (factory(nuisance_items={"i01", "i04"}), pool(6), F.recipe(), F.weights_subject())
    kwargs = dict(delta1=["fp16"], delta2=DELTA2, delta4=[{"prompt_suffix": "\n"}])
    one = sweep.run_sweep(*args, **kwargs)
    two = sweep.run_sweep(*args, **kwargs)
    assert canonical_bytes(one) == canonical_bytes(two)
    # A plain json round trip must survive too -- the CLI reads a record off disk.
    assert json.loads(json.dumps(one)) == one


def test_run_sweep_does_not_mutate_the_caller_recipe_or_subject():
    recipe = F.recipe()
    subject = F.weights_subject()
    before = (canonical_bytes(recipe), canonical_bytes(subject))
    sweep.run_sweep(factory(), pool(4), recipe, subject, delta1=["fp16"], delta2=DELTA2)
    assert (canonical_bytes(recipe), canonical_bytes(subject)) == before
    assert recipe["decoding"]["batch_size"] == 1


# --------------------------------------------------------------------------- delta families

def test_reference_is_batch_one_and_a_batch_one_delta2_reproduces_it_exactly():
    """The probe's finding in the mock: a rerun at batch 1 moves nothing, in any order."""
    rec = sweep.run_sweep(
        factory(nuisance_items={"i00", "i03"}), pool(6), F.recipe(), F.weights_subject(),
        delta1=[], delta2=[{"batch_size": 1, "order": "canonical"},
                           {"batch_size": 1, "order": "perm", "perm_seed": 3}],
    )
    for entry in rec["delta2"]:
        assert entry["results"] == rec["reference"], entry["config"]


def test_delta2_batch_change_moves_only_the_nuisance_items():
    rec = sweep.run_sweep(
        factory(nuisance_items={"i00", "i03"}), pool(6), F.recipe(), F.weights_subject(),
        delta1=[], delta2=[{"batch_size": 8, "order": "canonical"}],
    )
    moved = {
        iid for iid, r in rec["delta2"][0]["results"].items()
        if r["token_ids"] != rec["reference"][iid]["token_ids"]
    }
    assert moved == {"i00", "i03"}


def test_delta2_perm_is_a_seeded_permutation_of_the_a3_order():
    rec = sweep.run_sweep(
        factory(), pool(8), F.recipe(), F.weights_subject(),
        delta1=[], delta2=[{"batch_size": 8, "order": "canonical"},
                           {"batch_size": 8, "order": "perm", "perm_seed": 11},
                           {"batch_size": 8, "order": "perm", "perm_seed": 12}],
    )
    canonical, p11, p12 = rec["delta2"]
    assert canonical["order_sha256"] == rec["params"]["item_order_sha256"]
    assert p11["order_sha256"] != canonical["order_sha256"]
    assert p11["order_sha256"] != p12["order_sha256"]
    # A permutation, not a filter: the same items come back either way.
    assert sorted(p11["results"]) == sorted(canonical["results"])


def test_delta1_moves_only_the_items_that_are_sensitive_to_that_precision():
    def make(precision):
        return MockRunner(precision_items={"fp16": {"i02"}, "int8-bnb": {"i02", "i05"}})

    rec = sweep.run_sweep(
        make, pool(6), F.recipe(), F.weights_subject(),
        delta1=["fp16", "int8-bnb"], delta2=DELTA2,
    )
    def moved(block):
        return {iid for iid, r in block.items() if r["token_ids"] != rec["reference"][iid]["token_ids"]}

    assert moved(rec["delta1"]["fp16"]) == {"i02"}
    assert moved(rec["delta1"]["int8-bnb"]) == {"i02", "i05"}


def test_delta4_records_the_jitter_config_and_leaves_the_mock_unmoved():
    rec = sweep.run_sweep(
        factory(), pool(4), F.recipe(), F.weights_subject(),
        delta1=[], delta2=DELTA2,
        delta4=[{"id": "trailing-space", "prompt_suffix": " "},
                {"prompt_prefix": "\n"}],
    )
    assert [e["config"]["id"] for e in rec["delta4"]] == ["trailing-space", "jitter-1"]
    assert rec["delta4"][0]["config"]["prompt_suffix"] == " "
    assert rec["delta4"][1]["config"]["prompt_prefix"] == "\n"
    # MockRunner does not read prompt_text (module docstring), so nothing moves here.
    for entry in rec["delta4"]:
        assert entry["results"] == rec["reference"]


def test_alias_subject_without_logprobs_still_sweeps_when_delta1_is_empty():
    rec = sweep.run_sweep(
        factory(logprobs=False), pool(4), F.recipe(), F.alias_subject(),
        delta1=[], delta2=[{"batch_size": 8, "order": "canonical"}],
    )
    assert rec["params"]["reference"]["precision"] is None
    assert all(r["seq_logprob"] is None for r in rec["reference"].values())


# --------------------------------------------------------------------------- refusals

def test_delta1_refuses_the_reference_precision_and_duplicates():
    with pytest.raises(ValueError, match="reference precision"):
        sweep.run_sweep(factory(), pool(3), F.recipe(), F.weights_subject(),
                        delta1=["bf16"], delta2=DELTA2)
    with pytest.raises(ValueError, match="duplicate precision"):
        sweep.run_sweep(factory(), pool(3), F.recipe(), F.weights_subject(),
                        delta1=["fp16", "fp16"], delta2=DELTA2)


def test_delta1_refuses_a_subject_with_no_precision():
    with pytest.raises(ValueError, match="reference precision"):
        sweep.run_sweep(factory(), pool(3), F.recipe(), F.alias_subject(),
                        delta1=["fp16"], delta2=DELTA2)


@pytest.mark.parametrize("cfg, match", [
    ({"batch_size": 0, "order": "canonical"}, "batch_size"),
    ({"batch_size": 8, "order": "sideways"}, "order"),
    ({"batch_size": 8, "order": "perm"}, "perm_seed"),
    ({"batch_size": 8, "order": "canonical", "perm_seed": 3}, "meaningless"),
    ({"batch_size": 8, "device": 1, "gpu": "a"}, "unknown keys"),
])
def test_delta2_configuration_refusals(cfg, match):
    with pytest.raises(ValueError, match=match):
        sweep.run_sweep(factory(), pool(3), F.recipe(), F.weights_subject(),
                        delta1=[], delta2=[cfg])


@pytest.mark.parametrize("cfg, match", [
    ({"id": "x"}, "prompt_prefix or prompt_suffix"),
    ({"prompt_suffix": 3}, "must be a string"),
    ({"whitespace": " "}, "unknown keys"),
])
def test_delta4_configuration_refusals(cfg, match):
    with pytest.raises(ValueError, match=match):
        sweep.run_sweep(factory(), pool(3), F.recipe(), F.weights_subject(),
                        delta1=[], delta2=DELTA2, delta4=[cfg])


@pytest.mark.parametrize("items, match", [
    ([], "empty"),
    ([{"item_id": "a", "prompt_text": "p"}, {"item_id": "a", "prompt_text": "q"}], "duplicate"),
    ([{"item_id": "", "prompt_text": "p"}], "non-empty string"),
    ([{"item_id": "a"}], "prompt_text"),
    ([["a", "p"]], "not a mapping"),
])
def test_pool_refusals(items, match):
    with pytest.raises(ValueError, match=match):
        sweep.run_sweep(factory(), items, F.recipe(), F.weights_subject(),
                        delta1=[], delta2=DELTA2)


def test_a_runner_that_reorders_its_results_is_refused():
    class Reordering(MockRunner):
        def run(self, items, recipe, subject):
            return list(reversed(super().run(items, recipe, subject)))

    with pytest.raises(ValueError, match="out of order"):
        sweep.run_sweep(lambda p: Reordering(), pool(4), F.recipe(), F.weights_subject(),
                        delta1=[], delta2=DELTA2)


def test_a_runner_that_drops_a_result_is_refused():
    class Dropping(MockRunner):
        def run(self, items, recipe, subject):
            return super().run(items, recipe, subject)[:-1]

    with pytest.raises(ValueError, match="results for"):
        sweep.run_sweep(lambda p: Dropping(), pool(4), F.recipe(), F.weights_subject(),
                        delta1=[], delta2=DELTA2)


def test_runner_factory_is_called_once_per_precision():
    calls: list = []

    def make(precision):
        calls.append(precision)
        return MockRunner()

    sweep.run_sweep(make, pool(4), F.recipe(), F.weights_subject(),
                    delta1=["fp16", "int8-bnb"], delta2=DELTA2,
                    delta4=[{"prompt_suffix": " "}])
    assert calls == ["bf16", "fp16", "int8-bnb"]


# --------------------------------------------------------------------------- storage

def test_write_record_is_utf8_lf_no_bom_and_reads_back(tmp_path: Path):
    rec = sweep.run_sweep(factory(), pool(4), F.recipe(), F.weights_subject(),
                          delta1=["fp16"], delta2=DELTA2)
    path = sweep.write_record(rec, tmp_path / "sweep.json")
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw
    assert raw.endswith(b"\n")
    assert sweep.read_record(path) == rec


def test_read_record_refuses_a_json_array(tmp_path: Path):
    p = tmp_path / "not-a-record.json"
    p.write_bytes(b"[1, 2, 3]")
    with pytest.raises(ValueError, match="JSON object"):
        sweep.read_record(p)
