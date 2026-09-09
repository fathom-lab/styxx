"""Tests for styxx.v8.runner: the Runner protocol and the deterministic MockRunner."""
from __future__ import annotations

import copy
import math
import subprocess
import sys
from pathlib import Path

import pytest

from styxx.v8 import fingerprint as FP
from styxx.v8.runner import ItemResult, MockRunner, Runner, core_key, identity_key

ROOT = Path(__file__).resolve().parent.parent

WEIGHTS = {
    "kind": "weights",
    "model_family": "qwen2.5",
    "hf_repo": "Qwen/Qwen2.5-0.5B-Instruct",
    "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
    "weights_sha256": "a" * 64,
    "config_sha256": "b" * 64,
    "tokenizer_sha256": "c" * 64,
    "generation_config_sha256": "d" * 64,
    "precision": "bf16",
    "environment": {
        "runtime": {"framework": "mock", "version": "0", "backend": "cpu"},
        "hardware": {"gpu": "none", "driver": "none", "count": 0},
    },
}
ALIAS = {
    "kind": "alias",
    "model_family": "unknown",
    "provider": "acme",
    "alias": "acme-large",
    "region": "eu",
}
RECIPE = {
    "battery": "sha256:" + "0" * 64,
    "decoding": {
        "temperature": 0,
        "top_p": 1.0,
        "max_new_tokens": 16,
        "stop": ["\n"],
        "seed": 7,
        "batch_size": 1,
        "padding_side": "left",
    },
    "materials": {"chat_template": "", "chat_template_source": "inline", "system_prompt": "", "env_lock": ""},
    "chat_template_sha256": "e" * 64,
    "system_prompt_sha256": "f" * 64,
    "env_lock_sha256": "9" * 64,
    "harness": {"name": "styxx", "version": "8.0.0", "commit": "deadbeef"},
}
ITEMS = [{"item_id": f"i{k:02d}", "prompt_text": f"prompt {k}"} for k in range(8)]


def recipe_with(batch_size: int | None = None, **decoding) -> dict:
    r = copy.deepcopy(RECIPE)
    if batch_size is not None:
        r["decoding"]["batch_size"] = batch_size
    r["decoding"].update(decoding)
    return r


def subject_with(**fields) -> dict:
    s = copy.deepcopy(WEIGHTS)
    s.update(fields)
    return s


def ids_of(results: list[ItemResult]) -> dict[str, list[int]]:
    return {r["item_id"]: r["token_ids"] for r in results}


# ---------------------------------------------------------------- protocol and hygiene


def test_import_does_not_touch_torch():
    code = "import sys, styxx.v8.runner; print('torch' in sys.modules, 'transformers' in sys.modules)"
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT), check=False)
    assert p.returncode == 0, p.stderr
    assert p.stdout.split() == ["False", "False"]


def test_mock_satisfies_runner_protocol():
    assert isinstance(MockRunner(), Runner)
    assert not isinstance(object(), Runner)


def test_environment_shape():
    env = MockRunner().environment()
    assert set(env) == {"runtime", "hardware"}
    assert set(env["runtime"]) == {"framework", "version", "backend"}
    assert set(env["hardware"]) == {"gpu", "driver", "count"}
    assert env["hardware"]["count"] == 0


def test_identity_key_per_kind():
    w = identity_key(WEIGHTS)
    assert w["kind"] == "weights" and "precision" not in w and "environment" not in w
    assert w["weights_sha256"] == "a" * 64
    a = identity_key(ALIAS)
    assert a == {"kind": "alias", "provider": "acme", "alias": "acme-large", "region": "eu"}
    with pytest.raises(ValueError):
        identity_key({"kind": "gguf"})


def test_core_key_drops_the_two_nuisance_decoding_fields_only():
    c = core_key(RECIPE)
    assert set(c) == {"battery", "decoding", "chat_template_sha256", "system_prompt_sha256"}
    assert "batch_size" not in c["decoding"] and "padding_side" not in c["decoding"]
    assert c["decoding"]["max_new_tokens"] == 16 and c["decoding"]["stop"] == ["\n"]
    assert core_key(recipe_with(batch_size=8)) == c
    assert core_key(recipe_with(padding_side="right")) == c


# ---------------------------------------------------------------- result shape


def test_result_shape_and_order_preserved():
    res = MockRunner().run(ITEMS, RECIPE, WEIGHTS)
    assert [r["item_id"] for r in res] == [it["item_id"] for it in ITEMS]
    for r in res:
        assert set(r) == {
            "item_id", "token_ids", "output_text", "n_generated",
            "seq_logprob", "topk", "margin_by_position", "stay",
        }
        n = r["n_generated"]
        assert n == len(r["token_ids"]) >= 1
        assert n <= RECIPE["decoding"]["max_new_tokens"]
        assert all(isinstance(t, int) and 0 <= t < 32000 for t in r["token_ids"])
        assert r["output_text"] == " ".join(f"w{t}" for t in r["token_ids"])
        assert isinstance(r["seq_logprob"], float) and r["seq_logprob"] <= 0.0
        assert isinstance(r["stay"], float) and r["stay"] <= 0.0
        assert len(r["margin_by_position"]) == n and all(m > 0 for m in r["margin_by_position"])
        assert len(r["topk"]) == min(8, n)
        for p, entry in enumerate(r["topk"]):
            assert entry["pos"] == p
            assert len(entry["ids"]) == 5 and len(set(entry["ids"])) == 5
            assert entry["ids"][0] == r["token_ids"][p]  # top-1 is the greedy token
            assert entry["lps"] == sorted(entry["lps"], reverse=True)
            assert entry["lps"][0] <= 0.0


def test_seq_logprob_and_stay_derive_from_topk():
    res = MockRunner().run(ITEMS, recipe_with(max_new_tokens=8), WEIGHTS)
    for r in res:
        # every position is in topk when n <= 8, so the sums can be recomputed by hand
        assert r["n_generated"] <= 8
        seq = math.fsum(e["lps"][0] for e in r["topk"])
        assert r["seq_logprob"] == seq
        assert r["margin_by_position"] == [e["lps"][0] - e["lps"][1] for e in r["topk"]]
        stay = 0.0
        for e in r["topk"]:
            z = [v / 0.2 for v in e["lps"]]
            m = max(z)
            lse = m + math.log(math.fsum(math.exp(v - m) for v in z))
            stay += z[0] - lse
        assert abs(r["stay"] - stay) < 1e-9


def test_max_new_tokens_is_a_ceiling():
    res = MockRunner().run(ITEMS * 1, recipe_with(max_new_tokens=1), WEIGHTS)
    assert all(r["n_generated"] == 1 for r in res)


def test_reversed_order_returns_reversed_results_with_same_outputs():
    a = MockRunner().run(ITEMS, RECIPE, WEIGHTS)
    b = MockRunner().run(list(reversed(ITEMS)), RECIPE, WEIGHTS)
    assert [r["item_id"] for r in b] == [r["item_id"] for r in reversed(a)]
    assert ids_of(a) == ids_of(b)


# ---------------------------------------------------------------- determinism and the base key


def test_deterministic_across_instances_and_calls():
    a = MockRunner().run(ITEMS, RECIPE, WEIGHTS)
    b = MockRunner().run(ITEMS, RECIPE, WEIGHTS)
    assert a == b


def test_items_differ_from_each_other():
    res = MockRunner().run(ITEMS, RECIPE, WEIGHTS)
    seen = {tuple(r["token_ids"]) for r in res}
    assert len(seen) == len(ITEMS)


def test_identity_change_changes_every_output():
    a = ids_of(MockRunner().run(ITEMS, RECIPE, WEIGHTS))
    b = ids_of(MockRunner().run(ITEMS, RECIPE, subject_with(weights_sha256="1" * 64)))
    assert all(a[k] != b[k] for k in a)


def test_alias_identity_and_weights_identity_differ():
    a = ids_of(MockRunner().run(ITEMS, RECIPE, WEIGHTS))
    b = ids_of(MockRunner().run(ITEMS, RECIPE, ALIAS))
    c = ids_of(MockRunner().run(ITEMS, RECIPE, dict(ALIAS, region="us")))
    assert all(a[k] != b[k] for k in a)
    assert all(b[k] != c[k] for k in b)


def test_recipe_core_change_changes_every_output():
    base = ids_of(MockRunner().run(ITEMS, RECIPE, WEIGHTS))
    other_battery = copy.deepcopy(RECIPE)
    other_battery["battery"] = "sha256:" + "1" * 64
    assert all(base[k] != v for k, v in ids_of(MockRunner().run(ITEMS, other_battery, WEIGHTS)).items())
    other_template = copy.deepcopy(RECIPE)
    other_template["chat_template_sha256"] = "1" * 64
    assert all(base[k] != v for k, v in ids_of(MockRunner().run(ITEMS, other_template, WEIGHTS)).items())
    assert all(base[k] != v for k, v in ids_of(MockRunner().run(ITEMS, recipe_with(seed=8), WEIGHTS)).items())


def test_non_core_recipe_fields_do_not_change_outputs():
    base = MockRunner().run(ITEMS, RECIPE, WEIGHTS)
    r = copy.deepcopy(RECIPE)
    r["harness"]["version"] = "8.0.1"
    r["env_lock_sha256"] = "1" * 64
    assert MockRunner().run(ITEMS, r, WEIGHTS) == base


def test_environment_and_precision_do_not_change_plain_outputs():
    base = MockRunner().run(ITEMS, RECIPE, WEIGHTS)
    s = subject_with(precision="fp16")
    s["environment"] = {"runtime": {"framework": "x", "version": "y", "backend": "z"}, "hardware": {"gpu": "g", "driver": "d", "count": 2}}
    assert MockRunner().run(ITEMS, RECIPE, s) == base


def test_batch_size_and_padding_side_do_not_change_plain_outputs():
    base = MockRunner().run(ITEMS, RECIPE, WEIGHTS)
    assert MockRunner().run(ITEMS, recipe_with(batch_size=8), WEIGHTS) == base
    assert MockRunner().run(ITEMS, recipe_with(batch_size=3, padding_side="right"), WEIGHTS) == base


# ---------------------------------------------------------------- nuisance items


def test_nuisance_item_untouched_at_batch_1_in_any_order():
    base = ids_of(MockRunner().run(ITEMS, RECIPE, WEIGHTS))
    runner = MockRunner(nuisance_items={"i01", "i02"})
    assert ids_of(runner.run(ITEMS, RECIPE, WEIGHTS)) == base
    assert ids_of(runner.run(list(reversed(ITEMS)), RECIPE, WEIGHTS)) == base
    perm = [ITEMS[i] for i in (3, 0, 6, 1, 7, 2, 5, 4)]
    assert ids_of(runner.run(perm, RECIPE, WEIGHTS)) == base


def test_nuisance_item_flips_only_under_batch_gt_1():
    base = ids_of(MockRunner().run(ITEMS, RECIPE, WEIGHTS))
    runner = MockRunner(nuisance_items={"i01", "i02"})
    got = ids_of(runner.run(ITEMS, recipe_with(batch_size=8), WEIGHTS))
    assert got["i01"] != base["i01"] and got["i02"] != base["i02"]
    assert all(got[k] == base[k] for k in base if k not in {"i01", "i02"})


def test_a_plan_assignment_reaches_the_runner_through_the_recipe():
    """The link that was missing when the first floor was measured.

    A declared ``batch_size`` is applied by being written into that run's recipe, and the runner
    reads its batch size from ``recipe.decoding``. Without this step the plan is a description
    the runs ignore: five runs at batch 1, a floor of 0.0, and every later difference exceeding
    it (`papers/v8/vacuous_floor_2026_09_09/`). ``padding_side`` travels the same way.
    """
    plan = {"nuisance": [{"factor": "batch_size", "values": ["1", "8"]}]}
    settings = FP.plan_run_settings(plan, 2, RECIPE, [item["item_id"] for item in ITEMS])
    assert [s["recipe"]["decoding"]["batch_size"] for s in settings] == [1, 8]

    runner = MockRunner(nuisance_items={"i01"})
    first = ids_of(runner.run(ITEMS, settings[0]["recipe"], WEIGHTS))
    second = ids_of(runner.run(ITEMS, settings[1]["recipe"], WEIGHTS))
    assert second["i01"] != first["i01"]
    assert all(second[k] == first[k] for k in first if k != "i01")

    pads = FP.plan_run_settings(
        {"nuisance": [{"factor": "padding_side", "values": ["left", "right"]}]},
        2, RECIPE, [item["item_id"] for item in ITEMS],
    )
    assert [s["recipe"]["decoding"]["padding_side"] for s in pads] == ["left", "right"]


def test_nuisance_flip_changes_logprobs_too():
    base = {r["item_id"]: r for r in MockRunner().run(ITEMS, RECIPE, WEIGHTS)}
    got = {r["item_id"]: r for r in MockRunner(nuisance_items={"i01"}).run(ITEMS, recipe_with(batch_size=2), WEIGHTS)}
    assert got["i01"]["seq_logprob"] != base["i01"]["seq_logprob"]
    assert got["i01"]["topk"] != base["i01"]["topk"]
    assert got["i01"]["stay"] != base["i01"]["stay"]
    assert got["i00"] == base["i00"]


def test_nuisance_output_depends_on_position_parity_at_batch_gt_1():
    runner = MockRunner(nuisance_items={"i01"})
    rec = recipe_with(batch_size=4)
    at_odd = ids_of(runner.run(ITEMS, rec, WEIGHTS))["i01"]  # position 1
    at_even = ids_of(runner.run([ITEMS[1], ITEMS[0]] + ITEMS[2:], rec, WEIGHTS))["i01"]  # position 0
    at_odd_again = ids_of(runner.run([ITEMS[2], ITEMS[1], ITEMS[0]] + ITEMS[3:], rec, WEIGHTS))["i01"]  # position 1
    assert at_odd != at_even
    assert at_odd == at_odd_again
    base = ids_of(MockRunner().run(ITEMS, RECIPE, WEIGHTS))["i01"]
    assert base not in (at_odd, at_even)


def test_five_runs_batch_1_identical_and_batch_8_permuted_differ():
    """The fingerprint contract's floor scenario: nuisance items flip only under batch > 1."""
    import random

    runner = MockRunner(nuisance_items={"i01", "i02", "i05"})
    orders = [ITEMS] + [random.Random(s).sample(ITEMS, len(ITEMS)) for s in (11, 12, 13, 14)]
    at_1 = [ids_of(runner.run(o, recipe_with(batch_size=1), WEIGHTS)) for o in orders]
    assert all(x == at_1[0] for x in at_1)
    at_8 = [ids_of(runner.run(o, recipe_with(batch_size=8), WEIGHTS)) for o in orders]
    assert any(x != at_8[0] for x in at_8)
    for x in at_8:
        for k in x:
            if k not in {"i01", "i02", "i05"}:
                assert x[k] == at_1[0][k]


# ---------------------------------------------------------------- drift and precision items


def test_drift_items_always_differ_from_base():
    base = ids_of(MockRunner().run(ITEMS, RECIPE, WEIGHTS))
    for rec in (RECIPE, recipe_with(batch_size=8)):
        for order in (ITEMS, list(reversed(ITEMS))):
            got = ids_of(MockRunner(drift_items={"i03"}).run(order, rec, WEIGHTS))
            assert got["i03"] != base["i03"]
            assert all(got[k] == base[k] for k in base if k != "i03")


def test_drift_is_deterministic():
    a = MockRunner(drift_items={"i03"}).run(ITEMS, RECIPE, WEIGHTS)
    b = MockRunner(drift_items={"i03"}).run(ITEMS, RECIPE, WEIGHTS)
    assert a == b


def test_precision_items_flip_only_under_the_named_precision():
    base = ids_of(MockRunner().run(ITEMS, RECIPE, WEIGHTS))
    runner = MockRunner(precision_items={"fp16": {"i04"}, "int8-bnb": {"i06"}})
    assert ids_of(runner.run(ITEMS, RECIPE, WEIGHTS)) == base  # bf16: nothing named
    fp16 = ids_of(runner.run(ITEMS, RECIPE, subject_with(precision="fp16")))
    assert fp16["i04"] != base["i04"] and fp16["i06"] == base["i06"]
    assert all(fp16[k] == base[k] for k in base if k != "i04")
    int8 = ids_of(runner.run(ITEMS, RECIPE, subject_with(precision="int8-bnb")))
    assert int8["i06"] != base["i06"] and int8["i04"] == base["i04"]


def test_precision_variant_differs_from_drift_variant_and_combines():
    base = ids_of(MockRunner().run(ITEMS, RECIPE, WEIGHTS))["i04"]
    fp16 = subject_with(precision="fp16")
    p = ids_of(MockRunner(precision_items={"fp16": {"i04"}}).run(ITEMS, RECIPE, fp16))["i04"]
    d = ids_of(MockRunner(drift_items={"i04"}).run(ITEMS, RECIPE, fp16))["i04"]
    both = ids_of(MockRunner(drift_items={"i04"}, precision_items={"fp16": {"i04"}}).run(ITEMS, RECIPE, fp16))["i04"]
    assert len({tuple(base), tuple(p), tuple(d), tuple(both)}) == 4


def test_probe_scenario_precision_items_subset_of_nuisance_items():
    """precision_items within nuisance_items: at batch 1 under the reference precision every
    item is base; the delta-1 variant moves the precision items; the delta-2 variant (batch 8)
    moves every nuisance item, precision items included."""
    nuis = {"i01", "i02", "i05"}
    runner = MockRunner(nuisance_items=nuis, precision_items={"fp16": {"i01"}})
    base = ids_of(MockRunner().run(ITEMS, RECIPE, WEIGHTS))
    ref = ids_of(runner.run(ITEMS, RECIPE, WEIGHTS))
    assert ref == base
    d1 = ids_of(runner.run(ITEMS, RECIPE, subject_with(precision="fp16")))
    assert {k for k in d1 if d1[k] != ref[k]} == {"i01"}
    d2 = ids_of(runner.run(ITEMS, recipe_with(batch_size=8), WEIGHTS))
    assert {k for k in d2 if d2[k] != ref[k]} == nuis


# ---------------------------------------------------------------- logprobs=False


def test_logprobs_false_makes_the_four_fields_none_and_keeps_ids():
    with_lp = MockRunner().run(ITEMS, RECIPE, ALIAS)
    without = MockRunner(logprobs=False).run(ITEMS, RECIPE, ALIAS)
    for a, b in zip(with_lp, without):
        assert b["item_id"] == a["item_id"]
        assert b["token_ids"] == a["token_ids"]
        assert b["output_text"] == a["output_text"]
        assert b["n_generated"] == a["n_generated"]
        assert b["seq_logprob"] is None and b["topk"] is None
        assert b["margin_by_position"] is None and b["stay"] is None


# ---------------------------------------------------------------- refusals


def test_refuses_bad_items():
    r = MockRunner()
    with pytest.raises(ValueError):
        r.run([{"prompt_text": "x"}], RECIPE, WEIGHTS)
    with pytest.raises(ValueError):
        r.run([{"item_id": "", "prompt_text": "x"}], RECIPE, WEIGHTS)
    with pytest.raises(ValueError):
        r.run([{"item_id": "a"}], RECIPE, WEIGHTS)
    with pytest.raises(ValueError):
        r.run([{"item_id": "a", "prompt_text": "x"}, {"item_id": "a", "prompt_text": "y"}], RECIPE, WEIGHTS)
    with pytest.raises(ValueError):
        r.run(["not a dict"], RECIPE, WEIGHTS)


def test_refuses_bad_batch_size_and_max_new_tokens():
    r = MockRunner()
    with pytest.raises(ValueError):
        r.run(ITEMS, recipe_with(batch_size=0), WEIGHTS)
    with pytest.raises(ValueError):
        r.run(ITEMS, recipe_with(batch_size=True), WEIGHTS)
    with pytest.raises(ValueError):
        r.run(ITEMS, recipe_with(batch_size="8"), WEIGHTS)
    with pytest.raises(ValueError):
        r.run(ITEMS, recipe_with(max_new_tokens=0), WEIGHTS)


def test_refuses_unknown_subject_kind():
    with pytest.raises(ValueError):
        MockRunner().run(ITEMS, RECIPE, {"kind": "gguf", "model_family": "x"})


def test_empty_item_list_returns_empty():
    assert MockRunner().run([], RECIPE, WEIGHTS) == []


def test_constructor_copies_its_sets():
    nuis = {"i01"}
    r = MockRunner(nuisance_items=nuis, precision_items={"fp16": {"i02"}})
    nuis.add("i02")
    assert r.nuisance_items == frozenset({"i01"})
    assert r.precision_items == {"fp16": frozenset({"i02"})}
