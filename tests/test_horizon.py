# -*- coding: utf-8 -*-
"""styxx.horizon — the fitter's contract on synthetic bits; no torch, no model, no GPU.

Load-bearing: the planted-crossing self-test recovers its horizons and the datasheet refuses when it
does not; a censored horizon prints "> 1024" and never a number; a constant responder under exact
balance maps to exactly 0 and reads SCREEN_FAIL; the bootstrap is a function of the seed; the family
is a fact of a config and its A values, never of a name; the sheet's family aggregates satisfy a
declared composition in the lab's own scorer; nothing this module prints carries a word the lab
never prints.
"""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import numpy as np
import pytest

from styxx import horizon as H

ROOT = Path(__file__).resolve().parent.parent
RECALL_FLAT = [0.9, 0.88, 0.86, 0.84]


@pytest.fixture(scope="module")
def selftest_result():
    return H.selftest()


def _row(compare, recall=RECALL_FLAT, n=600, seed=1, **kw):
    rec = H.load_receipt(H.synthetic_receipt("m", compare, recall, n, seed, n_resamples=300, **kw))
    return H.fit_model(rec)


# -- the self-test and the refusal it gates ---------------------------------------------------------

def test_the_self_test_recovers_every_planted_horizon(selftest_result):
    st = selftest_result
    assert st["pass"], [c for c in st["checks"] if not c["ok"]]
    names = {c["name"] for c in st["checks"]}
    assert {"planted_32", "planted_128", "planted_203", "planted_600", "planted_order", "censored",
            "constant_responder", "compare_only_flag", "deterministic_under_seed"} <= names


def test_the_datasheet_refuses_when_the_self_test_fails(monkeypatch):
    row = _row(H._planted_curve(128.0))
    bad = {"pass": False, "checks": [{"name": "planted_128", "ok": False, "detail": "recovered 900"}]}
    with pytest.raises(H.HorizonRefusal, match="self-test failed; no datasheet"):
        H.datasheet(row, selftest_result=bad)
    monkeypatch.setattr(H, "selftest", lambda *a, **k: bad)
    with pytest.raises(H.HorizonRefusal):
        H.datasheet(row)


def test_the_planted_illustration_of_the_design_crosses_where_the_rule_says():
    # the design's text planted "128" with c = (0.9, 0.7, 0.4, 0.2); under the interpolation in log2 D
    # those values cross at about 203, so the self-test plants from the rule, not from that sentence
    x = H.crossing([0.9, 0.7, 0.4, 0.2])
    assert x["status"] == "FINITE" and x["bracket"] == (64, 256) and 200 < x["point"] < 206
    for h in (32.0, 128.0, 203.0, 600.0):
        assert H.crossing(H._planted_curve(h))["point"] == pytest.approx(h, rel=1e-9)


# -- crossing, censoring, non-monotone curves --------------------------------------------------------

def test_no_crossing_is_censored_and_prints_no_number(selftest_result):
    row = _row([0.9, 0.85, 0.8, 0.75])
    hz = row["horizon"]["compare"]
    assert hz["status"] == "CENSORED" and hz["point"] is None and hz["f_cens"] > 0.975
    assert "NO_HORIZON_INSIDE_RANGE" in row["flags"]
    text = H.datasheet(row, selftest_result=selftest_result)
    line = next(l for l in text.splitlines() if "H_compare:" in l)
    assert "> 1024 (none inside the tested range" in line and "~" not in line.split("H_compare:")[1]


def test_a_crossing_near_the_last_grid_point_censors_the_upper_end_of_its_interval():
    row = _row([0.9, 0.8, 0.6, 0.42])
    hz = row["horizon"]["compare"]
    assert hz["status"] == "FINITE" and hz["bracket"] == (256, 1024)
    assert 0.025 < hz["f_cens"] < 0.5 and hz["ci_hi_censored"] is True and hz["print_censored"] is False
    assert "> 1024]" in H._h_text(hz)


def test_c_at_D0_at_or_below_chance_is_not_estimable():
    assert H.crossing([0.0, 0.0, 0.0, 0.0])["status"] == "NOT_ESTIMABLE"
    assert H.crossing([-0.1, 0.2, 0.2, 0.2])["status"] == "NOT_ESTIMABLE"
    assert H.crossing([0.5, float("nan"), 0.2, 0.2])["status"] == "NOT_ESTIMABLE"


def test_a_non_monotone_curve_is_flagged_and_its_earliest_crossing_used():
    x = H.crossing([0.9, 0.3, 0.8, 0.2])
    assert x["status"] == "FINITE" and x["bracket"] == (16, 64)
    row = _row([0.9, 0.3, 0.8, 0.2])
    assert "NONMONOTONE@compare" in row["flags"] and row["horizon"]["compare"]["bracket"] == (16, 64)


def test_the_grid_is_validated():
    with pytest.raises(H.HorizonRefusal, match="increasing"):
        H.crossing([0.9, 0.5, 0.2], grid=[16, 16, 64])
    with pytest.raises(H.HorizonRefusal, match="points"):
        H.crossing([0.9, 0.5], grid=[16, 64, 256])


# -- chance normalisation, the constant responder, the screen -------------------------------------

def test_normalize_is_zero_at_chance_and_one_at_ceiling():
    assert H.normalize(0.5, 0.5) == 0.0 and H.normalize(1.0, 0.5) == 1.0
    assert H.normalize(1.0 / 3.0, 1.0 / 3.0) == 0.0 and H.normalize(0.0, 0.5) == -1.0
    with pytest.raises(H.HorizonRefusal):
        H.normalize(0.5, 1.0)


def test_a_constant_responder_under_exact_balance_maps_to_exactly_zero_and_fails_the_screen(selftest_result):
    row = _row([0, 0, 0, 0], [0, 0, 0, 0], constant_compare=True, constant_recall=0)
    assert row["curves"]["compare"] == [0.0, 0.0, 0.0, 0.0]
    assert row["curves"]["recall"] == [0.0, 0.0, 0.0, 0.0]
    assert row["flags"][0] == "SCREEN_FAIL" and not row["screen"]["pass"]
    assert any("compare guard failed (yes-rate 1.00)" in r for r in row["screen"]["reasons"])
    assert any("largest slot share 1.00" in r for r in row["screen"]["reasons"])
    assert row["horizon"]["compare"]["status"] == "NOT_ESTIMABLE"
    assert row["cells"]["compare"][16]["auroc"] == 0.5 and row["cells"]["compare"][16]["c_auc"] == 0.0
    text = H.datasheet(row, selftest_result=selftest_result)
    assert "not estimable" in text and "FAIL" in text


def test_the_d0_split_is_stratified_so_both_halves_keep_exact_balance():
    rec = H.load_receipt(H.synthetic_receipt("m", [0.6, 0.5, 0.4, 0.3], RECALL_FLAT, 600, 3, n_resamples=100))
    cell = rec["cells"]["compare"][16]
    s_mask, d_mask, note = H._split_masks(cell)
    assert s_mask.sum() == 300 and d_mask.sum() == 300 and not np.any(s_mask & d_mask)
    assert cell["_labels"][s_mask].sum() == 150 and cell["_labels"][d_mask].sum() == 150
    assert "stratified" in note
    row = H.fit_model(rec)
    assert row["screen"]["n_compare"] == 300 and row["cells"]["compare"][16]["n"] == 300
    assert row["cells"]["compare"][64]["n"] == 600 and "stratified" in row["split_note"]


def test_the_screen_cannot_pass_on_correctness_bits_alone():
    rng = np.random.default_rng(0)
    bits = {str(d): rng.random(600).tolist() for d in H.GRID}
    bits = {d: [v < 0.9 for v in vs] for d, vs in bits.items()}
    rec = H.load_receipt({"model": "bits-only", "per_item": {"compare": bits, "recall": bits}})
    row = H.fit_model(rec, n_resamples=50)
    assert not row["screen"]["pass"] and any("guard not recorded" in r for r in row["screen"]["reasons"])
    assert "LOGODDS_ABSENT" in row["flags"] and row["horizon"]["compare_auc"]["status"] == "NOT_RECORDED"


def test_guard_rules_are_the_bridge_v3_runner_s():
    assert H.guard(yes_rate=0.2) and H.guard(yes_rate=0.8) and not H.guard(yes_rate=0.81)
    assert H.guard(max_slot_frac=0.8) and not H.guard(max_slot_frac=0.801) and H.guard() is None


# -- the bootstrap: paired, deterministic, shared across models ---------------------------------------

def test_the_bootstrap_is_a_function_of_the_seed():
    rec = H.load_receipt(H.synthetic_receipt("m", H._planted_curve(203.0), RECALL_FLAT, 600, 5, n_resamples=200))
    a, b = H.fit_model(rec), H.fit_model(rec)
    for key in ("horizon", "ratio_min", "dissociation", "curves"):
        assert json.dumps(H._jsonable(a[key])) == json.dumps(H._jsonable(b[key]))
    c = H.fit_model(rec, seed=H.SEED_BOOT + 1)
    assert c["horizon"]["compare"]["ci_lo"] != a["horizon"]["compare"]["ci_lo"]


def test_two_models_with_identical_bits_share_one_resample_and_give_one_family_interval():
    r1 = H.synthetic_receipt("a", H._planted_curve(400.0), RECALL_FLAT, 600, 9, arch_family="ATTN",
                             controls={"ablated": {"compare": {16: (0.02, None)}}}, n_resamples=200)
    r2 = json.loads(json.dumps(r1))
    r2["model"] = "b"
    sheet = H.fit_zoo([H.load_receipt(r1), H.load_receipt(r2)])
    res = sheet["result"]
    assert res["shared_resample"] is True and res["family_set"] == ["a", "b"]
    ra, rb = sheet["rows"]
    assert ra["dissociation"] == rb["dissociation"]
    assert res["attention_family_min_dissoc_ci"]["lo"] == ra["dissociation"]["ci_lo"]
    assert res["attention_family_min_dissoc_ci"]["hi"] == ra["dissociation"]["ci_hi"]
    assert res["attention_family_min_dissoc"] == H._round6(ra["dissociation"]["point"])


def test_the_family_aggregates_satisfy_a_declared_composition_in_the_lab_s_scorer():
    from styxx.protocol import Experiment
    recs = [H.load_receipt(H.synthetic_receipt("attn-1", [0.9, 0.85, 0.8, 0.7], RECALL_FLAT, 600, 11, arch_family="ATTN",
                                               controls={"ablated": {"compare": {16: (0.02, None)}}}, n_resamples=100)),
            H.load_receipt(H.synthetic_receipt("attn-2", H._planted_curve(300.0), RECALL_FLAT, 600, 12, arch_family="ATTN",
                                               controls={"ablated": {"compare": {16: (0.30, None)}}}, n_resamples=100)),
            H.load_receipt(H.synthetic_receipt("decay-1", H._planted_curve(100.0), RECALL_FLAT, 600, 13, arch_family="DECAY",
                                               controls={"ablated": {"compare": {16: (0.01, None)}}}, n_resamples=100)),
            H.load_receipt(H.synthetic_receipt("T-copy", [0.9, 0.9, 0.9, 0.9], RECALL_FLAT, 600, 14, arch_family="ATTN",
                                               controls={"ablated": {"compare": {16: (0.01, None)}}}, n_resamples=100))]
    sheet = H.fit_zoo(recs, exclude=["T-copy"])
    res = sheet["result"]
    assert res["family_set"] == ["attn-1"] and res["n_attention_screen_pass"] == 1
    assert set(res["family_excluded"]) == {"attn-2", "decay-1", "T-copy"}
    assert res["family_excluded_reasons"]["attn-2"].startswith("PROBE_READS_PRIORS")
    # M1: the ablated maximum is over the screened set BEFORE the ablation exclusion, so it can fail
    assert res["family_set_ablated_max_norm_acc"] == pytest.approx(0.30, abs=0.02) and res["family_set_ablated_max_norm_acc"] > H.ABLATED_BAR
    exp = Experiment.__new__(Experiment)
    exp.composition = {"G": {"agg": "min", "over": "result.per_model.dissoc", "excluding": "result.family_excluded"}}
    exp._check_composition("G", res["attention_family_min_dissoc"], {"result": H._jsonable(res)})
    with pytest.raises(Exception, match="COMPOSITION VIOLATION"):
        exp._check_composition("G", res["attention_family_min_dissoc"] - 0.5, {"result": H._jsonable(res)})


# -- receipts: both shapes, the flat list, the refusals -----------------------------------------------

def test_the_flat_item_list_and_the_cell_block_give_the_same_row():
    rec = H.synthetic_receipt("m", H._planted_curve(128.0), RECALL_FLAT, 300, 21, n_resamples=100)
    items = []
    for fam, by_d in rec["per_item"].items():
        for d, cell in by_d.items():
            for i in range(len(cell["label"])):
                it = {"family": fam, "distance": int(d), "item": i, "label": cell["label"][i],
                      "prediction": cell["prediction"][i]}
                if fam == "compare":
                    it["logodds"] = cell["logodds"][i]
                items.append(it)
    flat = {"schema": H.RECEIPT_SCHEMA, "model": "m", "config": rec["config"], "items": items[::-1]}
    a, b = H.fit_model(H.load_receipt(rec)), H.fit_model(H.load_receipt(flat))
    assert a["curves"] == b["curves"] and a["horizon"]["compare"] == b["horizon"]["compare"]


def test_a_bridge_v3_receipt_is_read_by_arm_and_refused_without_per_item_bits():
    rec = H.synthetic_receipt("m", H._planted_curve(128.0), RECALL_FLAT, 300, 22, n_resamples=100)
    v3 = {"config": {"checkpoints": {"M1": "state-spaces/mamba-1.4b-hf"}, "distances": list(H.GRID), "D0": 16,
                     "seeds": {"bootstrap": 20261011}, "screen_bar": 0.4},
          "per_item": {"M1": rec["per_item"]}}
    loaded = H.load_receipt(v3, arm="M1")
    assert loaded["model"] == "state-spaces/mamba-1.4b-hf" and loaded["config"]["copied_from"] == "v3 receipt arm M1"
    with pytest.raises(H.HorizonRefusal, match="Z4"):
        H.load_receipt({"config": v3["config"]}, arm="M1")
    with pytest.raises(H.HorizonRefusal, match="no per_item bits for arm 'T'"):
        H.load_receipt(v3, arm="T")


@pytest.mark.parametrize("edit,match", [
    (lambda r: r["per_item"].pop("recall"), "no recall cells"),
    (lambda r: r["per_item"]["compare"].pop("1024"), "no cell at distance"),
    (lambda r: r.update(schema="styxx.horizon/receipt/9.9"), "schema"),
    (lambda r: r.update(controls={"windowed": {}}), "unknown control"),
    (lambda r: r["per_item"]["compare"]["16"].update(prediction=[not p for p in r["per_item"]["compare"]["16"]["prediction"]]),
     "disagree with the sign"),
    (lambda r: r["config"].update(pairing="sometimes"), "pairing"),
    (lambda r: r.update(arch_family={"family": "HYBRID"}), "arch_family"),
    (lambda r: r["per_item"]["recall"]["64"].update(label=[7] * 60), "slots must be 0, 1 or 2"),
])
def test_malformed_receipts_are_refused_by_name(edit, match):
    rec = H.synthetic_receipt("m", H._planted_curve(128.0), RECALL_FLAT, 60, 23, n_resamples=20)
    edit(rec)
    with pytest.raises(H.HorizonRefusal, match=match):
        H.load_receipt(rec)


def test_cells_of_unequal_size_are_refused_under_item_pairing_and_accepted_when_declared_independent():
    rec = H.synthetic_receipt("m", H._planted_curve(128.0), RECALL_FLAT, 60, 24, n_resamples=20)
    rng = np.random.default_rng(0)
    rec["per_item"]["compare"]["64"] = H.synthetic_cell("compare", 0.6, 40, rng)
    with pytest.raises(H.HorizonRefusal, match="differ in item count"):
        H.fit_model(H.load_receipt(rec))
    rec["config"]["pairing"] = "independent"
    row = H.fit_model(H.load_receipt(rec))
    assert row["pairing"] == "independent" and row["horizon"]["compare"]["n_resamples"] == 20


def test_rows_are_json_without_nan(tmp_path):
    row = _row([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], constant_compare=True, constant_recall=1)
    out = tmp_path / "row.json"
    H._write_json(str(out), row)
    text = out.read_text(encoding="utf-8")
    assert "NaN" not in text and "_rep" not in text
    back = json.loads(text)
    assert back["ratio_min"]["compare"]["point"] is None and back["model"] == "m"


# -- controls and their flags ------------------------------------------------------------------------

def test_controls_set_their_flags_and_their_absence_is_printed():
    ctrl = {"ablated": {"compare": {16: (0.02, None), 1024: (0.05, None)}, "recall": {16: (0.0, None)}},
            "near": {"compare": {1024: (0.85, None)}}, "overwrite": {"compare": {1024: (0.03, 200)}}}
    row = _row(H._planted_curve(203.0), controls=ctrl)
    assert {"COMPARE_ONLY_HORIZON", "FORMAT_DISTANCE_EXCLUDED", "POSITIVE_CONTROL_FIRED"} <= set(row["flags"])
    assert not row["controls"]["ablated"]["reads_priors"]
    bare = _row(H._planted_curve(203.0))
    assert {"ABLATED_CONTROL_ABSENT", "FORMAT_DISTANCE_NOT_EXCLUDED", "FAR_BINDING_CONTROL_ABSENT"} <= set(bare["flags"])
    decayed = _row(H._planted_curve(203.0), controls={"near": {"compare": {1024: (0.3, None)}},
                                                      "overwrite": {"compare": {1024: (0.6, 200)}},
                                                      "ablated": {"compare": {16: (0.4, None)}}})
    assert "FORMAT_DECAY@1024" in decayed["flags"] and "POSITIVE_CONTROL_SILENT@1024" in decayed["flags"]
    assert "PROBE_READS_PRIORS" in decayed["flags"]


def test_degenerate_cells_past_D0_are_flagged_with_both_numbers():
    rec = H.synthetic_receipt("m", [0.9, 0.8, 0.7, 0.6], RECALL_FLAT, 600, 31, n_resamples=50)
    rng = np.random.default_rng(1)
    rec["per_item"]["compare"]["1024"] = H.synthetic_cell("compare", 0.0, 600, rng, constant=True)
    row = H.fit_model(H.load_receipt(rec))
    flag = next(f for f in row["flags"] if f.startswith("DEGENERATE@compare:1024"))
    assert "c=+0.00" in flag and "c_auc=+0.00" in flag and row["cells"]["compare"][1024]["degenerate"]


# -- the architecture family --------------------------------------------------------------------------

def test_the_family_is_a_fact_of_config_and_weights_never_a_name():
    with pytest.raises(H.HorizonRefusal, match="not a fact"):
        H.assert_family("state-spaces/mamba-1.4b-hf")
    with pytest.raises(H.HorizonRefusal, match="no model_type"):
        H.assert_family({"architectures": ["MambaForCausalLM"]})
    with pytest.raises(H.HorizonRefusal, match="needs its A values"):
        H.assert_family({"model_type": "mamba"})
    ok = H.assert_family({"model_type": "mamba", "num_hidden_layers": 2}, A=[[-1.0, -3.7], [-0.18, -65.9]])
    assert ok["family"] == "DECAY" and ok["fact"]["max_A"] == -0.18 and ok["fact"]["layers"] == 2
    assert H.assert_family({"model_type": "mamba"}, A={"A_log": [[0.0, 1.0]]})["fact"]["min_A"] == pytest.approx(-np.e)
    with pytest.raises(H.HorizonRefusal, match="not strictly negative"):
        H.assert_family({"model_type": "mamba"}, A=[[-1.0, 0.0]])
    with pytest.raises(H.HorizonRefusal, match="complex"):
        H.assert_family({"model_type": "mamba"}, A=[np.array([-1.0 + 0.5j])])
    attn = H.assert_family({"model_type": "gemma2", "max_position_embeddings": 8192, "sliding_window": 4096})
    assert attn["family"] == "ATTN" and attn["fact"]["reach_tokens"] == 4096
    assert H.assert_family({"model_type": "qwen2", "max_position_embeddings": 32768, "sliding_window": 512,
                            "use_sliding_window": False})["fact"]["reach_tokens"] == 32768
    with pytest.raises(H.HorizonRefusal, match="reaches 512 tokens"):
        H.assert_family({"model_type": "llama", "max_position_embeddings": 131072, "sliding_window": 512})
    with pytest.raises(H.HorizonRefusal, match="refuses to guess"):
        H.assert_family({"model_type": "jamba"})


def _write_safetensors(path: Path, tensors: dict) -> None:
    header, blobs, off = {}, [], 0
    for name, (arr, dt) in tensors.items():
        b = ((arr.view(np.uint32) >> 16).astype(np.uint16).tobytes() if dt == "BF16"
             else arr.astype("<f2" if dt == "F16" else "<f4").tobytes())
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    hj = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(hj)) + hj + b"".join(blobs))


def test_A_log_is_read_from_safetensors_without_torch_and_without_a_model_load(tmp_path):
    a0 = np.log(np.arange(1, 7, dtype=np.float32).reshape(2, 3))
    a1 = np.log(np.array([[0.5, 2.0, 8.0]], dtype=np.float32))
    _write_safetensors(tmp_path / "m.safetensors", {
        "backbone.layers.1.mixer.A_log": (a1, "BF16"), "backbone.layers.0.mixer.A_log": (a0, "F32"),
        "backbone.layers.0.mixer.D": (np.ones(3, dtype=np.float32), "F32")})
    got = H.mamba_A_from_safetensors([str(tmp_path / "m.safetensors")])
    assert got["layers"] == 2 and got["names"][0].endswith("layers.0.mixer.A_log")
    assert np.allclose(got["A"][0], -np.arange(1, 7).reshape(2, 3))
    assert np.allclose(got["A"][1], [[-0.5, -2.0, -8.0]], rtol=1e-2)   # the BF16 layer
    fact = H.assert_family({"model_type": "mamba"}, A=got["A"])
    assert fact["family"] == "DECAY" and fact["fact"]["min_A"] == pytest.approx(-8.0, rel=1e-2)
    _write_safetensors(tmp_path / "none.safetensors", {"x.D": (np.ones(2, dtype=np.float32), "F32")})
    with pytest.raises(H.HorizonRefusal, match="no tensor named"):
        H.mamba_A_from_safetensors([str(tmp_path / "none.safetensors")])
    (tmp_path / "junk.safetensors").write_bytes(b"\x00" * 4)
    with pytest.raises(H.HorizonRefusal, match="short header"):
        H.mamba_A_from_safetensors([str(tmp_path / "junk.safetensors")])


# -- the datasheet and its words ---------------------------------------------------------------------

def _lab_banned_words():
    src = ROOT / "scripts" / "zenodo_publish_v7_48_0.py"
    words = list(H._HOUSE_WORDS)
    if src.exists():
        m = re.search(r"^BANNED = \((.*?)\)$", src.read_text(encoding="utf-8"), re.S | re.M)
        if m:
            words = sorted(set(words) | set(re.findall(r'"([^"]+)"', m.group(1))))
    return re.compile(r"\b(" + "|".join(re.escape(w) + "s?" for w in words) + r")\b", re.I)


def test_the_datasheet_prints_its_limits_in_the_lab_s_words(selftest_result):
    ctrl = {"ablated": {"compare": {16: (0.02, None)}}, "near": {"compare": {1024: (0.85, None)}}}
    rec = H.synthetic_receipt("decay-like", H._planted_curve(203.0), RECALL_FLAT, 600, 41, arch_family="DECAY",
                              controls=ctrl, n_resamples=200)
    rec["config"]["native_D"] = {"16": {"min": 15, "mean": 15.6, "max": 18}, "64": {"min": 61, "mean": 62.4, "max": 66},
                                 "256": {"min": 246, "mean": 249.5, "max": 258}, "1024": {"min": 990, "mean": 998.0, "max": 1010}}
    rec["config"].update({"dtype": "fp16", "params": "1372M", "ctx": 2048, "ctx_is_training_length": True})
    row = H.fit_model(H.load_receipt(rec))
    text = H.datasheet(row, selftest_result=selftest_result)
    assert "64..256 | ~" in text and "native" in text and "H_recall:  > 1024" in text
    assert "licenses no sentence about rotation, honesty, causation or ranking" in text
    assert "inside the tested range" in text and "inside context" not in text
    assert "2048*" in text and "COMPARE_ONLY_HORIZON" in text and "verdict:" not in text
    assert "gates sha256: none (unscored)" in text
    assert row["horizon"]["compare"]["native_point"] is not None
    banned = _lab_banned_words()
    assert not banned.findall(text), banned.findall(text)
    sheet = H.fit_zoo([H.load_receipt(rec)], exclude=[])
    full = H.datasheet(sheet, selftest_result=selftest_result)
    assert "verdict: unscored: no committed prereg" in full and "DECAY family" not in full
    assert not banned.findall(full), banned.findall(full)


def test_the_datasheet_refuses_to_print_a_house_word(monkeypatch, selftest_result):
    row = _row(H._planted_curve(128.0))
    monkeypatch.setattr(H, "LIMITS", H.LIMITS + ("a " + "ground" + "breaking" + " instrument",))
    with pytest.raises(H.HorizonRefusal, match="never prints"):
        H.datasheet(row, selftest_result=selftest_result)


def test_the_module_its_readme_and_this_file_carry_no_house_word():
    banned = _lab_banned_words()
    for p in (ROOT / "styxx" / "horizon.py", ROOT / "styxx" / "horizon_README.md", Path(__file__)):
        raw = p.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"{p.name} has a BOM"
        hits = banned.findall(raw.decode("utf-8"))
        assert not hits, f"{p.name}: {hits}"


# -- the CLI --------------------------------------------------------------------------------------------

def test_the_cli_fits_a_receipt_and_refuses_a_missing_one(tmp_path, capsys, monkeypatch, selftest_result):
    monkeypatch.setattr(H, "selftest", lambda *a, **k: selftest_result)
    rec = H.synthetic_receipt("cli", H._planted_curve(128.0), RECALL_FLAT, 300, 51, n_resamples=100)
    p = tmp_path / "horizon_cli.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    out = tmp_path / "row.json"
    assert H.main(["fit", "--receipt", str(p), "--json", str(out)]) == 0
    assert "styxx.horizon datasheet" in capsys.readouterr().out and json.loads(out.read_text(encoding="utf-8"))["model"] == "cli"
    assert H.main(["sheet", "--receipts", str(tmp_path)]) == 0
    assert "unscored: no committed prereg" in capsys.readouterr().out
    assert H.main(["fit", "--receipt", str(tmp_path / "absent.json")]) == 2
    assert "REFUSED: cannot read receipt" in capsys.readouterr().err
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"model_type": "llama", "max_position_embeddings": 131072}), encoding="utf-8")
    assert H.main(["family", "--config", str(cfg)]) == 0 and '"ATTN"' in capsys.readouterr().out
    assert H.main(["family", "--config", "a-name"]) == 2
