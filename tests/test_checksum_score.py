# -*- coding: utf-8 -*-
"""papers/checksum/score.py executes the two frozen PREREGs' reading rules on a certs file. These tests
bind the bands in the code to the frozen documents' text (so the code cannot drift from the PREREG
without a test failing), and pin what the scorer reads on the committed instrument checks and on
synthetic certs where a gate fires or a record lies."""
from __future__ import annotations

import copy
import importlib.util
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(ROOT, "papers", "checksum")


def _load():
    spec = importlib.util.spec_from_file_location("score", os.path.join(CK, "score.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


score = _load()


def _text(name):
    return open(os.path.join(CK, name), encoding="utf-8").read()


def test_every_band_in_the_code_is_in_the_frozen_document_it_is_attributed_to():
    b = score.BANDS["beacon_draw"]
    t = _text(b["prereg"])
    for needle in ("≤ 1e-3 nats/token", f"**[{b['h2_band'][0]:.2f}, {b['h2_band'][1]:.2f}]", f"r ≥ {b['h2_r_min']:.2f}",
                   f"≤ {b['h2_top1_loss_max']} of 48", f"r ≥ {b['h3_r_min']:.2f}", f"≤ {b['h3_top1_loss_max']} of 48",
                   f"> {int(b['h4_mean_min'])} nats/token", f"r < {b['h4_r_max']}", f"top-1 hits ≤ {b['h4_hits_max']} of 48",
                   "within\n  a factor of two", f"{b['h6_move_max']['Q4']:.2f}</sworn> nats/token", f"{b['h6_move_max']['R']:.2f}</sworn>",
                   "> 1e-2 nats/token", f"> {b['k3_mean']} nats/token, or argmax lost on > {b['k3_top1_loss']} of 48"):
        assert needle in t, f"beacon_draw: {needle!r} is not in the frozen PREREG"
    assert b["h1_floor_min_exclusive"] is None and "zero is\n  allowed" in t
    d = score.BANDS["deploy_quant"]
    t = _text(d["prereg"])
    for needle in ("> 0 and ≤ 1e-3 nats/token", f"**[{d['h2_band'][0]:.2f}, {d['h2_band'][1]:.2f}] nats/token**", f"r ≥ {d['h2_r_min']:.2f}",
                   f"top-1 loss ≤ {d['h2_top1_loss_max']} of 48", f"> {int(d['h4_mean_min'])} nats/token, r < {d['h4_r_max']}, top-1 ≤ {d['h4_hits_max']} of 48",
                   "Null floor > 1e-2", f"> {d['k3_mean']:.1f} nats/token or Q4 top-1 loss > {d['k3_top1_loss']}"):
        assert needle in t, f"deploy_quant: {needle!r} is not in the frozen PREREG"
    assert d["k1_floor"] == b["k1_floor"] == 1e-2


def test_the_beacon_draw_instrument_check_reads_every_band_held_and_is_not_a_result():
    certs = json.load(open(os.path.join(CK, "beacon_draw_certs_dryrun_qwen0.5b.json"), encoding="utf-8"))
    card = score.score(certs, "beacon_draw", expect_blob="d6a98261f44f31664cdedbc24769532d5a701bf46d2c8b9ce30cf6a227fb5519")
    assert card["is_the_experiment"] is False and card["counts_as_result"] is False
    assert {h: v["status"] for h, v in card["hypotheses"].items()} == {
        "H1": "HELD", "H2": "HELD", "H3": "HELD", "H4": "HELD", "H5": "PENDING", "H6": "PENDING"}
    assert card["gates"]["K5"]["fired"] is True                      # another model, not the experiment
    assert card["gates"]["K1"]["fired"] is False and card["gates"]["K2"]["fired"] is False
    assert card["gates"]["K3"]["fired"] is False and card["gates"]["K4"]["fired"] is False
    assert card["hypotheses"]["H1"]["preregistered"] is True
    assert card["run_reading"].startswith("INSTRUMENT CHECK")
    committed = json.load(open(os.path.join(CK, "beacon_draw_scorecard_dryrun_qwen0.5b.json"), encoding="utf-8"))
    for k in ("hypotheses", "gates", "counts_as_result", "run_reading"):
        assert committed[k] == card[k], f"the committed scorecard's {k} is not what the scorer reads today"


def test_the_hand_set_instrument_check_fails_the_bands_written_for_the_larger_model():
    # the 0.5B check under the 2026-09-13 PREREG: floor 0 fails "> 0" by the letter, NF4 0.427 sits above
    # [0.01, 0.30] — a wrong prediction in the safe direction is still a wrong prediction, and the scorer says so
    certs = json.load(open(os.path.join(CK, "deploy_quant_certs_dryrun_qwen0.5b.json"), encoding="utf-8"))
    card = score.score(certs, "deploy_quant")
    assert card["counts_as_result"] is False
    h1 = card["hypotheses"]["H1"]
    assert h1["status"] == "FAILED" and h1["clauses"][0]["holds"] is False and h1["clauses"][1]["holds"] is True
    assert h1["held_out_reading_unpreregistered"]["verdict"] == "SAME"
    assert card["hypotheses"]["H2"]["status"] == "FAILED"
    assert [c["holds"] for c in card["hypotheses"]["H2"]["clauses"]] == [True, False, True, True]
    assert card["hypotheses"]["H3"]["status"] == "HELD" and card["hypotheses"]["H4"]["status"] == "HELD"
    assert card["gates"]["K3"]["fired"] is False and card["gates"]["K4"]["fired"] is False
    committed = json.load(open(os.path.join(CK, "deploy_quant_scorecard_dryrun_qwen0.5b.json"), encoding="utf-8"))
    assert committed["hypotheses"] == card["hypotheses"] and committed["gates"] == card["gates"]


def test_k1_fired_evaluates_nothing_else():
    certs = json.load(open(os.path.join(CK, "beacon_draw_certs_dryrun_qwen0.5b.json"), encoding="utf-8"))
    certs = copy.deepcopy(certs)
    certs["k1"] = {"threshold_nats": 1e-2, "floor_nats": 0.02, "fired": True}
    certs["sanity"]["null_floor_nats"] = 0.02
    card = score.score(certs, "beacon_draw")
    assert card["gates"]["K1"]["fired"] is True
    assert all(v["status"] == "NOT_EVALUATED" for v in card["hypotheses"].values())
    assert card["run_reading"] == "INCONCLUSIVE (K1)"


def test_a_draw_record_whose_beacon_lies_fires_k5_and_the_expected_beacon_is_checked():
    certs = json.load(open(os.path.join(CK, "beacon_draw_certs_dryrun_qwen0.5b.json"), encoding="utf-8"))
    honest = certs["canaries"]["draw"]["beacon"]
    lying = copy.deepcopy(certs)
    lying["canaries"]["draw"]["beacon"] = "b" * 64
    card = score.score(lying, "beacon_draw")
    assert card["gates"]["K5"]["fired"] is True
    assert any("does not produce the canaries" in d for d in card["gates"]["K5"]["detail"])
    card = score.score(certs, "beacon_draw", expect_beacon="c" * 64)
    assert any("not the beacon the ANCHORED line prints" in d for d in card["gates"]["K5"]["detail"])
    card = score.score(certs, "beacon_draw", expect_beacon=honest)
    assert not any("ANCHORED" in d for d in card["gates"]["K5"]["detail"])
    assert not any("beacon clause was not checked" in n for n in card["notes"])


def test_the_sealed_experiment_shape_counts_as_a_result_only_when_everything_lines_up():
    certs = copy.deepcopy(json.load(open(os.path.join(CK, "beacon_draw_certs_dryrun_qwen0.5b.json"), encoding="utf-8")))
    certs["is_the_experiment"] = True
    certs["model"] = score.MODEL
    certs["tag"] = ""
    card = score.score(certs, "beacon_draw", expect_beacon=certs["canaries"]["draw"]["beacon"],
                       expect_blob=certs["provenance"]["prereg_blob_sha256"])
    assert card["gates"]["K5"]["fired"] is False and card["counts_as_result"] is True
    assert not card["run_reading"].startswith("INSTRUMENT CHECK")
    # the same certs with the sealed digest wrong: not a result
    card = score.score(certs, "beacon_draw", expect_beacon=certs["canaries"]["draw"]["beacon"], expect_blob="0" * 64)
    assert card["gates"]["K5"]["fired"] is True and card["counts_as_result"] is False


def test_h5_and_h6_read_their_inputs_when_given():
    certs = copy.deepcopy(json.load(open(os.path.join(CK, "beacon_draw_certs_dryrun_qwen0.5b.json"), encoding="utf-8")))
    hand = copy.deepcopy(json.load(open(os.path.join(CK, "deploy_quant_certs_dryrun_qwen0.5b.json"), encoding="utf-8")))
    card = score.score(certs, "beacon_draw", hand_set=hand)
    assert card["hypotheses"]["H5"]["status"] == "PENDING"            # the hand-set check is not the experiment
    hand["is_the_experiment"] = True
    card = score.score(certs, "beacon_draw", hand_set=hand)
    h5 = card["hypotheses"]["H5"]
    assert h5["status"] == "HELD" and h5["clauses"][-1]["observed"] == pytest.approx(0.34658 / 0.42703, rel=1e-2)
    port = {"schema": "styxx.portability/v1", "verdicts": "AGREE", "digest": "x",
            "arms": {"Q4": {"numbers": {"mean_abs_nats": {"max_abs_diff": 0.05}}},
                     "Q8": {"numbers": {"mean_abs_nats": {"max_abs_diff": 0.11}}},
                     "R": {"numbers": {"mean_abs_nats": {"max_abs_diff": 0.2}}}}}
    card = score.score(certs, "beacon_draw", portability=port)
    h6 = card["hypotheses"]["H6"]
    assert h6["status"] == "FAILED" and [c["holds"] for c in h6["clauses"]] == [True, True, False, True]


def test_the_cli_writes_a_scorecard(tmp_path):
    import subprocess
    import sys
    out = tmp_path / "card.json"
    r = subprocess.run([sys.executable, os.path.join(CK, "score.py"), "--prereg", "beacon_draw",
                        os.path.join(CK, "beacon_draw_certs_dryrun_qwen0.5b.json"), "--out", str(out)],
                       capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    card = json.loads(out.read_text(encoding="utf-8"))
    assert card["schema"] == score.SCHEMA and card["certs_file"] == "papers/checksum/beacon_draw_certs_dryrun_qwen0.5b.json"
    assert len(card["certs_sha256"]) == 64
