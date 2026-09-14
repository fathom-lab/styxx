# -*- coding: utf-8 -*-
"""papers/checksum/score.py executes the two frozen PREREGs' reading rules on a certs file.

v2 of these tests, after the red team of 2026-09-14 showed v1 bound bands by formatted needles (seven
band changes and six ≤/< flips passed every v1 test). Now: every number in BANDS is PARSED out of the
frozen text and compared with the code; every comparator is pinned by certs sitting exactly on each
edge; the sealed digests are the frozen files' bytes and the SEALS rows; and the scorer is shown to
re-derive what v1 trusted — K1 from the floor, the sealed digest without --expect-blob, the 48-item
draw, every arm grading the same set."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(ROOT, "papers", "checksum")


def _load():
    spec = importlib.util.spec_from_file_location("score", os.path.join(CK, "score.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


score = _load()


def _doc(name):
    t = open(os.path.join(CK, name), encoding="utf-8").read()
    t = re.sub(r"<sworn[^>]*>|</sworn>", "", t)
    return re.sub(r"\s+", " ", t)


def _certs(name):
    return copy.deepcopy(json.load(open(os.path.join(CK, name), encoding="utf-8")))


def _experiment(prereg):
    """The committed 0.5B instrument check reshaped as the experiment: the only fields changed are the ones
    that say it is not (is_the_experiment, model, tag). Everything the scorer re-derives stays real."""
    c = _certs("beacon_draw_certs_dryrun_qwen0.5b.json" if prereg == "beacon_draw" else "deploy_quant_certs_dryrun_qwen0.5b.json")
    c["is_the_experiment"], c["model"], c["tag"] = True, score.MODEL, ""
    return c


# ----------------------------------------------------------------------------- the bands are the frozen text

def test_every_beacon_draw_band_is_parsed_out_of_the_frozen_text_and_equals_the_code():
    b = score.BANDS["beacon_draw"]
    t = _doc(b["prereg"])
    m = re.search(r"\*\*H1 \(floor, held-out form\)\.\*\* The worst-pairwise null floor is ≤ (\S+) nats/token; zero is allowed", t)
    assert m and float(m.group(1)) == b["h1_floor_max"] and b["h1_floor_min_exclusive"] is None
    m = re.search(r"\*\*H2 \(NF4 on the drawn set\)\.\*\* A vs Q4 reads DRIFT; mean \|Δ log-prob\| in \*\*\[([\d.]+), ([\d.]+)\] "
                  r"nats/token\*\*; geometry r ≥ ([\d.]+); argmax lost on ≤ (\d+) of 48", t)
    assert m and (float(m.group(1)), float(m.group(2))) == b["h2_band"]
    assert float(m.group(3)) == b["h2_r_min"] and int(m.group(4)) == b["h2_top1_loss_max"]
    m = re.search(r"\*\*H3 \(int8\)\.\*\* A vs Q8 reads DRIFT with mean \|Δ log-prob\| below Q4's; r ≥ ([\d.]+); argmax lost on ≤ (\d+) of 48", t)
    assert m and float(m.group(1)) == b["h3_r_min"] and int(m.group(2)) == b["h3_top1_loss_max"]
    m = re.search(r"\*\*H4 \(far control\)\.\*\* A vs R: mean \|Δ log-prob\| > ([\d.]+) nats/token; r < ([\d.]+); top-1 hits ≤ (\d+) of 48", t)
    assert m and float(m.group(1)) == b["h4_mean_min"] and float(m.group(2)) == b["h4_r_max"] and int(m.group(3)) == b["h4_hits_max"]
    assert "NF4 mean lies within a factor of two of the hand set's, either way" in t and b["h5_ratio_band"] == (0.5, 2.0)
    m = re.search(r"Q4's and Q8's mean \|Δ log-prob\| move by at most ([\d.]+) nats/token and R's by at most ([\d.]+)", t)
    assert m and b["h6_move_max"] == {"Q4": float(m.group(1)), "Q8": float(m.group(1)), "R": float(m.group(2))}
    m = re.search(r"\*\*K1\.\*\* Worst-pairwise null floor > (\S+) nats/token", t)
    assert m and float(m.group(1)) == b["k1_floor"]
    m = re.search(r"\*\*K3\.\*\* A vs Q4 > ([\d.]+) nats/token, or argmax lost on > (\d+) of 48", t)
    assert m and float(m.group(1)) == b["k3_mean"] and int(m.group(2)) == b["k3_top1_loss"]


def test_every_deploy_quant_band_is_parsed_out_of_the_frozen_text_and_equals_the_code():
    b = score.BANDS["deploy_quant"]
    t = _doc(b["prereg"])
    m = re.search(r"\*\*H1 \(floor\)\.\*\* The null floor on GPU is > (\S+) and ≤ (\S+) nats/token\.", t)
    assert m and float(m.group(1)) == b["h1_floor_min_exclusive"] and float(m.group(2)) == b["h1_floor_max"]
    m = re.search(r"\*\*H2 \(NF4\)\.\*\* A vs Q4 reads DRIFT, mean \|Δ log-prob\| in \*\*\[([\d.]+), ([\d.]+)\] nats/token\*\*, "
                  r"belief geometry r ≥ ([\d.]+), top-1 loss ≤ (\d+) of 48", t)
    assert m and (float(m.group(1)), float(m.group(2))) == b["h2_band"]
    assert float(m.group(3)) == b["h2_r_min"] and int(m.group(4)) == b["h2_top1_loss_max"]
    m = re.search(r"\*\*H4 \(far control\)\.\*\* A vs R: mean \|Δ log-prob\| > ([\d.]+) nats/token, r < ([\d.]+), top-1 ≤ (\d+) of 48", t)
    assert m and float(m.group(1)) == b["h4_mean_min"] and float(m.group(2)) == b["h4_r_max"] and int(m.group(3)) == b["h4_hits_max"]
    m = re.search(r"\*\*K1\.\*\* Null floor > (\S+) nats/token", t)
    assert m and float(m.group(1)) == b["k1_floor"]
    m = re.search(r"\*\*K3\.\*\* A vs Q4 > ([\d.]+) nats/token or Q4 top-1 loss > (\d+)", t)
    assert m and float(m.group(1)) == b["k3_mean"] and int(m.group(2)) == b["k3_top1_loss"]


def test_the_sealed_digests_are_the_frozen_files_and_the_seal_rows_and_the_hand_set_is_the_named_one():
    seals = open(os.path.join(CK, "SEALS_2026_09_13.md"), encoding="utf-8").read()
    for b in score.BANDS.values():
        assert hashlib.sha256(open(os.path.join(CK, b["prereg"]), "rb").read()).hexdigest() == b["sealed_blob"]
        assert b["sealed_blob"] in seals
    from styxx import checksum as ck
    assert ck.canary_sha256(ck.CANARIES) == score.HAND_CANARY_SHA256
    assert score.HAND_CANARY_SHA256[:12] in _doc(score.BANDS["deploy_quant"]["prereg"])


# ----------------------------------------------------------------------------- every comparator, on its edge

def _set(c, **v):
    s = c["sanity"]
    if "floor" in v:
        s["null_floor_nats"] = v["floor"]
        c["k1"]["floor_nats"], c["k1"]["fired"] = v["floor"], v["floor"] > 1e-2
    for key, arm, field in (("m4", "Q4", "mean_abs_nats"), ("r4", "Q4", "rdm_r"), ("m8", "Q8", "mean_abs_nats"),
                            ("r8", "Q8", "rdm_r"), ("mr", "R", "mean_abs_nats"), ("rr", "R", "rdm_r")):
        if key in v:
            c[arm]["distance"][field] = v[key]
    if "l4" in v:
        s["top1_loss_vs_A"]["Q4"] = v["l4"]
    if "l8" in v:
        s["top1_loss_vs_A"]["Q8"] = v["l8"]
    if "hr" in v:
        s["first_token_top1_hits"]["R"] = v["hr"]
    return c


EDGES = [
    # (prereg, fields, where, expected) — where is (hypothesis, clause index) or ("gate", name)
    ("beacon_draw", {"floor": 1e-3}, ("H1", 0), True),
    ("beacon_draw", {"floor": 1.000001e-3}, ("H1", 0), False),
    ("beacon_draw", {"floor": 0.0}, ("H1", 0), True),
    ("beacon_draw", {"m4": 0.60}, ("H2", 1), True),
    ("beacon_draw", {"m4": 0.05, "m8": 0.01}, ("H2", 1), True),
    ("beacon_draw", {"m4": 0.6000001}, ("H2", 1), False),
    ("beacon_draw", {"m4": 0.0499999, "m8": 0.01}, ("H2", 1), False),
    ("beacon_draw", {"r4": 0.90}, ("H2", 2), True),
    ("beacon_draw", {"r4": 0.8999999}, ("H2", 2), False),
    ("beacon_draw", {"l4": 24}, ("H2", 3), True),
    ("beacon_draw", {"l4": 25}, ("H2", 3), False),
    ("beacon_draw", {"m8": 0.34657728324075127}, ("H3", 1), False),        # equal to Q4's mean: "below" is strict
    ("beacon_draw", {"r8": 0.98}, ("H3", 2), True),
    ("beacon_draw", {"r8": 0.9799999}, ("H3", 2), False),
    ("beacon_draw", {"l8": 8}, ("H3", 3), True),
    ("beacon_draw", {"l8": 9}, ("H3", 3), False),
    ("beacon_draw", {"mr": 5.0}, ("H4", 0), False),
    ("beacon_draw", {"mr": 5.0000001}, ("H4", 0), True),
    ("beacon_draw", {"rr": 0.6}, ("H4", 1), False),
    ("beacon_draw", {"rr": 0.5999999}, ("H4", 1), True),
    ("beacon_draw", {"hr": 4}, ("H4", 2), True),
    ("beacon_draw", {"hr": 5}, ("H4", 2), False),
    ("beacon_draw", {"m4": 1.5}, ("gate", "K3"), False),
    ("beacon_draw", {"m4": 1.5000001}, ("gate", "K3"), True),
    ("beacon_draw", {"l4": 36}, ("gate", "K3"), False),
    ("beacon_draw", {"l4": 37}, ("gate", "K3"), True),
    ("beacon_draw", {"mr": 0.60}, ("gate", "K4"), True),
    ("beacon_draw", {"mr": 0.6000001}, ("gate", "K4"), False),
    ("beacon_draw", {"floor": 1e-2}, ("gate", "K1"), False),
    ("beacon_draw", {"floor": 1.0000001e-2}, ("gate", "K1"), True),
    ("deploy_quant", {"floor": 0.0}, ("H1", 0), False),
    ("deploy_quant", {"floor": 1e-3}, ("H1", 0), True),
    ("deploy_quant", {"floor": 1.000001e-3}, ("H1", 0), False),
    ("deploy_quant", {"m4": 0.30}, ("H2", 1), True),
    ("deploy_quant", {"m4": 0.01, "m8": 0.001}, ("H2", 1), True),
    ("deploy_quant", {"m4": 0.3000001}, ("H2", 1), False),
    ("deploy_quant", {"r4": 0.95}, ("H2", 2), True),
    ("deploy_quant", {"r4": 0.9499999}, ("H2", 2), False),
    ("deploy_quant", {"l4": 4}, ("H2", 3), True),
    ("deploy_quant", {"l4": 5}, ("H2", 3), False),
    ("deploy_quant", {"mr": 5.0}, ("H4", 0), False),
    ("deploy_quant", {"rr": 0.3}, ("H4", 1), False),
    ("deploy_quant", {"rr": 0.2999999}, ("H4", 1), True),
    ("deploy_quant", {"hr": 2}, ("H4", 2), True),
    ("deploy_quant", {"hr": 3}, ("H4", 2), False),
    ("deploy_quant", {"m4": 1.0}, ("gate", "K3"), False),
    ("deploy_quant", {"l4": 12}, ("gate", "K3"), False),
    ("deploy_quant", {"l4": 13}, ("gate", "K3"), True),
]


@pytest.mark.parametrize("prereg,fields,where,expected", EDGES, ids=[f"{p}-{w[0]}{w[1]}-{f}" for p, f, w, _ in EDGES])
def test_every_comparator_on_its_edge(prereg, fields, where, expected):
    card = score.score(_set(_experiment(prereg), **fields), prereg)
    if where[0] == "gate":
        assert card["gates"][where[1]]["fired"] is expected
    else:
        assert card["hypotheses"][where[0]]["clauses"][where[1]]["holds"] is expected


def test_h5_ratio_band_is_inclusive_both_ways_and_h6_moves_are_inclusive():
    hand = _experiment("deploy_quant")
    hm4 = hand["Q4"]["distance"]["mean_abs_nats"]
    for m4, holds in ((2.0 * hm4, True), (0.5 * hm4, True), (2.0000001 * hm4, False)):
        c = _set(_experiment("beacon_draw"), m4=m4, m8=0.01, l4=10)
        h5 = score.score(c, "beacon_draw", hand_set=hand)["hypotheses"]["H5"]
        assert h5["clauses"][-1]["holds"] is holds, (m4 / hm4, h5)
    port = {"verdicts": "AGREE", "digest": "x", "arms": {a: {"numbers": {"mean_abs_nats": {"max_abs_diff": mx}}}
                                                         for a, mx in (("Q4", 0.10), ("Q8", 0.10), ("R", 0.22))}}
    assert score.score(_experiment("beacon_draw"), "beacon_draw", portability=port)["hypotheses"]["H6"]["status"] == "HELD"
    port["arms"]["Q8"]["numbers"]["mean_abs_nats"]["max_abs_diff"] = 0.1000001
    h6 = score.score(_experiment("beacon_draw"), "beacon_draw", portability=port)["hypotheses"]["H6"]
    assert h6["status"] == "FAILED" and [c["holds"] for c in h6["clauses"]] == [True, True, False, True]


def test_beacon_h1_applies_the_correction_rules_when_the_floor_is_above_zero():
    text = _doc("CORRECTION_prereg_beacon_draw_2026_09_14.md")
    c = _set(_experiment("beacon_draw"), floor=5e-4)
    c["h1_held_out"]["cert"]["distance"]["verdict"] = "INCONCLUSIVE"
    h1 = score.score(c, "beacon_draw")["hypotheses"]["H1"]
    assert h1["status"] == "HELD on the floor, INCONCLUSIVE on the pair by construction" and "rule 3" in h1["rule"]
    assert h1["status"] in text
    c["h1_held_out"]["cert"]["distance"]["verdict"] = "DRIFT"
    h1 = score.score(c, "beacon_draw")["hypotheses"]["H1"]
    assert h1["status"] == "FAILED" and "rule 4" in h1["rule"]
    c = _set(_experiment("beacon_draw"), floor=0.0)                   # bit-identical loads: evaluated as frozen
    c["h1_held_out"]["cert"]["distance"]["verdict"] = "INCONCLUSIVE"
    h1 = score.score(c, "beacon_draw")["hypotheses"]["H1"]
    assert h1["status"] == "FAILED" and h1["rule"] is None
    c = _set(_experiment("beacon_draw"), floor=2e-3)                  # the floor clause fails: no by-construction reading
    c["h1_held_out"]["cert"]["distance"]["verdict"] = "INCONCLUSIVE"
    assert score.score(c, "beacon_draw")["hypotheses"]["H1"]["status"] == "FAILED"


# ----------------------------------------------------------------------------- what v1 trusted, v2 re-derives

def test_the_experiment_shape_counts_as_a_result_without_any_flag():
    card = score.score(_experiment("beacon_draw"), "beacon_draw")
    assert card["gates"]["K5"]["fired"] is False and card["counts_as_result"] is True
    assert not card["run_reading"].startswith("INSTRUMENT CHECK")
    card = score.score(_experiment("deploy_quant"), "deploy_quant")
    assert card["gates"]["provenance"]["fired"] is False and card["counts_as_result"] is True


def test_k1_is_re_derived_from_the_floor_and_a_contradicting_k1_record_invalidates_the_certs():
    c = _experiment("beacon_draw")
    c["sanity"]["null_floor_nats"] = 0.05                      # the runner's k1 record still says fired: false
    card = score.score(c, "beacon_draw")
    assert card["gates"]["K1"]["fired"] is True and card["counts_as_result"] is False
    assert any("k1 record" in d for d in card["gates"]["K5"]["detail"])
    del c["k1"]
    assert score.score(c, "beacon_draw")["gates"]["K1"]["fired"] is True


def test_a_valid_sealed_run_whose_k1_fired_is_a_result_inconclusive_not_an_instrument_check():
    c = _set(_experiment("beacon_draw"), floor=0.05)
    card = score.score(c, "beacon_draw")
    assert card["counts_as_result"] is True and card["run_reading"] == "INCONCLUSIVE (K1)"
    assert all(v["status"] == "NOT_EVALUATED" for v in card["hypotheses"].values())


def test_the_sealed_digest_is_checked_without_expect_blob_and_a_wrong_expect_blob_is_refused():
    c = _experiment("beacon_draw")
    c["provenance"]["prereg_blob_sha256"] = "0" * 64
    card = score.score(c, "beacon_draw")
    assert card["gates"]["K5"]["fired"] is True and card["counts_as_result"] is False
    assert any("not the sealed digest" in d for d in card["gates"]["K5"]["detail"])
    card = score.score(_experiment("beacon_draw"), "beacon_draw", expect_blob="1" * 64)
    assert any("--expect-blob" in d for d in card["gates"]["K5"]["detail"])
    c = _experiment("deploy_quant")
    c["provenance"]["prereg_blob_sha256"] = "0" * 64
    assert score.score(c, "deploy_quant")["gates"]["provenance"]["fired"] is True


def test_an_honest_draw_of_the_wrong_size_and_arms_grading_another_set_fire_k5():
    from styxx import beacon, checksum as ck
    c = _experiment("beacon_draw")
    draw = c["canaries"]["draw"]
    eight = ck.canary_sha256(beacon.select(draw["beacon"], 8))
    for rec in [c["canaries"]["draw"]] + [c[a]["draw"] for a in score.ARMS] + [c["h1_held_out"]["cert"]["draw"]]:
        rec["n"], rec["canary_sha256"] = 8, eight
    c["canaries"]["canary_sha256"] = eight
    for a in score.ARMS:
        c[a]["canary_sha256"] = eight
    c["h1_held_out"]["cert"]["canary_sha256"] = eight
    card = score.score(c, "beacon_draw")
    assert card["gates"]["K5"]["fired"] is True and any("n=8" in d for d in card["gates"]["K5"]["detail"])
    c = _experiment("beacon_draw")
    c["Q4"]["canary_sha256"], c["Q4"]["draw"] = "f" * 64, dict(c["Q4"]["draw"], beacon="e" * 64)
    detail = score.score(c, "beacon_draw")["gates"]["K5"]["detail"]
    assert any("the Q4 cert grades canary set" in d for d in detail) and any("the Q4 cert carries a different draw" in d for d in detail)
    c = _experiment("beacon_draw")
    c["h1_held_out"]["unpreregistered"] = True
    assert any("not marked preregistered" in d for d in score.score(c, "beacon_draw")["gates"]["K5"]["detail"])
    c = _experiment("deploy_quant")
    c["R"]["canary_sha256"] = "f" * 64
    assert score.score(c, "deploy_quant")["gates"]["provenance"]["fired"] is True


def test_a_lying_beacon_fires_k5_and_the_expected_beacon_is_checked():
    c = _experiment("beacon_draw")
    honest = c["canaries"]["draw"]["beacon"]
    lying = copy.deepcopy(c)
    lying["canaries"]["draw"]["beacon"] = "b" * 64
    assert any("does not produce the canaries" in d for d in score.score(lying, "beacon_draw")["gates"]["K5"]["detail"])
    assert any("ANCHORED" in d for d in score.score(c, "beacon_draw", expect_beacon="c" * 64)["gates"]["K5"]["detail"])
    card = score.score(c, "beacon_draw", expect_beacon=honest)
    assert card["counts_as_result"] is True and not any("beacon clause was not checked" in n for n in card["notes"])


def test_h5_needs_a_valid_hand_set_experiment_on_the_same_device():
    c = _experiment("beacon_draw")
    assert score.score(c, "beacon_draw", hand_set=_certs("deploy_quant_certs_dryrun_qwen0.5b.json"))["hypotheses"]["H5"]["status"] == "PENDING"
    hand = _experiment("deploy_quant")
    h5 = score.score(c, "beacon_draw", hand_set=hand)["hypotheses"]["H5"]
    assert h5["status"] == "HELD" and h5["clauses"][-1]["observed"] == pytest.approx(0.34658 / 0.42703, rel=1e-3)
    hand["provenance"]["cuda_device"] = "another device"
    assert score.score(c, "beacon_draw", hand_set=hand)["hypotheses"]["H5"]["status"] == "PENDING"


# ----------------------------------------------------------------------------- the committed checks and the CLI

@pytest.mark.parametrize("prereg,certs,card_file,statuses,gate", [
    ("beacon_draw", "beacon_draw_certs_dryrun_qwen0.5b.json", "beacon_draw_scorecard_v2_dryrun_qwen0.5b.json",
     {"H1": "HELD", "H2": "HELD", "H3": "HELD", "H4": "HELD", "H5": "PENDING", "H6": "PENDING"}, "K5"),
    ("deploy_quant", "deploy_quant_certs_dryrun_qwen0.5b.json", "deploy_quant_scorecard_v2_dryrun_qwen0.5b.json",
     {"H1": "FAILED", "H2": "FAILED", "H3": "HELD", "H4": "HELD"}, "provenance"),
])
def test_the_committed_instrument_checks_read_as_committed(prereg, certs, card_file, statuses, gate):
    card = score.score(_certs(certs), prereg)
    assert card["counts_as_result"] is False and card["gates"][gate]["fired"] is True
    assert {h: v["status"] for h, v in card["hypotheses"].items()} == statuses
    committed = json.load(open(os.path.join(CK, card_file), encoding="utf-8"))
    assert committed["schema"] == score.SCHEMA
    for k in ("hypotheses", "gates", "counts_as_result", "run_reading"):
        assert committed[k] == card[k], f"the committed scorecard's {k} is not what the scorer reads today"
    assert committed["certs_sha256"] == hashlib.sha256(open(os.path.join(CK, certs), "rb").read()).hexdigest()


@pytest.mark.parametrize("prereg,certs", [("beacon_draw", "beacon_draw_certs_dryrun_qwen0.5b.json"),
                                          ("deploy_quant", "deploy_quant_certs_dryrun_qwen0.5b.json")])
def test_the_cli_writes_a_scorecard_even_on_a_cp1252_console(tmp_path, prereg, certs):
    out = tmp_path / "card.json"
    env = dict(os.environ, PYTHONIOENCODING="cp1252")           # v1 died here on the deploy_quant path
    env.pop("PYTHONUTF8", None)
    r = subprocess.run([sys.executable, os.path.join(CK, "score.py"), "--prereg", prereg, os.path.join(CK, certs), "--out", str(out)],
                       capture_output=True, timeout=300, cwd=ROOT, env=env)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    card = json.loads(out.read_text(encoding="utf-8"))
    assert card["schema"] == score.SCHEMA and card["certs_file"] == f"papers/checksum/{certs}" and len(card["certs_sha256"]) == 64
