# -*- coding: utf-8 -*-
"""styxx.portability v1 — the cross-machine floor and the verdict-level reading, on synthetic certs and on the
lab's own two-machine pair; and the refusals its red team forced on 2026-09-13: an absent leaf is never read as
survived, the base arm cannot choose the verdict, the digest names the inputs and not the labels, fingerprints
are bound to the certs."""
from __future__ import annotations

import copy
import json

import pytest

from styxx import portability as pt

CANARY = "c" * 64


def _cert(verdict, mean_abs, rdm_r, ci=(0.9, 1.1), a_hash=None, digest=None):
    c = {"schema": "styxx.checksum/compare/v1", "canary_sha256": CANARY, "n_items": 3,
         "a": {"model_id": "m", "rdm_sha256": a_hash}, "b": {"model_id": "x"},
         "distance": {"verdict": verdict, "mean_abs_nats": mean_abs, "rdm_r": rdm_r, "corr_dist": 0.1,
                      "ci_mean_abs": list(ci), "ci_rdm_r": [rdm_r - 0.05, rdm_r + 0.05]},
         "digest": digest or ("d" * 64)}
    return c


A_HASH = "a" * 64


def _run(int8=1.5, r=0.83, verdict="DRIFT", a_hash=A_HASH):
    return {"sanity": {"n_canaries": 3}, "reloaded": _cert("SAME", 0.0, 1.0, (0.0, 0.0), a_hash, "1" * 64),
            "int8": _cert(verdict, int8, r, (int8 - 0.3, int8 + 0.4), a_hash, "2" * 64), "random": _cert("DRIFT", 9.5, 0.07, (8.9, 10.1), a_hash, "3" * 64)}


def _fps(shift=0.0, a_hash=A_HASH):
    return {"A": {"canary_sha256": CANARY, "rdm_sha256": a_hash, "mean_lp_sha256": "e" * 64, "mean_lp": [-1.0 + shift, -2.0, -3.0]},
            "Q": {"canary_sha256": CANARY, "rdm_sha256": "q" * 64, "mean_lp_sha256": "f" * 64, "mean_lp": [-2.0 + 3 * shift, -2.5, -4.0]}}


def test_identical_runs_agree_within_floor_and_bind_the_base_arm():
    r = pt.compare([_run(), _run()], ["m1", "m2"], [_fps(), _fps()])
    assert r["verdicts"] == "AGREE" and r["magnitudes"] == "WITHIN-FLOOR"
    assert r["grading_floor_nats"] == pt.RESOLUTION_NATS
    assert r["base_arm_binding"].startswith("verified")
    assert r["cross_machine_floor"]["A"]["mean_abs_diff_mean_lp"] == 0.0
    assert r["intervals_graded"] is False and len(r["digest"]) == 64


def test_moved_magnitudes_with_agreeing_verdicts_read_agree_and_move_and_name_the_numbers():
    r = pt.compare([_run(int8=1.5145), _run(int8=1.4134, r=0.829)], ["build", "alienware"], [_fps(), _fps(1.7e-5)])
    assert r["verdicts"] == "AGREE" and r["magnitudes"] == "MOVE"
    assert "int8.mean_abs_nats" in r["magnitudes_moved"]
    assert not any(m.startswith("int8.ci") for m in r["magnitudes_moved"])   # intervals are reported, never graded
    assert r["arms"]["int8"]["intervals"]["ci_mean_abs"]["graded"] is False
    assert r["arms"]["int8"]["numbers"]["mean_abs_nats"]["max_abs_diff"] == pytest.approx(0.1011, abs=1e-4)
    assert r["cross_machine_floor"]["A"]["max_abs_diff_mean_lp"] == pytest.approx(1.7e-5)
    assert r["cross_machine_floor"]["A"]["mean_abs_diff_mean_lp"] == pytest.approx(1.7e-5 / 3)
    assert "floor on the tolerance" in r["reading"]


def test_a_verdict_flip_is_not_portable():
    r = pt.compare([_run(), _run(verdict="INCONCLUSIVE")], ["m1", "m2"])
    assert r["verdicts"] == "FLIP" and r["verdict_flips"] == ["int8"]


def test_an_absent_or_non_finite_leaf_on_one_machine_is_never_read_as_survived():
    b = _run()
    b["int8"]["distance"]["mean_abs_nats"] = None                  # a degenerate cert writes null
    r = pt.compare([_run(), b], ["m1", "m2"])
    assert r["magnitudes"] == "UNCOMPARABLE" and "int8.mean_abs_nats" in r["magnitudes_uncompared"]
    assert "nothing is said" in r["reading"]
    c = _run(); c["int8"]["distance"]["rdm_r"] = float("nan")
    assert pt.compare([_run(int8=1.6), c], ["m1", "m2"])["magnitudes"] == "MOVE"   # a move elsewhere still reads MOVE


def test_a_missing_verdict_is_refused():
    b = _run()
    del b["int8"]["distance"]["verdict"]
    with pytest.raises(ValueError):
        pt.compare([_run(), b], ["m1", "m2"])


def test_the_base_arm_cannot_choose_the_verdict():
    certs = [_run(int8=1.5), _run(int8=1.4)]
    with pytest.raises(ValueError):
        pt.compare(certs, ["m1", "m2"], [_fps(), _fps()], base_arm="ZZZ")        # absent from the fingerprints
    with pytest.raises(ValueError):
        pt.compare(certs, ["m1", "m2"], [_fps(), _fps()], base_arm="Q")          # not the `a` side of the certs


def test_fingerprints_are_bound_to_the_certs():
    with pytest.raises(ValueError):
        pt.compare([_run(), _run()], ["m1", "m2"], [_fps(), {**_fps(), "A": {**_fps()["A"], "canary_sha256": "d" * 64}}])
    short = _fps(); short["A"]["mean_lp"] = [-1.0, -2.0]
    with pytest.raises(ValueError):
        pt.compare([_run(), _run()], ["m1", "m2"], [_fps(), short])


def test_different_canary_sets_one_machine_or_no_shared_arm_are_refused():
    other = _run()
    for arm in ("reloaded", "int8", "random"):
        other[arm]["canary_sha256"] = "d" * 64
    with pytest.raises(ValueError):
        pt.compare([_run(), other], ["m1", "m2"])
    with pytest.raises(ValueError):
        pt.compare([_run()], ["m1"])
    with pytest.raises(ValueError):
        pt.compare([_run(), {"sanity": {}}], ["m1", "m2"])


def test_arms_present_on_one_machine_only_are_listed_not_hidden():
    b = _run(); b["fp16"] = _cert("SAME", 0.0, 1.0)
    r = pt.compare([_run(), b], ["m1", "m2"])
    assert r["arms_not_shared"] == ["fp16"] and "fp16" not in r["arms"]


def test_the_digest_covers_the_inputs_and_not_the_labels_or_the_clock():
    a = pt.compare([_run(), _run(int8=1.4)], ["m1", "m2"])
    b = pt.compare([_run(), _run(int8=1.4)], ["build", "alienware"])
    assert a["digest"] == b["digest"] and a["labels"] != b["labels"]
    c = pt.compare([_run(), _run(int8=1.3)], ["m1", "m2"])
    assert c["digest"] != a["digest"]
    d = _run(int8=1.4); d["int8"]["digest"] = "9" * 64                            # same numbers, different input cert
    assert pt.compare([_run(), d], ["m1", "m2"])["digest"] != a["digest"]
    e = pt.compare([_run(), _run(int8=1.4)], ["m1", "m2"], [_fps(), _fps()])
    f = pt.compare([_run(), _run(int8=1.4)], ["m1", "m2"], [_fps(), _fps(a_hash=A_HASH)])
    assert e["digest"] == f["digest"]
    g = _fps(); g["Q"]["mean_lp_sha256"] = "0" * 64
    assert pt.compare([_run(), _run(int8=1.4)], ["m1", "m2"], [_fps(), g])["digest"] != e["digest"]


def test_the_labs_own_two_machine_pair_reads_agree_and_move():
    # the finding of 2026-09-13, re-derived from the committed and the replication bytes (v0 certs: base-arm binding unverified)
    r = pt.compare([json.load(open("papers/checksum/smollm_quant_certs.json", encoding="utf-8")),
                    json.load(open("papers/checksum/replication_alienware_smollm_quant_certs.json", encoding="utf-8"))],
                   ["build_machine", "alienware"],
                   [json.load(open("papers/checksum/smollm_quant_fingerprints.json", encoding="utf-8")),
                    json.load(open("papers/checksum/replication_alienware_smollm_quant_fingerprints.json", encoding="utf-8"))])
    assert r["verdicts"] == "AGREE" and r["magnitudes"] == "MOVE"
    assert r["base_arm_binding"].startswith("unverified")
    assert r["cross_machine_floor"]["A"]["max_abs_diff_mean_lp"] == pytest.approx(1.7e-5, abs=1e-6)
    assert r["cross_machine_floor"]["A"]["mean_abs_diff_mean_lp"] == pytest.approx(8.17e-6, abs=1e-7)
    assert r["grading_floor_nats"] == pt.RESOLUTION_NATS
    assert r["arms"]["int8"]["numbers"]["mean_abs_nats"]["max_abs_diff"] == pytest.approx(0.1011, abs=1e-3)
    assert r["arms"]["reloaded"]["numbers"]["mean_abs_nats"]["max_abs_diff"] == 0.0
    assert not any(m.endswith(("ci_mean_abs", "ci_rdm_r")) for m in r["magnitudes_moved"])


def test_the_cli_writes_lf_exits_3_on_a_flip_and_2_on_a_refusal(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(_run())); b.write_text(json.dumps(_run(verdict="SAME")))
    out = tmp_path / "p.json"
    assert pt.main([str(a), str(b), "--out", str(out)]) == 3
    assert b"\r" not in out.read_bytes()
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["verdicts"] == "FLIP" and rec["labels"] == ["a.json", "b.json"]
    assert pt.main([str(a), str(a), "--base-arm", "ZZZ", "--fingerprints", str(a), str(a), "--out", str(out)]) == 2
