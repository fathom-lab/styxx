# -*- coding: utf-8 -*-
"""styxx.portability — the cross-machine floor and the verdict-level reading, on synthetic certs and on the
lab's own two-machine pair (committed vs replication bytes of the checksum RESULT)."""
from __future__ import annotations

import copy
import json

import pytest

from styxx import portability as pt

CANARY = "c" * 64


def _cert(verdict, mean_abs, rdm_r, ci=(0.9, 1.1)):
    return {"schema": "styxx.checksum/compare/v1", "canary_sha256": CANARY,
            "distance": {"verdict": verdict, "mean_abs_nats": mean_abs, "rdm_r": rdm_r, "corr_dist": 0.1,
                         "ci_mean_abs": list(ci), "ci_rdm_r": [rdm_r - 0.05, rdm_r + 0.05]}}


def _run(int8=1.5, r=0.83, verdict="DRIFT"):
    return {"sanity": {"n_canaries": 48}, "reloaded": _cert("SAME", 0.0, 1.0, (0.0, 0.0)),
            "int8": _cert(verdict, int8, r, (int8 - 0.3, int8 + 0.4)), "random": _cert("DRIFT", 9.5, 0.07, (8.9, 10.1))}


def _fps(shift=0.0):
    return {"A": {"mean_lp": [-1.0 + shift, -2.0, -3.0]}, "Q": {"mean_lp": [-2.0 + 3 * shift, -2.5, -4.0]}}


def test_identical_runs_agree_within_floor():
    r = pt.compare([_run(), _run()], ["m1", "m2"], [_fps(), _fps()])
    assert r["verdicts"] == "AGREE" and r["magnitudes"] == "WITHIN-FLOOR"
    assert r["grading_floor_nats"] == pt.RESOLUTION_NATS and "no fingerprints" not in r["grading_floor_source"]
    assert r["cross_machine_floor"]["A"]["max_abs_diff_mean_lp"] == 0.0
    assert len(r["digest"]) == 64


def test_moved_magnitudes_with_agreeing_verdicts_read_agree_and_move_and_name_the_numbers():
    r = pt.compare([_run(int8=1.5145), _run(int8=1.4134, r=0.829)], ["build", "alienware"], [_fps(), _fps(1.7e-5)])
    assert r["verdicts"] == "AGREE" and r["magnitudes"] == "MOVE"
    assert "int8.mean_abs_nats" in r["magnitudes_moved"] and "int8.ci_mean_abs" in r["magnitudes_moved"]
    assert r["arms"]["int8"]["numbers"]["mean_abs_nats"]["max_abs_diff"] == pytest.approx(0.1011, abs=1e-4)
    # the cross-machine null floor is below the resolution, so the resolution grades; both are reported
    assert r["cross_machine_floor"]["A"]["max_abs_diff_mean_lp"] == pytest.approx(1.7e-5)
    assert r["cross_machine_floor"]["A"]["n_items_differing"] == 1
    assert r["grading_floor_nats"] == pytest.approx(pt.RESOLUTION_NATS)
    assert "tolerance" in r["reading"]


def test_a_verdict_flip_is_not_portable():
    r = pt.compare([_run(), _run(verdict="INCONCLUSIVE")], ["m1", "m2"])
    assert r["verdicts"] == "FLIP" and r["verdict_flips"] == ["int8"]
    assert "not portable" in r["reading"]


def test_different_canary_sets_or_one_machine_are_refused():
    other = _run()
    for arm in ("reloaded", "int8", "random"):
        other[arm]["canary_sha256"] = "d" * 64
    with pytest.raises(ValueError):
        pt.compare([_run(), other], ["m1", "m2"])
    with pytest.raises(ValueError):
        pt.compare([_run()], ["m1"])


def test_the_digest_covers_the_comparison_not_the_labels_order_of_files_or_clock():
    a = pt.compare([_run(), _run(int8=1.4)], ["m1", "m2"])
    b = pt.compare([_run(), _run(int8=1.4)], ["m1", "m2"])
    assert a["digest"] == b["digest"]
    c = pt.compare([_run(), _run(int8=1.3)], ["m1", "m2"])
    assert c["digest"] != a["digest"]


def test_the_labs_own_two_machine_pair_reads_agree_and_move():
    # the finding of 2026-09-13, re-derived from the committed and the replication bytes
    r = pt.compare([json.load(open("papers/checksum/smollm_quant_certs.json", encoding="utf-8")),
                    json.load(open("papers/checksum/replication_alienware_smollm_quant_certs.json", encoding="utf-8"))],
                   ["build_machine", "alienware"],
                   [json.load(open("papers/checksum/smollm_quant_fingerprints.json", encoding="utf-8")),
                    json.load(open("papers/checksum/replication_alienware_smollm_quant_fingerprints.json", encoding="utf-8"))])
    assert r["verdicts"] == "AGREE" and r["magnitudes"] == "MOVE"
    assert r["cross_machine_floor"]["A"]["max_abs_diff_mean_lp"] == pytest.approx(1.7e-5, abs=1e-6)
    assert r["arms"]["int8"]["numbers"]["mean_abs_nats"]["max_abs_diff"] == pytest.approx(0.1011, abs=1e-3)
    assert r["arms"]["reloaded"]["numbers"]["mean_abs_nats"]["max_abs_diff"] == 0.0


def test_the_cli_writes_lf_and_exits_3_on_a_flip(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(_run())); b.write_text(json.dumps(_run(verdict="SAME")))
    out = tmp_path / "p.json"
    code = pt.main([str(a), str(b), "--labels", "x", "y", "--out", str(out)])
    assert code == 3 and b"\r" not in out.read_bytes()
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["verdicts"] == "FLIP" and rec["inputs"]["certs"] == [str(a), str(b)]
