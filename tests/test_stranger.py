# -*- coding: utf-8 -*-
"""styxx.stranger runs the checks a person who does not trust this lab runs (papers/checksum/STRANGER.md)
and prints one table. These tests pin: the ferry log passes against its own head (one real
re-derivation of every line, cached for the module) and the head-mismatch / no-head readings on a
stubbed report; every beacon-drawn certs file in the tree re-derives; every certs file the scorer knows
is read and the committed scorecards match; one receipt re-checks VERIFIED and a sample issued against
a temporary file is reported as not checkable, not as a failure; the CLI validates --only, writes its
report and exits by failure."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from styxx import charon, stranger

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "papers", "charon", "charon.log.jsonl")


def _head():
    _, entries, _, _ = charon.read_log(Path(LOG))
    return charon.head_of(entries)


@pytest.fixture(scope="module")
def ferry_log_pass():
    return stranger.step_ferry_log(Path(ROOT), _head())      # the one real re-derivation of every line


def test_the_ferry_log_passes_against_its_own_head(ferry_log_pass):
    ok = ferry_log_pass
    assert ok["status"] == "PASS" and ok["head_matches"] is True and ok["external_head_checked"] is True
    assert set(ok["by_status"]) >= {"SAME_LINE", "MOVED_VERIFIER"} and ok["chain_broken_at_line"] is None


def test_a_wrong_head_fails_and_no_head_is_internal_consistency_only(monkeypatch):
    real_head = _head()

    def fake_verify_log(log, repo, expect_head=None):
        return {"lines": [{}] * 3, "head": real_head, "head_expected": expect_head,
                "head_matches": (None if expect_head is None else expect_head.lower() == real_head),
                "by_status": {"SAME_LINE": 3}, "chain_problems": [], "chain_broken_at_line": None}
    monkeypatch.setattr(charon, "verify_log", fake_verify_log)
    bad = stranger.step_ferry_log(Path(ROOT), "0" * 64)
    assert bad["status"] == "FAIL" and bad["head_matches"] is False and "HEAD MISMATCH" in bad["detail"]
    none = stranger.step_ferry_log(Path(ROOT), None)
    assert none["status"] == "PASS" and none["external_head_checked"] is False and "internal consistency only" in none["detail"]
    tampered = dict(fake_verify_log(None, None, real_head), by_status={"SAME_LINE": 2, "TAMPER": 1})
    monkeypatch.setattr(charon, "verify_log", lambda *a, **k: tampered)
    assert stranger.step_ferry_log(Path(ROOT), real_head)["status"] == "FAIL"


def test_every_beacon_drawn_certs_file_in_the_tree_re_derives():
    s = stranger.step_draw(Path(ROOT))
    assert s["status"] == "PASS"
    drawn = [f for f in s["files"] if "beacon" in f["detail"]]
    assert len(drawn) >= 2 and all(f["status"] == "PASS" for f in drawn)


def test_every_known_certs_file_is_read_and_the_committed_scorecards_match():
    s = stranger.step_reading(Path(ROOT))
    assert s["status"] == "PASS"
    read = [f for f in s["files"] if f["status"] != "SKIP"]
    assert len(read) >= 3 and all(f["counts_as_result"] is False for f in read)   # nothing in the tree is the experiment
    compared = [f for f in read if "committed_scorecard" in f]
    assert len(compared) >= 2 and all(f["committed_scorecard_matches"] for f in compared)


def test_one_receipt_re_checks_verified_and_a_temporary_sample_is_not_a_failure():
    rc = os.path.join(ROOT, "papers", "checksum", "PREREG_checksum_beacon_draw_2026_09_14.sworn-receipt.json")
    row = stranger.check_receipt(Path(ROOT), rc)
    assert row["status"] == "PASS" and row["detail"].startswith("VERIFIED") and row["target"].endswith(".sworn.json")
    # a receipt issued under an earlier verifier build re-derives exactly when the sidecar is the target
    old = os.path.join(ROOT, "papers", "sworn", "RESULT_sworn_v01_ships_2026_09_01.sworn-receipt.json")
    if os.path.exists(old):
        row = stranger.check_receipt(Path(ROOT), old)
        assert row["status"] == "PASS" and "same-build=False" in row["detail"]
    sample = os.path.join(ROOT, "papers", "sworn", "sworn_action_sample.HELD.sworn-receipt.json")
    if os.path.exists(sample):
        row = stranger.check_receipt(Path(ROOT), sample)
        assert row["status"] == "SKIP" and "not in the tree" in row["detail"]


def test_seals_skip_when_no_anchor_exists_or_the_network_is_off(tmp_path):
    s = stranger.step_seals(tmp_path, network=True)
    assert s["status"] == "SKIP" and "no anchor exists yet" in s["detail"]
    (tmp_path / "papers" / "charon").mkdir(parents=True)
    (tmp_path / "papers" / "charon" / "anchors.jsonl").write_text("{}\n", encoding="utf-8")
    s = stranger.step_seals(tmp_path, network=False)
    assert s["status"] == "SKIP" and "--network" in s["detail"]


def test_run_selects_steps_renders_every_step_and_the_cli_exits_by_failure(tmp_path):
    rep = stranger.run(ROOT, only=["checkout", "draw", "reading"])
    assert rep["schema"] == stranger.SCHEMA and rep["commit"] and rep["failed"] == [] and rep["verdict"] == "NOTHING FAILED"
    assert rep["steps"]["ferry_log"]["status"] == "SKIP" and rep["steps"]["sworn"]["status"] == "SKIP"
    text = stranger.render(rep)
    assert all(name in text for name in stranger.STEPS)
    # not a git checkout: the checkout step fails and the exit code says so
    out = tmp_path / "rep.json"
    r = subprocess.run([sys.executable, "-m", "styxx.stranger", "--repo", str(tmp_path), "--only", "checkout", "--json", str(out)],
                       capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert r.returncode == 1, r.stdout + r.stderr
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["failed"] == ["checkout"] and "names no commit" in rep["steps"]["checkout"]["detail"]
    r = subprocess.run([sys.executable, "-m", "styxx.stranger", "--repo", ROOT, "--only", "nonsense"],
                       capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert r.returncode == 2
