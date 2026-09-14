# -*- coding: utf-8 -*-
"""styxx.stranger runs the checks a person who does not trust this lab runs (papers/checksum/STRANGER.md)
and prints one table. These tests pin: the ferry log passes against its own head (one real
re-derivation of every line, cached for the module), refuses a head that is not the full 64 hex before
spending 150 s, and reads mismatch / no-head / tamper on a stub; a document edited beside an untouched
sidecar FAILS the sworn step (the red team of 2026-09-14 found `check` on a sidecar never opens the .md);
document verdicts are tallied, not hidden; a tree whose tracked files differ from the commit FAILS unless
--allow-dirty; every beacon-drawn certs file re-derives and a forged fingerprint draw FAILS; a committed
scorecard written for other certs bytes FAILS; the CLI validates --only and exits by failure."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from styxx import charon, stranger

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "papers", "charon", "charon.log.jsonl")
CK = os.path.join(ROOT, "papers", "checksum")
PREREG_RECEIPT = os.path.join(CK, "PREREG_checksum_beacon_draw_2026_09_14.sworn-receipt.json")


def _has_history() -> bool:
    """CI checks out shallow; the receipts name commits such a clone may not have, so the steps that
    re-derive at a commit (the ferry log, `sworn check`) cannot run there — skip with the reason; never
    pass or fail a clone for being shallow. v1 probed one commit, which a partial-depth clone could have
    while missing older ones (red team 2026-09-14, ci F2); git's own answer is used now."""
    try:
        r = subprocess.run(["git", "-C", ROOT, "rev-parse", "--is-shallow-repository"], capture_output=True, text=True)
        return r.returncode == 0 and r.stdout.strip() == "false"
    except Exception:  # noqa: BLE001
        return False


needs_history = pytest.mark.skipif(not _has_history(), reason="a shallow clone: the receipts' commits may not be here")


def _head():
    _, entries, _, _ = charon.read_log(Path(LOG))
    return charon.head_of(entries)


@pytest.fixture(scope="module")
def ferry_log_pass():
    return stranger.step_ferry_log(Path(ROOT), _head())      # the one real re-derivation of every line


@needs_history
def test_the_ferry_log_passes_against_its_own_head(ferry_log_pass):
    ok = ferry_log_pass
    assert ok["status"] == "PASS" and ok["head_matches"] is True and ok["external_head_checked"] is True
    assert set(ok["by_status"]) >= {"SAME_LINE", "MOVED_VERIFIER"} and ok["chain_broken_at_line"] is None


def test_a_head_that_is_not_the_full_64_hex_is_refused_before_the_log_is_read(monkeypatch):
    def never(*a, **k):
        raise AssertionError("verify_log must not run for a malformed head")
    monkeypatch.setattr(charon, "verify_log", never)
    for bad in (_head()[:16], "xyz", _head() + "0"):
        s = stranger.step_ferry_log(Path(ROOT), bad)
        assert s["status"] == "FAIL" and "full 64-hex head" in s["detail"] and s["external_head_checked"] is False


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
    assert stranger.step_ferry_log(Path(ROOT), real_head.upper())["status"] == "PASS"
    tampered = dict(fake_verify_log(None, None, real_head), by_status={"SAME_LINE": 2, "TAMPER": 1})
    monkeypatch.setattr(charon, "verify_log", lambda *a, **k: tampered)
    assert stranger.step_ferry_log(Path(ROOT), real_head)["status"] == "FAIL"


def _receipt_triple(tmp_path, stem="PREREG_checksum_beacon_draw_2026_09_14"):
    d = tmp_path / "papers" / "checksum"
    d.mkdir(parents=True)
    for ext in (".md", ".sworn.json", ".sworn-receipt.json"):
        shutil.copyfile(os.path.join(CK, stem + ext), d / (stem + ext))
    return d / (stem + ".sworn-receipt.json"), d / (stem + ".md")


def _fake_check(monkeypatch, line):
    class R:
        returncode, stdout, stderr = 0, line + "\n", ""
    monkeypatch.setattr(stranger.subprocess, "run", lambda *a, **k: R())


def test_a_document_edited_beside_an_untouched_sidecar_fails_the_sworn_step(tmp_path, monkeypatch):
    rc, md = _receipt_triple(tmp_path)
    _fake_check(monkeypatch, "VERIFIED  digest=True verdict-reproduces=True same-build=True  document=SWORN-HELD")
    row = stranger.check_receipt(tmp_path, str(rc))
    assert row["status"] == "PASS" and row["document_matches_sidecar"] is True and row["document_verdict"] == "SWORN-HELD"
    md.write_bytes(md.read_bytes().replace(b"778", b"779", 1))
    row = stranger.check_receipt(tmp_path, str(rc))
    assert row["status"] == "FAIL" and row["document_matches_sidecar"] is False and "not the document this receipt swore to" in row["detail"]


def test_document_verdicts_are_tallied_and_rendered_not_hidden(tmp_path, monkeypatch):
    rc, _ = _receipt_triple(tmp_path)
    _fake_check(monkeypatch, "VERIFIED  digest=True verdict-reproduces=True same-build=False  document=SWORN-FAILED")
    s = stranger.step_sworn(tmp_path)
    assert s["status"] == "PASS" and s["document_verdicts"] == {"SWORN-FAILED": 1} and "SWORN-FAILED 1" in s["detail"]
    rep = {"repo": "r", "commit": "c", "dirty": False, "steps": {"sworn": s}, "verdict": "NOTHING FAILED", "seconds": 0}
    assert "the document it swears to reads SWORN-FAILED" in stranger.render(rep)


@needs_history
def test_one_real_receipt_re_checks_verified_and_a_temporary_sample_is_not_a_failure():
    row = stranger.check_receipt(Path(ROOT), PREREG_RECEIPT)
    assert row["status"] == "PASS" and row["detail"].startswith("VERIFIED") and row["target"].endswith(".sworn.json")
    assert row["document_matches_sidecar"] is True
    old = os.path.join(ROOT, "papers", "sworn", "RESULT_sworn_v01_ships_2026_09_01.sworn-receipt.json")
    if os.path.exists(old):
        row = stranger.check_receipt(Path(ROOT), old)
        assert row["status"] == "PASS" and "same-build=False" in row["detail"]
    sample = os.path.join(ROOT, "papers", "sworn", "sworn_action_sample.HELD.sworn-receipt.json")
    if os.path.exists(sample):
        row = stranger.check_receipt(Path(ROOT), sample)
        assert row["status"] == "SKIP" and "not in the tree" in row["detail"]


def test_a_tree_whose_tracked_files_differ_from_the_commit_fails_unless_allowed(monkeypatch):
    def fake_git(repo, *args):
        return "abc123" * 7 if args[0] == "rev-parse" else " M papers/checksum/beacon_draw_certs_dryrun_qwen0.5b.json"
    monkeypatch.setattr(stranger, "_git", fake_git)
    s = stranger.step_checkout(Path(ROOT))
    assert s["status"] == "FAIL" and s["dirty"] is True and "--allow-dirty" in s["detail"]
    s = stranger.step_checkout(Path(ROOT), allow_dirty=True)
    assert s["status"] == "PASS" and "checked knowingly" in s["detail"]


def test_every_beacon_drawn_certs_file_in_the_tree_re_derives_and_a_forged_fingerprint_draw_fails(tmp_path):
    s = stranger.step_draw(Path(ROOT))
    assert s["status"] == "PASS"
    drawn = [f for f in s["files"] if "beacon" in f["detail"]]
    assert len(drawn) >= 2 and all(f["status"] == "PASS" and "fingerprints re-checked" in f["detail"] for f in drawn)
    d = tmp_path / "papers" / "checksum"
    d.mkdir(parents=True)
    shutil.copyfile(os.path.join(CK, "beacon_draw_certs_dryrun_qwen0.5b.json"), d / "beacon_draw_certs_dryrun_qwen0.5b.json")
    fps = json.load(open(os.path.join(CK, "beacon_draw_fingerprints_dryrun_qwen0.5b.json"), encoding="utf-8"))
    fps["Q4"]["draw"] = dict(fps["Q4"]["draw"], beacon="e" * 64)
    (d / "beacon_draw_fingerprints_dryrun_qwen0.5b.json").write_text(json.dumps(fps), encoding="utf-8")
    s = stranger.step_draw(tmp_path)
    assert s["status"] == "FAIL" and "fingerprint Q4" in s["files"][0]["detail"]


def test_every_known_certs_file_is_read_and_the_committed_scorecards_match():
    s = stranger.step_reading(Path(ROOT))
    assert s["status"] == "PASS"
    read = [f for f in s["files"] if f["status"] != "SKIP"]
    assert len(read) >= 3 and all(f["counts_as_result"] is False for f in read)   # nothing in the tree is the experiment
    compared = [f for f in read if "committed_scorecard" in f]
    assert len(compared) >= 2 and all(f["committed_scorecard_matches"] for f in compared)
    assert all("_v2_" in f["committed_scorecard"] for f in compared)             # v1 cards are history, not compared


def test_a_committed_scorecard_written_for_other_certs_bytes_fails(tmp_path):
    d = tmp_path / "papers" / "checksum"
    d.mkdir(parents=True)
    for f in ("score.py", "deploy_quant_certs_dryrun_qwen0.5b.json", "deploy_quant_scorecard_v2_dryrun_qwen0.5b.json"):
        shutil.copyfile(os.path.join(CK, f), d / f)
    assert stranger.step_reading(tmp_path)["status"] == "PASS"
    card = json.loads((d / "deploy_quant_scorecard_v2_dryrun_qwen0.5b.json").read_text(encoding="utf-8"))
    card["certs_sha256"] = "0" * 64
    (d / "deploy_quant_scorecard_v2_dryrun_qwen0.5b.json").write_text(json.dumps(card), encoding="utf-8")
    s = stranger.step_reading(tmp_path)
    assert s["status"] == "FAIL" and "other certs bytes" in s["files"][0]["detail"]


def test_seals_skip_when_no_anchor_exists_or_the_network_is_off(tmp_path):
    s = stranger.step_seals(tmp_path, network=True)
    assert s["status"] == "SKIP" and "no anchor exists yet" in s["detail"]
    (tmp_path / "papers" / "charon").mkdir(parents=True)
    (tmp_path / "papers" / "charon" / "anchors.jsonl").write_text("{}\n", encoding="utf-8")
    s = stranger.step_seals(tmp_path, network=False)
    assert s["status"] == "SKIP" and "--network" in s["detail"]


def test_run_selects_steps_renders_every_step_and_the_cli_exits_by_failure(tmp_path):
    rep = stranger.run(ROOT, only=["checkout", "draw", "reading"], allow_dirty=True)
    assert rep["schema"] == stranger.SCHEMA and rep["commit"] and rep["failed"] == [] and rep["verdict"] == "NOTHING FAILED"
    assert rep["steps"]["ferry_log"]["status"] == "SKIP" and rep["steps"]["sworn"]["status"] == "SKIP"
    text = stranger.render(rep)
    assert all(name in text for name in stranger.STEPS)
    out = tmp_path / "rep.json"
    r = subprocess.run([sys.executable, "-m", "styxx.stranger", "--repo", str(tmp_path), "--only", "checkout", "--json", str(out)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300, cwd=ROOT)
    assert r.returncode == 1, r.stdout + r.stderr
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["failed"] == ["checkout"] and "names no commit" in rep["steps"]["checkout"]["detail"]
    r = subprocess.run([sys.executable, "-m", "styxx.stranger", "--repo", ROOT, "--only", "nonsense"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, cwd=ROOT)
    assert r.returncode == 2
