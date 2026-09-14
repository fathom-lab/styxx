# -*- coding: utf-8 -*-
"""styxx.stranger runs the checks a person who does not trust this lab runs (papers/checksum/STRANGER.md)
and prints one table. These tests pin: the ferry log passes against its own head (one real
re-derivation of every line, cached for the module), refuses a head that is not the full 64 hex before
spending 150 s, and reads mismatch / no-head / tamper on a stub; a document edited beside an untouched
sidecar FAILS the sworn step (the red team of 2026-09-14 found `check` on a sidecar never opens the .md);
document verdicts are tallied, not hidden; a tree whose tracked files differ from the commit FAILS unless
--allow-dirty; every beacon-drawn certs file re-derives and a forged fingerprint draw FAILS; a committed
scorecard written for other certs bytes FAILS; the CLI validates --only and exits by failure.

The verification of the night repairs (2026-09-14) added: every committed scorecard is accounted for — an
unreadable one, two current-schema cards for one certs file, a card naming no certs file the step reads,
all FAIL, and a card of another schema is an info row (STR-1); checkout runs whatever --only says (STR-2);
drawn certs with no fingerprints beside them FAIL (STR-3); a sidecar with no .md, and a check line with no
document=, PASS with an info row (STR-4); an unreadable certs file FAILS the reading (STR-5); and the
reading takes the seal's beacon from the seals step, or counts no beacon-draw card as a result (S1)."""
from __future__ import annotations

import json
import os
import re
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


# --- the verification of the night repairs, 2026-09-14 ---------------------------------------------------------

def test_checkout_runs_on_every_invocation_whatever_only_says(monkeypatch):
    rep = stranger.run(ROOT, only=["recipe"], allow_dirty=True)          # --only leaves checkout out; it runs anyway
    assert rep["steps"]["checkout"]["status"] == "PASS" and "not selected" not in rep["steps"]["checkout"]["detail"]
    assert rep["commit"] and re.fullmatch(r"[0-9a-f]{40,64}", rep["commit"])
    assert rep["steps"]["draw"]["status"] == "SKIP"

    def fake_git(repo, *args):
        return "abc123" * 7 if args[0] == "rev-parse" else " M papers/checksum/beacon_draw_certs_dryrun_qwen0.5b.json"
    monkeypatch.setattr(stranger, "_git", fake_git)
    for only in (["recipe"], ["draw", "reading"]):
        rep = stranger.run(ROOT, only=only)
        assert rep["failed"] == ["checkout"] and rep["verdict"] == "FAILED: checkout"
        assert rep["commit"] == "abc123" * 7 and rep["dirty"] is True


def _reading_tree(tmp_path):
    d = tmp_path / "papers" / "checksum"
    d.mkdir(parents=True)
    for f in ("score.py", "deploy_quant_certs_dryrun_qwen0.5b.json", "deploy_quant_scorecard_v2_dryrun_qwen0.5b.json"):
        shutil.copyfile(os.path.join(CK, f), d / f)
    card = json.loads((d / "deploy_quant_scorecard_v2_dryrun_qwen0.5b.json").read_text(encoding="utf-8"))
    return d, card


def _render_reading(s):
    return stranger.render({"repo": "r", "commit": "c", "dirty": False, "steps": {"reading": s}, "verdict": "v", "seconds": 0})


def test_an_unreadable_scorecard_fails_the_reading_with_a_row(tmp_path):
    d, _ = _reading_tree(tmp_path)
    (d / "deploy_quant_scorecard_zz.json").write_text("{not json", encoding="utf-8")
    (d / "deploy_quant_scorecard_zz_list.json").write_text("[]", encoding="utf-8")
    s = stranger.step_reading(tmp_path)
    assert s["status"] == "FAIL"
    rows = {r["scorecard"]: r for r in s["scorecards"]}
    assert rows["papers/checksum/deploy_quant_scorecard_zz.json"]["status"] == "FAIL"
    assert "unreadable scorecard" in rows["papers/checksum/deploy_quant_scorecard_zz.json"]["detail"]
    assert "not a JSON object" in rows["papers/checksum/deploy_quant_scorecard_zz_list.json"]["detail"]
    assert "FAIL  papers/checksum/deploy_quant_scorecard_zz.json: unreadable scorecard" in _render_reading(s)


def test_two_current_schema_scorecards_naming_one_certs_file_fail(tmp_path):
    d, card = _reading_tree(tmp_path)
    forged = dict(card, run_reading="FORGED HELD", counts_as_result=True)       # sorts before the real card
    (d / "deploy_quant_scorecard_a_forged.json").write_text(json.dumps(forged), encoding="utf-8")
    s = stranger.step_reading(tmp_path)
    row = next(r for r in s["files"] if r["certs"] == "papers/checksum/deploy_quant_certs_dryrun_qwen0.5b.json")
    assert s["status"] == "FAIL" and row["status"] == "FAIL" and "committed_scorecard" not in row
    assert row["committed_scorecards"] == ["papers/checksum/deploy_quant_scorecard_a_forged.json",
                                           "papers/checksum/deploy_quant_scorecard_v2_dryrun_qwen0.5b.json"]
    assert "2 current-schema scorecards name these certs" in row["detail"]


def test_a_current_schema_scorecard_naming_no_certs_file_the_step_reads_fails(tmp_path):
    d, card = _reading_tree(tmp_path)
    (d / "deploy_quant_scorecard_v2_gone.json").write_text(json.dumps(dict(card, certs_file="papers/checksum/deploy_quant_certs_gone.json")), encoding="utf-8")
    (d / "deploy_quant_scorecard_v2_scorer.json").write_text(json.dumps(dict(card, certs_file="papers/checksum/score.py")), encoding="utf-8")
    nameless = dict(card)
    del nameless["certs_file"]
    (d / "deploy_quant_scorecard_v2_nameless.json").write_text(json.dumps(nameless), encoding="utf-8")
    s = stranger.step_reading(tmp_path)
    rows = {r["scorecard"]: r for r in s["scorecards"]}
    assert s["status"] == "FAIL" and all(r["status"] == "FAIL" for r in rows.values()) and len(rows) == 3
    assert "not in the tree" in rows["papers/checksum/deploy_quant_scorecard_v2_gone.json"]["detail"]
    assert "not a certs file this step reads" in rows["papers/checksum/deploy_quant_scorecard_v2_scorer.json"]["detail"]
    assert "no certs_file" in rows["papers/checksum/deploy_quant_scorecard_v2_nameless.json"]["detail"]
    real = next(r for r in s["files"] if r["certs"] == "papers/checksum/deploy_quant_certs_dryrun_qwen0.5b.json")
    assert real["status"] == "PASS" and real["committed_scorecard_matches"] is True


def test_a_certs_file_spelled_with_backslashes_is_the_same_file_and_is_compared(tmp_path):
    d, card = _reading_tree(tmp_path)
    path = d / "deploy_quant_scorecard_v2_dryrun_qwen0.5b.json"
    spelled = "./papers" + chr(92) + "checksum" + chr(92) + "deploy_quant_certs_dryrun_qwen0.5b.json"
    path.write_text(json.dumps(dict(card, certs_file=spelled)), encoding="utf-8")
    s = stranger.step_reading(tmp_path)
    assert s["status"] == "PASS" and "1 committed scorecards compared" in s["detail"] and s["scorecards"] == []
    path.write_text(json.dumps(dict(card, certs_file=spelled, run_reading="FORGED HELD")), encoding="utf-8")
    s = stranger.step_reading(tmp_path)
    assert s["status"] == "FAIL" and "not what the scorer reads today" in s["files"][0]["detail"]


def test_a_scorecard_of_another_schema_is_an_info_row_never_silently_dropped(tmp_path):
    d, card = _reading_tree(tmp_path)
    (d / "deploy_quant_scorecard_v2_dryrun_qwen0.5b.json").write_text(
        json.dumps(dict(card, schema="styxx.checksum/scorecard/v1", run_reading="FORGED HELD")), encoding="utf-8")
    s = stranger.step_reading(tmp_path)
    assert s["status"] == "PASS" and "0 committed scorecards compared, 1 of another schema listed and not compared" in s["detail"]
    (row,) = s["scorecards"]
    assert row["status"] == "INFO" and "not compared" in row["detail"] and "scorecard/v1" in row["detail"]
    assert "info  papers/checksum/deploy_quant_scorecard_v2_dryrun_qwen0.5b.json: schema" in _render_reading(s)


def test_an_unreadable_certs_file_fails_the_reading_with_a_row(tmp_path):
    d, _ = _reading_tree(tmp_path)
    (d / "deploy_quant_certs_broken.json").write_text("{not json", encoding="utf-8")
    s = stranger.step_reading(tmp_path)
    row = next(r for r in s["files"] if r["certs"] == "papers/checksum/deploy_quant_certs_broken.json")
    assert s["status"] == "FAIL" and row["status"] == "FAIL" and "unreadable certs file" in row["detail"]
    assert "failing: papers/checksum/deploy_quant_certs_broken.json" in s["detail"]


def test_a_beacon_drawn_certs_file_with_no_fingerprints_beside_it_fails(tmp_path):
    d = tmp_path / "papers" / "checksum"
    d.mkdir(parents=True)
    shutil.copyfile(os.path.join(CK, "beacon_draw_certs_dryrun_qwen0.5b.json"), d / "beacon_draw_certs_dryrun_qwen0.5b.json")
    s = stranger.step_draw(tmp_path)
    assert s["status"] == "FAIL" and "no beacon_draw_fingerprints_dryrun_qwen0.5b.json beside it" in s["files"][0]["detail"]
    assert "cannot be checked against the ids" in s["files"][0]["detail"]
    fp = d / "beacon_draw_fingerprints_dryrun_qwen0.5b.json"
    fp.write_text("{}", encoding="utf-8")
    s = stranger.step_draw(tmp_path)
    assert s["status"] == "FAIL" and "carries no fingerprint" in s["files"][0]["detail"]
    shutil.copyfile(os.path.join(CK, "beacon_draw_fingerprints_dryrun_qwen0.5b.json"), fp)
    assert stranger.step_draw(tmp_path)["status"] == "PASS"


def test_a_sidecar_with_no_document_and_a_line_with_no_document_field_pass_with_info_rows(tmp_path, monkeypatch):
    rc, md = _receipt_triple(tmp_path)
    md.unlink()
    _fake_check(monkeypatch, "VERIFIED  digest=True verdict-reproduces=True same-build=True  document=SWORN-HELD")
    s = stranger.step_sworn(tmp_path)
    (row,) = s["receipts"]
    assert s["status"] == "PASS" and row["status"] == "PASS" and row["document_matches_sidecar"] is None
    assert "no document was compared" in row["document_note"] and "1 with no .md beside the sidecar" in s["detail"]
    rep = {"repo": "r", "commit": "c", "dirty": False, "steps": {"sworn": s}, "verdict": "NOTHING FAILED", "seconds": 0}
    assert "info  papers/checksum/PREREG_checksum_beacon_draw_2026_09_14.sworn-receipt.json: no papers/checksum/" in stranger.render(rep)
    shutil.copyfile(os.path.join(CK, "PREREG_checksum_beacon_draw_2026_09_14.md"), md)
    _fake_check(monkeypatch, "VERIFIED  digest=True verdict-reproduces=True same-build=True")
    s = stranger.step_sworn(tmp_path)
    (row,) = s["receipts"]
    assert row["status"] == "PASS" and row["document_verdict"] is None and s["document_verdicts"] == {"?": 1}
    rep["steps"] = {"sworn": s}
    assert "carries no document= field" in stranger.render(rep)


_STUB_SCORER = '''
SCHEMA = "styxx.checksum/scorecard/v2"
BANDS = {"deploy_quant": {"prereg": "PREREG_dq.md", "sealed_blob": "a" * 64},
         "beacon_draw": {"prereg": "PREREG_bd.md", "sealed_blob": "d" * 64}}


def score(certs, prereg, expect_beacon=None, expect_blob=None, hand_set=None, portability=None):
    # a scorer from before S1: it counts whatever the certs say, beacon or not
    return {"hypotheses": {}, "gates": {}, "run_reading": prereg + " beacon=" + str(expect_beacon),
            "counts_as_result": certs.get("counts") is True}
'''


def test_the_reading_takes_the_seals_beacon_for_beacon_draw_certs_or_counts_none(tmp_path, monkeypatch):
    d = tmp_path / "papers" / "checksum"
    d.mkdir(parents=True)
    (d / "score.py").write_text(_STUB_SCORER, encoding="utf-8")
    (d / "beacon_draw_certs_x.json").write_text(json.dumps({"prereg": "PREREG_bd.md", "counts": True}), encoding="utf-8")
    (d / "deploy_quant_certs_x.json").write_text(json.dumps({"prereg": "PREREG_dq.md", "counts": True}), encoding="utf-8")
    monkeypatch.setattr(stranger, "step_checkout", lambda repo, allow_dirty=False: {"status": "PASS", "commit": "c" * 40, "dirty": False, "detail": "stub"})
    seals = {"status": "PASS", "lines": []}
    monkeypatch.setattr(stranger, "step_seals", lambda repo, network: seals)

    def reading(only=("seals", "reading")):
        rep = stranger.run(tmp_path, only=list(only))
        s = rep["steps"]["reading"]
        return s, {os.path.basename(r["certs"]): r for r in s["files"]}

    seals["lines"] = [{"n": 6, "kind": "sealed-prereg", "digest": "D" * 64, "status": "ANCHORED", "beacon": "B" * 64}]
    s, rows = reading()
    assert rows["beacon_draw_certs_x.json"]["reading"] == "beacon_draw beacon=" + "b" * 64
    assert rows["beacon_draw_certs_x.json"]["counts_as_result"] is True and rows["beacon_draw_certs_x.json"]["expect_beacon"] == "b" * 64
    assert rows["deploy_quant_certs_x.json"]["reading"] == "deploy_quant beacon=None"      # only beacon-draw certs take it
    assert s["expect_beacon"] == "b" * 64 and "read with the seal's beacon bbbbbbbbbbbb" in s["detail"]

    seals["lines"] = [{"n": 5, "kind": "sealed-prereg", "digest": "e" * 64, "status": "ANCHORED", "beacon": "1" * 64},
                      {"n": 6, "kind": "sealed-prereg", "digest": "d" * 64, "status": "EARLIEST_UNKNOWN", "beacon": "2" * 64},
                      {"n": 7, "kind": "sealed-canaries", "digest": "d" * 64, "status": "ANCHORED", "beacon": "3" * 64}]
    for only in (("seals", "reading"), ("reading",)):
        s, rows = reading(only)
        bd = rows["beacon_draw_certs_x.json"]
        assert bd["reading"] == "beacon_draw beacon=None" and bd["counts_as_result"] is False and bd["scorer_counts_as_result"] is True
        assert "NOT counted here" in bd["detail"] and s["expect_beacon"] is None
        assert "no beacon-draw card can count as a result without the seal's beacon (--network)" in s["detail"]
        assert "1 count as a result" in s["detail"]                                        # the deploy_quant stub only
    assert "the seals step did not run (not selected (--only))" in s["detail"]


def test_the_real_scorer_takes_the_beacon_by_keyword_and_a_committed_beaconless_instrument_check_still_matches():
    ck_score = stranger._load_scorer(Path(ROOT))
    seals = {"status": "PASS", "lines": [{"kind": "sealed-prereg", "status": "ANCHORED", "beacon": "ab" * 32,
                                          "digest": ck_score.BANDS["beacon_draw"]["sealed_blob"]}]}
    s = stranger.step_reading(Path(ROOT), seals)
    assert s["status"] == "PASS" and s["expect_beacon"] == "ab" * 32
    bd = next(r for r in s["files"] if r["certs"] == "papers/checksum/beacon_draw_certs_dryrun_qwen0.5b.json")
    # the beacon reached the scorer: the reading differs from the beaconless card, which matches only because it claims no result
    assert bd["expect_beacon"] == "ab" * 32 and bd["committed_scorecard_matches"] is True and "without the seal's beacon" in bd["committed_scorecard_note"]
    assert bd["counts_as_result"] is False and "K5" in bd["reading"]
