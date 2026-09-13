# -*- coding: utf-8 -*-
"""styxx.challenge v1 — the three refusals, the split agreement, the receipt beside the record, and
the record's honest status as a self-report. Every case here was produced by the 2026-09-13 red team
against v0, which paid (by the letter of BOUNTY.md) for a shallow clone, a renamed copy, and a
modified verifier."""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from styxx import challenge

DOC = "papers/checksum/RESULT_checksum_smollm_quant_2026_09_13.md"
RCPT = DOC.replace(".md", ".sworn-receipt.json")


def _lab():
    return json.load(open(RCPT, encoding="utf-8"))


def test_a_checkout_without_the_named_commit_is_refused_not_paid(tmp_path):
    lab = _lab()
    lab["commit"] = "0" * 40
    p = tmp_path / "lab.json"
    p.write_text(json.dumps(lab))
    with pytest.raises(SystemExit) as e:
        challenge.run(DOC, str(p), repo=".", out=str(tmp_path / "mine.json"))
    assert "not in" in str(e.value) and "shallow" in str(e.value)


def test_a_renamed_copy_is_refused_because_sworn_digests_the_basename(tmp_path):
    other = tmp_path / "other.md"
    shutil.copyfile(DOC, other)
    with pytest.raises(SystemExit) as e:
        challenge.run(str(other), RCPT, repo=".", out=str(tmp_path / "mine.json"))
    assert "Keep the name" in str(e.value)


def test_a_receipt_with_nothing_held_is_refused(tmp_path, monkeypatch):
    mine = tmp_path / "mine.json"

    def fake_verify(cmd, **kw):
        mine.write_text(json.dumps({"digest": "a" * 64, "document_verdict": "SWORN-HELD",
                                    "counts": {"HELD": 0, "UNRESOLVED": 13}, "verifier": {"sworn_sha256": "b" * 64}}))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(challenge.subprocess, "run", fake_verify)
    monkeypatch.setattr(challenge, "_has_commit", lambda repo, commit: True)
    with pytest.raises(SystemExit) as e:
        challenge.run(DOC, RCPT, repo=".", out=str(mine))
    assert "HELD=0 UNRESOLVED=13" in str(e.value)


def test_an_honest_replication_carries_the_strangers_receipt_and_both_builds(tmp_path):
    mine = tmp_path / "mine.json"
    rec = challenge.run(DOC, RCPT, repo=".", out=str(mine))
    assert rec["agree"] is True and rec["agree_digest"] is True and rec["agree_verdict"] is True
    assert rec["same_build"] is True and rec["lab_build"] == rec["my_build"]
    assert rec["mine_receipt_sha256"] == challenge._sha(str(mine))
    assert rec["lab_document"] == "RESULT_checksum_smollm_quant_2026_09_13.md"
    assert rec["document_sha256_at_commit"] is not None
    assert rec["why"] == ""
    assert rec["record_sha256"] == challenge.record_sha256(rec)


def test_a_tampered_lab_verdict_is_a_challenge_that_says_which_half_disagrees(tmp_path):
    lab = _lab()
    lab["document_verdict"] = "SWORN-FAILED"
    p = tmp_path / "lab.json"
    p.write_text(json.dumps(lab))
    rec = challenge.run(DOC, str(p), repo=".", out=str(tmp_path / "mine.json"))
    assert rec["agree"] is False
    assert rec["agree_verdict"] is False and rec["agree_digest"] is True
    assert "verdict differs" in rec["why"]


def test_a_different_verifier_build_is_named_as_the_reason_before_it_is_called_a_challenge(tmp_path):
    lab = _lab()
    lab["verifier"] = {"sworn_sha256": "f" * 64}
    lab["digest"] = "e" * 64
    p = tmp_path / "lab.json"
    p.write_text(json.dumps(lab))
    rec = challenge.run(DOC, str(p), repo=".", out=str(tmp_path / "mine.json"))
    assert rec["same_build"] is False and rec["agree"] is False
    assert "verifier build differs" in rec["why"]


def test_the_record_is_a_self_report_and_the_module_says_so(tmp_path):
    # the hash names the record's own bytes and nothing else: editing and recomputing validates.
    # This is pinned so no document can claim the record "cannot be edited to say something else".
    rec = challenge.run(DOC, RCPT, repo=".", out=str(tmp_path / "mine.json"))
    forged = dict(rec)
    forged["agree"] = False
    forged["record_sha256"] = challenge.record_sha256(forged)
    assert forged["record_sha256"] != rec["record_sha256"]
    assert challenge.record_sha256(forged) == forged["record_sha256"]
    assert "self-report" in challenge.__doc__


def test_the_cli_writes_lf_bytes_and_the_receipt_beside_the_record_and_exits_3_on_a_challenge(tmp_path):
    lab = _lab()
    lab["digest"] = "0" * 64
    p = tmp_path / "lab.json"
    p.write_text(json.dumps(lab))
    out = tmp_path / "challenge.json"
    code = challenge.main([DOC, str(p), "--repo", ".", "--out", str(out)])
    assert code == 3
    assert b"\r" not in out.read_bytes()
    assert (tmp_path / "challenge.mine.sworn-receipt.json").exists()
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["schema"] == "styxx.challenge/v1" and rec["agree"] is False
    assert challenge.main([DOC, RCPT, "--repo", ".", "--out", str(tmp_path / "rep.json")]) == 0
