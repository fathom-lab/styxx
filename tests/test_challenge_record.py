# -*- coding: utf-8 -*-
"""styxx.challenge v1 — the three refusals, the split agreement, the receipt beside the record, and
the record's honest status as a self-report. Every case here was produced by the 2026-09-13 red team
against v0, which paid (by the letter of BOUNTY.md) for a shallow clone, a renamed copy, and a
modified verifier. The version-skew cases were added when the 7.48.0 bump made every 7.47.0 receipt
issued by the current sworn.py read as "a span-level difference"."""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from styxx import challenge, sworn
from styxx._version import __version__ as RUNNING

DOC = "papers/checksum/RESULT_checksum_smollm_quant_2026_09_13.md"
RCPT = DOC.replace(".md", ".sworn-receipt.json")
OTHER = "0.0.0+not-this-release"     # a styxx version no build of this tree carries
assert OTHER != RUNNING

_LAB_COMMIT = json.load(open(RCPT, encoding="utf-8")).get("commit", "")


@pytest.fixture(autouse=True, scope="module")
def _the_commit_the_receipt_names(full_git_history):
    # CI checks out at fetch-depth 1 and the receipt names an ancestor of HEAD. conftest's session
    # fixture `full_git_history` unshallows before any test runs, and asking for it here puts this
    # check after it. The check used to run at import, which is collection, before the unshallow, so
    # CI skipped this whole file. If the commit is still absent (a shallow clone with no remote to
    # fetch from), skip with the reason, the way tests/test_receipt_provenance_audit.py does: never
    # fail a clone for being shallow, and never pass it either. In CI,
    # tests/test_suite_preconditions.py fails when the unshallow does.
    if not challenge._has_commit(".", _LAB_COMMIT):
        pytest.skip(f"the receipt names {_LAB_COMMIT[:12]}, which this checkout does not have (a shallow clone "
                    "that could not be unshallowed); the challenge tests need full history")


def _lab():
    return json.load(open(RCPT, encoding="utf-8"))


def _issued_by(version, tmp_path, *, edit=None, name="lab.json"):
    """The committed receipt as styxx `version`, with this sworn.py, would have issued it: the version
    stamp replaced, `edit` applied, and the digest re-issued by styxx.sworn.issue_receipt itself.
    Written under tmp_path; the committed receipt is never touched."""
    lab = _lab()
    lab["verifier"] = dict(lab["verifier"], styxx_version=version)
    if edit is not None:
        edit(lab)
    lab["digest"] = sworn.issue_receipt(lab, timestamp=lab["timestamp"])["digest"]
    p = tmp_path / name
    p.write_text(json.dumps(lab), encoding="utf-8")
    return p


def _a_span_moved(lab):
    # one span now names different resolved bytes. Every span is still HELD and the document verdict
    # is unchanged, so only the span-level comparison can see it.
    lab["spans"][0]["resolved_sha256"] = "0" * 64


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
    # the lab's receipt as the running release would have issued it: every field but the version
    # stamp is the committed one, so this replicates exactly when nothing else moved
    mine = tmp_path / "mine.json"
    rec = challenge.run(DOC, str(_issued_by(RUNNING, tmp_path)), repo=".", out=str(mine))
    assert rec["agree"] is True and rec["agree_digest"] is True and rec["agree_verdict"] is True
    assert rec["same_build"] is True and rec["lab_build"] == rec["my_build"]
    assert rec["lab_styxx_version"] == rec["my_styxx_version"] == RUNNING and rec["version_skew"] is False
    assert rec["lab_digest_reissues"] is True and rec["agree_without_version"] is True
    assert rec["mine_receipt_sha256"] == challenge._sha(str(mine))
    assert rec["lab_document"] == "RESULT_checksum_smollm_quant_2026_09_13.md"
    assert rec["document_sha256_at_commit"] is not None
    assert rec["why"] == ""
    assert rec["record_sha256"] == challenge.record_sha256(rec)


def test_a_tampered_lab_verdict_is_a_challenge_that_says_which_half_disagrees(tmp_path):
    lab = json.loads(_issued_by(RUNNING, tmp_path).read_text(encoding="utf-8"))
    lab["document_verdict"] = "SWORN-FAILED"     # edited after issue: the digest is left as issued
    p = tmp_path / "lab.json"
    p.write_text(json.dumps(lab))
    rec = challenge.run(DOC, str(p), repo=".", out=str(tmp_path / "mine.json"))
    assert rec["agree"] is False
    assert rec["agree_verdict"] is False and rec["agree_digest"] is True
    assert rec["lab_digest_reissues"] is False
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
    forged["agree"] = not rec["agree"]
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
    same = _issued_by(RUNNING, tmp_path, name="same.json")
    assert challenge.main([DOC, str(same), "--repo", ".", "--out", str(tmp_path / "rep.json")]) == 0


def test_a_version_only_difference_on_a_same_build_receipt_is_version_skew_not_a_span_level_difference(
        tmp_path, capsys):
    out = tmp_path / "skew.json"
    code = challenge.main([DOC, str(_issued_by(OTHER, tmp_path)), "--repo", ".", "--out", str(out)])
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["same_build"] is True and rec["agree_verdict"] is True
    assert rec["lab_digest_reissues"] is True and rec["agree_without_version"] is True
    assert rec["version_skew"] is True
    assert rec["lab_styxx_version"] == OTHER and rec["my_styxx_version"] == RUNNING
    assert rec["why"].startswith("version skew, not a span-level difference")
    assert OTHER in rec["why"] and RUNNING in rec["why"]
    assert "the verdict and every span agree" in rec["why"]
    # `agree` is digest and verdict, and the digest was not reproduced byte for byte, so it stays a
    # CHALLENGE with exit 3, one whose `why` is an instruction (the module docstring says why)
    assert rec["agree_digest"] is False and rec["agree"] is False and code == 3
    assert "version_skew=True" in capsys.readouterr().out


def test_a_span_level_difference_under_the_same_version_is_still_one(tmp_path):
    p = _issued_by(RUNNING, tmp_path, edit=_a_span_moved)
    rec = challenge.run(DOC, str(p), repo=".", out=str(tmp_path / "mine.json"))
    assert rec["same_build"] is True and rec["agree_verdict"] is True and rec["lab_digest_reissues"] is True
    assert rec["lab_styxx_version"] == rec["my_styxx_version"] == RUNNING
    assert rec["agree_without_version"] is False and rec["version_skew"] is False and rec["agree"] is False
    assert rec["why"] == "the verdict agrees but the digest differs: a span-level difference; compare the two receipts"


def test_a_span_level_difference_across_versions_is_not_excused_as_version_skew(tmp_path):
    p = _issued_by(OTHER, tmp_path, edit=_a_span_moved)
    rec = challenge.run(DOC, str(p), repo=".", out=str(tmp_path / "mine.json"))
    assert rec["same_build"] is True and rec["agree_verdict"] is True and rec["lab_digest_reissues"] is True
    assert rec["lab_styxx_version"] == OTHER and rec["my_styxx_version"] == RUNNING
    assert rec["agree_without_version"] is False and rec["version_skew"] is False and rec["agree"] is False
    assert rec["why"].startswith("the verdict agrees but the digest differs: a span-level difference")
    assert OTHER in rec["why"] and RUNNING in rec["why"]
    assert "still differ with the version set aside" in rec["why"]


def test_a_lab_receipt_edited_after_issue_is_named_not_excused_as_version_skew(tmp_path):
    # the committed body under another version, with a digest that is not its own. The two bodies
    # differ only by the version, so a check that skipped the lab's own digest would call this skew.
    lab = json.loads(_issued_by(OTHER, tmp_path).read_text(encoding="utf-8"))
    lab["digest"] = "0" * 64
    p = tmp_path / "lab.json"
    p.write_text(json.dumps(lab))
    rec = challenge.run(DOC, str(p), repo=".", out=str(tmp_path / "mine.json"))
    assert rec["agree_without_version"] is True and rec["lab_digest_reissues"] is False
    assert rec["version_skew"] is False and rec["agree"] is False
    assert "does not re-issue to its own digest" in rec["why"] and "span-level" not in rec["why"]
