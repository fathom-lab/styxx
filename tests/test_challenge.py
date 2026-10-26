# -*- coding: utf-8 -*-
"""styxx.challenge — a record that reproduces the lab's receipt says REPLICATION; one that does not says CHALLENGE."""
from __future__ import annotations

import json

import pytest

from styxx import challenge

DOC = "papers/checksum/RESULT_checksum_smollm_quant_2026_09_13.md"
RCPT = DOC.replace(".md", ".sworn-receipt.json")

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


def test_the_lab_receipt_replicates_from_its_own_repo(tmp_path):
    # the committed receipt, as committed, against the styxx this checkout runs. Every span and the
    # verdict reproduce. The verifier build is sworn.py and the styxx version. While both are the ones
    # that issued the receipt the digest reproduces too; after a version bump alone the record says
    # version skew, naming both versions; after sworn.py itself moved (the /v2 step of 2026-10-24,
    # papers/sworn/NOTE_certifies_correction_2026_10_24.md, is the earliest such edit since this test
    # was written, and it pinned an unmoved sworn.py without saying so) the record says the build
    # differs, and the committed receipt still re-derives at its commit under styxx.sworn. In no case
    # does the record take the shape the bounty pays (agree false with same_build true).
    from styxx import sworn
    from styxx._version import __version__ as running
    rec = challenge.run(DOC, RCPT, repo=".", out=str(tmp_path / "mine.json"))
    assert rec["lab_verdict"] == rec["my_verdict"] == "SWORN-HELD"
    assert rec["lab_digest_reissues"] is True
    assert rec["my_styxx_version"] == running
    if rec["lab_build"] == rec["my_build"]:
        assert rec["same_build"] is (rec["lab_styxx_version"] == running)
        assert rec["agree_without_version"] is True
        if rec["lab_styxx_version"] == running:
            assert rec["agree"] is True and rec["version_skew"] is False and rec["why"] == ""
        else:
            assert rec["agree"] is False and rec["version_skew"] is True and rec["same_build"] is False
            assert rec["why"].startswith("version skew, not a disagreement")
            assert rec["lab_styxx_version"] in rec["why"] and running in rec["why"]
    else:
        assert rec["same_build"] is False and rec["agree"] is False and rec["version_skew"] is False
        assert rec["why"].startswith("the verifier build differs, so the digest differs by construction")
        lab = json.load(open(RCPT, encoding="utf-8"))
        side = sworn.load_sidecar(json.load(open(DOC.replace(".md", ".sworn.json"), encoding="utf-8")))
        res = sworn.verify_receipt(lab, sidecar=side, tree=sworn.GitTree(".", side["commit"]))
        assert res["status"] == "VERIFIED" and res["same_verifier_build"] is False, res
    assert not (rec["agree"] is False and rec["same_build"] is True)
    assert len(rec["record_sha256"]) == 64


def test_a_tampered_lab_receipt_is_a_challenge(tmp_path):
    lab = json.load(open(RCPT))
    lab["digest"] = "0" * 64
    p = tmp_path / "lab.json"; p.write_text(json.dumps(lab))
    rec = challenge.run(DOC, str(p), repo=".", out=str(tmp_path / "mine.json"))
    assert rec["agree"] is False
    # a digest that is not the receipt's own is named, never excused as version skew
    assert rec["version_skew"] is False and rec["lab_digest_reissues"] is False
