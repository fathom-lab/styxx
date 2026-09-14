# -*- coding: utf-8 -*-
"""styxx.challenge — a record that reproduces the lab's receipt says REPLICATION; one that does not says CHALLENGE."""
from __future__ import annotations

import json
import subprocess

import pytest
import sys

from styxx import challenge

DOC = "papers/checksum/RESULT_checksum_smollm_quant_2026_09_13.md"
RCPT = DOC.replace(".md", ".sworn-receipt.json")

# CI checks out at fetch-depth 1; the receipt names an ancestor of HEAD, and without it nothing here can
# re-derive. Skip with the reason, the way tests/test_receipt_provenance_audit.py does — never fail a
# clone for being shallow, and never pass it either.
import subprocess as _sp
_LAB_COMMIT = json.load(open(RCPT, encoding="utf-8")).get("commit", "")
if _sp.run(["git", "cat-file", "-e", f"{_LAB_COMMIT}^{{commit}}"], capture_output=True).returncode != 0:
    pytest.skip(f"the receipt names {_LAB_COMMIT[:12]}, which this checkout does not have (a shallow clone); "
                "the challenge tests need full history", allow_module_level=True)


def test_the_lab_receipt_replicates_from_its_own_repo(tmp_path):
    rec = challenge.run(DOC, RCPT, repo=".", out=str(tmp_path / "mine.json"))
    assert rec["agree"] is True
    assert rec["same_build"] is True
    assert rec["lab_verdict"] == rec["my_verdict"] == "SWORN-HELD"
    assert len(rec["record_sha256"]) == 64


def test_a_tampered_lab_receipt_is_a_challenge(tmp_path):
    lab = json.load(open(RCPT))
    lab["digest"] = "0" * 64
    p = tmp_path / "lab.json"; p.write_text(json.dumps(lab))
    rec = challenge.run(DOC, str(p), repo=".", out=str(tmp_path / "mine.json"))
    assert rec["agree"] is False
