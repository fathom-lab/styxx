# -*- coding: utf-8 -*-
"""styxx.challenge — a record that reproduces the lab's receipt says REPLICATION; one that does not says CHALLENGE."""
from __future__ import annotations

import json
import subprocess
import sys

from styxx import challenge

DOC = "papers/checksum/RESULT_checksum_smollm_quant_2026_09_13.md"
RCPT = DOC.replace(".md", ".sworn-receipt.json")


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
