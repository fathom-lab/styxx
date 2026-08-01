# -*- coding: utf-8 -*-
"""styxx.seal — the trust seal for agent work. The refusals are the product."""
import json
import subprocess

from styxx.seal import seal, verify_seal


def _repo(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)


def _prereg(tmp_path, commit=True):
    spec = {"gates": {"G1": {"metric": "auc", "op": ">=", "value": 0.9}},
            "outcomes": [{"when": {"G1": True}, "verdict": "SURVIVED"},
                         {"when": {"G1": False}, "verdict": "CLOSED_NEGATIVE"}],
            "smoke_verdict": "INVALID__smoke"}
    p = tmp_path / "PREREG_x.md"
    p.write_text("# P\n\n```gates\n" + json.dumps(spec) + "\n```\n", encoding="utf-8")
    if commit:
        subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-qm", "freeze"], cwd=tmp_path, check=True)
    return p


def _doc_and_receipt(tmp_path, value="0.95"):
    doc = tmp_path / "report.md"
    doc.write_text(f"# report\n\nsome preamble sentence here.\n\n"
                   f"The measured AUC was {value} on the held-out set.\n", encoding="utf-8")
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps({"auc": 0.95}), encoding="utf-8")
    return doc, rp


def test_grounded_document_seals(tmp_path):
    doc, rp = _doc_and_receipt(tmp_path)
    s = seal(doc, [rp])
    assert s.verdict == "SEALED" and not s.refusals
    assert verify_seal(s.to_dict())


def test_ungrounded_claim_refuses_and_names_the_token(tmp_path):
    doc, rp = _doc_and_receipt(tmp_path, value="0.99")   # not in the receipt
    s = seal(doc, [rp])
    assert s.verdict == "REFUSED"
    assert any(r["layer"] == "oath" and r["token"] == "0.99" for r in s.refusals)


def test_protocol_verdict_mismatch_refuses(tmp_path):
    _repo(tmp_path)
    p = _prereg(tmp_path)
    doc, rp = _doc_and_receipt(tmp_path)
    res = tmp_path / "result.json"
    res.write_text(json.dumps({"auc": 0.5, "verdict": "SURVIVED"}), encoding="utf-8")
    s = seal(doc, [rp], preregs=[(p, res)])              # frozen table says CLOSED_NEGATIVE
    assert s.verdict == "REFUSED"
    assert any(r["layer"] == "protocol" and "CLOSED_NEGATIVE" in r["why"]
               for r in s.refusals)


def test_protocol_honest_verdict_seals(tmp_path):
    _repo(tmp_path)
    p = _prereg(tmp_path)
    doc, rp = _doc_and_receipt(tmp_path)
    res = tmp_path / "result.json"
    res.write_text(json.dumps({"auc": 0.95, "verdict": "SURVIVED"}), encoding="utf-8")
    s = seal(doc, [rp], preregs=[(p, res)])
    assert s.verdict == "SEALED"
    assert s.protocol[0]["frozen_verdict"] == "SURVIVED"
    assert len(s.protocol[0]["prereg_commit"]) == 40


def test_uncommitted_prereg_refuses(tmp_path):
    _repo(tmp_path)
    p = _prereg(tmp_path, commit=False)
    doc, rp = _doc_and_receipt(tmp_path)
    res = tmp_path / "result.json"
    res.write_text(json.dumps({"auc": 0.95, "verdict": "SURVIVED"}), encoding="utf-8")
    s = seal(doc, [rp], preregs=[(p, res)])
    assert s.verdict == "REFUSED"
    assert any("PrologueError" in r["why"] for r in s.refusals)


def test_nothing_checkable_seals_vacuous_loudly(tmp_path):
    doc = tmp_path / "empty.md"
    doc.write_text("# thoughts\n\nno numbers live here at all.\n", encoding="utf-8")
    s = seal(doc, [])
    assert s.verdict == "SEALED_VACUOUS"


def test_tampered_seal_fails_verification(tmp_path):
    doc, rp = _doc_and_receipt(tmp_path)
    d = seal(doc, [rp]).to_dict()
    assert verify_seal(d)
    d["oath"]["counts"]["VERIFIED"] += 5
    assert not verify_seal(d)
