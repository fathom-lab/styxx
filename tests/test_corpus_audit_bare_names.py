"""corpus_audit re-derives a certificate over the bytes its receipts had at the issuing commit.

It writes those bytes into a temporary directory under the receipt names the audited certificate
gives, so it writes only bare file names: a certificate in a repository someone else wrote must
not choose where the auditor's machine writes.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _repo_with_escaping_receipt(base: Path, name: str) -> Path:
    repo = base / "audited"
    (repo / "a" / "b").mkdir(parents=True)
    payload = b"bytes the certificate's author chose\n"
    (repo / "evil.txt").write_bytes(payload)
    doc = b"# A note\n\nThe value was 42.\n"
    (repo / "a" / "b" / "x.md").write_bytes(doc)
    cert = {
        "document": "x.md",
        "document_sha256": hashlib.sha256(doc).hexdigest(),
        "receipts_sha256": {name: hashlib.sha256(payload).hexdigest()},
        "receipt_binding": {"receipts": [{"name": name, "path": "evil.txt"}]},
        "verdict": "OATH-HELD",
        "counts": {},
    }
    (repo / "a" / "b" / "x.certificate.json").write_text(json.dumps(cert, indent=1) + "\n", encoding="utf-8")
    g = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
         "-c", "core.autocrlf=false", "-c", "commit.gpgsign=false"]
    subprocess.run(g[:3] + ["init", "-q"], check=True)
    subprocess.run(g + ["add", "-A"], check=True)
    subprocess.run(g + ["commit", "-q", "-m", "audited"], check=True)
    return repo


def _audit(repo: Path, tmp: Path, out: Path):
    env = dict(os.environ, PYTHONPATH=str(ROOT), TMP=str(tmp), TEMP=str(tmp), TMPDIR=str(tmp),
               PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-m", "styxx.corpus_audit", str(repo), "--history", "on",
                           "--json", str(out)], cwd=str(out.parent), env=env, capture_output=True,
                          text=True, timeout=600)


def test_a_receipt_name_that_climbs_out_is_not_written(tmp_path):
    base = tmp_path / "tmp" / "base"
    base.mkdir(parents=True)
    target = base.parent / "ESCAPED_by_corpus_audit.txt"     # a temp dir under base, then ../.. lands here
    repo = _repo_with_escaping_receipt(tmp_path, "../../ESCAPED_by_corpus_audit.txt")
    r = _audit(repo, base, tmp_path / "report.json")
    assert r.returncode in (0, 1), r.stderr[-800:]
    assert not target.exists()
    rep = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    rb = rep["documents"][0].get("receipt_binding") or {}
    assert rb.get("stands_over_sworn_bytes") is None
    assert "not a bare file name" in str(rb.get("stands_reason"))


def test_an_absolute_receipt_name_is_not_written(tmp_path):
    base = tmp_path / "tmp"
    base.mkdir()
    target = tmp_path / "elsewhere" / "ESCAPED_absolute.txt"
    target.parent.mkdir()
    repo = _repo_with_escaping_receipt(tmp_path, str(target))
    r = _audit(repo, base, tmp_path / "report.json")
    assert r.returncode in (0, 1), r.stderr[-800:]
    assert not target.exists()
