# -*- coding: utf-8 -*-
"""styxx.charon v0.1 (log/entry v1) — the spec and its errata, pinned.

The load-bearing tests are the ones the adversarial pass of 2026-09-02 forced: a rebuilt chain
or a truncated tail is a DIFFERENT log and only --expect-head says so; verify compares the whole
core, not the class; a capsule's verdict is re-derived, not copied; a headerless log is refused;
a malformed line is TAMPER, not a traceback; no absolute path is ever written; the page shows
and proves nothing; the receipt set on an OATH line is the resolved one.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from styxx import charon, sworn
from styxx.charon import (ENTRY_SCHEMA, LOG_SCHEMA, check_chain, header_digest, ingest, make_entry,
                          read_log, render_page, verify_log)

ROOT = Path(__file__).resolve().parent.parent
CAPSULE = ROOT / "papers" / "closed-model-frontier" / "CORPUS_STATE_2026_08_31.capsule.html"
CAPSULE_V02 = ROOT / "papers" / "closed-model-frontier" / "HANDOFF_capsule_v02_2026_08_31.capsule.html"
CERT = ROOT / "POSITIONING.certificate.json"


def _git(repo, *args):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=env,
                          check=True).stdout.strip()


@pytest.fixture
def sworn_repo(tmp_path):
    """A repository with one receipt committed and one sworn document canonicalised at that commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "res.json").write_bytes(b'{"recall": 0.91, "passed": 296}')
    _git(repo, "add", "res.json")
    _git(repo, "commit", "-q", "-m", "receipt")
    sha = _git(repo, "rev-parse", "HEAD")
    doc = repo / "RESULT_x.md"
    doc.write_bytes(b'<sworn r="path:res.json#/recall" k="numeric">recall was 0.91.</sworn> narrative.\n')
    side = sworn.to_sidecar(doc.read_bytes(), doc.name, sha)
    sworn._write_json_lf(repo / "RESULT_x.sworn.json", side)
    core = sworn.verify(sidecar=side, tree=sworn.GitTree(repo, sha), commit=sha)
    sworn._write_json_lf(repo / "RESULT_x.sworn-receipt.json", sworn.issue_receipt(core))
    return repo, sha


def _log_with(repo, tmp_path, *artifacts, root=None):
    log = tmp_path / "c.log.jsonl"
    ingest(list(artifacts), log, root or repo, timestamp="2026-09-02T00:00:00Z", population="test")
    return log


def _rechain(lines):
    """A forger's nine lines: rebuild seq/prev/entry_id over an altered set, honestly."""
    header = json.loads(lines[0])
    prev = header_digest(header)
    out = [lines[0]]
    for i, ln in enumerate(lines[1:], start=1):
        e = json.loads(ln)
        e.pop("entry_id", None)
        e["seq"], e["prev"] = i, prev
        core = {k: v for k, v in e.items() if k not in ("timestamp", "note")}
        e["entry_id"] = charon._sha256(charon.jcs(core).encode())
        prev = e["entry_id"]
        out.append(json.dumps(e, ensure_ascii=False))
    return out


class TestEntry:
    def test_the_entry_is_content_addressed_and_the_timestamp_is_outside_it(self, sworn_repo):
        repo, sha = sworn_repo
        d = charon.derive(repo / "RESULT_x.sworn.json", repo)
        a = make_entry(d, 1, "0" * 64, timestamp="2026-09-02T00:00:00Z")
        b = make_entry(d, 1, "0" * 64, timestamp="2030-01-01T00:00:00Z")
        assert a["entry_id"] == b["entry_id"] and a["timestamp"] != b["timestamp"]
        assert a["schema"] == ENTRY_SCHEMA and a["kind"] == "sworn"
        assert a["verdict"] == "SWORN-HELD" and a["verdict_class"] == "HELD" and a["reproduced"] is True
        assert a["at"]["commit"] == sha and a["subject"]["path"] == "RESULT_x.md"
        assert a["at"]["document_at_commit"] is False          # the document was never committed there
        assert a["receipts"]["n"] == 1 and a["receipts"]["vacuous"] is False
        assert a["floor"] == 0.5 and a["rungs"] == {"committed": 1}
        assert [m for m, _ in a["verifier"]["modules"]][:2] == ["styxx.charon", "styxx.sworn"]
        assert len(a["verifier"]["digest"]) == 64
        assert a["counts"]["receipt_check"]["status"] == "VERIFIED"

    def test_a_sworn_line_with_no_receipt_is_null_not_false(self, sworn_repo):
        repo, _ = sworn_repo
        (repo / "RESULT_x.sworn-receipt.json").unlink()
        d = charon.derive(repo / "RESULT_x.sworn.json", repo)
        assert d["reproduced"] is None and d["counts"]["recorded_verdict"] is None

    def test_a_forged_receipt_string_is_not_reproduced(self, sworn_repo):
        repo, _ = sworn_repo
        (repo / "RESULT_x.sworn-receipt.json").write_text('{"document_verdict": "SWORN-HELD"}', encoding="utf-8")
        d = charon.derive(repo / "RESULT_x.sworn.json", repo)
        assert d["reproduced"] is False and d["counts"]["receipt_check"]["status"] == "FAILED"

    def test_an_absent_commit_is_unresolved_never_an_accusation(self, sworn_repo, tmp_path):
        repo, sha = sworn_repo
        side = json.loads((repo / "RESULT_x.sworn.json").read_text(encoding="utf-8"))
        side["commit"] = "f" * 40
        sworn._write_json_lf(repo / "RESULT_y.sworn.json", side)
        (repo / "RESULT_y.md").write_bytes((repo / "RESULT_x.md").read_bytes())
        d = charon.derive(repo / "RESULT_y.sworn.json", repo)
        assert d["verdict"] == "UNRESOLVED" and d["verdict_class"] == "UNRESOLVED"
        assert d["counts"]["reason"] == "commit_absent" and d["reproduced"] is None

    def test_an_artifact_outside_the_repository_is_refused_no_absolute_path_is_written(self, sworn_repo, tmp_path):
        repo, _ = sworn_repo
        with pytest.raises(SystemExit, match="outside --repo"):
            charon.derive(CAPSULE, repo)

    def test_anything_else_is_refused_by_name(self, tmp_path):
        p = tmp_path / "notes.txt"
        p.write_text("x")
        with pytest.raises(SystemExit):
            charon.derive(p, tmp_path)
        with pytest.raises(SystemExit, match="no such artifact"):
            charon.derive(tmp_path / "nope.capsule.html", tmp_path)


class TestCapsuleAndCertificate:
    def test_a_capsule_verdict_is_re_derived_not_copied(self):
        d = charon.derive(CAPSULE, ROOT)
        assert d["kind"] == "capsule-oath" and d["verdict_class"] in ("OATH-HELD", "OATH-FAILED")
        assert d["counts"]["recorded_verdict"] is not None and d["receipts"]["n"] >= 1

    def test_a_forged_embedded_verdict_does_not_reach_the_line(self, tmp_path):
        """Edit the embedded certificate's verdict and leave the bytes: the live re-derivation
        disagrees, so the line carries the live verdict and reproduced=false."""
        html = CAPSULE.read_text(encoding="utf-8")
        i = html.index('id="oath-capsule">') + len('id="oath-capsule">')
        j = html.index("</script>", i)
        payload = json.loads(html[i:j])
        live_verdict = payload["certificate"]["verdict"]
        forged_verdict = "OATH-FAILED" if "HELD" in live_verdict else "OATH-HELD"
        payload["certificate"]["verdict"] = forged_verdict
        forged = html[:i] + json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c") + html[j:]
        p = tmp_path / "forged.capsule.html"
        p.write_text(forged, encoding="utf-8")
        d = charon.derive(p, tmp_path)
        assert d["counts"]["recorded_verdict"] == forged_verdict
        assert d["verdict"] != forged_verdict and d["reproduced"] is False

    def test_a_diffgate_capsule_verdict_is_the_live_gate(self):
        d = charon.derive(CAPSULE_V02, ROOT)
        assert d["kind"] == "capsule-diffgate" and d["verdict_class"] in ("PASS", "FAIL")
        assert d["receipts"]["n"] == 3                          # summary, diff, gate bindings

    def test_a_certificate_line_carries_resolved_and_cited_receipts(self):
        """Both sets, and they are allowed to differ.

        The resolved digests are content identity modulo newlines (`_content_sha256`), so the line
        reads the same on a CRLF checkout as on an LF one. The cited digests are whatever the
        certificate recorded, and every `receipts_sha256` in this corpus was recorded from a
        Windows working tree — CRLF hashes, as `corpus_audit._receipt_sha_matches` documents. So
        the two sets differ on both platforms, the line carries both, and the certificate still
        reproduces because the auditor compares content, not bytes."""
        d = charon.derive(CERT, ROOT)
        assert d["kind"] == "oath-certificate" and d["verdict_class"].startswith("OATH-")
        assert d["receipts"]["cited"]["n"] == 4 and d["receipts"]["n"] == 4
        assert d["reproduced"] is True and d["counts"]["pinned_verifier_sha256"]

    def test_a_certificate_line_reads_the_same_on_crlf_and_on_lf(self, tmp_path):
        """The promise the log makes is that a stranger re-derives the same lines. A raw hash of a
        working-tree document would break it on the other platform's checkout: 213 OATH lines would
        read as moved. This is that guarantee, pinned."""
        cert = json.loads(CERT.read_text(encoding="utf-8"))
        doc_lf = (ROOT / "POSITIONING.md").read_bytes().replace(b"\r\n", b"\n")
        names = list(cert.get("receipts_sha256") or {})
        for eol in (b"\n", b"\r\n"):
            d = tmp_path / eol.hex()
            d.mkdir()
            (d / "POSITIONING.md").write_bytes(doc_lf.replace(b"\n", eol))
            (d / "POSITIONING.certificate.json").write_text(json.dumps(cert), encoding="utf-8")
            for n in names:
                src = next(p for p in ROOT.rglob(n) if ".claude" not in p.parts)
                (d / n).write_bytes(src.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", eol))
        a = charon.derive(tmp_path / b"\n".hex() / "POSITIONING.certificate.json", tmp_path / b"\n".hex())
        b = charon.derive(tmp_path / b"\r\n".hex() / "POSITIONING.certificate.json", tmp_path / b"\r\n".hex())
        assert a["subject"]["sha256"] == b["subject"]["sha256"]
        assert a["receipts"]["sha256"] == b["receipts"]["sha256"]

    def test_a_certificate_whose_document_is_gone_is_a_line_not_a_skip(self, tmp_path):
        cert = json.loads(CERT.read_text(encoding="utf-8"))
        p = tmp_path / "GONE.certificate.json"
        p.write_text(json.dumps(cert), encoding="utf-8")
        d = charon.derive(p, tmp_path)
        assert d["verdict_class"] == "UNRESOLVED" and d["counts"]["reason"] == "MISSING_DOC"
        assert d["receipts"]["cited"]["n"] == 4 and d["counts"]["recorded_verdict"] == cert["verdict"]


class TestLog:
    def test_ingest_writes_a_chained_header_then_dense_chained_entries_with_lf(self, sworn_repo, tmp_path):
        repo, _ = sworn_repo
        log = _log_with(repo, tmp_path, repo / "RESULT_x.sworn.json", repo / "RESULT_x.sworn.json")
        raw = log.read_bytes()
        assert b"\r" not in raw
        header, entries, problems, facts = read_log(log)
        assert header["schema"] == LOG_SCHEMA and header["population"] == "test"
        assert "NOT a claim that any verdict is true" in header["certifies"]
        assert entries[0]["prev"] == header_digest(header) and entries[1]["prev"] == entries[0]["entry_id"]
        assert [e["seq"] for e in entries] == [1, 2] and check_chain(header, entries) == [] and problems == []
        assert facts["eol"] == "LF" and facts["bom"] is False

    def test_an_edited_header_is_tamper(self, sworn_repo, tmp_path):
        repo, _ = sworn_repo
        log = _log_with(repo, tmp_path, repo / "RESULT_x.sworn.json")
        lines = log.read_text(encoding="utf-8").split("\n")
        h = json.loads(lines[0])
        h["certifies"] = "every verdict here is TRUE and signed"
        lines[0] = json.dumps(h, ensure_ascii=False)
        log.write_bytes("\n".join(lines).encode())
        assert verify_log(log, repo)["by_status"]["TAMPER"] == 1

    def test_an_edited_or_removed_interior_line_is_tamper(self, sworn_repo, tmp_path):
        repo, _ = sworn_repo
        log = _log_with(repo, tmp_path, repo / "RESULT_x.sworn.json", repo / "RESULT_x.sworn.json")
        lines = log.read_text(encoding="utf-8").split("\n")
        edited = tmp_path / "t.log.jsonl"
        edited.write_bytes("\n".join([lines[0], lines[1].replace('"SWORN-HELD"', '"SWORN-FAILED"', 1), lines[2], ""]).encode())
        rep = verify_log(edited, repo)
        assert rep["by_status"]["TAMPER"] == 2 and rep["chain_broken_at_line"] == 2
        assert rep["chain_problems"][-1]["problem"] == "UNVERIFIABLE_AFTER_BREAK"
        dropped = tmp_path / "d.log.jsonl"
        dropped.write_bytes("\n".join([lines[0], lines[2], ""]).encode())
        assert verify_log(dropped, repo)["by_status"]["TAMPER"] >= 1
        with pytest.raises(SystemExit, match="does not chain"):
            ingest([repo / "RESULT_x.sworn.json"], dropped, repo)

    def test_a_rebuilt_chain_or_a_truncated_tail_is_only_caught_by_the_expected_head(self, sworn_repo, tmp_path):
        repo, _ = sworn_repo
        log = _log_with(repo, tmp_path, repo / "RESULT_x.sworn.json", repo / "RESULT_x.sworn.json",
                        repo / "RESULT_x.sworn.json")
        head = read_log(log)[1][-1]["entry_id"]
        lines = [ln for ln in log.read_text(encoding="utf-8").split("\n") if ln]
        # truncate the tail: a valid shorter log with a different head
        short = tmp_path / "short.log.jsonl"
        short.write_bytes("\n".join(lines[:3] + [""]).encode())
        rep = verify_log(short, repo)
        assert rep["by_status"]["TAMPER"] == 0 and rep["head"] != head
        rep = verify_log(short, repo, expect_head=head)
        assert rep["head_matches"] is False and rep["by_status"]["HEAD_MISMATCH"] == 1
        assert charon.main(["verify", "--log", str(short), "--repo", str(repo), "--expect-head", head]) == 1
        # drop the middle line and rebuild the suffix honestly: chain ok, head different
        rebuilt = tmp_path / "rebuilt.log.jsonl"
        rebuilt.write_bytes("\n".join(_rechain([lines[0], lines[1], lines[3]]) + [""]).encode())
        rep = verify_log(rebuilt, repo, expect_head=head)
        assert rep["by_status"]["TAMPER"] == 0 and rep["head_matches"] is False
        assert charon.main(["status", "--log", str(rebuilt), "--expect-head", head]) == 1
        assert charon.main(["status", "--log", str(log), "--expect-head", head]) == 0

    def test_a_headerless_log_is_refused_on_ingest_and_tamper_on_verify(self, sworn_repo, tmp_path):
        repo, _ = sworn_repo
        log = _log_with(repo, tmp_path, repo / "RESULT_x.sworn.json")
        lines = [ln for ln in log.read_text(encoding="utf-8").split("\n") if ln]
        nohead = tmp_path / "nohead.log.jsonl"
        nohead.write_bytes("\n".join(lines[1:] + [""]).encode())
        with pytest.raises(SystemExit, match="does not chain"):
            ingest([repo / "RESULT_x.sworn.json"], nohead, repo)
        rep = verify_log(nohead, repo)
        assert rep["by_status"]["TAMPER"] == 1 and any("no header" in p["problem"] for p in rep["chain_problems"])

    def test_malformed_lines_and_a_bom_are_tamper_not_a_traceback(self, sworn_repo, tmp_path):
        repo, _ = sworn_repo
        log = _log_with(repo, tmp_path, repo / "RESULT_x.sworn.json")
        raw = log.read_bytes()
        bad = tmp_path / "bad.log.jsonl"
        bad.write_bytes(b"\xef\xbb\xbf" + raw + b"[1,2]\nnot json\n" + b'{"schema": "styxx.charon/entry/v1", "seq": 3}\n')
        rep = verify_log(bad, repo)
        probs = [p["problem"] for p in rep["chain_problems"]]
        assert any("BOM" in p for p in probs) and any("not JSON" in p for p in probs) \
            and any("not a JSON object" in p for p in probs) and any("missing keys" in p for p in probs)
        assert rep["by_status"]["TAMPER"] >= 1 and rep["file"]["bom"] is True

    def test_crlf_is_reported_and_the_file_digest_travels(self, sworn_repo, tmp_path):
        repo, _ = sworn_repo
        log = _log_with(repo, tmp_path, repo / "RESULT_x.sworn.json")
        crlf = tmp_path / "crlf.log.jsonl"
        crlf.write_bytes(log.read_bytes().replace(b"\n", b"\r\n"))
        rep = verify_log(crlf, repo)
        assert rep["file"]["eol"] == "CRLF" and rep["file"]["file_sha256"] != verify_log(log, repo)["file"]["file_sha256"]

    def test_a_missing_log_is_refused_not_a_clean_zero(self, tmp_path):
        with pytest.raises(SystemExit, match="no such log"):
            verify_log(tmp_path / "nope.jsonl", tmp_path)

    def test_verify_compares_the_whole_core_same_line_moved_verifier_skew_drift(self, sworn_repo, tmp_path, monkeypatch):
        repo, _ = sworn_repo
        log = _log_with(repo, tmp_path, repo / "RESULT_x.sworn.json")
        assert verify_log(log, repo)["lines"][0]["status"] == "SAME_LINE"
        real_derive = charon.derive

        def receipts_grew(path, r):
            d = real_derive(path, r)
            d["receipts"] = {"n": 2, "sha256": sorted(d["receipts"]["sha256"] + ["a" * 64]), "cited": None, "vacuous": False}
            return d
        monkeypatch.setattr(charon, "derive", receipts_grew)
        row = verify_log(log, repo)["lines"][0]
        assert row["status"] == "DRIFT" and row["fields_changed"] == ["receipts"] and row["receipts_moved"] is True

        def verifier_and_receipts(path, r):
            d = receipts_grew(path, r)
            d["verifier"]["digest"] = "0" * 64
            return d
        monkeypatch.setattr(charon, "derive", verifier_and_receipts)
        assert verify_log(log, repo)["lines"][0]["status"] == "SKEW"

        def verifier_only(path, r):
            d = real_derive(path, r)
            d["verifier"]["digest"] = "0" * 64
            return d
        monkeypatch.setattr(charon, "derive", verifier_only)
        assert verify_log(log, repo)["lines"][0]["status"] == "MOVED_VERIFIER"

    def test_a_missing_artifact_is_unresolved(self, sworn_repo, tmp_path):
        repo, _ = sworn_repo
        log = _log_with(repo, tmp_path, repo / "RESULT_x.sworn.json")
        (repo / "RESULT_x.sworn.json").unlink()
        rep = verify_log(log, repo)
        assert rep["lines"][0]["status"] == "UNRESOLVED" and rep["lines"][0]["reason"] == "artifact_missing"

    def test_duplicates_are_lines_and_distinct_subjects_is_reported(self, sworn_repo, tmp_path):
        repo, _ = sworn_repo
        log = _log_with(repo, tmp_path, repo / "RESULT_x.sworn.json", repo / "RESULT_x.sworn.json")
        rep = verify_log(log, repo)
        assert rep["entries"] == 2 and rep["distinct_subjects"] == 1
        assert rep["receipts_n"]["held_total"] == 2 and rep["reproduced_at_ingest"]["sworn"]["true"] == 2

    def test_a_vacuous_held_is_excluded_from_the_held_count(self, tmp_path):
        repo = tmp_path / "r"
        repo.mkdir()
        _git(repo, "init", "-q")
        (repo / "x.txt").write_text("x")
        _git(repo, "add", "x.txt")
        _git(repo, "commit", "-q", "-m", "one")
        sha = _git(repo, "rev-parse", "HEAD")
        doc = repo / "RESULT_v.md"
        doc.write_bytes(b'<sworn r="path:gone.json#/a" k="numeric">1 item.</sworn>\n')
        sworn._write_json_lf(repo / "RESULT_v.sworn.json", sworn.to_sidecar(doc.read_bytes(), doc.name, sha))
        d = charon.derive(repo / "RESULT_v.sworn.json", repo)
        assert d["verdict"] == "SWORN-HELD" and d["receipts"]["vacuous"] is True and d["receipts"]["n"] == 0
        log = _log_with(repo, tmp_path, repo / "RESULT_v.sworn.json")
        rep = verify_log(log, repo)
        assert rep["receipts_n"]["held_total"] == 0 and rep["receipts_n"]["by_kind"]["sworn"]["vacuous_excluded"] == 1


class TestPage:
    def test_the_page_has_every_entry_id_no_script_no_handler_no_request_and_lf(self, sworn_repo, tmp_path):
        repo, _ = sworn_repo
        log = _log_with(repo, tmp_path, repo / "RESULT_x.sworn.json")
        rep = verify_log(log, repo)
        out = render_page(log, tmp_path / "index.html", rep, repo=tmp_path)
        raw = out.read_bytes()
        html = raw.decode("utf-8").lower()
        assert b"\r" not in raw and "<script" not in html and "javascript:" not in html
        assert "href=" not in html and "src=" not in html and "http://" not in html and "https://" not in html
        for e in read_log(log)[1]:
            assert e["entry_id"][:16] in html
        assert "it proves nothing" in html and "nothing here is signed" in html
        assert "python -m styxx.charon derive --repo . result_x.sworn.json" in html
        assert "--expect-head" in html and "this log's header says" in html


class TestCLI:
    def test_ingest_verify_derive_status_page_round_trip(self, sworn_repo, tmp_path, capsys):
        repo, _ = sworn_repo
        log = tmp_path / "c.log.jsonl"
        assert charon.main(["ingest", "--log", str(log), "--repo", str(repo), "--population", "one sworn document",
                            str(repo / "RESULT_x.sworn.json")]) == 0
        assert charon.main(["verify", "--log", str(log), "--repo", str(repo), "--out", str(tmp_path / "rep.json")]) == 0
        assert charon.main(["derive", "--repo", str(repo), str(repo / "RESULT_x.sworn.json")]) == 0
        assert charon.main(["status", "--log", str(log)]) == 0
        assert charon.main(["page", "--log", str(log), "--out", str(tmp_path / "i.html"), "--repo", str(tmp_path),
                            "--report", str(tmp_path / "rep.json")]) == 0
        out = capsys.readouterr().out
        assert "SAME_LINE=1" in out and "chain ok" in out and '"kind": "sworn"' in out
        assert b"\r" not in (tmp_path / "rep.json").read_bytes()
        rep = json.loads((tmp_path / "rep.json").read_text(encoding="utf-8"))
        assert rep["header"]["population"] == "one sworn document" and rep["head_matches"] is None

    def test_verify_exits_one_on_tamper_and_on_head_mismatch_only(self, sworn_repo, tmp_path):
        repo, _ = sworn_repo
        log = _log_with(repo, tmp_path, repo / "RESULT_x.sworn.json")
        head = read_log(log)[1][-1]["entry_id"]
        assert charon.main(["verify", "--log", str(log), "--repo", str(repo), "--expect-head", head]) == 0
        assert charon.main(["verify", "--log", str(log), "--repo", str(repo), "--expect-head", "0" * 64]) == 1
        lines = log.read_text(encoding="utf-8").split("\n")
        log.write_bytes("\n".join([lines[0], lines[1].replace("HELD", "FAILED", 1), ""]).encode())
        assert charon.main(["verify", "--log", str(log), "--repo", str(repo)]) == 1
