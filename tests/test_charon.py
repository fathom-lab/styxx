# -*- coding: utf-8 -*-
"""styxx.charon v0.1 — every clause of the frozen spec, pinned.

papers/charon/SPEC_charon_v01_2026_09_02.md. The load-bearing tests are the ones about what the
log refuses to hide: a removed or reordered line is TAMPER, a moved verifier is SKEW and not
DRIFT, an absent commit is UNRESOLVED and never an accusation, the receipt-set size is on the
line, and the page carries no script.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from styxx import charon, sworn
from styxx.charon import (ENTRY_SCHEMA, LOG_SCHEMA, check_chain, ingest, make_entry, read_log,
                          render_page, verify_log)

ROOT = Path(__file__).resolve().parent.parent
CAPSULE = ROOT / "papers" / "closed-model-frontier" / "CORPUS_STATE_2026_08_31.capsule.html"
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


class TestEntry:
    def test_the_entry_is_content_addressed_and_the_timestamp_is_outside_it(self, sworn_repo):
        repo, sha = sworn_repo
        d = charon.derive(repo / "RESULT_x.sworn.json", repo)
        a = make_entry(d, 1, None, timestamp="2026-09-02T00:00:00Z")
        b = make_entry(d, 1, None, timestamp="2030-01-01T00:00:00Z")
        assert a["entry_id"] == b["entry_id"] and a["timestamp"] != b["timestamp"]
        assert a["schema"] == ENTRY_SCHEMA and a["kind"] == "sworn"
        assert a["verdict"] == "SWORN-HELD" and a["verdict_class"] == "HELD" and a["reproduced"] is True
        assert a["at"]["commit"] == sha and a["subject"]["path"] == "RESULT_x.md"
        assert a["receipts"]["n"] == 1 and len(a["receipts"]["sha256"][0]) == 64
        assert a["floor"] == 0.5 and a["rungs"] == {"committed": 1}
        assert a["verifier"]["module"] == "styxx.sworn" and len(a["verifier"]["module_sha256"]) == 64

    def test_a_different_seq_or_prev_is_a_different_entry(self, sworn_repo):
        repo, _ = sworn_repo
        d = charon.derive(repo / "RESULT_x.sworn.json", repo)
        assert make_entry(d, 1, None)["entry_id"] != make_entry(d, 2, "a" * 64)["entry_id"]

    def test_an_absent_commit_is_unresolved_never_an_accusation(self, sworn_repo, tmp_path):
        repo, sha = sworn_repo
        side = json.loads((repo / "RESULT_x.sworn.json").read_text(encoding="utf-8"))
        side["commit"] = "f" * 40
        other = tmp_path / "RESULT_x.sworn.json"
        sworn._write_json_lf(other, side)
        (tmp_path / "RESULT_x.md").write_bytes((repo / "RESULT_x.md").read_bytes())
        d = charon.derive(other, repo)
        assert d["verdict"] == "UNRESOLVED" and d["verdict_class"] == "UNRESOLVED"
        assert d["counts"]["reason"] == "commit_absent" and d["reproduced"] is False

    def test_anything_else_is_refused_by_name(self, tmp_path):
        p = tmp_path / "notes.txt"
        p.write_text("x")
        with pytest.raises(SystemExit):
            charon.derive(p, tmp_path)


class TestLog:
    def test_ingest_writes_a_header_then_dense_chained_entries_with_lf(self, sworn_repo, tmp_path):
        repo, _ = sworn_repo
        log = tmp_path / "c.log.jsonl"
        ingest([repo / "RESULT_x.sworn.json"], log, repo, timestamp="2026-09-02T00:00:00Z")
        ingest([CAPSULE, CERT], log, ROOT, timestamp="2026-09-02T00:00:01Z")
        raw = log.read_bytes()
        assert b"\r" not in raw
        header, entries = read_log(log)
        assert header["schema"] == LOG_SCHEMA and "NOT a claim that any verdict is true" in header["certifies"]
        assert [e["seq"] for e in entries] == [1, 2, 3]
        assert entries[0]["prev"] is None and entries[1]["prev"] == entries[0]["entry_id"]
        assert entries[2]["prev"] == entries[1]["entry_id"]
        assert check_chain(entries) == []
        kinds = [e["kind"] for e in entries]
        assert kinds == ["sworn", "capsule-oath", "oath-certificate"]
        assert entries[1]["receipts"]["n"] >= 1 and entries[2]["receipts"]["n"] == 4

    def test_a_removed_or_edited_line_is_tamper(self, sworn_repo, tmp_path):
        repo, _ = sworn_repo
        log = tmp_path / "c.log.jsonl"
        ingest([repo / "RESULT_x.sworn.json", repo / "RESULT_x.sworn.json"], log, repo)
        lines = log.read_text(encoding="utf-8").split("\n")
        # edit a verdict in place: the entry_id no longer re-derives
        edited = lines[1].replace('"SWORN-HELD"', '"SWORN-FAILED"', 1)
        tampered = tmp_path / "t.log.jsonl"
        tampered.write_bytes("\n".join([lines[0], edited, lines[2], ""]).encode())
        rep = verify_log(tampered, repo)
        assert rep["by_status"]["TAMPER"] >= 1 and rep["lines"][0]["status"] == "TAMPER"
        # drop the first entry: the second's prev points at nothing
        dropped = tmp_path / "d.log.jsonl"
        dropped.write_bytes("\n".join([lines[0], lines[2], ""]).encode())
        rep = verify_log(dropped, repo)
        assert rep["by_status"]["TAMPER"] >= 1
        with pytest.raises(SystemExit):
            ingest([repo / "RESULT_x.sworn.json"], dropped, repo)      # refuses to extend a broken chain

    def test_verify_reproduces_an_unchanged_log(self, sworn_repo, tmp_path):
        repo, _ = sworn_repo
        log = tmp_path / "c.log.jsonl"
        ingest([repo / "RESULT_x.sworn.json"], log, repo)
        rep = verify_log(log, repo)
        assert rep["by_status"]["REPRODUCED"] == 1 and rep["entries"] == 1
        assert rep["head"] == read_log(log)[1][-1]["entry_id"]

    def test_skew_is_a_moved_verifier_and_drift_is_moved_bytes(self, sworn_repo, tmp_path, monkeypatch):
        repo, _ = sworn_repo
        log = tmp_path / "c.log.jsonl"
        ingest([repo / "RESULT_x.sworn.json"], log, repo)
        real_derive = charon.derive

        def moved_verdict(path, r):
            d = real_derive(path, r)
            d["verdict"], d["verdict_class"] = "SWORN-FAILED", "FAILED"
            return d
        monkeypatch.setattr(charon, "derive", moved_verdict)
        assert verify_log(log, repo)["lines"][0]["status"] == "DRIFT"       # same build, verdict moved

        def moved_verifier_and_verdict(path, r):
            d = moved_verdict(path, r)
            d["verifier"]["module_sha256"] = "0" * 64
            return d
        monkeypatch.setattr(charon, "derive", moved_verifier_and_verdict)
        assert verify_log(log, repo)["lines"][0]["status"] == "SKEW"        # build moved too

        def moved_verifier_only(path, r):
            d = real_derive(path, r)
            d["verifier"]["module_sha256"] = "0" * 64
            return d
        monkeypatch.setattr(charon, "derive", moved_verifier_only)
        assert verify_log(log, repo)["lines"][0]["status"] == "MOVED_VERIFIER"

    def test_a_capsule_that_stops_reproducing_moves_the_line(self, tmp_path, monkeypatch):
        log = tmp_path / "c.log.jsonl"
        ingest([CAPSULE], log, ROOT)
        real_derive = charon.derive

        def now_broken(path, r):
            d = real_derive(path, r)
            d["reproduced"] = not d["reproduced"]
            return d
        monkeypatch.setattr(charon, "derive", now_broken)
        assert verify_log(log, ROOT)["lines"][0]["status"] == "DRIFT"

    def test_a_missing_artifact_is_unresolved(self, sworn_repo, tmp_path):
        repo, _ = sworn_repo
        log = tmp_path / "c.log.jsonl"
        ingest([repo / "RESULT_x.sworn.json"], log, repo)
        (repo / "RESULT_x.sworn.json").unlink()
        rep = verify_log(log, repo)
        assert rep["lines"][0]["status"] == "UNRESOLVED" and rep["lines"][0]["reason"] == "artifact_missing"


class TestPage:
    def test_the_page_has_every_entry_id_no_script_and_lf(self, sworn_repo, tmp_path):
        repo, _ = sworn_repo
        log = tmp_path / "c.log.jsonl"
        ingest([repo / "RESULT_x.sworn.json"], log, repo)
        ingest([CAPSULE], log, ROOT)
        rep = verify_log(log, ROOT)
        out = render_page(log, tmp_path / "index.html", rep)
        raw = out.read_bytes()
        html = raw.decode("utf-8")
        assert b"\r" not in raw and "<script" not in html.lower()
        for e in read_log(log)[1]:
            assert e["entry_id"][:16] in html
        assert "receipts.n is a column" in html and "nothing here is signed" in html
        assert "python -m styxx.charon verify" in html


class TestCLI:
    def test_ingest_verify_status_page_round_trip(self, sworn_repo, tmp_path, capsys):
        repo, _ = sworn_repo
        log = tmp_path / "c.log.jsonl"
        assert charon.main(["ingest", "--log", str(log), "--repo", str(repo),
                            str(repo / "RESULT_x.sworn.json")]) == 0
        assert charon.main(["verify", "--log", str(log), "--repo", str(repo),
                            "--out", str(tmp_path / "rep.json")]) == 0
        assert charon.main(["status", "--log", str(log)]) == 0
        assert charon.main(["page", "--log", str(log), "--out", str(tmp_path / "i.html"),
                            "--report", str(tmp_path / "rep.json")]) == 0
        out = capsys.readouterr().out
        assert "REPRODUCED=1" in out and "chain ok" in out
        assert b"\r" not in (tmp_path / "rep.json").read_bytes()

    def test_verify_exits_one_only_on_tamper(self, sworn_repo, tmp_path):
        repo, _ = sworn_repo
        log = tmp_path / "c.log.jsonl"
        ingest([repo / "RESULT_x.sworn.json"], log, repo)
        lines = log.read_text(encoding="utf-8").split("\n")
        log.write_bytes("\n".join([lines[0], lines[1].replace("HELD", "FAILED", 1), ""]).encode())
        assert charon.main(["verify", "--log", str(log), "--repo", str(repo)]) == 1
