"""styxx.worklog v0.1 — the record, and the checks it can honestly support.

Spec: papers/closed-model-frontier/SPEC_worklog_v01_2026_08_31.md

The load-bearing tests are the negative ones: a worklog must carry no verdict,
must refuse a tampered digest, and must never claim completeness.
"""
from __future__ import annotations

import json

import pytest

from styxx.worklog import SPEC, Worklog, load, verify_worklog


@pytest.fixture
def wl(tmp_path):
    w = Worklog(session="s-1", harness="test-harness")
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")
    w.record_file(tmp_path / "a.py", "edit")
    w.record_file(tmp_path / "b.py", "create")
    return w, tmp_path


# ------------------------------------------------------------------ recording

def test_records_digests_never_contents(wl):
    w, _ = wl
    blob = json.dumps(w.to_dict())
    assert "x = 1" not in blob and "y = 2" not in blob, "worklog leaked file contents"
    assert all(len(e["after_sha256"]) == 64 for e in w.entries)


def test_sequence_is_dense_and_ordered(wl):
    w, _ = wl
    assert [e["seq"] for e in w.entries] == [1, 2]
    w.record("c.py", "edit", b"z = 3\n")
    assert [e["seq"] for e in w.entries] == [1, 2, 3]


def test_round_trip(wl, tmp_path):
    w, _ = wl
    p = w.write(tmp_path / "log.json")
    again = load(p)
    assert again.digest() == w.digest()
    assert len(again.entries) == 2


def test_verify_intact(wl, tmp_path):
    w, _ = wl
    p = w.write(tmp_path / "log.json")
    rep = verify_worklog(p)
    assert rep["ok"], rep["problems"]
    assert rep["entries"] == 2 and rep["distinct_paths"] == 2


# ------------------------------------------------------------------ tampering

def test_tampered_entry_breaks_the_digest(wl, tmp_path):
    w, _ = wl
    p = w.write(tmp_path / "log.json")
    d = json.loads(p.read_text(encoding="utf-8"))
    d["entries"][0]["path"] = "somewhere/else.py"
    p.write_text(json.dumps(d), encoding="utf-8")
    rep = verify_worklog(p)
    assert not rep["ok"]
    assert any("digest" in x for x in rep["problems"])


def test_removed_entry_breaks_the_sequence(wl, tmp_path):
    w, _ = wl
    p = w.write(tmp_path / "log.json")
    d = json.loads(p.read_text(encoding="utf-8"))
    d["entries"] = d["entries"][1:]          # drop the first write
    p.write_text(json.dumps(d), encoding="utf-8")
    rep = verify_worklog(p)
    assert not rep["ok"]
    assert any("sequence" in x or "digest" in x for x in rep["problems"])


def test_unknown_spec_refused(tmp_path):
    p = tmp_path / "log.json"
    p.write_text(json.dumps({"spec": "something/else", "entries": []}), encoding="utf-8")
    rep = verify_worklog(p)
    assert not rep["ok"] and rep["stage"] == "spec"


def test_entry_with_no_digest_at_all_is_rejected(wl, tmp_path):
    w, _ = wl
    w.entries.append({"seq": 3, "path": "c.py", "tool": "edit",
                      "at": "2026-08-31T00:00:00Z",
                      "after_sha256": None, "before_sha256": None})
    p = w.write(tmp_path / "log.json")
    rep = verify_worklog(p)
    assert not rep["ok"]
    assert any("neither a before nor an after" in x for x in rep["problems"])


# ------------------------------------------------ the honest-boundary contract

def test_a_worklog_never_carries_a_verdict(wl, tmp_path):
    """The whole point. If this ever fails, someone gave the record an opinion."""
    w, _ = wl
    p = w.write(tmp_path / "log.json")
    rep = verify_worklog(p)
    assert rep["verdict"] == "UNGATED"
    blob = json.dumps(w.to_dict())
    for banned in ("CONTRADICTED", "VERIFIED", "PASS", "FAIL", "UNDECLARED"):
        assert banned not in blob, f"a worklog must not contain {banned}"


def test_the_incompleteness_boundary_is_stated_not_implied(wl, tmp_path):
    w, _ = wl
    p = w.write(tmp_path / "log.json")
    b = verify_worklog(p)["boundary"]
    assert "instrumented surface" in b
    assert "only as trustworthy as the harness" in b


def test_spec_string_is_pinned():
    assert SPEC == "styxx-worklog/v0.1"
