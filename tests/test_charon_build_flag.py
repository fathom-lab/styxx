# -*- coding: utf-8 -*-
"""charon reads a sworn line's advisory build flag apart, and only that flag, only when the sworn.py the
line names moved (papers/charon/AMENDMENT_charon_same_verifier_build_flag_2026_10_25.md).

A sworn line carries counts.receipt_check.same_verifier_build: whether the verdict receipt it re-checks
names, as its issuing build, the sworn.py that is installed. That flag sits inside the line's compared
core. Under SPEC E5 a line whose only moved field is that flag reads SKEW when the verifier moved, and the
fourteen sworn lines of the committed log written 2026-09-13..15 read SKEW on the earliest sworn.py edit since
then. The amendment reads such a line MOVED_VERIFIER when, and only when:
  * the line is a sworn line, and the verifier moved (its digest differs);
  * the only compared field that moved is counts, and inside counts the only value that moved is that flag;
  * the line recorded it true and the fresh reading is false;
  * the styxx.sworn sha the line's verifier names is not the installed one;
  * the receipt on disk still names, as its issuing build, the styxx.sworn sha the line names.
Then the flag moved because the installed sworn.py moved, and for no other reason the line can show.

What is held here, each against the reading SPEC E5 gives (`_e5`): the case above reads MOVED_VERIFIER and
the row names what was read apart; a receipt re-issued in place under the same build reads DRIFT; the
flag beside any other moved compared field reads exactly what E5 reads; a flag that moved false -> true
(a receipt re-issued by the installed build), a receipt that names another build, a verifier that moved
outside styxx.sworn, and a line of another kind are not read apart; and, where the history is here, the
fourteen committed lines are read apart or not moved at all.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from styxx import charon, sworn
from styxx.charon import ingest, verify_log

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "papers" / "charon" / "charon.log.jsonl"
FLAG = "counts.receipt_check.same_verifier_build"


def _git(repo, *args):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=env,
                          check=True).stdout.strip()


@pytest.fixture
def sworn_repo(tmp_path):
    """One receipt committed, one sworn document canonicalised at that commit, and its verdict receipt
    issued by the installed sworn.py, so the line ingest writes records the flag true."""
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
    return repo


def _log(repo, tmp_path):
    log = tmp_path / "c.log.jsonl"
    ingest([repo / "RESULT_x.sworn.json"], log, repo, timestamp="2026-10-25T00:00:00Z", population="test")
    return log


def _line(log):
    _, entries, _, _ = charon.read_log(log)
    return entries[0]


def _sha_of(verifier, name):
    return dict(verifier["modules"]).get(name)


def _e5(row):
    """The status SPEC E5 (and the 2026-10-12 amendment's table) gives a re-derived row: SAME_LINE and
    MOVED_VERIFIER need every compared field to match."""
    same_build = row["verifier_moved"] == []
    if not row["fields_changed"]:
        return "SAME_LINE" if same_build else "MOVED_VERIFIER"
    return "DRIFT" if same_build else "SKEW"


def _module_moved(d, name, sha="e" * 64):
    """The fresh verifier with one module's sha moved and its digest recomputed, as an edit to that
    module produces."""
    mods = [[n, (sha if n == name else s)] for n, s in d["verifier"]["modules"]]
    d["verifier"] = dict(d["verifier"], modules=mods,
                         digest=charon._verifier_digest(mods, d["verifier"]["runtime"]))
    return d


def _flag(d, value):
    d["counts"]["receipt_check"] = dict(d["counts"]["receipt_check"], same_verifier_build=value)
    return d


def _sworn_moved(d):
    """What an edit to sworn.py makes the fresh derivation read: styxx.sworn's sha moved, and the
    receipt, issued by the old build, no longer names the installed one."""
    return _flag(_module_moved(d, "styxx.sworn"), False)


def _patch(monkeypatch, *steps):
    real = charon.derive

    def derive(path, repo):
        d = real(path, repo)
        for step in steps:
            d = step(d)
        return d
    monkeypatch.setattr(charon, "derive", derive)


def _reissue(repo, build):
    """The receipt rewritten in place, as another sworn build would issue it: the verifier block names
    `build` and the digest is recomputed, so it still VERIFIES and nothing but its build moved."""
    p = repo / "RESULT_x.sworn-receipt.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    rec["verifier"] = dict(rec["verifier"], sworn_sha256=build)
    body = {k: v for k, v in rec.items() if k not in sworn._RECEIPT_OUTSIDE_DIGEST}
    rec["digest"] = sworn._sha256(sworn._jcs(body).encode("utf-8"))
    sworn._write_json_lf(p, rec)


def test_the_fixture_is_the_case_the_amendment_names(sworn_repo, tmp_path):
    # the line records the flag true, and the receipt names the styxx.sworn sha the line names
    # (sworn.py is -text, so its raw sha and its LF sha are one string)
    line = _line(_log(sworn_repo, tmp_path))
    assert line["counts"]["receipt_check"]["same_verifier_build"] is True
    rec = json.loads((sworn_repo / "RESULT_x.sworn-receipt.json").read_text(encoding="utf-8"))
    assert rec["verifier"]["sworn_sha256"] == _sha_of(line["verifier"], "styxx.sworn")
    assert charon._receipt_build(sworn_repo / "RESULT_x.sworn.json") == rec["verifier"]["sworn_sha256"]


def test_only_the_flag_moved_and_sworn_moved_reads_moved_verifier(sworn_repo, tmp_path, monkeypatch):
    log = _log(sworn_repo, tmp_path)
    _patch(monkeypatch, _sworn_moved)
    rep = verify_log(log, sworn_repo)
    row = rep["lines"][0]
    assert _e5(row) == "SKEW"                                 # what E5 gave
    assert row["status"] == "MOVED_VERIFIER" and row["fields_changed"] == ["counts"]
    assert row["read_apart"] == [FLAG] and row["verifier_moved"] == ["modules"]
    assert rep["read_apart"] == {FLAG: 1}
    assert rep["by_status"]["MOVED_VERIFIER"] == 1 and rep["by_status"]["SKEW"] == 0
    assert FLAG in rep["note"]


def test_a_receipt_reissued_in_place_under_the_same_build_reads_drift(sworn_repo, tmp_path):
    # no derive patched: the receipt is rewritten to name another build and still VERIFIES; only the flag
    # moved, under the same verifier, and that is the bytes moving
    log = _log(sworn_repo, tmp_path)
    _reissue(sworn_repo, "d" * 64)
    rep = verify_log(log, sworn_repo)
    row = rep["lines"][0]
    rc = charon.derive(sworn_repo / "RESULT_x.sworn.json", sworn_repo)["counts"]["receipt_check"]
    assert rc["status"] == "VERIFIED" and rc["same_verifier_build"] is False
    assert row["fields_changed"] == ["counts"] and row["verifier_moved"] == []
    assert row["status"] == "DRIFT" == _e5(row) and row["read_apart"] == []
    assert rep["read_apart"] == {FLAG: 0}


def test_the_flag_alone_under_the_same_build_reads_drift(sworn_repo, tmp_path, monkeypatch):
    log = _log(sworn_repo, tmp_path)
    _patch(monkeypatch, lambda d: _flag(d, False))
    row = verify_log(log, sworn_repo)["lines"][0]
    assert row["fields_changed"] == ["counts"] and row["status"] == "DRIFT" == _e5(row) and row["read_apart"] == []


def _other(field):
    def counts_held(d):
        d["counts"]["HELD"] += 1
        return d

    def recorded_verdict(d):
        d["counts"]["recorded_verdict"] = "SWORN-FAILED"
        return d

    def rc_key(key, value):
        def step(d):
            d["counts"]["receipt_check"] = dict(d["counts"]["receipt_check"], **{key: value})
            return d
        return step

    def new_count(d):
        d["counts"]["WITHHELD_NEW"] = 1
        return d

    def receipts(d):
        d["receipts"] = dict(d["receipts"], n=d["receipts"]["n"] + 1,
                             sha256=sorted(d["receipts"]["sha256"] + ["a" * 64]))
        return d

    def subject(d):
        d["subject"] = dict(d["subject"], sha256="b" * 64)
        return d

    def verdict(d):
        d["verdict"], d["verdict_class"] = "SWORN-FAILED", "FAILED"
        return d

    def at(d):
        d["at"] = dict(d["at"], commit="c" * 40)
        return d

    def floor(d):
        d["floor"] = (d["floor"] or 0) + 0.125
        return d

    def reproduced(d):
        d["reproduced"] = False
        return d
    return {"counts.HELD": counts_held, "counts.recorded_verdict": recorded_verdict,
            "counts.receipt_check.status": rc_key("status", "FAILED"),
            "counts.receipt_check.digest_match": rc_key("digest_match", False),
            "counts.receipt_check.verdict_reproduces": rc_key("verdict_reproduces", False),
            "counts.receipt_check.schema": rc_key("schema", "styxx.sworn.verdict-receipt/v9"),
            "counts.new_key": new_count, "receipts": receipts, "subject": subject, "verdict": verdict,
            "at": at, "floor": floor, "reproduced": reproduced}[field]


OTHER_FIELDS = ("counts.HELD", "counts.recorded_verdict", "counts.receipt_check.status",
                "counts.receipt_check.digest_match", "counts.receipt_check.verdict_reproduces",
                "counts.receipt_check.schema", "counts.new_key", "receipts", "subject", "verdict", "at",
                "floor", "reproduced")


@pytest.mark.parametrize("field", OTHER_FIELDS)
@pytest.mark.parametrize("sworn_moved", (True, False), ids=("sworn-moved", "same-build"))
def test_the_flag_beside_any_other_moved_field_reads_what_e5_reads(sworn_repo, tmp_path, monkeypatch, field,
                                                                   sworn_moved):
    log = _log(sworn_repo, tmp_path)
    lead = _sworn_moved if sworn_moved else (lambda d: _flag(d, False))
    _patch(monkeypatch, lead, _other(field))
    # the other value did move: with the flag put back, the fresh line still differs from the recorded one
    line = _line(log)
    fresh = _flag(charon.derive(sworn_repo / "RESULT_x.sworn.json", sworn_repo), True)
    top = field.split(".")[0]
    assert charon.jcs(fresh[top]) != charon.jcs(line[top]), field
    row = verify_log(log, sworn_repo)["lines"][0]
    assert top in row["fields_changed"] and row["read_apart"] == []
    assert row["status"] == _e5(row) == ("SKEW" if sworn_moved else "DRIFT")


def test_a_flag_that_moved_false_to_true_is_not_read_apart(sworn_repo, tmp_path, monkeypatch):
    # the line was written while the receipt named an older build (false); a receipt re-issued by the
    # installed build reads true. sworn.py moved too, and the line still reads what E5 reads
    _patch(monkeypatch, lambda d: _flag(d, False))
    log = _log(sworn_repo, tmp_path)
    assert _line(log)["counts"]["receipt_check"]["same_verifier_build"] is False
    _patch(monkeypatch, lambda d: _flag(_module_moved(d, "styxx.sworn"), True))
    row = verify_log(log, sworn_repo)["lines"][0]
    assert row["fields_changed"] == ["counts"] and row["read_apart"] == []
    assert row["status"] == "SKEW" == _e5(row)


def test_a_receipt_that_names_another_build_is_not_read_apart(sworn_repo, tmp_path, monkeypatch):
    # sworn.py moved AND the receipt was rewritten in place to name a third build: the flag moved, and
    # the receipt no longer names the build the line names, so the move is not the instrument's alone
    log = _log(sworn_repo, tmp_path)
    _reissue(sworn_repo, "d" * 64)
    _patch(monkeypatch, _sworn_moved)
    row = verify_log(log, sworn_repo)["lines"][0]
    assert row["fields_changed"] == ["counts"] and row["read_apart"] == []
    assert row["status"] == "SKEW" == _e5(row)


def test_a_receipt_that_is_gone_or_unreadable_is_not_read_apart(sworn_repo, tmp_path, monkeypatch):
    log = _log(sworn_repo, tmp_path)
    _patch(monkeypatch, _sworn_moved)
    (sworn_repo / "RESULT_x.sworn-receipt.json").write_bytes(b"{not json")
    row = verify_log(log, sworn_repo)["lines"][0]
    assert row["read_apart"] == [] and row["status"] == "SKEW" == _e5(row)
    assert charon._receipt_build(sworn_repo / "RESULT_x.sworn.json") is None


@pytest.mark.parametrize("module", ("styxx.charon", "styxx.attestation", "styxx.claimdetect"))
def test_a_verifier_that_moved_outside_styxx_sworn_is_not_read_apart(sworn_repo, tmp_path, monkeypatch, module):
    # the flag names sworn.py's build; when sworn.py did not move, a moved flag is the receipt moving
    log = _log(sworn_repo, tmp_path)
    _patch(monkeypatch, lambda d: _flag(_module_moved(d, module), False))
    row = verify_log(log, sworn_repo)["lines"][0]
    assert row["verifier_moved"] == ["modules"] and row["fields_changed"] == ["counts"]
    assert row["read_apart"] == [] and row["status"] == "SKEW" == _e5(row)


def test_a_runtime_that_moved_is_not_read_apart(sworn_repo, tmp_path, monkeypatch):
    log = _log(sworn_repo, tmp_path)

    def runtime_moved(d):
        rt = dict(d["verifier"]["runtime"], version="3.99.0")
        d["verifier"] = dict(d["verifier"], runtime=rt, digest=charon._verifier_digest(d["verifier"]["modules"], rt))
        return _flag(d, False)
    _patch(monkeypatch, runtime_moved)
    row = verify_log(log, sworn_repo)["lines"][0]
    assert row["verifier_moved"] == ["runtime"] and row["read_apart"] == [] and row["status"] == "SKEW" == _e5(row)


@pytest.mark.parametrize("kind", ("capsule-oath", "capsule-diffgate", "oath-certificate"))
def test_a_line_of_another_kind_is_untouched(sworn_repo, tmp_path, monkeypatch, kind):
    # a line of another kind carrying the same counts and the same move reads what E5 reads
    def as_kind(d):
        d["kind"] = kind
        return d
    _patch(monkeypatch, as_kind)
    monkeypatch.setattr(charon, "_artifact_for", lambda e, r: Path(r) / "RESULT_x.sworn.json")
    log = _log(sworn_repo, tmp_path)
    assert _line(log)["kind"] == kind
    _patch(monkeypatch, as_kind, _sworn_moved)
    row = verify_log(log, sworn_repo)["lines"][0]
    assert row["fields_changed"] == ["counts"] and row["read_apart"] == []
    assert row["status"] == "SKEW" == _e5(row)


def test_a_line_whose_counts_are_not_what_a_writer_writes_never_raises(sworn_repo, tmp_path, monkeypatch):
    # receipt_check missing on the fresh side, or not an object: not read apart, and no traceback
    log = _log(sworn_repo, tmp_path)
    for bad in (None, "VERIFIED", ["x"]):
        def step(d, bad=bad):
            d = _module_moved(d, "styxx.sworn")
            if bad is None:
                d["counts"].pop("receipt_check")
            else:
                d["counts"]["receipt_check"] = bad
            return d
        _patch(monkeypatch, step)
        row = verify_log(log, sworn_repo)["lines"][0]
        assert row["read_apart"] == [] and row["status"] == "SKEW" == _e5(row)


def test_a_tampered_row_names_no_read_apart(sworn_repo, tmp_path):
    log = _log(sworn_repo, tmp_path)
    lines = log.read_text(encoding="utf-8").splitlines()
    e = json.loads(lines[1])
    e["counts"]["HELD"] += 1
    lines[1] = json.dumps(e, ensure_ascii=False)
    log.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    row = verify_log(log, sworn_repo)["lines"][0]
    assert row["status"] == "TAMPER" and row["read_apart"] is None


def _has_history() -> bool:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--is-shallow-repository"], capture_output=True,
                           text=True)
        return r.returncode == 0 and r.stdout.strip() == "false"
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _has_history(), reason="a shallow clone: the receipts' commits may not be here")
def test_the_fourteen_committed_lines_are_read_apart_or_not_moved():
    # the sworn lines of the committed log that recorded the flag true (2026-09-13..15): each re-derives
    # with nothing moved (the installed sworn.py is the one they name) or with only the flag moved and
    # read apart. No other compared field of any of them moved. The log is read, never written
    _, entries, _, _ = charon.read_log(LOG)
    flagged = [e for e in entries if e.get("kind") == "sworn"
               and ((e.get("counts") or {}).get("receipt_check") or {}).get("same_verifier_build") is True]
    assert [e["seq"] for e in flagged] == list(range(244, 258)), [e["seq"] for e in flagged]
    for e in flagged:
        art = charon._artifact_for(e, ROOT)
        fresh = charon.derive(art, ROOT)
        changed = [k for k in charon._COMPARED_KEYS if charon.jcs(fresh[k]) != charon.jcs(e[k])]
        same_build = fresh["verifier"]["digest"] == e["verifier"]["digest"]
        apart = charon._read_apart(e, fresh, changed, same_build, art)
        assert changed == [] or apart == [FLAG], (e["seq"], changed, apart)
