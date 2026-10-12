# -*- coding: utf-8 -*-
"""charon's verifier names its runtime: verifier scheme styxx.charon/verifier/v1
(papers/charon/AMENDMENT_charon_verifier_names_its_runtime_2026_10_12.md, phase 3 of the 2026-10-11 design).

What is held here: a fresh line's verifier carries the scheme, module shas over LF bytes (a CRLF copy of a
module hashes as its LF copy), the three runtime fields the capsule reader compares and nothing else, and a
digest over both; capsule-diffgate ends with styxx.instrument, styxx.declare and styxx.attestation, and the
other kinds keep their names; no sys.version and no path is written. Under a monkeypatched derive, runtime
and module hash, verify reads every case the amendment names: MOVED_VERIFIER and SKEW naming runtime,
modules or both, a verdict the interpreter moved reading SKEW ["runtime"] and not DRIFT, DRIFT only under the
same module bytes and the same runtime, unattributed for a hand-edited digest. A legacy-shaped line never
reads SAME_LINE, even when its shas are the LF ones. Lines seq 89 and 95 of the committed log, re-derived
from the working tree, move no compared key and read MOVED_VERIFIER ["legacy"], and the 257 lines written
before the amendment are legacy. The note names the runtime; by_verifier_part counts every row; a scheme-v1
sha is recorded with the lf: prefix. A verifier no writer wrote is read, never raised on, chained or not.

The full re-derivation of the committed log under this verifier is tests/test_stranger.py's ferry-log
step, unedited: every line must read SAME_LINE or MOVED_VERIFIER.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

from styxx import charon, diffgate, sworn
from styxx import instrument as I
from styxx.attestation import jcs
from styxx.charon import VERIFIER_SCHEME, header_digest, ingest, read_log, verify_log

ROOT = Path(__file__).resolve().parent.parent
PAPERS = ROOT / "papers" / "closed-model-frontier"
DOGFOOD = PAPERS / "DOGFOOD_session_2026_08_31.capsule.html"
HANDOFF = PAPERS / "HANDOFF_capsule_v02_2026_08_31.capsule.html"
LOG = ROOT / "papers" / "charon" / "charon.log.jsonl"
LEGACY_LINES = 257          # the lines written before the amendment; an ingest after it appends scheme-v1 lines


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


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
    return repo


def _log_with(repo, tmp_path, *artifacts):
    log = tmp_path / "c.log.jsonl"
    ingest(list(artifacts), log, repo, timestamp="2026-10-12T00:00:00Z", population="test")
    return log


def _rechain(lines):
    """The header as it is, then every line with seq, prev and entry_id rebuilt over what it now holds."""
    header = json.loads(lines[0])
    prev = header_digest(header)
    out = [lines[0]]
    for i, ln in enumerate(lines[1:], start=1):
        e = json.loads(ln)
        e.pop("entry_id", None)
        e["seq"], e["prev"] = i, prev
        core = {k: v for k, v in e.items() if k not in ("timestamp", "note")}
        e["entry_id"] = _sha(jcs(core).encode("utf-8"))
        prev = e["entry_id"]
        out.append(json.dumps(e, ensure_ascii=False))
    return out


def _write_log(path: Path, lines) -> Path:
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return path


def _with_verifier(log: Path, tmp_path: Path, verifier, rechain: bool = True) -> Path:
    """The one-line log with its line's verifier replaced, re-chained (or not: a TAMPER line)."""
    lines = [ln for ln in log.read_text(encoding="utf-8").split("\n") if ln]
    e = json.loads(lines[1])
    e["verifier"] = verifier
    lines = [lines[0], json.dumps(e, ensure_ascii=False)]
    return _write_log(tmp_path / "edited.log.jsonl", _rechain(lines) if rechain else lines)


def _legacy(verifier: dict) -> dict:
    """A verifier as every line before the amendment wrote it: no scheme, no runtime, a digest over the
    module list alone. The shas are kept as given."""
    mods = [list(p) for p in verifier["modules"]]
    return {"styxx_version": verifier["styxx_version"], "modules": mods, "digest": _sha(jcs(mods).encode("utf-8"))}


# ------------------------------------------------------------------------- 1. a fresh line's verifier

def test_a_fresh_line_names_its_scheme_lf_module_shas_its_runtime_and_a_digest_over_both(sworn_repo):
    v = charon.derive(sworn_repo / "RESULT_x.sworn.json", sworn_repo)["verifier"]
    assert VERIFIER_SCHEME == "styxx.charon/verifier/v1" and v["scheme"] == VERIFIER_SCHEME
    assert list(v) == ["scheme", "styxx_version", "modules", "runtime", "digest"]
    assert [m for m, _ in v["modules"]] == list(charon._MODULES["sworn"])
    for name, sha in v["modules"]:
        f = importlib.import_module(name).__file__
        assert sha == I.lf_sha256(f) == _sha(Path(f).read_bytes().replace(b"\r\n", b"\n")), name
    assert v["runtime"] == {"implementation": sys.implementation.name, "version": platform.python_version(),
                            "unicode": unicodedata.unidata_version}
    assert v["digest"] == _sha(jcs({"modules": v["modules"], "runtime": v["runtime"]}).encode("utf-8"))
    assert v["digest"] != _legacy(v)["digest"]          # the legacy rule over the same shas gives another digest
    # names and hex only: no path, no sys.version, no case-table label (O9)
    assert all(re.fullmatch(r"styxx\.[a-z_]+", m) and re.fullmatch(r"[0-9a-f]{64}", s) for m, s in v["modules"])
    text = json.dumps(v)
    assert sys.version not in text and "case_floor" not in text and "detail" not in text


def test_a_crlf_copy_of_a_module_hashes_as_its_lf_copy_and_a_lone_cr_is_kept(tmp_path, monkeypatch):
    src = b'"""a probe module for charon\'s LF rule."""\nX = 1\n\nY = "two"\n'
    (tmp_path / "p3_charon_lf_probe.py").write_bytes(src)
    (tmp_path / "p3_charon_crlf_probe.py").write_bytes(src.replace(b"\n", b"\r\n"))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    try:
        lf = charon._module_pair("p3_charon_lf_probe")
        crlf = charon._module_pair("p3_charon_crlf_probe")
    finally:
        for name in ("p3_charon_lf_probe", "p3_charon_crlf_probe"):
            sys.modules.pop(name, None)
    assert lf == ("p3_charon_lf_probe", _sha(src)) and crlf == ("p3_charon_crlf_probe", _sha(src))
    assert _sha(src.replace(b"\n", b"\r\n")) != crlf[1]       # the raw rule of a legacy line would differ
    assert charon._content_sha256(b"a\rb") != charon._content_sha256(b"a\nb")
    assert charon._module_pair("p3_charon_no_such_module") == ("p3_charon_no_such_module", None)


def test_capsule_diffgate_ends_with_the_three_added_modules_and_the_other_kinds_keep_their_names():
    assert charon._MODULES == {
        "sworn": ("styxx.charon", "styxx.sworn", "styxx.attestation", "styxx.claimdetect"),
        "capsule-oath": ("styxx.charon", "styxx.capsule", "styxx.certify"),
        "capsule-diffgate": ("styxx.charon", "styxx.capsule", "styxx.diffgate", "styxx.evidence",
                             "styxx.claimdetect", "styxx.instrument", "styxx.declare", "styxx.attestation"),
        "oath-certificate": ("styxx.charon", "styxx.corpus_audit", "styxx.certify"),
    }
    v = charon.derive(DOGFOOD, ROOT)["verifier"]
    assert v["scheme"] == VERIFIER_SCHEME
    assert [m for m, _ in v["modules"]] == list(charon._MODULES["capsule-diffgate"])
    shas = dict(v["modules"])
    assert all(shas.values())
    # one LF rule on both sides: the styxx.diffgate sha charon names is the one a record names (phase 2)
    assert shas["styxx.diffgate"] == I.record_identity(diffgate.__file__)["build"]["modules"][0][1]
    # every kind's line carries the scheme and the runtime, an UNRESOLVED one included
    for kind in charon.KINDS:
        u = charon._unresolved(kind, "x.md", "x.md", "test")["verifier"]
        assert u["scheme"] == VERIFIER_SCHEME and u["runtime"] == v["runtime"]
        assert [m for m, _ in u["modules"]] == list(charon._MODULES[kind])


# ------------------------------------------------------------------------- 2. what verify says moved

def _elsewhere():
    """Another interpreter, as the runtime reads it."""
    rt = dict(_REAL["runtime"]())
    rt.update(version="0.0.0-elsewhere", unicode="99.0.0")
    return rt


def _module_moved(name):
    n, sha = _REAL["pair"](name)
    return (n, "f" * 64) if name == "styxx.sworn" else (n, sha)


def _receipts_grew(path, repo):
    d = _REAL["derive"](path, repo)
    d["receipts"] = {"n": 2, "sha256": sorted(d["receipts"]["sha256"] + ["a" * 64]), "cited": None,
                     "vacuous": False}
    return d


def _verdict_moved(path, repo):
    d = _REAL["derive"](path, repo)
    d["verdict"], d["verdict_class"] = "SWORN-FAILED", "FAILED"
    return d


def _hand_edited(path, repo):
    d = _REAL["derive"](path, repo)
    d["verifier"]["digest"] = "0" * 64
    return d


def _hand_edited_receipts(path, repo):
    d = _receipts_grew(path, repo)
    d["verifier"]["digest"] = "0" * 64
    return d


_REAL = {"runtime": I.runtime, "pair": charon._module_pair, "derive": charon.derive}


@pytest.mark.parametrize("runtime, pair, derive, status, moved", [
    ("real", "real", "real", "SAME_LINE", []),
    ("elsewhere", "real", "real", "MOVED_VERIFIER", ["runtime"]),
    ("elsewhere", "real", "receipts", "SKEW", ["runtime"]),
    ("elsewhere", "real", "verdict", "SKEW", ["runtime"]),         # the interpreter moved it: SKEW, not DRIFT
    ("real", "moved", "real", "MOVED_VERIFIER", ["modules"]),
    ("real", "moved", "receipts", "SKEW", ["modules"]),
    ("elsewhere", "moved", "receipts", "SKEW", ["modules", "runtime"]),
    ("real", "real", "receipts", "DRIFT", []),
    ("real", "real", "verdict", "DRIFT", []),
    ("real", "real", "hand_edited", "MOVED_VERIFIER", ["unattributed"]),
    ("real", "real", "hand_edited_receipts", "SKEW", ["unattributed"]),
])
def test_verify_names_what_moved_in_the_verifier(sworn_repo, tmp_path, monkeypatch, runtime, pair, derive,
                                                 status, moved):
    log = _log_with(sworn_repo, tmp_path, sworn_repo / "RESULT_x.sworn.json")
    monkeypatch.setattr(I, "runtime", {"real": _REAL["runtime"], "elsewhere": _elsewhere}[runtime])
    monkeypatch.setattr(charon, "_module_pair", {"real": _REAL["pair"], "moved": _module_moved}[pair])
    monkeypatch.setattr(charon, "derive", {"real": _REAL["derive"], "receipts": _receipts_grew,
                                           "verdict": _verdict_moved, "hand_edited": _hand_edited,
                                           "hand_edited_receipts": _hand_edited_receipts}[derive])
    rep = verify_log(log, sworn_repo)
    row = rep["lines"][0]
    assert (row["status"], row["verifier_moved"], row["verifier_scheme"]) == (status, moved, "v1")
    assert rep["by_status"][status] == 1
    assert rep["by_verifier_part"]["+".join(moved) or "same"] == 1 and sum(rep["by_verifier_part"].values()) == 1
    real_rt = {k: _REAL["runtime"]()[k] for k in I.RUNTIME_COMPARED}
    assert row["runtime_recorded"] == real_rt
    assert row["runtime_now"] == ({k: _elsewhere()[k] for k in I.RUNTIME_COMPARED} if runtime == "elsewhere"
                                  else real_rt)


# ------------------------------------------------------------------------------ 3. a legacy line

def test_a_legacy_shaped_line_reads_moved_verifier_legacy_and_never_same_line(sworn_repo, tmp_path, monkeypatch):
    log = _log_with(sworn_repo, tmp_path, sworn_repo / "RESULT_x.sworn.json")
    fresh = read_log(log)[1][0]["verifier"]
    # the LF shas themselves, under the legacy digest rule: no fresh digest can equal it
    legacy = _with_verifier(log, tmp_path, _legacy(fresh))
    rep = verify_log(legacy, sworn_repo)
    row = rep["lines"][0]
    assert rep["chain_problems"] == [] and row["fields_changed"] == []
    assert (row["status"], row["verifier_moved"], row["verifier_scheme"]) == ("MOVED_VERIFIER", ["legacy"], "legacy")
    assert row["runtime_recorded"] is None and row["runtime_now"] == fresh["runtime"]
    assert rep["by_verifier_part"]["legacy"] == 1
    assert not any(s.startswith("lf:") for v in rep["verifier_builds_recorded"].values() for s in v)
    # the core moved too: SKEW ["legacy"], never DRIFT, whatever moved it
    for moved_it in (_receipts_grew, _verdict_moved):
        monkeypatch.setattr(charon, "derive", moved_it)
        row = verify_log(legacy, sworn_repo)["lines"][0]
        assert (row["status"], row["verifier_moved"]) == ("SKEW", ["legacy"])


def _committed_lines():
    lines = LOG.read_text(encoding="utf-8").split("\n")
    return lines[0], [ln for ln in lines[1:] if ln]


def test_lines_89_and_95_from_the_working_tree_move_no_compared_key_and_read_moved_verifier_legacy(tmp_path):
    seqs = (89, 95)
    header, body = _committed_lines()
    picked = [ln for ln in body if json.loads(ln)["seq"] in seqs]
    assert len(picked) == len(seqs)
    names = {89: DOGFOOD.name, 95: HANDOFF.name}
    for ln in picked:
        e = json.loads(ln)
        assert e["kind"] == "capsule-diffgate" and e["subject"]["name"] == names[e["seq"]]
        assert "scheme" not in e["verifier"] and "runtime" not in e["verifier"]
    small = _write_log(tmp_path / "picked.log.jsonl", _rechain([header] + picked))
    rep = verify_log(small, ROOT)
    assert rep["chain_problems"] == [] and rep["chain_broken_at_line"] is None
    for row, ln in zip(rep["lines"], picked):
        assert row["fields_changed"] == [], row
        assert (row["status"], row["verifier_moved"], row["verifier_scheme"]) == ("MOVED_VERIFIER", ["legacy"], "legacy")
        assert row["runtime_recorded"] is None and row["runtime_now"] == charon._verifier_runtime()
        assert row["verifier_recorded"] == json.loads(ln)["verifier"]["digest"][:12]
    assert rep["by_verifier_part"] == dict({p: 0 for p in charon._VERIFIER_PARTS}, legacy=len(seqs))
    # a legacy sha is the raw one the line names (seq 89's styxx.diffgate is the CRLF rendering), unprefixed
    assert rep["verifier_builds_recorded"]["styxx.diffgate"] == ["405b7d426380"]


def test_the_lines_written_before_the_amendment_are_legacy_so_none_reads_same_line_again():
    header, entries, problems, _facts = read_log(LOG)
    assert problems == [] and charon.check_chain(header, entries) == []
    before = [e for e in entries if e["seq"] <= LEGACY_LINES]
    assert len(before) == LEGACY_LINES
    for e in before:
        v = e["verifier"]
        assert charon._verifier_scheme(v) == "legacy" and "runtime" not in v, e["seq"]
        # the legacy digest is over the module list alone, a JSON array; a scheme-v1 digest is over an object
        assert v["digest"] == _sha(jcs(v["modules"]).encode("utf-8")), e["seq"]


# ------------------------------------------------------------------------------- 4. the report

def test_the_note_names_the_runtime_and_the_report_counts_what_moved(sworn_repo, tmp_path, capsys):
    log = _log_with(sworn_repo, tmp_path, sworn_repo / "RESULT_x.sworn.json",
                    sworn_repo / "RESULT_x.sworn.json")
    rep = verify_log(log, sworn_repo)
    note = rep["note"]
    assert "runtime" in note and "a legacy line names no runtime" in note
    assert "DRIFT = the core moved under the same module bytes and the same runtime" in note
    assert "SKEW with runtime" in note and "unattributed" in note
    assert charon.STATUSES == ("SAME_LINE", "MOVED_VERIFIER", "SKEW", "DRIFT", "UNRESOLVED", "HEAD_MISMATCH",
                               "TAMPER")
    assert list(rep["by_status"]) == list(charon.STATUSES)
    assert list(rep["by_verifier_part"]) == list(charon._VERIFIER_PARTS)
    assert rep["by_verifier_part"]["same"] == 2 and sum(rep["by_verifier_part"].values()) == rep["entries"] == 2
    # a scheme-v1 sha is recorded with the lf: prefix, so it is never counted as the raw sha of a legacy line
    assert rep["verifier_builds_recorded"]["styxx.charon"] == ["lf:" + charon._module_pair("styxx.charon")[1][:12]]
    assert charon.main(["verify", "--log", str(log), "--repo", str(sworn_repo)]) == 0
    out = capsys.readouterr().out
    assert "SAME_LINE=2" in out and "verifier moved: same=2  modules=0  runtime=0" in out


# ------------------------------------------------------------- 5. a verifier no writer wrote

@pytest.mark.parametrize("verifier, scheme, moved", [
    ({"modules": [[["x"], 5]], "digest": 7}, "legacy", ["legacy"]),
    ({"modules": [["styxx.charon", 5]], "digest": "0" * 64}, "legacy", ["legacy"]),
    ({}, "legacy", ["legacy"]),
    ({"scheme": ["v1"], "modules": "text", "runtime": "text", "digest": None}, "unknown", ["unattributed"]),
    ({"scheme": "styxx.charon/verifier/v9", "modules": [], "runtime": {}, "digest": "0" * 64}, "unknown",
     ["unattributed"]),
    ({"scheme": VERIFIER_SCHEME, "modules": [[1, 2, 3]], "runtime": [1], "digest": "0" * 64}, "v1",
     ["modules", "runtime"]),
    ({"scheme": VERIFIER_SCHEME, "modules": [["styxx.charon", None]], "runtime": None, "digest": []}, "v1",
     ["modules", "runtime"]),
])
def test_a_verifier_no_writer_wrote_is_read_never_raised_on(sworn_repo, tmp_path, verifier, scheme, moved):
    log = _log_with(sworn_repo, tmp_path, sworn_repo / "RESULT_x.sworn.json")
    rep = verify_log(_with_verifier(log, tmp_path, verifier), sworn_repo)
    row = rep["lines"][0]
    assert rep["chain_problems"] == []
    assert (row["status"], row["verifier_scheme"], row["verifier_moved"]) == ("MOVED_VERIFIER", scheme, moved)
    assert row["verifier_recorded"] == (verifier.get("digest")[:12] if isinstance(verifier.get("digest"), str) else "")
    # and not chained: TAMPER, not re-derived, nothing compared, no traceback
    rep = verify_log(_with_verifier(log, tmp_path, verifier, rechain=False), sworn_repo)
    row = rep["lines"][0]
    assert row["status"] == "TAMPER" and row["verifier_moved"] is None and row["runtime_now"] is None
    assert row["verifier_scheme"] == scheme and rep["by_verifier_part"]["not_compared"] == 1


def test_a_hand_edited_digest_on_a_line_with_both_parts_as_fresh_is_unattributed(sworn_repo, tmp_path):
    log = _log_with(sworn_repo, tmp_path, sworn_repo / "RESULT_x.sworn.json")
    v = dict(read_log(log)[1][0]["verifier"], digest="1" * 64)
    row = verify_log(_with_verifier(log, tmp_path, v), sworn_repo)["lines"][0]
    assert (row["status"], row["verifier_moved"]) == ("MOVED_VERIFIER", ["unattributed"])
