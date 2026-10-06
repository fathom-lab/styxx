"""A v0.1 capsule names the files its verifier writes when it re-runs the certifier.

The capsule chooses those names and the hashes they are checked against, so the only names the
verifier (and charon, which re-runs a capsule the same way) may write are bare file names: a name
that can only mean a file directly inside the temporary directory, on POSIX and on Windows.
"""
import json
import tempfile
from pathlib import Path

import pytest

from styxx import capsule as C
from styxx import charon

ROOT = Path(__file__).resolve().parents[1]
HONEST = ROOT / "papers" / "closed-model-frontier" / "RESULT_obligate1_does_not_ship_2026_08_31.capsule.html"


def _split(html):
    i = html.index(C._BEGIN) + len(C._BEGIN)
    j = html.index(C._END, i)
    return html[:i], json.loads(html[i:j]), html[j:]


def _forged(tmp_path, receipt_name=None, document_name=None):
    head, payload, tail = _split(HONEST.read_text(encoding="utf-8"))
    if receipt_name is not None:
        old = payload["receipts"][0]["name"]
        payload["receipts"][0]["name"] = receipt_name
        shas = payload["certificate"]["receipts_sha256"]
        shas[receipt_name] = shas.pop(old)
    if document_name is not None:
        payload["document"]["name"] = document_name
    out = tmp_path / "forged.capsule.html"
    out.write_text(head + json.dumps(payload).replace("</", "<\\/") + tail, encoding="utf-8")
    return out


@pytest.fixture
def own_tempdir(tmp_path, monkeypatch):
    """Put the verifier's temporary directories under tmp_path, so a climb out of one stays visible."""
    t = tmp_path / "t"
    t.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(t))
    return t


def test_the_honest_capsule_still_verifies():
    rep = C.verify_capsule(HONEST)
    assert rep["ok"], rep["problems"]


def test_an_absolute_receipt_name_is_refused_and_nothing_is_written(tmp_path, own_tempdir):
    target = tmp_path / "escape" / "written_by_verify.json"
    target.parent.mkdir()
    rep = C.verify_capsule(_forged(tmp_path, receipt_name=str(target)))
    assert not rep["ok"]
    assert any("not a bare file name" in p for p in rep["problems"])
    assert not target.exists()


def test_a_relative_climb_is_refused_and_nothing_is_written(tmp_path, own_tempdir):
    rep = C.verify_capsule(_forged(tmp_path, receipt_name="../climbed.json"))
    assert not rep["ok"]
    assert not (own_tempdir / "climbed.json").exists()
    assert not list(own_tempdir.rglob("climbed.json"))


def test_a_document_name_with_a_directory_is_refused(tmp_path, own_tempdir):
    rep = C.verify_capsule(_forged(tmp_path, document_name="../doc.md"))
    assert not rep["ok"]
    assert not (own_tempdir / "doc.md").exists()


def test_charon_does_not_write_where_a_capsule_points(tmp_path, own_tempdir):
    target = tmp_path / "escape" / "written_by_charon.json"
    target.parent.mkdir()
    line = charon.derive_capsule(_forged(tmp_path, receipt_name=str(target)), tmp_path)
    assert not target.exists()
    assert "OATH-HELD" not in json.dumps(line.get("verdict"))


def test_two_names_that_meet_on_a_case_insensitive_file_system_are_refused(tmp_path, own_tempdir):
    _, payload, _ = _split(HONEST.read_text(encoding="utf-8"))
    rep = C.verify_capsule(_forged(tmp_path, receipt_name=payload["document"]["name"].upper()))
    assert not rep["ok"]


@pytest.mark.parametrize("name", ["a.json", "UPDATE_2026_10_05.md", "receipts.v2.json", "obligate1_result.json",
                                  "console.json", "nullable.md", "com10.json"])
def test_bare_names(name):
    assert C._bare_name(name)


@pytest.mark.parametrize("name", ["", ".", "..", "a/b.json", "a\\b.json", "C:x.json", "/etc/x", "../x.json",
                                  "x.json.", "x.json ", "CON", "con.json", "nul .txt", "COM1.json", "lpt9",
                                  "a\x00b", "a\nb", None, 3])
def test_names_that_are_not_bare(name):
    assert not C._bare_name(name)
