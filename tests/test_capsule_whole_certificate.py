# -*- coding: utf-8 -*-
"""Layer 2 of a v0.1 capsule compares the WHOLE certificate, and says what it cannot check.

On 2026-10-05 a check of the lab's own published capsule found that `capsule verify` compared a
chosen few fields: the verdict by class (the ", N uncovered" suffix stripped), the counts, and the
status of each embedded ledger row, one way. A certificate hand-edited anywhere else verified, and
printed, exactly like the genuine one while the page drew the edited values:

* D1: the verdict suffix, `uncovered` and `uncovered_items` edited to report 0 uncovered while the
  document holds a number nothing checked;
* D2: an advisory computed about a moved verdict string, never printed by the v0.1 command;
* D3: a ledger row deleted, a receipt_ref repointed, rows marked obligated, the epistemics summary
  rewritten, the mint time and minting version rewritten.

Each test below fails on 43b3b608 and passes with the repair. The rule they pin: a field the
certificate carries must re-derive from the embedded bytes; a field the installed certify writes
and the certificate lacks is printed NOT CHECKED by name with the installed verifier's value; the
mint environment is printed as stated by the minter, never as verified.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from styxx.capsule import _BEGIN, _END, SPEC, create_capsule, main, verify_capsule
from styxx.certify import certify_doc

ROOT = Path(__file__).resolve().parent.parent
RECEIPT = {"eval": {"accuracy": 0.75, "items": 40}}
CLEAN = "The run scored 0.75 accuracy over 40 items.\n"
UNCOVERED = "The run scored 0.75 accuracy over 40 items. It was wrong on 12.\n"


def _mint(tmp_path, text, receipts=None, name="d"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = tmp_path / f"{name}.md"
    doc.write_text(text, encoding="utf-8")
    rps = []
    for rn, obj in (receipts or {"r.json": RECEIPT}).items():
        rp = tmp_path / rn
        rp.write_text(json.dumps(obj), encoding="utf-8")
        rps.append(rp)
    cert = certify_doc(doc, rps)
    cp = tmp_path / f"{name}.certificate.json"
    cp.write_text(json.dumps(cert), encoding="utf-8")
    out = tmp_path / f"{name}.capsule.html"
    create_capsule(doc, rps, cp, out)
    return out


def _forge(src, dst, edit):
    html = src.read_text(encoding="utf-8")
    i = html.index(_BEGIN) + len(_BEGIN)
    j = html.index(_END, i)
    payload = json.loads(html[i:j])
    edit(payload)
    dst.write_text(html[:i] + json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
                   + html[j:], encoding="utf-8")
    return dst


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    return _mint(tmp_path_factory.mktemp("clean"), CLEAN)


@pytest.fixture(scope="module")
def uncovered(tmp_path_factory):
    return _mint(tmp_path_factory.mktemp("uncovered"), UNCOVERED)


def test_the_fixtures_are_what_the_tests_assume(clean, uncovered):
    a, b = verify_capsule(clean), verify_capsule(uncovered)
    assert a["ok"] and b["ok"], (a["problems"], b["problems"])
    assert a["verdict"] == "OATH-HELD" and b["verdict"] == "OATH-HELD, 1 uncovered"


# ---------------------------------------------------------------- D1: the coverage band

def test_d1_a_certificate_edited_to_report_no_uncovered_number_fails(uncovered, tmp_path):
    def edit(p):
        c = p["certificate"]
        c["verdict"], c["uncovered"], c["uncovered_items"] = "OATH-HELD", 0, []
    rep = verify_capsule(_forge(uncovered, tmp_path / "d1.capsule.html", edit))
    assert rep["ok"] is False
    assert any(p.startswith("verdict not reproduced") for p in rep["problems"])
    assert any(p.startswith("uncovered not reproduced") for p in rep["problems"])
    assert any(p.startswith("uncovered_items not reproduced") and '"token": "12"' in p
               for p in rep["problems"])


def test_d1_a_suffix_alone_cannot_be_added_or_dropped(clean, uncovered, tmp_path):
    def add(p):
        p["certificate"]["verdict"] = "OATH-HELD, 7 uncovered"
    def drop(p):
        p["certificate"]["verdict"] = "OATH-HELD"
    for src, edit in ((clean, add), (uncovered, drop)):
        rep = verify_capsule(_forge(src, tmp_path / f"{edit.__name__}.capsule.html", edit))
        assert rep["ok"] is False
        assert any(p.startswith("verdict not reproduced") for p in rep["problems"])


def test_d1_a_forger_posing_as_an_old_certificate_is_shown_the_live_coverage(uncovered, tmp_path):
    """Deleting the band's fields and the suffix makes the certificate look older than the band.
    That is not a failure (an older certify really wrote such certificates), but the reader is
    told, by name, what was not checked and what the installed verifier finds instead."""
    def edit(p):
        c = p["certificate"]
        c["verdict"] = "OATH-HELD"
        for k in ("uncovered", "uncovered_items", "uncovered_excluded_by_rule", "uncovered_policy"):
            del c[k]
    rep = verify_capsule(_forge(uncovered, tmp_path / "old.capsule.html", edit))
    assert rep["ok"] is True
    assert rep["live_verdict"] == "OATH-HELD, 1 uncovered"
    nc = "\n".join(rep["not_checked"])
    assert "coverage suffix" in nc and "certificate.uncovered:" in nc and "installed verifier: 1" in nc
    assert any("coverage suffix" in a and "line 1 '12'" in a for a in rep["advisory"])
    assert "verdict class" in rep["compared"] and "verdict" not in rep["compared"]


# ---------------------------------------------------------------- D3: every field, both ways

def _fails_on(src, tmp_path, edit, needle):
    rep = verify_capsule(_forge(src, tmp_path / f"{edit.__name__}.capsule.html", edit))
    assert rep["ok"] is False, f"{edit.__name__} verified"
    assert any(needle in p for p in rep["problems"]), rep["problems"]


def test_d3_a_deleted_ledger_row_fails(clean, tmp_path):
    def deleted_row(p):
        p["certificate"]["ledger"] = [e for e in p["certificate"]["ledger"] if e["token"] != "40"]
    _fails_on(clean, tmp_path, deleted_row, "ledger omits a row the installed verifier finds")


def test_d3_an_added_ledger_row_fails(clean, tmp_path):
    def added_row(p):
        row = dict(p["certificate"]["ledger"][0], line=9)
        p["certificate"]["ledger"].append(row)
    _fails_on(clean, tmp_path, added_row, "ledger row not reproduced: line 9")


def test_d3_a_repointed_receipt_ref_fails(clean, tmp_path):
    def repointed(p):
        for e in p["certificate"]["ledger"]:
            if e["token"] == "0.75":
                e["receipt_ref"] = "r.json:eval.items"
    _fails_on(clean, tmp_path, repointed, "receipt_ref embedded 'r.json:eval.items'")


def test_d3_an_obligated_flag_fails(clean, tmp_path):
    def flag(p):
        for e in p["certificate"]["ledger"]:
            e["epistemics"]["obligated"] = not e["epistemics"]["obligated"]
    _fails_on(clean, tmp_path, flag, "ledger divergence at line 1 token '0.75': epistemics")


def test_d3_a_rewritten_epistemics_summary_fails(clean, tmp_path):
    def summary(p):
        p["certificate"]["epistemics_summary"]["obligated_total"] += 1
    _fails_on(clean, tmp_path, summary, "epistemics_summary not reproduced")


def test_d3_a_field_the_installed_certify_does_not_write_fails(clean, tmp_path):
    def extra(p):
        p["certificate"]["reviewed_by_a_human"] = True
    _fails_on(clean, tmp_path, extra, "certificate.reviewed_by_a_human cannot be reproduced")


def test_d3_a_receipt_the_capsule_does_not_carry_fails(clean, tmp_path):
    def phantom(p):
        p["certificate"]["receipts_sha256"]["phantom.json"] = "0" * 64
    _fails_on(clean, tmp_path, phantom, "receipts_sha256 not reproduced")


def test_d3_a_receipt_binding_digest_fails(tmp_path):
    src = _mint(tmp_path / "rb", CLEAN)
    html = src.read_text(encoding="utf-8")
    assert '"receipt_binding"' in html

    def digest(p):
        p["certificate"]["receipt_binding"]["receipts"][0]["content_sha256"] = "0" * 64
    _fails_on(src, tmp_path, digest, "receipt_binding content_sha256 of 'r.json' not reproduced")


def test_d3_mint_fields_are_printed_as_stated_never_as_verified(clean, tmp_path, capsys):
    def mint_env(p):
        p["created"] = "2020-01-01T00:00:00Z"
        p["verifier"]["styxx_version"] = "1.0.0"
    forged = _forge(clean, tmp_path / "mint.capsule.html", mint_env)
    rep = verify_capsule(forged)
    assert rep["ok"] is True                     # nothing in the bytes can contradict them
    st = "\n".join(rep["stated"])
    assert "created 2020-01-01T00:00:00Z" in st and "styxx_version 1.0.0" in st
    assert main(["verify", str(forged)]) == 0
    out = capsys.readouterr().out
    assert "stated by the minter, not checked: created 2020-01-01T00:00:00Z" in out
    verified_line = [ln for ln in out.splitlines() if ln.startswith("VERIFIED")][0]
    assert "2020-01-01" not in verified_line and "1.0.0" not in verified_line


def test_the_receipts_are_re_read_in_the_order_the_certificate_lists_them(tmp_path):
    """Two receipts carry the same value; receipt_ref names the one the certifier read earlier.
    The capsule stores receipts sorted by name, so verifying in that order would move the ref."""
    src = _mint(tmp_path / "order", "The run scored 0.75 accuracy over 40 items.\n",
                receipts={"z.json": RECEIPT, "a.json": RECEIPT})
    rep = verify_capsule(src)
    assert rep["ok"] is True, rep["problems"]
    html = src.read_text(encoding="utf-8")
    i = html.index(_BEGIN) + len(_BEGIN)
    payload = json.loads(html[i:html.index(_END, i)])
    assert [r["name"] for r in payload["receipts"]] == ["a.json", "z.json"]
    assert {e["receipt_ref"].split(":")[0] for e in payload["certificate"]["ledger"]} == {"z.json"}


@pytest.mark.parametrize("where", ["absolute", "climbing"])
def test_a_capsule_that_names_a_path_writes_nothing_outside_the_verifier(clean, tmp_path, where,
                                                                         monkeypatch):
    """The re-run writes each embedded file under the name the capsule gives it. Until
    2026-10-05 a receipt named with an absolute path (or ../) was written there, on the reader's
    machine, with bytes the capsule chose."""
    import tempfile
    (tmp_path / "t").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "t"))   # the re-run's temp dirs
    if where == "absolute":
        target = tmp_path / "outside" / "written_by_verify.json"
        target.parent.mkdir()
        name = str(target)
    else:
        target = tmp_path / "t" / "written_by_verify.json"
        name = "../written_by_verify.json"

    def named(p):
        old = p["receipts"][0]["name"]
        p["receipts"][0]["name"] = name
        c = p["certificate"]
        c["receipts_sha256"] = {name: c["receipts_sha256"][old]}
    rep = verify_capsule(_forge(clean, tmp_path / "named.capsule.html", named))
    assert rep["ok"] is False and rep["live_verdict"] is None
    assert any("is not a bare file name" in p for p in rep["problems"])
    assert not target.exists()


# ---------------------------------------------------------------- D2: the command prints it all

def test_d2_the_v01_command_prints_every_advisory_and_not_checked_field(capsys):
    """A committed capsule minted before the uncovered band: verify_capsule computes the advisory;
    until 2026-10-05 the v0.1 branch of the command dropped it."""
    cap = (ROOT / "papers" / "closed-model-frontier"
           / "RESULT_v14_naming_the_defects_did_not_save_it_2026_09_01.capsule.html")
    rep = verify_capsule(cap)
    assert rep["advisory"]
    assert main(["verify", str(cap)]) == 0
    out = capsys.readouterr().out
    for adv in rep["advisory"]:
        assert f"  advisory: {adv}" in out
    for nc in rep["not_checked"]:
        assert f"  NOT CHECKED: {nc}" in out
    assert "installed verifier's verdict: OATH-HELD, 5 uncovered" in out


def test_d2_the_command_exits_nonzero_when_a_carried_field_does_not_reproduce(uncovered, tmp_path,
                                                                             capsys):
    def edit(p):
        c = p["certificate"]
        c["verdict"], c["uncovered"], c["uncovered_items"] = "OATH-HELD", 0, []
    assert main(["verify", str(_forge(uncovered, tmp_path / "d1.capsule.html", edit))]) == 1
    out = capsys.readouterr().out
    assert "CAPSULE FAILS VERIFICATION" in out and "uncovered not reproduced" in out


# ---------------------------------------------------------------- the committed capsules

def _payload_of(path):
    html = path.read_text(encoding="utf-8")
    i = html.index(_BEGIN) + len(_BEGIN)
    return json.loads(html[i:html.index(_END, i)])


def _committed_v01():
    try:
        files = subprocess.run(["git", "-C", str(ROOT), "ls-files", "*.capsule.html"],
                               capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        files = [p.relative_to(ROOT).as_posix() for p in ROOT.glob("papers/**/*.capsule.html")]
    return sorted(f for f in files if _payload_of(ROOT / f).get("spec") == SPEC)


@pytest.mark.parametrize("rel", _committed_v01())
def test_every_committed_v01_capsule_still_verifies(rel):
    """None of them may be edited (a receipt is history). Each predates a field or two the
    installed certify writes; those are NOT CHECKED by name, never silently skipped."""
    rep = verify_capsule(ROOT / rel)
    assert rep["ok"] is True, rep["problems"]
    carried = set(_payload_of(ROOT / rel)["certificate"])
    if "uncovered" not in carried:
        assert any(n.startswith("certificate.uncovered:") for n in rep["not_checked"])
    if "receipt_binding" not in carried:
        assert any(n.startswith("certificate.receipt_binding:") for n in rep["not_checked"])
