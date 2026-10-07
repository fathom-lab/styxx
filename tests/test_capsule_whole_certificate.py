# -*- coding: utf-8 -*-
"""Layer 2 of a v0.1 capsule compares the WHOLE certificate and the page, and says what it cannot
check.

On 2026-10-05 a check of the lab's own published capsule found that `capsule verify` compared a
chosen few fields: the verdict by class (the ", N uncovered" suffix stripped), the counts, and the
status of each embedded ledger row, one way. A certificate hand-edited anywhere else verified, and
printed, exactly like the genuine one while the page drew the edited values:

* D1: the verdict suffix, `uncovered` and `uncovered_items` edited to report 0 uncovered while the
  document holds a number nothing checked;
* D2: an advisory computed about a moved verdict string, never printed by the v0.1 command;
* D3: a ledger row deleted, a receipt_ref repointed, rows marked obligated, the epistemics summary
  rewritten.

A review of the earlier repair found more that verified like the genuine capsule: a
decoy payload in an HTML comment (layer 2 read it, the browser drew the other one), values
re-typed (30.0 and false for 30 and 0, a row on line `true`), fields nested in the receipt
binding or the payload, a free-text install line, a certificate posing as older than the band
while keeping a field certify wrote later, and a ledger in another order. The earlier repair had also
let `capsule create` mint certificates with no `col`, whose pages painted other numbers.

Each test below fails on 43b3b608 or on the earlier repair (daa05b66) and passes here. The rule they
pin: a field the certificate carries must re-derive from the embedded bytes, type for type; a field
the installed certify writes and the certificate lacks is NOT CHECKED by name, unless the
certificate itself shows it is not that old; the page must be the page a styxx renders for exactly
this payload; the mint environment is printed as stated by the minter, never as verified.
"""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from styxx.capsule import (_BEGIN, _END, SPEC, _render_html, create_capsule, main,
                           verify_capsule)
from styxx.certify import certify_doc

ROOT = Path(__file__).resolve().parent.parent
RECEIPT = {"eval": {"accuracy": 0.75, "items": 40}}
CLEAN = "The run scored 0.75 accuracy over 40 items.\n"
UNCOVERED = "The run scored 0.75 accuracy over 40 items. It was wrong on 12.\n"


def _mint(tmp_path, text, receipts=None, name="d", cert_edit=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = tmp_path / f"{name}.md"
    doc.write_text(text, encoding="utf-8")
    rps = []
    for rn, obj in (receipts or {"r.json": RECEIPT}).items():
        rp = tmp_path / rn
        rp.write_text(json.dumps(obj), encoding="utf-8")
        rps.append(rp)
    cert = certify_doc(doc, rps)
    if cert_edit:
        cert_edit(cert)
    cp = tmp_path / f"{name}.certificate.json"
    cp.write_text(json.dumps(cert), encoding="utf-8")
    out = tmp_path / f"{name}.capsule.html"
    create_capsule(doc, rps, cp, out)
    return out


def _payload_of(path):
    html = path.read_text(encoding="utf-8")
    i = html.index(_BEGIN) + len(_BEGIN)
    return json.loads(html[i:html.index(_END, i)])


def _forge(src, dst, edit):
    """What a forger does: edit the payload, then render the page this styxx renders for it."""
    payload = _payload_of(src)
    edit(payload)
    dst.write_text(_render_html(payload), encoding="utf-8")
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
    assert a["not_checked"] == [] and b["not_checked"] == []


def _fails_on(src, tmp_path, edit, needle):
    rep = verify_capsule(_forge(src, tmp_path / f"{edit.__name__}.capsule.html", edit))
    assert rep["ok"] is False, f"{edit.__name__} verified"
    assert any(needle in p for p in rep["problems"]), rep["problems"]
    return rep


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


_BAND = ("uncovered", "uncovered_items", "uncovered_excluded_by_rule", "uncovered_policy")


def _pose_as_pre_band(c):
    c["verdict"] = "OATH-HELD"
    for k in _BAND:
        del c[k]


def test_d1_a_pre_band_pose_that_keeps_a_later_field_fails(uncovered, tmp_path):
    """The band arrived on 2026-09-01, receipt_binding on 2026-09-05. A certificate carrying the
    binding and not the band was issued by no certify: the band was deleted. Until this repair it
    verified with ok True and the lines a genuinely old capsule prints (review finding, forge3)."""
    def kept_binding(p):
        _pose_as_pre_band(p["certificate"])
    rep = _fails_on(uncovered, tmp_path, kept_binding, "no certify issued that combination")
    assert any(p.startswith("certificate.uncovered is absent") and "receipt_binding" in p
               for p in rep["problems"])
    assert any(p.startswith("verdict not reproduced") for p in rep["problems"])


def test_d1_a_pre_band_pose_that_names_the_installed_certify_fails(uncovered, tmp_path):
    """Deleting the binding too leaves verifier_sha256 naming the installed certify.py, which
    writes the band."""
    def same_issuer(p):
        _pose_as_pre_band(p["certificate"])
        del p["certificate"]["receipt_binding"]
    _fails_on(uncovered, tmp_path, same_issuer, "names the installed certify.py")


def test_d1_a_pose_with_nothing_to_date_it_is_shown_the_live_coverage(uncovered, tmp_path):
    """What remains: delete the band and the binding and restate the issuer's hash, and the
    certificate is shaped like one issued between 2026-08-30 and 2026-09-01, which really exist.
    It verifies, with the reader told by name what was not checked and what the installed verifier
    finds instead, and create_capsule refuses to mint from it."""
    def old_issuer(p):
        _pose_as_pre_band(p["certificate"])
        del p["certificate"]["receipt_binding"]
        p["certificate"]["verifier_sha256"] = p["verifier"]["sha256"] = "1" * 64
    rep = verify_capsule(_forge(uncovered, tmp_path / "old.capsule.html", old_issuer))
    assert rep["ok"] is True, rep["problems"]
    assert rep["live_verdict"] == "OATH-HELD, 1 uncovered"
    nc = "\n".join(rep["not_checked"])
    assert "coverage suffix" in nc and "certificate.uncovered:" in nc and "installed verifier: 1" in nc
    assert any("coverage suffix" in a and "line 1 '12'" in a for a in rep["advisory"])
    assert "verdict class" in rep["compared"] and "verdict" not in rep["compared"]
    assert rep["mint_refusals"]


# ---------------------------------------------------------------- D3: every field, both ways

def test_d3_a_deleted_ledger_row_fails(clean, tmp_path):
    def deleted_row(p):
        p["certificate"]["ledger"] = [e for e in p["certificate"]["ledger"] if e["token"] != "40"]
    _fails_on(clean, tmp_path, deleted_row, "ledger omits a row the installed verifier finds")


def test_d3_an_added_ledger_row_fails(clean, tmp_path):
    def added_row(p):
        row = dict(p["certificate"]["ledger"][0], line=9)
        p["certificate"]["ledger"].append(row)
    _fails_on(clean, tmp_path, added_row, "ledger row not reproduced: line 9")


def test_d3_a_reordered_ledger_fails(clean, tmp_path):
    """certify writes rows in document order; an honest mint never carries another."""
    def reordered(p):
        p["certificate"]["ledger"].reverse()
    _fails_on(clean, tmp_path, reordered, "ledger rows are not in the order")


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


def test_d3_a_receipt_binding_digest_fails(clean, tmp_path):
    def digest(p):
        p["certificate"]["receipt_binding"]["receipts"][0]["content_sha256"] = "0" * 64
    _fails_on(clean, tmp_path, digest, "receipt_binding content_sha256 of 'r.json' not reproduced")


# ---------------------------------------------------------------- types: 1 is not 1.0 or true

def test_re_typed_counts_and_band_fail(clean, tmp_path):
    """Python's == says 30 == 30.0 and 0 == False; the page prints 'false' on the cards."""
    def retyped(p):
        c = p["certificate"]
        c["counts"] = {"VERIFIED": float(c["counts"]["VERIFIED"]), "ABSTAIN": False,
                       "UNGROUNDED": False}
        c["uncovered"] = False
    rep = _fails_on(clean, tmp_path, retyped, "counts not reproduced")
    assert any(p.startswith("uncovered not reproduced") for p in rep["problems"])


def test_a_row_on_line_true_is_not_the_row_on_line_1(clean, tmp_path):
    """(true, token, 0) aligned with (1, token, 0) and true != 1 is False, so the row passed
    while the page filed it under no line and left its number unpainted."""
    def line_true(p):
        p["certificate"]["ledger"][0]["line"] = True
    _fails_on(clean, tmp_path, line_true, "ledger row not reproduced: line true")


def test_a_re_typed_flag_inside_a_row_fails(clean, tmp_path):
    def flag_as_int(p):
        for e in p["certificate"]["ledger"]:
            e["epistemics"]["obligated"] = int(e["epistemics"]["obligated"])
    _fails_on(clean, tmp_path, flag_as_int, "epistemics embedded")


# ---------------------------------------------------------------- nested fields, the payload

@pytest.mark.parametrize("where", ["binding", "binding_row", "document", "receipt", "verifier"])
def test_a_field_no_styxx_writes_fails_at_any_depth(clean, tmp_path, where):
    def nested(p):
        rb = p["certificate"]["receipt_binding"]
        {"binding": lambda: rb.__setitem__("attested_by", "external auditor"),
         "binding_row": lambda: rb["receipts"][0].__setitem__("signature", "ab" * 32),
         "document": lambda: p["document"].__setitem__("note", "reviewed by an auditor"),
         "receipt": lambda: p["receipts"][0].__setitem__("provenance", "signed by the lab"),
         "verifier": lambda: p["verifier"].__setitem__("signed", True)}[where]()
    needle = {"binding": "receipt_binding.attested_by", "binding_row": "].signature",
              "document": "payload.document.note", "receipt": "payload.receipts[0].provenance",
              "verifier": "payload.verifier.signed"}[where]
    _fails_on(clean, tmp_path, nested, needle)


def test_the_install_line_must_be_the_one_create_writes(clean, tmp_path):
    """The page tells its reader which package checks it; a forged one pointed elsewhere."""
    def pip(p):
        p["verifier"]["pip"] = "styxx-capsule-tools==" + p["verifier"]["styxx_version"]
    _fails_on(clean, tmp_path, pip, "payload.verifier.pip 'styxx-capsule-tools")


def test_the_binding_s_repository_facts_must_be_a_combination_certify_writes(clean, tmp_path):
    """No repository at mint, yet a head, every receipt committed, and no blob: the review's T4."""
    def repo_claims(p):
        rb = p["certificate"]["receipt_binding"]
        rb["head"], rb["all_receipts_committed"] = "a" * 40, True
        for r in rb["receipts"]:
            r["committed"], r["path"] = True, "papers/forged/r.json"
    rep = _fails_on(clean, tmp_path, repo_claims, "no repository at mint and names a head")
    assert any("is not a combination certify writes" in p for p in rep["problems"])


def test_d3_mint_fields_are_printed_as_stated_never_as_verified(clean, tmp_path, capsys):
    def mint_env(p):
        p["created"] = "2020-01-01T00:00:00Z"
        p["verifier"]["styxx_version"] = "1.0.0"
        p["verifier"]["pip"] = "styxx==1.0.0"
    forged = _forge(clean, tmp_path / "mint.capsule.html", mint_env)
    rep = verify_capsule(forged)
    assert rep["ok"] is True, rep["problems"]   # nothing in the bytes can contradict them
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
    payload = _payload_of(src)
    assert [r["name"] for r in payload["receipts"]] == ["a.json", "z.json"]
    assert {e["receipt_ref"].split(":")[0] for e in payload["certificate"]["ledger"]} == {"z.json"}


# ---------------------------------------------------------------- the page around the payload

def test_an_edited_page_fails(clean, tmp_path):
    """Layer 2 used to name an edited page NOT CHECKED and pass; every capsule minted before
    2026-10-05 printed the same line, so an edited one could not be told from a genuine one."""
    assert any(c.startswith("the page") for c in verify_capsule(clean)["compared"])
    edited = tmp_path / "edited.capsule.html"
    edited.write_text(clean.read_text(encoding="utf-8").replace(
        "<main>", "<main><p>Reviewed and approved.</p>", 1), encoding="utf-8")
    rep = verify_capsule(edited)
    assert rep["ok"] is False
    assert any(p.startswith("the page around the payload is not the page any styxx renders")
               for p in rep["problems"])


def _with_decoy(html, genuine_json):
    k = html.index("<body>") + len("<body>")
    return html[:k] + "<!-- " + _BEGIN + genuine_json + _END + " -->" + html[k:]


def test_a_decoy_payload_in_a_comment_fails(clean, tmp_path):
    """Layer 2 found the payload by text, so it read a genuine payload hidden in an HTML comment
    while the browser, which skips comments, drew a forged one placed after it."""
    html = clean.read_text(encoding="utf-8")
    i = html.index(_BEGIN) + len(_BEGIN)
    genuine_json = html[i:html.index(_END, i)]
    p = _payload_of(clean)
    doc = base64.b64decode(p["document"]["b64"]).replace(b"0.75", b"0.95")
    p["document"]["b64"] = base64.b64encode(doc).decode("ascii")
    p["certificate"]["document_sha256"] = hashlib.sha256(doc).hexdigest()
    forged = tmp_path / "decoy.capsule.html"
    forged.write_text(_with_decoy(_render_html(p), genuine_json), encoding="utf-8")
    rep = verify_capsule(forged)
    assert rep["ok"] is False
    assert any("not the page any styxx renders" in x for x in rep["problems"])


# ---------------------------------------------------------------- names, crashes

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


def test_a_name_this_system_cannot_hold_fails_without_a_traceback(clean, tmp_path):
    def named(p):
        p["receipts"][0]["name"] = "r<1>.json"
        c = p["certificate"]
        c["receipts_sha256"] = {"r<1>.json": c["receipts_sha256"]["r.json"]}
    rep = verify_capsule(_forge(clean, tmp_path / "lt.capsule.html", named))
    assert rep["ok"] is False


def test_a_v01_page_relabelled_v02_fails_cleanly(clean, tmp_path):
    """The v0.2 path indexed payload['summary'] directly: a traceback, not a list of problems."""
    def relabel(p):
        p["spec"] = "styxx-oath/capsule/v0.2"
    forged = _forge(clean, tmp_path / "v02.capsule.html", relabel)
    rep = verify_capsule(forged)
    assert rep["ok"] is False and any("payload.summary" in p for p in rep["problems"])


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


# ---------------------------------------------------------------- the mint gate

def _strip_rows(field):
    def edit(cert):
        for e in cert["ledger"]:
            del e[field]
    return edit


def _june_shaped(cert):
    """A certificate shaped like those issued before 2026-08-24: no column, no epistemics, no
    band, no binding, another certify.py."""
    for e in cert["ledger"]:
        del e["col"], e["epistemics"]
    for k in _BAND + ("epistemics_summary", "receipt_binding"):
        del cert[k]
    cert["verifier_sha256"] = "1" * 64


@pytest.mark.parametrize("edit,why", [
    (_strip_rows("col"), "does not verify"),             # a later field dates it: verify fails
    (_strip_rows("epistemics"), "does not verify"),
    (_june_shaped, "verifies only with what its page draws from NOT CHECKED"),
])
def test_create_refuses_a_certificate_whose_rows_lack_what_the_page_draws_from(tmp_path, edit,
                                                                               why):
    """The 2026-09-01 refusal (commit 0f9b9e3f) kept pre-column certificates from being minted;
    the earlier round of this repair let them through, and their pages painted other numbers. A
    capsule already minted from one still verifies, with the fields NOT CHECKED."""
    d = tmp_path / "m"
    with pytest.raises(SystemExit) as e:
        _mint(d, CLEAN, cert_edit=edit)
    msg = str(e.value)
    assert f"REFUSED: the minted capsule {why}" in msg and "certificate.ledger[]." in msg
    assert not (d / "d.capsule.html").exists()


def test_certify_s_own_binding_failure_block_still_mints_and_verifies(tmp_path):
    """certify writes this block when binding fails, and promises the failure never blocks a
    certificate (R7). Its digests are NOT CHECKED; the receipts' bytes still are, by hash."""
    def failed(cert):
        rb = cert["receipt_binding"]
        cert["receipt_binding"] = {"schema": rb["schema"], "content_rule": rb["content_rule"],
                                   "head": None, "all_receipts_committed": False, "receipts": [],
                                   "note": "binding failed: OSError: probe"}
    cap = _mint(tmp_path / "bf", CLEAN, cert_edit=failed)
    rep = verify_capsule(cap)
    assert rep["ok"] is True, rep["problems"]
    assert any("binding failed at mint" in n for n in rep["not_checked"])


# ---------------------------------------------------------------- the committed capsules

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


# ---------------------------------------------------------------- the binding's repository facts
#
# Review round 2 (forgery, blocker): bind_at_mint marks a receipt committed, with its blob, only
# when the blob at head IS the receipt's bytes (as they are, or with LF or CRLF line ends). The
# blob is therefore a function of the embedded bytes, but layer 2 checked only its form, so a
# certificate certified over edited bytes and given the repository facts of an honest mint (a
# real head, the real path and the real committed blob) verified with output identical to the
# honest mint's.

def _blob(b: bytes) -> str:
    from styxx.receipt_binding import git_blob_id
    return git_blob_id(b)


def _as_committed(p, blob_of):
    """The binding a mint inside a repository writes, with each row's blob chosen by blob_of."""
    rb = p["certificate"]["receipt_binding"]
    rb.pop("note", None)
    rb["head"] = "a" * 40
    for r in rb["receipts"]:
        r["path"], r["committed"], r["blob"] = "papers/" + r["name"], True, blob_of(r)
    rb["all_receipts_committed"] = True


def _receipt_bytes(capsule):
    return {r["name"]: base64.b64decode(r["b64"]) for r in _payload_of(capsule)["receipts"]}


def test_a_committed_receipt_must_name_the_blob_of_its_embedded_bytes(clean, tmp_path):
    def blob_of_other_bytes(p):
        _as_committed(p, lambda r: _blob(json.dumps({"eval": {"accuracy": 0.95}}).encode()))
    _fails_on(clean, tmp_path, blob_of_other_bytes, "is not the git blob of the embedded receipt")


@pytest.mark.parametrize("form", ["as embedded", "LF", "CRLF"])
def test_a_committed_receipt_naming_its_own_blob_verifies_and_the_blob_is_printed(clean, tmp_path,
                                                                                 form):
    recs = _receipt_bytes(clean)

    def own_blob(r):
        b = recs[r["name"]].replace(b"\r\n", b"\n")
        return _blob({"as embedded": recs[r["name"]], "LF": b,
                      "CRLF": b.replace(b"\n", b"\r\n")}[form])

    rep = verify_capsule(_forge(clean, tmp_path / "own.capsule.html",
                                lambda p: _as_committed(p, own_blob)))
    assert rep["ok"] is True, rep["problems"]
    assert any(f"blob {own_blob({'name': 'r.json'})}" in s for s in rep["stated"]), rep["stated"]


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                    "-c", "core.autocrlf=false", *args], check=True, capture_output=True)


def test_a_forged_receipt_given_an_honest_mint_s_repository_facts_fails(tmp_path):
    """The review's f9, end to end: an honest mint inside a repository, then the receipt edited,
    certified outside it, and given the honest binding's head, paths, blobs and committed flags."""
    import shutil
    if not shutil.which("git"):
        pytest.skip("git is not on PATH")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "r.json").write_text(json.dumps(RECEIPT), encoding="utf-8")
    (repo / "d.md").write_text(CLEAN, encoding="utf-8")
    _git(repo, "add", "r.json", "d.md")
    _git(repo, "commit", "-q", "-m", "receipt")
    cert = certify_doc(repo / "d.md", [repo / "r.json"])
    assert cert["receipt_binding"]["all_receipts_committed"] is True
    cp = repo / "d.certificate.json"
    cp.write_text(json.dumps(cert), encoding="utf-8")
    honest = create_capsule(repo / "d.md", [repo / "r.json"], cp, tmp_path / "honest.capsule.html")
    assert verify_capsule(honest)["ok"] is True

    edited = {"eval": {"accuracy": 0.75, "items": 40, "reviewed": 1}}
    forged = _mint(tmp_path / "outside", CLEAN, receipts={"r.json": edited})
    real = cert["receipt_binding"]

    def borrow(p):
        rb = p["certificate"]["receipt_binding"]
        rb.pop("note", None)
        rb["head"], rb["all_receipts_committed"] = real["head"], True
        for r, h in zip(rb["receipts"], real["receipts"]):
            r["path"], r["blob"], r["committed"] = h["path"], h["blob"], h["committed"]
    rep = verify_capsule(_forge(forged, tmp_path / "forged.capsule.html", borrow))
    assert rep["ok"] is False
    assert any("is not the git blob of the embedded receipt" in p for p in rep["problems"])


@pytest.mark.parametrize("path", ["/etc/r.json", "papers\\r.json", "papers/../r.json", "C:/r.json",
                                  "papers/other.json", "papers/\x1b[2Kr.json", "papers//r.json"])
def test_a_binding_path_must_be_a_repository_path_ending_in_the_receipt_s_name(clean, tmp_path,
                                                                              path):
    recs = _receipt_bytes(clean)

    def pathed(p):
        _as_committed(p, lambda r: _blob(recs[r["name"]]))
        p["certificate"]["receipt_binding"]["receipts"][0]["path"] = path
    _fails_on(clean, tmp_path, pathed, "is not a repository path certify writes")


@pytest.mark.parametrize("note", ["no receipts", "reviewed by the lab", "binding failed: x"])
def test_a_binding_note_must_be_one_certify_writes_where_it_writes_it(clean, tmp_path, note):
    recs = _receipt_bytes(clean)

    def noted(p):
        _as_committed(p, lambda r: _blob(recs[r["name"]]))
        p["certificate"]["receipt_binding"]["note"] = note
    _fails_on(clean, tmp_path, noted, "receipt_binding.note")


# ---------------------------------------------------------------- the issuer's hash
#
# Review round 2 (forgery lens, blocker): certificate.verifier_sha256 could be absent, null, an
# object or free text, and the certificate still verified, printing it as stated. Every certify
# since the earliest (9ed6f3b5, 2026-06-10) writes it as 64 lowercase hex digits, and
# create_capsule copies it into payload.verifier.sha256.

@pytest.mark.parametrize("value", ["absent", None, {"sha256": "0" * 64}, "A" * 64, "0" * 63])
def test_the_issuer_s_hash_must_have_the_form_every_certify_writes(clean, tmp_path, value):
    def issuer(p):
        c = p["certificate"]
        if value == "absent":
            del c["verifier_sha256"]
        else:
            c["verifier_sha256"] = value
        p["verifier"]["sha256"] = c.get("verifier_sha256")
    _fails_on(clean, tmp_path, issuer, "certificate.verifier_sha256")


def test_the_payload_s_copy_of_the_issuer_s_hash_must_be_the_certificate_s(clean, tmp_path):
    def copy(p):
        p["verifier"]["sha256"] = "1" * 64
    _fails_on(clean, tmp_path, copy, "payload.verifier.sha256")


# ---------------------------------------------------------------- fields every certify writes
#
# Review round 2 (forgery lens, major): with the issuer's hash moved off the installed certify.py
# (free: it is stated, not checked), deleting `status` from the accused rows was NOT CHECKED, not a
# failure, on every committed capsule with no other edit, and both pages then painted the accused
# numbers verified. Every certify since the earliest (9ed6f3b5, 2026-06-10) writes status,
# receipt_ref, value, decimals and context in every ledger and ungrounded row, and oath, prereg,
# document, ungrounded and abstained at the top level, so their absence is an edit whoever issued
# the certificate. A row field that some rows carry and others lack, where the installed verifier
# writes it in all of them, is an edit too.

def _moved_issuer(p):
    p["certificate"]["verifier_sha256"] = p["verifier"]["sha256"] = "1" * 64


@pytest.mark.parametrize("field", ["status", "receipt_ref", "value", "decimals", "context"])
def test_a_row_field_every_certify_writes_cannot_be_deleted(clean, tmp_path, field):
    def deleted(p):
        _moved_issuer(p)
        del p["certificate"]["ledger"][0][field]
    _fails_on(clean, tmp_path, deleted, f"certificate.ledger[].{field} is absent from 1 of 2")


@pytest.mark.parametrize("field", ["oath", "prereg", "document", "ungrounded", "abstained"])
def test_a_top_level_field_every_certify_writes_cannot_be_deleted(clean, tmp_path, field):
    def deleted(p):
        _moved_issuer(p)
        del p["certificate"][field]
    _fails_on(clean, tmp_path, deleted, f"certificate.{field} is missing")


def test_the_review_s_status_deletion_on_a_committed_capsule_fails():
    """f1: the accused rows of the committed obligate1 capsule lose `status`, in the ledger and in
    `ungrounded`, under the page that capsule carries. Before this repair it verified."""
    from styxx._capsule_page_v01_legacy import render_html_v01_legacy
    src = ROOT / "papers" / "closed-model-frontier" / "RESULT_obligate1_does_not_ship_2026_08_31.capsule.html"
    p = _payload_of(src)
    for name in ("ledger", "ungrounded"):
        for e in p["certificate"][name]:
            if e.get("status") == "UNGROUNDED":
                del e["status"]
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        forged = Path(td) / "f1.capsule.html"
        forged.write_text(render_html_v01_legacy(p), encoding="utf-8")
        rep = verify_capsule(forged)
    assert rep["ok"] is False
    assert any(x.startswith("certificate.ledger[].status is absent from 3 of 57")
               for x in rep["problems"]), rep["problems"]


TABLE = "| run | accuracy | items |\n|---|---|---|\n| a | 0.75 | 40 |\n"


def test_a_row_field_some_rows_carry_and_others_lack_fails(tmp_path):
    """binding_context is written only on table rows; a table row without it, beside one with it,
    was removed."""
    src = _mint(tmp_path / "table", TABLE)
    assert all("binding_context" in e for e in _payload_of(src)["certificate"]["ledger"])

    def partial(p):
        _moved_issuer(p)
        del p["certificate"]["ledger"][0]["binding_context"]
    _fails_on(src, tmp_path, partial, "certificate.ledger[].binding_context is absent from 1 row")


def test_a_row_list_with_fields_not_checked_never_reads_every_field(tmp_path):
    def june_shaped(p):
        _june_shaped(p["certificate"])
        p["verifier"]["sha256"] = p["certificate"]["verifier_sha256"]
    src = _mint(tmp_path / "src", CLEAN)
    rep = verify_capsule(_forge(src, tmp_path / "june.capsule.html", june_shaped))
    assert rep["ok"] is True, rep["problems"]
    ledger = [c for c in rep["compared"] if c.startswith("ledger (")]
    assert ledger and "every field)" not in ledger[0] and "NOT CHECKED" in ledger[0], ledger


def test_a_certificate_without_its_epistemics_summary_is_not_minted(tmp_path):
    """The page draws its volunteered-share card from epistemics_summary. A certificate whose rows
    carry epistemics and which lacks the summary exists (two are committed, from the 26 minutes
    between the two changes on 2026-08-30), so verify prints it NOT CHECKED; create refuses it."""
    def no_summary(cert):
        for k in _BAND + ("epistemics_summary", "receipt_binding"):
            del cert[k]
        cert["verifier_sha256"] = "1" * 64
    d = tmp_path / "m"
    with pytest.raises(SystemExit) as e:
        _mint(d, CLEAN, cert_edit=no_summary)
    assert "certificate.epistemics_summary" in str(e.value)
    assert not (d / "d.capsule.html").exists()


# ---------------------------------------------------------------- the stated version and the floor

@pytest.mark.parametrize("version,below", [("0.1", True), ("7.48.0", True), ("7.48.1", True),
                                           ("7.49.0rc1", True), ("7.49.0", False),
                                           ("7.49.0.post1", False), ("7.50", False)])
def test_verify_advises_when_the_stated_version_is_below_the_floor(clean, tmp_path, version, below):
    """The stated version is printed as stated; below the floor, layer 2 also says that a styxx of
    that version passes certificates this one fails (PyPI 7.48.0 passes the D1 forgery)."""
    def stated(p):
        p["verifier"]["styxx_version"], p["verifier"]["pip"] = version, f"styxx=={version}"
    rep = verify_capsule(_forge(clean, tmp_path / "v.capsule.html", stated))
    assert rep["ok"] is True, rep["problems"]
    floor = [a for a in rep["advisory"] if "styxx>=7.49.0" in a]
    assert bool(floor) is below, rep["advisory"]
    if below:
        assert f"styxx {version}" in floor[0]


# ---------------------------------------------------------------- a renamed copy
#
# Review round 2 (both lenses, major and minor): layer 2 compares certificate.document, which
# certify writes as the document's file name, with the name the capsule gives its document. A
# capsule of an honest copy under another name, which 7.48.0 minted and verified, now fails, and
# create's refusal blamed the ledger schema. The comparison stays; the messages name the rename.

def test_a_renamed_copy_fails_and_verify_names_the_rename(clean, tmp_path):
    def renamed(p):
        p["document"]["name"] = "d_for_readers.md"
    rep = _fails_on(clean, tmp_path, renamed, "certificate.document 'd.md' is not the name")
    msg = [p for p in rep["problems"] if p.startswith("certificate.document")][0]
    assert "'d_for_readers.md'" in msg and "renamed copy" in msg


def test_create_refuses_a_renamed_copy_and_says_so(tmp_path):
    d = tmp_path / "ren"
    d.mkdir()
    doc = d / "report.md"
    doc.write_text(CLEAN, encoding="utf-8")
    rec = d / "r.json"
    rec.write_text(json.dumps(RECEIPT), encoding="utf-8")
    cp = d / "report.certificate.json"
    cp.write_text(json.dumps(certify_doc(doc, [rec])), encoding="utf-8")
    copy = d / "report_v2.md"
    copy.write_bytes(doc.read_bytes())
    with pytest.raises(SystemExit) as e:
        create_capsule(copy, [rec], cp, d / "report_v2.capsule.html")
    msg = str(e.value)
    assert "'report.md'" in msg and "'report_v2.md'" in msg and "renamed" in msg
    assert "ledger schema" not in msg
    assert not (d / "report_v2.capsule.html").exists()


# ---------------------------------------------------------------- review round 2 minors

def test_a_file_that_is_not_utf8_fails_without_a_traceback(clean, tmp_path, capsys):
    """D9 left one case: verify_capsule read the file outside any guard, so one 0xFF byte ended in
    UnicodeDecodeError, in the command and in charon."""
    from styxx import charon
    bad = tmp_path / "bad.capsule.html"
    bad.write_bytes(clean.read_bytes().replace(b"<title>", b"<title>\xff", 1))
    rep = verify_capsule(bad)
    assert rep["ok"] is False and rep["stage"] == "parse"
    assert any("UTF-8" in p for p in rep["problems"]), rep["problems"]
    assert main(["verify", str(bad)]) == 1
    assert "CAPSULE FAILS VERIFICATION" in capsys.readouterr().out
    line = charon.derive_capsule(bad, tmp_path)
    assert line["verdict"] == "UNRESOLVED"


@pytest.mark.parametrize("created", ["9999-99-99T99:99:99Z", "2026-02-30T12:00:00Z",
                                     "2026-10-06T24:00:00Z"])
def test_a_mint_time_no_clock_shows_fails(clean, tmp_path, created):
    def when(p):
        p["created"] = created
    _fails_on(clean, tmp_path, when, "payload.created")
