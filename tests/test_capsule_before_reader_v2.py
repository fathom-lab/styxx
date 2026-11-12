# -*- coding: utf-8 -*-
"""A v0.1 capsule whose certificate was issued before the numeral reader v2 still verifies.

Found on 2026-11-12, after styxx 8.0.0 was published: `python -m styxx.capsule verify` in 8.0.0
exits 1 with `uncovered_policy not reproduced` on an intact capsule minted by 7.50.0. The reader
v2 rewrote the policy block (schema string, note, counted reasons), and layer 2 compares every
certificate field whole, so a block that describes the reader failed a certificate no byte of
which had changed. The page of such a capsule tells its reader to `pip install "styxx>=7.49.0"`,
which installs the release that fails it.

The fixture `tests/fixtures/capsule_v01_minted_by_styxx_7_50_0.html` was minted on 2026-11-12 by
the `styxx` package unpacked from tag v7.50.0 with its line ends as committed
(`git -c core.autocrlf=false archive v7.50.0 styxx`; `python -m styxx.certify`, then
`python -m styxx.capsule create`), outside any repository, from a one-line document and a receipt
of three leaves. Its certificate names the sha-256 of the tag's own `styxx/certify.py`. It is a
record: the test pins the digest of its payload, and a new case is a new file, never this one
minted again.

The rule these tests pin. For a certificate that names no numeral reader, the installed certify
is run a second time on the embedded bytes with the reader v2 off, in a private copy of the
module. The policy block is accepted only if the whole certificate is, field for field and type
for type, what that earlier reader issues. Every field but the policy block must also equal what
the installed reader issues, as before. So an edited or retyped block fails, the earlier block
beside a reader block fails, a certificate of the reader v2 dressed as an older one fails when a
row of it is the reader v2's, and a row the two readers disagree on fails.

One thing passes that no certify of the day issued, and it is pinned here so that it is seen: a
certificate that is byte for byte what the earlier reader issues for a document, produced today
by removing the reader block from a new certificate and changing the issuer hash. Nothing in the
bytes tells it from one issued then, and it says of the document what the earlier reader says.

The draft of this repair switched the installed module's reader flag off around one call. A
review ran it in threads and it passed a capsule it must fail. `test_threads_*` pins that the
repair writes nothing shared.

A second thing 8.0.0 got wrong, found by review of this repair on 2026-11-12. The reader v2 adds a
key to some ledger rows (`ellipsis`, `magnitude`, `sign_form`). Where it adds the key and moves
nothing else (a numeral directly before U+2026 on a spec, historical or notation line), 8.0.0
printed on an intact capsule of 7.49.0 or 7.50.0, beside the policy line, "no styxx minted this
page around it", which was false: those releases minted that page. With the policy block
repaired, that line was the one failure left, while the same certificate around the page 7.48.0
and 7.48.1 rendered verified: the page decided the verdict. The rule now, pinned at the end of
this file, does not ask which page it is. A certificate whose every carried field is what the
earlier reader issues (and that does not name the installed certify.py as its issuer) has the
key NOT CHECKED, which is no reason to fail. A certificate that shows the reader v2 and lacks the
key fails as edited. Where a row's status moves, the capsule still fails, with its row.

Not tested here: capsules minted by 7.48.0, 7.48.1 and 7.49.0 (run by hand on 2026-11-12, see the
CHANGELOG entry of this change; no fixture of theirs is committed), and the mirror case, 7.50.0
verifying a capsule minted by a later styxx, which this change cannot reach.
"""
from __future__ import annotations

import base64
import hashlib
import importlib
import json
import threading
from pathlib import Path

import pytest

from styxx.capsule import (_BEGIN, _END, _carries_the_earlier_readers_fields,
                           _certify_before_reader_v2, _earlier_reader_certificate,
                           _issued_by_the_earlier_reader, _render_html, create_capsule,
                           verify_capsule)

# The module by its name: the package also exports a function named `certify`, and which of the
# two the attribute `styxx.certify` holds depends on what was imported last.
certify = importlib.import_module("styxx.certify")
capsule_mod = importlib.import_module("styxx.capsule")

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "capsule_v01_minted_by_styxx_7_50_0.html"
# sha-256 of the payload text between the page's payload tags (one line, so line ends cannot move it)
FIXTURE_PAYLOAD_SHA256 = "88b97a1d6238b5e3a7e59922d5422854017e36c5a879732a1bdc9180f50e1d0e"
# sha-256 of styxx/certify.py at tag v7.50.0 (`git show v7.50.0:styxx/certify.py | sha256sum`)
CERTIFY_7_50_0_SHA256 = "c448ded108df0f54de7a68cb0755fd31f574e21d4ace8c6b6b4b427cd8e9ea59"
V1 = "styxx-oath/uncovered-band/v1"
V2 = "styxx-oath/uncovered-band/v2"
# what 8.0.0 printed on an intact capsule of 7.49.0 or 7.50.0, and what must not be printed on one
FALSE_LINE = "no styxx minted this page"

RECEIPT = {"eval": {"accuracy": 0.75, "items": 40}}
CLEAN = "The run scored 0.75 accuracy over 40 items. It was wrong on 12.\n"
# the reader v2 reads a magnitude after a currency symbol; the reader before it did not
MAGNITUDE = "The run scored 0.75 accuracy over 40 items. The bill was $99.9k.\n"
# a numeral directly before U+2026 on a spec line: the reader v2 adds `ellipsis` to the row and
# moves nothing else of it
ELLIPSIS_SPEC = ("The run scored 0.75 accuracy over 40 items.\n"
                 "The check requires 0.85\u2026 or more to pass.\n")
# the same on a plain line: the reader v2 also withdraws the row's VERIFIED
ELLIPSIS_PLAIN = "The run scored 0.75\u2026 accuracy over 40 items.\n"
# both in one document: one row the reader v2 moves, one row it only adds the key to
ELLIPSIS_BOTH = ("The run scored 0.75\u2026 accuracy over 40 items.\n"
                 "The check requires 0.85\u2026 or more to pass.\n")


def _payload_text(path: Path) -> str:
    html = path.read_text(encoding="utf-8")
    i = html.index(_BEGIN) + len(_BEGIN)
    return html[i:html.index(_END, i)]


def _payload_of(path: Path) -> dict:
    return json.loads(_payload_text(path))


def _forge(src: Path, dst: Path, edit) -> Path:
    """What a forger does: edit the payload, then render the page this styxx renders for it."""
    payload = _payload_of(src)
    edit(payload)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(_render_html(payload), encoding="utf-8")
    return dst


def _mint(tmp_path: Path, text: str, name: str = "d") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = tmp_path / f"{name}.md"
    doc.write_text(text, encoding="utf-8")
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps(RECEIPT), encoding="utf-8")
    cp = tmp_path / f"{name}.certificate.json"
    cp.write_text(json.dumps(certify.certify_doc(doc, [rp])), encoding="utf-8")
    out = tmp_path / f"{name}.capsule.html"
    create_capsule(doc, [rp], cp, out)
    return out


def _issuer_of_the_fixture() -> dict:
    return dict(_payload_of(FIXTURE)["verifier"])


def _earlier_block(count: int) -> dict:
    return dict(_payload_of(FIXTURE)["certificate"]["uncovered_policy"], excluded_by_rule=count)


def _mint_as_7_50_0(tmp_path: Path, monkeypatch, text: str) -> Path:
    """A capsule as the earlier reader issues it: certified and minted in this one thread with the
    installed reader switched off for the minting only, then given the issuer the fixture states
    (the 7.50.0 certify.py), since a certificate that names the installed certify.py and lacks a
    field that file writes fails as edited."""
    with monkeypatch.context() as m:
        m.setattr(certify, "V15_READER_V2", False)
        minted = _mint(tmp_path / "earlier", text)
    assert certify.V15_READER_V2 is True
    issuer = _issuer_of_the_fixture()

    def redress(payload: dict) -> None:
        payload["certificate"]["verifier_sha256"] = issuer["sha256"]
        payload["verifier"] = issuer

    return _forge(minted, tmp_path / "as_7_50_0.capsule.html", redress)


def _dress_as_older(minted: Path, dst: Path) -> Path:
    """What V-2 of the review did to a certificate of the reader v2: delete the reader block, put
    the earlier policy block in its place, and name the 7.50.0 certify.py as the issuer."""
    issuer = _issuer_of_the_fixture()

    def redress(payload: dict) -> None:
        cert = payload["certificate"]
        del cert["numeral_reader"]
        cert["uncovered_policy"] = _earlier_block(cert["uncovered_policy"]["excluded_by_rule"])
        cert["verifier_sha256"] = issuer["sha256"]
        payload["verifier"] = issuer

    return _forge(minted, dst, redress)


def _problems(rep: dict) -> str:
    return "\n".join(rep["problems"])


# ── the record ────────────────────────────────────────────────────

def test_the_fixture_is_the_record_it_is_named_for():
    assert hashlib.sha256(_payload_text(FIXTURE).encode("utf-8")).hexdigest() == FIXTURE_PAYLOAD_SHA256
    payload = _payload_of(FIXTURE)
    cert = payload["certificate"]
    assert payload["verifier"] == {"sha256": CERTIFY_7_50_0_SHA256, "styxx_version": "7.50.0",
                                   "pip": "styxx==7.50.0"}
    assert cert["verifier_sha256"] == CERTIFY_7_50_0_SHA256
    assert cert["uncovered_policy"]["schema"] == V1
    assert "numeral_reader" not in cert
    assert cert["verdict"] == "OATH-HELD, 1 uncovered"
    assert cert["receipt_binding"]["head"] is None, "minted outside any repository"


def test_a_capsule_minted_by_7_50_0_verifies():
    assert certify.V15_READER_V2 is True, "the reader v2 ships on; this test is about that state"
    rep = verify_capsule(FIXTURE)
    assert rep["ok"], _problems(rep)
    assert rep["verdict"] == rep["live_verdict"] == "OATH-HELD, 1 uncovered"
    policy = [c for c in rep["compared"] if c.startswith("uncovered_policy")]
    assert policy and "numeral reader v2 off" in policy[0], rep["compared"]
    # the reader is named beside the result, and its own block is listed as not checked
    assert any("numeral_reader" in a for a in rep["advisory"])
    assert any("certificate.numeral_reader" in n for n in rep["not_checked"])
    assert certify.V15_READER_V2 is True


# ── what must still fail ──────────────────────────────────────────

def _count_plus_one(payload):
    payload["certificate"]["uncovered_policy"]["excluded_by_rule"] += 1


def _count_as_float(payload):
    block = payload["certificate"]["uncovered_policy"]
    block["excluded_by_rule"] = float(block["excluded_by_rule"])


def _count_as_bool(payload):
    block = payload["certificate"]["uncovered_policy"]
    assert block["excluded_by_rule"] == 0
    block["excluded_by_rule"] = False


def _note_rewritten(payload):
    payload["certificate"]["uncovered_policy"]["scope_limit"] = "every number was checked."


def _reasons_emptied(payload):
    payload["certificate"]["uncovered_policy"]["counted_reasons"] = []


def _key_added(payload):
    payload["certificate"]["uncovered_policy"]["reviewed"] = True


@pytest.mark.parametrize("edit", [_count_plus_one, _count_as_float, _count_as_bool, _note_rewritten,
                                  _reasons_emptied, _key_added], ids=lambda f: f.__name__)
def test_an_edited_or_retyped_policy_block_of_an_older_certificate_fails(tmp_path, edit):
    """`0.0` and `False` equal `0` to Python's `==`; the comparison keeps types, so they fail."""
    rep = verify_capsule(_forge(FIXTURE, tmp_path / "edited.html", edit))
    assert not rep["ok"], edit.__name__
    assert "uncovered_policy not reproduced" in _problems(rep), edit.__name__


def test_the_earlier_block_beside_a_reader_block_fails(tmp_path):
    """The exception is for a certificate that names no reader. One issued under the reader v2
    carries that reader's block, so the earlier block in it was put there afterwards."""
    minted = _mint(tmp_path / "now", CLEAN)
    assert "numeral_reader" in _payload_of(minted)["certificate"]

    def swap(payload):
        cert = payload["certificate"]
        cert["uncovered_policy"] = _earlier_block(cert["uncovered_policy"]["excluded_by_rule"])

    rep = verify_capsule(_forge(minted, tmp_path / "swapped.html", swap))
    assert not rep["ok"]
    assert "uncovered_policy not reproduced" in _problems(rep)


def test_a_reader_block_removed_from_a_certificate_of_the_installed_certify_fails(tmp_path):
    """Removing the reader block and putting the earlier policy block in its place does not turn a
    certificate of the installed certify into an older one: it still names its issuer."""
    minted = _mint(tmp_path / "now", CLEAN)

    def redress(payload):
        cert = payload["certificate"]
        del cert["numeral_reader"]
        cert["uncovered_policy"] = _earlier_block(cert["uncovered_policy"]["excluded_by_rule"])

    rep = verify_capsule(_forge(minted, tmp_path / "redressed.html", redress))
    assert not rep["ok"]
    assert "a field was removed" in _problems(rep)


def test_a_reader_v2_certificate_dressed_as_an_older_one_fails(tmp_path):
    """The review's V-2. Under the reader v2 `$99.9k` is a ledger row with a `magnitude`; no earlier
    reader wrote that row. With the reader block deleted, the earlier policy block swapped in and
    the 7.50.0 certify.py named as issuer, the draft of this repair verified the file. It is not
    what the earlier reader issues for these bytes, so its policy block fails, as under 8.0.0."""
    minted = _mint(tmp_path / "now", MAGNITUDE)
    cert = _payload_of(minted)["certificate"]
    assert any("magnitude" in row for row in cert["ledger"])
    rep = verify_capsule(_dress_as_older(minted, tmp_path / "dressed.html"))
    assert not rep["ok"]
    assert "uncovered_policy not reproduced" in _problems(rep)


def test_a_row_the_two_readers_disagree_on_still_fails(tmp_path, monkeypatch):
    """The exception covers the policy block and nothing else. `$99.9k` is no ledger row under the
    earlier reader and an UNGROUNDED row under the reader v2, so the verdict moves and the capsule
    fails, with the advisory naming the two readers."""
    capsule = _mint_as_7_50_0(tmp_path, monkeypatch, MAGNITUDE)
    assert _payload_of(capsule)["certificate"]["verdict"].startswith("OATH-HELD")
    rep = verify_capsule(capsule)
    assert not rep["ok"]
    text = _problems(rep)
    assert "verdict not reproduced" in text
    assert "uncovered_policy not reproduced" not in text, "the policy block is the earlier reader's own"
    assert any("numeral_reader" in a for a in rep["advisory"])


# ── what passes, and is said ──────────────────────────────────────

def test_an_older_certificate_minted_here_with_the_reader_off_verifies(tmp_path, monkeypatch):
    """The same rule on a second document, minted by this code under the earlier reader."""
    capsule = _mint_as_7_50_0(tmp_path, monkeypatch, CLEAN)
    cert = _payload_of(capsule)["certificate"]
    assert cert["uncovered_policy"]["schema"] == V1 and "numeral_reader" not in cert
    rep = verify_capsule(capsule)
    assert rep["ok"], _problems(rep)
    assert any(c.startswith("uncovered_policy (") for c in rep["compared"])


def test_a_certificate_that_is_what_the_earlier_reader_issues_passes_whoever_made_it(tmp_path):
    """Pinned so that it is seen. For a document the two readers read alike, a certificate of the
    reader v2 with its reader block deleted, the earlier policy block swapped in and another
    issuer named is, byte for byte, the certificate the earlier reader issues. Verify passes it,
    with the reader block listed as not checked and the advisory naming the two readers."""
    minted = _mint(tmp_path / "now", CLEAN)
    dressed = _dress_as_older(minted, tmp_path / "dressed.html")
    rep = verify_capsule(dressed)
    assert rep["ok"], _problems(rep)
    assert any("certificate.numeral_reader" in n for n in rep["not_checked"])
    assert any("numeral_reader" in a for a in rep["advisory"])


def test_the_installed_block_in_a_certificate_that_names_no_reader_is_left_as_it_was(tmp_path):
    """Not changed here, and pinned so that it is seen: a certificate that names no reader and
    carries the reader v2's block verifies, as it does under 8.0.0, with the reader block listed as
    not checked. The block equals what the installed reader re-derives, so it is compared as that
    reader's block and the exception is not used. No certify issued that combination; whether
    verify should refuse it is a separate question from the one this file settles."""
    live_block = _payload_of(_mint(tmp_path / "now", CLEAN))["certificate"]["uncovered_policy"]
    assert live_block["schema"] == V2

    def swap(payload):
        payload["certificate"]["uncovered_policy"] = dict(
            live_block, excluded_by_rule=payload["certificate"]["uncovered_policy"]["excluded_by_rule"])

    rep = verify_capsule(_forge(FIXTURE, tmp_path / "swapped.html", swap))
    assert rep["ok"], _problems(rep)
    assert "uncovered_policy" in rep["compared"]
    assert any("certificate.numeral_reader" in n for n in rep["not_checked"])


def test_create_mints_from_a_7_50_0_certificate_when_no_row_moves(tmp_path):
    """`capsule create` keeps a capsule only if it verifies, so it follows the same rule."""
    payload = _payload_of(FIXTURE)
    doc = tmp_path / payload["document"]["name"]
    doc.write_bytes(base64.b64decode(payload["document"]["b64"]))
    rps = []
    for r in payload["receipts"]:
        rp = tmp_path / r["name"]
        rp.write_bytes(base64.b64decode(r["b64"]))
        rps.append(rp)
    cp = tmp_path / "older.certificate.json"
    cp.write_text(json.dumps(payload["certificate"]), encoding="utf-8")
    out = create_capsule(doc, rps, cp, tmp_path / "again.capsule.html")
    rep = verify_capsule(out)
    assert rep["ok"], _problems(rep)
    assert _payload_of(out)["certificate"]["uncovered_policy"]["schema"] == V1


# ── nothing shared is written ─────────────────────────────────────

def test_the_earlier_reader_is_a_private_copy_and_the_installed_flag_is_not_written():
    earlier = _certify_before_reader_v2()
    assert earlier is not certify
    assert earlier is _certify_before_reader_v2(), "loaded once"
    assert earlier.V15_READER_V2 is False and certify.V15_READER_V2 is True
    assert Path(earlier.__file__) == Path(certify.__file__), "so the issuer hash it writes is the installed file's"
    import sys
    assert sys.modules["styxx.certify"] is certify
    assert not any(m is earlier for m in list(sys.modules.values()))


def test_the_installed_reader_is_never_switched(tmp_path, monkeypatch):
    """Every write of the installed module's flag is recorded; a verify of each kind makes none."""
    writes = []
    module_type = type(certify)

    class Watched(module_type):
        def __setattr__(self, name, value):
            if name == "V15_READER_V2":
                writes.append(value)
            super().__setattr__(name, value)

    new = _mint(tmp_path / "now", CLEAN)
    older = _mint_as_7_50_0(tmp_path, monkeypatch, MAGNITUDE)
    monkeypatch.setattr(certify, "__class__", Watched)
    for path, want in ((FIXTURE, True), (new, True), (older, False)):
        assert verify_capsule(path)["ok"] is want
    assert writes == []
    assert certify.V15_READER_V2 is True


def test_threads_give_what_one_thread_gives(tmp_path, monkeypatch):
    """The draft switched a shared flag: on four threads it passed the `$99.9k` capsule in most
    runs and failed intact ones. Here every thread gets the one-thread answer every time."""
    new = _mint(tmp_path / "now", CLEAN)
    older_moved = _mint_as_7_50_0(tmp_path, monkeypatch, MAGNITUDE)
    cases = [(FIXTURE, True), (new, True), (older_moved, False)]
    for path, want in cases:
        assert verify_capsule(path)["ok"] is want
    wrong, raised = [], []

    def work(path, want):
        try:
            for _ in range(4):
                got = verify_capsule(path)["ok"]
                if got is not want:
                    wrong.append((path.name, got))
        except BaseException as e:  # noqa: BLE001 - reported below
            raised.append(repr(e))

    threads = [threading.Thread(target=work, args=case) for case in cases for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not raised, raised
    assert not wrong, wrong
    assert certify.V15_READER_V2 is True


# ── the helper's own contract ─────────────────────────────────────

class _Recorder:
    def __init__(self, result=None, error=None):
        self.calls, self.result, self.error = 0, result, error

    def certify_doc(self, doc, receipts):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


@pytest.mark.parametrize("cert, live", [
    ({"uncovered_policy": {}, "numeral_reader": {}}, {"numeral_reader": {}}),   # names a reader
    ({"uncovered_policy": {}, "numeral_reader": None}, {"numeral_reader": {}}),  # names one, badly
    ({"numeral_reader": {}}, {"numeral_reader": {}}),                           # no policy block
    ({"uncovered_policy": {}}, {}),                                             # installed reader off
    ([], {"numeral_reader": {}}),                                               # not an object
], ids=["names_a_reader", "reader_is_null", "no_policy", "installed_reader_off", "not_an_object"])
def test_no_second_derivation_where_the_question_does_not_arise(cert, live, monkeypatch):
    rec = _Recorder(result={"uncovered_policy": {}})
    monkeypatch.setattr(capsule_mod, "_certify_before_reader_v2", lambda: rec)
    assert _earlier_reader_certificate(cert, live, Path("d.md"), []) is None
    assert rec.calls == 0, "certify was run a second time"


def test_one_second_derivation_where_it_does(monkeypatch):
    rec = _Recorder(result={"uncovered_policy": {"schema": V1}, "counts": (1, 2)})
    monkeypatch.setattr(capsule_mod, "_certify_before_reader_v2", lambda: rec)
    got = _earlier_reader_certificate({"uncovered_policy": {}}, {"numeral_reader": {}}, Path("d.md"), [])
    assert rec.calls == 1
    assert got == {"uncovered_policy": {"schema": V1}, "counts": [1, 2]}, "as a reader of the file sees it"
    # a certificate with no policy block names no reader either: the row keys of the reader v2
    # are judged against the earlier certificate for it too
    assert _earlier_reader_certificate({}, {"numeral_reader": {}}, Path("d.md"), []) == got
    assert rec.calls == 2


def test_the_carried_fields_test_lets_a_certificate_lack_fields_and_nothing_else():
    earlier = {"verdict": "OATH-HELD", "counts": {"VERIFIED": 1}, "ledger": [{"line": 1, "value": 1.0}],
               "uncovered": 0, "verifier_sha256": "a" * 64, "receipt_binding": {"head": None}}
    older = {"verdict": "OATH-HELD", "counts": {"VERIFIED": 1}, "ledger": [{"line": 1, "value": 1.0}],
             "verifier_sha256": "b" * 64}
    assert _carries_the_earlier_readers_fields(older, earlier)
    assert not _issued_by_the_earlier_reader(older, earlier), "the stricter test wants every field"
    for change in (lambda c: c.__setitem__("verdict", "OATH-FAILED"),
                   lambda c: c["counts"].__setitem__("VERIFIED", 1.0),
                   lambda c: c["ledger"][0].__setitem__("ellipsis", True),
                   lambda c: c.__setitem__("numeral_reader", {})):
        edited = json.loads(json.dumps(older))
        change(edited)
        assert not _carries_the_earlier_readers_fields(edited, earlier)
    assert not _carries_the_earlier_readers_fields(older, None)
    assert not _carries_the_earlier_readers_fields([], earlier)


def test_a_second_derivation_that_cannot_run_gives_nothing_to_compare_with(monkeypatch):
    rec = _Recorder(error=RuntimeError("certify refused"))
    monkeypatch.setattr(capsule_mod, "_certify_before_reader_v2", lambda: rec)
    assert _earlier_reader_certificate({"uncovered_policy": {}}, {"numeral_reader": {}}, Path("d.md"), []) is None
    assert rec.calls == 1


def test_issued_by_the_earlier_reader_is_whole_and_keeps_types():
    earlier = {"verdict": "OATH-HELD", "uncovered": 0, "ledger": [{"line": 1, "token": "40"}],
               "uncovered_policy": {"schema": V1}, "verifier_sha256": "a" * 64,
               "receipt_binding": {"head": None}}
    same = json.loads(json.dumps(earlier))
    same["verifier_sha256"] = "b" * 64                      # the mint environment is left out
    same["receipt_binding"] = {"head": "c" * 40}
    assert _issued_by_the_earlier_reader(same, earlier)
    for edit in (lambda c: c.__setitem__("uncovered", 0.0),
                 lambda c: c.__setitem__("uncovered", False),
                 lambda c: c["ledger"][0].__setitem__("magnitude", {"suffix": "k"}),
                 lambda c: c["ledger"].append({"line": 2, "token": "7"}),
                 lambda c: c.__setitem__("numeral_reader", {}),
                 lambda c: c.pop("verdict"),
                 lambda c: c["uncovered_policy"].__setitem__("schema", V2)):
        other = json.loads(json.dumps(same))
        edit(other)
        assert not _issued_by_the_earlier_reader(other, earlier)
    assert not _issued_by_the_earlier_reader([], earlier)
    assert not _issued_by_the_earlier_reader(same, None)


# ── added after the second review ─────────────────────────────────

def test_every_field_of_the_certificate_is_in_the_whole_comparison():
    """`uncovered_items` and `uncovered_excluded_by_rule` among them: leaving a field out of the
    whole-certificate comparison passed every test of this file until this one."""
    cert = _payload_of(FIXTURE)["certificate"]
    earlier = json.loads(json.dumps(cert))
    assert _issued_by_the_earlier_reader(cert, earlier)
    assert {"uncovered_items", "uncovered_excluded_by_rule", "ledger", "verdict", "counts"} <= set(cert)
    for field in cert:
        other = json.loads(json.dumps(cert))
        other[field] = {"edited": True}
        mint_environment = field in ("verifier_sha256", "receipt_binding")
        assert _issued_by_the_earlier_reader(other, earlier) is mint_environment, field


def test_only_the_policy_block_is_excused():
    """The exception is for `uncovered_policy` and for no other field. Where the installed reader
    differs from the embedded certificate in another field (one more span excluded by rule, a band
    item under another reason), that field is a problem even though the embedded certificate is,
    whole, what the earlier reader issues."""
    payload = _payload_of(FIXTURE)
    cert = payload["certificate"]
    earlier = json.loads(json.dumps(cert))
    recs = {r["name"]: base64.b64decode(r["b64"]) for r in payload["receipts"]}
    live = json.loads(json.dumps(cert))
    live["verifier_sha256"] = "0" * 64                       # another certify.py than the issuer's
    live["uncovered_policy"] = dict(cert["uncovered_policy"], schema=V2)
    live["numeral_reader"] = {"schema": "styxx-oath/numerals/v2"}
    compare = capsule_mod._compare_certificate_v01

    base = compare(cert, live, payload, recs, earlier)
    assert base["problems"] == [], base["problems"]
    assert [c for c in base["compared"] if c.startswith("uncovered_policy (")]

    def one_more_excluded(v):
        return v + [{"line": 9, "col": 1, "token": "7", "counted": False, "reason": "year", "context": "x"}]

    def another_reason(v):
        return [dict(v[0], reason="unexplained")] + v[1:]

    assert cert["uncovered_items"], "the fixture has one band item"
    for field, change in (("uncovered_excluded_by_rule", one_more_excluded), ("uncovered_items", another_reason)):
        moved = json.loads(json.dumps(live))
        moved[field] = change(moved[field])
        rep = compare(cert, moved, payload, recs, earlier)
        assert any(p.startswith(f"{field} not reproduced") for p in rep["problems"]), (field, rep["problems"])
    # and with no earlier certificate to compare with, the policy block is a problem again
    rep = compare(cert, live, payload, recs, None)
    assert any(p.startswith("uncovered_policy not reproduced") for p in rep["problems"])


def test_the_second_derivation_reads_the_receipts_in_the_order_the_certificate_lists(tmp_path, monkeypatch):
    """A value found in two receipts binds to the one read earlier, so the order the certifier read
    them in is part of what the certificate says. Here z.json is read before a.json; a capsule
    stores its receipts sorted by name. Passing them sorted to the second derivation failed this
    intact capsule and no other test of the file."""
    work = tmp_path / "two"
    work.mkdir()
    doc = work / "d.md"
    doc.write_text("The run scored 0.75 accuracy over 40 items.\n", encoding="utf-8")
    z = work / "z.json"
    z.write_text(json.dumps({"eval": {"accuracy": 0.75}}), encoding="utf-8")
    a = work / "a.json"
    a.write_text(json.dumps({"other": {"accuracy": 0.75, "items": 40}}), encoding="utf-8")
    with monkeypatch.context() as m:
        m.setattr(certify, "V15_READER_V2", False)
        cp = work / "d.certificate.json"
        cp.write_text(json.dumps(certify.certify_doc(doc, [z, a])), encoding="utf-8")
        minted = create_capsule(doc, [z, a], cp, work / "d.capsule.html")
    issuer = _issuer_of_the_fixture()

    def redress(payload: dict) -> None:
        payload["certificate"]["verifier_sha256"] = issuer["sha256"]
        payload["verifier"] = issuer

    capsule = _forge(minted, tmp_path / "two_as_7_50_0.capsule.html", redress)
    cert = _payload_of(capsule)["certificate"]
    assert list(cert["receipts_sha256"]) == ["z.json", "a.json"]
    assert [r["name"] for r in _payload_of(capsule)["receipts"]] == ["a.json", "z.json"]
    assert any(str(row.get("receipt_ref", "")).startswith("z.json") for row in cert["ledger"])
    rep = verify_capsule(capsule)
    assert rep["ok"], _problems(rep)


# ── a row key the reader v2 began writing ─────────────────────────

def _reader_lines(rep: dict) -> list:
    return [a for a in rep["advisory"] if a.startswith("numeral_reader:")]


def _ellipsis_rows(cert: dict) -> list:
    return [r for r in cert["ledger"] if "ellipsis" in r]


def test_the_row_keys_of_the_reader_v2_are_these_three():
    assert capsule_mod._READER_V2_ROW_FIELDS == ("magnitude", "sign_form", "ellipsis")


def test_a_numeral_before_an_ellipsis_on_a_spec_line_verifies_around_both_pages(tmp_path, monkeypatch):
    """8.0.0 failed both of these on the policy block and printed a false line on the one that
    carries this styxx's page. With the policy block repaired, an earlier commit of this change
    left that line as the one failure of that capsule and passed the other: one certificate, two
    verdicts, by the page around it."""
    from styxx._capsule_page_v01_legacy import render_html_v01_legacy

    capsule = _mint_as_7_50_0(tmp_path, monkeypatch, ELLIPSIS_SPEC)
    payload = _payload_of(capsule)
    cert = payload["certificate"]
    # the case is the one meant: the installed reader writes the key on one row, the earlier
    # reader on none, and the row's status is the same under both
    doc = tmp_path / "live.md"
    doc.write_text(ELLIPSIS_SPEC, encoding="utf-8")
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps(RECEIPT), encoding="utf-8")
    live = certify.certify_doc(doc, [rp])
    assert len(_ellipsis_rows(live)) == 1 and _ellipsis_rows(cert) == []
    assert [r["status"] for r in live["ledger"]] == [r["status"] for r in cert["ledger"]]
    assert live["verdict"] == cert["verdict"]

    older_page = tmp_path / "older_page.capsule.html"
    older_page.write_text(render_html_v01_legacy(payload), encoding="utf-8")
    reports = []
    for path, page in ((capsule, "the page this styxx renders"), (older_page, "before 2026-10-05")):
        rep = verify_capsule(path)
        assert rep["ok"], _problems(rep)
        assert rep["mint_refusals"] == []
        assert any(c.startswith("the page (") and page in c for c in rep["compared"]), rep["compared"]
        lines = [n for n in rep["not_checked"] if "ledger[].ellipsis" in n]
        assert len(lines) == 1 and "absent from 1 of 3 row(s)" in lines[0] and "reader off" in lines[0]
        # not "whole": the issuer hash and the binding are the minter's and differ here
        assert "apart from the two its minter states" in lines[0] and "whole" not in lines[0]
        assert any("NOT CHECKED" in c for c in rep["compared"] if c.startswith("ledger ("))
        reports.append(lines)
    assert reports[0] == reports[1], "one certificate, one account of it, whichever page carries it"


def test_the_reader_advisory_names_the_flag_and_says_nothing_is_above(tmp_path, monkeypatch):
    """Through 8.0.0 the advisory ended "A row the two readers disagree on is reported above" on
    every certificate that names no reader: on a verifying one nothing is reported, and on a
    failing one the rows come below. It says where a difference is listed and names the flag, on
    a capsule that verifies too (tests/test_certify_numerals_v2.py asks for the flag there)."""
    for text in (CLEAN, ELLIPSIS_SPEC):
        rep = verify_capsule(_mint_as_7_50_0(tmp_path / str(len(text)), monkeypatch, text))
        assert rep["ok"], _problems(rep)
        (line,) = _reader_lines(rep)
        assert "above" not in line and "as a problem or as NOT CHECKED" in line
        assert "V15_READER_V2 = False set on the module styxx.certify" in line


def test_every_difference_between_the_readers_is_a_problem_or_not_checked(tmp_path, monkeypatch):
    """An earlier commit of this change said every difference "is among the problems listed
    below". On a document with one row the reader v2 moves and one row it only adds a key to,
    the second is in no problem line: it is NOT CHECKED. The sentence says both now, and this
    checks it row by row and not by its words."""
    capsule = _mint_as_7_50_0(tmp_path, monkeypatch, ELLIPSIS_BOTH)
    cert = _payload_of(capsule)["certificate"]
    doc = tmp_path / "live.md"
    doc.write_text(ELLIPSIS_BOTH, encoding="utf-8")
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps(RECEIPT), encoding="utf-8")
    live = json.loads(json.dumps(certify.certify_doc(doc, [rp])))
    moved = [r for r, c in zip(live["ledger"], cert["ledger"]) if r["status"] != c["status"]]
    key_only = [r for r, c in zip(live["ledger"], cert["ledger"])
                if r["status"] == c["status"] and "ellipsis" in r]
    assert len(moved) == 1 and len(key_only) == 1, "the document is not the one meant"
    rep = verify_capsule(capsule)
    assert not rep["ok"]
    text = _problems(rep)
    assert f"line {moved[0]['line']} token" in text and "status" in text and FALSE_LINE not in text
    assert f"line {key_only[0]['line']} token" not in text, "the key-only row is not a problem"
    assert any("ledger[].ellipsis" in n and "absent from 2 of" in n for n in rep["not_checked"])
    (line,) = _reader_lines(rep)
    assert "as a problem or as NOT CHECKED" in line


def test_create_mints_from_a_certificate_that_lacks_only_such_a_key(tmp_path, monkeypatch):
    capsule = _mint_as_7_50_0(tmp_path, monkeypatch, ELLIPSIS_SPEC)
    payload = _payload_of(capsule)
    work = tmp_path / "again"
    work.mkdir()
    doc = work / payload["document"]["name"]
    doc.write_bytes(base64.b64decode(payload["document"]["b64"]))
    rp = work / payload["receipts"][0]["name"]
    rp.write_bytes(base64.b64decode(payload["receipts"][0]["b64"]))
    cp = work / "older.certificate.json"
    cp.write_text(json.dumps(payload["certificate"]), encoding="utf-8")
    out = create_capsule(doc, [rp], cp, work / "again.capsule.html")
    assert verify_capsule(out)["ok"]
    assert _ellipsis_rows(_payload_of(out)["certificate"]) == []


def test_a_numeral_before_an_ellipsis_on_a_plain_line_still_fails_with_its_row(tmp_path, monkeypatch):
    """Here the two readers disagree on what the row says: the earlier one VERIFIED it, the reader
    v2 abstains. The capsule fails as before, the row is named, the false line is not printed
    beside it, and the advisory says where the row is and how to verify under the earlier reader."""
    capsule = _mint_as_7_50_0(tmp_path, monkeypatch, ELLIPSIS_PLAIN)
    rep = verify_capsule(capsule)
    assert not rep["ok"]
    text = _problems(rep)
    assert "ledger divergence" in text and "status" in text
    assert FALSE_LINE not in text
    assert "uncovered_policy not reproduced" not in text
    (line,) = _reader_lines(rep)
    assert "listed below, as a problem or as NOT CHECKED" in line and "above" not in line
    assert "V15_READER_V2 = False set on the module styxx.certify" in line


@pytest.mark.parametrize("keep_reader_block, shown", [
    (False, "a policy block of schema " + V2),
    (True, "certificate.numeral_reader"),
], ids=["reader_block_and_key_removed", "key_removed"])
def test_a_reader_v2_certificate_with_the_row_key_removed_fails_around_either_page(tmp_path, keep_reader_block, shown):
    """What the allowance must not cover: a certificate of the reader v2 from which the row key
    was deleted, with or without its reader block. Its policy block is the reader v2's, and the
    reader that wrote that wrote the key. Through 8.0.0, and at the earlier commits of this
    change, it failed around the page create_capsule renders and verified around the older page
    with the key NOT CHECKED: the page decided. It fails as edited around both now."""
    from styxx._capsule_page_v01_legacy import render_html_v01_legacy

    minted = _mint(tmp_path / "now", ELLIPSIS_SPEC)
    assert len(_ellipsis_rows(_payload_of(minted)["certificate"])) == 1
    issuer = _issuer_of_the_fixture()

    def strip(payload: dict) -> None:
        cert = payload["certificate"]
        if not keep_reader_block:
            del cert["numeral_reader"]
        for name in ("ledger", "ungrounded", "abstained"):
            for row in cert[name]:
                row.pop("ellipsis", None)
        cert["verifier_sha256"] = issuer["sha256"]
        payload["verifier"] = issuer

    current = _forge(minted, tmp_path / "stripped.html", strip)
    older = tmp_path / "stripped_older_page.html"
    older.write_text(render_html_v01_legacy(_payload_of(current)), encoding="utf-8")
    for path in (current, older):
        rep = verify_capsule(path)
        assert not rep["ok"], path.name
        gone = [p_ for p_ in rep["problems"] if "certificate.ledger[].ellipsis is absent from 1 of 3 row(s)" in p_]
        assert len(gone) == 1 and shown in gone[0] and "so it was removed" in gone[0], _problems(rep)
        assert FALSE_LINE not in _problems(rep)
        assert rep["mint_refusals"] == []
        assert not any("ledger[].ellipsis" in n for n in rep["not_checked"])


def test_a_certificate_older_than_the_band_verifies_around_both_pages_too(tmp_path, monkeypatch):
    """7.49.0 and 7.50.0 also minted this page around certificates issued before the uncovered
    band (the certify of tag v7.47.0 writes no band and no binding). Such a certificate lacks
    fields, so it is not "whole" what the earlier reader issues, and the first form of the row-key
    rule left it failing with the one false line (a review minted 10 such capsules with the
    released code; this builds one by removing the later fields from an earlier-reader
    certificate, which is what such a certificate carries). What it does carry is the earlier
    reader's, so the key is NOT CHECKED and it verifies, around either page."""
    from styxx._capsule_page_v01_legacy import render_html_v01_legacy

    capsule = _mint_as_7_50_0(tmp_path, monkeypatch, ELLIPSIS_SPEC)
    assert "uncovered" not in _payload_of(capsule)["certificate"]["verdict"], "the document has a band"

    def older(payload: dict) -> None:
        for name in ("uncovered", "uncovered_items", "uncovered_excluded_by_rule", "uncovered_policy",
                     "receipt_binding"):
            del payload["certificate"][name]

    current = _forge(capsule, tmp_path / "pre_band.html", older)
    older_page = tmp_path / "pre_band_older_page.html"
    older_page.write_text(render_html_v01_legacy(_payload_of(current)), encoding="utf-8")
    for path in (current, older_page):
        rep = verify_capsule(path)
        assert rep["ok"], _problems(rep)
        assert rep["mint_refusals"] == []
        assert any("ledger[].ellipsis" in n and "reader off" in n for n in rep["not_checked"])
        assert any("certificate.uncovered_policy" in n for n in rep["not_checked"])
    # and one field of it edited: no allowance, and the page line says what is true
    def edited(payload: dict) -> None:
        older(payload)
        payload["certificate"]["oath"] = payload["certificate"]["oath"] + " "

    rep = verify_capsule(_forge(capsule, tmp_path / "pre_band_edited.html", edited))
    assert not rep["ok"]
    assert any("was not shown to be what the installed certify issues" in p_ for p_ in rep["problems"])
    assert FALSE_LINE not in "\n".join(p_ for p_ in rep["problems"] if "ellipsis" in p_)


def test_another_refusal_keeps_its_sentence(tmp_path):
    """The sentence stays where it is true: no styxx renders this page around a certificate that
    predates the uncovered band and the epistemics summary."""
    def older(payload: dict) -> None:
        cert = payload["certificate"]
        for name in ("epistemics_summary", "uncovered", "uncovered_items",
                     "uncovered_excluded_by_rule", "uncovered_policy", "receipt_binding"):
            cert.pop(name, None)
        cert["verdict"] = "OATH-HELD"

    rep = verify_capsule(_forge(FIXTURE, tmp_path / "older_cert.html", older))
    assert not rep["ok"]
    assert rep["mint_refusals"] and all(FALSE_LINE in p_ for p_ in rep["problems"]), _problems(rep)
    assert any("epistemics_summary" in p_ for p_ in rep["problems"])
    assert all(type(r) is str for r in rep["mint_refusals"]), "the report holds plain strings"


def test_a_certificate_minted_here_with_the_reader_off_and_naming_this_certify_still_fails(tmp_path, monkeypatch):
    """The verdict is not changed: a certificate that names the installed certify.py as its issuer
    and lacks what that file writes fails as edited, the row key among the fields named. The
    allowance is for a certificate another certify.py issued. One line is changed: the page line
    that 8.0.0 printed beside these (the false one) is gone."""
    with monkeypatch.context() as m:
        m.setattr(certify, "V15_READER_V2", False)
        minted = _mint(tmp_path / "off", ELLIPSIS_SPEC)
    rep = verify_capsule(minted)
    assert not rep["ok"]
    text = _problems(rep)
    assert "certificate.numeral_reader" in text and "a field was removed" in text
    assert "certificate.ledger[].ellipsis is absent from 1 of 3 row(s), but the certificate names" in text
    assert FALSE_LINE not in text and "was not shown to be" not in text
    assert rep["mint_refusals"] == []
