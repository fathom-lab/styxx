# -*- coding: utf-8 -*-
"""#196: a hostile capsule ends `capsule verify` in CAPSULE FAILS VERIFICATION, with a named
problem and exit 1, and `styxx.charon` in an UNRESOLVED or not-reproduced line. Never a traceback.

7.49.0 left the issue open. Its CHANGELOG named two classes that still ended verify in a traceback
(a stated `styxx_version` of thousands of digits, an `ungrounded` row that is a list), and the
issue names three more:

* a v0.1 payload whose spec says v0.2;
* a receipt whose bytes are not JSON, with its hash restated in the certificate;
* a receipt named with a character Windows refuses (r<1>.json): writing it raised OSError there,
  and on Linux the verifier wrote it and re-ran the certifier, so the verdict depended on the
  verifier's system.

A sweep of every single-field mutation of a v0.1, a v0.2 and a sworn payload found more on
origin/main (566f7769): a v0.2 binding that is not an object or a gate record JCS cannot
canonicalise (AttributeError, ValueError), a sworn receipt that is not an object (AttributeError),
a payload nested deeper than the JSON decoder recurses (RecursionError), and in charon a v0.2
binding value that is not text (TypeError). The sweep is below, and each class has its own test.
A name some system refuses is now refused on every system, and so is a pair of names one file
system holds as one file.
"""
from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path

import pytest

from styxx import capsule as C
from styxx import charon
from styxx._capsule_page_v01_legacy import render_html_v01_legacy
from styxx.capsule import (_BEGIN, _END, SPEC_V02, create_capsule, create_capsule_diffgate,
                           create_capsule_sworn, main, verify_capsule)
from styxx.certify import certify_doc

RECEIPT = {"eval": {"accuracy": 0.75, "items": 40}}
TEXT = ("The run scored 0.75 accuracy over 40 items. It was wrong on 12.\n"
        "Line two says 40 again.\n")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _no_repository(self, path):
    from styxx.receipt_binding import RepoUnavailable
    raise RepoUnavailable(f"{path} is not inside a git repository")


@pytest.fixture(scope="module")
def v01(tmp_path_factory):
    """Minted outside any repository wherever pytest keeps its temporary files, as on CI: inside
    one, bind_at_mint states a head and paths, which verify states rather than checks, so the
    sweep's mutations of them verified there and nowhere else."""
    from styxx import receipt_binding as rb
    d = tmp_path_factory.mktemp("v01")
    doc = d / "d.md"
    doc.write_text(TEXT, encoding="utf-8")
    rp = d / "r.json"
    rp.write_text(json.dumps(RECEIPT), encoding="utf-8")
    cp = d / "d.certificate.json"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rb.Repo, "__init__", _no_repository)
        cp.write_text(json.dumps(certify_doc(doc, [rp])), encoding="utf-8")
        return create_capsule(doc, [rp], cp, d / "d.capsule.html")


@pytest.fixture(scope="module")
def v02(tmp_path_factory):
    d = tmp_path_factory.mktemp("v02")
    sp = d / "s.md"
    sp.write_text("This change is small: 7 files changed in total across the codebase.\n",
                  encoding="utf-8")
    dp = d / "s.diff"
    dp.write_text("diff --git a/x.py b/x.py\nnew file mode 100644\n--- /dev/null\n+++ b/x.py\n"
                  "@@ -0,0 +1,1 @@\n+x = 1\n", encoding="utf-8")
    return create_capsule_diffgate(sp, dp, d / "s.capsule.html")


@pytest.fixture(scope="module")
def sworn(tmp_path_factory):
    from styxx.sworn import Manifest, issue_receipt, verify
    d = tmp_path_factory.mktemp("sworn")
    m = Manifest("pytest", "turn-1", "2026-09-01T00:00:00Z", rung="L1")
    m.add("r1", b'{"passed": 12}\n', "test_report", complete=True,
          captured_at="2026-09-01T00:00:00Z")
    mp = d / "m.manifest.json"
    m.write(mp)
    doc = d / "note.md"
    doc.write_bytes(b'# n\n\n<sworn r="r1#/passed" k="numeric">The battery passed 12 checks.'
                    b'</sworn>\n')
    core = verify(doc.read_bytes(), name=doc.name,
                  manifest=Manifest.from_dict(json.loads(mp.read_text(encoding="utf-8"))))
    rp = d / "r.sworn-receipt.json"
    rp.write_bytes((json.dumps(issue_receipt(core), indent=1) + "\n").encode("utf-8"))
    return create_capsule_sworn(doc, mp, rp, d / "note.capsule.html")


def _split(path: Path):
    html = path.read_text(encoding="utf-8")
    i = html.index(_BEGIN) + len(_BEGIN)
    j = html.index(_END, i)
    return html[:i], json.loads(html[i:j].replace("\\u003c", "<")), html[j:]


def _splice(src: Path, payload) -> str:
    head, _, tail = _split(src)
    return head + json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c") + tail


def _write(dst: Path, src: Path, edit, render=None) -> Path:
    """A forger's page: the payload edited, then rendered by `render`, or spliced back into the
    page it came in (ensure_ascii, so a lone surrogate travels as a JSON escape)."""
    payload = _split(src)[1]
    edit(payload)
    dst.write_text(render(payload) if render else _splice(src, payload), encoding="utf-8")
    return dst


def _command(path: Path, rep=None):
    """(exit code, output) of `capsule verify`, its stdout a strict UTF-8 stream; with `rep`, the
    command prints that report rather than computing it again."""
    out = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="strict")
    with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(out):
        if rep is not None:
            mp.setattr(C, "verify_capsule", lambda p: rep)
        code = main(["verify", str(path)])
    out.flush()
    return code, out.buffer.getvalue().decode("utf-8")


def _line(path: Path, repo: Path, rep=None) -> dict:
    """charon's line for the capsule, digested and encoded as ingest writes it; with `rep`, charon
    reads that report rather than computing it again."""
    with pytest.MonkeyPatch.context() as mp:
        if rep is not None:
            mp.setattr(C, "verify_capsule", lambda p: rep)
        line = charon.derive(path, repo)
    entry = charon.make_entry(line, 1, "0" * 64, timestamp="2026-10-07T00:00:00Z")
    json.dumps(entry, ensure_ascii=False).encode("utf-8")
    return line


def _fails(path: Path, needle: str, repo: Path):
    """The full contract: verify fails with a problem naming `needle`, the command exits 1 with
    the failure line, and charon gives an UNRESOLVED or not-reproduced line."""
    rep = verify_capsule(path)
    assert rep["ok"] is False
    assert any(needle in p for p in rep["problems"]), rep["problems"]
    code, out = _command(path)
    assert code == 1 and "CAPSULE FAILS VERIFICATION" in out, out[-600:]
    line = _line(path, repo)
    assert line["verdict"] == "UNRESOLVED" or line["reproduced"] is False, line
    return rep, line


# ---------------------------------------------------------------- the three classes #196 names

def test_a_v01_payload_whose_spec_says_v02_fails_by_name(v01, tmp_path):
    def relabel(p):
        p["spec"] = SPEC_V02
    _fails(_write(tmp_path / "relabel.capsule.html", v01, relabel), "payload.summary", tmp_path)


def test_a_receipt_that_is_not_json_with_its_hash_restated_fails_by_name(v01, tmp_path):
    bad = b"this receipt is not JSON {"

    def edit(p):
        p["receipts"][0]["b64"] = _b64(bad)
        c = p["certificate"]
        c["receipts_sha256"]["r.json"] = _sha(bad)
        row = c["receipt_binding"]["receipts"][0]
        row["raw_sha256"] = row["content_sha256"] = _sha(bad)
    for render in (None, C._render_html, render_html_v01_legacy):
        f = _write(tmp_path / "notjson.capsule.html", v01, edit, render)
        _, line = _fails(f, "the installed verifier could not re-run on the embedded bytes",
                         tmp_path)
        assert line["verdict"] == "UNRESOLVED"


@pytest.mark.parametrize("name", ["r<1>.json", "r>1.json", 'r"1".json', "r|1.json", "r?.json",
                                  "r*.json"])
def test_a_receipt_name_windows_refuses_is_refused_on_every_system(v01, tmp_path, name,
                                                                   monkeypatch):
    """Until #196 a POSIX verifier wrote r<1>.json and re-ran the certifier, and Windows failed
    to write it: one capsule, two verdicts. Now the name is refused before anything is written,
    by the same rule everywhere."""
    import tempfile
    t = tmp_path / "t"
    t.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(t))

    def named(p):
        p["receipts"][0]["name"] = name
        c = p["certificate"]
        c["receipts_sha256"] = {name: c["receipts_sha256"]["r.json"]}
        c["receipt_binding"]["receipts"][0]["name"] = name
    f = _write(tmp_path / "named.capsule.html", v01, named, C._render_html)
    rep, line = _fails(f, "which Windows refuses in a file name; it is refused on every system",
                       tmp_path)
    assert rep["live_verdict"] is None
    assert "unsafe_embedded_name" in line["counts"]["reason"]
    assert not list(t.rglob("*.json")), "a file was written"


def test_the_name_rule_is_the_same_on_every_system():
    """The rule is a function of the name alone: nothing in it asks which system runs it."""
    refused = ["r<1>.json", "a>b", 'a"b', "a|b", "a?b", "a*b", "\ud800.json", "x" * 256,
               "\u00e9" * 128, "\U0001F600" * 64,
               "ABCDEF~1.JSO", "abcdef~1.jso", "RECEIP~1.JSO", "ABCDEF~1", "AB12F3~1.JSO", "~"]
    held = ["r.json", "x" * 255, "\u00e9" * 127, "\U0001F600" * 63, "r(1).json", "r[1].json",
            "r#1.json", "caf\u00e9.json", "r~1.json", "notes~draft.md", "abcdefghi~.jso",
            "abcdefghijkl.json"]
    for n in refused:
        assert not C._bare_name(n) and C._portable_name_problem(n), n
    for n in held:
        assert C._bare_name(n) and C._portable_name_problem(n) is None, n


@pytest.mark.parametrize("names, needle", [
    (("\u00e9.json", "e\u0301.json"), "the same file on HFS+"),
    (("\u00c9.json", "e\u0301.json"), "the same file on HFS+"),
    (("r\u200c.json", "r.json"), "the same file on HFS+"),
    (("r\ufeff.json", "R.json"), "the same file on HFS+"),
    (("abcdefghijkl.json", "ABCDEF~1.JSO"), "has the form of an 8.3 short name"),
    (("R.json", "r.JSON"), "the same file on a case-insensitive file system")])
def test_two_names_one_file_system_holds_as_one_file_are_refused(v01, tmp_path, names,
                                                                 needle):
    """HFS+ stores names decomposed, so an NFC and an NFD spelling are one file there, as two
    spellings in different case are on NTFS: the later write would replace the earlier there and
    not on ext4. HFS+ also ignores some code points when it compares names (Apple TN1150), and
    NTFS opens a file under its 8.3 short name. The case-insensitive pair keeps the message 7.48.1
    gave it."""
    def two(p):
        a, b = names
        rb = base64.b64decode(p["receipts"][0]["b64"])
        p["receipts"] = [{"name": a, "b64": _b64(rb)}, {"name": b, "b64": _b64(rb)}]
        p["certificate"]["receipts_sha256"] = {a: _sha(rb), b: _sha(rb)}
    _fails(_write(tmp_path / "two.capsule.html", v01, two), needle, tmp_path)


def test_a_long_name_and_its_short_name_are_refused_on_every_system(tmp_path, monkeypatch):
    """Two receipts certified as two files (in two directories, or on a system that generates no
    8.3 short names), abcdefghijkl.json and ABCDEF~1.JSO. Where NTFS generates short names the
    verifier wrote them as one file, re-ran the certifier over the second receipt twice, and
    failed with 'verdict not reproduced', which a reader takes for tamper; elsewhere the capsule
    verified. Now verify, create and charon refuse the short-name form on every system, before
    anything is written."""
    import tempfile
    from styxx import receipt_binding as rb
    from styxx._version import __version__
    monkeypatch.setattr(rb.Repo, "__init__", _no_repository)
    long_, short = "abcdefghijkl.json", "ABCDEF~1.JSO"
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    doc = tmp_path / "d.md"
    doc.write_text(TEXT, encoding="utf-8")
    ra, rb_ = tmp_path / "a" / long_, tmp_path / "b" / short
    ra.write_text(json.dumps({"eval": {"accuracy": 0.75}}), encoding="utf-8")
    rb_.write_text(json.dumps({"eval": {"items": 40}}), encoding="utf-8")
    cert = certify_doc(doc, [ra, rb_])
    cp = tmp_path / "d.certificate.json"
    cp.write_text(json.dumps(cert), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        create_capsule(doc, [ra, rb_], cp, tmp_path / "d.capsule.html")
    assert "'ABCDEF~1.JSO' has the form of an 8.3 short name" in str(e.value)

    payload = {"spec": C.SPEC, "created": "2026-10-07T00:00:00Z",
               "document": {"name": "d.md", "b64": _b64(TEXT.encode("utf-8"))},
               "receipts": [{"name": n, "b64": _b64(p.read_bytes())}
                            for n, p in sorted([(long_, ra), (short, rb_)])],
               "certificate": cert,
               "verifier": {"sha256": cert.get("verifier_sha256"), "styxx_version": __version__,
                            "pip": f"styxx=={__version__}"}}
    t = tmp_path / "t"
    t.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(t))
    page = tmp_path / "x.capsule.html"
    page.write_text(C._render_html(payload), encoding="utf-8")
    rep, line = _fails(page, "'ABCDEF~1.JSO' has the form of an 8.3 short name", tmp_path)
    assert not any("not reproduced" in p for p in rep["problems"]), rep["problems"]
    assert "unsafe_embedded_name" in line["counts"]["reason"]
    assert not list(t.rglob("*")), "a file was written"


def test_create_refuses_a_name_windows_refuses_before_reading_anything(tmp_path):
    """No file of that name can exist on Windows, so create checks the name before anything else
    (none of the paths given exists, so reading any of them would raise). Until #196 a POSIX
    system minted and kept such a capsule, and a Windows verifier failed it."""
    with pytest.raises(SystemExit) as e:
        create_capsule(tmp_path / "d.md", [tmp_path / "r<1>.json"], tmp_path / "c.json",
                       tmp_path / "d.capsule.html")
    assert "'r<1>.json'" in str(e.value) and "Windows refuses" in str(e.value)
    assert "ledger schema" not in str(e.value)
    assert not (tmp_path / "d.capsule.html").exists()


@pytest.mark.parametrize("names, needle", [
    (("R.json", "r.json"), "one file on a case-insensitive file system"),
    (("\u00e9.json", "e\u0301.json"), "one file on HFS+"),
    (("r\u200c.json", "r.json"), "one file on HFS+"),
    (("d.md", "D.MD"), "one file on a case-insensitive file system")])
def test_create_refuses_a_pair_of_names_one_file_system_holds_as_one_before_reading_anything(
        tmp_path, names, needle):
    """Two receipts kept in two directories (or the document and a receipt) whose names verify
    refuses as a pair: until #196 create minted the capsule, and its own verify then removed it
    with a hint that blamed the ledger schema. Now create refuses first, naming the pair."""
    a, b = names
    doc = tmp_path / a if a == "d.md" else tmp_path / "d.md"
    receipts = [tmp_path / "x" / b] if a == "d.md" else [tmp_path / "x" / a, tmp_path / "y" / b]
    with pytest.raises(SystemExit) as e:
        create_capsule(doc, receipts, tmp_path / "c.json", tmp_path / "d.capsule.html")
    msg = str(e.value)
    assert needle in msg and repr(a) in msg and repr(b) in msg, msg
    assert "ledger schema" not in msg
    assert not (tmp_path / "d.capsule.html").exists()


@pytest.mark.parametrize("name", ["con.json", "nul.md", "x.json."])
def test_create_refuses_a_name_that_is_not_bare_before_reading_anything(tmp_path, name):
    """The 7.48.1 rule verify applies, checked at create too: a name verify refuses was found
    only after the mint, and its refusal blamed the ledger schema."""
    with pytest.raises(SystemExit) as e:
        create_capsule(tmp_path / "d.md", [tmp_path / "x" / name], tmp_path / "c.json",
                       tmp_path / "d.capsule.html")
    assert "is not a bare file name" in str(e.value) and "ledger schema" not in str(e.value)


def test_create_still_takes_one_receipt_given_twice(tmp_path):
    """A receipt path given twice is one name, not a pair, as before."""
    doc = tmp_path / "d.md"
    doc.write_text(TEXT, encoding="utf-8")
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps(RECEIPT), encoding="utf-8")
    cp = tmp_path / "d.certificate.json"
    with pytest.MonkeyPatch.context() as mp:
        from styxx import receipt_binding as rb
        mp.setattr(rb.Repo, "__init__", _no_repository)
        cp.write_text(json.dumps(certify_doc(doc, [rp])), encoding="utf-8")
        out = create_capsule(doc, [rp, rp], cp, tmp_path / "d.capsule.html")
        assert verify_capsule(out)["ok"]


# ---------------------------------------------------------------- the two 7.49.0 named

@pytest.mark.parametrize("digits", [4300, 5000])
def test_a_stated_version_of_thousands_of_digits_fails_by_name(v01, tmp_path, digits):
    """Python reads at most 4300 digits as an integer by default; the advisory's version
    comparison ended verify in a ValueError (7.49.0 CHANGELOG)."""
    v = "7." + "9" * digits

    def stated(p):
        p["verifier"]["styxx_version"], p["verifier"]["pip"] = v, f"styxx=={v}"
    for render in (C._render_html, render_html_v01_legacy):
        f = _write(tmp_path / "ver.capsule.html", v01, stated, render)
        _fails(f, "characters long", tmp_path)


def test_a_stated_version_of_64_characters_is_read(v01, tmp_path):
    v = "7." + "1" * 62

    def stated(p):
        p["verifier"]["styxx_version"], p["verifier"]["pip"] = v, f"styxx=={v}"
    rep = verify_capsule(_write(tmp_path / "v64.capsule.html", v01, stated, C._render_html))
    assert not any("characters long" in p for p in rep["problems"]), rep["problems"]


@pytest.mark.parametrize("rows", [[["a", 1]], [[["a"]]], ["xy"], [None], [{"line": 1}, ["z", 2]]])
def test_an_ungrounded_row_that_is_not_an_object_fails_by_name(v01, tmp_path, rows):
    """A list row ended verify in a TypeError (sorting its items, or hashing them)."""
    def edit(p):
        p["certificate"]["ungrounded"] = rows
    _fails(_write(tmp_path / "ung.capsule.html", v01, edit, C._render_html),
           "certificate.ungrounded is not the ledger's UNGROUNDED rows", tmp_path)


# ---------------------------------------------------------------- the sweep found these

@pytest.mark.parametrize("value", ["x", [["a"]], 7, float("nan")])
def test_a_v02_binding_that_is_not_an_object_fails_by_name(v02, tmp_path, value):
    def edit(p):
        p["binding"]["summary"] = value
    _fails(_write(tmp_path / "b.capsule.html", v02, edit), "summary bytes != binding.summary",
           tmp_path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), [float("-inf")]])
def test_a_v02_gate_record_jcs_cannot_canonicalise_fails_by_name(v02, tmp_path, value):
    def edit(p):
        p["gate"]["diffgate"] = value
    rep, line = _fails(_write(tmp_path / "g.capsule.html", v02, edit),
                       "the record cannot be canonicalised", tmp_path)
    assert line["verdict"] == "UNRESOLVED"


@pytest.mark.parametrize("which", ["summary", "diff"])
def test_v02_bytes_bound_to_their_hash_that_are_not_utf8_fail_by_name(v02, tmp_path, which):
    bad = b"\xff\xfe not text"

    def edit(p):
        p[which]["b64"] = _b64(bad)
        p["binding"][which]["value"] = _sha(bad)
    _fails(_write(tmp_path / "u.capsule.html", v02, edit), "is not UTF-8 text", tmp_path)


@pytest.mark.parametrize("verifier", [["x"], "x", {"pip": [1]}, {"pip": "\ud800"}])
def test_v02_instrument_skew_with_a_verifier_that_is_not_an_object_fails_by_name(v02, tmp_path,
                                                                               verifier):
    def edit(p):
        p["gate"]["diffgate"] = "v9"
        p["binding"]["gate"]["value"] = C._gate_binding_hash(p["gate"])
        p["verifier"] = verifier
    _fails(_write(tmp_path / "s.capsule.html", v02, edit), "INSTRUMENT SKEW", tmp_path)


@pytest.mark.parametrize("value", ["x", [1], 7])
def test_a_sworn_receipt_that_is_not_an_object_fails_by_name(sworn, tmp_path, value):
    def edit(p):
        p["receipt"] = value
    f = _write(tmp_path / "r.capsule.html", sworn, edit)
    rep = verify_capsule(f)
    assert rep["ok"] is False and any("payload.receipt is not an object" in p
                                      for p in rep["problems"]), rep["problems"]
    code, out = _command(f)
    assert code == 1 and "CAPSULE FAILS VERIFICATION" in out


@pytest.mark.parametrize("edit", ["counts_nan", "document_text", "name_list", "verifier_text"])
def test_a_sworn_receipt_edited_with_its_core_restated_fails_without_a_traceback(sworn, tmp_path,
                                                                               edit):
    def change(p):
        r = p["receipt"]
        if edit == "counts_nan":
            r["counts"] = float("nan")
        elif edit == "document_text":
            r["document"] = "x"
        elif edit == "name_list":
            r["document"]["name"] = [1]
        else:
            r["verifier"] = "x"
        try:
            p["core_sha256"] = C._sworn_core_sha256(r)
        except ValueError:
            pass
    f = _write(tmp_path / "c.capsule.html", sworn, change)
    rep = verify_capsule(f)
    assert isinstance(rep["problems"], list)
    needle = {"counts_nan": "the receipt cannot be canonicalised, so its verdict core has no digest",
              "document_text": "payload.receipt.document is not an object",
              "verifier_text": "payload.receipt.verifier is not an object"}.get(edit)
    if needle:
        # (name_list: the document name is what styxx.sworn re-derives the core under, so a core
        # restated over a name that is a list re-derives, and the page verifies here as it does
        # at origin/main)
        assert rep["ok"] is False and any(needle in p for p in rep["problems"]), rep
    code, out = _command(f)
    assert code == (0 if rep["ok"] else 1)


@pytest.mark.parametrize("value", ["x", [1], 7, "absent", {}])
def test_a_sworn_receipt_verifier_block_that_is_not_an_object_fails_by_name(sworn, tmp_path,
                                                                            value):
    """The verifier block is outside the verdict core, so a block read as absent gave only an
    INSTRUMENT SKEW advisory and verified: text, a list or a number there printed VERIFIED, exit
    0, where origin/main ended in a traceback. Now each fails by name. An empty block, or none,
    reads as it did at origin/main: an advisory that the build is not named."""
    def edit(p):
        if value == "absent":
            del p["receipt"]["verifier"]
        else:
            p["receipt"]["verifier"] = value
    f = _write(tmp_path / "v.capsule.html", sworn, edit)
    rep = verify_capsule(f)
    code, out = _command(f)
    assert any("INSTRUMENT SKEW: the receipt was issued by styxx.sworn None" in a
               for a in rep["advisory"]), rep
    if isinstance(value, dict) or value == "absent":
        assert rep["ok"] is True and code == 0, rep
    else:
        assert rep["ok"] is False and code == 1 and "CAPSULE FAILS VERIFICATION" in out
        assert any("payload.receipt.verifier is not an object: styxx.sworn writes its build "
                   "there" in p for p in rep["problems"]), rep["problems"]


@pytest.mark.parametrize("depth", [65, 900, 100000])
def test_a_payload_nested_too_deeply_fails_at_parse(v01, tmp_path, depth):
    """100000 levels ended the decoder in a RecursionError; a few hundred passed the decoder and
    could end the comparison, JCS or printing in one."""
    head, payload, tail = _split(v01)
    body = json.dumps(payload).replace('"certificate": {',
                                       '"certificate": {"x": ' + "[" * depth + "]" * depth + ", ",
                                       1)
    f = tmp_path / "deep.capsule.html"
    f.write_text(head + body + tail, encoding="utf-8")
    rep = verify_capsule(f)
    assert rep["ok"] is False and rep["stage"] == "parse", rep
    code, out = _command(f)
    assert code == 1 and "CAPSULE FAILS VERIFICATION" in out
    assert _line(f, tmp_path)["verdict"] == "UNRESOLVED"


def test_a_lone_surrogate_is_printed_as_an_escape(v01, tmp_path):
    """JSON can carry a lone surrogate and UTF-8 cannot: printing it ended the command in a
    UnicodeEncodeError, and charon's line could be derived and then not written."""
    def edit(p):
        p["certificate"]["verdict"] = "OATH-HELD\ud800"
    f = _write(tmp_path / "sur.capsule.html", v01, edit)
    _, line = _fails(f, "verdict not reproduced", tmp_path)
    assert "\\ud800" in line["counts"]["recorded_verdict"]
    code, out = _command(f)
    assert "\\ud800" in out


def test_a_name_the_terminal_cannot_encode_is_printed_as_an_escape(v01, tmp_path, monkeypatch):
    """A document name in another script, on a stdout whose encoding cannot write it (a Windows
    pipe is cp1252), ended the command in a UnicodeEncodeError."""
    import sys
    out = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", out)
    assert main(["verify", str(v01)]) == 0
    C._say("document \u4e2d\u6587.md")
    out.flush()
    assert b"document \\u4e2d\\u6587.md" in out.buffer.getvalue()


# ---------------------------------------------------------------- what the installed code raises

def test_charon_names_an_exception_verify_raises(v01, tmp_path, monkeypatch):
    """verify answers hostile bytes with problems; should it ever raise, charon's line is
    UNRESOLVED and names the exception's class instead of ending derive."""
    def boom(path):
        raise ValueError("raised inside verify")
    monkeypatch.setattr(C, "verify_capsule", boom)
    line = _line(v01, v01.parent)
    assert line["verdict"] == "UNRESOLVED"
    assert line["counts"]["reason"] == "capsule_verify_raised: ValueError"


@pytest.mark.parametrize("exc", [ValueError, RecursionError, OSError])
def test_a_v02_gate_that_raises_on_the_embedded_text_fails_by_name(v02, tmp_path, monkeypatch,
                                                                   exc):
    import importlib
    diffgate_mod = importlib.import_module("styxx.diffgate")

    def boom(*a, **k):
        raise exc("raised inside the gate")
    monkeypatch.setattr(diffgate_mod, "gate_diff_text", boom)
    rep = verify_capsule(v02)
    assert rep["ok"] is False
    assert rep["problems"] == [f"the installed diffgate could not re-run on the embedded text "
                               f"({exc.__name__}: raised inside the gate)"]
    code, out = _command(v02)
    assert code == 1 and "CAPSULE FAILS VERIFICATION" in out
    line = _line(v02, v02.parent)
    assert line["verdict"] == "UNRESOLVED" or line["reproduced"] is False, line


@pytest.mark.parametrize("exc", [ValueError, KeyError, SystemExit])
def test_a_sworn_manifest_that_does_not_load_fails_by_name(sworn, monkeypatch, exc):
    from styxx.sworn import Manifest

    def boom(d):
        raise exc("raised loading the manifest")
    monkeypatch.setattr(Manifest, "from_dict", staticmethod(boom))
    rep = verify_capsule(sworn)
    assert rep["ok"] is False
    assert any(p.startswith("the sealed manifest does not load: ") and "raised loading" in p
               for p in rep["problems"]), rep["problems"]
    code, out = _command(sworn)
    assert code == 1 and "CAPSULE FAILS VERIFICATION" in out


@pytest.mark.parametrize("exc", [ValueError, TypeError, SystemExit])
def test_sworn_bytes_the_installed_instrument_refuses_fail_by_name(sworn, monkeypatch, exc):
    import importlib
    sworn_mod = importlib.import_module("styxx.sworn")

    def boom(*a, **k):
        raise exc("raised inside styxx.sworn")
    monkeypatch.setattr(sworn_mod, "verify", boom)
    rep = verify_capsule(sworn)
    assert rep["ok"] is False
    assert any(p.startswith("the sealed bytes do not verify at the installed instrument: ")
               and "raised inside styxx.sworn" in p for p in rep["problems"]), rep["problems"]
    code, out = _command(sworn)
    assert code == 1 and "CAPSULE FAILS VERIFICATION" in out


@pytest.mark.parametrize("exc", [OSError, ValueError])
def test_v01_files_this_system_cannot_write_fail_by_name(v01, tmp_path, monkeypatch, exc):
    """A name the rule allows that the verifier's own system still cannot write (on Windows
    without long paths, a path over 260 characters): the problem says the limit is this
    system's, and nothing is re-run."""
    import importlib
    certify_mod = importlib.import_module("styxx.certify")
    ran = []
    monkeypatch.setattr(certify_mod, "certify_doc", lambda *a, **k: ran.append(a))

    def refuse(self, data):
        raise exc("this system refuses the write")
    monkeypatch.setattr(Path, "write_bytes", refuse)
    rep = verify_capsule(v01)
    assert rep["ok"] is False and not ran
    assert any(p.startswith("the embedded files could not be written under their names here "
                            f"({exc.__name__}: this system refuses the write)")
               and "verify the capsule where its files can be written" in p
               for p in rep["problems"]), rep["problems"]
    code, out = _command(v01)
    assert code == 1 and "CAPSULE FAILS VERIFICATION" in out
    line = _line(v01, v01.parent)
    assert line["verdict"] == "UNRESOLVED" or line["reproduced"] is False, line


@pytest.mark.parametrize("stated, recorded", [(7, None), (float("nan"), None),
                                              ({"\u00e9": 1}, None), (["7.48.1"], None),
                                              ("7.48.1", "7.48.1")])
def test_charon_records_a_stated_version_only_when_it_is_text(v01, tmp_path, stated, recorded):
    """A version that is not text could not be digested (NaN) or was refused (an object with a
    key outside ASCII); charon records it as null and still derives the line."""
    def edit(p):
        p["verifier"]["styxx_version"] = stated
    f = _write(tmp_path / "pin.capsule.html", v01, edit)
    line = _line(f, tmp_path)
    assert line["counts"]["pinned_verifier_version"] == recorded


# ---------------------------------------------------------------- the sweep

# Values put in place of one field of the payload at a time (and each field is also deleted, and
# each object given one field no minter writes): the kinds that ended verify, the command or
# charon in a traceback before #196, and their neighbours. By default each field takes
# SWEEP_PER_FIELD of them, a window that moves along SWEEP from one field to the next, so every
# value meets about a quarter of the fields: 1,848 mutations of the fixtures above.
# STYXX_CAPSULE_SWEEP=full puts every value in every field: 6,059 mutations. Both pass. Writing
# and reading a page is most of the time a mutation takes, and on Windows a scanner reads every
# page written, so the full sweep takes minutes there.
SWEEP = [None, True, 0, -1, 1.5, float("nan"), float("inf"), "", "x", "7." + "9" * 5000,
         "\ud800", "a<b>&c", [], {}, [None], [["a", 1]], ["a", 1], {"a": [1]},
         [[[[[[[[["deep"]]]]]]]]]]
# The single-field mutations of the v0.1 payload that verify, and why: a binding block with no
# note is what bind_at_mint writes in a repository with no commit, and its facts are stated by
# the minter. Every other mutation of the v0.1 payload around its current page is refused.
V01_VERIFIES = {("certificate", "receipt_binding", "note")}
SWEEP_PER_FIELD = (len(SWEEP) if os.environ.get("STYXX_CAPSULE_SWEEP") == "full" else 5)


def _paths(obj, pre=()):
    if pre:
        yield pre
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _paths(v, pre + (k,))
    elif isinstance(obj, list):
        for n, v in enumerate(obj):
            yield from _paths(v, pre + (n,))


def _mutants(base, per_field=None):
    per_field = per_field or SWEEP_PER_FIELD
    for i, path in enumerate(_paths(base)):
        parent = base
        for p in path[:-1]:
            parent = parent[p]
        cur = parent[path[-1]]
        for v in (SWEEP[(i * per_field + j) % len(SWEEP)] for j in range(per_field)):
            if not (type(v) is type(cur) and (v == cur or isinstance(v, float) and math.isnan(v)
                                              and math.isnan(cur))):
                yield path, f"{path} = {repr(v)[:24]}", ("set", v)
        yield path, f"{path} deleted", ("del", None)
        if isinstance(parent[path[-1]], dict):
            yield path, f"{path} + a field", ("add", None)


def _mutate(base, path, how):
    p = copy.deepcopy(base)
    parent = p
    for k in path[:-1]:
        parent = parent[k]
    op, v = how
    if op == "set":
        parent[path[-1]] = copy.deepcopy(v)
    elif op == "del":
        del parent[path[-1]]
    else:
        parent[path[-1]]["zz_not_written"] = 1
    return p


@pytest.mark.parametrize("which", ["v01", "v02", "sworn"])
def test_no_single_field_mutation_ends_in_a_traceback(which, request, tmp_path, monkeypatch):
    """Every single-field mutation of a minted payload, around the page this styxx renders for it
    (spliced into the minted page where that page cannot be rendered): verify returns a report,
    the command prints it on a strict UTF-8 stream and exits 0 exactly when the report is ok, and
    charon derives a line it can digest and write, UNRESOLVED or not reproduced whenever verify
    failed. Around the v0.1 page a mutation is refused unless V01_VERIFIES names it.

    Two things keep the sweep to seconds. The re-run certifier looks for a git repository around
    the temporary directory it writes in, and finds none; here it is told so without running git
    (the binding note is stated by the minter, never compared). And certify_doc, the diff gate and
    styxx.sworn's verify are functions of the bytes and names they are given, which most mutations
    leave alone:
    each runs once per distinct input, and its result (or its exception) is handed back for the
    rest. What the sweep tests is what verify and charon do with those results."""
    import importlib                # by module name: the package exports a function `certify`
    certify_mod = importlib.import_module("styxx.certify")
    diffgate_mod = importlib.import_module("styxx.diffgate")
    sworn_mod = importlib.import_module("styxx.sworn")
    rb = importlib.import_module("styxx.receipt_binding")
    monkeypatch.setattr(rb.Repo, "__init__", _no_repository)

    def once(fn, key_of):
        seen = {}

        def run(*a, **k):
            key = key_of(*a, **k)
            if key not in seen:
                try:
                    seen[key] = (True, fn(*a, **k))
                except (Exception, SystemExit) as e:  # noqa: BLE001 - handed back as raised
                    seen[key] = (False, e)
            done, value = seen[key]
            if not done:
                raise value
            return copy.deepcopy(value)
        return run
    monkeypatch.setattr(certify_mod, "certify_doc", once(
        certify_mod.certify_doc,
        lambda d, rps, *a, **k: (d.name, d.read_bytes(), tuple((p.name, p.read_bytes())
                                                               for p in rps), repr(a), repr(k))))
    monkeypatch.setattr(diffgate_mod, "gate_diff_text", once(
        diffgate_mod.gate_diff_text, lambda s, d, *a, **k: (s, d, repr(a), repr(k))))
    monkeypatch.setattr(sworn_mod, "verify", once(
        sworn_mod.verify,
        lambda doc=None, name="", manifest=None, commit=None, **k: (
            doc, repr(name), repr(commit), repr(k),
            None if manifest is None else manifest.digest_or_none())))
    src = request.getfixturevalue(which)
    base = _split(src)[1]
    render = {"v01": C._render_html, "v02": C._render_html_v02,
              "sworn": lambda p: C._render_html_sworn(p, C._sworn_verifier_js().decode("utf-8"))
              }[which]
    f = tmp_path / "m.capsule.html"
    escaped, verified, n = [], [], 0
    for path, label, how in _mutants(base):
        payload = _mutate(base, path, how)
        try:
            html = render(payload)
        except Exception:  # noqa: BLE001 - a payload the renderer cannot draw is spliced in
            html = _splice(src, payload)
        f.write_text(html, encoding="utf-8", errors="surrogatepass")
        n += 1
        try:
            rep = verify_capsule(f)
            assert isinstance(rep["problems"], list) and isinstance(rep["ok"], bool)
            code, out = _command(f, rep)
            assert code == (0 if rep["ok"] else 1), (code, rep["ok"])
            assert rep["ok"] or "CAPSULE FAILS VERIFICATION" in out
            line = _line(f, tmp_path, rep)
            assert rep["ok"] or line["verdict"] == "UNRESOLVED" or line["reproduced"] is False
        except (Exception, SystemExit) as e:  # noqa: BLE001 - every escape is collected
            escaped.append(f"{label}: {type(e).__name__}: {str(e)[:160]}")
            continue
        if which == "v01" and rep["ok"] and path not in V01_VERIFIES:
            verified.append(label)
    assert n > 200, n
    assert not escaped, f"{len(escaped)} of {n} mutations escaped: " + "; ".join(escaped[:8])
    assert not verified, f"{len(verified)} v0.1 mutations verified: " + "; ".join(verified[:8])
