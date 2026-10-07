# -*- coding: utf-8 -*-
"""styxx.capsule — the proof-carrying document (OATH Capsule v0.1).

A certificate proves a document against its receipts, but the proof lives in a repository.
The capsule makes it portable: one self-contained HTML file carrying the document's exact
bytes, every receipt's exact bytes, the certificate verbatim, and two layers of verification
the READER runs:

* Layer 1, any browser, offline: WebCrypto re-hashes every embedded byte against the
  certificate and paints every token with its epistemic band. Tamper-evidence in one second.
* Layer 2, one command: ``python -m styxx.capsule verify FILE`` re-runs the real verifier
  over the embedded bytes and compares the whole certificate with what it re-derives, type for
  type: the verdict string (its class only, for a certificate that predates the uncovered band),
  the counts, the coverage band, the epistemics summary and the ledger in both directions and in
  order, every field of every row. It checks that the page around the payload is the page a
  styxx renders for exactly that payload, so what a browser draws is what was compared. What it
  cannot re-derive from the bytes (when, and with which styxx, the capsule was minted; where the
  receipts were committed) it prints as stated by the minter, never as verified; a field an
  older certify did not write it prints as NOT CHECKED, by name. Reproducibility, not assertion.

Creation refuses to lie: a capsule cannot be minted unless every hash matches and the
certificate re-verifies live. What no layer proves — that receipts truthfully record
reality — is printed in the capsule's own footer, because a portable binding that implied
portable provenance would be the green-lamp half-truth this instrument exists to reject.

Spec: papers/closed-model-frontier/SPEC_oath_capsule_v01_2026_08_31.md
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import hashlib
import html as _html
import json
import re
import sys
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import List, Optional

# The v0.13 UNCOVERED band appends ", N uncovered" to a verdict string. That suffix is a
# COVERAGE report travelling in the headline, not a verdict change: `counts["UNGROUNDED"]` is
# untouched and no token's status moved (styxx.corpus_audit.verdict_class learned this on
# 2026-09-01, when bucketing on the whole string put 131 certificates in neither class).
#
# Layer 2 compares the verdict by CLASS only for a certificate that predates the band (it carries
# no `uncovered` field and nothing certify began writing later, and does not name the installed
# certify.py as its issuer, so its verdict could never carry the suffix), and says so as NOT
# CHECKED.
# Until 2026-10-05 it compared the class for EVERY certificate and never compared `uncovered`, so
# a certificate edited to drop the suffix and report 0 uncovered verified like a clean one while
# its document held a number nothing checked. A certificate that carries the band is compared
# string for string.
_UNCOVERED_SUFFIX = re.compile(r",\s*\d+\s+uncovered\s*$")


def _verdict_class(verdict) -> str:
    return _UNCOVERED_SUFFIX.sub("", str(verdict))

__all__ = ["create_capsule", "create_capsule_diffgate", "verify_capsule", "main"]

SPEC = "styxx-oath/capsule/v0.1"
SPEC_V02 = "styxx-oath/capsule/v0.2"
SPEC_SWORN = "styxx-oath/capsule/sworn/v0.1"
_PAYLOAD_ID = "oath-capsule"
_BEGIN = f'<script type="application/json" id="{_PAYLOAD_ID}">'
_END = "</script>"


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


# Re-running the certifier writes the embedded document and receipts into a temporary
# directory under the names the capsule gives them. The capsule chooses those names, so a name
# must be one that can only mean a file directly inside that directory, under POSIX and under
# Windows rules alike: no separator, no drive, no "." or "..", no control character, no trailing
# dot or space (Windows drops them, so two names could meet in one file), and no Windows device.
_WINDOWS_DEVICES = {"con", "prn", "aux", "nul", "conin$", "conout$"}


def _bare_name(name) -> bool:
    if not isinstance(name, str) or name in ("", ".", ".."):
        return False
    if any(c in name for c in "/\\:") or any(ord(c) < 32 for c in name):
        return False
    if name != name.rstrip(" ."):
        return False
    if PurePosixPath(name).name != name or PureWindowsPath(name).name != name:
        return False
    stem = name.split(".", 1)[0].rstrip(" ").lower()
    if stem in _WINDOWS_DEVICES or (len(stem) == 4 and stem[:3] in ("com", "lpt")
                                    and stem[3] in "0123456789¹²³"):
        return False
    return True


def _unsafe_names(payload: dict) -> List[str]:
    """Problems with the file names a v0.1 payload asks the verifier to write; empty when all are safe."""
    names = [(payload.get("document") or {}).get("name")]
    names += [(r or {}).get("name") for r in payload.get("receipts") or []]
    out = [f"embedded name {n!r} is not a bare file name; nothing was written and the "
           f"certifier was not re-run" for n in names if not _bare_name(n)]
    if not out:
        folded = [n.casefold() for n in names]
        if len(set(folded)) != len(folded):
            out.append("two embedded names are the same file on a case-insensitive file system; "
                       "nothing was written and the certifier was not re-run")
    return out


# ---------------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------------

def create_capsule(doc: Path, receipts: List[Path], cert: Path, out: Path) -> Path:
    """Mint a capsule — refusing, loudly, to mint one that lies."""
    from styxx.certify import certify_doc
    from styxx._version import __version__

    # The certificate hashes the document as certify_doc READ it — read_text with universal
    # newlines, re-encoded UTF-8 — so on a CRLF checkout the on-disk bytes are NOT what the
    # certificate attested. The capsule embeds the exact bytes the certificate hashed
    # (newline-canonical text bytes), which is what makes it byte-faithful across newline
    # conventions and lets both verification layers hash the embedded bytes directly.
    doc_bytes = doc.read_text(encoding="utf-8").encode("utf-8")
    cert_obj = json.loads(cert.read_text(encoding="utf-8"))

    # 1. the certificate must describe THESE bytes
    if _sha256(doc_bytes) != cert_obj.get("document_sha256"):
        raise SystemExit("REFUSED: document bytes do not match certificate.document_sha256")
    rec_map = {}
    for r in receipts:
        rb = r.read_bytes()
        want = (cert_obj.get("receipts_sha256") or {}).get(r.name)
        if want is None:
            raise SystemExit(f"REFUSED: certificate carries no hash for receipt {r.name!r}")
        if _sha256(rb) != want:
            raise SystemExit(f"REFUSED: receipt {r.name!r} bytes do not match the certificate")
        rec_map[r.name] = rb
    missing = set(cert_obj.get("receipts_sha256") or {}) - set(rec_map)
    if missing:
        raise SystemExit(f"REFUSED: certificate names receipts not provided: {sorted(missing)}")

    # 2. the certificate must be REPRODUCIBLE at the live verifier, right now
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / doc.name
        d.write_bytes(doc_bytes)
        rps = []
        for name, rb in rec_map.items():
            rp = Path(td) / name
            rp.write_bytes(rb)
            rps.append(rp)
        live = certify_doc(d, rps)
    if live["verdict"] != cert_obj["verdict"] or live["counts"] != cert_obj["counts"]:
        raise SystemExit(
            "REFUSED: certificate is not reproducible at the installed verifier "
            f"(live {live['verdict']} {live['counts']} vs stored "
            f"{cert_obj['verdict']} {cert_obj['counts']})")

    payload = {
        "spec": SPEC,
        "created": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "document": {"name": doc.name, "b64": _b64(doc_bytes)},
        "receipts": [{"name": n, "b64": _b64(b)} for n, b in sorted(rec_map.items())],
        "certificate": cert_obj,
        "verifier": {"sha256": cert_obj.get("verifier_sha256"),
                     "styxx_version": __version__,
                     "pip": f"styxx=={__version__}"},
    }
    html = _render_html(payload)
    out.write_text(html, encoding="utf-8")

    # A CAPSULE THAT CANNOT VERIFY MUST NOT EXIST.
    #
    # The gate above compares only `verdict` and `counts`, and that is strictly weaker
    # than what verify_capsule checks. On 2026-09-01 that gap minted two capsules of
    # June papers that failed verification on EVERY token: certificates issued before
    # the ledger gained `status`/`col`/`receipt_ref` embed a ledger the current verifier
    # re-derives as None, so `capsule verify` reported a divergence per token while
    # `capsule create` had reported success. Both were about to be sent to an external
    # reader as evidence the instrument works.
    #
    # Re-verifying what we just wrote is the only gate that cannot drift away from the
    # verifier, because it IS the verifier. On failure the file is removed rather than
    # left on disk — a broken capsule that exists will eventually be sent to someone.
    #
    # Verification passing is not the whole gate. Layer 2 prints a field an older certify did not
    # write as NOT CHECKED and still passes, so that capsules already minted keep verifying; a new
    # capsule is not minted from a certificate whose ledger rows lack a field the page draws its
    # bands from (col, status, epistemics), or whose verdict is not the installed verifier's
    # string. That keeps the 2026-09-01 refusal of pre-column certificates, which the earlier round
    # of the 2026-10-05 repair had let through, minting pages that painted other numbers.
    report = verify_capsule(out)
    why = "does not verify"
    if report.get("ok") and report.get("mint_refusals"):
        why = "verifies only with what its page draws from NOT CHECKED"
        report = dict(report, ok=False, problems=report["mint_refusals"])
    if not report.get("ok"):
        problems = report.get("problems") or [report.get("error", "unknown")]
        out.unlink(missing_ok=True)
        raise SystemExit(
            f"REFUSED: the minted capsule {why}, so it was not kept.\n"
            + "\n".join(f"  - {p}" for p in problems[:6])
            + (f"\n  ... and {len(problems) - 6} more" if len(problems) > 6 else "")
            + "\n\nIf the ledger's rows lack fields, or it diverges on every token, this "
              "certificate predates the current ledger schema. Re-certify the document "
              "(the verdict is expected to be unchanged; a re-issue is a new commit and the "
              "drift is tracked), then mint again.")
    return out


# ---------------------------------------------------------------------------------
# verify (layer 2 — the real instrument, re-run)
# ---------------------------------------------------------------------------------

def verify_capsule(path: Path) -> dict:
    html = path.read_text(encoding="utf-8")
    try:
        i = html.index(_BEGIN) + len(_BEGIN)
        j = html.index(_END, i)
        payload = json.loads(html[i:j])
    except (ValueError, json.JSONDecodeError) as e:
        return {"ok": False, "stage": "parse", "error": f"no capsule payload: {e}",
                "problems": [f"no capsule payload: {e}"]}
    if not isinstance(payload, dict):
        return {"ok": False, "stage": "parse", "problems": ["the capsule payload is not an object"]}

    # spec dispatch; the v0.1 path compares the whole certificate and the page since 2026-10-05
    spec = payload.get("spec")
    if spec == SPEC_V02:
        return _verify_capsule_v02(html, payload)
    if spec == SPEC_SWORN:
        return _verify_capsule_sworn(html, payload)
    if spec != SPEC:
        return {"ok": False, "stage": "spec",
                "problems": [f"unknown capsule spec {spec!r} — this verifier knows "
                             f"{SPEC}, {SPEC_V02} and {SPEC_SWORN}"]}
    return _verify_capsule_v01(html, payload)


def _verify_capsule_v01(html: str, payload: dict) -> dict:
    from styxx.certify import certify_doc

    problems: List[str] = []
    advisory: List[str] = []
    cmp: dict = {"not_checked": [], "stated": [], "compared": [], "mint_refusals": []}
    live = None
    doc = payload.get("document") if isinstance(payload.get("document"), dict) else {}
    cert = payload.get("certificate") if isinstance(payload.get("certificate"), dict) else {}

    def report() -> dict:
        return {"ok": not problems, "problems": problems, "advisory": advisory,
                "not_checked": cmp["not_checked"], "stated": cmp["stated"],
                "compared": cmp["compared"], "mint_refusals": cmp["mint_refusals"],
                "verdict": cert.get("verdict"), "counts": cert.get("counts"),
                "live_verdict": None if live is None else live.get("verdict"),
                "document": doc.get("name"), "spec": payload.get("spec"),
                "reproduced_at": None if live is None else "installed verifier"}

    # the payload's own shape: what no minter writes fails before anything is decoded or written
    shape = _payload_problems_v01(payload)
    if shape:
        problems.extend(shape)
        return report()
    doc_bytes = base64.b64decode(doc["b64"])
    if _sha256(doc_bytes) != cert.get("document_sha256"):
        problems.append("document bytes != certificate.document_sha256")
    rsha = cert.get("receipts_sha256") if isinstance(cert.get("receipts_sha256"), dict) else {}
    recs = {}
    for r in payload["receipts"]:
        rb = base64.b64decode(r["b64"])
        recs[r["name"]] = rb
        want = rsha.get(r["name"])
        if _sha256(rb) != want:
            problems.append(f"receipt {r['name']!r} bytes != certificate hash")
    problems.extend(_unsafe_names(payload))

    if not problems:
        # The receipts are handed to the certifier in the order the certificate lists them,
        # which is the order its certifier read them in: when a value appears in more than one
        # receipt, `receipt_ref` names the earliest read, so the order is part of what the
        # certificate says. The capsule stores the receipts sorted by name, so their order
        # there is not it.
        listed = [n for n in rsha if n in recs]
        order = listed + sorted(n for n in recs if n not in listed)
        with tempfile.TemporaryDirectory() as td:
            try:
                d = Path(td) / doc["name"]
                d.write_bytes(doc_bytes)
                rps = []
                for name in order:
                    rp = Path(td) / name
                    rp.write_bytes(recs[name])
                    rps.append(rp)
            except OSError as e:
                # a name this system cannot hold (on Windows, one with < or : in it)
                problems.append(f"the embedded files could not be written under their names "
                                f"here ({type(e).__name__}: {e}), so the verifier was not re-run")
            else:
                try:
                    live = certify_doc(d, rps)
                except Exception as e:  # noqa: BLE001 - bytes the certifier cannot read fail
                    problems.append(f"the installed verifier could not re-run on the embedded "
                                    f"bytes ({type(e).__name__}: {str(e)[:160]})")
    if live is not None:
        # the certificate as certify's own command writes it: through JSON, so a tuple is a list
        # and every number has the type a reader of the file sees
        live = json.loads(json.dumps(live, ensure_ascii=False))
        cmp = _compare_certificate_v01(cert, live, payload, recs)
        problems.extend(cmp["problems"])
        advisory.extend(cmp["advisory"])

    # The page around the payload is what a reader's browser runs. It must be the page some
    # styxx renders for exactly this payload, or what it shows was never checked by anything.
    _check_page_v01(html, payload, cert, live if not problems else None, doc_bytes, problems,
                    advisory, cmp["compared"])
    return report()


# ---------------------------------------------------------------------------------
# layer 2 of a v0.1 capsule: the payload, the whole certificate, and the page
# ---------------------------------------------------------------------------------
#
# Until 2026-10-05 layer 2 compared a chosen few fields (the verdict class, the counts, and the
# status of each embedded ledger row, one way), so a certificate edited anywhere else verified
# exactly like the genuine one while the page drew the edited values. A second round of that
# repair (after its review) closed what the earlier round left: a decoy payload the browser never
# reads, values re-typed (1.0 for 1, true for 1), fields nested inside the receipt binding or the
# payload, a free-text install line the page shows, a certificate whose band was deleted while
# fields certify wrote later stayed, and the page drawing a band where the certificate puts none.
#
# The rule: what the page shows from the payload, layer 2 re-derives and compares, or names as not
# checked. Every field certify_doc writes is a function of the document and receipt bytes at the
# installed verifier, so it is re-derived and compared, type for type, except these, which
# describe the minting environment and cannot be re-derived from the bytes: the hash of the
# certify.py that issued the certificate, and the receipt binding's repository facts (head, paths,
# blobs, committed flags). Those are printed as stated by the minter, never as verified, after
# checking they are a combination certify writes.
_CERT_MINT_FIELDS = ("verifier_sha256", "receipt_binding")
# Every certificate any styxx has issued carries these, and every ledger (and ungrounded) row
# carries the row fields: the earliest certify (9ed6f3b5, 2026-06-10) already wrote each of them.
# One missing was removed, whoever issued the certificate. Until review round 3 only five were
# required, so with the issuer's hash moved off the installed certify.py, deleting `status` from
# the accused rows was printed NOT CHECKED and verified, while both pages painted those numbers
# verified.
_CERT_REQUIRED = ("verdict", "counts", "ledger", "document_sha256", "receipts_sha256",
                  "verifier_sha256", "oath", "prereg", "document", "ungrounded", "abstained")
_ROW_REQUIRED = ("line", "token", "value", "decimals", "context", "status", "receipt_ref")
# what every certify writes into verifier_sha256: the sha-256 of its own certify.py, in hex
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
# Row lists, compared row by row in both directions and in order, every field of every row.
_CERT_ROW_LISTS = ("ledger", "ungrounded", "abstained")
# The fields certify began writing after its earliest certificates, in the order it began writing
# them (`git log -S` on styxx/certify.py). A certificate carrying one of them was issued by a
# certify that wrote every one listed before it, so an earlier one missing beside it was removed.
# Measured on 2026-10-06 over the 223 certificates committed here (213 files and 10 capsules):
# every one of them obeys this order.
_CERT_GENERATIONS = (
    ("2026-08-24", ("ledger[].col",)),
    ("2026-08-30", ("ledger[].epistemics",)),
    ("2026-08-30", ("epistemics_summary",)),     # 26 minutes after the per-row field
    ("2026-09-01", ("uncovered", "uncovered_items", "uncovered_excluded_by_rule",
                    "uncovered_policy")),
    ("2026-09-05", ("receipt_binding",)),
)
# The receipt binding block and its rows, as styxx.receipt_binding.bind_at_mint writes them (and
# certify's own fallback when binding fails). `note` is the only optional key.
_BINDING_FIELDS = ("schema", "content_rule", "head", "all_receipts_committed", "receipts")
_BINDING_ROW_FIELDS = ("name", "path", "raw_sha256", "content_sha256", "blob", "committed")
_BINDING_BYTE_FIELDS = ("raw_sha256", "content_sha256")
_GIT_ID = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_PAYLOAD_FIELDS = ("spec", "created", "document", "receipts", "certificate", "verifier")
_PAYLOAD_FILE_FIELDS = ("name", "b64")
_PAYLOAD_VERIFIER_FIELDS = ("sha256", "styxx_version", "pip")
# what create_capsule writes into `created` and `verifier.styxx_version`
_CREATED = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_VERSION = re.compile(r"\d+(?:\.\d+){1,3}(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?")


def _same(a, b) -> bool:
    """JSON equality that keeps types: true is not 1, 1.0 is not 1, and key order is ignored.

    Python's == says 1 == 1.0 == True, so until 2026-10-05 a certificate whose counts were
    re-typed (30.0, false) or whose row said `line: true` compared equal, while the page drew the
    re-typed values (a row on line `true` is drawn on no line at all)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and a != a:
        return b != b
    return a == b


def _short(v, n: int = 200) -> str:
    s = repr(v) if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True)
    return s if len(s) <= n else f"{s[:n]}... ({len(s)} chars)"


def _canonical_b64(s) -> Optional[bytes]:
    """The bytes of a base64 string create_capsule could have written, else None. Python's
    decoder skips characters outside the alphabet and a browser's atob throws on them, so a
    string that is not canonical would be read differently by the two layers."""
    if not isinstance(s, str):
        return None
    try:
        b = base64.b64decode(s, validate=True)
    except (ValueError, TypeError):
        return None
    return b if _b64(b) == s else None


def _payload_problems_v01(payload: dict) -> List[str]:
    """What no minter writes into a v0.1 payload. A field at any level that create_capsule does not
    write fails: until 2026-10-05 only the top level was looked at, so a note on the document or a
    provenance claim on a receipt travelled in the capsule unremarked."""
    out: List[str] = []
    for k in payload:
        if k not in _PAYLOAD_FIELDS:
            out.append(f"payload.{k}: a field create_capsule does not write")
    for k in _PAYLOAD_FIELDS:
        if k not in payload:
            out.append(f"payload.{k} is missing: create_capsule writes it")
    files = [("payload.document", payload.get("document"))]
    recs = payload.get("receipts")
    if not isinstance(recs, list):
        out.append("payload.receipts is not a list")
    else:
        files += [(f"payload.receipts[{n}]", r) for n, r in enumerate(recs)]
        names = [r.get("name") for r in recs if isinstance(r, dict)]
        if len(set(map(str, names))) != len(names):
            out.append(f"payload.receipts names a receipt more than once: {names}")
    for where, f in files:
        if not isinstance(f, dict):
            out.append(f"{where} is not an object")
            continue
        for k in f:
            if k not in _PAYLOAD_FILE_FIELDS:
                out.append(f"{where}.{k}: a field create_capsule does not write")
        if not isinstance(f.get("name"), str):
            out.append(f"{where}.name is not a string")
        if _canonical_b64(f.get("b64")) is None:
            out.append(f"{where}.b64 is not the base64 create_capsule writes")
    if not isinstance(payload.get("certificate"), dict):
        out.append("payload.certificate is not an object")
    ver = payload.get("verifier")
    if not isinstance(ver, dict):
        out.append("payload.verifier is not an object")
    else:
        for k in ver:
            if k not in _PAYLOAD_VERIFIER_FIELDS:
                out.append(f"payload.verifier.{k}: a field create_capsule does not write")
        v = ver.get("styxx_version")
        if not (isinstance(v, str) and _VERSION.fullmatch(v)):
            out.append(f"payload.verifier.styxx_version {_short(v)} is not a styxx version")
        # The page prints this line as the command that installs layer 2. create_capsule writes
        # exactly "styxx==" and the version; anything else would have the page tell its reader to
        # install some other package to check it.
        elif ver.get("pip") != f"styxx=={v}":
            out.append(f"payload.verifier.pip {_short(ver.get('pip'))} is not 'styxx=={v}', the "
                       f"install line create_capsule writes and the page shows")
    c = payload.get("created")
    if not (isinstance(c, str) and _CREATED.fullmatch(c)):
        out.append(f"payload.created {_short(c)} is not the UTC time create_capsule writes")
    return out


def _row_key(e: dict) -> tuple:
    # typed, so a row on line `true` or `1.0` is not the row on line 1
    return (json.dumps(e.get("line")), json.dumps(e.get("token"), ensure_ascii=False))


def _at(e: dict) -> str:
    return f"line {json.dumps(e.get('line'))} token {e.get('token')!r}"


def _rows_by_key(rows: list) -> dict:
    """Rows keyed (line, token, k), typed: the k-th row with that line and token, in ledger order.

    `col` is deliberately not in the key, so a certificate whose rows lack a field still aligns
    row for row and that field is reported missing rather than every row being reported lost."""
    seen: dict = {}
    out: dict = {}
    for e in rows:
        base = _row_key(e)
        k = seen.get(base, 0)
        seen[base] = k + 1
        out[base + (k,)] = e
    return out


def _compare_rows(name: str, stored, fresh: list, problems: List[str], fields: dict) -> None:
    """Row by row, both ways and in order. `fields` gathers, for each field the installed verifier
    writes in a matched row, [rows lacking it, rows carrying it, rows in the list]."""
    if not isinstance(stored, list) or not all(isinstance(e, dict) for e in stored):
        problems.append(f"{name} not reproduced: the embedded {name} is not a list of rows")
        return
    s, f = _rows_by_key(stored), _rows_by_key(fresh)
    for k, e in s.items():
        if k not in f:
            problems.append(f"{name} row not reproduced: {_at(e)} (embedded "
                            f"{e.get('status', 'row')}) is not a row at the installed verifier")
            continue
        le = f[k]
        for fld, lv in le.items():
            slot = fields.setdefault(f"{name}[].{fld}", [0, 0, len(s)])
            slot[0 if fld not in e else 1] += 1
            if fld not in e:
                continue
            if not _same(e[fld], lv):
                problems.append(f"{name} divergence at {_at(e)}: {fld} embedded "
                                f"{_short(e[fld])} vs live {_short(lv)}")
        for fld in e:
            if fld not in le:
                problems.append(f"{name} divergence at {_at(e)}: the row carries {fld!r}, which "
                                f"the installed verifier does not write")
    for k, le in f.items():
        if k not in s:
            problems.append(f"{name} omits a row the installed verifier finds: {_at(le)} (live "
                            f"{le.get('status', 'row')})")
    if set(s) == set(f) and list(s) != list(f):
        problems.append(f"{name} rows are not in the order the installed verifier writes them "
                        f"(document order)")


def _presence(cert: dict, field: str) -> tuple:
    """(carried in full, carried at all) for a generation field; a row field is carried in full
    when every ledger row carries it."""
    if field.startswith("ledger[]."):
        k = field[len("ledger[]."):]
        ledger = cert.get("ledger")
        rows = [e for e in ledger if isinstance(e, dict)] if isinstance(ledger, list) else []
        n = sum(k in e for e in rows)
        return n == len(rows), n > 0
    return field in cert, field in cert


def _generation_problems(cert: dict) -> dict:
    """{field: problem} for every dated field missing beside one certify began writing later."""
    shown = [g for g, (_, fields) in enumerate(_CERT_GENERATIONS)
             if any(_presence(cert, f)[1] for f in fields)]
    if not shown:
        return {}
    top = max(shown)
    top_date, top_fields = _CERT_GENERATIONS[top]
    carried = next(f for f in top_fields if _presence(cert, f)[1])
    out = {}
    for g in range(top + 1):
        date, fields = _CERT_GENERATIONS[g]
        for f in fields:
            if _presence(cert, f)[0]:
                continue
            when = (f"in the same change that began writing {f}" if g == top else
                    f"on {top_date}, after it began writing {f} on {date}")
            out[f] = (f"certificate.{f} is absent{' from some rows' if '[]' in f else ''}, but "
                      f"the certificate carries certificate.{carried}, which certify began "
                      f"writing {when}: no certify issued that combination, so a field was "
                      f"removed")
    return out


def _compare_certificate_v01(cert: dict, live: dict, payload: dict, recs: dict) -> dict:
    """Compare an embedded certificate with the one certify_doc re-derives from the embedded
    bytes. A field both carry must be equal, type for type. A field the installed certify writes
    and the certificate lacks is NOT CHECKED, by name, with the value the installed verifier finds
    (an older certify did not write it), unless the certificate shows it is not that old: it
    carries a field certify began writing later, or names the installed certify.py as its issuer.
    Then the missing field fails. A field the certificate carries and the installed certify does
    not write cannot be reproduced, so it fails. Mint-environment fields are listed as stated."""
    problems: List[str] = []
    advisory: List[str] = []
    not_checked: List[str] = []
    stated: List[str] = []
    compared: List[str] = []
    # what create_capsule refuses to mint even though verify passes: a ledger missing fields the
    # page draws its bands from, or a verdict compared by class only
    mint_refusals: List[str] = []

    for k in _CERT_REQUIRED:
        if k not in cert:
            problems.append(f"certificate.{k} is missing: every certificate carries it")
    vs, lvs = cert.get("verifier_sha256"), live.get("verifier_sha256")
    same_issuer = isinstance(vs, str) and vs == lvs
    # The issuer's hash is stated, not checked, but its form is fixed: until review round 3 it
    # could be absent, null, an object or free text (terminal escapes included) and still verify.
    if "verifier_sha256" in cert and not (isinstance(vs, str) and _SHA256_HEX.fullmatch(vs)):
        problems.append(f"certificate.verifier_sha256 {_short(vs, 80)} is not the 64 lowercase "
                        f"hex digits every certify writes there")
    pv = (payload.get("verifier") or {}).get("sha256")
    if not _same(pv, vs):
        problems.append(f"payload.verifier.sha256 {_short(pv, 80)} is not "
                        f"certificate.verifier_sha256 {_short(vs, 80)}; create_capsule copies "
                        f"one into the other")
    removed = _generation_problems(cert)
    problems.extend(removed.values())

    def absent_field(name: str, what: str) -> None:
        if name in removed:
            return                  # already failed, with the field that dates the certificate
        if same_issuer:
            problems.append(f"certificate.{name} is {what}, but the certificate names the "
                            f"installed certify.py (verifier_sha256 {vs}) as its issuer, and that "
                            f"file writes it: a field was removed")
        else:
            not_checked.append(f"certificate.{name}: {what}; the installed certify writes it, a "
                               f"certificate from an older certify does not")

    # the verdict: the whole string, except for a certificate that predates the uncovered band
    ev, lv = cert.get("verdict"), live["verdict"]
    if "verdict" in cert:
        if _same(ev, lv):
            compared.append("verdict")
        elif ("uncovered" not in cert and not removed and not same_issuer
              and isinstance(ev, str) and _verdict_class(ev) == ev and _verdict_class(lv) == ev):
            compared.append("verdict class")
            mint_refusals.append(f"the verdict {ev!r} is not the installed verifier's {lv!r}")
            items = live.get("uncovered_items") or []
            where = "; ".join(f"line {u.get('line')} {u.get('token')!r} ({u.get('reason')})"
                              for u in items[:10]) + (" ..." if len(items) > 10 else "")
            not_checked.append(
                f"the verdict's coverage suffix: this certificate carries no `uncovered` field "
                f"(it predates the uncovered band), so its verdict {ev!r} was compared by class "
                f"only; the installed verifier's verdict is {lv!r}")
            advisory.append(
                f"verdict string moved without a class change: live {lv!r} vs embedded {ev!r}, a "
                f"coverage suffix the installed verifier appends to a certificate that predates "
                f"the band. The installed verifier finds {len(items)} numeric span(s) nothing "
                f"checked: {where}")
        else:
            problems.append(f"verdict not reproduced: live {_short(lv)} vs embedded {_short(ev)}")

    for k, lval in live.items():
        if k == "verdict" or k in _CERT_ROW_LISTS or k in _CERT_MINT_FIELDS:
            continue
        if k not in cert:
            if k not in _CERT_REQUIRED:
                absent_field(k, f"not carried (installed verifier: {_short(lval, 160)})")
        elif not _same(cert[k], lval):
            if isinstance(lval, list) and isinstance(cert[k], list):
                js = lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True)  # noqa: E731
                only_live = [x for x in lval if js(x) not in {js(y) for y in cert[k]}]
                only_cert = [x for x in cert[k] if js(x) not in {js(y) for y in lval}]
                problems.append(f"{k} not reproduced: {len(cert[k])} embedded vs {len(lval)} "
                                f"live; only live {_short(only_live)}; only embedded "
                                f"{_short(only_cert)}")
            else:
                problems.append(f"{k} not reproduced: live {_short(lval)} vs embedded "
                                f"{_short(cert[k])}")
        else:
            compared.append(k)
    for k in cert:
        if k not in live and k not in _CERT_MINT_FIELDS:
            problems.append(f"certificate.{k} cannot be reproduced: the installed certify does "
                            f"not write it")
    # The page draws its volunteered-share card from the summary. A certificate whose rows carry
    # epistemics and which lacks it exists (two are committed, from the 26 minutes between those
    # changes on 2026-08-30), so verify names it NOT CHECKED, but create does not mint from it.
    if "epistemics_summary" in live and "epistemics_summary" not in cert:
        mint_refusals.append("certificate.epistemics_summary is absent, and the page draws its "
                             "volunteered share from it")

    for name in _CERT_ROW_LISTS:
        if name not in live:
            continue
        if name not in cert:
            if name not in _CERT_REQUIRED:
                absent_field(name, "not carried")
            continue
        before = len(problems)
        fields: dict = {}
        _compare_rows(name, cert[name], live[name], problems, fields)
        unchecked = 0
        for fld, (n, carried, total) in fields.items():
            if not n or fld in removed:
                continue            # carried by every row, or already failed with its date
            if fld.split("[].", 1)[1] in _ROW_REQUIRED:
                problems.append(f"certificate.{fld} is absent from {n} of {total} row(s): every "
                                f"certify writes it in every row, so it was removed")
            elif carried:
                problems.append(f"certificate.{fld} is absent from {n} row(s) and carried by "
                                f"{carried}, where the installed verifier writes it in all of "
                                f"them: no certify writes it into some of those rows and not "
                                f"others, so it was removed")
            else:
                unchecked += 1
                absent_field(fld, f"absent from {n} of {total} row(s)")
                mint_refusals.append(f"certificate.{fld} is absent from {n} of {total} row(s), "
                                     f"and the page draws each band from its row")
        if len(problems) == before:
            # "every field" only when no field of these rows went unchecked (review round 3)
            compared.append(f"{name} ({len(live[name])} rows, both directions, in order, "
                            + ("every field)" if not unchecked else
                               f"every field it carries; {unchecked} field(s) NOT CHECKED, "
                               f"listed)"))

    # the minting environment: stated, never verified
    stated.append(f"certificate.verifier_sha256 {vs} (the certify.py that issued it; the "
                  f"installed certify.py is "
                  f"{'the same file' if same_issuer else lvs})")
    if "receipt_binding" not in cert:
        absent_field("receipt_binding", "not carried (receipt digests, and the repository head, "
                                        "paths and committed flags at mint)")
    else:
        _compare_binding(cert["receipt_binding"], live.get("receipt_binding") or {}, recs,
                         problems, not_checked, stated, compared)

    ver = payload.get("verifier") or {}
    stated.append(f"created {payload.get('created')} (when the capsule was minted)")
    stated.append(f"verifier.styxx_version {ver.get('styxx_version')}, pip "
                  f"{ver.get('pip')} (the styxx the minter ran)")
    sv = ver.get("styxx_version")
    if isinstance(sv, str) and _version_key(sv) and _version_key(sv) < _version_key(_LAYER2_FLOOR):
        advisory.append(f"the capsule states it was minted with styxx {sv}, below "
                        f"{_LAYER2_FLOOR}, the release whose layer 2 compares the whole "
                        f"certificate and the page: a styxx below it passes certificates this one "
                        f"fails, so check a capsule with styxx>={_LAYER2_FLOOR}, not with the "
                        f"version it states")
    return {"problems": problems, "advisory": advisory, "not_checked": not_checked,
            "stated": stated, "compared": compared, "mint_refusals": mint_refusals}


def _binding_path_problem(path, name) -> Optional[str]:
    """Why `path` is not one bind_at_mint writes for a receipt named `name`, else None.

    bind_at_mint writes the receipt's path relative to the repository root, in POSIX form
    (Repo.rel_or_none: os.path.relpath, as_posix, and None for anything outside the root), so it
    ends in the receipt's own name. That name can differ only in case, where a case-insensitive file
    system resolves the name to the case on disk."""
    if not isinstance(path, str):
        return "is not a string"
    parts = path.split("/")
    if (PurePosixPath(path).is_absolute() or PureWindowsPath(path).drive or "\\" in path
            or any(p in ("", ".", "..") for p in parts)):
        return "is not a relative POSIX path inside the repository"
    if any(ord(c) < 32 or 127 <= ord(c) < 160 for c in path):
        return "holds a control character"
    if not isinstance(name, str) or parts[-1].casefold() != name.casefold():
        return f"does not end in the receipt's name {_short(name)}"
    return None


def _compare_binding(rb, lrb: dict, recs: dict, problems: List[str], not_checked: List[str],
                     stated: List[str], compared: List[str]) -> None:
    """The receipt binding: its digests are functions of the receipt bytes and are compared; its
    repository facts are stated, after checking they are a combination bind_at_mint writes. One
    of those facts is also a function of the bytes and is compared: bind_at_mint marks a receipt
    committed only when the blob at head is the receipt's bytes (as they are, or with LF or CRLF
    line ends), and writes that blob. Until review round 3 only the blob's form was checked, so a
    certificate certified over edited bytes and given an honest mint's head, paths and blobs
    verified exactly like that mint."""
    from styxx.receipt_binding import git_blob_id
    if not isinstance(rb, dict):
        problems.append("certificate.receipt_binding is not an object")
        return
    before = len(problems)
    for k in rb:
        if k not in _BINDING_FIELDS and k != "note":
            problems.append(f"certificate.receipt_binding.{k}: a field certify does not write")
    for k in _BINDING_FIELDS:
        if k not in rb:
            problems.append(f"certificate.receipt_binding.{k} is missing: certify writes it in "
                            f"every binding block")
    for k in ("schema", "content_rule"):
        if k in rb and not _same(rb[k], lrb.get(k)):
            problems.append(f"receipt_binding.{k} not reproduced: live {_short(lrb.get(k))} "
                            f"vs embedded {_short(rb[k])}")
    rows = rb.get("receipts")
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        problems.append("receipt_binding.receipts not reproduced: not a list of receipts")
        return
    for r in rows:
        for k in r:
            if k not in _BINDING_ROW_FIELDS:
                problems.append(f"certificate.receipt_binding.receipts[{r.get('name')!r}].{k}: a "
                                f"field certify does not write")
        for k in _BINDING_ROW_FIELDS:
            if k not in r:
                problems.append(f"certificate.receipt_binding.receipts[{r.get('name')!r}].{k} is "
                                f"missing: certify writes it for every receipt")
    # what bind_at_mint can write: no path, blob or committed flag without a head; a blob exactly
    # when committed; all_receipts_committed exactly when every receipt is; and a note only in
    # one of three forms, each where it writes it ("no receipts" with no rows; "no repository at
    # mint: ..." with no head; certify's own "binding failed: ..." with neither)
    head, note, allc = rb.get("head"), rb.get("note"), rb.get("all_receipts_committed")
    if head is not None and not (isinstance(head, str) and _GIT_ID.fullmatch(head)):
        problems.append(f"receipt_binding.head {_short(head)} is not a commit id")
    if note is not None and not isinstance(note, str):
        problems.append("receipt_binding.note is not a string")
    elif note is None:
        if rows == []:
            problems.append("receipt_binding lists no receipts and carries no note; certify "
                            "writes the note 'no receipts' there")
    elif note.startswith("no repository at mint: "):
        if head is not None:
            problems.append("receipt_binding says there was no repository at mint and names a head")
    elif note.startswith("binding failed: "):
        if rows or head is not None:
            problems.append("receipt_binding.note says the binding failed at mint, and the block "
                            "names a head or receipts; certify's fallback carries neither")
    elif note == "no receipts":
        if rows:
            problems.append(f"receipt_binding.note says there were no receipts, and the block "
                            f"lists {len(rows)}")
    else:
        problems.append(f"receipt_binding.note {_short(note, 80)} is not a note certify writes")
    for r in rows:
        blob, committed, path = r.get("blob"), r.get("committed"), r.get("path")
        if (not isinstance(committed, bool)
                or (blob is not None and not (isinstance(blob, str) and _GIT_ID.fullmatch(blob)))
                or committed != (blob is not None)
                or (head is None and (path is not None or committed))):
            problems.append(f"receipt_binding row {r.get('name')!r} (path {_short(path)}, blob "
                            f"{_short(blob)}, committed {_short(committed)}, head {_short(head)}) "
                            f"is not a combination certify writes")
        why = None if path is None else _binding_path_problem(path, r.get("name"))
        if why:
            problems.append(f"receipt_binding path {_short(path)} of {r.get('name')!r} is not a "
                            f"repository path certify writes: it {why}")
        raw = recs.get(r.get("name")) if isinstance(r.get("name"), str) else None
        if committed is True and isinstance(blob, str) and raw is not None:
            lf = raw.replace(b"\r\n", b"\n")
            if blob not in {git_blob_id(raw), git_blob_id(lf),
                            git_blob_id(lf.replace(b"\n", b"\r\n"))}:
                problems.append(f"receipt_binding row {r.get('name')!r} says it was committed as "
                                f"blob {blob}, which is not the git blob of the embedded receipt's "
                                f"bytes (as they are, or with LF or CRLF line ends); certify "
                                f"marks a receipt committed only when the blob at head is those "
                                f"bytes")
    if not _same(allc, bool(rows) and all(r.get("committed") is True for r in rows)):
        problems.append(f"receipt_binding.all_receipts_committed {_short(allc)} does not follow "
                        f"from its rows")

    if isinstance(note, str) and note.startswith("binding failed:") and rows == []:
        # certify's own fallback (R7: a binding failure never blocks a certificate)
        not_checked.append(f"certificate.receipt_binding digests: the block says its binding "
                           f"failed at mint ({note!r}), so it carries none; the receipts' bytes "
                           f"are compared through receipts_sha256")
    else:
        names = [r.get("name") for r in rows]
        lnames = [r.get("name") for r in (lrb.get("receipts") or [])]
        if not _same(names, lnames):
            problems.append(f"receipt_binding.receipts not reproduced: names {names} vs the "
                            f"embedded receipts {lnames}")
        lrows = {r.get("name"): r for r in (lrb.get("receipts") or [])}
        for r in rows:
            lr = lrows.get(r.get("name")) or {}
            for k in _BINDING_BYTE_FIELDS:
                if k in r and not _same(r[k], lr.get(k)):
                    problems.append(f"receipt_binding {k} of {r.get('name')!r} not "
                                    f"reproduced: live {lr.get(k)} vs embedded {r[k]}")
        if len(problems) == before:
            compared.append("receipt_binding digests")
    # The head, the paths and the commits behind them cannot be checked from the bytes: a capsule
    # carries no repository. A blob is printed beside its flag; it names the receipt's own bytes.
    stated.append(
        f"certificate.receipt_binding: head {head}, all_receipts_committed {allc}; "
        + "; ".join(f"{r.get('name')} path {r.get('path')} committed {r.get('committed')}"
                    + (f" blob {r.get('blob')}" if r.get("blob") is not None else "")
                    for r in rows)
        + (f"; note {note!r}" if note else ""))


# The marks the page before 2026-10-05 read differently from certify: its private band markers,
# and every line break str.splitlines() honours beyond \n (it split only at \n and \r\n).
_PAGE_V01_LEGACY_MISREAD = re.compile("[\x01-\x03\r\x0b\x0c\x1c-\x1e\x85\u2028\u2029]")


def _check_page_v01(html: str, payload: dict, cert: dict, live: Optional[dict], doc_bytes,
                    problems: List[str], advisory: List[str], compared: List[str]) -> None:
    """The page must be what some styxx renders for exactly this payload.

    The payload is located by text; a browser locates it as an element, skipping comments. Until
    2026-10-05 a page carrying a second, decoy payload inside an HTML comment ahead of the real
    one verified the decoy while the browser drew the other, and the output was the genuine
    capsule's byte for byte. Re-rendering the payload and requiring the page to equal it closes
    every variant of that: an edited script, a decoy, a moved payload. Two renderers exist: this
    one, and the one every styxx used before 2026-10-05 (styxx._capsule_page_v01_legacy). For the
    older page, whose script reads documents differently from certify, the drawing is re-derived
    here (_legacy_page_problems)."""
    from styxx._capsule_page_v01_legacy import render_html_v01_legacy

    page = html[1:] if html.startswith("\ufeff") else html

    def renders(fn) -> bool:
        try:
            return fn(payload) == page
        except Exception:   # noqa: BLE001 - a payload a renderer cannot read is not its page
            return False

    if renders(_render_html):
        compared.append("the page (the page this styxx renders for this payload)")
        return
    if not renders(render_html_v01_legacy):
        problems.append("the page around the payload is not the page any styxx renders for this "
                        "payload: it was edited after minting (a script, a decoy payload, a "
                        "moved element), so what a browser shows was not checked")
        return
    if live is None:
        return           # the payload did not verify; its drawing is moot
    found = _legacy_page_problems(payload, cert, live, doc_bytes)
    if found:
        problems.extend(found)
        return
    compared.append("the page (the page styxx rendered before 2026-10-05 for this payload; it "
                    "draws every ledger row where the certificate puts it)")
    advisory.append("this capsule carries the page styxx minted before 2026-10-05: its badge "
                    "shows the certificate's verdict as fixed text before any check, and still "
                    "shows it when a byte is doctored or the script does not run. A capsule "
                    "minted with this styxx shows a verdict only after its hashes match.")


def _u16(s: str) -> str:
    """The string as a browser holds it: one character per UTF-16 code unit."""
    b = s.encode("utf-16-le", "surrogatepass")
    return "".join(chr(b[i] | (b[i + 1] << 8)) for i in range(0, len(b), 2))


def _band(e: dict) -> str:
    if e.get("status") == "UNGROUNDED":
        return "un"
    if e.get("status") == "ABSTAIN":
        return "ab"
    ep = e.get("epistemics")
    return "vo" if isinstance(ep, dict) and ep.get("obligated") else "vv"


def _legacy_page_problems(payload: dict, cert: dict, live: dict, doc_bytes: bytes) -> List[str]:
    """Where the page styxx minted before 2026-10-05 would draw this certificate wrong.

    Its script splits the document at \\n and \\r\\n only, while certify splits as str.splitlines
    (form feed, U+2028 and five more), so rows land on other lines; it reads `col` as a UTF-16
    index and falls back to the token's earliest occurrence on the line, so an astral character or a
    U+2212 minus moves a band to another number or drops it; it marks bands with U+0001 to U+0003,
    so those characters in the document draw bands no row gives; and it writes receipt names into
    the page unescaped. Each is re-derived here from the payload, and any one fails."""
    out: List[str] = []
    for r in payload["receipts"]:
        if any(c in r["name"] for c in "<&"):
            out.append(f"the page (minted before 2026-10-05) writes receipt names into its HTML "
                       f"unescaped, and {r['name']!r} holds markup (< or &), so the page would "
                       f"not show that name")
    # Its 'volunteered share' card reads epistemics_summary and, without one, counts every
    # verified number volunteered. Where the certificate's own rows say otherwise, the card
    # contradicts them (review round 2, f8: deleting the summary moved the card from 34% to 100%).
    if "epistemics_summary" not in cert:
        obligated = sum(1 for e in cert.get("ledger") or [] if isinstance(e, dict)
                        and e.get("status") == "VERIFIED" and isinstance(e.get("epistemics"), dict)
                        and e["epistemics"].get("obligated") is True)
        if obligated:
            out.append(f"the page (minted before 2026-10-05) draws its volunteered share from "
                       f"certificate.epistemics_summary, which this certificate lacks, so it shows "
                       f"every verified number volunteered while {obligated} of its verified "
                       f"ledger rows say obligated")
    text = doc_bytes.decode("utf-8")
    m = _PAGE_V01_LEGACY_MISREAD.search(text)
    if m or text.startswith("\ufeff"):
        at = m.start() if m else 0
        out.append(f"the page (minted before 2026-10-05) cannot draw this document as certify "
                   f"reads it: it holds U+{ord(text[at]):04X} on line "
                   f"{text.count(chr(10), 0, at) + 1}, which its script reads differently")
        return out
    lines = text.split("\n")
    emb, liv = _rows_by_key(cert.get("ledger") or []), _rows_by_key(live.get("ledger") or [])
    by_line: dict = {}
    for key, e in emb.items():
        by_line.setdefault(e.get("line"), []).append((e, liv[key]))
    for n, pairs in sorted(by_line.items()):
        line = lines[n - 1]
        drawn = _u16(line)
        for e in sorted((e for e, _ in pairs), key=lambda e: -(e.get("col") or 0)):
            t = _u16(str(e.get("token")))
            col = e.get("col")
            at = col if isinstance(col, int) and drawn.startswith(t, col) else drawn.find(t)
            if at >= 0:
                drawn = drawn[:at] + "\x01" + _band(e) + "\x02" + t + "\x03" + drawn[at + len(t):]
        if not all(isinstance(le.get("col"), int) for _, le in pairs):
            continue     # no installed column to draw against (certify always writes one)
        want = _u16(line)
        for e, le in sorted(pairs, key=lambda p: -p[1]["col"]):
            c, w = le["col"], len(le["token"])
            a, z = len(_u16(line[:c])), len(_u16(line[:c + w]))
            want = want[:a] + "\x01" + _band(e) + "\x02" + want[a:z] + "\x03" + want[z:]
        if drawn != want:
            toks = ", ".join(repr(e.get("token")) for e, _ in pairs)
            out.append(f"the page (minted before 2026-10-05) draws line {n} differently from the "
                       f"certificate: its script places a band for {toks} on another number, or "
                       f"on none")
    return out


# ---------------------------------------------------------------------------------
# the rendered capsule (layer 1 lives here, inline, zero external requests)
# ---------------------------------------------------------------------------------

# The page's install line names this floor, written into its template (_TEMPLATE, twice in the
# HTML and once as FLOOR in its script): 7.49.0, the next minor release after 7.48.1, which will
# carry this layer 2 (the whole-certificate comparison and the page check). Until review round 3
# the line was built from the minter's stated version, which a forger chooses and which can name
# a styxx that passes forged certificates (PyPI 7.48.0 passes the D1 forgery). verify advises
# when a capsule states a version below the floor. Changing it changes every page this renderer
# produces, so a later change keeps this renderer for the pages it minted, as
# styxx._capsule_page_v01_legacy keeps the one before it.
_LAYER2_FLOOR = "7.49.0"


def _version_key(v: str) -> tuple:
    """Order on the versions _VERSION accepts: the release numbers, then a pre-release or a
    development release before the release itself, a post-release after it."""
    m = re.fullmatch(r"(\d+(?:\.\d+){1,3})((?:a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?", v)
    if not m:
        return ()
    rel = tuple(int(x) for x in m.group(1).split("."))
    rel += (0,) * (4 - len(rel))
    early = bool(m.group(2)) or (bool(m.group(4)) and not m.group(3))
    return rel + (0 if early else 1,)


def _render_html(payload: dict) -> str:
    # The verdict is NOT written into the page at mint (2026-10-05). The badge used to carry it
    # from the moment the page opened and the script only recoloured it, so with one doctored
    # byte the red TAMPERED banner sat under a badge that still read OATH-HELD and cards that
    # still read "verified N", and a page whose script never ran (scripts off, or no WebCrypto
    # on plain http) read OATH-HELD and "checking integrity" forever. The badge now starts
    # neutral; only the script writes the certificate's verdict, after every embedded hash has
    # matched, and a mismatch writes TAMPERED on the badge itself.
    #
    # Every "<" in the payload is written as <, so no text inside it can open or close an
    # element, and the placeholders are filled in one pass, so a document name cannot carry one.
    payload_json = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    fill = {"TITLE": _html.escape(f"OATH Capsule — {payload['document']['name']}"),
            "PAYLOAD": payload_json}
    return re.sub(r"__(TITLE|PAYLOAD)__", lambda m: fill[m.group(1)], _TEMPLATE)


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<style>
:root{--paper:#1A0F26;--ink:#F3EBE0;--bone:#D0C5DA;--mute:#5C4E70;--sig:#C4B5FD;
--ok:#B7E4C7;--warn:#F5C5B0;--bad:#D89886;--rule:#3A2B47;}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);
font-family:ui-monospace,Consolas,monospace;font-size:14px;line-height:1.6}
header{padding:18px 24px;border-bottom:1px solid var(--rule);position:sticky;top:0;
background:var(--paper);z-index:5}
.badge{display:inline-block;padding:4px 14px;border-radius:2px;font-weight:700;
letter-spacing:.08em}
.badge.pending{background:#241830;color:var(--mute)}
.badge.held{background:var(--ok);color:#123}.badge.failed{background:var(--bad);color:#210}
.badge.warn{background:var(--warn);color:#321}
.badge.tampered{background:#f33;color:#fff}
.meta{color:var(--mute);font-size:12px;margin-top:6px}
.notrun{background:var(--warn);color:#321;padding:12px 24px;font-weight:700;margin-top:6px}
main{max-width:1080px;margin:0 auto;padding:24px}
h2{font-size:13px;letter-spacing:.14em;color:var(--sig);text-transform:uppercase;
margin:28px 0 10px}
pre.doc{white-space:pre-wrap;word-wrap:break-word;background:#150c20;border:1px solid
var(--rule);padding:18px;border-radius:3px;color:var(--bone)}
.tok{border-radius:2px;padding:0 2px;font-weight:700}
.tok.vo{background:rgba(196,181,253,.25);color:var(--sig)}
.tok.vv{background:rgba(196,181,253,.12);color:var(--sig);outline:1px dashed var(--mute)}
.tok.ab{color:var(--mute);outline:1px dotted var(--mute)}
.tok.un{background:rgba(216,152,134,.35);color:#ffd9cf}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}
.card{border:1px solid var(--rule);border-radius:3px;padding:12px;background:#150c20}
.card b{font-size:20px;display:block}.card span{color:var(--mute);font-size:11px}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--rule);
padding:6px 8px;text-align:left;font-size:12px}th{color:var(--mute)}
.hash{color:var(--mute);word-break:break-all;font-size:11px}
.match{color:var(--ok)}.mismatch{color:#f66;font-weight:700}
footer{border-top:1px solid var(--rule);margin-top:32px;padding:18px 24px;
color:var(--mute);font-size:12px;max-width:1080px;margin-left:auto;margin-right:auto}
.legend span{margin-right:14px}
#tamper{display:none;background:#f33;color:#fff;padding:14px 24px;font-weight:700}
</style></head><body>
<noscript><div class="notrun">THIS PAGE DID NOT RUN. Scripts are off, so it checked nothing
and shows no verdict. Check the capsule with layer 2: pip install "styxx>=7.49.0", then
python -m styxx.capsule verify on this file.</div></noscript>
<div id="tamper">TAMPERED — embedded bytes do not match this capsule's certificate. Its
verdict, counts and bands are not shown, because they do not describe these bytes.</div>
<header>
  <span class="badge pending" id="verdict">checking…</span>
  <span class="badge pending" id="integrity">integrity not checked yet</span>
  <div class="meta" id="pending">No verdict yet: this page shows one only after it has
  re-hashed every embedded byte. If this line stays, its check did not run; use layer 2,
  below.</div>
  <div class="meta" id="meta"></div>
</header>
<main>
  <h2>the boundary, up front</h2><div class="grid" id="cards"></div>
  <h2>document — every number wearing its band</h2>
  <div class="legend meta"><span class="tok vo">verified·obligated</span>
  <span class="tok vv">verified·volunteered</span><span class="tok ab">abstained</span>
  <span class="tok un">accused</span></div>
  <pre class="doc" id="doc"></pre>
  <h2>receipts — byte integrity</h2><table id="receipts"><tr><th>receipt</th>
  <th>sha-256 (recomputed in your browser)</th><th></th></tr></table>
  <h2>re-run it yourself (layer 2 — the real verifier)</h2>
  <pre class="doc">pip install "styxx>=7.49.0"
python -m styxx.capsule verify this_file.html</pre>
</main>
<footer id="foot"></footer>
<script type="application/json" id="oath-capsule">__PAYLOAD__</script>
<script>
(() => {
  const vb = document.getElementById('verdict');
  const ib = document.getElementById('integrity');
  const pend = document.getElementById('pending');
  let settled = false;
  // The page says it did not check, rather than leaving "checking…" up forever.
  const notChecked = why => {
    if (settled) return;
    settled = true;
    vb.textContent = 'NOT CHECKED'; vb.className = 'badge warn';
    ib.textContent = 'integrity: not checked'; ib.className = 'badge pending';
    pend.textContent = 'THIS PAGE DID NOT FINISH ITS CHECK: ' + why + '. It shows no ' +
      'verdict. Check the capsule with layer 2, below.';
    pend.className = 'notrun';
  };
  const timer = setTimeout(() => notChecked('it had not finished after 10 seconds'), 10000);
  const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  (async () => {
    const P = JSON.parse(document.getElementById('oath-capsule').textContent);
    const C = P.certificate;
    const subtle = (typeof crypto === 'object' && crypto) ? crypto.subtle : undefined;
    if (!subtle) {
      notChecked('this browser gives the page no WebCrypto, which it offers only to a page ' +
                 'opened from a file, from localhost or over https');
      return;
    }
    const b64b = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
    const hex = b => [...new Uint8Array(b)].map(x=>x.toString(16).padStart(2,'0')).join('');
    const sha = async u8 => hex(await subtle.digest('SHA-256', u8));
    const ver = String(P.verifier.styxx_version);
    // The styxx a reader installs to check this capsule in full: 7.49.0, the next minor release
    // after 7.48.1, which will carry this repair (layer 2's whole-certificate comparison and its
    // page check). Set here in the template, never taken from the payload, whose stated version
    // the minter chooses; an older styxx passes certificates this one fails.
    const FLOOR = '7.49.0';
    document.getElementById('meta').textContent =
      P.document.name + ' · capsule ' + P.spec + ' · minted ' + P.created +
      ' · verifier styxx ' + ver + ' (as stated by the minter)';
    document.querySelectorAll('main pre.doc')[1] &&
      (document.querySelectorAll('main pre.doc')[1].textContent =
       'pip install "styxx>=' + FLOOR + '"\n' +
       'python -m styxx.capsule verify ' + location.pathname.split('/').pop());

    // integrity: every embedded byte vs the certificate
    let tampered = false;
    const docBytes = b64b(P.document.b64);
    if (await sha(docBytes) !== C.document_sha256) tampered = true;
    const rows = [];
    for (const r of P.receipts) {
      const h = await sha(b64b(r.b64));
      const ok = h === (C.receipts_sha256 || {})[r.name];
      if (!ok) tampered = true;
      rows.push(`<tr><td>${esc(r.name)}</td><td class="hash">${h}</td>` +
        `<td class="${ok?'match':'mismatch'}">${ok?'matches certificate':'MISMATCH'}</td></tr>`);
    }
    if (settled) return;          // the timeout has already said this page did not finish
    // the text as certify reads it: a byte-order mark is kept, as Python keeps it
    const text = new TextDecoder('utf-8', { ignoreBOM: true }).decode(docBytes);
    const footText =
      'What this page checks, offline: that these exact bytes are the bytes the certificate ' +
      'hashed (SHA-256, recomputed in your browser). It draws the verdict and the bands from ' +
      'the certificate only when they match, and each band only where its ledger row says the ' +
      'number is. It re-runs nothing, and it cannot prove its own script honest. Layer 2 ' +
      're-runs the real verifier over the embedded bytes, compares the whole certificate with ' +
      'what it re-derives, checks that this page is the page styxx renders for them, and ' +
      'prints as stated by the minter, not checked, what bytes cannot show: when, and with ' +
      'which styxx, this capsule was minted. What neither proves: that the receipts ' +
      'truthfully record reality (that chain lives in repository provenance), or who minted ' +
      'this capsule (nothing in it is signed). A capsule is a portable binding, not a ' +
      'portable oath of origin. Nothing crosses unseen.';
    const unmarked = (badge, why, cardWhy) => {
      settled = true; clearTimeout(timer);
      document.getElementById('receipts').insertAdjacentHTML('beforeend', rows.join(''));
      document.getElementById('foot').textContent = footText;
      vb.textContent = badge;
      pend.textContent = why; pend.className = 'notrun';
      document.getElementById('cards').innerHTML =
        '<div class="card"><b>not drawn</b><span>' + cardWhy + '</span></div>';
      document.getElementById('doc').textContent = text;
    };
    if (tampered) {
      unmarked('TAMPERED', 'The certificate does not describe these bytes, so its verdict, ' +
        'counts and bands are not shown. The document below is the embedded text, unmarked.',
        'the certificate does not describe these bytes');
      vb.className = 'badge tampered';
      ib.textContent = 'INTEGRITY: FAILED'; ib.className = 'badge tampered';
      document.getElementById('tamper').style.display = 'block';
      return;
    }

    // everything the page draws from the certificate is built BEFORE the verdict is shown, so
    // a certificate the script cannot read ends in NOT CHECKED, never in a half-drawn verdict
    const verdict = String(C.verdict);
    const es = C.epistemics_summary || {}; const v = (es.verified)||{};
    const vm = v.value_match || {}; const dv = v.derived || {};
    const obl = (vm.obligated_integer_filter_ran||0)+(vm.obligated_integer_filter_na||0)
              +(dv.obligated||0);
    const tot = v.total || C.counts.VERIFIED || 0;
    // without a summary the share is unknown, not 100%: no obligated count was read
    const hasEs = !!C.epistemics_summary && typeof C.epistemics_summary === 'object';
    const cards = [
      ['verdict', verdict],
      ['verified', C.counts.VERIFIED],
      ['abstained', C.counts.ABSTAIN],
      ['accused', C.counts.UNGROUNDED],
      ['volunteered share', hasEs && tot ? Math.round(100*(tot-obl)/tot)+'%' : '—'],
    ];
    const cardsHtml = cards.map(
      ([k,val]) => `<div class="card"><b>${esc(val)}</b><span>${k}</span></div>`).join('');

    // Paint the document at certify's own coordinates. certify reads the text with universal
    // newlines and splits it as Python's str.splitlines() does (form feed, U+2028 and the rest
    // included); `line` counts those lines from 1 and `col` counts code points, with U+2212
    // read as '-'. Each band is a span element built from text, so no character in the document
    // can open one. A row whose token is not at its line and column is never moved to another
    // number: the page then draws no verdict and no bands.
    const parts = text.split(/(\r\n|[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029])/);
    const ledger = Array.isArray(C.ledger) ? C.ledger : [];
    const byLine = new Map();
    for (const e of ledger) {
      if (!e || !Number.isInteger(e.line)) continue;
      if (!byLine.has(e.line)) byLine.set(e.line, []);
      byLine.get(e.line).push(e);
    }
    // A band is read from the row itself, or the row is not drawn: a row with no status, a status
    // this page does not know, or a verified row with no obligation flag used to fall through to
    // a verified band, so an accused number stripped of its status was painted verified.
    const band = e => e.status==='UNGROUNDED' ? 'un' : e.status==='ABSTAIN' ? 'ab'
      : (e.status==='VERIFIED' && e.epistemics && typeof e.epistemics.obligated === 'boolean')
        ? (e.epistemics.obligated ? 'vo' : 'vv') : '';
    const at = e => Number.isInteger(e.col) ? e.col : -1;
    const segs = [];
    let placed = 0;
    for (let k = 0; k < parts.length; k += 2) {
      const cps = Array.from(parts[k]);
      const norm = cps.map(c => c === '\u2212' ? '-' : c);
      let pos = 0;
      for (const e of (byLine.get(k / 2 + 1) || []).slice().sort((a, b) => at(a) - at(b))) {
        const t = Array.from(String(e.token)); const c = at(e);
        if (!band(e) || c < pos || !t.length ||
            norm.slice(c, c + t.length).join('') !== t.join('')) continue;
        if (c > pos) segs.push(['', cps.slice(pos, c).join('')]);
        segs.push([band(e), cps.slice(c, c + t.length).join('')]);
        pos = c + t.length; placed++;
      }
      if (pos < cps.length) segs.push(['', cps.slice(pos).join('')]);
      if (k + 1 < parts.length) segs.push(['', parts[k + 1]]);
    }
    if (placed !== ledger.length) {
      unmarked('NOT CHECKED', 'The certificate does not fit these bytes: ' +
        (ledger.length - placed) + ' of its ' + ledger.length + ' ledger rows do not sit at ' +
        'their recorded line and column, or carry no band this page can read. Its verdict, ' +
        'counts and bands are not shown; check the capsule with layer 2, below.',
        'the ledger does not fit these bytes');
      vb.className = 'badge warn';
      ib.textContent = 'integrity: all hashes match'; ib.className = 'badge';
      return;
    }
    const nodes = segs.map(([c, s]) => {
      if (!c) return document.createTextNode(s);
      const sp = document.createElement('span');
      sp.className = 'tok ' + c; sp.textContent = s;
      return sp;
    });

    settled = true; clearTimeout(timer);
    document.getElementById('receipts').insertAdjacentHTML('beforeend', rows.join(''));
    document.getElementById('foot').textContent = footText;
    ib.textContent = 'integrity: all hashes match'; ib.className = 'badge';
    ib.style.background = 'rgba(183,228,199,.15)'; ib.style.color = 'var(--ok)';
    vb.textContent = verdict;
    vb.className = 'badge ' + (verdict === 'OATH-HELD' ? 'held'
      : /^OATH-HELD,/.test(verdict) ? 'warn' : 'failed');
    pend.textContent = 'Every embedded byte matches the certificate, so its verdict is shown. ' +
      'This page re-runs nothing; layer 2, below, re-derives the verdict.';
    document.getElementById('cards').innerHTML = cardsHtml;
    const docEl = document.getElementById('doc');
    docEl.textContent = '';
    for (const n of nodes) docEl.appendChild(n);
  })().catch(e => notChecked('its script failed (' + ((e && e.message) || e) + ')'));
})();
</script>
</body></html>
"""


# ---------------------------------------------------------------------------------
# v0.2 — the agent-handoff capsule (diffgate record over summary + diff)
# Spec: papers/closed-model-frontier/SPEC_oath_capsule_v02_2026_08_31.md
# ---------------------------------------------------------------------------------

def _render_html_v02(payload: dict) -> str:
    g = payload["gate"]
    payload_json = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    title = (f"OATH Capsule — {payload['summary']['name']} × "
             f"{payload['diff']['name']}").replace("<", "")
    return (_TEMPLATE_V02
            .replace("__TITLE__", title)
            .replace("__VERDICT__", str(g.get("verdict", "?")))
            .replace("__PIP__", payload["verifier"]["pip"])
            .replace("__PAYLOAD__", payload_json))


_TEMPLATE_V02 = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<style>
:root{--paper:#1A0F26;--ink:#F3EBE0;--bone:#D0C5DA;--mute:#5C4E70;--sig:#C4B5FD;
--ok:#B7E4C7;--warn:#F5C5B0;--bad:#D89886;--rule:#3A2B47;}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);
font-family:ui-monospace,Consolas,monospace;font-size:14px;line-height:1.6}
header{padding:18px 24px;border-bottom:1px solid var(--rule);position:sticky;top:0;
background:var(--paper);z-index:5}
.badge{display:inline-block;padding:4px 14px;border-radius:2px;font-weight:700;
letter-spacing:.06em}
.badge.held{background:var(--ok);color:#123}.badge.failed{background:var(--bad);color:#210}
.badge.warn{background:var(--warn);color:#321}.badge.tampered{background:#f33;color:#fff}
.meta{color:var(--mute);font-size:12px;margin-top:6px}
main{max-width:1080px;margin:0 auto;padding:24px}
h2{font-size:13px;letter-spacing:.14em;color:var(--sig);text-transform:uppercase;
margin:28px 0 10px}
pre.doc{white-space:pre-wrap;word-wrap:break-word;background:#150c20;border:1px solid
var(--rule);padding:18px;border-radius:3px;color:var(--bone);max-height:480px;
overflow:auto}
.band{border-radius:2px;padding:0 2px}
.band.vg{background:rgba(183,228,199,.18);color:var(--ok)}
.band.cb{background:rgba(216,152,134,.35);color:#ffd9cf;font-weight:700}
.band.ua{background:rgba(245,197,176,.18);color:var(--warn)}
.band.uc{color:var(--mute)}
.band .trunc{color:var(--mute);font-size:10px;font-weight:400}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.card{border:1px solid var(--rule);border-radius:3px;padding:12px;background:#150c20}
.card b{font-size:20px;display:block}.card span{color:var(--mute);font-size:11px}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--rule);
padding:6px 8px;text-align:left;font-size:12px;vertical-align:top}th{color:var(--mute)}
td.v-VERIFIED{color:var(--ok)}td.v-CONTRADICTED{color:#ffb3a3;font-weight:700}
td.v-UNCHECKABLE{color:var(--warn)}
.dl-add{color:var(--ok)}.dl-del{color:var(--bad)}
.note{color:var(--mute);font-size:12px;margin:6px 0}
.disclose{color:var(--warn);font-size:12px;margin:6px 0}
footer{border-top:1px solid var(--rule);margin-top:32px;padding:18px 24px;
color:var(--mute);font-size:12px;max-width:1080px;margin-left:auto;margin-right:auto;
white-space:pre-wrap}
#tamper{display:none;background:#f33;color:#fff;padding:14px 24px;font-weight:700}
#why{color:#fff;font-weight:400;font-size:12px}
</style></head><body>
<noscript><div style="background:#f5c5b0;color:#321;padding:14px 24px;font-weight:700">
UNVERIFIED RENDERING — this page proves nothing without its verifier; run layer 2.
</div></noscript>
<div id="tamper">TAMPERED — this capsule does not verify. Nothing below can be trusted.
<div id="why"></div></div>
<header>
  <span class="badge" id="verdict">__VERDICT__</span>
  <span class="badge" id="integrity" style="background:#241830;color:var(--mute)">checking
  integrity…</span>
  <div class="meta" id="meta"></div>
</header>
<main>
  <h2>the gate, up front</h2><div class="grid" id="cards"></div>
  <h2>summary — every recorded sentence wearing its verdict</h2>
  <div class="meta"><span class="band vg">verified</span>
  <span class="band cb">contradicted</span> <span class="band ua">uncheckable</span>
  <span class="band uc">uncovered — listed, never judged</span></div>
  <div class="disclose" id="disclose"></div>
  <pre class="doc" id="doc"></pre>
  <h2>claims — the record, verbatim</h2>
  <table id="claims"><tr><th>kind</th><th>text</th><th>detail</th><th>verdict</th>
  <th>why</th></tr></table>
  <h2>the diff — display only, not parsed, not verified by this page</h2>
  <pre class="doc" id="diff"></pre>
  <h2>re-run it yourself (layer 2 — the real instrument)</h2>
  <pre class="doc">pip install __PIP__
python -m styxx.capsule verify this_file.html</pre>
</main>
<footer>what layer 1 proves (this page, offline): the embedded summary, diff, and gate
record are byte-for-byte the material this capsule binds — sha-256, recomputed in your
browser just now — and the sealed verdict follows arithmetically from the sealed claims.
this page re-runs nothing. the badge is a convenience, not an authority: this page cannot
prove its own javascript honest. decisions go through layer 2.

what layer 2 proves (pip install __PIP__): the gate record — verdict, every claim verdict,
every why-string, every count, every uncovered sentence — re-derives from the embedded
summary and diff by re-running the real instrument. one exception, printed when it
applies: unparsed_claims is observational and depends on whether styxx.claimdetect is
importable where the verifier runs; divergence there is reported, never treated as tamper.

what no layer proves: who minted this — no signatures; anyone can mint an internally
honest capsule over bytes of their choosing, and a re-mint over different bytes is a
different honest capsule, not a forgery this format can catch. when — the timestamp is
unsealed and unproven. that this diff was ever applied to any repository, branch, or
deployment — the capsule pins diff bytes, not repo state. that tests passed — environment
legs are refused at mint; tests_pass can only appear here as UNCHECKABLE, by construction.
that the summary's uncovered prose is true — uncovered sentences are listed, never judged;
coverage is not correctness. that this run is the only run — a capsule proves this
artifact, never the absence of others.

a capsule is a portable binding, not a portable oath of origin. nothing crosses unseen.
</footer>
<script type="application/json" id="oath-capsule">__PAYLOAD__</script>
<script>
(async () => {
  const $ = id => document.getElementById(id);
  const fail = msgs => {
    $('tamper').style.display = 'block';
    $('why').textContent = msgs.join(' · ');
    const ib = $('integrity');
    ib.textContent = 'INTEGRITY: FAILED'; ib.className = 'badge tampered';
  };
  const blocks = document.querySelectorAll('script#oath-capsule');
  if (blocks.length !== 1) return fail(['ambiguous payload: marker not unique']);
  let P;
  try { P = JSON.parse(blocks[0].textContent); }
  catch (e) { return fail(['payload unparseable']); }
  if (P.spec !== 'styxx-oath/capsule/v0.2') return fail(['unknown spec: ' + P.spec]);
  const g = P.gate;

  const b64b = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
  const hex = b => [...new Uint8Array(b)].map(x => x.toString(16).padStart(2, '0')).join('');
  const sha = async u8 => hex(await crypto.subtle.digest('SHA-256', u8));
  // RFC 8785 JCS, exact for the float-free gate record (parity-tested vs Python)
  const jcs = o => o === null ? 'null' : o === true ? 'true' : o === false ? 'false'
    : typeof o === 'number' ? String(o)
    : typeof o === 'string' ? JSON.stringify(o)
    : Array.isArray(o) ? '[' + o.map(jcs).join(',') + ']'
    : '{' + Object.keys(o).sort().map(k => JSON.stringify(k) + ':' + jcs(o[k])).join(',') + '}';

  // header — qualified badge, never bare
  const vb = $('verdict');
  vb.textContent = g.verdict + ' · ' + g.claims.length + ' claims checked · ' +
    g.uncovered_sentences + '/' + g.sentences_total + ' sentences uncovered';
  vb.className = 'badge ' +
    (g.verdict === 'PASS' ? (g.claims.length ? 'held' : 'warn') : 'failed');
  $('meta').textContent = P.summary.name + ' + ' + P.diff.name + ' · capsule ' + P.spec +
    ' · minted ' + P.created + ' (timestamp unsealed) · verifier styxx ' +
    P.verifier.styxx_version;

  // layer 1: hashes + the arithmetic folds the instrument guarantees
  const sumBytes = b64b(P.summary.b64), diffBytes = b64b(P.diff.b64);
  const bad = [];
  if (await sha(sumBytes) !== P.binding.summary.value) bad.push('summary bytes do not match binding');
  if (await sha(diffBytes) !== P.binding.diff.value) bad.push('diff bytes do not match binding');
  if (await sha(new TextEncoder().encode(jcs(g))) !== P.binding.gate.value) bad.push('gate record does not match binding');
  const nCon = g.claims.filter(c => c.verdict === 'CONTRADICTED').length;
  if (g.verdict !== (nCon ? 'FAIL' : 'PASS')) bad.push('verdict does not follow from the sealed claims');
  if (g.uncovered_sentences !== g.uncovered_texts.length) bad.push('uncovered count fold broken');
  if (g.measured !== true) bad.push('unmeasured record inside a minted capsule');
  if (g.base !== '(diff-text)' || g.head !== '(diff-text)') bad.push('base/head invariant broken');
  const ib = $('integrity');
  if (bad.length) { fail(bad); } else {
    ib.textContent = 'integrity: hashes match — verdict shown as recorded and follows from claims';
    ib.className = 'badge';
    ib.style.background = 'rgba(183,228,199,.15)'; ib.style.color = 'var(--ok)';
  }

  // cards
  const nVer = g.claims.filter(c => c.verdict === 'VERIFIED').length;
  const nUnc = g.claims.filter(c => c.verdict === 'UNCHECKABLE').length;
  const covered = g.sentences_total - g.uncovered_sentences;
  const cards = [
    ['claims verified', nVer], ['contradicted', nCon], ['uncheckable', nUnc],
    ['sentences covered', covered + '/' + g.sentences_total],
    ['uncovered — never judged', g.uncovered_sentences],
    ['unparsed claim-shaped', g.unparsed_claims.length],
  ];
  const cardsEl = $('cards');
  for (const [k, v] of cards) {
    const d = document.createElement('div'); d.className = 'card';
    const b = document.createElement('b'); b.textContent = String(v);
    const s = document.createElement('span'); s.textContent = k;
    d.appendChild(b); d.appendChild(s); cardsEl.appendChild(d);
  }

  // control characters (except newline/tab) render visibly, with a count
  let ctrl = 0;
  const clean = s => s.replace(
    /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f\u200e\u200f\u202a-\u202e\u2066-\u2069]/g,
    () => { ctrl++; return '�'; });

  // paint the summary: locate recorded texts — never re-split, never re-judge
  const text = new TextDecoder('utf-8').decode(sumBytes);
  const ranges = [];
  let unlocated = 0;
  const locate = (t, cls) => {
    if (!t) return;
    let from = 0, at;
    while ((at = text.indexOf(t, from)) !== -1) {
      if (!ranges.some(r => at < r.end && at + t.length > r.start)) {
        ranges.push({ start: at, end: at + t.length, cls, trunc: t.length === 160 });
        return;
      }
      from = at + 1;
    }
    unlocated++;
  };
  for (const c of g.claims)
    locate(c.text, c.verdict === 'VERIFIED' ? 'vg'
      : c.verdict === 'CONTRADICTED' ? 'cb' : 'ua');
  for (const u of g.uncovered_texts) locate(u, 'uc');
  ranges.sort((a, b) => a.start - b.start);
  const pre = $('doc');
  let pos = 0;
  for (const r of ranges) {
    if (r.start > pos) pre.appendChild(document.createTextNode(clean(text.slice(pos, r.start))));
    const span = document.createElement('span'); span.className = 'band ' + r.cls;
    span.textContent = clean(text.slice(r.start, r.end));
    if (r.trunc) {
      const m = document.createElement('span'); m.className = 'trunc';
      m.textContent = ' …record truncates at 160'; span.appendChild(m);
    }
    pre.appendChild(span); pos = r.end;
  }
  if (pos < text.length) pre.appendChild(document.createTextNode(clean(text.slice(pos))));
  const dis = [];
  if (unlocated) dis.push(unlocated + ' recorded sentence(s) could not be located for painting — see the claims table');
  if (ctrl) dis.push(ctrl + ' invisible control character(s) rendered as �');
  $('disclose').textContent = dis.join(' · ');

  // claims table — every why-string verbatim
  const tbl = $('claims');
  for (const c of g.claims) {
    const tr = document.createElement('tr');
    for (const [val, cls] of [[c.kind, ''], [c.text, ''],
        [JSON.stringify(c.detail), ''], [c.verdict, 'v-' + c.verdict],
        [c.why + (c.kind === 'tests_pass' ? ' [environment leg — refused at mint by construction]' : ''), '']]) {
      const td = document.createElement('td');
      if (cls) td.className = cls;
      td.textContent = clean(String(val));
      tr.appendChild(td);
    }
    tbl.appendChild(tr);
  }

  // diff panel — display only
  const dpre = $('diff');
  for (const line of new TextDecoder('utf-8').decode(diffBytes).split('\n')) {
    const span = document.createElement('span');
    if (line.startsWith('+') && !line.startsWith('+++')) span.className = 'dl-add';
    else if (line.startsWith('-') && !line.startsWith('---')) span.className = 'dl-del';
    span.textContent = clean(line) + '\n';
    dpre.appendChild(span);
  }

  // layer-2 command with this file's actual name
  document.querySelectorAll('main pre.doc')[3] &&
    (document.querySelectorAll('main pre.doc')[3].textContent =
      'pip install ' + P.verifier.pip + '\n' +
      'python -m styxx.capsule verify ' + location.pathname.split('/').pop());
})();
</script>
</body></html>
"""


def _canonical_text_bytes(path: Path, what: str) -> tuple[str, bytes]:
    """Read as the instruments read: universal-newline text, re-encoded UTF-8.

    The v0.1 CRLF lesson, applied to every v0.2 input before it bites: the gate
    consumes TEXT, so the capsule binds the text bytes, not the checkout's.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise SystemExit(f"REFUSED: cannot read {what} as UTF-8 text: {e}")
    return text, text.encode("utf-8")


def _v02_folds(gate: dict) -> List[str]:
    """The arithmetic invariants a genuine strict=False, run=None gate record
    satisfies — mirrored verbatim by the capsule's layer-1 JS. A violation can
    only mean tampering (or an instrument this verifier does not know)."""
    problems = []
    contradicted = sum(1 for c in gate.get("claims", [])
                       if c.get("verdict") == "CONTRADICTED")
    want = "FAIL" if contradicted else "PASS"
    if gate.get("verdict") != want:
        problems.append(f"verdict fold: {gate.get('verdict')!r} does not follow from "
                        f"{contradicted} CONTRADICTED claims (expected {want!r})")
    if gate.get("uncovered_sentences") != len(gate.get("uncovered_texts", [])):
        problems.append("uncovered count fold: uncovered_sentences != len(uncovered_texts)")
    if gate.get("measured") is not True:
        problems.append("measured invariant: a minted v0.2 capsule can only carry a "
                        "measured gate")
    if gate.get("base") != "(diff-text)" or gate.get("head") != "(diff-text)":
        problems.append("base/head invariant: v0.2 gates are minted from diff text only")
    return problems


def _gate_binding_hash(gate: dict) -> str:
    from styxx.attestation import jcs
    return _sha256(jcs(gate).encode("utf-8"))


def create_capsule_diffgate(summary: Path, diff: Path, out: Path,
                            gate_path: Path | None = None) -> Path:
    """Mint the agent-handoff capsule — refusing, loudly, to mint one that lies.

    The gate embedded is ALWAYS the live mint-time re-run over the canonical
    bytes (a supplied --gate is only cross-checked, never sealed), so layer-2
    reproduction succeeds by construction and the record is a pure function of
    (summary bytes, diff bytes): strict=False, run=None, nothing self-reported.
    """
    from styxx.diffgate import gate_diff_text
    from styxx._version import __version__

    summary_text, summary_bytes = _canonical_text_bytes(summary, "summary")   # R1/R3
    diff_text, diff_bytes = _canonical_text_bytes(diff, "diff")               # R1/R3

    live = gate_diff_text(summary_text, diff_text, run=None, strict=False).to_dict()

    if gate_path is not None:
        try:
            supplied = json.loads(gate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:                          # R1
            raise SystemExit(f"REFUSED: cannot parse supplied gate: {e}")
        if supplied.get("diffgate") != "v0":                                  # R2
            raise SystemExit(f"REFUSED: unknown diffgate version "
                             f"{supplied.get('diffgate')!r} — cannot re-run it")
        for c in supplied.get("claims", []):                                  # R4
            if c.get("kind") == "tests_pass" and c.get("verdict") != "UNCHECKABLE":
                raise SystemExit(
                    "REFUSED: environment legs cannot be capsuled in v0.2; a "
                    "--run-resolved tests_pass verdict would require executing an "
                    "embedded shell string to verify. re-gate without --run.")
        diverged = [k for k in live                                           # R5
                    if k not in ("base", "head") and supplied.get(k) != live[k]]
        if diverged == ["unparsed_claims"]:
            raise SystemExit(
                "REFUSED: styxx.claimdetect availability differs between the gate's "
                "environment and this mint environment — re-gate here or omit --gate.")
        if (diverged == ["verdict"]
                and any(c.get("verdict") == "UNCHECKABLE"
                        for c in supplied.get("claims", []))):
            raise SystemExit(
                "REFUSED: v0.2 gates are non-strict by policy; strictness is a "
                "read-side policy — every UNCHECKABLE is visible in the record. "
                "re-gate non-strict or omit --gate.")
        if diverged:
            raise SystemExit(
                "REFUSED: supplied gate does not reproduce from these bytes — stale, "
                "forged, or produced by a different code path "
                f"(diverging fields: {diverged}); re-gate from the exported diff or "
                "omit --gate.")

    if not live.get("measured", False):                                       # R6
        raise SystemExit(
            f"REFUSED: gate measured nothing (why_unmeasured: "
            f"{live.get('why_unmeasured')!r}) — a capsule cannot carry proof of a "
            "non-measurement.")

    folds = _v02_folds(live)
    if folds:  # cannot happen at the pinned instrument; a fold here means skew
        raise SystemExit(f"REFUSED: live gate violates its own invariants: {folds}")

    payload = {
        "spec": SPEC_V02,
        "created": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {"name": summary.name, "b64": _b64(summary_bytes)},
        "diff": {"name": diff.name, "b64": _b64(diff_bytes)},
        "gate": live,
        "binding": {
            "summary": {"alg": "sha256", "value": _sha256(summary_bytes)},
            "diff": {"alg": "sha256", "value": _sha256(diff_bytes)},
            "gate": {"alg": "sha256-jcs", "value": _gate_binding_hash(live)},
        },
        "verifier": {"styxx_version": __version__, "pip": f"styxx=={__version__}"},
    }
    html = _render_html_v02(payload)
    if html.count(_BEGIN) != 1:                                               # R7
        raise SystemExit("REFUSED: payload marker is not unique in the rendered "
                         "capsule — refusing to write an ambiguous artifact")
    out.write_text(html, encoding="utf-8")
    rep = verify_capsule(out)                                                 # R7
    if not rep["ok"]:
        out.unlink(missing_ok=True)
        raise SystemExit(f"REFUSED: the freshly minted capsule fails its own "
                         f"layer-2 verify: {rep['problems']}")
    return out


def _verify_capsule_v02(html: str, payload: dict) -> dict:
    from styxx.diffgate import gate_diff_text

    result = {"ok": False, "spec": SPEC_V02, "stage": "parse",
              "problems": [], "advisory": [],
              "verdict": None, "gate_reproduced": False, "reproduced_at": None,
              "summary": None, "diff": None}
    problems: List[str] = []
    advisory: List[str] = []

    if html.count(_BEGIN) != 1:
        result["problems"] = ["ambiguous payload: marker occurs more than once"]
        return result

    obj = lambda k: payload.get(k) if isinstance(payload.get(k), dict) else {}  # noqa: E731
    gate = obj("gate")
    binding = obj("binding")
    result["verdict"] = gate.get("verdict")
    result["summary"] = obj("summary").get("name")
    result["diff"] = obj("diff").get("name")

    # stage: binding — every embedded byte vs its sealed hash
    result["stage"] = "binding"
    # a payload without the two embedded files fails here rather than in a traceback (2026-10-05)
    for k in ("summary", "diff"):
        f = payload.get(k)
        if not isinstance(f, dict) or not isinstance(f.get("b64"), str):
            problems.append(f"payload.{k} is not an embedded file (an object with b64)")
    if problems:
        result["problems"] = problems
        return result
    try:
        summary_bytes = base64.b64decode(payload["summary"]["b64"])
        diff_bytes = base64.b64decode(payload["diff"]["b64"])
    except ValueError as e:
        result["problems"] = [f"embedded bytes are not base64: {e}"]
        return result
    if _sha256(summary_bytes) != (binding.get("summary") or {}).get("value"):
        problems.append("summary bytes != binding.summary")
    if _sha256(diff_bytes) != (binding.get("diff") or {}).get("value"):
        problems.append("diff bytes != binding.diff")
    if _gate_binding_hash(gate) != (binding.get("gate") or {}).get("value"):
        problems.append("gate record != binding.gate (sha256-jcs)")
    if problems:
        result["problems"] = problems
        return result

    # stage: re-execution — the decisive leg. run=None, strict=False, always.
    result["stage"] = "reproduced"
    live = gate_diff_text(summary_bytes.decode("utf-8"), diff_bytes.decode("utf-8"),
                          run=None, strict=False).to_dict()
    if live.get("diffgate") != gate.get("diffgate"):
        problems.append(
            f"INSTRUMENT SKEW: installed diffgate {live.get('diffgate')!r} vs embedded "
            f"{gate.get('diffgate')!r} — reproduce under `pip install "
            f"{payload.get('verifier', {}).get('pip', 'styxx')}` before treating this "
            "as tamper")
    else:
        try:
            import styxx.claimdetect  # noqa: F401  — availability probe only
            _claimdetect = True
        except Exception:
            _claimdetect = False
        for k in live:
            if k == "unparsed_claims":
                if not _claimdetect:
                    advisory.append("SKIPPED: unparsed_claims (claimdetect unavailable "
                                    "here — observational field)")
                elif live[k] != gate.get(k):
                    advisory.append(
                        f"unparsed_claims diverges (embedded {gate.get(k)!r} vs live "
                        f"{live[k]!r}) — observational, environment-dependent, never "
                        "treated as tamper")
                continue
            if live[k] != gate.get(k):
                problems.append(f"gate.{k} not reproduced: embedded {gate.get(k)!r} "
                                f"vs live {live[k]!r}")

    result["problems"] = problems
    result["advisory"] = advisory
    result["gate_reproduced"] = not problems
    result["ok"] = not problems
    result["reproduced_at"] = "installed verifier" if not problems else None
    return result


# ---------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------

# =================================================================================
# the sworn profile (SPEC_sworn_browser_verifier_v01_2026_09_05.md, B6)
#
# What it seals: the document bytes, the manifest the spans resolve against, the verdict receipt
# styxx.sworn issued, and the browser verifier's own bytes. What it claims, and the words are the
# plan's:
#
#     re-derives sworn span verdicts offline; a forger controlling the whole file passes both
#     browser layers; the package at the named commit is the check
#
# Layer 1 (browser) re-derives the PORTABLE core — the receipt minus `verifier` and minus
# `coverage`, which is the number the conformance vectors pin — and compares its digest to the
# sealed one. It cannot check `verifier`, because that block names a Python build it has never
# seen; layer 2 does. Neither layer makes the file honest: a forger who controls the whole file
# controls both, and the package at the named commit is the check.
# =================================================================================

SWORN_REFUSALS = ("sworn_no_manifest", "sworn_receipt_mismatch", "sworn_manifest_mismatch",
                  "sworn_tree_receipt", "sworn_document_mismatch")

_SWORN_LABEL = ("re-derives sworn span verdicts offline; a forger controlling the whole file "
                "passes both browser layers; the package at the named commit is the check")

# the receipt fields that sit OUTSIDE the portable core: `digest`/`timestamp`/`coverage` travel
# outside the receipt digest already (sworn R9), and `verifier` names a build a second
# implementation cannot reproduce.
_SWORN_OUTSIDE_CORE = ("digest", "timestamp", "coverage", "coverage_sha256", "verifier")


def _sworn_verifier_js() -> bytes:
    """The browser verifier's bytes, from the installed package."""
    return (Path(__file__).parent / "_data" / "sworn_verify.js").read_bytes()


def _sworn_portable_core(receipt: dict) -> dict:
    return {k: v for k, v in receipt.items() if k not in _SWORN_OUTSIDE_CORE}


def _sworn_core_sha256(receipt: dict) -> str:
    from styxx.attestation import jcs
    return _sha256(jcs(_sworn_portable_core(receipt)).encode("utf-8"))


def create_capsule_sworn(doc: Path, manifest: Optional[Path], receipt: Path, out: Path) -> Path:
    """Mint a sworn capsule. Refuses, by name, rather than sealing something the browser could
    only ever call UNRESOLVED — or something that does not re-derive here and now."""
    from styxx.sworn import Manifest, issue_receipt, scan, verify
    from styxx._version import __version__

    doc_bytes = doc.read_bytes()
    rec_obj = json.loads(receipt.read_text(encoding="utf-8"))
    man_obj = json.loads(manifest.read_text(encoding="utf-8")) if manifest else None

    # sworn_document_mismatch — the sealed bytes must be the bytes the receipt was issued over
    if _sha256(doc_bytes) != (rec_obj.get("document") or {}).get("inline_sha256"):
        raise SystemExit("REFUSED sworn_document_mismatch: the document bytes do not hash to the "
                         "receipt's document.inline_sha256")

    # sworn_tree_receipt — v0.1 seals no tree, so a path:/prereg: span could only be UNRESOLVED
    sc = scan(doc_bytes)
    for d in sc["declarations"]:
        r = d.get("receipt") or ""
        if r.startswith("path:") or r.startswith("prereg:"):
            raise SystemExit(f"REFUSED sworn_tree_receipt: the span at {d['at']} names {r!r}; "
                             "this profile seals no tree and the browser could only call it "
                             "UNRESOLVED. Seal a document whose spans resolve against the "
                             "manifest, or wait for a profile that carries a snapshot.")

    # sworn_no_manifest — an rN with nothing to resolve against
    needs_manifest = any((d.get("receipt") or "").startswith("r") and
                         not (d.get("receipt") or "").startswith(("path:", "prereg:"))
                         for d in sc["declarations"] if d.get("receipt"))
    if needs_manifest and man_obj is None:
        raise SystemExit("REFUSED sworn_no_manifest: the document binds an rN span and no "
                         "manifest was given to seal beside it")

    man = Manifest.from_dict(man_obj) if man_obj is not None else None

    # sworn_manifest_mismatch — the receipt must name the manifest being sealed
    if rec_obj.get("manifest_digest") != (man.digest_or_none() if man is not None else None):
        raise SystemExit("REFUSED sworn_manifest_mismatch: the receipt names manifest digest "
                         f"{rec_obj.get('manifest_digest')!r} and the sealed manifest digests to "
                         f"{man.digest_or_none() if man is not None else None!r}")

    # sworn_receipt_mismatch — the receipt must re-derive from the sealed bytes, here and now
    live = verify(doc_bytes, name=(rec_obj.get("document") or {}).get("name", ""),
                  manifest=man, commit=rec_obj.get("commit"))
    if _sworn_core_sha256(issue_receipt(live)) != _sworn_core_sha256(rec_obj):
        raise SystemExit("REFUSED sworn_receipt_mismatch: the receipt's core does not re-derive "
                         "from the sealed bytes at the installed verifier "
                         f"(live {live['document_verdict']} {live['counts']} vs sealed "
                         f"{rec_obj.get('document_verdict')} {rec_obj.get('counts')})")

    js = _sworn_verifier_js()
    payload = {
        "spec": SPEC_SWORN,
        "created": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label": _SWORN_LABEL,
        "document": {"name": doc.name, "b64": _b64(doc_bytes),
                     "sha256": _sha256(doc_bytes)},
        "manifest": man_obj,
        "receipt": rec_obj,
        "core_sha256": _sworn_core_sha256(rec_obj),
        "verifier_js": {"sha256": _sha256(js), "b64": _b64(js)},
        "verifier": {"styxx_version": __version__,
                     "sworn_sha256": (rec_obj.get("verifier") or {}).get("sworn_sha256"),
                     "pip": f"styxx=={__version__}"},
    }
    # LF, explicitly: the verifier is sealed as bytes AND inlined in the page, and a
    # platform newline would make the inlined copy differ from the sealed one on disk.
    out.write_bytes(_render_html_sworn(payload, js.decode("utf-8")).encode("utf-8"))

    # A CAPSULE THAT CANNOT VERIFY MUST NOT EXIST — the v0.1 rule, kept.
    report = verify_capsule(out)
    if not report.get("ok"):
        problems = report.get("problems") or [report.get("error", "unknown")]
        out.unlink(missing_ok=True)
        raise SystemExit("REFUSED: the minted capsule does not verify, so it was not kept.\n"
                         + "\n".join(f"  - {p}" for p in problems[:6]))
    return out


def _verify_capsule_sworn(html: str, payload: dict) -> dict:
    """Layer 2: re-run styxx.sworn over the sealed bytes. INSTRUMENT SKEW is named apart from
    tamper — one is the instrument having moved, the other is the bytes having moved."""
    from styxx.sworn import Manifest, issue_receipt, verify

    problems: List[str] = []
    advisory: List[str] = []
    rec = payload.get("receipt") or {}
    out = {"ok": False, "spec": SPEC_SWORN,
           "document": (payload.get("document") or {}).get("name"),
           "verdict": rec.get("document_verdict"), "counts": rec.get("counts"),
           "problems": problems, "advisory": advisory, "label": _SWORN_LABEL}

    try:
        doc_bytes = base64.b64decode((payload["document"])["b64"], validate=True)
    except Exception as e:                                  # noqa: BLE001
        problems.append(f"document bytes are not decodable: {e}")
        return out
    if _sha256(doc_bytes) != (payload["document"]).get("sha256"):
        problems.append("document bytes != payload document.sha256 (tamper)")
    if _sha256(doc_bytes) != (rec.get("document") or {}).get("inline_sha256"):
        problems.append("document bytes != receipt document.inline_sha256 (tamper)")

    # the sealed browser verifier must be the one inlined in the page, byte for byte
    try:
        js = base64.b64decode((payload["verifier_js"])["b64"], validate=True)
    except Exception as e:                                  # noqa: BLE001
        problems.append(f"the sealed browser verifier is not decodable: {e}")
        js = b""
    if js and _sha256(js) != (payload["verifier_js"]).get("sha256"):
        problems.append("the sealed browser verifier does not hash to its sealed digest (tamper)")
    # Content identity modulo newlines, both sides. `html` came through read_text, which
    # normalises CRLF to LF, so comparing it against raw sealed bytes accused a correctly minted
    # capsule of tamper on any checkout that holds the verifier CRLF. The sealed DIGEST above
    # still pins the exact bytes; this check asks the weaker question it was always asking —
    # is the code the page runs the code this capsule sealed?
    if js:
        inlined = html.replace("\r\n", "\n")
        sealed_text = js.decode("utf-8", errors="replace").replace("\r\n", "\n")
        if sealed_text not in inlined:
            problems.append("the browser verifier inlined in the page is not the sealed one "
                            "(tamper) — layer 1 ran something this capsule did not seal")
    installed = _sworn_verifier_js()
    if js and js != installed:
        advisory.append("INSTRUMENT SKEW: the sealed browser verifier differs from the installed "
                        "one; the sealed bytes are what layer 1 ran")

    man_obj = payload.get("manifest")
    man = None
    if man_obj is not None:
        try:
            man = Manifest.from_dict(man_obj)
        except SystemExit as e:
            problems.append(f"the sealed manifest does not load: {e}")
    if rec.get("manifest_digest") != (man.digest_or_none() if man is not None else None):
        problems.append("the receipt's manifest_digest is not the sealed manifest's (tamper)")

    try:
        live = verify(doc_bytes, name=(rec.get("document") or {}).get("name", ""),
                      manifest=man, commit=rec.get("commit"))
        live_rec = issue_receipt(live)
    except SystemExit as e:
        problems.append(f"the sealed bytes do not verify at the installed instrument: {e}")
        return out

    sealed_build = (rec.get("verifier") or {}).get("sworn_sha256")
    live_build = (live_rec.get("verifier") or {}).get("sworn_sha256")
    same_build = sealed_build == live_build
    if not same_build:
        advisory.append("INSTRUMENT SKEW: the receipt was issued by styxx.sworn "
                        f"{str(sealed_build)[:12]} and this is {str(live_build)[:12]}")
    if _sworn_core_sha256(live_rec) != _sworn_core_sha256(rec):
        problems.append(
            "the verdict core does not re-derive from the sealed bytes"
            + (" — and the instrument moved, so this is SKEW, not tamper" if not same_build
               else " under the same build, which is tamper")
            + f" (live {live['document_verdict']} {live['counts']} vs sealed "
              f"{rec.get('document_verdict')} {rec.get('counts')})")
    if payload.get("core_sha256") != _sworn_core_sha256(rec):
        problems.append("the sealed core_sha256 is not the receipt's own (tamper) — layer 1 "
                        "compares against it")

    out["ok"] = not problems
    out["same_build"] = same_build
    out["core_sha256"] = payload.get("core_sha256")
    return out


def _render_html_sworn(payload: dict, js_source: str) -> str:
    """The sworn capsule page. Layer 1 loads the sealed verifier and re-derives the portable core
    in the reader's browser; nothing here is a claim that the file is honest."""
    body = json.dumps(payload, indent=1).replace("<", "\\u003c")
    doc_name = (payload.get("document") or {}).get("name", "document")
    rec = payload.get("receipt") or {}
    counts = rec.get("counts") or {}
    return _SWORN_HTML.replace("__PAYLOAD__", body).replace("__JS__", js_source) \
        .replace("__DOCNAME__", doc_name) \
        .replace("__VERDICT__", str(rec.get("document_verdict"))) \
        .replace("__COUNTS__", " ".join(f"{k.lower()}={v}" for k, v in counts.items())) \
        .replace("__LABEL__", _SWORN_LABEL) \
        .replace("__PIP__", (payload.get("verifier") or {}).get("pip", "styxx"))


_SWORN_HTML = """<!doctype html>
<meta charset="utf-8">
<title>sworn capsule — __DOCNAME__</title>
<style>
 body{font:14px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;margin:2rem auto;max-width:52rem;
      color:#111;background:#fff}
 h1{font-size:1.1rem} .k{color:#555} pre{white-space:pre-wrap;word-break:break-word}
 .box{border:1px solid #ccc;padding:.8rem 1rem;margin:1rem 0}
 .ok{border-color:#0a0} .bad{border-color:#c00} .note{color:#555;font-size:.92em}
 code{background:#f4f4f4;padding:0 .2em}
</style>
<h1>sworn capsule — __DOCNAME__</h1>
<p class="k">sealed verdict <b>__VERDICT__</b> &middot; __COUNTS__</p>
<div class="box note"><b>What this page is.</b> __LABEL__<br>
Layer 1 below re-derives the verdict core from the sealed bytes, in your browser, with no network.
Layer 2 is <code>python -m styxx.capsule verify THIS_FILE</code> after <code>__PIP__</code>, and it
is the one that checks the build the receipt names.</div>
<div id="layer1" class="box">layer 1: running…</div>
<h2 style="font-size:1rem">the document</h2>
<pre id="doc" class="box"></pre>
<script type="application/json" id="oath-capsule">__PAYLOAD__</script>
<script>__JS__</script>
<script>
(function () {
  const api = globalThis.swornVerifyApi;
  const el = document.getElementById("layer1");
  const P = JSON.parse(document.getElementById("oath-capsule").textContent);
  const b64 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
  const lines = [];
  let ok = true;
  try {
    const doc = b64(P.document.b64);
    document.getElementById("doc").textContent = new TextDecoder().decode(doc);
    const jsBytes = b64(P.verifier_js.b64);
    const jsOk = api.sha256Bytes(jsBytes) === P.verifier_js.sha256;
    lines.push((jsOk ? "OK  " : "BAD ") + "the sealed verifier hashes to its sealed digest");
    ok = ok && jsOk;
    const docOk = api.sha256Bytes(doc) === P.document.sha256;
    lines.push((docOk ? "OK  " : "BAD ") + "the document hashes to its sealed digest");
    ok = ok && docOk;
    const man = P.manifest === null || P.manifest === undefined ? null
              : api.jsonPlain(JSON.stringify(P.manifest));
    const core = api.swornVerify(doc, man,
      { name: P.receipt.document.name, commit: P.receipt.commit });
    const got = api.coreDigest(core);
    const coreOk = got === P.core_sha256;
    lines.push((coreOk ? "OK  " : "BAD ") + "the verdict core re-derives here: " +
               core.document_verdict + " " + JSON.stringify(core.counts));
    ok = ok && coreOk;
    if (!coreOk) lines.push("     sealed " + P.core_sha256 + "\n     here   " + got);
  } catch (e) {
    ok = false;
    lines.push("BAD the browser verifier raised: " + e);
  }
  el.className = "box " + (ok ? "ok" : "bad");
  el.innerHTML = "<b>layer 1 — " + (ok ? "re-derived in this browser" : "DID NOT re-derive") +
    "</b><pre>" + lines.join("\n").replace(/[&<>]/g, c =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])) + "</pre>" +
    "<span class=note>A forger controlling this whole file controls this layer too. " +
    "Layer 2 is the check.</span>";
})();
</script>
"""


# What `verify` must never hand a terminal raw: C0 controls (escape, carriage return, a line feed
# inside a value), DEL and the C1 controls, the line and paragraph separators, and the
# bidirectional marks and overrides. Every one of them can come from the capsule (a document or
# receipt name, the verdict, the issuer's hash, a binding path), and until review round 3 they
# were printed as they came, so a forgery could move the cursor up and erase its own NOT CHECKED
# lines or failure list from the reader's screen while stdout still held them.
_UNPRINTABLE = re.compile("[\x00-\x1f\x7f-\x9f؜‎‏  ‪-‮"
                          "⁦-⁩]")


def _printable(line) -> str:
    """The line with each character above shown as a visible escape (\\x1b, \\u202e)."""
    def esc(m):
        c = ord(m.group(0))
        return f"\\x{c:02x}" if c < 0x100 else f"\\u{c:04x}"
    return _UNPRINTABLE.sub(esc, str(line))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="styxx.capsule",
        description="Proof-carrying documents (v0.1) and agent-handoff capsules (v0.2)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create", help="mint a capsule: DOC RECEIPTS... --cert CERT "
                                      "(v0.1) | SUMMARY DIFF [--gate GATE] (v0.2)")
    c.add_argument("document", help="the document (v0.1) or the agent summary (v0.2)")
    c.add_argument("inputs", nargs="+",
                   help="receipts (v0.1, with --cert) or the unified diff (v0.2)")
    c.add_argument("--cert", default=None, help="certificate JSON — selects v0.1")
    c.add_argument("--gate", default=None,
                   help="optional diffgate GATE.json to cross-check (v0.2)")
    c.add_argument("--sworn-receipt", default=None,
                   help="a styxx.sworn verdict receipt — selects the sworn profile; the "
                        "positionals are then DOC and (optionally) the manifest")
    c.add_argument("--out", default=None)
    v = sub.add_parser("verify", help="layer-2: re-run the real instrument on a capsule")
    v.add_argument("capsule")
    a = ap.parse_args(argv)

    if a.cmd == "create":
        if sum(bool(x) for x in (a.cert, a.gate, a.sworn_receipt)) > 1:
            ap.error("--cert (v0.1), --gate (v0.2) and --sworn-receipt (sworn) are exclusive")
        out = Path(a.out) if a.out else Path(a.document).with_suffix(".capsule.html")
        if a.sworn_receipt:
            man = Path(a.inputs[0]) if a.inputs and a.inputs[0] != "-" else None
            p = create_capsule_sworn(Path(a.document), man, Path(a.sworn_receipt), out)
            print(f"capsule minted -> {p}")
            return 0
        if a.cert:
            p = create_capsule(Path(a.document), [Path(r) for r in a.inputs],
                               Path(a.cert), out)
        else:
            if len(a.inputs) != 1:
                ap.error("v0.2 mint takes exactly two positionals: SUMMARY DIFF "
                         "(use --cert for a v0.1 document capsule)")
            p = create_capsule_diffgate(Path(a.document), Path(a.inputs[0]), out,
                                        Path(a.gate) if a.gate else None)
        print(f"capsule minted -> {p}")
        return 0

    rep = verify_capsule(Path(a.capsule))
    say = lambda line: print(_printable(line))  # noqa: E731 - every line, whatever it quotes
    if rep.get("spec") == SPEC_SWORN:
        say(f"capsule: {rep.get('document')}  spec {rep.get('spec')}")
        say(f"sealed verdict: {rep.get('verdict')}  counts {rep.get('counts')}")
        for adv in rep.get("advisory", []):
            say(f"  advisory: {adv}")
        if rep["ok"]:
            say("VERIFIED: the sealed bytes re-derive the sealed verdict core at the installed "
                "instrument.")
            say(f"  {rep.get('label')}")
            return 0
        say("CAPSULE FAILS VERIFICATION:")
        for p_ in rep.get("problems", []):
            say(f"  - {p_}")
        return 1
    if rep.get("spec") == SPEC_V02:
        say(f"capsule: {rep.get('summary')} + {rep.get('diff')}  spec {rep.get('spec')}")
        say(f"embedded gate verdict: {rep.get('verdict')}")
        for adv in rep.get("advisory", []):
            say(f"  advisory: {adv}")
        if rep["ok"]:
            say("VERIFIED: bytes match their bindings and the gate record re-derives "
                "at the installed instrument.")
            return 0
    else:
        say(f"capsule: {rep.get('document')}  spec {rep.get('spec')}")
        say(f"embedded verdict: {rep.get('verdict')}  counts {rep.get('counts')}")
        if rep.get("live_verdict") is not None and rep.get("live_verdict") != rep.get("verdict"):
            say(f"installed verifier's verdict: {rep.get('live_verdict')}")
        for adv in rep.get("advisory", []):
            say(f"  advisory: {adv}")
        for nc in rep.get("not_checked", []):
            say(f"  NOT CHECKED: {nc}")
        for st in rep.get("stated", []):
            say(f"  stated by the minter, not checked: {st}")
        if rep["ok"]:
            nc = len(rep.get("not_checked", []))
            say("VERIFIED: the embedded bytes match the certificate's hashes, and the "
                "installed verifier re-derives what the certificate says of them: "
                + ", ".join(rep.get("compared", [])) + "."
                + (f" {nc} item(s) NOT CHECKED, listed above." if nc else ""))
            return 0
    say("CAPSULE FAILS VERIFICATION:")
    for p_ in rep.get("problems", []):
        say(f"  - {p_}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
