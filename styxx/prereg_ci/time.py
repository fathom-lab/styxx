# -*- coding: utf-8 -*-
"""styxx.prereg_ci.time: the time ladder (DESIGN 3.5, 3.6; step B6).

What a seal's time can show, rung by rung, each read from committed objects and printed; only T0
and, under a policy, T3a decide a FAIL (DESIGN 3.5, the table):

    T0  NOT_ORDERED   the order commit is not a strict ancestor of R, the receipt's commit (C3 fails)
    T1  AUTHOR_ORDER  it is; the author set the commits and their dates, so this shows order, not time
    T2  SERVER_ORDER  GitHub's merge times; v0 reads no server, so this prints "not read" (PLAN K6)
    T3a TSA_TIME      an RFC 3161 token committed beside the sealed file (the files a manifest line's
                      ``time`` lists, then ``<path>.tsr``) verifies against the seal bytes and the
                      certificates committed under ``prereg-ci/tsa/`` at the rev the caller names
                      (``trust_rev``: the base side in a check, so a branch cannot commit the
                      certificate that verifies its own token), through
                      ``openssl ts -verify -data SEAL -in TOKEN -CAfile TRUST -attime GENTIME``, and
                      its time precedes R's. ``-attime`` is the token's own genTime, so the same
                      committed objects read the same way whenever they are read (PLAN K5); without it
                      openssl would check the chain at the wall clock and the reading would move the
                      day the pinned certificate expires. The rung trusts the TSA's clock.
    T3b OTS_PROOF     an OpenTimestamps proof (``<path>.ots``) parsed offline: its magic, its digest
                      against the seal bytes, and the attestations it carries. A Bitcoin attestation is
                      the proof's claim, which only a Bitcoin node verifies; no client is run here, and a
                      pending proof shows nothing yet.
    T3c STYXX_CLOCK   lines of ``styxx.clock``'s anchors format about the seal (``papers/charon/
                      anchors.jsonl``, or a ``.jsonl`` the manifest line lists), read for shape alone
                      through ``research_capsule.check_anchor_offline`` and never on chain (K6).

Policies (DESIGN 3.5). ``require_time`` is T1 (the default) or T3a; under T3a a seal read below it is
``FAIL TIME_BELOW_POLICY``, and ``openssl`` absent while a token waits to be verified is
``UNCHECKABLE OPENSSL_ABSENT`` (a NOTE under T1). ``preregistered_word``: a RESULT that calls itself
preregistered while its seal reads below T3a is ``NOTE PREREGISTERED_ON_AUTHOR_ORDER``, a FAIL under
``"fail"``; reading the RESULT's title is the report's (step B7), this module takes the fact.
``close_after`` (3.6) is read against the receipt's time: T3a when a token committed beside the
receipt verifies (a FAIL under ``require_time`` T3a), else T1, printed and never deciding (RED_TEAM M7).

``stamp`` (DESIGN 3.3) is the one networked command: it sends the sealed blob's sha256 to an RFC 3161
TSA, or the blob to the OpenTimestamps client, which sends its digest to the calendars, and writes the
token beside the file. Without ``go`` it refuses: it then opens no socket and writes nothing, and this
module imports its one URL module only inside the ``go`` branch (``import styxx`` itself loads
``http.server`` for ``styxx.dashboard``, so a process running this module has network modules loaded
whether or not it stamps); with ``go`` it is the operator's act (DESIGN 9). It stamps the blob at a commit,
never the working copy, so a CRLF checkout cannot move the digest; a token that exists is never
rewritten.

The token is read without openssl: a small DER reader finds TSTInfo (RFC 3161 section 2.4.2) inside
the response or the bare token, so the imprint, the time, the policy, the serial and the TSA's name
print whether or not openssl is on PATH, and a tampered seal byte is seen in the imprint before any
signature is checked; openssl verifies the signature and the certificate chain. The OpenTimestamps
reader follows the proof serialisation of python-opentimestamps, the library under the
opentimestamps-client (magic, version 1, the file hash op and digest, then ops with 0xff forks and
0x00 attestations), as this build recalled it: neither its source nor a real proof was fetched in
this build (that would be a download), so the tests pin proofs serialised by hand.

Nothing here is public API (DESIGN 3.1). The module is named as the design names it; inside the
package every import is absolute, so the standard library's ``time`` is reached as ``time`` from every
module, and nothing puts the package directory itself on ``sys.path`` (PLAN K16).
"""
from __future__ import annotations

import calendar
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from styxx.prereg_ci import git as _git
from styxx.research_capsule import check_anchor_offline as _check_anchor_offline

_RUNGS = ("T0", "T1", "T2", "T3a", "T3b", "T3c")
_RUNG_NAME = {"T0": "NOT_ORDERED", "T1": "AUTHOR_ORDER", "T2": "SERVER_ORDER", "T3a": "TSA_TIME",
              "T3b": "OTS_PROOF", "T3c": "STYXX_CLOCK"}
_REQUIRE = ("T1", "T3a")                  # require_time (DESIGN 3.5): the default, and the one outside rung
_WORD_POLICIES = ("note", "fail")         # preregistered_word
_TSA_DIR = "prereg-ci/tsa"                # the pinned TSA certificates (DESIGN 3.5, T3a)
_TRUST_EXTS = (".pem", ".crt")
_ANCHORS = "papers/charon/anchors.jsonl"  # styxx.clock's anchors file (none exists in the lab's tree today)
_TOKEN_KIND = {".tsr": "rfc3161", ".ots": "ots", ".jsonl": "anchors"}
_TOKEN_MAX_BYTES = 1 << 20                # a token over this is listed unread (read at call time)
_OPENSSL_TIMEOUT_S = 60
_HTTP_TIMEOUT_S = 30
_OTS_CLIENT_TIMEOUT_S = 60
# a TSA URL: http or https, a host (a name or a bracketed IPv6 address), an optional port, then a path,
# query or fragment; no user information, no whitespace (RFC 3161 section 3.4, the protocol via HTTP)
_TSA_URL = re.compile(r"(?i:https?)://(?:[A-Za-z0-9._~-]+|\[[0-9A-Fa-f:.]+\])(?::[0-9]{1,5})?(?:[/?#][^\s]*)?")
_UNSET = object()

_TIME_BELOW_POLICY = "TIME_BELOW_POLICY"
_TSA_TOKEN_UNVERIFIED = "TSA_TOKEN_UNVERIFIED"
_TSA_TRUST_ABSENT = "TSA_TRUST_ABSENT"
_OPENSSL_ABSENT = "OPENSSL_ABSENT"
_TIME_TOKEN_UNREAD = "TIME_TOKEN_UNREAD"
_PREREGISTERED_ON_AUTHOR_ORDER = "PREREGISTERED_ON_AUTHOR_ORDER"
_RECEIPT_AFTER_CLOSE = "RECEIPT_AFTER_CLOSE"
# the kind each code has on its own; _findings lifts OPENSSL_ABSENT to a NOTE under require_time T1,
# PREREGISTERED_ON_AUTHOR_ORDER to a FAIL under preregistered_word "fail", and RECEIPT_AFTER_CLOSE to a
# FAIL when the receipt's time is read at T3a under require_time T3a
_KIND = {_TIME_BELOW_POLICY: "FAIL", _TSA_TOKEN_UNVERIFIED: "NOTE", _TSA_TRUST_ABSENT: "NOTE",
         _OPENSSL_ABSENT: "UNCHECKABLE", _TIME_TOKEN_UNREAD: "NOTE", _PREREGISTERED_ON_AUTHOR_ORDER: "NOTE",
         _RECEIPT_AFTER_CLOSE: "NOTE"}


# ---- time as text -------------------------------------------------------------------------------------------

_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z")


def _iso(epoch: int) -> str:
    """Seconds since the epoch as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.datetime.fromtimestamp(int(epoch), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch(text: Any) -> Optional[int]:
    """``YYYY-MM-DDTHH:MM:SSZ`` as seconds since the epoch, or None for anything else."""
    m = _ISO.fullmatch(text) if isinstance(text, str) else None
    if m is None:
        return None
    try:
        return calendar.timegm(datetime.datetime(*(int(x) for x in m.groups())).utctimetuple())
    except (ValueError, OverflowError):
        return None


# ---- the RFC 3161 token, read without openssl -------------------------------------------------------------

class _TokenError(ValueError):
    """Bytes that are not an RFC 3161 response or token, with why."""


_SEQ, _SET, _INT, _OID, _OCTETS, _GENTIME = 0x30, 0x31, 0x02, 0x06, 0x04, 0x18
_CTX0 = 0xA0
# An INTEGER or an OID arc longer than these is refused as malformed. RFC 3161 serials run to 160 bits and a
# 2.25 UUID arc to 19 bytes, so a token from a TSA stays under them; past them Python would not print the
# number (no int past 4300 digits by default) and an arc of a megabyte would take minutes to read.
_INT_MAX_BYTES = 128
_ARC_MAX_BYTES = 20
_OID_SIGNED_DATA = "1.2.840.113549.1.7.2"
_OID_TSTINFO = "1.2.840.113549.1.9.16.1.4"
_HASH_OIDS = {"1.3.14.3.2.26": "sha1", "2.16.840.1.101.3.4.2.4": "sha224", "2.16.840.1.101.3.4.2.1": "sha256",
              "2.16.840.1.101.3.4.2.2": "sha384", "2.16.840.1.101.3.4.2.3": "sha512"}
_STATUS = {0: "granted", 1: "granted with modifications", 2: "rejection", 3: "waiting",
           4: "revocation warning", 5: "revocation notification"}
_NAME_OIDS = {"2.5.4.3": "CN", "2.5.4.6": "C", "2.5.4.7": "L", "2.5.4.8": "ST", "2.5.4.10": "O", "2.5.4.11": "OU",
              "2.5.4.5": "serialNumber", "1.2.840.113549.1.9.1": "emailAddress", "0.9.2342.19200300.100.1.25": "DC"}
_STRINGS = {0x0C: "utf-8", 0x13: "ascii", 0x16: "ascii", 0x1A: "ascii", 0x14: "latin-1", 0x1E: "utf-16-be"}


def _tlv(data: bytes, pos: int, end: Optional[int] = None) -> Tuple[int, int, int]:
    """(tag, start of the value, end of the value) of the DER element at *pos*; refused where it runs
    past *end*, the enclosing element's end, or past the bytes."""
    end = len(data) if end is None else end
    if pos + 2 > end:
        raise _TokenError(f"truncated: an element header at byte {pos} runs past the end")
    tag, n = data[pos], data[pos + 1]
    if tag & 0x1F == 0x1F:
        raise _TokenError(f"a multi-byte tag at byte {pos}, which this format does not use")
    start = pos + 2
    if n & 0x80:
        k = n & 0x7F
        if k == 0 or k > 4:
            raise _TokenError(f"an indefinite or oversized length at byte {pos}")
        if start + k > end:
            raise _TokenError(f"truncated: a length at byte {pos} runs past the end")
        n = int.from_bytes(data[start:start + k], "big")
        start += k
    if start + n > end:
        raise _TokenError(f"truncated: the element at byte {pos} is {n} bytes long and the bytes end before it")
    return tag, start, start + n


def _children(data: bytes, start: int, end: int) -> List[Tuple[int, int, int]]:
    out = []
    pos = start
    while pos < end:
        child = _tlv(data, pos, end)
        out.append(child)
        pos = child[2]
    return out


def _expect(child: Tuple[int, int, int], tag: int, what: str) -> Tuple[int, int]:
    if child[0] != tag:
        raise _TokenError(f"{what} is not the element the format names (tag {child[0]:#04x}, not {tag:#04x})")
    return child[1], child[2]


def _oid(data: bytes, s: int, e: int) -> str:
    if s >= e:
        raise _TokenError("an empty object identifier")
    if data[e - 1] & 0x80:                                  # the last byte says an arc goes on: none ends here
        raise _TokenError("an object identifier cut mid-arc (its last byte continues an arc)")
    arcs: List[int] = []
    val = width = 0
    for b in data[s:e]:
        val = (val << 7) | (b & 0x7F)
        width += 1
        if width > _ARC_MAX_BYTES:
            raise _TokenError(f"an object identifier arc of over {_ARC_MAX_BYTES} bytes, past what this reader reads")
        if not b & 0x80:
            arcs.append(val)
            val = width = 0
    lead = arcs[0]
    head = [0, lead] if lead < 40 else [1, lead - 40] if lead < 80 else [2, lead - 80]
    return ".".join(str(a) for a in head + arcs[1:])


def _int(data: bytes, s: int, e: int) -> int:
    if s >= e:
        raise _TokenError("an empty integer")
    if e - s > _INT_MAX_BYTES:
        raise _TokenError(f"an integer of {e - s} bytes, over the {_INT_MAX_BYTES} this reader reads")
    return int.from_bytes(data[s:e], "big", signed=True)


def _gentime(text: str) -> int:
    """A GeneralizedTime in UTC (``YYYYMMDDHHMMSS[.fff]Z``, as DER writes it) as seconds since the epoch."""
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(?:\.\d+)?Z", text)
    if m is None:
        raise _TokenError(f"genTime {text!r} is not a GeneralizedTime in UTC")
    try:
        return calendar.timegm(datetime.datetime(*(int(x) for x in m.groups())).utctimetuple())
    except (ValueError, OverflowError) as e:
        raise _TokenError(f"genTime {text!r}: {e}")


def _text(data: bytes, child: Tuple[int, int, int]) -> str:
    enc = _STRINGS.get(child[0])
    return data[child[1]:child[2]].decode(enc, "replace") if enc else data[child[1]:child[2]].hex()


def _name(data: bytes, s: int, e: int) -> str:
    """An X.501 Name as openssl prints it: ``/CN=.../O=...``."""
    parts = []
    for rdn in _children(data, s, e):
        for atv in _children(data, *_expect(rdn, _SET, "a name's relative distinguished name")):
            kids = _children(data, *_expect(atv, _SEQ, "an attribute"))
            if len(kids) != 2:
                raise _TokenError("an attribute is not a type and a value")
            oid = _oid(data, *_expect(kids[0], _OID, "an attribute type"))
            parts.append(f"{_NAME_OIDS.get(oid, oid)}={_text(data, kids[1])}")
    return "/" + "/".join(parts)


def _general_name(data: bytes, child: Tuple[int, int, int]) -> str:
    n = child[0] & 0x1F
    if n == 4:                                              # directoryName [4] EXPLICIT Name
        kids = _children(data, child[1], child[2])
        if len(kids) != 1:
            raise _TokenError("directoryName holds more than one Name")
        return _name(data, *_expect(kids[0], _SEQ, "directoryName"))
    if n in (1, 2, 6):                                      # rfc822Name, dNSName, uniformResourceIdentifier
        return data[child[1]:child[2]].decode("ascii", "replace")
    return f"[{n}] {data[child[1]:child[2]].hex()}"


def _tstinfo(token: Any) -> Dict[str, Any]:
    """What an RFC 3161 TimeStampResp (``.tsr``) or a bare TimeStampToken says, read from its DER alone:
    the form, the response status, and TSTInfo's version, policy, message imprint (algorithm and
    digest), serial, genTime, nonce and TSA name. A rejection carries no token, so its TSTInfo fields
    are None. Anything else is ``_TokenError`` with why."""
    if not isinstance(token, (bytes, bytearray)):
        raise _TokenError(f"not bytes (a {type(token).__name__})")
    data = bytes(token)
    out: Dict[str, Any] = {"form": None, "status": None, "status_text": None, "status_detail": None, "version": None,
                           "policy": None, "imprint_alg_oid": None, "imprint_alg": None, "imprint": None,
                           "serial": None, "gen_time": None, "gen_time_epoch": None, "accuracy_seconds": None,
                           "nonce": None, "tsa": None}
    tag, s, e = _tlv(data, 0)
    if tag != _SEQ:
        raise _TokenError("does not open with a SEQUENCE: neither a TimeStampResp nor a ContentInfo")
    if e != len(data):
        raise _TokenError(f"{len(data) - e} byte(s) follow the outer element")
    kids = _children(data, s, e)
    if not kids:
        raise _TokenError("an empty outer SEQUENCE")
    if kids[0][0] == _SEQ:                                  # TimeStampResp: PKIStatusInfo, then the token
        out["form"] = "response"
        st = _children(data, kids[0][1], kids[0][2])
        if not st:
            raise _TokenError("PKIStatusInfo holds no status")
        out["status"] = _int(data, *_expect(st[0], _INT, "the PKI status"))
        out["status_text"] = _STATUS.get(out["status"], f"status {out['status']}")
        if len(st) > 1 and st[1][0] == _SEQ:
            texts = [_text(data, c) for c in _children(data, st[1][1], st[1][2])]
            out["status_detail"] = "; ".join(texts) or None
        if len(kids) == 1:
            return out                                      # a rejection carries no token
        if len(kids) != 2:
            raise _TokenError("TimeStampResp holds more than a status and a token")
        info = kids[1]
    elif kids[0][0] == _OID:
        out["form"] = "token"
        info = (_SEQ, s, e)
    else:
        raise _TokenError("neither a TimeStampResp nor a ContentInfo")
    ci = _children(data, *_expect(info, _SEQ, "ContentInfo"))
    if len(ci) != 2 or _oid(data, *_expect(ci[0], _OID, "contentType")) != _OID_SIGNED_DATA:
        raise _TokenError("the token is not CMS SignedData")
    exp = _children(data, *_expect(ci[1], _CTX0, "the signed content"))
    if len(exp) != 1:
        raise _TokenError("the signed content is not one SignedData")
    sd = _children(data, *_expect(exp[0], _SEQ, "SignedData"))
    if len(sd) < 3:
        raise _TokenError("SignedData lacks its encapsulated content")
    eci = _children(data, *_expect(sd[2], _SEQ, "encapContentInfo"))
    if len(eci) != 2 or _oid(data, *_expect(eci[0], _OID, "eContentType")) != _OID_TSTINFO:
        raise _TokenError("the signed content is not TSTInfo")
    econt = _children(data, *_expect(eci[1], _CTX0, "eContent"))
    if len(econt) != 1:
        raise _TokenError("eContent is not one OCTET STRING")
    s0, e0 = _expect(econt[0], _OCTETS, "eContent")
    tag, s1, e1 = _tlv(data, s0, e0)
    if tag != _SEQ or e1 != e0:
        raise _TokenError("TSTInfo is not one SEQUENCE")
    fields = _children(data, s1, e1)
    if len(fields) < 5:
        raise _TokenError("TSTInfo lacks its five required fields")
    out["version"] = _int(data, *_expect(fields[0], _INT, "the TSTInfo version"))
    out["policy"] = _oid(data, *_expect(fields[1], _OID, "the policy"))
    mi = _children(data, *_expect(fields[2], _SEQ, "messageImprint"))
    if len(mi) != 2:
        raise _TokenError("messageImprint is not an algorithm and a digest")
    alg = _children(data, *_expect(mi[0], _SEQ, "the hash algorithm"))
    if not alg:
        raise _TokenError("the hash algorithm names no identifier")
    out["imprint_alg_oid"] = _oid(data, *_expect(alg[0], _OID, "the hash algorithm"))
    out["imprint_alg"] = _HASH_OIDS.get(out["imprint_alg_oid"])
    s2, e2 = _expect(mi[1], _OCTETS, "hashedMessage")
    out["imprint"] = data[s2:e2].hex()
    out["serial"] = _int(data, *_expect(fields[3], _INT, "serialNumber"))
    s3, e3 = _expect(fields[4], _GENTIME, "genTime")
    out["gen_time_epoch"] = _gentime(data[s3:e3].decode("ascii", "replace"))
    out["gen_time"] = _iso(out["gen_time_epoch"])
    for f in fields[5:]:                                    # accuracy, ordering, nonce, tsa, extensions
        if f[0] == _INT:
            out["nonce"] = _int(data, f[1], f[2])
        elif f[0] == _SEQ:
            acc = _children(data, f[1], f[2])
            if acc and acc[0][0] == _INT:
                out["accuracy_seconds"] = _int(data, acc[0][1], acc[0][2])
        elif f[0] == _CTX0:
            kids2 = _children(data, f[1], f[2])
            if len(kids2) == 1:
                out["tsa"] = _general_name(data, kids2[0])
    return out


# ---- offline verification through openssl ------------------------------------------------------------------

def _openssl() -> Optional[str]:
    """The ``openssl`` on PATH, or None."""
    return shutil.which("openssl")


def _openssl_version(exe: str) -> Optional[str]:
    try:
        r = subprocess.run([exe, "version"], stdin=subprocess.DEVNULL, capture_output=True, timeout=_OPENSSL_TIMEOUT_S)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    lines = r.stdout.decode("utf-8", "replace").strip().splitlines()
    return lines[0].strip() if r.returncode == 0 and lines else None


def _verify_argv(exe: str, data_file: str, token_file: str, trust_file: str, at: int, form: str) -> List[str]:
    """DESIGN 3.5's command, with the verification time fixed (PLAN K5) and ``-token_in`` for a bare token."""
    argv = [exe, "ts", "-verify", "-data", data_file, "-in", token_file, "-CAfile", trust_file, "-attime", str(int(at))]
    if form == "token":
        argv.append("-token_in")
    return argv


_ERR = re.compile(r"error:[0-9A-Fa-f]+:[^:]*:[^:]*:(.*)$")
_SRC = re.compile(r"[A-Za-z0-9_./\\-]+\.c:\d+:")


def _openssl_reason(stdout: bytes, stderr: bytes) -> str:
    """The reasons openssl gave, stripped of their error codes and source positions."""
    reasons: List[str] = []
    for ln in stderr.decode("utf-8", "replace").splitlines():
        if "error" not in ln.lower():
            continue
        m = _ERR.search(ln)
        txt = _SRC.sub("", m.group(1) if m else ln).strip(": ")
        if txt and txt not in reasons:
            reasons.append(txt)
    if reasons:
        return "; ".join(reasons)[:400]
    out = stdout.decode("utf-8", "replace").strip().splitlines()
    return (out[-1] if out else "openssl gave no reason")[:400]


def _rfc3161(seal: bytes, token: bytes, trust: Optional[bytes], *, exe: Any = _UNSET, at: Optional[int] = None,
             timeout: int = _OPENSSL_TIMEOUT_S) -> Dict[str, Any]:
    """An RFC 3161 token read against *seal*, the sealed bytes, and *trust*, the PEM certificates
    committed under ``prereg-ci/tsa/`` (None where none is). ``status``: VERIFIED (the imprint is the
    seal's digest and ``openssl ts -verify`` says OK at the token's own time, or *at*), FAILED (the
    imprint differs, the response is not granted, or openssl refuses, with its reason), NO_TRUST (no
    certificate to verify against: read, not verified), UNCHECKABLE (openssl absent or not answering),
    MALFORMED (not a token). The TSTInfo fields are read whatever the status allows."""
    out: Dict[str, Any] = {"status": None, "detail": "", "form": None, "imprint_matches": None, "imprint_alg": None,
                           "imprint": None, "gen_time": None, "gen_time_epoch": None, "at": None, "openssl": None,
                           "tsa": None, "policy": None, "serial": None}
    try:
        info = _tstinfo(token)
    except _TokenError as e:
        out.update(status="MALFORMED", detail=f"not an RFC 3161 response or token: {e}")
        return out
    for k in ("form", "imprint_alg", "imprint", "gen_time", "gen_time_epoch", "tsa", "policy", "serial"):
        out[k] = info[k]
    if info["form"] == "response" and info["status"] not in (0, 1):
        why = f" ({info['status_detail']})" if info["status_detail"] else ""
        out.update(status="FAILED", detail=f"the response's status is {info['status_text']}{why}; it carries no token")
        return out
    if info["gen_time_epoch"] is None:
        out.update(status="FAILED", detail="the response carries no token")
        return out
    expected = None
    if info["imprint_alg"] is not None:
        expected = hashlib.new(info["imprint_alg"], seal).hexdigest()
        out["imprint_matches"] = expected == info["imprint"]
    mismatch = (f"message imprint mismatch: the token's {info['imprint_alg']} imprint {info['imprint'][:16]}... is not "
                f"the seal bytes' {(expected or '')[:16]}..." if out["imprint_matches"] is False else "")
    if trust is None:
        out.update(status="FAILED" if mismatch else "NO_TRUST",
                   detail=mismatch or f"no certificate under {_TSA_DIR}/ to verify the TSA's signature against; "
                                      f"the token is read, not verified")
        return out
    exe = _openssl() if exe is _UNSET else exe
    if exe is None:
        out.update(status="FAILED" if mismatch else "UNCHECKABLE",
                   detail=mismatch or "openssl is not on PATH; the token is read, not verified")
        return out
    out["openssl"] = _openssl_version(exe)
    out["at"] = int(info["gen_time_epoch"] if at is None else at)
    with tempfile.TemporaryDirectory(prefix="prereg-ci-ts-") as d:
        files = {}
        for name, data in (("seal.bin", seal), ("token.der", bytes(token)), ("trust.pem", trust)):
            files[name] = os.path.join(d, name)
            with open(files[name], "wb") as f:
                f.write(data)
        argv = _verify_argv(exe, files["seal.bin"], files["token.der"], files["trust.pem"], out["at"], info["form"])
        try:
            r = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            out.update(status="FAILED" if mismatch else "UNCHECKABLE",
                       detail=mismatch or f"openssl ts -verify ran past {timeout} s")
            return out
        except (OSError, ValueError) as e:
            out.update(status="FAILED" if mismatch else "UNCHECKABLE", detail=mismatch or f"openssl could not run: {e}")
            return out
    ok = r.returncode == 0 and b"Verification: OK" in r.stdout
    reason = "Verification: OK" if ok else _openssl_reason(r.stdout, r.stderr)
    if mismatch:
        out.update(status="FAILED", detail=f"{mismatch}; openssl: {reason}")
    elif ok and info["imprint_alg"] is not None:
        out.update(status="VERIFIED", detail=f"openssl ts -verify: OK at {out['at']} ({_iso(out['at'])}, the token's own "
                                             f"time); imprint {info['imprint_alg']} equals the seal bytes' digest")
    elif ok:
        out.update(status="FAILED", detail=f"the imprint algorithm {info['imprint_alg_oid']} is not one this layer "
                                           f"computes, so the imprint is not read against the seal bytes")
    else:
        out.update(status="FAILED", detail=f"openssl ts -verify: {reason}")
    return out


# ---- the OpenTimestamps proof, parsed offline -------------------------------------------------------------

class _ProofError(ValueError):
    """Bytes that are not an OpenTimestamps proof, with why."""


_OTS_MAGIC = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"
_OTS_VERSION = 1
_OTS_HASH = {0x02: ("sha1", 20), 0x03: ("ripemd160", 20), 0x08: ("sha256", 32), 0x67: ("keccak256", 32)}
_OTS_BINARY = (0xF0, 0xF1)                # append, prepend: one varbytes argument
_OTS_UNARY = (0xF2, 0xF3)                 # reverse, hexlify
_OTS_ATTEST = {bytes.fromhex("0588960d73d71901"): "bitcoin", bytes.fromhex("06869a0d73d71b45"): "litecoin",
               bytes.fromhex("30fe8087b5c7ead7"): "ethereum", bytes.fromhex("83dfe30d2ef90c8e"): "pending"}
_OTS_MAX_ARG = 4096
_OTS_MAX_PAYLOAD = 8192
_OTS_MAX_DEPTH = 256


class _OtsReader:
    def __init__(self, data: bytes, pos: int = 0):
        self.data, self.pos = data, pos

    def byte(self) -> int:
        if self.pos >= len(self.data):
            raise _ProofError(f"truncated at byte {self.pos}")
        self.pos += 1
        return self.data[self.pos - 1]

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise _ProofError(f"truncated at byte {self.pos}: {n} bytes wanted, {len(self.data) - self.pos} left")
        self.pos += n
        return self.data[self.pos - n:self.pos]

    def varuint(self) -> int:
        value = shift = 0
        while True:
            b = self.byte()
            value |= (b & 0x7F) << shift
            if not b & 0x80:
                return value
            shift += 7
            if shift > 63:
                raise _ProofError("a varuint over 64 bits")

    def varbytes(self, cap: int) -> bytes:
        n = self.varuint()
        if n > cap:
            raise _ProofError(f"a {n}-byte field over the {cap}-byte cap")
        return self.take(n)


def _ots_walk(rd: _OtsReader, att: Dict[str, Any], depth: int) -> None:
    if depth > _OTS_MAX_DEPTH:
        raise _ProofError(f"a proof nested past {_OTS_MAX_DEPTH} levels")
    tag = rd.byte()
    while tag == 0xFF:                                      # a fork: this branch, then the next tag
        _ots_step(rd, rd.byte(), att, depth)
        tag = rd.byte()
    _ots_step(rd, tag, att, depth)


def _ots_step(rd: _OtsReader, tag: int, att: Dict[str, Any], depth: int) -> None:
    if tag == 0x00:                                         # an attestation: 8-byte tag, varbytes payload
        kind = _OTS_ATTEST.get(rd.take(8))
        payload = _OtsReader(rd.varbytes(_OTS_MAX_PAYLOAD))
        if kind in ("bitcoin", "litecoin", "ethereum"):
            att[kind].append(payload.varuint())
        elif kind == "pending":
            att["pending"].append(payload.varbytes(_OTS_MAX_PAYLOAD).decode("utf-8", "replace"))
        else:
            att["unknown"] += 1
            return
        if payload.pos != len(payload.data):
            raise _ProofError(f"an attestation payload with {len(payload.data) - payload.pos} byte(s) after its value")
        return
    if tag in _OTS_BINARY:
        rd.varbytes(_OTS_MAX_ARG)
    elif tag not in _OTS_UNARY and tag not in _OTS_HASH:
        raise _ProofError(f"an op tag this reader does not know ({tag:#04x})")
    _ots_walk(rd, att, depth + 1)


def _ots(proof: bytes, seal: bytes) -> Dict[str, Any]:
    """An OpenTimestamps proof read offline against *seal*: its size and sha256, whether it opens with
    the proof magic, the file digest's algorithm and value and whether it is the seal bytes' digest,
    and the attestations it carries (block heights per chain, the calendars of pending ones).
    ``status``: ATTESTED (a block header attestation is claimed; a node verifies it, nothing here does),
    PENDING (calendar attestations alone: nothing is shown yet), NO_ATTESTATION, MALFORMED."""
    out: Dict[str, Any] = {"status": None, "detail": "", "size": len(proof), "sha256": hashlib.sha256(proof).hexdigest(),
                           "magic": False, "version": None, "digest_alg": None, "digest": None, "matches_seal": None,
                           "attestations": {"bitcoin": [], "litecoin": [], "ethereum": [], "pending": [], "unknown": 0}}
    if not proof.startswith(_OTS_MAGIC):
        out.update(status="MALFORMED", detail="does not open with the OpenTimestamps proof magic")
        return out
    out["magic"] = True
    rd = _OtsReader(proof, len(_OTS_MAGIC))
    try:
        out["version"] = rd.varuint()
        if out["version"] != _OTS_VERSION:
            raise _ProofError(f"version {out['version']}, not {_OTS_VERSION}")
        tag = rd.byte()
        if tag not in _OTS_HASH:
            raise _ProofError(f"a file hash op this reader does not know ({tag:#04x})")
        alg, n = _OTS_HASH[tag]
        out["digest_alg"], out["digest"] = alg, rd.take(n).hex()
        try:
            out["matches_seal"] = hashlib.new(alg, seal).hexdigest() == out["digest"]
        except ValueError:                                  # keccak256, or a ripemd160 this build lacks
            out["matches_seal"] = None
        _ots_walk(rd, out["attestations"], 0)
        if rd.pos != len(proof):
            raise _ProofError(f"{len(proof) - rd.pos} byte(s) after the proof")
    except _ProofError as e:
        out.update(status="MALFORMED", detail=str(e))
        return out
    att = out["attestations"]
    chains = [f"{k} block {h}" for k in ("bitcoin", "litecoin", "ethereum") for h in att[k]]
    if chains:
        out.update(status="ATTESTED", detail=f"the proof claims {', '.join(chains)}; the claim is not verified here, "
                                             f"a node of that chain reads the block header")
    elif att["pending"]:
        out.update(status="PENDING", detail=f"pending at {', '.join(att['pending'])}: no block attestation yet, so the "
                                            f"proof shows nothing yet; not verified here")
    else:
        out.update(status="NO_ATTESTATION", detail=f"the proof carries no attestation this reader knows "
                                                   f"({att['unknown']} of an unknown kind)")
    return out


# ---- the clock rung, shape only -------------------------------------------------------------------------------

_CHAIN = "not read on chain"


def _anchors(data: bytes, digests: Iterable[str]) -> Dict[str, Any]:
    """Lines of ``styxx.clock``'s anchors format, each about this seal (its digest one of *digests*)
    read through ``research_capsule.check_anchor_offline``: SHAPE_OK or MALFORMED_LINE, and never
    ANCHORED, since nothing here reads the chain. ``n`` is the line's own sequence number as it
    writes it; ``line`` is where it sits in the file."""
    wanted = set(digests)
    out: Dict[str, Any] = {"lines": 0, "malformed": 0, "about_seal": [],
                           "detail": "offline: each line's shape alone, through research_capsule.check_anchor_offline; "
                                     f"{_CHAIN} (styxx.clock.verify reads the chain, and v0 opens no socket)"}
    for n, raw in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
        if not raw.strip():
            continue
        out["lines"] += 1
        try:
            obj = json.loads(raw)
        except ValueError:
            out["malformed"] += 1
            continue
        if not isinstance(obj, dict):
            out["malformed"] += 1
            continue
        if obj.get("digest") not in wanted:
            continue
        reading = dict(_check_anchor_offline(obj))
        reading["line"] = n
        reading["chain"] = _CHAIN
        out["about_seal"].append(reading)
    return out


# ---- tokens, trust and the ladder, read from a repository ----------------------------------------------------

def _kind_of(path: str) -> Optional[str]:
    return _TOKEN_KIND.get(os.path.splitext(path)[1].lower())


def _tokens(repo: Any, rev: str, path: str, line: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The time tokens of the sealed *path* at *rev*: the files a manifest *line*'s ``time`` lists, in
    its order, then ``<path>.tsr`` and ``<path>.ots`` where they are in the tree and the line does not
    name them. The manifest is append-only and a line's ``time`` is written when the line is, so a
    token ``stamp`` writes beside the file later is read without a new line. Each ``{path, kind,
    bytes, sha256, why}``; a path that is not a blob at *rev*, is over ``_TOKEN_MAX_BYTES`` or has an
    extension this layer does not read has ``bytes`` None and ``why``."""
    c = repo.commit(rev)
    named = list(line.get("time") or []) if line is not None else []
    named += [p for p in (path + ".tsr", path + ".ots") if p not in named and repo.blob_id(c, p) is not None]
    out = []
    for p in named:
        t: Dict[str, Any] = {"path": p, "kind": _kind_of(p), "bytes": None, "sha256": None, "why": None}
        if t["kind"] is None:
            t["why"] = f"not a token this layer reads ({', '.join(_TOKEN_KIND)})"
        else:
            try:
                data = repo.read(c, p)
            except _git._Refusal as e:
                data, t["why"] = None, f"{e.code}: {e.detail}"
            if data is None and t["why"] is None:
                t["why"] = f"not in the tree at {c[:12]}"
            elif data is not None and len(data) > _TOKEN_MAX_BYTES:
                t["why"] = f"{len(data)} bytes, over the {_TOKEN_MAX_BYTES} this layer reads"
            elif data is not None:
                t["bytes"], t["sha256"] = data, hashlib.sha256(data).hexdigest()
        out.append(t)
    return out


def _trust(repo: Any, rev: str, trust_dir: str = _TSA_DIR) -> Tuple[Optional[bytes], List[str]]:
    """Every ``.pem`` and ``.crt`` blob under *trust_dir* at *rev*, in path order, joined with a newline
    into one trust store, and the paths; (None, []) where there is none."""
    c = repo.commit(rev)
    paths = sorted(p for p in repo.paths(c) if p.startswith(trust_dir + "/") and p.lower().endswith(_TRUST_EXTS))
    blobs = []
    kept = []
    for p in paths:
        try:
            data = repo.read(c, p)
        except _git._Refusal:
            continue
        if data is not None:
            blobs.append(data)
            kept.append(p)
    return (b"\n".join(blobs) if blobs else None), kept


def _rung(ordered: Optional[bool], t3a: Sequence[Dict[str, Any]]) -> Optional[str]:
    """The highest rung that holds and can be read from committed objects: T0 when the receipt does not
    descend from the order commit, whatever a token says; T3a when a token verifies and its time
    precedes the receipt's (or no receipt is there yet); T1 when the order holds; None with no receipt
    and no token."""
    if ordered is False:
        return "T0"
    if any(t["status"] == "VERIFIED" and t.get("precedes_receipt") is not False for t in t3a):
        return "T3a"
    return "T1" if ordered else None


def _reading(repo: Any, *, path: str, rev: str, seal: bytes, digests: Iterable[str] = (),
             line: Optional[Dict[str, Any]] = None, receipt: Optional[str] = None, order: Optional[str] = None,
             receipt_path: Optional[str] = None, receipt_bytes: Optional[bytes] = None, exe: Any = _UNSET,
             trust_dir: str = _TSA_DIR, trust_rev: Optional[str] = None) -> Dict[str, Any]:
    """The time ladder of the seal of *path* whose bytes are *seal*, read at *rev* (the tree the tokens
    are read from) against *receipt*, the commit R that added the receipt (None before data), and
    *order*, the order commit. The trust store is read at *trust_rev* (*rev* when None): a check
    passes its BASE, so a branch that commits a certificate of its own beside a token from its own
    TSA does not verify it. *digests* are the seal's other digests (the decision digest) an anchor
    line may name. With *receipt_path*, the tokens beside the receipt are read against
    *receipt_bytes* (the blob at *rev* when None) for ``close_after``."""
    c = repo.commit(rev)
    tc = repo.commit(trust_rev) if trust_rev is not None else c
    seal_sha = hashlib.sha256(seal).hexdigest()
    all_digests = [seal_sha] + [d for d in digests if isinstance(d, str) and d and d != seal_sha]
    r = repo.commit(receipt) if receipt is not None else None
    o = repo.commit(order) if order is not None else None
    ordered: Optional[bool] = None
    if r is not None:
        ordered = repo.is_strict_ancestor(o, r) if o is not None else False
    r_at = repo.commit_time(r) if r is not None else None
    exe_path = _openssl() if exe is _UNSET else exe
    version = _openssl_version(exe_path) if exe_path else None
    trust, trust_paths = _trust(repo, tc, trust_dir)
    out: Dict[str, Any] = {
        "path": path, "rev": c, "seal_sha256": seal_sha, "digests": all_digests, "order": o, "receipt": r,
        "ordered": ordered, "receipt_committed_at_epoch": r_at, "receipt_committed_at": _iso(r_at) if r_at else None,
        "T1": {"holds": ordered, "detail": ("no receipt to order against" if r is None else
                                           "the order commit is a strict ancestor of the receipt's commit; the author "
                                           "set both and their dates" if ordered else
                                           "the order commit is not a strict ancestor of the receipt's commit (C3)")},
        "T2": {"read": False, "detail": "not read: v0 consults no server (GitHub's merge times would be SERVER_ORDER)"},
        "T3a": [], "T3b": [], "T3c": {}, "unread": [], "trust": trust_paths, "trust_rev": tc, "openssl": version,
        "receipt_time": {"T3a": [], "T1": _iso(r_at) if r_at else None}, "rung": None, "name": None}
    anchors: List[Tuple[str, bytes]] = []
    for t in _tokens(repo, c, path, line):
        if t["bytes"] is None:
            out["unread"].append({"path": t["path"], "why": t["why"]})
        elif t["kind"] == "rfc3161":
            v = _rfc3161(seal, t["bytes"], trust, exe=exe_path)
            v["path"], v["sha256"] = t["path"], t["sha256"]
            v["precedes_receipt"] = (None if r_at is None or v["gen_time_epoch"] is None
                                     else v["gen_time_epoch"] < r_at)
            out["T3a"].append(v)
        elif t["kind"] == "ots":
            p = _ots(t["bytes"], seal)
            p["path"], p["sha256"] = t["path"], t["sha256"]
            out["T3b"].append(p)
        else:
            anchors.append((t["path"], t["bytes"]))
    if _ANCHORS not in [a[0] for a in anchors] and repo.blob_id(c, _ANCHORS) is not None:
        try:
            data = repo.read(c, _ANCHORS)
        except _git._Refusal as e:
            data = None
            out["unread"].append({"path": _ANCHORS, "why": f"{e.code}: {e.detail}"})
        if data is not None and len(data) <= _TOKEN_MAX_BYTES:
            anchors.append((_ANCHORS, data))
        elif data is not None:
            out["unread"].append({"path": _ANCHORS, "why": f"{len(data)} bytes, over the {_TOKEN_MAX_BYTES} this layer reads"})
    if anchors:
        t3c: Dict[str, Any] = {"files": [], "lines": 0, "malformed": 0, "about_seal": [], "detail": ""}
        for p, data in anchors:
            a = _anchors(data, all_digests)
            t3c["files"].append(p)
            t3c["lines"] += a["lines"]
            t3c["malformed"] += a["malformed"]
            for reading in a["about_seal"]:
                reading["path"] = p
            t3c["about_seal"].extend(a["about_seal"])
            t3c["detail"] = a["detail"]
        out["T3c"] = t3c
    else:
        out["T3c"] = {"files": [], "lines": 0, "malformed": 0, "about_seal": [],
                      "detail": f"no anchors file in the tree at {c[:12]} (none named by the manifest line, no {_ANCHORS})"}
    if receipt_path is not None and r is not None:
        rb = receipt_bytes if receipt_bytes is not None else repo.read(c, receipt_path)
        for t in _tokens(repo, c, receipt_path, None):
            if t["kind"] == "rfc3161" and t["bytes"] is not None and rb is not None:
                v = _rfc3161(rb, t["bytes"], trust, exe=exe_path)
                v["path"], v["sha256"] = t["path"], t["sha256"]
                out["receipt_time"]["T3a"].append(v)
    out["rung"] = _rung(ordered, out["T3a"])
    out["name"] = _RUNG_NAME.get(out["rung"]) if out["rung"] else None
    return out


def _describe(r: Dict[str, Any]) -> str:
    """One line for the report: ``time AUTHOR_ORDER (T2 not read, T3 none)`` (DESIGN 4.A), or the rung
    with its time and each T3 reading."""
    head = f"time {r['name']}" if r["name"] else "time (no receipt to order against, no outside rung)"
    if r["rung"] == "T3a":
        lifted = next(t for t in r["T3a"] if t["status"] == "VERIFIED" and t.get("precedes_receipt") is not False)
        head += f" {lifted['gen_time']}"
    t3c = r["T3c"]
    if not r["T3a"] and not r["T3b"] and not t3c.get("files"):
        return head + " (T2 not read, T3 none)"
    parts = ["T2 not read"]
    for t in r["T3a"]:
        s = f"T3a {t['status'].lower()} {t['path']}"
        if t["status"] == "VERIFIED" and t.get("precedes_receipt") is False:
            s += " (its time does not precede the receipt's)"
        parts.append(s)
    if not r["T3a"]:
        parts.append("T3a none")
    parts.extend(f"T3b {p['status'].lower()} {p['path']}" for p in r["T3b"])
    if not r["T3b"]:
        parts.append("T3b none")
    if t3c.get("files"):
        n = t3c["lines"]
        parts.append(f"T3c {n} line{'s' if n != 1 else ''} in {', '.join(t3c['files'])}, {len(t3c['about_seal'])} about "
                     f"this seal, {_CHAIN}")
    else:
        parts.append("T3c none")
    return head + " (" + "; ".join(parts) + ")"


# ---- the policies ---------------------------------------------------------------------------------------------

def _finding(code: str, detail: str, kind: Optional[str] = None) -> Dict[str, str]:
    return {"code": code, "kind": kind or _KIND[code], "detail": detail}


def _findings(reading: Dict[str, Any], *, require_time: str = "T1", preregistered: bool = False,
              preregistered_word: str = "note", close_after: Optional[str] = None) -> List[Dict[str, str]]:
    """The findings of a time reading under the policies (DESIGN 3.5, 3.6): what each token read, the
    rung against ``require_time``, the preregistered word, and the receipt's time against
    ``close_after`` (``YYYY-MM-DDTHH:MM:SSZ``). A policy value the design does not name is ValueError."""
    if require_time not in _REQUIRE:
        raise ValueError(f"require_time is one of {_REQUIRE}, not {require_time!r}")
    if preregistered_word not in _WORD_POLICIES:
        raise ValueError(f"preregistered_word is one of {_WORD_POLICIES}, not {preregistered_word!r}")
    out: List[Dict[str, str]] = []
    waiting: List[str] = []
    for t in reading.get("T3a", ()):
        if t["status"] in ("FAILED", "MALFORMED"):
            out.append(_finding(_TSA_TOKEN_UNVERIFIED, f"{t['path']}: {t['detail']}"))
        elif t["status"] == "NO_TRUST":
            out.append(_finding(_TSA_TRUST_ABSENT, f"{t['path']}: {t['detail']}"))
        elif t["status"] == "UNCHECKABLE":
            waiting.append(t["path"])
    if waiting:
        out.append(_finding(_OPENSSL_ABSENT, f"openssl is not on PATH (or did not answer), so {', '.join(waiting)} "
                                             f"{'is' if len(waiting) == 1 else 'are'} read and not verified",
                            kind="UNCHECKABLE" if require_time == "T3a" else "NOTE"))
    for u in reading.get("unread", ()):
        out.append(_finding(_TIME_TOKEN_UNREAD, f"{u['path']}: {u['why']}"))
    rung = reading.get("rung")
    name = reading.get("name") or "no rung (no receipt, no token)"
    if require_time == "T3a" and rung != "T3a" and not waiting:
        tokens = reading.get("T3a", ())
        if not tokens:
            why = "no RFC 3161 token is committed beside the sealed file"
        else:
            why = "; ".join(f"{t['path']} {t['status'].lower()}"
                            + (" but its time does not precede the receipt's" if t["status"] == "VERIFIED" else "")
                            for t in tokens)
        out.append(_finding(_TIME_BELOW_POLICY, f"require_time is T3a and the seal reads {name}: {why}"))
    if preregistered and rung != "T3a":
        out.append(_finding(_PREREGISTERED_ON_AUTHOR_ORDER,
                            f"the RESULT calls itself preregistered while its seal reads {name}, below T3a TSA_TIME; "
                            f"only an outside clock would show the plan came before the data",
                            kind="FAIL" if preregistered_word == "fail" else "NOTE"))
    if close_after is not None:
        bound = _epoch(close_after)
        if bound is None:
            raise ValueError(f"close_after {close_after!r} is not YYYY-MM-DDTHH:MM:SSZ")
        rt = reading.get("receipt_time") or {}
        verified = [t["gen_time_epoch"] for t in rt.get("T3a", ()) if t["status"] == "VERIFIED"
                    and t.get("gen_time_epoch") is not None]
        at: Optional[int] = None
        read_at = ""
        if verified:
            at, read_at = min(verified), "T3a"
        elif reading.get("receipt_committed_at_epoch") is not None:
            at, read_at = reading["receipt_committed_at_epoch"], "T1"
        if at is not None and at > bound:
            kind = "FAIL" if read_at == "T3a" and require_time == "T3a" else "NOTE"
            tail = "" if kind == "FAIL" else " (printed, not deciding: it bounds when a receipt may be submitted, " \
                                             "not when its data were seen)"
            out.append(_finding(_RECEIPT_AFTER_CLOSE, f"the receipt's time read at {read_at} ({_iso(at)}) is after "
                                                      f"close_after {close_after}{tail}", kind=kind))
    return out


# ---- stamp: the one networked command --------------------------------------------------------------------------

class _StampError(ValueError):
    """What stamp refuses as an input error (the CLI's exit 2)."""


def _stamp(repo: Any, path: str, *, rfc3161: Optional[str] = None, ots: bool = False, commit: str = "HEAD",
           go: bool = False, exe: Any = _UNSET) -> Dict[str, Any]:
    """Stamp the blob of *path* at *commit*: with *rfc3161*, send its sha256 to that TSA and write
    ``<path>.tsr`` beside it; with *ots*, hand the bytes to the OpenTimestamps client on PATH, which
    sends their digest to its calendars, and write ``<path>.ots``. Without *go* the result is REFUSED
    and says what would be sent where; nothing is written and no socket is opened. A token that exists
    is never rewritten."""
    if bool(rfc3161) == bool(ots):
        raise ValueError("name exactly one of --rfc3161 TSA_URL and --ots")
    if rfc3161:
        _tsa_url(rfc3161)
    c = repo.commit(commit)
    data = repo.read(c, path)
    if data is None:
        raise _StampError(f"{path} is not a file in the tree at {c[:12]}")
    digest = hashlib.sha256(data).hexdigest()
    writes = path + (".tsr" if rfc3161 else ".ots")
    out: Dict[str, Any] = {"status": None, "detail": "", "path": path, "commit": c, "sha256": digest, "bytes": len(data),
                           "writes": writes, "to": rfc3161 if rfc3161 else "the OpenTimestamps calendars, through the "
                                                                          "ots client on PATH"}
    if not go:
        out.update(status="REFUSED", detail=f"stamp sends the sha256 of {path}'s blob at {c[:12]} to {out['to']}, an "
                                            f"outside server, and writes {writes}; pass --go to send it (the "
                                            f"operator's act, DESIGN 9)")
        return out
    target = os.path.join(repo._where(), *writes.split("/"))
    if os.path.exists(target):
        out.update(status="REFUSED", detail=f"{writes} exists already; a token is history and is not rewritten")
        return out
    if rfc3161:
        token, info = _request_rfc3161(data, rfc3161, exe)
        out.update(gen_time=info["gen_time"], tsa=info["tsa"], policy=info["policy"], serial=info["serial"])
        advice = (f"commit {writes} beside {path}, and the TSA's certificate chain under {_TSA_DIR}/, so check "
                  f"verifies it offline; a manifest line may list {writes} under time")
    else:
        token = _request_ots(data, os.path.basename(path))
        advice = f"commit {writes} beside {path}; the proof is pending until a calendar writes it into a block"
    with open(target, "xb") as f:
        f.write(token)
    with open(target, "rb") as f:
        back = f.read()
    if back != token:
        raise _StampError(f"{writes} did not read back as written")
    out.update(status="WRITTEN", detail=f"{len(token)} bytes written to {writes}; {advice}")
    return out


def _tsa_url(url: str) -> None:
    """A TSA is reached over HTTP or HTTPS at a named host (RFC 3161 section 3.4); a ``file:`` or any
    other URL ``urllib`` would also open is refused before anything is sent. Read by a pattern, so
    no URL module is imported on the path that sends nothing."""
    if not isinstance(url, str) or _TSA_URL.fullmatch(url) is None:
        raise _StampError(f"--rfc3161 {url!r} is not an http or https URL with a host")


def _request_rfc3161(data: bytes, url: str, exe: Any = _UNSET) -> Tuple[bytes, Dict[str, Any]]:
    """A TimeStampReq over *data* (openssl ts -query, sha256, with a nonce and the certificate asked
    for), posted to *url*; the response, read and held to the request (granted, the imprint, the
    nonce). The one place this module opens a connection."""
    _tsa_url(url)
    exe = _openssl() if exe is _UNSET else exe
    if exe is None:
        raise _StampError("openssl is not on PATH; the query cannot be made")
    with tempfile.TemporaryDirectory(prefix="prereg-ci-tsq-") as d:
        seal, query = os.path.join(d, "seal.bin"), os.path.join(d, "query.tsq")
        with open(seal, "wb") as f:
            f.write(data)
        try:
            r = subprocess.run([exe, "ts", "-query", "-data", seal, "-sha256", "-cert", "-out", query],
                               stdin=subprocess.DEVNULL, capture_output=True, timeout=_OPENSSL_TIMEOUT_S)
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            raise _StampError(f"openssl ts -query could not run: {e}")
        if r.returncode != 0:
            raise _StampError(f"openssl ts -query: {_openssl_reason(r.stdout, r.stderr)}")
        with open(query, "rb") as f:
            q = f.read()
    nonce = _query_nonce(q)
    import urllib.request as _http                          # the one network import, inside the one networked path
    req = _http.Request(url, data=q, method="POST", headers={"Content-Type": "application/timestamp-query",
                                                             "Accept": "application/timestamp-reply"})
    try:
        with _http.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
            body = resp.read(_TOKEN_MAX_BYTES + 1)
    except (OSError, ValueError) as e:
        raise _StampError(f"the TSA at {url} did not answer: {e}")
    if len(body) > _TOKEN_MAX_BYTES:
        raise _StampError(f"the response is over {_TOKEN_MAX_BYTES} bytes")
    try:
        info = _tstinfo(body)
    except _TokenError as e:
        raise _StampError(f"the TSA answered with bytes that are not a TimeStampResp: {e}")
    if info["form"] != "response" or info["status"] not in (0, 1):
        raise _StampError(f"the TSA did not grant the request: {info['status_text']} ({info['status_detail'] or 'no detail'})")
    if info["imprint_alg"] != "sha256" or info["imprint"] != hashlib.sha256(data).hexdigest():
        raise _StampError("the response's imprint is not the sha256 of the bytes stamped")
    if nonce is not None and info["nonce"] != nonce:
        raise _StampError("the response's nonce is not the request's")
    return body, info


def _query_nonce(query: bytes) -> Optional[int]:
    """The nonce of a TimeStampReq (the INTEGER after the message imprint), or None."""
    try:
        tag, s, e = _tlv(query, 0)
        fields = _children(query, s, e)
        for f in fields[2:]:
            if f[0] == _INT:
                return _int(query, f[1], f[2])
    except _TokenError:
        return None
    return None


def _request_ots(data: bytes, basename: str) -> bytes:
    """The OpenTimestamps client on PATH stamping *data* (it sends the digest to its calendars)."""
    exe = shutil.which("ots")
    if exe is None:
        raise _StampError("the OpenTimestamps client (ots) is not on PATH")
    with tempfile.TemporaryDirectory(prefix="prereg-ci-ots-") as d:
        target = os.path.join(d, basename)
        with open(target, "wb") as f:
            f.write(data)
        try:
            r = subprocess.run([exe, "stamp", target], stdin=subprocess.DEVNULL, capture_output=True,
                               timeout=_OTS_CLIENT_TIMEOUT_S)
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            raise _StampError(f"ots stamp could not run: {e}")
        if r.returncode != 0 or not os.path.exists(target + ".ots"):
            raise _StampError(f"ots stamp: {r.stderr.decode('utf-8', 'replace').strip()[-300:] or 'wrote no proof'}")
        with open(target + ".ots", "rb") as f:
            proof = f.read()
    reading = _ots(proof, data)
    if reading["status"] == "MALFORMED" or reading["matches_seal"] is not True:
        raise _StampError(f"the proof the client wrote does not read as a proof over the bytes stamped: {reading['detail']}")
    return proof
