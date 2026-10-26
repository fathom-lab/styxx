# -*- coding: utf-8 -*-
"""prereg-ci B6: the time ladder (DESIGN 3.5, 3.6), read from committed objects and never from a server.

T0 ``NOT_ORDERED`` and T1 ``AUTHOR_ORDER`` come from git's ancestry (C3) and the dates the author
set; T2 ``SERVER_ORDER`` is never read in v0 and prints so (PLAN K6); T3a ``TSA_TIME`` is an RFC 3161
token committed beside the sealed file, verified offline with ``openssl ts -verify`` against the seal
bytes and a certificate committed under ``prereg-ci/tsa/``, at the token's own time rather than the
wall clock (PLAN K5), and it is the one rung a policy may decide a FAIL on; T3b ``OTS_PROOF`` is an
OpenTimestamps proof parsed offline (presence, size, digest, attestations; no client is run); T3c
``STYXX_CLOCK`` is ``styxx.clock``'s anchors format read for shape only through
``research_capsule.check_anchor_offline``, never on chain (K6). ``stamp`` is the one networked
command (DESIGN 3.3) and refuses without ``--go``; no test here passes ``--go``.

Done when (DESIGN 8, B6; PLAN B6): a token from a test TSA made here in tmp_path with ``openssl``
(a CA, a TSA certificate with critical ``extendedKeyUsage = timeStamping``, ``ts -query`` and
``ts -reply`` from a config written here) verifies offline against the seal bytes materialised from
the blob, at a fixed verification time; one flipped seal byte fails it.

Every repository here is made with ``git init`` in tmp_path; every openssl run is local; during
``stamp`` without ``--go`` a socket or an HTTP open would raise, and none happens.
"""
from __future__ import annotations

import ast
import datetime
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time as _clock
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx import clock as C  # noqa: E402
from styxx.prereg_ci import cli  # noqa: E402
from styxx.prereg_ci import git as G  # noqa: E402
from styxx.prereg_ci import time as T  # noqa: E402
from tests.test_prereg_ci_git import Git, digest, prereg, receipt  # noqa: E402

P = "p/PREREG_a.md"
R = "p/receipt.json"
R_EARLY = "p/receipt_early.json"
SEAL = prereg(0.5).encode("utf-8")
OPENSSL = shutil.which("openssl")
needs_openssl = pytest.mark.skipif(OPENSSL is None, reason="openssl is not on PATH, so no test TSA can be made here")
FAR_FUTURE = 4_102_444_800            # 2100-01-01T00:00:00Z, past the test certificates
FAR_PAST = 946_684_800                # 2000-01-01T00:00:00Z, before them
POLICY_OID = "1.3.6.1.4.1.99999.1"

# The test TSA's configuration: a CA, a TSA certificate with the critical timeStamping purpose, and the
# ts section openssl ts -reply reads (OpenSSL 3.x wants signer_digest named).
CNF = """[ req ]
distinguished_name = dn
prompt = no
[ dn ]
CN = prereg-ci test
[ v3_ca ]
basicConstraints = critical, CA:TRUE
keyUsage = critical, keyCertSign, cRLSign
subjectKeyIdentifier = hash
[ v3_tsa ]
basicConstraints = critical, CA:FALSE
keyUsage = critical, digitalSignature, nonRepudiation
extendedKeyUsage = critical, timeStamping
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid
[ tsa ]
default_tsa = tsa_config1
[ tsa_config1 ]
dir = .
serial = ./tsaserial
signer_cert = ./tsa.pem
signer_key = ./tsa.key
signer_digest = sha256
certs = ./ca.pem
default_policy = %s
other_policies = %s
digests = sha256
accuracy = secs:1
ordering = yes
tsa_name = yes
ess_cert_id_chain = no
ess_cert_id_alg = sha256
""" % (POLICY_OID, POLICY_OID)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Tsa:
    """A time-stamping authority made in a directory with openssl: its own CA, a TSA certificate
    signed by it, and ``stamp``, which answers a query over given bytes with a response (.tsr)."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        (root / "openssl.cnf").write_bytes(CNF.encode("utf-8"))
        self.env = dict(os.environ, OPENSSL_CONF=str(root / "openssl.cnf"))
        self.n = 0
        ec = ("genpkey", "-algorithm", "EC", "-pkeyopt", "ec_paramgen_curve:P-256")
        self.run(*ec, "-out", "ca.key")
        self.run("req", "-x509", "-new", "-key", "ca.key", "-config", "openssl.cnf", "-extensions", "v3_ca",
                 "-days", "3650", "-subj", "/CN=prereg-ci test CA", "-out", "ca.pem")
        self.run(*ec, "-out", "tsa.key")
        self.run("req", "-new", "-key", "tsa.key", "-config", "openssl.cnf", "-subj", "/CN=prereg-ci test TSA",
                 "-out", "tsa.csr")
        self.run("x509", "-req", "-in", "tsa.csr", "-CA", "ca.pem", "-CAkey", "ca.key", "-CAcreateserial",
                 "-days", "3650", "-extfile", "openssl.cnf", "-extensions", "v3_tsa", "-out", "tsa.pem")
        (root / "tsaserial").write_bytes(b"01\n")
        self.ca = (root / "ca.pem").read_bytes()
        self.tsa_pem = (root / "tsa.pem").read_bytes()

    def run(self, *args: str) -> bytes:
        r = subprocess.run([OPENSSL, *args], cwd=str(self.root), env=self.env, stdin=subprocess.DEVNULL,
                           capture_output=True, timeout=120)
        assert r.returncode == 0, (args, r.stderr.decode("utf-8", "replace"))
        return r.stdout

    def stamp(self, data: bytes, token_only: bool = False) -> bytes:
        self.n += 1
        n = self.n
        (self.root / f"d{n}.bin").write_bytes(data)
        self.run("ts", "-query", "-data", f"d{n}.bin", "-sha256", "-cert", "-no_nonce", "-out", f"q{n}.tsq")
        self.run("ts", "-reply", "-config", "openssl.cnf", "-queryfile", f"q{n}.tsq", "-out", f"r{n}.tsr")
        if token_only:
            self.run("ts", "-reply", "-in", f"r{n}.tsr", "-token_out", "-out", f"t{n}.tkn")
            return (self.root / f"t{n}.tkn").read_bytes()
        return (self.root / f"r{n}.tsr").read_bytes()


@pytest.fixture(scope="module")
def tsa(tmp_path_factory):
    if OPENSSL is None:
        pytest.skip("openssl is not on PATH, so no test TSA can be made here")
    return Tsa(tmp_path_factory.mktemp("tsa"))


@pytest.fixture(scope="module")
def token(tsa):
    before = int(_clock.time())
    tok = tsa.stamp(SEAL)
    return SimpleNamespace(bytes=tok, before=before, after=int(_clock.time()) + 1)


# ---- an OpenTimestamps proof, serialised here so the parser is pinned against the format as read -----------

OTS_MAGIC = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"
PENDING, BITCOIN, LITECOIN = "83dfe30d2ef90c8e", "0588960d73d71901", "06869a0d73d71b45"


def varuint(n: int) -> bytes:
    out = b""
    while True:
        b, n = n & 0x7F, n >> 7
        if n:
            out += bytes([b | 0x80])
        else:
            return out + bytes([b])


def varbytes(b: bytes) -> bytes:
    return varuint(len(b)) + b


def attest(tag: str, payload: bytes) -> bytes:
    return b"\x00" + bytes.fromhex(tag) + varbytes(payload)


def ots(file_digest: bytes, *branches: bytes, alg: bytes = b"\x08", magic: bytes = OTS_MAGIC) -> bytes:
    body = b"".join(b"\xff" + br for br in branches[:-1]) + branches[-1]
    return magic + varuint(1) + alg + file_digest + body


PENDING_BRANCH = b"\xf0" + varbytes(b"\x01\x02") + b"\x08" + attest(PENDING, varbytes(b"https://a.calendar.example"))
BITCOIN_BRANCH = b"\xf1" + varbytes(b"\xaa") + b"\x08" + attest(BITCOIN, varuint(812345))


# ---- the module's surface --------------------------------------------------------------------------------------

def test_the_time_module_keeps_every_top_level_name_private_and_the_cli_exports_main_alone():
    """DESIGN 3.1: the public surface is core.rescore, .compare, .licences and the CLI."""
    tree = ast.parse((ROOT / "styxx" / "prereg_ci" / "time.py").read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(t.id for t in targets if isinstance(t, ast.Name))
    assert all(n.startswith("_") for n in names), sorted(n for n in names if not n.startswith("_"))
    assert cli.__all__ == ["main"]
    import styxx.prereg_ci as pkg
    assert set(pkg.__all__) <= {"rescore", "compare", "licences"}


def test_the_module_opens_no_network_module_at_import_and_stamp_alone_names_one():
    """The one networked path is stamp's --go branch: urllib is imported inside it, never at module
    level, and socket, http and requests appear nowhere."""
    src = (ROOT / "styxx" / "prereg_ci" / "time.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            assert not any(m.split(".")[0] in ("urllib", "socket", "http", "requests") for m in mods), ast.dump(node)
    inside = [n for n in ast.walk(tree) if isinstance(n, ast.Import)
              and any(a.name.startswith("urllib") for a in n.names)]
    assert inside, "stamp's --go branch imports urllib where it sends"
    assert "socket" not in src.replace("opens no socket", "").replace("no socket", "")
    assert "import requests" not in src and "import http" not in src


def test_the_rungs_are_the_designs_six_and_the_policies_their_two_values():
    assert T._RUNGS == ("T0", "T1", "T2", "T3a", "T3b", "T3c")
    assert T._RUNG_NAME == {"T0": "NOT_ORDERED", "T1": "AUTHOR_ORDER", "T2": "SERVER_ORDER", "T3a": "TSA_TIME",
                            "T3b": "OTS_PROOF", "T3c": "STYXX_CLOCK"}
    assert T._REQUIRE == ("T1", "T3a")
    assert T._WORD_POLICIES == ("note", "fail")
    assert T._TSA_DIR == "prereg-ci/tsa"
    assert T._ANCHORS == "papers/charon/anchors.jsonl"
    assert T._KIND[T._TIME_BELOW_POLICY] == "FAIL"
    assert T._KIND[T._OPENSSL_ABSENT] == "UNCHECKABLE"
    assert all(k in ("FAIL", "NOTE", "UNCHECKABLE") for k in T._KIND.values())


# ---- the token, read without openssl -----------------------------------------------------------------------

@needs_openssl
def test_a_response_from_the_test_tsa_parses_to_its_tstinfo(token):
    info = T._tstinfo(token.bytes)
    assert info["form"] == "response" and info["status"] == 0 and info["status_text"] == "granted"
    assert info["version"] == 1 and info["policy"] == POLICY_OID
    assert info["imprint_alg"] == "sha256" and info["imprint"] == sha(SEAL)
    assert isinstance(info["serial"], int) and info["serial"] >= 1
    assert token.before <= info["gen_time_epoch"] <= token.after
    assert info["gen_time"].endswith("Z") and len(info["gen_time"]) == 20
    assert info["tsa"] == "/CN=prereg-ci test TSA"
    assert info["nonce"] is None


@needs_openssl
def test_a_bare_token_parses_too_and_names_its_form(tsa):
    info = T._tstinfo(tsa.stamp(SEAL, token_only=True))
    assert info["form"] == "token" and info["status"] is None
    assert info["imprint"] == sha(SEAL) and info["gen_time_epoch"] > 0


def test_a_rejection_response_parses_with_no_token_inside():
    """TimeStampResp with status 2 (rejection) and a status string, built by hand."""
    status_text = b"\x0c\x02no"
    pki = b"\x30" + bytes([3 + 2 + len(status_text)]) + b"\x02\x01\x02" + b"\x30" + bytes([len(status_text)]) + status_text
    resp = b"\x30" + bytes([len(pki)]) + pki
    info = T._tstinfo(resp)
    assert info["form"] == "response" and info["status"] == 2 and info["status_text"] == "rejection"
    assert info["gen_time"] is None and info["imprint"] is None


@pytest.mark.parametrize("data", [b"", b"\x30", b"\x30\x05\x02\x01", b"not a token at all", b"\x30\x80\x00\x00",
                                  b"\x30\x03\x06\x01\x2a"])
def test_bytes_that_are_not_a_token_are_refused_with_a_reason(data):
    with pytest.raises(T._TokenError) as e:
        T._tstinfo(data)
    assert str(e.value)


@needs_openssl
def test_a_truncated_token_is_refused(token):
    with pytest.raises(T._TokenError):
        T._tstinfo(token.bytes[:-40])


# ---- bytes a branch can commit as a token: each reads MALFORMED with a reason, none raises past the reader ----

def der(tag: int, body: bytes) -> bytes:
    n = len(body)
    if n < 0x80:
        return bytes([tag, n]) + body
    nb = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([tag, 0x80 | len(nb)]) + nb + body


def oid_bytes(arcs) -> bytes:
    out = b""
    for a in [40 * arcs[0] + arcs[1], *arcs[2:]]:
        chunk = [a & 0x7F]
        a >>= 7
        while a:
            chunk.append(0x80 | (a & 0x7F))
            a >>= 7
        out += bytes(reversed(chunk))
    return out


def hand_token(gen_time: bytes, imprint: bytes = hashlib.sha256(SEAL).digest(), serial: bytes = b"\x2b") -> bytes:
    """A bare TimeStampToken's shape written by hand (ContentInfo, SignedData, TSTInfo), with no signer:
    the DER reader reads it, and openssl would refuse it."""
    tst = der(0x30, der(0x02, b"\x01") + der(0x06, oid_bytes([1, 3, 6, 1, 4, 1, 99999, 1]))
              + der(0x30, der(0x30, der(0x06, oid_bytes([2, 16, 840, 1, 101, 3, 4, 2, 1]))) + der(0x04, imprint))
              + der(0x02, serial) + der(0x18, gen_time))
    eci = der(0x30, der(0x06, oid_bytes([1, 2, 840, 113549, 1, 9, 16, 1, 4])) + der(0xA0, der(0x04, tst)))
    sd = der(0x30, der(0x02, b"\x03") + der(0x31, b"") + eci)
    return der(0x30, der(0x06, oid_bytes([1, 2, 840, 113549, 1, 7, 2])) + der(0xA0, sd))


def test_a_hand_written_token_reads_as_the_test_tsas_do():
    info = T._tstinfo(hand_token(b"20261025000000Z"))
    assert info["form"] == "token" and info["policy"] == "1.3.6.1.4.1.99999.1" and info["serial"] == 0x2B
    assert info["imprint_alg"] == "sha256" and info["imprint"] == sha(SEAL)
    assert info["gen_time"] == "2026-10-25T00:00:00Z"


@pytest.mark.parametrize("data, words", [
    (bytes.fromhex("3005060180a000"), "cut mid-arc"),                    # an OID of the one byte 0x80 (B6 review A1)
    (der(0x30, der(0x06, b"\x2a" + b"\xff" * 3000 + b"\x7f") + der(0xA0, b"")), "arc of"),  # one arc 3001 bytes long
    (der(0x30, der(0x30, der(0x02, b"\x01" * 2000))), "integer of 2000 bytes"),          # a PKI status 2000 bytes long
    (hand_token(b"20261025000000Z", serial=b"\x01" * 129), "integer of 129 bytes"),      # a serial past 1024 bits
], ids=["oid-0x80-alone", "oid-arc-3001-bytes", "status-2000-bytes", "serial-129-bytes"])
def test_a_token_whose_oid_or_integer_python_would_not_read_or_print_is_malformed(data, words):
    """Before the fix the A1 bytes raised IndexError, the long arc and status ValueError (Python prints no
    int past 4300 digits) and the long serial read as an int nothing prints; the contract is MALFORMED."""
    with pytest.raises(T._TokenError) as e:
        T._tstinfo(data)
    assert words in str(e.value)
    r = T._rfc3161(SEAL, data, None)
    assert r["status"] == "MALFORMED" and words in r["detail"]


@pytest.mark.parametrize("hexed", ["80", "2a80", "2a8686", "2a864886f7"])
def test_an_oid_whose_last_byte_continues_an_arc_is_cut_mid_arc(hexed):
    """B6 review A7: ``2a 80`` read as "1.2", since a trailing 0x80 leaves the arc's value at 0."""
    data = bytes.fromhex(hexed)
    with pytest.raises(T._TokenError) as e:
        T._oid(data, 0, len(data))
    assert "cut mid-arc" in str(e.value)


def test_the_reader_keeps_a_uuid_arc_and_a_160_bit_serial():
    """The caps sit past what RFC 3161 tokens carry: a 2.25 UUID arc is 19 bytes, a serial runs to 160 bits."""
    uuid_oid = oid_bytes([2, 25, 2 ** 128 - 1])
    assert T._oid(uuid_oid, 0, len(uuid_oid)) == f"2.25.{2 ** 128 - 1}"
    serial = b"\x7f" + b"\xff" * 19
    assert T._tstinfo(hand_token(b"20261025000000Z", serial=serial))["serial"] == int.from_bytes(serial, "big")
    edge = b"\x7f" + b"\xff" * (T._INT_MAX_BYTES - 1)
    assert T._int(edge, 0, len(edge)) == int.from_bytes(edge, "big")


def test_a_crafted_token_committed_beside_the_seal_reads_malformed_through_the_ladder(tmp_path):
    """B6 review A1, through a repository: the ladder prints the token MALFORMED and reads on."""
    g = Git(tmp_path / "r")
    g.write(P, SEAL)
    c0 = g.commit("seal the plan")
    g.write(P + ".tsr", bytes.fromhex("3005060180a000"))
    c1 = g.commit("a crafted token")
    g.write(R, receipt(m=0.7))
    rc = g.commit("a result")
    with G._Repo(g.root) as repo:
        r = T._reading(repo, path=P, rev=rc, seal=SEAL, order=c0, receipt=rc, trust_rev=c0)
    [t] = r["T3a"]
    assert t["path"] == P + ".tsr" and t["status"] == "MALFORMED" and "cut mid-arc" in t["detail"]
    assert r["rung"] == "T1" and c1 != rc
    assert codes(T._findings(r)) == [(T._TSA_TOKEN_UNVERIFIED, "NOTE")]


# ---- offline verification through openssl (the done-when) -------------------------------------------------

@needs_openssl
def test_done_when_a_token_from_the_test_tsa_verifies_offline_against_the_seal_bytes(tsa, token):
    r = T._rfc3161(SEAL, token.bytes, tsa.ca)
    assert r["status"] == "VERIFIED", r
    assert r["imprint_matches"] is True
    assert r["at"] == r["gen_time_epoch"], "K5: verified at the token's own time, not the wall clock"
    assert r["openssl"].startswith("OpenSSL")
    assert r["gen_time"] == T._iso(r["gen_time_epoch"])
    assert r["tsa"] == "/CN=prereg-ci test TSA"


@needs_openssl
def test_done_when_one_flipped_seal_byte_fails_the_token(tsa, token):
    flipped = bytearray(SEAL)
    flipped[len(flipped) // 2] ^= 0x01
    r = T._rfc3161(bytes(flipped), token.bytes, tsa.ca)
    assert r["status"] == "FAILED", r
    assert r["imprint_matches"] is False
    assert "imprint" in r["detail"]


@needs_openssl
def test_done_when_through_a_repository_the_blob_verifies_and_a_tampered_seal_byte_fails_it(tmp_path, tsa,
                                                                                           no_network):
    """DESIGN 8 B6, PLAN B6: the seal bytes are materialised from the blob at a commit (a CRLF
    working copy is not what is read), the token from the local test TSA verifies offline against
    them at its own time, and a commit that flips one byte of the sealed file reads that token
    FAILED. Python opens no socket meanwhile; openssl ts -verify fetches nothing."""
    g = Git(tmp_path / "r")
    g.write(P, SEAL)
    c0 = g.commit("seal the plan")
    with G._Repo(g.root) as repo:
        seal = repo.read(c0, P)
        assert seal == SEAL
        g.write(P + ".tsr", tsa.stamp(seal))
        g.write(T._TSA_DIR + "/ca.pem", tsa.ca)
        g.commit("stamp the plan")
        g.write(R, receipt(m=0.7))
        r1 = g.commit("a result, tomorrow", at=int(_clock.time()) + 86_400 - 1_760_000_000)
        g.write(P, SEAL.replace(b"\n", b"\r\n"))
        r = T._reading(repo, path=P, rev=r1, seal=repo.read(r1, P), receipt=r1, order=c0)
        assert (r["rung"], r["name"]) == ("T3a", "TSA_TIME")
        [t] = r["T3a"]
        assert t["status"] == "VERIFIED" and t["imprint_matches"] is True and t["at"] == t["gen_time_epoch"]
        assert t["precedes_receipt"] is True and r["trust"] == [T._TSA_DIR + "/ca.pem"]
        assert T._findings(r, require_time="T3a") == []
        flipped = bytearray(SEAL)
        flipped[len(flipped) // 2] ^= 0x01
        g.write(P, bytes(flipped))
        c2 = g.commit("one byte of the sealed file")
        tampered = repo.read(c2, P)
        assert tampered != SEAL and len(tampered) == len(SEAL)
        r = T._reading(repo, path=P, rev=c2, seal=tampered, receipt=r1, order=c0)
        [t] = r["T3a"]
        assert t["status"] == "FAILED" and t["imprint_matches"] is False and "imprint" in t["detail"]
        assert (r["rung"], r["name"]) == ("T1", "AUTHOR_ORDER")
        assert codes(T._findings(r)) == [(T._TSA_TOKEN_UNVERIFIED, "NOTE")]
        assert codes(T._findings(r, require_time="T3a")) == [(T._TSA_TOKEN_UNVERIFIED, "NOTE"),
                                                             (T._TIME_BELOW_POLICY, "FAIL")]


@needs_openssl
def test_a_token_over_other_bytes_fails_against_the_seal(tsa):
    other = tsa.stamp(b"other bytes\n")
    r = T._rfc3161(SEAL, other, tsa.ca)
    assert r["status"] == "FAILED" and r["imprint_matches"] is False


@needs_openssl
def test_a_tampered_token_byte_fails_the_signature_though_the_imprint_still_matches(tsa, token):
    """openssl checks the TSA's signature over TSTInfo: one changed digit of genTime is refused even
    though the imprint is still the seal's digest."""
    info = T._tstinfo(token.bytes)
    gt = datetime.datetime.fromtimestamp(info["gen_time_epoch"], datetime.timezone.utc)
    der = b"\x18\x0f" + gt.strftime("%Y%m%d%H%M%S").encode("ascii") + b"Z"
    assert token.bytes.count(der) == 1, "the genTime element is found once in the response"
    raw = bytearray(token.bytes)
    i = token.bytes.index(der) + 2 + 13                    # the units digit of the seconds
    raw[i] = ord("0") + (raw[i] - ord("0") + 1) % 10
    moved = T._tstinfo(bytes(raw))
    assert moved["imprint"] == sha(SEAL) and moved["gen_time_epoch"] != info["gen_time_epoch"]
    r = T._rfc3161(SEAL, bytes(raw), tsa.ca)
    assert r["status"] == "FAILED" and r["imprint_matches"] is True, r
    assert "openssl ts -verify" in r["detail"]


@needs_openssl
def test_a_trust_store_that_does_not_reach_the_tsa_fails_the_token(tsa, token):
    r = T._rfc3161(SEAL, token.bytes, tsa.tsa_pem)
    assert r["status"] == "FAILED" and r["imprint_matches"] is True
    assert "certificate" in r["detail"]


@needs_openssl
def test_the_trust_store_is_a_concatenation_of_certificates(tsa, token):
    r = T._rfc3161(SEAL, token.bytes, tsa.tsa_pem + b"\n" + tsa.ca)
    assert r["status"] == "VERIFIED"


@needs_openssl
def test_the_verification_time_is_the_tokens_own_and_a_fixed_other_time_moves_the_verdict(tsa, token):
    """PLAN K5: the same committed objects verify the same way whenever they are read."""
    assert T._rfc3161(SEAL, token.bytes, tsa.ca, at=FAR_FUTURE)["status"] == "FAILED"
    assert "expired" in T._rfc3161(SEAL, token.bytes, tsa.ca, at=FAR_FUTURE)["detail"]
    assert "not yet valid" in T._rfc3161(SEAL, token.bytes, tsa.ca, at=FAR_PAST)["detail"]
    ok = T._rfc3161(SEAL, token.bytes, tsa.ca, at=T._tstinfo(token.bytes)["gen_time_epoch"])
    assert ok["status"] == "VERIFIED"


def test_the_verify_command_line_is_the_designs_with_the_time_fixed():
    argv = T._verify_argv("openssl", "s.bin", "t.tsr", "ca.pem", 1234, "response")
    assert argv == ["openssl", "ts", "-verify", "-data", "s.bin", "-in", "t.tsr", "-CAfile", "ca.pem", "-attime", "1234"]
    assert T._verify_argv("openssl", "s", "t", "c", 1, "token")[-3:] == ["-attime", "1", "-token_in"]


@needs_openssl
def test_a_bare_token_verifies_through_token_in(tsa):
    r = T._rfc3161(SEAL, tsa.stamp(SEAL, token_only=True), tsa.ca)
    assert r["status"] == "VERIFIED" and r["form"] == "token"


@needs_openssl
def test_without_openssl_the_token_is_uncheckable_and_nothing_runs(tsa, token, monkeypatch):
    ran = []
    monkeypatch.setattr(T.subprocess, "run", lambda *a, **k: ran.append(a))
    r = T._rfc3161(SEAL, token.bytes, tsa.ca, exe=None)
    assert r["status"] == "UNCHECKABLE" and "openssl" in r["detail"] and not ran
    assert r["imprint_matches"] is True, "the imprint is still read from the token alone"
    assert r["openssl"] is None


@needs_openssl
def test_without_a_trust_store_the_token_is_read_but_not_verified(token):
    r = T._rfc3161(SEAL, token.bytes, None)
    assert r["status"] == "NO_TRUST" and r["imprint_matches"] is True and T._TSA_DIR in r["detail"]


def test_bytes_that_are_not_a_token_read_malformed():
    r = T._rfc3161(SEAL, b"junk", b"-----BEGIN CERTIFICATE-----\n")
    assert r["status"] == "MALFORMED" and r["imprint_matches"] is None


@needs_openssl
def test_a_rejection_response_is_not_verified(tsa):
    status_text = b"\x0c\x02no"
    pki = b"\x30" + bytes([5 + len(status_text)]) + b"\x02\x01\x02" + b"\x30" + bytes([len(status_text)]) + status_text
    r = T._rfc3161(SEAL, b"\x30" + bytes([len(pki)]) + pki, tsa.ca)
    assert r["status"] == "FAILED" and "rejection" in r["detail"]


@needs_openssl
def test_an_openssl_that_hangs_is_uncheckable_and_a_mismatch_still_fails(tsa, token, monkeypatch):
    """The run is reached (a real token, a trust store): the timeout is UNCHECKABLE, not a verdict;
    an imprint that does not match the seal fails whatever openssl does."""
    def hang(*a, **k):
        raise subprocess.TimeoutExpired(a[0], k.get("timeout", 0))
    monkeypatch.setattr(T.subprocess, "run", hang)
    r = T._rfc3161(SEAL, token.bytes, tsa.ca, exe="openssl-that-hangs")
    assert r["status"] == "UNCHECKABLE" and "ran past" in r["detail"] and r["imprint_matches"] is True
    flipped = bytearray(SEAL)
    flipped[0] ^= 0x01
    r = T._rfc3161(bytes(flipped), token.bytes, tsa.ca, exe="openssl-that-hangs")
    assert r["status"] == "FAILED" and "imprint" in r["detail"]


def test_which_openssl_is_shutil_which(monkeypatch):
    monkeypatch.setattr(T.shutil, "which", lambda name: "/x/openssl" if name == "openssl" else None)
    assert T._openssl() == "/x/openssl"
    monkeypatch.setattr(T.shutil, "which", lambda name: None)
    assert T._openssl() is None


# ---- the OpenTimestamps proof, parsed offline --------------------------------------------------------------

def test_a_pending_proof_over_the_seal_reads_pending_and_names_its_calendar():
    r = T._ots(ots(hashlib.sha256(SEAL).digest(), PENDING_BRANCH), SEAL)
    assert r["status"] == "PENDING" and r["magic"] is True
    assert r["digest_alg"] == "sha256" and r["digest"] == sha(SEAL) and r["matches_seal"] is True
    assert r["attestations"] == {"bitcoin": [], "litecoin": [], "ethereum": [], "pending": ["https://a.calendar.example"],
                                 "unknown": 0}
    assert r["size"] == len(ots(hashlib.sha256(SEAL).digest(), PENDING_BRANCH))
    assert r["sha256"] == sha(ots(hashlib.sha256(SEAL).digest(), PENDING_BRANCH))
    assert "not verified" in r["detail"] or "Bitcoin" in r["detail"]


def test_a_proof_with_a_bitcoin_attestation_reads_attested_with_the_height_the_proof_claims():
    r = T._ots(ots(hashlib.sha256(SEAL).digest(), PENDING_BRANCH, BITCOIN_BRANCH), SEAL)
    assert r["status"] == "ATTESTED"
    assert r["attestations"]["bitcoin"] == [812345] and r["attestations"]["pending"] == ["https://a.calendar.example"]
    assert "812345" in r["detail"] and "node" in r["detail"]


def test_a_proof_over_other_bytes_does_not_match_the_seal():
    r = T._ots(ots(hashlib.sha256(b"other").digest(), PENDING_BRANCH), SEAL)
    assert r["status"] == "PENDING" and r["matches_seal"] is False


def test_a_sha1_proof_names_its_algorithm_and_still_compares():
    r = T._ots(ots(hashlib.sha1(SEAL).digest(), PENDING_BRANCH, alg=b"\x02"), SEAL)
    assert r["digest_alg"] == "sha1" and r["matches_seal"] is True


@pytest.mark.parametrize("data, words", [
    (b"", "magic"),
    (b"\x00OpenTimestamps\x00\x00Proof\x00" + b"\x00" * 8 + b"\x01\x08" + b"\x00" * 32, "magic"),
    (ots(b"\x00" * 32, PENDING_BRANCH)[:-5], "truncated"),
    (ots(b"\x00" * 32, PENDING_BRANCH) + b"\x00", "after"),
    (OTS_MAGIC + varuint(2) + b"\x08" + b"\x00" * 32 + attest(PENDING, varbytes(b"u")), "version"),
    (OTS_MAGIC + varuint(1) + b"\x09" + b"\x00" * 32 + attest(PENDING, varbytes(b"u")), "hash"),
])
def test_bytes_that_are_not_a_proof_read_malformed_with_a_reason(data, words):
    r = T._ots(data, SEAL)
    assert r["status"] == "MALFORMED" and words in r["detail"], r


def test_an_attestation_of_an_unknown_kind_is_counted_and_a_proof_with_none_says_so():
    unknown = attest("0102030405060708", varbytes(b"?"))
    r = T._ots(ots(hashlib.sha256(SEAL).digest(), unknown), SEAL)
    assert r["status"] == "NO_ATTESTATION" and r["attestations"]["unknown"] == 1
    r2 = T._ots(ots(hashlib.sha256(SEAL).digest(), b"\x08" + attest(LITECOIN, varuint(7))), SEAL)
    assert r2["status"] == "ATTESTED" and r2["attestations"]["litecoin"] == [7]


# ---- the clock rung, shape only ----------------------------------------------------------------------------

def test_anchor_lines_about_the_seal_read_through_check_anchor_offline_and_are_labelled_not_read_on_chain():
    good = {"n": 1, "kind": "sealed-prereg", "digest": sha(SEAL), "tx": "5" * 88, "slot": 1, "block_time": "2026-10-25T00:00:00Z"}
    other = {"n": 2, "kind": "sealed-prereg", "digest": "0" * 64, "tx": "6" * 88}
    bad = {"n": 3, "kind": "no-such-kind", "digest": sha(SEAL), "tx": "7" * 88}
    data = "\n".join(json.dumps(x) for x in (good, other, bad)).encode("utf-8") + b"\nnot json\n"
    r = T._anchors(data, [sha(SEAL)])
    assert r["lines"] == 4 and r["malformed"] == 1
    assert [a["n"] for a in r["about_seal"]] == [1, 3] and [a["line"] for a in r["about_seal"]] == [1, 3]
    assert r["about_seal"][0]["status"] == "SHAPE_OK" and r["about_seal"][1]["status"] == "MALFORMED_LINE"
    assert all(a["chain"] == "not read on chain" for a in r["about_seal"])
    assert r["about_seal"][0]["memo_expected"] == C.memo("sealed-prereg", sha(SEAL))


def test_an_anchor_line_may_name_the_decision_digest_instead_of_the_blob():
    line = {"n": 1, "kind": "sealed-prereg", "digest": digest(0.5), "tx": "5" * 88}
    r = T._anchors(json.dumps(line).encode("utf-8") + b"\n", [sha(SEAL), digest(0.5)])
    assert [a["n"] for a in r["about_seal"]] == [1]


def test_an_anchor_lines_own_n_is_kept_apart_from_its_line_in_the_file():
    """``n`` is the anchors format's sequence number, as the line writes it (None when it writes
    none); ``line`` is where the line sits in the file."""
    lines = [{"kind": "sealed-prereg", "digest": "0" * 64, "tx": "4" * 88},
             {"n": 9, "kind": "sealed-prereg", "digest": sha(SEAL), "tx": "5" * 88},
             {"kind": "sealed-prereg", "digest": sha(SEAL), "tx": "6" * 88}]
    r = T._anchors(b"\n".join(json.dumps(x).encode("utf-8") for x in lines) + b"\n", [sha(SEAL)])
    assert [(a["n"], a["line"]) for a in r["about_seal"]] == [(9, 2), (None, 3)]


DEEP_LINES = ["[" * 50_000, '{"a": ' * 50_000]      # under the 1 MiB cap, nested past Python's recursion limit


def test_an_anchors_line_nested_too_deep_is_malformed_and_one_whose_digest_is_no_string_is_not_about_the_seal():
    """B6 review A2 and A3: a line whose digest is a JSON list raised TypeError (a list is not hashable)
    and a line of 50,000 '[' raised RecursionError out of the reader. A digest that is not a string names
    no seal, since a seal's digests are hex strings: the line is counted and not read further."""
    good = {"n": 1, "kind": "sealed-prereg", "digest": sha(SEAL), "tx": "5" * 88}
    lines = [json.dumps({"n": 2, "kind": "sealed-prereg", "digest": [sha(SEAL)], "tx": "t"}),
             json.dumps({"n": 3, "kind": "sealed-prereg", "digest": {"sha256": sha(SEAL)}, "tx": "t"}),
             *DEEP_LINES, json.dumps(good)]
    r = T._anchors("\n".join(lines).encode("utf-8") + b"\n", [sha(SEAL)])
    assert r["lines"] == 5 and r["malformed"] == 2
    assert [(a["n"], a["line"], a["status"]) for a in r["about_seal"]] == [(1, 5, "SHAPE_OK")]


def test_a_committed_anchors_file_holding_those_lines_leaves_the_ladder_reading(tmp_path):
    """The anchors file is read at the rev for every seal, so one such line on a branch took down every
    reading there; now the ladder reads and prints the count."""
    good = {"n": 1, "kind": "sealed-prereg", "digest": sha(SEAL), "tx": "5" * 88}
    g = Git(tmp_path / "r")
    g.write(P, SEAL)
    c0 = g.commit("seal the plan")
    g.write(T._ANCHORS, "\n".join([json.dumps({"n": 2, "kind": "sealed-prereg", "digest": ["x"], "tx": "t"}),
                                   DEEP_LINES[0], json.dumps(good)]) + "\n")
    g.write(R, receipt(m=0.7))
    rc = g.commit("an anchors file, and a result")
    with G._Repo(g.root) as repo:
        r = T._reading(repo, path=P, rev=rc, seal=SEAL, order=c0, receipt=rc, trust_rev=c0)
    assert r["rung"] == "T1" and r["T3c"]["files"] == [T._ANCHORS]
    assert (r["T3c"]["lines"], r["T3c"]["malformed"]) == (3, 1)
    assert [(a["n"], a["status"]) for a in r["T3c"]["about_seal"]] == [(1, "SHAPE_OK")]
    assert "T3c 3 lines in papers/charon/anchors.jsonl, 1 about this seal" in T._describe(r)


# ---- repositories: tokens beside the sealed file, the trust store, the ladder ---------------------------------

@pytest.fixture(scope="module")
def stamped(tmp_path_factory, tsa):
    """main: c0 seals the PREREG; r_early adds a receipt dated by the fixture's clock (2025, before any
    token's time); c1 commits the token, a pending OTS proof and the CA under prereg-ci/tsa/; c2 adds a
    token for a second path under a manifest line's own list; r_late adds a receipt dated tomorrow, so
    the token's time precedes it; c3 stamps that receipt. A branch ``other`` holds a commit no receipt
    descends from."""
    g = Git(tmp_path_factory.mktemp("stamped"))
    g.write("README.md", "x\n")
    m0 = g.commit("start")
    g.write(P, SEAL)
    c0 = g.commit("seal the plan")
    g.write(R_EARLY, receipt(m=0.7))
    r_early = g.commit("an early result")
    tok = tsa.stamp(SEAL)
    g.write(P + ".tsr", tok)
    g.write(P + ".ots", ots(hashlib.sha256(SEAL).digest(), PENDING_BRANCH))
    g.write(T._TSA_DIR + "/ca.pem", tsa.ca)
    g.write(T._TSA_DIR + "/other.crt", tsa.tsa_pem)
    g.write(T._TSA_DIR + "/README.md", "the pinned certificates\n")
    c1 = g.commit("stamp the plan")
    g.write("p/alt/plan.tsr", tok)
    g.write("p/alt/bad.tsr", tsa.stamp(b"other\n"))
    c2 = g.commit("a token under another name, and one over other bytes")
    tomorrow = int(_clock.time()) + 86_400
    g.write(R, receipt(m=0.7))
    r_late = g.commit("a result, tomorrow", at=tomorrow - 1_760_000_000)
    rtok = tsa.stamp(receipt(m=0.7).encode("utf-8"))
    g.write(R + ".tsr", rtok)
    c3 = g.commit("stamp the result")
    g("switch", "-qc", "other", m0)
    g.write("other.txt", "y\n")
    o = g.commit("elsewhere")
    g("switch", "-q", "main")
    with G._Repo(g.root) as repo:
        yield SimpleNamespace(g=g, repo=repo, m0=m0, c0=c0, r_early=r_early, c1=c1, c2=c2, r_late=r_late, c3=c3,
                              o=o, tok=tok, rtok=rtok)


def test_commit_time_is_the_committer_time_git_recorded(stamped):
    repo, g = stamped.repo, stamped.g
    for c in (stamped.c0, stamped.r_late):
        assert repo.commit_time(c) == int(g("log", "-1", "--format=%ct", c))
    assert repo.commit_time(stamped.r_late) > int(_clock.time())
    with pytest.raises(G._Refusal) as e:
        repo.commit_time(g("rev-parse", stamped.c0 + "^{tree}"))
    assert e.value.code == "COMMIT_ABSENT"


def test_the_default_tokens_are_the_tsr_and_ots_beside_the_sealed_file(stamped):
    got = T._tokens(stamped.repo, stamped.c1, P, None)
    assert [(t["path"], t["kind"], t["bytes"] is not None) for t in got] == [
        (P + ".tsr", "rfc3161", True), (P + ".ots", "ots", True)]
    assert got[0]["sha256"] == sha(stamped.tok)
    assert T._tokens(stamped.repo, stamped.c0, P, None) == []


def test_a_manifest_lines_time_list_names_the_tokens_and_an_absent_one_is_listed_unread(stamped):
    """The line's list, in its order, then the files beside the sealed file it does not name: the
    manifest is append-only and a line's ``time`` is written when it is sealed, so a token stamped
    later beside the file is still read."""
    line = {"id": "1", "kind": "prereg", "path": P, "commit": stamped.c0, "blob_sha256": sha(SEAL), "amends": None,
            "decision": "own", "seen": None, "time": ["p/alt/plan.tsr", "p/missing.tsr", "p/alt/plan.txt"]}
    got = T._tokens(stamped.repo, stamped.c2, P, line)
    assert [(t["path"], t["kind"], t["bytes"] is not None) for t in got] == [
        ("p/alt/plan.tsr", "rfc3161", True), ("p/missing.tsr", "rfc3161", False), ("p/alt/plan.txt", None, False),
        (P + ".tsr", "rfc3161", True), (P + ".ots", "ots", True)]
    assert "not in the tree" in got[1]["why"] and "reads" in got[2]["why"]
    named = dict(line, time=[P + ".ots"])
    assert [t["path"] for t in T._tokens(stamped.repo, stamped.c2, P, named)] == [P + ".ots", P + ".tsr"]
    empty = dict(line, time=[])
    assert [t["path"] for t in T._tokens(stamped.repo, stamped.c2, P, empty)] == [P + ".tsr", P + ".ots"]
    assert T._tokens(stamped.repo, stamped.c0, P, empty) == []


def test_a_token_over_the_cap_is_listed_unread(stamped, monkeypatch):
    monkeypatch.setattr(T, "_TOKEN_MAX_BYTES", 10)
    got = T._tokens(stamped.repo, stamped.c1, P, None)
    assert all(t["bytes"] is None and "bytes" in t["why"] for t in got)


def test_the_trust_store_is_every_pem_and_crt_under_prereg_ci_tsa_in_path_order(stamped, tsa):
    trust, paths = T._trust(stamped.repo, stamped.c1)
    assert paths == [T._TSA_DIR + "/ca.pem", T._TSA_DIR + "/other.crt"]
    assert trust == tsa.ca + b"\n" + tsa.tsa_pem
    assert T._trust(stamped.repo, stamped.c0) == (None, [])


@pytest.fixture(scope="module")
def rogue(tmp_path_factory):
    """A second TSA with its own CA, standing for one a branch makes for itself."""
    if OPENSSL is None:
        pytest.skip("openssl is not on PATH, so no test TSA can be made here")
    return Tsa(tmp_path_factory.mktemp("rogue"))


@needs_openssl
def test_the_trust_store_is_read_at_the_rev_the_caller_names_so_a_branch_cannot_pin_its_own_tsa(tmp_path, tsa,
                                                                                                 rogue):
    """A branch that commits a TSA certificate of its own beside a token from that TSA verifies
    against the trust store at its own head; read at the base side, the same token fails. So the
    pinned certificates are read where the caller says (``trust_rev``; step B7's check passes BASE),
    as the locked scorer runs from the base side."""
    g = Git(tmp_path / "r")
    g.write(P, SEAL)
    g.write(T._TSA_DIR + "/ca.pem", tsa.ca)
    base = g.commit("seal the plan; pin the TSA")
    g.write(P + ".tsr", rogue.stamp(SEAL))
    g.write(T._TSA_DIR + "/mine.pem", rogue.ca)
    head = g.commit("a token, and the certificate that verifies it")
    with G._Repo(g.root) as repo:
        at_head = T._reading(repo, path=P, rev=head, seal=SEAL, order=base)
        assert at_head["trust_rev"] == head and at_head["T3a"][0]["status"] == "VERIFIED"
        assert at_head["rung"] == "T3a"
        at_base = T._reading(repo, path=P, rev=head, seal=SEAL, order=base, trust_rev=base)
        assert at_base["trust_rev"] == base and at_base["trust"] == [T._TSA_DIR + "/ca.pem"]
        [t] = at_base["T3a"]
        assert t["status"] == "FAILED" and t["imprint_matches"] is True and "certificate" in t["detail"], t
        assert at_base["rung"] is None
        assert codes(T._findings(at_base, require_time="T3a")) == [(T._TSA_TOKEN_UNVERIFIED, "NOTE"),
                                                                   (T._TIME_BELOW_POLICY, "FAIL")]


def test_the_ladder_reads_tsa_time_when_the_token_verifies_and_precedes_the_receipt(stamped):
    r = T._reading(stamped.repo, path=P, rev=stamped.c3, seal=SEAL, digests=[digest(0.5)], receipt=stamped.r_late,
                   order=stamped.c0)
    assert (r["rung"], r["name"]) == ("T3a", "TSA_TIME")
    assert r["ordered"] is True and r["T1"]["holds"] is True
    assert r["T2"]["read"] is False and "not read" in r["T2"]["detail"]
    [t] = r["T3a"]
    assert t["path"] == P + ".tsr" and t["status"] == "VERIFIED" and t["precedes_receipt"] is True
    assert t["gen_time_epoch"] < r["receipt_committed_at_epoch"]
    assert r["receipt_committed_at"] == T._iso(r["receipt_committed_at_epoch"])
    [o] = r["T3b"]
    assert o["path"] == P + ".ots" and o["status"] == "PENDING" and o["matches_seal"] is True
    assert r["T3c"]["files"] == [] and "no anchors file" in r["T3c"]["detail"]
    assert r["trust"] == [T._TSA_DIR + "/ca.pem", T._TSA_DIR + "/other.crt"]
    assert r["openssl"].startswith("OpenSSL")
    assert r["unread"] == []
    assert r["seal_sha256"] == sha(SEAL) and digest(0.5) in r["digests"]
    line = T._describe(r)
    assert line.startswith("time TSA_TIME " + t["gen_time"]) and "T2 not read" in line and P + ".tsr" in line
    assert "T3b pending" in line and "T3c none" in line


def test_a_verified_token_whose_time_does_not_precede_the_receipts_leaves_the_rung_at_author_order(stamped):
    r = T._reading(stamped.repo, path=P, rev=stamped.c3, seal=SEAL, receipt=stamped.r_early, order=stamped.c0)
    assert (r["rung"], r["name"]) == ("T1", "AUTHOR_ORDER")
    [t] = r["T3a"]
    assert t["status"] == "VERIFIED" and t["precedes_receipt"] is False
    assert "does not precede" in T._describe(r)


def test_with_no_token_the_ladder_prints_the_designs_author_order_line(stamped):
    r = T._reading(stamped.repo, path=P, rev=stamped.r_early, seal=SEAL, receipt=stamped.r_early, order=stamped.c0)
    assert (r["rung"], r["name"]) == ("T1", "AUTHOR_ORDER")
    assert r["T3a"] == [] and r["T3b"] == [] and r["T3c"]["files"] == []
    assert T._describe(r) == "time AUTHOR_ORDER (T2 not read, T3 none)"
    assert r["openssl"] is None or r["openssl"].startswith("OpenSSL")


def test_a_receipt_the_order_commit_does_not_reach_is_not_ordered_whatever_the_token_says(stamped):
    r = T._reading(stamped.repo, path=P, rev=stamped.c3, seal=SEAL, receipt=stamped.r_late, order=stamped.o)
    assert (r["rung"], r["name"]) == ("T0", "NOT_ORDERED") and r["ordered"] is False
    assert r["T3a"][0]["status"] == "VERIFIED"
    assert T._describe(r).startswith("time NOT_ORDERED")


def test_before_any_receipt_the_ladder_reads_the_seals_own_rung(stamped):
    r = T._reading(stamped.repo, path=P, rev=stamped.c1, seal=SEAL, order=stamped.c0)
    assert r["receipt"] is None and r["ordered"] is None and r["receipt_committed_at"] is None
    assert (r["rung"], r["name"]) == ("T3a", "TSA_TIME") and r["T3a"][0]["precedes_receipt"] is None
    r0 = T._reading(stamped.repo, path=P, rev=stamped.c0, seal=SEAL, order=stamped.c0)
    assert r0["rung"] is None and r0["name"] is None and "no receipt" in T._describe(r0)


def test_a_manifest_line_decides_which_tokens_are_read_and_a_token_over_other_bytes_fails(stamped):
    line = {"id": "1", "kind": "prereg", "path": P, "commit": stamped.c0, "blob_sha256": sha(SEAL), "amends": None,
            "decision": "own", "seen": None, "time": ["p/alt/bad.tsr", "p/alt/plan.tsr"]}
    r = T._reading(stamped.repo, path=P, rev=stamped.c3, seal=SEAL, receipt=stamped.r_late, order=stamped.c0, line=line)
    assert [(t["path"], t["status"]) for t in r["T3a"]] == [("p/alt/bad.tsr", "FAILED"), ("p/alt/plan.tsr", "VERIFIED"),
                                                            (P + ".tsr", "VERIFIED")]
    assert r["rung"] == "T3a" and [o["path"] for o in r["T3b"]] == [P + ".ots"]
    assert codes(T._findings(r)) == [(T._TSA_TOKEN_UNVERIFIED, "NOTE")]


def test_without_openssl_the_ladder_reads_author_order_and_the_token_uncheckable(stamped):
    r = T._reading(stamped.repo, path=P, rev=stamped.c3, seal=SEAL, receipt=stamped.r_late, order=stamped.c0, exe=None)
    assert r["rung"] == "T1" and r["T3a"][0]["status"] == "UNCHECKABLE" and r["openssl"] is None


def test_without_a_trust_store_the_token_is_read_and_not_verified(stamped):
    r = T._reading(stamped.repo, path=P, rev=stamped.c3, seal=SEAL, receipt=stamped.r_late, order=stamped.c0,
                   trust_dir="prereg-ci/no-such-dir")
    assert r["rung"] == "T1" and r["T3a"][0]["status"] == "NO_TRUST" and r["trust"] == []


def test_the_receipts_own_tokens_are_read_beside_it_for_close_after(stamped):
    r = T._reading(stamped.repo, path=P, rev=stamped.c3, seal=SEAL, receipt=stamped.r_late, order=stamped.c0,
                   receipt_path=R, receipt_bytes=receipt(m=0.7).encode("utf-8"))
    [t] = r["receipt_time"]["T3a"]
    assert t["path"] == R + ".tsr" and t["status"] == "VERIFIED"
    assert r["receipt_time"]["T1"] == r["receipt_committed_at"]
    r2 = T._reading(stamped.repo, path=P, rev=stamped.c2, seal=SEAL, receipt=stamped.r_early, order=stamped.c0,
                    receipt_path=R_EARLY, receipt_bytes=receipt(m=0.7).encode("utf-8"))
    assert r2["receipt_time"]["T3a"] == []


def test_the_anchors_file_at_the_rev_is_read_for_shape_when_it_exists(stamped):
    g = stamped.g
    line = {"n": 1, "kind": "sealed-prereg", "digest": sha(SEAL), "tx": "5" * 88}
    g.write(T._ANCHORS, json.dumps(line) + "\n")
    c4 = g.commit("an anchors file")
    r = T._reading(stamped.repo, path=P, rev=c4, seal=SEAL, receipt=stamped.r_late, order=stamped.c0)
    assert r["T3c"]["files"] == [T._ANCHORS] and r["T3c"]["lines"] == 1
    assert r["T3c"]["about_seal"][0]["status"] == "SHAPE_OK" and r["T3c"]["about_seal"][0]["chain"] == "not read on chain"
    assert "not read on chain" in r["T3c"]["detail"]
    assert "T3c 1 line" in T._describe(r)
    g("reset", "-q", "--hard", stamped.c3)


# ---- the policies -------------------------------------------------------------------------------------------

def reading(rung, *, t3a=(), openssl="OpenSSL 3.5.4", unread=(), receipt_at=None, receipt_t3a=()):
    return {"rung": rung, "name": T._RUNG_NAME.get(rung), "T3a": list(t3a), "openssl": openssl, "unread": list(unread),
            "receipt_committed_at_epoch": receipt_at, "receipt_committed_at": T._iso(receipt_at) if receipt_at else None,
            "receipt_time": {"T3a": list(receipt_t3a), "T1": T._iso(receipt_at) if receipt_at else None}}


def codes(findings):
    return [(f["code"], f["kind"]) for f in findings]


def test_under_the_default_policy_author_order_passes_and_prints_nothing():
    assert T._findings(reading("T1")) == []
    assert T._findings(reading("T3a", t3a=[{"path": "a.tsr", "status": "VERIFIED", "detail": ""}])) == []


def test_require_time_t3a_fails_a_seal_read_below_it():
    f = T._findings(reading("T1"), require_time="T3a")
    assert codes(f) == [(T._TIME_BELOW_POLICY, "FAIL")] and "AUTHOR_ORDER" in f[0]["detail"]
    assert codes(T._findings(reading("T0"), require_time="T3a")) == [(T._TIME_BELOW_POLICY, "FAIL")]
    assert T._findings(reading("T3a", t3a=[{"path": "a.tsr", "status": "VERIFIED", "detail": ""}]), require_time="T3a") == []


def test_time_below_policy_names_what_kept_the_rung_below_t3a():
    """B6 review A8: under T0, a verified token whose time precedes the receipt's was described as one
    whose time does not precede it; the suffix went on every verified token, whatever kept the rung down.
    Only T0 or a time that does not precede R keeps a verified token's seal below T3a (_rung)."""
    ahead = {"path": "p/x.tsr", "status": "VERIFIED", "detail": "", "precedes_receipt": True}
    late = {"path": "p/x.tsr", "status": "VERIFIED", "detail": "", "precedes_receipt": False}
    [f] = T._findings(reading("T0", t3a=[ahead]), require_time="T3a")
    assert (f["code"], f["kind"]) == (T._TIME_BELOW_POLICY, "FAIL") and "does not precede" not in f["detail"]
    assert f["detail"] == ("require_time is T3a and the seal reads NOT_ORDERED: the order commit is not a strict "
                           "ancestor of the receipt's commit (C3), which no token lifts; p/x.tsr verified")
    [f] = T._findings(reading("T1", t3a=[late]), require_time="T3a")
    assert f["detail"] == ("require_time is T3a and the seal reads AUTHOR_ORDER: p/x.tsr verified but its time "
                           "does not precede the receipt's")
    [f] = T._findings(reading("T0"), require_time="T3a")
    assert "not a strict ancestor" in f["detail"] and "no RFC 3161 token is committed" in f["detail"]


def test_under_t0_the_policy_finding_names_the_missing_order_through_a_repository(stamped):
    r = T._reading(stamped.repo, path=P, rev=stamped.c3, seal=SEAL, receipt=stamped.r_late, order=stamped.o)
    assert r["rung"] == "T0" and r["T3a"][0]["status"] == "VERIFIED" and r["T3a"][0]["precedes_receipt"] is True
    [f] = [x for x in T._findings(r, require_time="T3a") if x["code"] == T._TIME_BELOW_POLICY]
    assert f["kind"] == "FAIL" and "not a strict ancestor" in f["detail"] and "does not precede" not in f["detail"]
    early = T._reading(stamped.repo, path=P, rev=stamped.c3, seal=SEAL, receipt=stamped.r_early, order=stamped.c0)
    [f] = [x for x in T._findings(early, require_time="T3a") if x["code"] == T._TIME_BELOW_POLICY]
    assert "does not precede" in f["detail"] and "not a strict ancestor" not in f["detail"]


def test_a_token_that_fails_or_has_no_trust_prints_under_t1_and_is_the_fails_detail_under_t3a():
    failed = {"path": "p/x.tsr", "status": "FAILED", "detail": "message imprint mismatch"}
    f = T._findings(reading("T1", t3a=[failed]))
    assert codes(f) == [(T._TSA_TOKEN_UNVERIFIED, "NOTE")] and "p/x.tsr" in f[0]["detail"]
    f3 = T._findings(reading("T1", t3a=[failed]), require_time="T3a")
    assert codes(f3) == [(T._TSA_TOKEN_UNVERIFIED, "NOTE"), (T._TIME_BELOW_POLICY, "FAIL")]
    assert "p/x.tsr" in f3[1]["detail"]
    notrust = {"path": "p/x.tsr", "status": "NO_TRUST", "detail": "no certificate"}
    assert codes(T._findings(reading("T1", t3a=[notrust]))) == [(T._TSA_TRUST_ABSENT, "NOTE")]
    malformed = {"path": "p/x.tsr", "status": "MALFORMED", "detail": "not DER"}
    assert codes(T._findings(reading("T1", t3a=[malformed]))) == [(T._TSA_TOKEN_UNVERIFIED, "NOTE")]


def test_openssl_absent_is_uncheckable_under_t3a_and_a_note_under_t1_and_nothing_with_no_token():
    unch = {"path": "p/x.tsr", "status": "UNCHECKABLE", "detail": "openssl is not on PATH"}
    assert codes(T._findings(reading("T1", t3a=[unch], openssl=None), require_time="T3a")) == [
        (T._OPENSSL_ABSENT, "UNCHECKABLE")]
    assert codes(T._findings(reading("T1", t3a=[unch], openssl=None))) == [(T._OPENSSL_ABSENT, "NOTE")]
    assert codes(T._findings(reading("T1", openssl=None), require_time="T3a")) == [(T._TIME_BELOW_POLICY, "FAIL")]
    assert T._findings(reading("T1", openssl=None)) == []


def test_an_unread_token_path_is_a_note():
    f = T._findings(reading("T1", unread=[{"path": "p/missing.tsr", "why": "not in the tree at abc"}]))
    assert codes(f) == [(T._TIME_TOKEN_UNREAD, "NOTE")] and "p/missing.tsr" in f[0]["detail"]


def test_a_result_calling_itself_preregistered_below_t3a_is_noted_or_refused_by_policy():
    f = T._findings(reading("T1"), preregistered=True)
    assert codes(f) == [(T._PREREGISTERED_ON_AUTHOR_ORDER, "NOTE")]
    assert codes(T._findings(reading("T1"), preregistered=True, preregistered_word="fail")) == [
        (T._PREREGISTERED_ON_AUTHOR_ORDER, "FAIL")]
    assert T._findings(reading("T3a", t3a=[{"path": "a", "status": "VERIFIED", "detail": ""}]), preregistered=True) == []
    assert codes(T._findings(reading("T0"), preregistered=True)) == [(T._PREREGISTERED_ON_AUTHOR_ORDER, "NOTE")]


def test_close_after_is_printed_against_the_receipts_time_and_decides_only_at_t3a_under_t3a():
    late = {"path": "p/r.json.tsr", "status": "VERIFIED", "detail": "", "gen_time_epoch": 1_800_000_000,
            "gen_time": T._iso(1_800_000_000)}
    close = "2026-01-01T00:00:00Z"
    f = T._findings(reading("T1", receipt_at=1_800_000_000), close_after=close)
    assert codes(f) == [(T._RECEIPT_AFTER_CLOSE, "NOTE")] and "T1" in f[0]["detail"]
    f = T._findings(reading("T1", receipt_at=1_800_000_000, receipt_t3a=[late]), close_after=close)
    assert codes(f) == [(T._RECEIPT_AFTER_CLOSE, "NOTE")] and "T3a" in f[0]["detail"]
    f = T._findings(reading("T1", receipt_at=1_800_000_000, receipt_t3a=[late]), close_after=close, require_time="T3a")
    assert (T._RECEIPT_AFTER_CLOSE, "FAIL") in codes(f)
    assert T._findings(reading("T1", receipt_at=1_700_000_000), close_after=close) == []
    assert T._findings(reading("T1"), close_after=close) == [], "no receipt: nothing to read against it"
    with pytest.raises(ValueError):
        T._findings(reading("T1", receipt_at=1), close_after="2026-01-01")


def test_a_policy_value_the_design_does_not_name_is_a_caller_error():
    with pytest.raises(ValueError):
        T._findings(reading("T1"), require_time="T2")
    with pytest.raises(ValueError):
        T._findings(reading("T1"), preregistered_word="warn")


def test_iso_renders_utc_seconds_and_parses_them_back():
    assert T._iso(0) == "1970-01-01T00:00:00Z"
    assert T._iso(1_761_400_000) == "2025-10-25T13:46:40Z"
    assert T._epoch("2025-10-25T13:46:40Z") == 1_761_400_000
    assert T._epoch("2025-10-25 14:26:40") is None and T._epoch("") is None and T._epoch(None) is None


def test_iso_renders_every_year_a_datetime_holds_and_prints_any_other_time_as_git_writes_it():
    """B6 review A4: a committer time of 99999999999999 (git records it as written) raised OSError out of
    _iso, which went through the platform's gmtime; on the Windows box this was written on that raised
    for any time before 1969-12-31T12:00:00Z or after 3001-01-19T21:59:59Z, so a genTime of year 0001
    raised too. Years 1 to 9999 print with four digits; any other time prints as ``@<seconds>``, git's
    own form, and is still compared as the number it is."""
    for epoch, text in ((-62_135_596_800, "0001-01-01T00:00:00Z"), (-1, "1969-12-31T23:59:59Z"),
                        (-43_201, "1969-12-31T11:59:59Z"), (32_536_850_400, "3001-01-19T22:00:00Z"),
                        (32_503_680_000, "3000-01-01T00:00:00Z"), (253_402_300_799, "9999-12-31T23:59:59Z")):
        assert T._iso(epoch) == text and T._epoch(text) == epoch
    for epoch in (253_402_300_800, 99_999_999_999_999, 9_999_999_999_999_999_999, -62_135_596_801):
        assert T._iso(epoch) == f"@{epoch}" and T._epoch(T._iso(epoch)) is None


@pytest.mark.parametrize("gen_time, text", [(b"00010101000000Z", "0001-01-01T00:00:00Z"),
                                            (b"99991231235959Z", "9999-12-31T23:59:59Z")])
def test_a_token_whose_gen_time_is_year_one_or_9999_reads_and_prints(gen_time, text):
    info = T._tstinfo(hand_token(gen_time))
    assert info["gen_time"] == text and info["gen_time_epoch"] == T._epoch(text)
    r = T._rfc3161(SEAL, hand_token(gen_time), None)
    assert r["status"] == "NO_TRUST" and r["gen_time"] == text


def commit_dated(g, msg: str, committer: str) -> str:
    """Commit everything with the committer date given as git reads it (``@<seconds> +0000``)."""
    g("add", "-A")
    env = {k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")}
    env.update(GIT_AUTHOR_DATE="1760000000 +0000", GIT_COMMITTER_DATE=committer)
    done = subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", msg], cwd=str(g.root), env=env,
                          capture_output=True, stdin=subprocess.DEVNULL)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    return g("rev-parse", "HEAD")


@pytest.mark.parametrize("at, text", [(99_999_999_999_999, "@99999999999999"), (0, "1970-01-01T00:00:00Z")])
def test_a_receipt_commit_dated_past_year_9999_or_at_the_epoch_reads_its_time(tmp_path, at, text):
    """B6 review A4 through a repository: the receipt's time prints, the rung reads, close_after prints it.
    At 0 the time printed None before, read as if there were no receipt to time."""
    g = Git(tmp_path / "r")
    g.write(P, SEAL)
    c0 = g.commit("seal the plan")
    g.write(R, receipt(m=0.7))
    rc = commit_dated(g, "a result, dated by hand", f"@{at} +0000")
    with G._Repo(g.root) as repo:
        if repo.commit_time(rc) != at:
            pytest.skip(f"this git recorded committer time {repo.commit_time(rc)}, not {at}")
        r = T._reading(repo, path=P, rev=rc, seal=SEAL, order=c0, receipt=rc, trust_rev=c0)
    assert r["receipt_committed_at_epoch"] == at and r["receipt_committed_at"] == text
    assert r["receipt_time"]["T1"] == text and r["rung"] == "T1"
    f = T._findings(r, close_after="1971-01-01T00:00:00Z")
    if at:
        assert codes(f) == [(T._RECEIPT_AFTER_CLOSE, "NOTE")] and text in f[0]["detail"]
    else:
        assert f == []


# ---- stamp: the one networked command, which refuses without --go ------------------------------------------

@pytest.fixture
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a network call was made without --go")
    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(urllib.request, "urlopen", boom)


def test_stamp_without_go_refuses_writes_nothing_and_opens_no_socket(stamped, no_network, monkeypatch, capsys):
    g = stamped.g
    monkeypatch.chdir(g.root)
    before = set(p.relative_to(g.root).as_posix() for p in g.root.rglob("*") if p.is_file() and ".git" not in p.parts)
    rc = cli.main(["stamp", P, "--rfc3161", "http://127.0.0.1:9/tsa", "--commit", stamped.c0])
    out = capsys.readouterr().out
    assert rc == 2 and "--go" in out and sha(SEAL) in out and "http://127.0.0.1:9/tsa" in out
    assert P + ".tsr" in out
    rc = cli.main(["stamp", P, "--ots", "--commit", stamped.c0])
    out = capsys.readouterr().out
    assert rc == 2 and "--go" in out and P + ".ots" in out
    after = set(p.relative_to(g.root).as_posix() for p in g.root.rglob("*") if p.is_file() and ".git" not in p.parts)
    assert after == before
    assert g("status", "--short") == ""


def test_stamp_wants_exactly_one_of_rfc3161_and_ots(stamped, no_network, monkeypatch, capsys):
    monkeypatch.chdir(stamped.g.root)
    assert cli.main(["stamp", P, "--commit", stamped.c0]) == 2
    assert "--rfc3161" in capsys.readouterr().out
    assert cli.main(["stamp", P, "--rfc3161", "http://127.0.0.1:9/", "--ots", "--commit", stamped.c0]) == 2
    assert "one of" in capsys.readouterr().out


def test_stamp_reads_the_blob_at_the_commit_not_the_working_copy(stamped, no_network, monkeypatch, capsys):
    g = stamped.g
    monkeypatch.chdir(g.root)
    (g.root / "p" / "PREREG_a.md").write_bytes(SEAL.replace(b"\n", b"\r\n"))
    try:
        rc = cli.main(["stamp", P, "--rfc3161", "http://127.0.0.1:9/tsa", "--commit", stamped.c0])
        out = capsys.readouterr().out
        assert rc == 2 and sha(SEAL) in out and sha(SEAL.replace(b"\n", b"\r\n")) not in out
    finally:
        g("checkout", "--", P)


def test_stamp_refuses_a_path_that_is_not_a_blob_at_the_commit(stamped, no_network, monkeypatch, capsys):
    monkeypatch.chdir(stamped.g.root)
    assert cli.main(["stamp", "p/no_such.md", "--ots", "--commit", stamped.c0]) == 2
    assert "p/no_such.md" in capsys.readouterr().out


def test_the_stamp_function_refuses_without_go_and_says_what_it_would_send(stamped, no_network):
    r = T._stamp(stamped.repo, P, rfc3161="http://127.0.0.1:9/tsa", commit=stamped.c0)
    assert r["status"] == "REFUSED" and r["sha256"] == sha(SEAL) and r["writes"] == P + ".tsr"
    assert r["to"] == "http://127.0.0.1:9/tsa" and "--go" in r["detail"] and r["commit"] == stamped.c0
    r = T._stamp(stamped.repo, P, ots=True, commit=stamped.c0)
    assert r["status"] == "REFUSED" and r["writes"] == P + ".ots" and "OpenTimestamps" in r["to"]
    with pytest.raises(ValueError):
        T._stamp(stamped.repo, P, commit=stamped.c0)
    with pytest.raises(ValueError):
        T._stamp(stamped.repo, P, rfc3161="x", ots=True, commit=stamped.c0)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://127.0.0.1/tsa", "127.0.0.1:9/tsa", "http:///tsa",
                                 "javascript:x"])
def test_stamp_refuses_a_tsa_url_that_is_not_http_or_https_with_a_host(stamped, no_network, monkeypatch, capsys,
                                                                       url):
    with pytest.raises(T._StampError) as e:
        T._stamp(stamped.repo, P, rfc3161=url, commit=stamped.c0)
    assert "http" in str(e.value)
    monkeypatch.chdir(stamped.g.root)
    assert cli.main(["stamp", P, "--rfc3161", url, "--commit", stamped.c0]) == 2
    assert "REFUSED" in capsys.readouterr().out


@needs_openssl
def test_the_query_nonce_is_read_from_a_query_openssl_makes_here(tsa):
    """stamp holds a TSA's response to its request's nonce; the reader is pinned on queries openssl
    writes locally (ts -query sends nothing)."""
    (tsa.root / "n.bin").write_bytes(SEAL)
    tsa.run("ts", "-query", "-data", "n.bin", "-sha256", "-cert", "-out", "n.tsq")
    nonce = T._query_nonce((tsa.root / "n.tsq").read_bytes())
    text = tsa.run("ts", "-query", "-in", "n.tsq", "-text").decode("utf-8", "replace")
    [printed] = [ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.strip().startswith("Nonce:")]
    assert isinstance(nonce, int) and nonce == int(printed, 16)
    tsa.run("ts", "-query", "-data", "n.bin", "-sha256", "-no_nonce", "-out", "n0.tsq")
    assert T._query_nonce((tsa.root / "n0.tsq").read_bytes()) is None
    assert T._query_nonce(b"junk") is None


def test_no_prereg_ci_test_runs_stamp_with_go():
    """The stamp command exists and is never run with --go in this build (DESIGN 9: the operator's
    act): no call in any prereg-ci test passes ``--go`` or ``go=`` anything but False."""
    files = sorted((ROOT / "tests").glob("test_prereg_ci*.py"))
    assert Path(__file__).resolve() in [f.resolve() for f in files]
    for f in files:
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                assert kw.arg != "go" or (isinstance(kw.value, ast.Constant) and kw.value.value is False), (
                    f.name, node.lineno)
            for a in node.args:
                if isinstance(a, (ast.List, ast.Tuple)):
                    assert not any(isinstance(e, ast.Constant) and e.value == "--go" for e in a.elts), (
                        f.name, node.lineno)
