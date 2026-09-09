"""Tests for styxx.v8.keys — the six items of INTERFACES_foundations.md plus properties.

Every pinned number is asserted.  The node interop test fails (never skips) when
``node`` is not on PATH.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from hypothesis import example, given, settings
from hypothesis import strategies as st

from styxx.v8 import keys

# ------------------------------------------------------------------ RFC 8032 §7.1

# (secret seed, public key, message, signature) — TEST 1 and TEST 2 verbatim.
RFC8032_VECTORS = [
    (
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
        "",
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
    ),
    (
        "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
        "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
        "72",
        "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00",
    ),
]


@pytest.mark.parametrize("seed_hex, pub_hex, msg_hex, sig_hex", RFC8032_VECTORS)
def test_rfc8032_vectors(seed_hex, pub_hex, msg_hex, sig_hex):
    seed = bytes.fromhex(seed_hex)
    pub = bytes.fromhex(pub_hex)
    msg = bytes.fromhex(msg_hex)
    sig = bytes.fromhex(sig_hex)
    assert keys.public_from_private(seed) == pub
    assert keys.sign(seed, msg) == sig
    assert len(sig) == 64
    assert keys.verify(pub, msg, sig) is True
    # Determinism: pure Ed25519 signatures do not depend on randomness.
    assert keys.sign(seed, msg) == keys.sign(seed, msg)


def test_generate_yields_valid_pair():
    seed, pub = keys.generate()
    assert len(seed) == 32 and len(pub) == 32
    assert keys.public_from_private(seed) == pub
    sig = keys.sign(seed, b"hello")
    assert keys.verify(pub, b"hello", sig) is True
    seed2, pub2 = keys.generate()
    assert seed2 != seed and pub2 != pub


def test_public_from_private_rejects_bad_seed():
    with pytest.raises(ValueError):
        keys.public_from_private(b"\x00" * 31)
    with pytest.raises(ValueError):
        keys.public_from_private(b"\x00" * 33)
    with pytest.raises(TypeError):
        keys.public_from_private("9d61" * 16)  # type: ignore[arg-type]


# ------------------------------------------------------------------ 2. wire round trips

def test_encode_public_shape():
    seed = bytes.fromhex(RFC8032_VECTORS[0][0])
    pub = keys.public_from_private(seed)
    s = keys.encode_public(pub)
    assert s.startswith("ed25519:")
    body = s[len("ed25519:"):]
    assert len(body) == 43           # ceil(32 * 8 / 6) without padding
    assert "=" not in body
    assert "+" not in body and "/" not in body
    # Derived by hand from the RFC 8032 TEST 1 public key.
    assert s == "ed25519:11qYAYKxCrfVS_7TyWQHOg7hcvPapiMlrwIaaPcHURo"
    assert keys.decode_public(s) == pub


def test_encode_signature_shape():
    sig = bytes.fromhex(RFC8032_VECTORS[0][3])
    s = keys.encode_signature(sig)
    assert s.startswith("ed25519:")
    body = s[len("ed25519:"):]
    assert len(body) == 86           # ceil(64 * 8 / 6) without padding
    assert "=" not in body
    assert keys.decode_signature(s) == sig


def test_decode_rejects_padding():
    pub = b"\x01" * 32
    with pytest.raises(ValueError):
        keys.decode_public(keys.encode_public(pub) + "=")
    sig = b"\x02" * 64
    with pytest.raises(ValueError):
        keys.decode_signature(keys.encode_signature(sig) + "==")


def test_decode_rejects_wrong_length():
    # A 31-byte and a 33-byte payload under the right prefix.
    import base64
    for n in (31, 33, 0, 64):
        body = base64.urlsafe_b64encode(b"\x07" * n).decode().rstrip("=")
        with pytest.raises(ValueError):
            keys.decode_public("ed25519:" + body)
    for n in (63, 65, 32):
        body = base64.urlsafe_b64encode(b"\x07" * n).decode().rstrip("=")
        with pytest.raises(ValueError):
            keys.decode_signature("ed25519:" + body)


def test_decode_rejects_bad_prefix_and_alphabet():
    pub = b"\x03" * 32
    good = keys.encode_public(pub)
    body = good[len("ed25519:"):]
    for bad in ("ED25519:" + body, "ed25519-" + body, body, "rsa:" + body, "ed25519:"):
        with pytest.raises(ValueError):
            keys.decode_public(bad)
    # Standard-base64 alphabet characters are not base64url.
    with pytest.raises(ValueError):
        keys.decode_public("ed25519:" + body[:-1] + "+")
    with pytest.raises(ValueError):
        keys.decode_public("ed25519:" + body[:-1] + "/")
    with pytest.raises(ValueError):
        keys.decode_public("ed25519:" + body + "\n")
    with pytest.raises(ValueError):
        keys.decode_public("ed25519:" + body[:-1] + "é")


def test_decode_rejects_non_canonical_trailing_bits():
    # 32 bytes = 256 bits = 42 sextets + 4 bits; the 43rd character carries two
    # zero bits.  Setting them decodes to the same bytes under a lenient decoder,
    # so two different strings would name one key.  Refuse the non-canonical one.
    good = keys.encode_public(b"\x00" * 32)
    assert good.endswith("A")
    with pytest.raises(ValueError):
        keys.decode_public(good[:-1] + "B")


def test_encode_rejects_wrong_length():
    with pytest.raises(ValueError):
        keys.encode_public(b"\x00" * 31)
    with pytest.raises(ValueError):
        keys.encode_signature(b"\x00" * 63)
    with pytest.raises(TypeError):
        keys.encode_public("x" * 32)  # type: ignore[arg-type]


@given(st.binary(min_size=32, max_size=32))
@settings(max_examples=200, deadline=None)
def test_public_round_trip_property(raw):
    # The wire layer round-trips every 32-byte string; decode_public additionally
    # validates the point (2026-09-08), so the round trip completes iff
    # validate_public accepts the bytes, and otherwise raises ValueError.
    s = keys.encode_public(raw)
    try:
        keys.validate_public(raw)
    except ValueError:
        with pytest.raises(ValueError):
            keys.decode_public(s)
        return
    assert keys.decode_public(s) == raw


@given(st.binary(min_size=32, max_size=32))
@settings(max_examples=200, deadline=None)
def test_public_round_trip_property_on_derived_keys(seed):
    pub = keys.public_from_private(seed)
    assert keys.decode_public(keys.encode_public(pub)) == pub


@given(st.binary(min_size=64, max_size=64))
@settings(max_examples=200, deadline=None)
def test_signature_round_trip_property(raw):
    assert keys.decode_signature(keys.encode_signature(raw)) == raw


# ------------------------------------------------------------------ 3. verify never raises

SEED_A = bytes.fromhex(RFC8032_VECTORS[0][0])
PUB_A = bytes.fromhex(RFC8032_VECTORS[0][1])
SEED_B = bytes.fromhex(RFC8032_VECTORS[1][0])
PUB_B = bytes.fromhex(RFC8032_VECTORS[1][1])


def test_verify_false_on_wrong_key():
    msg = b"the message"
    sig = keys.sign(SEED_A, msg)
    assert keys.verify(PUB_A, msg, sig) is True
    assert keys.verify(PUB_B, msg, sig) is False


def test_verify_false_on_wrong_message():
    sig = keys.sign(SEED_A, b"the message")
    assert keys.verify(PUB_A, b"the message.", sig) is False
    assert keys.verify(PUB_A, b"", sig) is False


def test_verify_false_on_every_flipped_signature_byte():
    msg = b"flip"
    sig = keys.sign(SEED_A, msg)
    for i in range(64):
        bad = bytearray(sig)
        bad[i] ^= 0x01
        assert keys.verify(PUB_A, msg, bytes(bad)) is False, i


def test_verify_false_on_malformed_lengths():
    msg = b"m"
    sig = keys.sign(SEED_A, msg)
    assert keys.verify(PUB_A, msg, sig[:63]) is False
    assert keys.verify(PUB_A, msg, sig + b"\x00") is False
    assert keys.verify(PUB_A[:31], msg, sig) is False
    assert keys.verify(PUB_A + b"\x00", msg, sig) is False
    assert keys.verify(b"", msg, sig) is False
    assert keys.verify(PUB_A, msg, b"") is False


def test_verify_false_on_wrong_types():
    msg = b"m"
    sig = keys.sign(SEED_A, msg)
    assert keys.verify(keys.encode_public(PUB_A), msg, sig) is False  # type: ignore[arg-type]
    assert keys.verify(PUB_A, "m", sig) is False  # type: ignore[arg-type]
    assert keys.verify(PUB_A, msg, keys.encode_signature(sig)) is False  # type: ignore[arg-type]
    assert keys.verify(None, None, None) is False  # type: ignore[arg-type]


def test_verify_false_on_key_that_is_not_a_curve_point():
    # 32 bytes of 0xff is not a canonical encoding of a point on the curve.
    msg = b"m"
    sig = keys.sign(SEED_A, msg)
    assert keys.verify(b"\xff" * 32, msg, sig) is False


def test_sign_rejects_bad_inputs():
    with pytest.raises(ValueError):
        keys.sign(SEED_A[:31], b"m")
    with pytest.raises(TypeError):
        keys.sign(SEED_A, "m")  # type: ignore[arg-type]


@given(st.binary(min_size=32, max_size=32), st.binary(max_size=256))
@settings(max_examples=100, deadline=None)
def test_sign_verify_property(seed, msg):
    pub = keys.public_from_private(seed)
    sig = keys.sign(seed, msg)
    assert len(sig) == 64
    assert keys.verify(pub, msg, sig) is True
    assert keys.verify(pub, msg + b"\x00", sig) is False


# ------------------------------------------------------------------ 4. tagged

def test_tagged_exact_preimage():
    d = bytes(range(32))
    assert keys.tagged("styxx.v8/cert/1", d) == b"styxx.v8/cert/1\x00" + d
    assert keys.tagged("styxx.v8/sth/1", d) == b"styxx.v8/sth/1\x00" + d


def test_tagged_distinct_tags_distinct_preimages():
    d = b"\xab" * 32
    a = keys.tagged("styxx.v8/cert/1", d)
    b = keys.tagged("styxx.v8/sth/1", d)
    assert a != b
    assert a[-32:] == b[-32:] == d
    # A tag cannot be confused with a prefix of another: the NUL separator sits at
    # exactly one position, so the tag is recoverable from the preimage.
    assert a.index(b"\x00") == len("styxx.v8/cert/1")
    assert b.index(b"\x00") == len("styxx.v8/sth/1")


def test_tagged_refuses_empty_and_nul():
    d = b"\x00" * 32
    with pytest.raises(ValueError):
        keys.tagged("", d)
    with pytest.raises(ValueError):
        keys.tagged("styxx.v8/cert/1\x00", d)
    with pytest.raises(ValueError):
        keys.tagged("\x00styxx", d)
    with pytest.raises(ValueError):
        keys.tagged("a\x00b", d)


def test_tagged_refuses_non_ascii_and_bad_digest():
    with pytest.raises(ValueError):
        keys.tagged("styxx€", b"\x00" * 32)
    with pytest.raises(ValueError):
        keys.tagged("styxx.v8/cert/1", b"\x00" * 31)
    with pytest.raises(ValueError):
        keys.tagged("styxx.v8/cert/1", b"\x00" * 33)
    with pytest.raises(TypeError):
        keys.tagged(b"styxx.v8/cert/1", b"\x00" * 32)  # type: ignore[arg-type]


# ------------------------------------------------------------------ 5. node interop

NODE_VERIFY = (
    'const c=require("crypto");'
    'const [m,p,s]=process.argv.slice(1).map(h=>Buffer.from(h,"hex"));'
    'const r=c.verify(null,m,{key:Buffer.concat([Buffer.from("302a300506032b6570032100","hex"),p]),'
    'format:"der",type:"spki"},s);'
    'process.stdout.write(String(r));'
)


def _node_verify(msg: bytes, pub: bytes, sig: bytes) -> str:
    node = shutil.which("node")
    if node is None:
        pytest.fail("node is not on PATH; the interop test cannot run (this is a failure, not a skip)")
    proc = subprocess.run(
        [node, "-e", NODE_VERIFY, msg.hex(), pub.hex(), sig.hex()],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_node_verifies_python_signature():
    seed, pub = keys.generate()
    msg = b"styxx.v8 interop \xe2\x82\xac \x00 bytes"
    sig = keys.sign(seed, msg)
    assert _node_verify(msg, pub, sig) == "true"
    flipped = bytearray(sig)
    flipped[7] ^= 0x80
    assert _node_verify(msg, pub, bytes(flipped)) == "false"
    assert _node_verify(msg + b"!", pub, sig) == "false"


def test_node_verifies_rfc8032_vectors():
    for _seed, pub_hex, msg_hex, sig_hex in RFC8032_VECTORS:
        assert _node_verify(bytes.fromhex(msg_hex), bytes.fromhex(pub_hex), bytes.fromhex(sig_hex)) == "true"


def test_node_version_is_present():
    node = shutil.which("node")
    if node is None:
        pytest.fail("node is not on PATH")
    proc = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0
    assert proc.stdout.startswith("v")


# ------------------------------------------------------------------ 6. PEM / public files

def test_private_pem_round_trip(tmp_path):
    seed, pub = keys.generate()
    path = tmp_path / "signer.pem"
    keys.save_private_pem(seed, path)
    raw = path.read_bytes()
    assert raw.startswith(b"-----BEGIN PRIVATE KEY-----\n")
    assert raw.endswith(b"-----END PRIVATE KEY-----\n")
    assert b"\r" not in raw
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert keys.load_private_pem(path) == seed
    # The file is a PKCS#8 the library itself reads back to the same key.
    key = serialization.load_pem_private_key(raw, password=None)
    assert isinstance(key, Ed25519PrivateKey)
    assert key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw) == pub


def test_private_pem_rfc8032_seed_round_trip(tmp_path):
    path = tmp_path / "rfc.pem"
    keys.save_private_pem(SEED_A, path)
    assert keys.load_private_pem(path) == SEED_A
    assert keys.public_from_private(keys.load_private_pem(path)) == PUB_A


def test_private_pem_mode_best_effort(tmp_path, monkeypatch):
    seed, _ = keys.generate()
    path = tmp_path / "k.pem"
    keys.save_private_pem(seed, path)
    if os.name != "posix":
        # Windows: no ACL is set (documented); the file exists and reads back.
        assert keys.load_private_pem(path) == seed
        return
    assert (path.stat().st_mode & 0o777) == 0o600
    # The mode comes from creation (O_CREAT with 0600), not from a chmod after the
    # write: with a permissive umask and fchmod disabled the file is still 0600.
    def no_fchmod(fd, mode):
        raise OSError("fchmod disabled for the test")
    old = os.umask(0)
    try:
        monkeypatch.setattr(os, "fchmod", no_fchmod)
        p2 = tmp_path / "k2.pem"
        keys.save_private_pem(seed, p2)
        assert (p2.stat().st_mode & 0o777) == 0o600
    finally:
        os.umask(old)


def test_load_private_pem_rejects_garbage(tmp_path):
    path = tmp_path / "bad.pem"
    path.write_bytes(b"-----BEGIN PRIVATE KEY-----\nnope\n-----END PRIVATE KEY-----\n")
    with pytest.raises(ValueError):
        keys.load_private_pem(path)
    with pytest.raises(FileNotFoundError):
        keys.load_private_pem(tmp_path / "missing.pem")


def test_public_file_is_exactly_encoded_string_plus_lf(tmp_path):
    seed, pub = keys.generate()
    path = tmp_path / "signer.pub"
    keys.save_public(pub, path)
    raw = path.read_bytes()
    assert raw == keys.encode_public(pub).encode("ascii") + b"\n"
    assert raw.count(b"\n") == 1
    assert b"\r" not in raw
    assert keys.load_public(path) == pub


def test_load_public_tolerates_crlf_but_not_bom_or_padding(tmp_path):
    _, pub = keys.generate()
    s = keys.encode_public(pub)
    p = tmp_path / "crlf.pub"
    p.write_bytes(s.encode("ascii") + b"\r\n")
    assert keys.load_public(p) == pub
    p2 = tmp_path / "bom.pub"
    p2.write_bytes(b"\xef\xbb\xbf" + s.encode("ascii") + b"\n")
    with pytest.raises(ValueError):
        keys.load_public(p2)
    p3 = tmp_path / "pad.pub"
    p3.write_bytes(s.encode("ascii") + b"=\n")
    with pytest.raises(ValueError):
        keys.load_public(p3)


def test_files_accept_str_paths(tmp_path):
    seed, pub = keys.generate()
    priv = str(tmp_path / "s.pem")
    pubf = str(tmp_path / "s.pub")
    keys.save_private_pem(seed, priv)
    keys.save_public(pub, pubf)
    assert keys.load_private_pem(priv) == seed
    assert keys.load_public(pubf) == pub


# ================================================================== hostile review
# Appended by the hostile tester (2026-09-08).  Nothing above this line was edited.

import array
import base64
import sys

from cryptography.hazmat.primitives.asymmetric import ed448, rsa, x25519
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

# RFC 8032 §7.1 TEST 3 (message af82).  Not in the contract; recalled and then checked
# against cryptography directly in the test below, never against keys.py alone.
RFC8032_TEST3 = (
    "c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
    "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
    "af82",
    "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a",
)

ED25519_L = 2**252 + 27742317777372353535851937790883648493
ED25519_P = 2**255 - 19
IDENTITY_POINT = bytes([1]) + bytes(31)                 # y = 1, x = 0: the neutral element
ORDER2_POINT = (ED25519_P - 1).to_bytes(32, "little")    # y = -1: the point of order 2
B64URL_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"


@pytest.mark.parametrize("seed_hex, pub_hex, msg_hex, sig_hex", RFC8032_VECTORS + [RFC8032_TEST3])
def test_hostile_rfc8032_vectors_independent_of_module(seed_hex, pub_hex, msg_hex, sig_hex):
    # The same vectors through cryptography directly, bypassing keys.py, so the
    # module's agreement with the RFC is not the module checking itself.
    key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed_hex))
    pub = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    assert pub == bytes.fromhex(pub_hex)
    assert key.sign(bytes.fromhex(msg_hex)) == bytes.fromhex(sig_hex)
    Ed25519PublicKey.from_public_bytes(pub).verify(bytes.fromhex(sig_hex), bytes.fromhex(msg_hex))
    # And the module agrees with that independent computation.
    assert keys.public_from_private(bytes.fromhex(seed_hex)) == pub
    assert keys.sign(bytes.fromhex(seed_hex), bytes.fromhex(msg_hex)) == bytes.fromhex(sig_hex)
    assert keys.verify(pub, bytes.fromhex(msg_hex), bytes.fromhex(sig_hex)) is True


def test_hostile_public_from_private_extreme_seeds():
    # All-zero seed: a well-known value (clamped SHA-512(0^32) times B).
    assert keys.public_from_private(bytes(32)).hex() == (
        "3b6a27bcceb6a42d62a3a8d02a6f0d73653215771de243a63ac048a18b59da29"
    )
    for seed in (bytes(32), b"\xff" * 32, bytes(range(32))):
        independent = Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        assert keys.public_from_private(seed) == independent


# ------------------------------------------------------------------ verify never raises

class _LyingBytes(bytes):
    """A bytes subclass whose __len__ lies; len() says 32, the buffer holds 5."""
    def __len__(self):
        return 32


class _B(bytes):
    pass


def test_hostile_verify_never_raises_on_garbage_types():
    sig = keys.sign(SEED_A, b"m")
    released = memoryview(bytearray(32))
    released.release()
    garbage_keys = [
        True, 32, 32.0, _LyingBytes(b"\x00" * 5), released,
        array.array("B", [0] * 32), array.array("I", [0] * 8), list(PUB_A), tuple(PUB_A),
        {"k": PUB_A}, PUB_A.hex(), object(), type, lambda: PUB_A,
    ]
    for k in garbage_keys:
        assert keys.verify(k, b"m", sig) is False, repr(k)
        assert keys.verify(PUB_A, k, sig) is False, repr(k)
        assert keys.verify(PUB_A, b"m", k) is False, repr(k)
    # Non-contiguous memoryview as signature: 64 items strided out of 128 bytes.
    assert keys.verify(PUB_A, b"m", memoryview(bytearray(128))[::2]) is False
    # Huge message with a signature over something else.
    assert keys.verify(PUB_A, b"\x00" * (4 * 1024 * 1024), sig) is False


def test_hostile_verify_accepts_bytes_like_variants():
    sig = keys.sign(SEED_A, b"m")
    assert keys.verify(_B(PUB_A), _B(b"m"), _B(sig)) is True
    assert keys.verify(bytearray(PUB_A), bytearray(b"m"), bytearray(sig)) is True
    assert keys.verify(memoryview(PUB_A), memoryview(b"m"), memoryview(sig)) is True


@given(st.binary(max_size=80), st.binary(max_size=80), st.binary(max_size=80))
@settings(max_examples=300, deadline=None)
def test_hostile_verify_never_raises_property(k, m, s):
    r = keys.verify(k, m, s)
    assert r is False or r is True
    if len(k) != 32 or len(s) != 64:
        assert r is False


def test_hostile_sign_accepts_bytes_like_and_refuses_others():
    assert keys.sign(bytearray(SEED_A), b"") == bytes.fromhex(RFC8032_VECTORS[0][3])
    assert keys.sign(_B(SEED_A), _B(b"")) == bytes.fromhex(RFC8032_VECTORS[0][3])
    assert keys.sign(memoryview(SEED_A), memoryview(b"")) == bytes.fromhex(RFC8032_VECTORS[0][3])
    # A strided memoryview of the message is copied, not misread.
    strided = memoryview(bytearray(b"\x00\xff" * 8))[::2]
    assert keys.sign(SEED_A, strided) == keys.sign(SEED_A, bytes(8))
    with pytest.raises(TypeError):
        keys.sign(True, b"")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        keys.sign(SEED_A, array.array("B", [0]))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        keys.sign(_LyingBytes(b"\x00" * 5), b"")


# ------------------------------------------------------------------ RFC 8032 §5.1.7 boundaries

def test_hostile_verify_rejects_s_plus_l_malleation():
    # RFC 8032 §5.1.7: reject unless 0 <= S < L.  S + L still fits in 32 bytes for
    # the TEST 1 signature and would pass a verifier that reduces S mod L.
    sig = bytes.fromhex(RFC8032_VECTORS[0][3])
    r, s = sig[:32], int.from_bytes(sig[32:], "little")
    assert s + ED25519_L < 2**256
    malleated = r + (s + ED25519_L).to_bytes(32, "little")
    assert keys.verify(PUB_A, b"", malleated) is False
    high_bit = r + (s | (1 << 255)).to_bytes(32, "little")
    assert keys.verify(PUB_A, b"", high_bit) is False
    assert keys.verify(PUB_A, b"", r + bytes(32)) is False


def test_hostile_small_order_public_key_is_refused():
    # Repaired 2026-09-08: validate_public refuses the eight small-order points, so
    # the "universal" signature under the identity key verifies on no message.
    universal = IDENTITY_POINT + bytes(32)      # R = identity, S = 0
    assert keys.verify(IDENTITY_POINT, b"cert A", universal) is False
    assert keys.verify(IDENTITY_POINT, b"cert B", universal) is False
    assert keys.verify(ORDER2_POINT, b"cert A", ORDER2_POINT + bytes(32)) is False


def _backend_verifies(pub: bytes, msg: bytes, sig: bytes) -> bool:
    """The verdict of ``cryptography`` alone, bypassing keys.py's key validation."""
    try:
        Ed25519PublicKey.from_public_bytes(pub).verify(sig, msg)
        return True
    except Exception:
        return False


def test_hostile_small_order_public_key_current_behaviour_is_documented():
    # The receipt behind the repair: what the BACKEND does, asserted so a change in
    # the backend is noticed, next to what the module now does (refuse).
    universal = IDENTITY_POINT + bytes(32)
    assert _backend_verifies(IDENTITY_POINT, b"cert A", universal) is True
    assert _backend_verifies(IDENTITY_POINT, b"cert B", universal) is True
    assert keys.verify(IDENTITY_POINT, b"cert A", universal) is False
    assert keys.verify(IDENTITY_POINT, b"cert B", universal) is False
    # The order-2 point (0, -1) as both A and R with S = 0 is NOT universal for the
    # backend: the RFC 8032 §5.1.7 check [S]B == R + [k]A reduces to A + [k]A ==
    # identity, which holds iff k = SHA-512(R || A || M) mod L is odd.  A
    # cofactorless verifier (the backend) accepts it on a message-dependent parity;
    # a cofactored verifier would accept it on all of them; the module accepts it on
    # none.  Derived and asserted per message rather than pinned as a constant.
    import hashlib
    order2_sig = ORDER2_POINT + bytes(32)
    outcomes = set()
    for i in range(64):
        msg = b"m%d" % i
        k = int.from_bytes(hashlib.sha512(ORDER2_POINT + ORDER2_POINT + msg).digest(), "little") % ED25519_L
        expected = (k % 2 == 1)
        got = _backend_verifies(ORDER2_POINT, msg, order2_sig)
        assert got is expected, (msg, k % 2, got)
        outcomes.add(got)
        assert keys.verify(ORDER2_POINT, msg, order2_sig) is False
    assert outcomes == {True, False}
    # Encoding still frames the bytes (it does not validate); decoding refuses them.
    s = keys.encode_public(IDENTITY_POINT)
    assert s == "ed25519:AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    with pytest.raises(ValueError, match="small-order"):
        keys.decode_public(s)
    with pytest.raises(ValueError, match="small-order"):
        keys.validate_public(IDENTITY_POINT)


def test_hostile_non_canonical_public_key_encoding():
    # RFC 8032 §5.1.3: decoding y >= p fails.  y = p + 1 is the identity written
    # non-canonically; the backend reduces it and accepts (pinned so a stricter
    # backend is noticed); the module refuses it before the backend sees it.
    non_canonical = (ED25519_P + 1).to_bytes(32, "little")
    assert non_canonical != IDENTITY_POINT
    assert _backend_verifies(non_canonical, b"m", IDENTITY_POINT + bytes(32)) is True
    assert keys.verify(non_canonical, b"m", IDENTITY_POINT + bytes(32)) is False
    with pytest.raises(ValueError, match="y >= p"):
        keys.validate_public(non_canonical)
    assert keys.encode_public(non_canonical) != keys.encode_public(IDENTITY_POINT)
    with pytest.raises(ValueError):
        keys.decode_public(keys.encode_public(non_canonical))


# ------------------------------------------------------------------ wire decoding, hostile

def test_hostile_decode_wrong_type_is_typeerror_not_valueerror():
    # The contract names ValueError for prefix/length/padding and says nothing about
    # non-str input; the module raises TypeError.  Pinned so the choice is visible.
    for bad in (None, True, 7, b"ed25519:" + b"A" * 43, ["ed25519:"], 1.5):
        with pytest.raises(TypeError):
            keys.decode_public(bad)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            keys.decode_signature(bad)  # type: ignore[arg-type]


def test_hostile_decode_unicode_lookalikes_and_controls():
    good = keys.encode_public(PUB_A)
    body = good[len("ed25519:"):]
    hostile = [
        "ed25519:" + body[:-1] + "\x00" + body[-1],   # embedded NUL
        "ed25519:" + body + "\u00a0",                 # NBSP (isspace)
        "ed25519:" + body + "\u200b",                 # zero-width space (not isspace)
        "ed25519:" + body[:-1] + "\ud800",            # lone surrogate
        "ed25519:" + body[:-1] + "\u0661",            # Arabic-Indic digit one
        "ed25519:" + body[:-1] + "\uff21",            # fullwidth A
        "ed25519:" + body + "A",                      # 44 chars -> 33 bytes
        "ed25519:" + body + "AA",                     # 45 chars: impossible length
        "ed25519:" + body[:-1],                       # 42 chars
        "ed25519:ed25519:" + body,                    # prefix twice
        "ed25519:" + "A" * (1 << 20),                 # huge
        "\ufeff" + good,                              # BOM in front
        good + "\x00",
        " " + good,
        good.upper() if good.upper() != good else good + "x",
    ]
    for s in hostile:
        with pytest.raises(ValueError):
            keys.decode_public(s)
    # A str subclass carrying the exact encoding still decodes.
    class S(str):
        pass
    assert keys.decode_public(S(good)) == PUB_A


def test_hostile_decode_public_last_char_canonical_set(monkeypatch):
    # 32 bytes = 42 full sextets + 4 bits; the 43rd char has two zero low bits.
    # Exactly the 16 alphabet chars with value % 4 == 0 are canonical there.
    # decode_public also validates the point since the 2026-09-08 repair, so the
    # base64url layer is measured with validation switched off, and the two
    # measurements are related: with validation on, exactly those of the 16 that are
    # valid keys survive ("A" is bytes(32), the order-4 point, and is refused).
    real_validate = keys.validate_public
    prefix = keys.encode_public(bytes(32))[:-1]
    with_validation = [c for c in B64URL_ALPHABET if _decodes(keys.decode_public, prefix + c)]
    monkeypatch.setattr(keys, "validate_public", lambda raw: None)
    accepted = [c for c in B64URL_ALPHABET if _decodes(keys.decode_public, prefix + c)]
    assert accepted == [c for i, c in enumerate(B64URL_ALPHABET) if i % 4 == 0]
    assert len(accepted) == 16
    assert "A" not in with_validation
    assert set(with_validation) < set(accepted)
    for c in accepted:
        raw = keys.decode_public(prefix + c)
        assert (c in with_validation) is _validates(raw, real_validate)


def test_hostile_decode_signature_last_char_canonical_set():
    # 64 bytes = 85 full sextets + 2 bits; the 86th char has four zero low bits.
    prefix = keys.encode_signature(bytes(64))[:-1]
    accepted = [c for c in B64URL_ALPHABET if _decodes(keys.decode_signature, prefix + c)]
    assert accepted == ["A", "Q", "g", "w"]


def _decodes(fn, s) -> bool:
    try:
        fn(s)
        return True
    except ValueError:
        return False


def _validates(raw, validate=None) -> bool:
    """True iff ``validate`` (default ``keys.validate_public``) accepts ``raw``."""
    try:
        (validate or keys.validate_public)(raw)
        return True
    except ValueError:
        return False


@given(st.text(min_size=0, max_size=60))
@settings(max_examples=300, deadline=None)
def test_hostile_decode_only_raises_valueerror_on_str(s):
    # Any str either decodes to exactly 32 bytes or raises ValueError; nothing else.
    for fn, n in ((keys.decode_public, 32), (keys.decode_signature, 64)):
        try:
            out = fn(s)
        except ValueError:
            continue
        assert isinstance(out, bytes) and len(out) == n
        assert (keys.encode_public if n == 32 else keys.encode_signature)(out) == s


def test_hostile_encode_bytes_like_variants_and_refusals():
    zero = "ed25519:" + "A" * 43
    assert keys.encode_public(memoryview(bytearray(64))[::2]) == zero
    assert keys.encode_public(memoryview(array.array("I", [0] * 8))) == zero
    assert keys.encode_public(_B(bytes(32))) == zero
    for bad in (True, 32, array.array("B", [0] * 32), list(bytes(32)), None):
        with pytest.raises(TypeError):
            keys.encode_public(bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        keys.encode_public(_LyingBytes(b"\x00" * 5))


# ------------------------------------------------------------------ tagged, hostile

def test_hostile_tagged_accepts_controls_and_whitespace_in_tag():
    # Contract forbids only empty and NUL.  Every other C0 control, DEL and
    # whitespace pass through unchanged.  Pinned so the gap is visible.
    d = bytes(32)
    for tag in ("a\nb", "\x7f", " ", "\t", "\x01", "a b\r\n"):
        assert keys.tagged(tag, d) == tag.encode("ascii") + b"\x00" + d
    assert len(keys.tagged("x" * 1_000_000, d)) == 1_000_000 + 1 + 32


def test_hostile_tagged_type_and_encoding_refusals():
    d = bytes(32)
    with pytest.raises(ValueError):
        keys.tagged("\ud800", d)          # lone surrogate
    with pytest.raises(ValueError):
        keys.tagged("\x00", d)
    with pytest.raises(ValueError):
        keys.tagged("\u00e9", d)          # Latin-1 range, still not ASCII
    with pytest.raises(TypeError):
        keys.tagged("t", True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        keys.tagged(None, d)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        keys.tagged("t", "0" * 32)  # type: ignore[arg-type]
    out = keys.tagged("t", memoryview(bytearray(32)))
    assert type(out) is bytes and out == b"t\x00" + d


# ------------------------------------------------------------------ node interop, reverse direction

NODE_SIGN = (
    'const c=require("crypto");'
    'const seed=Buffer.from(process.argv[1],"hex");'
    'const k=c.createPrivateKey({key:Buffer.concat([Buffer.from("302e020100300506032b657004220420","hex"),seed]),'
    'format:"der",type:"pkcs8"});'
    'process.stdout.write(c.sign(null,Buffer.from(process.argv[2],"hex"),k).toString("hex"));'
)

NODE_PEM_TO_SPKI = (
    'const c=require("crypto");const fs=require("fs");'
    'const k=c.createPrivateKey(fs.readFileSync(process.argv[1]));'
    'process.stdout.write(c.createPublicKey(k).export({format:"der",type:"spki"}).toString("hex"));'
)


def _node():
    node = shutil.which("node")
    if node is None:
        pytest.fail("node is not on PATH; the interop test cannot run (this is a failure, not a skip)")
    return node


def test_hostile_node_signs_python_verifies():
    node = _node()
    seed, pub = keys.generate()
    msg = b"node signs \x00\xff python verifies"
    proc = subprocess.run([node, "-e", NODE_SIGN, seed.hex(), msg.hex()],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    node_sig = bytes.fromhex(proc.stdout.strip())
    assert len(node_sig) == 64
    # Pure Ed25519 is deterministic: two implementations, one signature.
    assert node_sig == keys.sign(seed, msg)
    assert keys.verify(pub, msg, node_sig) is True
    assert keys.verify(pub, msg + b"\x00", node_sig) is False


def test_hostile_node_reads_python_pem(tmp_path):
    node = _node()
    path = tmp_path / "rfc.pem"
    keys.save_private_pem(SEED_A, path)
    proc = subprocess.run([node, "-e", NODE_PEM_TO_SPKI, str(path)],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "302a300506032b6570032100" + PUB_A.hex()


def test_hostile_signature_identical_across_two_processes():
    code = (
        "import sys; sys.path.insert(0, %r); from styxx.v8 import keys; "
        "print(keys.sign(bytes.fromhex(%r), b'x').hex())"
        % (os.getcwd(), SEED_A.hex())
    )
    outs = []
    for _ in range(2):
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr
        outs.append(proc.stdout.strip())
    assert outs[0] == outs[1]
    assert len(outs[0]) == 128
    assert bytes.fromhex(outs[0]) == keys.sign(SEED_A, b"x")


# ------------------------------------------------------------------ files, hostile

def test_hostile_private_pem_exact_bytes_for_rfc_seed(tmp_path):
    # PKCS#8 v1 for Ed25519 is fixed-shape: 16-byte prefix + seed, base64 in one line.
    path = tmp_path / "rfc.pem"
    keys.save_private_pem(SEED_A, path)
    raw = path.read_bytes()
    assert raw == (
        b"-----BEGIN PRIVATE KEY-----\n"
        b"MC4CAQAwBQYDK2VwBCIEIJ1hsZ3v/VpguoRK9JLsLMREScVpezJpGXA7rAMcrn9g\n"
        b"-----END PRIVATE KEY-----\n"
    )
    der = base64.b64decode(b"".join(raw.splitlines()[1:-1]))
    assert der == bytes.fromhex("302e020100300506032b657004220420") + SEED_A


def test_hostile_load_private_pem_refuses_other_key_kinds(tmp_path):
    def pem(key):
        return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                 serialization.NoEncryption())
    cases = {
        "rsa.pem": pem(rsa.generate_private_key(65537, 2048)),
        "ed448.pem": pem(ed448.Ed448PrivateKey.generate()),
        "x25519.pem": pem(x25519.X25519PrivateKey.generate()),
        "enc.pem": Ed25519PrivateKey.from_private_bytes(SEED_A).private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
            serialization.BestAvailableEncryption(b"pw")),
        "pub.pem": Ed25519PrivateKey.from_private_bytes(SEED_A).public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo),
        "der.pem": Ed25519PrivateKey.from_private_bytes(SEED_A).private_bytes(
            serialization.Encoding.DER, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()),
        "empty.pem": b"",
        "bogus_oid.pem": b"-----BEGIN PRIVATE KEY-----\n" + base64.encodebytes(
            bytes.fromhex("302e020100300506032a030404220420") + SEED_A) + b"-----END PRIVATE KEY-----\n",
    }
    for name, data in cases.items():
        p = tmp_path / name
        p.write_bytes(data)
        with pytest.raises(ValueError):
            keys.load_private_pem(p)
    # Lenient on framing the library tolerates: CRLF, a leading BOM, a second key after ours.
    keys.save_private_pem(SEED_A, tmp_path / "ok.pem")
    raw = (tmp_path / "ok.pem").read_bytes()
    (tmp_path / "crlf.pem").write_bytes(raw.replace(b"\n", b"\r\n"))
    assert keys.load_private_pem(tmp_path / "crlf.pem") == SEED_A
    (tmp_path / "bom.pem").write_bytes(b"\xef\xbb\xbf" + raw)
    assert keys.load_private_pem(tmp_path / "bom.pem") == SEED_A
    (tmp_path / "two.pem").write_bytes(raw + cases["rsa.pem"])
    assert keys.load_private_pem(tmp_path / "two.pem") == SEED_A


def test_hostile_load_public_file_variants(tmp_path):
    good = keys.encode_public(PUB_A).encode("ascii")
    refused = {
        "empty.pub": b"",
        "two_lf.pub": good + b"\n\n",
        "bare_cr.pub": good + b"\r",
        "lead_space.pub": b" " + good + b"\n",
        "utf16.pub": keys.encode_public(PUB_A).encode("utf-16"),
        "trailing_nul.pub": good + b"\x00\n",
        "lf_cr.pub": good + b"\n\r",
        "comment.pub": good + b" # signer\n",
        "two_keys.pub": good + b"\n" + good + b"\n",
    }
    for name, data in refused.items():
        p = tmp_path / name
        p.write_bytes(data)
        with pytest.raises(ValueError):
            keys.load_public(p)
    (tmp_path / "no_lf.pub").write_bytes(good)
    assert keys.load_public(tmp_path / "no_lf.pub") == PUB_A
    with pytest.raises(FileNotFoundError):
        keys.load_public(tmp_path / "missing.pub")
    with pytest.raises(TypeError):
        keys.load_public(3)  # type: ignore[arg-type]


def test_hostile_save_validates_before_touching_disk(tmp_path):
    with pytest.raises(ValueError):
        keys.save_private_pem(b"\x00" * 31, tmp_path / "nope.pem")
    with pytest.raises(ValueError):
        keys.save_public(b"\x00" * 33, tmp_path / "nope.pub")
    with pytest.raises(TypeError):
        keys.save_public("ed25519:" + "A" * 43, tmp_path / "nope2.pub")  # type: ignore[arg-type]
    assert not (tmp_path / "nope.pem").exists()
    assert not (tmp_path / "nope.pub").exists()
    assert not (tmp_path / "nope2.pub").exists()
    # A directory as the target is an OSError, not a silent no-op.
    with pytest.raises(OSError):
        keys.save_public(PUB_A, tmp_path)


def test_hostile_files_accept_bytes_paths(tmp_path):
    seed, pub = keys.generate()
    priv = os.fsencode(str(tmp_path / "b.pem"))
    pubf = os.fsencode(str(tmp_path / "b.pub"))
    keys.save_private_pem(seed, priv)
    keys.save_public(pub, pubf)
    assert keys.load_private_pem(priv) == seed
    assert keys.load_public(pubf) == pub


# ================================================================== hostile review, second pass
# Appended by the second hostile tester (2026-09-08).  Nothing above this line was edited.
#
# Independent oracle: a compact transcription of the RFC 8032 section 6 reference
# implementation (pure Python, no third-party code).  Everything below that says
# "reference" was computed by this code, not by ``cryptography`` and not by keys.py.

import hashlib

_P = 2**255 - 19
_Q = 2**252 + 27742317777372353535851937790883648493


def _modp_inv(x):
    return pow(x, _P - 2, _P)


_D = -121665 * _modp_inv(121666) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)


def _pt_add(P, Q):
    A = (P[1] - P[0]) * (Q[1] - Q[0]) % _P
    B = (P[1] + P[0]) * (Q[1] + Q[0]) % _P
    C = 2 * P[3] * Q[3] * _D % _P
    Dd = 2 * P[2] * Q[2] % _P
    E, F, G, H = B - A, Dd - C, Dd + C, B + A
    return (E * F % _P, G * H % _P, F * G % _P, E * H % _P)


def _pt_mul(s, P):
    Q = (0, 1, 1, 0)
    while s > 0:
        if s & 1:
            Q = _pt_add(Q, P)
        P = _pt_add(P, P)
        s >>= 1
    return Q


def _pt_equal(P, Q):
    return (P[0] * Q[2] - Q[0] * P[2]) % _P == 0 and (P[1] * Q[2] - Q[1] * P[2]) % _P == 0


def _recover_x(y, sign):
    if y >= _P:                       # RFC 8032 section 5.1.3 step 1: decoding fails
        return None
    x2 = (y * y - 1) * _modp_inv(_D * y * y + 1) % _P   # reduce: x2 == 0 is a mod-p question
    if x2 == 0:
        return None if sign else 0    # step 4: x = 0 with x_0 = 1 fails
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P != 0:
        x = x * _SQRT_M1 % _P
    if (x * x - x2) % _P != 0:
        return None
    if (x & 1) != sign:
        x = _P - x
    return x


_GY = 4 * _modp_inv(5) % _P
_GX = _recover_x(_GY, 0)
_G = (_GX, _GY, 1, _GX * _GY % _P)


def _pt_compress(P):
    zinv = _modp_inv(P[2])
    x, y = P[0] * zinv % _P, P[1] * zinv % _P
    return int.to_bytes(y | ((x & 1) << 255), 32, "little")


def _pt_decompress(s):
    y = int.from_bytes(s, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    return None if x is None else (x, y, 1, x * y % _P)


def _sha512_modq(s):
    return int.from_bytes(hashlib.sha512(s).digest(), "little") % _Q


def _secret_expand(secret):
    h = hashlib.sha512(secret).digest()
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, h[32:]


def ref_public(secret: bytes) -> bytes:
    a, _ = _secret_expand(secret)
    return _pt_compress(_pt_mul(a, _G))


def ref_sign(secret: bytes, msg: bytes) -> bytes:
    a, prefix = _secret_expand(secret)
    A = _pt_compress(_pt_mul(a, _G))
    r = _sha512_modq(prefix + msg)
    Rs = _pt_compress(_pt_mul(r, _G))
    h = _sha512_modq(Rs + A + msg)
    return Rs + int.to_bytes((r + h * a) % _Q, 32, "little")


def ref_verify(public: bytes, msg: bytes, signature: bytes) -> bool:
    if len(public) != 32 or len(signature) != 64:
        return False
    A = _pt_decompress(public)
    if A is None:
        return False
    Rs = signature[:32]
    R = _pt_decompress(Rs)
    if R is None:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= _Q:
        return False
    h = _sha512_modq(Rs + public + msg)
    return _pt_equal(_pt_mul(s, _G), _pt_add(R, _pt_mul(h, A)))


IDENTITY_SIGNBIT = bytes([1]) + bytes(30) + bytes([0x80])        # x = 0 with x_0 = 1
NONCANON_IDENTITY = (ED25519_P + 1).to_bytes(32, "little")        # y = p + 1
NONCANON_IDENTITY_SIGN = ((ED25519_P + 1) | (1 << 255)).to_bytes(32, "little")
Y_EQ_P = ED25519_P.to_bytes(32, "little")                         # y = p, i.e. 0 written non-canonically
Y_ZERO = bytes(32)                                                # y = 0: a canonical point of order 4
Y_MAX = (2**255 - 1).to_bytes(32, "little")                       # y = p + 18
UNIVERSAL_SIG = IDENTITY_POINT + bytes(32)                        # R = identity, S = 0


def test_hostile2_reference_reproduces_rfc8032_vectors():
    # The oracle proves itself on the RFC vectors before it is used against the module.
    for seed_hex, pub_hex, msg_hex, sig_hex in RFC8032_VECTORS + [RFC8032_TEST3]:
        seed, msg = bytes.fromhex(seed_hex), bytes.fromhex(msg_hex)
        assert ref_public(seed) == bytes.fromhex(pub_hex)
        assert ref_sign(seed, msg) == bytes.fromhex(sig_hex)
        assert ref_verify(bytes.fromhex(pub_hex), msg, bytes.fromhex(sig_hex)) is True
        assert ref_verify(bytes.fromhex(pub_hex), msg + b"x", bytes.fromhex(sig_hex)) is False


@given(st.binary(min_size=32, max_size=32), st.binary(max_size=200))
@settings(max_examples=40, deadline=None)
def test_hostile2_module_agrees_with_reference_on_public_and_sign(seed, msg):
    assert keys.public_from_private(seed) == ref_public(seed)
    sig = keys.sign(seed, msg)
    assert sig == ref_sign(seed, msg)
    assert ref_verify(keys.public_from_private(seed), msg, sig) is True
    assert keys.verify(keys.public_from_private(seed), msg, sig) is True


def test_hostile2_module_agrees_with_reference_on_a_large_message():
    msg = bytes(range(256)) * 4096          # 1 MiB
    sig = keys.sign(SEED_A, msg)
    assert sig == ref_sign(SEED_A, msg)
    assert keys.verify(PUB_A, msg, sig) is True
    assert ref_verify(PUB_A, msg, sig) is True


@given(st.binary(min_size=32, max_size=32), st.binary(min_size=64, max_size=64), st.binary(max_size=16))
@example(bytes(32), bytes(64), b"\x002\x01\x00")
@settings(max_examples=40, deadline=None)
def test_hostile2_random_well_formed_garbage_agrees_with_reference(pub, sig, msg):
    # Random 32/64-byte strings: no exception, and the verdict is the reference's
    # verdict AND the key passing validate_public.  The reference alone is NOT always
    # False: hypothesis found pub = 0^32, sig = 0^64, msg = b"\x002\x01\x00" verifying
    # (the pinned example; see the next test) — the module refuses that key.  y >= p
    # no longer needs excluding: the reference and the module both refuse it.
    got = keys.verify(pub, msg, sig)
    assert got is (ref_verify(pub, msg, sig) and _validates(pub))


def test_hostile2_all_zero_key_is_refused_where_the_reference_accepts_one_in_four():
    # FINDING (protocol-level, RFC 8032 permits it): y = 0 is a point of order 4, so with
    # A = R = that point and S = 0 the check [S]B == R + [k]A holds iff (k + 1) % 4 == 0.
    # One in four messages "verifies" under the all-zero key with the all-zero signature
    # for the reference and for the backend — that is what no key validation means.
    # Repaired 2026-09-08: the module refuses the key, so it accepts none of them.
    assert ref_verify(bytes(32), b"\x002\x01\x00", bytes(64)) is True
    assert _backend_verifies(bytes(32), b"\x002\x01\x00", bytes(64)) is True
    assert keys.verify(bytes(32), b"\x002\x01\x00", bytes(64)) is False
    hits = 0
    for i in range(64):
        msg = b"z%d" % i
        k = int.from_bytes(hashlib.sha512(bytes(64) + msg).digest(), "little") % ED25519_L
        expected = ((k + 1) % 4 == 0)
        assert ref_verify(bytes(32), msg, bytes(64)) is expected, (msg, k % 4)
        assert _backend_verifies(bytes(32), msg, bytes(64)) is expected, (msg, k % 4)
        assert keys.verify(bytes(32), msg, bytes(64)) is False, msg
        hits += expected
    assert 0 < hits < 64
    # Encoding frames the bytes without validating; decoding refuses them.
    assert keys.encode_public(bytes(32)) == "ed25519:" + "A" * 43
    assert keys.encode_signature(bytes(64)) == "ed25519:" + "A" * 86
    with pytest.raises(ValueError, match="small-order"):
        keys.decode_public("ed25519:" + "A" * 43)


def test_hostile2_signature_mutations_agree_with_reference():
    msg = b"mutate"
    sig = keys.sign(SEED_A, msg)
    r, s = sig[:32], int.from_bytes(sig[32:], "little")
    variants = {
        "s+q": r + (s + ED25519_L).to_bytes(32, "little"),
        "s=q": r + ED25519_L.to_bytes(32, "little"),
        "s=q-1": r + (ED25519_L - 1).to_bytes(32, "little"),
        "s=0": r + bytes(32),
        "s|2^255": r + (s | (1 << 255)).to_bytes(32, "little"),
        "r^1": bytes([sig[0] ^ 1]) + sig[1:],
        "r sign bit": sig[:31] + bytes([sig[31] ^ 0x80]) + sig[32:],
        "swap halves": sig[32:] + sig[:32],
    }
    for name, bad in variants.items():
        assert keys.verify(PUB_A, msg, bad) is False, name
        assert ref_verify(PUB_A, msg, bad) is False, name
    assert keys.verify(PUB_A, msg, sig) is True and ref_verify(PUB_A, msg, sig) is True


def test_hostile2_canonical_edge_points_are_refused_where_the_reference_is_message_dependent():
    # Canonically encoded small-order keys: the backend and the reference give the same
    # (message-dependent) verdict on every message -- the RFC itself permits decoding
    # these -- and the module, since the 2026-09-08 repair, gives False on every message.
    for A in (IDENTITY_POINT, ORDER2_POINT, Y_ZERO):
        for i in range(16):
            msg = b"c%d" % i
            assert _backend_verifies(A, msg, UNIVERSAL_SIG) is ref_verify(A, msg, UNIVERSAL_SIG), (A.hex(), msg)
            assert keys.verify(A, msg, UNIVERSAL_SIG) is False, (A.hex(), msg)
    assert all(ref_verify(IDENTITY_POINT, b"c%d" % i, UNIVERSAL_SIG) for i in range(16))
    assert not any(keys.verify(IDENTITY_POINT, b"c%d" % i, UNIVERSAL_SIG) for i in range(16))


def test_hostile2_backend_accepts_an_encoding_rfc8032_refuses_to_decode():
    # (p-1) | (1<<255) is the order-2 point with the sign bit set: x = 0 and x_0 = 1,
    # which RFC 8032 s5.1.3 step 4 refuses to decode. cryptography decodes and verifies
    # it on the messages where the cofactor works out. The divergence is the backend's,
    # and closing it is what validate_public is for. Both halves are asserted, so
    # neither this finding nor the repair can go vacuous.
    A = ((ED25519_P - 1) | (1 << 255)).to_bytes(32, "little")
    assert A.hex() == "ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    msgs = [b"c%d" % i for i in range(16)]
    assert any(_backend_verifies(A, m, UNIVERSAL_SIG) for m in msgs)      # the backend accepts it
    assert not any(ref_verify(A, m, UNIVERSAL_SIG) for m in msgs)         # RFC 8032 does not decode it
    assert not any(keys.verify(A, m, UNIVERSAL_SIG) for m in msgs)        # nor does this module
    with pytest.raises(ValueError):
        keys.validate_public(A)


def test_hostile2_non_canonical_public_key_encodings_are_refused():
    # Repaired 2026-09-08: validate_public applies RFC 8032 section 5.1.3 decoding
    # itself, so y >= p and x = 0 with the sign bit set never reach the backend.
    for A in (IDENTITY_SIGNBIT, NONCANON_IDENTITY, NONCANON_IDENTITY_SIGN, Y_EQ_P):
        with pytest.raises(ValueError):
            keys.validate_public(A)
        for i in range(16):
            assert keys.verify(A, b"c%d" % i, UNIVERSAL_SIG) is False, A.hex()


def test_hostile2_non_canonical_public_key_encodings_current_behaviour():
    # Receipt behind the repair.  Per RFC 8032 section 5.1.3 all four fail decoding
    # (reference False on every message); the backend reduces y mod p and accepts
    # (pinned so a stricter backend is noticed); the module refuses all of them.
    for A in (IDENTITY_SIGNBIT, NONCANON_IDENTITY, NONCANON_IDENTITY_SIGN):
        for i in range(16):
            msg = b"c%d" % i
            assert ref_verify(A, msg, UNIVERSAL_SIG) is False
            assert _backend_verifies(A, msg, UNIVERSAL_SIG) is True, (A.hex(), msg)
            assert keys.verify(A, msg, UNIVERSAL_SIG) is False, (A.hex(), msg)
    # y = p is 0 mod p, a point of order 4: the backend accepts iff k % 4 == 0, so on
    # some message; the reference and the module on none.
    hits = [i for i in range(16) if _backend_verifies(Y_EQ_P, b"c%d" % i, UNIVERSAL_SIG)]
    assert hits, "expected at least one message on which the backend accepts y=p"
    assert all(ref_verify(Y_EQ_P, b"c%d" % i, UNIVERSAL_SIG) is False for i in range(16))
    assert all(keys.verify(Y_EQ_P, b"c%d" % i, UNIVERSAL_SIG) is False for i in range(16))
    # y = p + 18 (all 0xff, 0x7f) is not on the curve after reduction: all three refuse.
    for i in range(4):
        assert keys.verify(Y_MAX, b"c%d" % i, UNIVERSAL_SIG) is False
        assert ref_verify(Y_MAX, b"c%d" % i, UNIVERSAL_SIG) is False
        assert _backend_verifies(Y_MAX, b"c%d" % i, UNIVERSAL_SIG) is False
    # Two byte strings, one key for the backend; the wire layer encodes both, and the
    # decoder now refuses the non-canonical one instead of returning it.
    assert keys.encode_public(NONCANON_IDENTITY) != keys.encode_public(IDENTITY_POINT)
    with pytest.raises(ValueError, match="y >= p"):
        keys.decode_public(keys.encode_public(NONCANON_IDENTITY))


def test_hostile2_non_canonical_r_in_signature_is_refused():
    # The same non-canonical encodings in the R half of a signature ARE refused by the
    # backend (it re-encodes R and compares bytes), matching the reference.  The
    # identity key is the probe (under it the check reduces to R == identity), so
    # the backend is asked directly; the module refuses the key itself, and refuses
    # the same R values under a real key too.
    for R in (IDENTITY_SIGNBIT, NONCANON_IDENTITY, NONCANON_IDENTITY_SIGN, Y_EQ_P, Y_MAX, ORDER2_POINT, Y_ZERO):
        bad = R + bytes(32)
        assert _backend_verifies(IDENTITY_POINT, b"m", bad) is False, R.hex()
        assert ref_verify(IDENTITY_POINT, b"m", bad) is False, R.hex()
        assert keys.verify(IDENTITY_POINT, b"m", bad) is False, R.hex()
        assert keys.verify(PUB_A, b"m", bad) is False, R.hex()
        assert ref_verify(PUB_A, b"m", bad) is False, R.hex()
    assert _backend_verifies(IDENTITY_POINT, b"m", UNIVERSAL_SIG) is True
    assert keys.verify(IDENTITY_POINT, b"m", UNIVERSAL_SIG) is False


def test_hostile2_node_shares_the_backend_verdict_on_non_canonical_keys():
    # A second OpenSSL consumer (node) agrees with the backend, not with the reference:
    # the deviation is a backend property, not a keys.py transcription error.  The
    # module deliberately diverges from both on these keys (it refuses them).
    for A in (IDENTITY_POINT, IDENTITY_SIGNBIT, NONCANON_IDENTITY):
        assert _node_verify(b"m", A, UNIVERSAL_SIG) == "true", A.hex()
        assert keys.verify(A, b"m", UNIVERSAL_SIG) is False, A.hex()
    assert _node_verify(b"m", Y_MAX, UNIVERSAL_SIG) == "false"


# ------------------------------------------------------------------ PKCS#8 framing, by hand

_ED25519_ALG = bytes.fromhex("300506032b6570")


def _pkcs8_v1(inner: bytes) -> bytes:
    body = bytes([0x02, 0x01, 0x00]) + _ED25519_ALG + bytes([0x04, len(inner)]) + inner
    assert len(body) < 128
    return bytes([0x30, len(body)]) + body


def _pem(der: bytes) -> bytes:
    return b"-----BEGIN PRIVATE KEY-----\n" + base64.encodebytes(der) + b"-----END PRIVATE KEY-----\n"


def test_hostile2_load_private_pem_accepts_hand_framed_pkcs8(tmp_path):
    # Framed by hand from the RFC seed, not by cryptography's serializer.
    p = tmp_path / "hand.pem"
    p.write_bytes(_pem(_pkcs8_v1(bytes([0x04, 32]) + SEED_A)))
    assert keys.load_private_pem(p) == SEED_A
    assert _pkcs8_v1(bytes([0x04, 32]) + SEED_A) == bytes.fromhex("302e020100300506032b657004220420") + SEED_A


def test_hostile2_load_private_pem_wrong_seed_length_is_valueerror(tmp_path):
    # Repaired 2026-09-08: every backend exception becomes a one-line ValueError.
    for n in (0, 1, 31, 33, 64):
        p = tmp_path / f"seed{n}.pem"
        p.write_bytes(_pem(_pkcs8_v1(bytes([0x04, n]) + bytes(n))))
        with pytest.raises(ValueError) as info:
            keys.load_private_pem(p)
        assert str(info.value) and "\n" not in str(info.value), n


def test_hostile2_load_private_pem_wrong_seed_length_current_behaviour(tmp_path):
    # Receipt behind the repair: the backend still raises InternalError on these
    # files (pinned so a backend change is noticed); the module never lets it out.
    from cryptography.exceptions import InternalError
    for n in (0, 1, 31, 33, 64):
        p = tmp_path / f"seed{n}.pem"
        data = _pem(_pkcs8_v1(bytes([0x04, n]) + bytes(n)))
        p.write_bytes(data)
        with pytest.raises(InternalError):
            serialization.load_pem_private_key(data, password=None)
        with pytest.raises(ValueError) as info:
            keys.load_private_pem(p)
        assert not isinstance(info.value, InternalError), n
        assert info.value.__cause__ is None and info.value.__suppress_context__
        assert str(info.value).startswith("not an unencrypted PEM private key: "), n
        assert "\n" not in str(info.value) and "\r" not in str(info.value), n
    # The X25519 OID with a 31-byte body takes the same path.
    x = bytes([0x02, 0x01, 0x00]) + bytes.fromhex("300506032b656e") + bytes([0x04, 33, 0x04, 31]) + bytes(31)
    p = tmp_path / "x25519_31.pem"
    data = _pem(bytes([0x30, len(x)]) + x)
    p.write_bytes(data)
    with pytest.raises(InternalError):
        serialization.load_pem_private_key(data, password=None)
    with pytest.raises(ValueError):
        keys.load_private_pem(p)


def test_hostile2_load_private_pem_other_malformed_pkcs8_is_valueerror(tmp_path):
    cases = {
        "raw32_no_inner_octet": _pkcs8_v1(bytes(32)),
        "inner_bitstring": _pkcs8_v1(bytes([0x03, 33, 0x00]) + bytes(32)),
        "inner_seq": _pkcs8_v1(bytes([0x30, 32]) + bytes(32)),
        "seed32_trailing_byte": _pkcs8_v1(bytes([0x04, 32]) + SEED_A + b"\x00"),
    }
    for name, der in cases.items():
        p = tmp_path / (name + ".pem")
        p.write_bytes(_pem(der))
        with pytest.raises(ValueError):
            keys.load_private_pem(p)
    p = tmp_path / "wrong_label.pem"
    p.write_bytes(b"-----BEGIN EC PRIVATE KEY-----\n" + base64.encodebytes(_pkcs8_v1(bytes([0x04, 32]) + SEED_A))
                  + b"-----END EC PRIVATE KEY-----\n")
    with pytest.raises(ValueError):
        keys.load_private_pem(p)
    p = tmp_path / "openssh.pem"
    p.write_bytes(Ed25519PrivateKey.from_private_bytes(SEED_A).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.OpenSSH, serialization.NoEncryption()))
    with pytest.raises(ValueError):
        keys.load_private_pem(p)


# ------------------------------------------------------------------ files, second pass

def test_hostile2_save_raises_when_the_write_does_not_land():
    # os.devnull accepts the write and reads back empty: the read-back check must fire
    # for the public file.  For the key file the no-clobber rule fires first, since
    # the device node exists: FileExistsError, an OSError either way.
    with pytest.raises(OSError):
        keys.save_public(PUB_A, os.devnull)
    with pytest.raises(FileExistsError):
        keys.save_private_pem(SEED_A, os.devnull)


def test_hostile2_save_private_pem_refuses_to_overwrite_unless_asked(tmp_path):
    # Repaired 2026-09-08: an existing key file is never clobbered by default.
    p = tmp_path / "k.pem"
    keys.save_private_pem(SEED_A, p)
    before = p.read_bytes()
    with pytest.raises(FileExistsError):
        keys.save_private_pem(SEED_B, p)
    assert p.read_bytes() == before
    assert keys.load_private_pem(p) == SEED_A
    keys.save_private_pem(SEED_B, p, overwrite=True)
    assert keys.load_private_pem(p) == SEED_B


def test_hostile2_load_rejects_nul_in_path_and_directory_targets(tmp_path):
    with pytest.raises(ValueError):
        keys.load_public("a\x00b")
    with pytest.raises(ValueError):
        keys.load_private_pem("a\x00b")
    with pytest.raises(OSError):
        keys.load_public(tmp_path)
    with pytest.raises(OSError):
        keys.load_private_pem(tmp_path)


def test_hostile2_public_file_round_trip_is_byte_stable_across_two_writes(tmp_path):
    a, b = tmp_path / "a.pub", tmp_path / "b.pub"
    keys.save_public(PUB_A, a)
    keys.save_public(PUB_A, b)
    assert a.read_bytes() == b.read_bytes() == b"ed25519:11qYAYKxCrfVS_7TyWQHOg7hcvPapiMlrwIaaPcHURo\n"


# ------------------------------------------------------------------ generate, second pass

def test_hostile2_generate_is_not_degenerate():
    pairs = [keys.generate() for _ in range(64)]
    seeds = {s for s, _ in pairs}
    pubs = {p for _, p in pairs}
    assert len(seeds) == 64 and len(pubs) == 64
    assert bytes(32) not in seeds and b"\xff" * 32 not in seeds
    for s, p in pairs[:3]:
        assert ref_public(s) == p
        assert IDENTITY_POINT != p and ORDER2_POINT != p


# ------------------------------------------------------------------ duck-typed liars (observations)

def test_hostile2_subclass_overrides_are_trusted():
    # Python semantics, not a module defect: a str subclass that lies in encode() and a
    # bytes subclass that lies in __bytes__() are believed.  Pinned so it is visible.
    class LyingStr(str):
        def encode(self, *a, **k):
            return b"other"

    class LyingBytes(bytes):
        def __bytes__(self):
            return bytes(32)

    assert keys.tagged(LyingStr("styxx.v8/cert/1"), bytes(32)) == b"other\x00" + bytes(32)
    assert keys.encode_public(LyingBytes(b"\x00" * 5)) == "ed25519:" + "A" * 43
    assert keys.verify(LyingBytes(b"\x00" * 5), b"m", b"\x00" * 64) is False


# ================================================================== repairs (2026-09-08)
# Tests for the four repaired defects: public-key validation, backend errors in
# load_private_pem, key-file creation, and the load_public framing rule.  Every
# curve fact below is checked against the reference above (independent of keys.py)
# and, where it applies, against ``cryptography`` directly.

# The eight canonical encodings of the cofactor subgroup.
SMALL_ORDER_HEX = [
    "01" + "00" * 31,                                                    # identity (0, 1)
    "ec" + "ff" * 30 + "7f",                                             # (0, -1), order 2
    "00" * 32,                                                           # y = 0, order 4
    "00" * 31 + "80",                                                    # y = 0, sign bit, order 4
    "c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac037a",  # order 8
    "c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac03fa",  # order 8, sign bit
    "26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc05",  # order 8
    "26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc85",  # order 8, sign bit
]
SMALL_ORDER = [bytes.fromhex(h) for h in SMALL_ORDER_HEX]

# Encodings RFC 8032 section 5.1.3 says fail to decode: y >= p, and x = 0 with x_0 = 1.
NON_CANONICAL = {
    "y=p": Y_EQ_P,
    "y=p+1": NONCANON_IDENTITY,
    "y=p+1,sign": NONCANON_IDENTITY_SIGN,
    "y=2^255-1": Y_MAX,
    "y=2^255-1,sign": b"\xff" * 32,
    "y=p+3,sign": ((ED25519_P + 3) | (1 << 255)).to_bytes(32, "little"),
    "x=0,sign(identity)": IDENTITY_SIGNBIT,
    "x=0,sign(order2)": ((ED25519_P - 1) | (1 << 255)).to_bytes(32, "little"),
}


@pytest.mark.parametrize("hex_key", SMALL_ORDER_HEX)
def test_repair_small_order_key_is_refused_everywhere(hex_key):
    A = bytes.fromhex(hex_key)
    # The reference: canonical, decodes, and [8]P is the identity.
    P = _pt_decompress(A)
    assert P is not None
    assert _pt_equal(_pt_mul(8, P), (0, 1, 1, 0))
    # The backend loads it without complaint: the module's check is the only gate.
    Ed25519PublicKey.from_public_bytes(A)
    with pytest.raises(ValueError, match="small-order"):
        keys.validate_public(A)
    with pytest.raises(ValueError, match="small-order"):
        keys.decode_public(keys.encode_public(A))
    for i in range(8):
        msg = b"cert %d" % i
        assert keys.verify(A, msg, UNIVERSAL_SIG) is False
        assert keys.verify(A, msg, A + bytes(32)) is False
        assert keys.verify(A, msg, bytes(64)) is False


@pytest.mark.parametrize("name", list(NON_CANONICAL))
def test_repair_non_canonical_key_encoding_is_refused(name):
    A = NON_CANONICAL[name]
    assert _pt_decompress(A) is None                     # RFC 8032 section 5.1.3 fails
    Ed25519PublicKey.from_public_bytes(A)                # the backend still loads it
    with pytest.raises(ValueError):
        keys.validate_public(A)
    with pytest.raises(ValueError):
        keys.decode_public(keys.encode_public(A))
    for i in range(4):
        assert keys.verify(A, b"c%d" % i, UNIVERSAL_SIG) is False


def test_repair_off_curve_encoding_is_refused():
    # y = 2 is the smallest canonical y with no x on the curve; the reference agrees.
    off = (2).to_bytes(32, "little")
    assert _recover_x(2, 0) is None
    with pytest.raises(ValueError, match="x recovery failed"):
        keys.validate_public(off)
    assert keys.verify(off, b"m", UNIVERSAL_SIG) is False
    with pytest.raises(ValueError, match="x recovery failed"):
        keys.validate_public(b"\xff" * 31 + b"\x7f"[:0] + bytes([0x7f]) if False else off)


def test_repair_rfc_and_generated_keys_validate():
    for _seed, pub_hex, _msg, _sig in RFC8032_VECTORS + [RFC8032_TEST3]:
        pub = bytes.fromhex(pub_hex)
        keys.validate_public(pub)
        assert keys._encode_point(keys._decode_point(pub)) == pub
    # Cross-check against cryptography on 50 fresh keys: every one validates, the
    # pure-Python decoder re-encodes to the same bytes, and the affine point agrees
    # with the independent reference.
    for _ in range(50):
        seed, pub = keys.generate()
        assert Ed25519PublicKey.from_public_bytes(pub).public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw) == pub
        keys.validate_public(pub)
        x, y = keys._decode_point(pub)
        assert keys._encode_point((x, y)) == pub
        ref = _pt_decompress(pub)
        assert (x, y) == (ref[0], ref[1])
        assert keys.decode_public(keys.encode_public(pub)) == pub
        assert keys.verify(pub, b"m", keys.sign(seed, b"m")) is True
    # Bytes-like variants are accepted.
    keys.validate_public(bytearray(PUB_A))
    keys.validate_public(memoryview(PUB_A))
    keys.validate_public(_B(PUB_A))


def test_repair_validate_public_type_and_length():
    with pytest.raises(TypeError):
        keys.validate_public(keys.encode_public(PUB_A))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        keys.validate_public(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        keys.validate_public(list(PUB_A))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        keys.validate_public(PUB_A[:31])
    with pytest.raises(ValueError):
        keys.validate_public(PUB_A + b"\x00")
    with pytest.raises(ValueError):
        keys.validate_public(b"")
    with pytest.raises(ValueError):
        keys.validate_public(_LyingBytes(b"\x00" * 5))
    assert keys.validate_public(PUB_A) is None


def test_repair_small_order_subgroup_is_exactly_the_eight():
    # Multiples 0..7 of an order-8 point under the module's own affine arithmetic
    # produce exactly the eight listed encodings and [8]P is the identity; the
    # reference (extended coordinates, independent code) agrees on every multiple.
    P = keys._decode_point(SMALL_ORDER[4])
    R = _pt_decompress(SMALL_ORDER[4])
    Q = (0, 1)
    seen = []
    for k in range(8):
        seen.append(keys._encode_point(Q))
        assert _pt_compress(_pt_mul(k, R)) == keys._encode_point(Q), k
        Q = keys._point_add(Q, P)
    assert Q == (0, 1)
    assert len(set(seen)) == 8
    assert sorted(seen) == sorted(SMALL_ORDER)
    for A in SMALL_ORDER:
        assert keys._is_small_order(keys._decode_point(A)) is True


def test_repair_generated_keys_have_order_l_under_the_module_arithmetic():
    # [L]A = identity and [8]A != identity for real keys: the affine addition law and
    # the decoder agree with the group structure, not only with the reference.
    def mul(n, P):
        R = None
        while n:
            if n & 1:
                R = P if R is None else keys._point_add(R, P)
            P = keys._point_add(P, P)
            n >>= 1
        return R
    for pub in (PUB_A, keys.generate()[1], keys.generate()[1]):
        A = keys._decode_point(pub)
        assert keys._is_small_order(A) is False
        assert mul(8, A) != (0, 1)
        assert mul(ED25519_L, A) == (0, 1)
    # And the base point itself.
    assert mul(ED25519_L, (_GX, _GY)) == (0, 1)


@given(st.binary(min_size=32, max_size=32))
@example(bytes(32))
@example(IDENTITY_POINT)
@example(ORDER2_POINT)
@example(SMALL_ORDER[4])
@example(SMALL_ORDER[7])
@example(NONCANON_IDENTITY)
@example(IDENTITY_SIGNBIT)
@example(PUB_A)
@settings(max_examples=300, deadline=None)
def test_repair_point_decoder_agrees_with_reference(raw):
    # On every 32-byte string the module's decoder and the reference decompressor
    # agree on decodability and on the affine point; validate_public then accepts
    # exactly the decodable points that are not small-order.
    ref = _pt_decompress(raw)
    try:
        x, y = keys._decode_point(raw)
    except ValueError:
        assert ref is None
        assert _validates(raw) is False
        return
    assert ref is not None
    assert (x, y) == (ref[0], ref[1])
    assert keys._encode_point((x, y)) == raw
    small = _pt_equal(_pt_mul(8, ref), (0, 1, 1, 0))
    assert keys._is_small_order((x, y)) is small
    assert _validates(raw) is (not small)


def test_repair_verify_refuses_small_order_key_before_the_backend(monkeypatch):
    # The key check runs before the backend is asked. cryptography's verify lives in
    # the Rust object, not on the Python class, so patching the class method records
    # nothing; the observable boundary is key CONSTRUCTION in the module's namespace.
    calls = []
    real_from_public_bytes = Ed25519PublicKey.from_public_bytes

    class _Recorder:
        @staticmethod
        def from_public_bytes(data):
            calls.append(bytes(data))
            return real_from_public_bytes(data)

    monkeypatch.setattr(keys, "Ed25519PublicKey", _Recorder)
    assert keys.verify(IDENTITY_POINT, b"m", UNIVERSAL_SIG) is False
    assert keys.verify(NONCANON_IDENTITY, b"m", UNIVERSAL_SIG) is False
    assert calls == []
    assert keys.verify(PUB_A, b"m", UNIVERSAL_SIG) is False   # reaches the backend, bad sig
    assert calls == [PUB_A]
    good_sig = bytes.fromhex(RFC8032_VECTORS[0][3])
    assert keys.verify(PUB_A, b"", good_sig) is True
    assert calls == [PUB_A, PUB_A]


# ------------------------------------------------------------------ 2. load_private_pem errors

def test_repair_load_private_pem_backend_errors_are_one_line_valueerrors(tmp_path):
    from cryptography.exceptions import InternalError, UnsupportedAlgorithm
    cases = {f"seed{n}.pem": _pem(_pkcs8_v1(bytes([0x04, n]) + bytes(n))) for n in (0, 1, 31, 33, 64)}
    x = bytes([0x02, 0x01, 0x00]) + bytes.fromhex("300506032b656e") + bytes([0x04, 33, 0x04, 31]) + bytes(31)
    cases["x25519_31.pem"] = _pem(bytes([0x30, len(x)]) + x)
    cases["garbage.pem"] = b"-----BEGIN PRIVATE KEY-----\nnope\n-----END PRIVATE KEY-----\n"
    cases["empty.pem"] = b""
    cases["bogus_oid.pem"] = _pem(bytes.fromhex("302e020100300506032a030404220420") + SEED_A)
    cases["encrypted.pem"] = Ed25519PrivateKey.from_private_bytes(SEED_A).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.BestAvailableEncryption(b"pw"))
    for name, data in cases.items():
        p = tmp_path / name
        p.write_bytes(data)
        with pytest.raises(ValueError) as info:
            keys.load_private_pem(p)
        assert not isinstance(info.value, (InternalError, UnsupportedAlgorithm, TypeError)), name
        text = str(info.value)
        assert text.startswith("not an unencrypted PEM private key: "), name
        assert len(text) > len("not an unencrypted PEM private key: "), name
        assert "\n" not in text and "\r" not in text, name
        assert info.value.__cause__ is None and info.value.__suppress_context__, name
    # Errors from opening the file are not rewrapped.
    with pytest.raises(FileNotFoundError):
        keys.load_private_pem(tmp_path / "missing.pem")
    with pytest.raises(ValueError, match="null"):
        keys.load_private_pem("a\x00b")
    with pytest.raises(OSError):
        keys.load_private_pem(tmp_path)
    # A good file still loads.
    keys.save_private_pem(SEED_A, tmp_path / "ok.pem")
    assert keys.load_private_pem(tmp_path / "ok.pem") == SEED_A


def test_repair_one_line_helper():
    assert keys._one_line(ValueError("a\n  b\r\n\tc")) == "a b c"
    assert keys._one_line(ValueError("")) == "ValueError"
    assert keys._one_line(ValueError("   ")) == "ValueError"
    assert keys._one_line(RuntimeError("x")) == "x"


# ------------------------------------------------------------------ 3. save_private_pem creation

def test_repair_save_private_pem_refuses_existing_paths(tmp_path):
    p = tmp_path / "k.pem"
    keys.save_private_pem(SEED_A, p)
    before = p.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite=True"):
        keys.save_private_pem(SEED_B, p)
    assert p.read_bytes() == before
    # The seed is validated before the path is looked at.
    with pytest.raises(ValueError):
        keys.save_private_pem(b"\x00" * 31, p)
    with pytest.raises(TypeError):
        keys.save_private_pem(SEED_A.hex(), p)  # type: ignore[arg-type]
    assert p.read_bytes() == before
    # A directory, an empty file, and a non-key file are all "existing".
    d = tmp_path / "dir"
    d.mkdir()
    with pytest.raises(FileExistsError):
        keys.save_private_pem(SEED_A, d)
    with pytest.raises(OSError):
        keys.save_private_pem(SEED_A, d, overwrite=True)
    e = tmp_path / "empty.pem"
    e.write_bytes(b"")
    with pytest.raises(FileExistsError):
        keys.save_private_pem(SEED_A, e)
    assert e.read_bytes() == b""
    n = tmp_path / "notes.pem"
    n.write_bytes(b"not a key")
    with pytest.raises(FileExistsError):
        keys.save_private_pem(SEED_A, n)
    assert n.read_bytes() == b"not a key"
    # overwrite=True replaces the contents and the result reads back.
    keys.save_private_pem(SEED_B, p, overwrite=True)
    assert keys.load_private_pem(p) == SEED_B
    keys.save_private_pem(SEED_A, n, overwrite=True)
    assert keys.load_private_pem(n) == SEED_A
    # overwrite=True on a fresh path creates it.
    q = tmp_path / "fresh.pem"
    keys.save_private_pem(SEED_A, q, overwrite=True)
    assert keys.load_private_pem(q) == SEED_A
    # overwrite is keyword-only.
    with pytest.raises(TypeError):
        keys.save_private_pem(SEED_A, tmp_path / "pos.pem", True)  # type: ignore[misc]
    assert not (tmp_path / "pos.pem").exists()
    # str and bytes paths both honour the rule.
    with pytest.raises(FileExistsError):
        keys.save_private_pem(SEED_A, str(p))
    with pytest.raises(FileExistsError):
        keys.save_private_pem(SEED_A, os.fsencode(str(p)))
    if os.name == "posix":
        # A dangling symlink counts as existing (lexists), and is not followed.
        link = tmp_path / "dangling.pem"
        os.symlink(tmp_path / "nowhere", link)
        with pytest.raises(FileExistsError):
            keys.save_private_pem(SEED_A, link)
        assert not (tmp_path / "nowhere").exists()


def test_repair_save_private_pem_file_is_exact_and_lf(tmp_path):
    # Creation through os.open must not translate newlines (O_BINARY on Windows).
    p = tmp_path / "rfc.pem"
    keys.save_private_pem(SEED_A, p)
    raw = p.read_bytes()
    assert raw == (
        b"-----BEGIN PRIVATE KEY-----\n"
        b"MC4CAQAwBQYDK2VwBCIEIJ1hsZ3v/VpguoRK9JLsLMREScVpezJpGXA7rAMcrn9g\n"
        b"-----END PRIVATE KEY-----\n"
    )
    assert b"\r" not in raw
    keys.save_private_pem(SEED_A, p, overwrite=True)
    assert p.read_bytes() == raw


def test_repair_save_private_pem_mode_on_overwrite(tmp_path):
    if os.name != "posix":
        # Windows: no ACL is set (documented).  Nothing to measure beyond read-back.
        p = tmp_path / "w.pem"
        p.write_bytes(b"old")
        keys.save_private_pem(SEED_B, p, overwrite=True)
        assert keys.load_private_pem(p) == SEED_B
        return
    # Overwriting a permissive file re-protects it (fchmod) before the write.
    q = tmp_path / "loose.pem"
    q.write_bytes(b"old")
    q.chmod(0o644)
    keys.save_private_pem(SEED_B, q, overwrite=True)
    assert (q.stat().st_mode & 0o777) == 0o600
    assert keys.load_private_pem(q) == SEED_B
    # A fresh file under a permissive umask is 0600 from creation.
    old = os.umask(0)
    try:
        r = tmp_path / "fresh.pem"
        keys.save_private_pem(SEED_A, r)
        assert (r.stat().st_mode & 0o777) == 0o600
    finally:
        os.umask(old)


def test_repair_save_private_pem_read_back_check_fires(tmp_path, monkeypatch):
    # If the bytes on disk differ from what was written, the save raises OSError.
    real_read_back = keys._read_back

    def read_back_sees_other_bytes(target, data):
        real_read_back(target, data)                     # the real check passes...
        raise OSError("read back differs")               # ...then simulate a mismatch

    monkeypatch.setattr(keys, "_read_back", read_back_sees_other_bytes)
    with pytest.raises(OSError, match="read back differs"):
        keys.save_private_pem(SEED_A, tmp_path / "rb.pem")
    monkeypatch.undo()
    # And the real check on a real mismatch: a file whose content is replaced by
    # a hook between write and read-back (simulated by patching open for reading).
    real_open = open
    state = {"armed": False}

    def open_hook(path, mode="r", *a, **k):
        fh = real_open(path, mode, *a, **k)
        if state["armed"] and "r" in mode and "b" in mode:
            import io
            return io.BytesIO(b"tampered")
        return fh

    monkeypatch.setattr(keys, "open", open_hook, raising=False)
    state["armed"] = True
    with pytest.raises(OSError, match="did not land"):
        keys.save_private_pem(SEED_A, tmp_path / "rb2.pem")


# ------------------------------------------------------------------ 4. load_public framing

def test_repair_load_public_framing_rule(tmp_path):
    s = keys.encode_public(PUB_A).encode("ascii")
    accepted = {"bare": s, "lf": s + b"\n", "crlf": s + b"\r\n"}
    for name, data in accepted.items():
        p = tmp_path / (name + ".pub")
        p.write_bytes(data)
        assert keys.load_public(p) == PUB_A, name
    refused = {
        "lf_lf": s + b"\n\n",
        "crlf_crlf": s + b"\r\n\r\n",
        "crlf_lf": s + b"\r\n\n",
        "lf_crlf": s + b"\n\r\n",
        "cr": s + b"\r",
        "cr_cr_lf": s + b"\r\r\n",
        "lf_cr": s + b"\n\r",
        "leading_lf": b"\n" + s + b"\n",
        "leading_crlf": b"\r\n" + s,
        "leading_space": b" " + s + b"\n",
        "leading_tab": b"\t" + s,
        "trailing_space_lf": s + b" \n",
        "lf_space": s + b"\n ",
        "trailing_tab": s + b"\t",
        "bom": b"\xef\xbb\xbf" + s + b"\n",
        "utf16": keys.encode_public(PUB_A).encode("utf-16"),
        "two_lines": s + b"\n" + s + b"\n",
        "comment": s + b" # signer\n",
        "empty": b"",
        "only_lf": b"\n",
        "only_crlf": b"\r\n",
        "nul_lf": s + b"\x00\n",
        "small_order_key": keys.encode_public(IDENTITY_POINT).encode("ascii") + b"\n",
        "non_canonical_key": keys.encode_public(NONCANON_IDENTITY).encode("ascii") + b"\r\n",
    }
    for name, data in refused.items():
        p = tmp_path / (name + ".pub")
        p.write_bytes(data)
        with pytest.raises(ValueError):
            keys.load_public(p)
    # The helper itself: exactly one newline, LF or CRLF, nothing more.
    assert keys._strip_one_newline(b"x\n") == b"x"
    assert keys._strip_one_newline(b"x\r\n") == b"x"
    assert keys._strip_one_newline(b"x") == b"x"
    assert keys._strip_one_newline(b"x\n\n") == b"x\n"
    assert keys._strip_one_newline(b"x\r\n\r\n") == b"x\r\n"
    assert keys._strip_one_newline(b"x\r") == b"x\r"
    assert keys._strip_one_newline(b"x\n\r") == b"x\n\r"   # a trailing bare CR is not a newline
    assert keys._strip_one_newline(b"\n") == b""
    assert keys._strip_one_newline(b"") == b""
