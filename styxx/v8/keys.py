"""styxx.v8.keys — ed25519 keys, signatures and the domain-separated preimage.

Backend: ``cryptography`` (pure Ed25519 per RFC 8032, no prehash).  Wire form for
public keys and signatures is ``"ed25519:" + base64url`` without padding.  Private keys
are stored as unencrypted PKCS#8 PEM; public keys as the wire string plus one LF.

Nothing here decides what is signed.  ``tagged`` only frames a 32-byte digest under a
NUL-terminated ASCII tag so that a cert id and a tree head never share a preimage; the
tags themselves are fixed by the v0.2 spec (``"styxx.v8/cert/1"``, ``"styxx.v8/sth/1"``).

Decisions
---------
- Public keys are validated: ``validate_public`` decodes the point in pure Python per
  RFC 8032 §5.1.3 (y >= p refused, failed x recovery refused, x = 0 with the sign bit
  refused) and refuses the eight small-order points ([8]P = identity), under which one
  64-byte string verifies over many or all messages; ``decode_public`` raises
  ValueError and ``verify`` returns False on such a key, ``encode_public`` only frames
  bytes and does not validate, and signing/verifying still go through ``cryptography``.
- ``load_private_pem`` turns every exception the backend raises (ValueError, TypeError,
  UnsupportedAlgorithm, and ``cryptography.exceptions.InternalError`` on a malformed
  seed OCTET STRING) into a ValueError with a one-line reason; errors from opening the
  file (OSError, or ValueError on a NUL in the path) pass through unchanged.
- ``save_private_pem`` creates the file with ``O_CREAT | O_EXCL`` and mode 0600 in one
  ``os.open`` call on POSIX, so no world-readable window exists (a following ``fchmod``
  is best-effort against a stricter umask); on Windows the same call runs but the mode
  is not an ACL and none is set, so the file inherits the directory's ACL; an existing
  path (including a dangling symlink) is refused with FileExistsError unless
  ``overwrite=True``, in which case the file is truncated and re-protected before any
  byte is written.
- ``load_public`` strips exactly one trailing LF or CRLF (a missing newline is also
  accepted); any other byte outside the wire string — leading whitespace, a bare CR, a
  second line, a BOM, a comment — is a ValueError.
- Wrong argument types raise TypeError everywhere except in ``verify``, which returns
  False.
"""
from __future__ import annotations

import base64
import binascii
import os
from typing import Tuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

__all__ = [
    "PREFIX",
    "generate",
    "public_from_private",
    "validate_public",
    "encode_public",
    "decode_public",
    "encode_signature",
    "decode_signature",
    "sign",
    "verify",
    "tagged",
    "save_private_pem",
    "load_private_pem",
    "save_public",
    "load_public",
]

PREFIX = "ed25519:"
SEED_LEN = 32
PUBLIC_LEN = 32
SIGNATURE_LEN = 64
DIGEST_LEN = 32


# ---------------------------------------------------------------- validation

def _check_bytes(value, length: int, what: str) -> bytes:
    """Return ``value`` as ``bytes`` if it is a bytes-like of exactly ``length`` bytes.

    TypeError for a non-bytes type; ValueError for a wrong length.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        value = bytes(value)
    else:
        raise TypeError(f"{what} must be bytes, got {type(value).__name__}")
    if len(value) != length:
        raise ValueError(f"{what} must be {length} bytes, got {len(value)}")
    return value


def _private_key(private_seed_32: bytes) -> Ed25519PrivateKey:
    seed = _check_bytes(private_seed_32, SEED_LEN, "private seed")
    return Ed25519PrivateKey.from_private_bytes(seed)


# ---------------------------------------------------------------- point validation
#
# Pure-Python RFC 8032 §5.1.3 decoding plus a small-order check, in affine coordinates
# on the twisted Edwards curve  -x^2 + y^2 = 1 + d x^2 y^2  over GF(p), p = 2^255 - 19,
# d = -121665/121666.  Validation only: signing and verifying go through ``cryptography``.
# Correctness over speed — a validation costs about ten modular inversions.

_P = 2**255 - 19
_D = (-121665 * pow(121666, _P - 2, _P)) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)          # a square root of -1 mod p
_IDENTITY = (0, 1)


def _inv(x: int) -> int:
    return pow(x, _P - 2, _P)


def _decode_point(public_32: bytes) -> Tuple[int, int]:
    """RFC 8032 §5.1.3: the affine ``(x, y)`` of a canonical encoding; ValueError otherwise.

    Refuses y >= p (non-canonical), an encoding whose x cannot be recovered (not on
    the curve), and x = 0 carrying the sign bit.
    """
    y = int.from_bytes(public_32, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    if y >= _P:
        raise ValueError("public key is not a canonical point encoding (y >= p)")
    u = (y * y - 1) % _P
    v = (_D * y * y + 1) % _P
    if v == 0:
        # Unreachable on this curve (d is a non-square, so -1/d is not a square);
        # kept so a zero denominator can never pass as x = 0.
        raise ValueError("public key is not a point on the curve")
    x2 = u * _inv(v) % _P
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P != 0:
        x = x * _SQRT_M1 % _P
    if (x * x - x2) % _P != 0:
        raise ValueError("public key is not a point on the curve (x recovery failed)")
    if x == 0:
        if sign:
            raise ValueError("public key encodes x = 0 with the sign bit set")
    elif (x & 1) != sign:
        x = _P - x
    if (-x * x + y * y - 1 - _D * x * x * y * y) % _P != 0:
        raise ValueError("public key is not a point on the curve")
    return x, y


def _encode_point(point: Tuple[int, int]) -> bytes:
    """RFC 8032 §5.1.2 encoding of an affine point."""
    x, y = point
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def _point_add(p: Tuple[int, int], q: Tuple[int, int]) -> Tuple[int, int]:
    """Affine twisted Edwards addition with a = -1; complete on this curve."""
    x1, y1 = p
    x2, y2 = q
    k = _D * x1 * x2 * y1 * y2 % _P
    x3 = (x1 * y2 + y1 * x2) * _inv((1 + k) % _P) % _P
    y3 = (y1 * y2 + x1 * x2) * _inv((1 - k) % _P) % _P
    return x3, y3


def _is_small_order(point: Tuple[int, int]) -> bool:
    """True iff [8]P is the identity — the cofactor subgroup has exactly eight points."""
    q = point
    for _ in range(3):
        q = _point_add(q, q)
    return q == _IDENTITY


def validate_public(public_32: bytes) -> None:
    """Raise ValueError unless ``public_32`` is a canonical encoding of a point of order L.

    Refuses non-canonical encodings (y >= p), encodings that are not on the curve,
    x = 0 with the sign bit set, and the eight small-order points (the identity and
    every P with [8]P = identity).  TypeError for a non-bytes argument.
    """
    public = _check_bytes(public_32, PUBLIC_LEN, "public key")
    point = _decode_point(public)
    if _is_small_order(point):
        raise ValueError("public key is a small-order point")


# ----------------------------------------------------------------- key material

def generate() -> Tuple[bytes, bytes]:
    """Return ``(private_seed_32, public_32)`` for a fresh Ed25519 key pair."""
    key = Ed25519PrivateKey.generate()
    seed = key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return seed, public


def public_from_private(private_seed_32: bytes) -> bytes:
    """Derive the 32-byte public key from a 32-byte private seed (RFC 8032 §5.1.5)."""
    return _private_key(private_seed_32).public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


# ----------------------------------------------------------------- wire encoding

def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(s: str, length: int, what: str) -> bytes:
    """Decode ``"ed25519:" + base64url`` (no padding) to exactly ``length`` bytes.

    ValueError on a bad prefix, on padding characters, on characters outside the
    base64url alphabet, on a non-canonical encoding (trailing bits set), and on a
    decoded length other than ``length``.
    """
    if not isinstance(s, str):
        raise TypeError(f"{what} must be str, got {type(s).__name__}")
    if not s.startswith(PREFIX):
        raise ValueError(f"{what} must start with {PREFIX!r}")
    body = s[len(PREFIX):]
    if not body:
        raise ValueError(f"{what} has an empty body")
    if "=" in body:
        raise ValueError(f"{what} must not carry base64 padding")
    if any(c.isspace() for c in body):
        raise ValueError(f"{what} must not contain whitespace")
    pad = (-len(body)) % 4
    if pad == 3:
        raise ValueError(f"{what} has an impossible base64url length")
    try:
        raw = base64.urlsafe_b64decode(body + "=" * pad)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{what} is not base64url: {exc}") from None
    if _b64url_encode(raw) != body:
        # Either a character outside the alphabet that the decoder tolerated,
        # or a non-canonical final sextet.
        raise ValueError(f"{what} is not canonical base64url")
    if len(raw) != length:
        raise ValueError(f"{what} must decode to {length} bytes, got {len(raw)}")
    return raw


def encode_public(public_32: bytes) -> str:
    """``"ed25519:" + base64url(public)`` without padding.  Frames bytes; does not validate."""
    return PREFIX + _b64url_encode(_check_bytes(public_32, PUBLIC_LEN, "public key"))


def decode_public(s: str) -> bytes:
    """Inverse of ``encode_public``; ValueError on bad prefix, length, padding present,
    or a key ``validate_public`` refuses."""
    raw = _b64url_decode(s, PUBLIC_LEN, "public key")
    validate_public(raw)
    return raw


def encode_signature(sig_64: bytes) -> str:
    """``"ed25519:" + base64url(signature)`` without padding."""
    return PREFIX + _b64url_encode(_check_bytes(sig_64, SIGNATURE_LEN, "signature"))


def decode_signature(s: str) -> bytes:
    """Inverse of ``encode_signature``; ValueError on bad prefix, length, or padding present."""
    return _b64url_decode(s, SIGNATURE_LEN, "signature")


# ----------------------------------------------------------------- sign / verify

def sign(private_seed_32: bytes, message: bytes) -> bytes:
    """Pure Ed25519 (RFC 8032 §5.1.6) signature over ``message``; 64 bytes, no prehash."""
    key = _private_key(private_seed_32)
    if isinstance(message, (bytearray, memoryview)):
        message = bytes(message)
    if not isinstance(message, bytes):
        raise TypeError(f"message must be bytes, got {type(message).__name__}")
    return key.sign(message)


def verify(public_32: bytes, message: bytes, sig_64: bytes) -> bool:
    """True iff ``sig_64`` is a valid Ed25519 signature by ``public_32`` over ``message``.

    Returns False on every failure — wrong key, wrong message, malformed key or
    signature, a key ``validate_public`` refuses, wrong types — and never raises.
    """
    try:
        public = _check_bytes(public_32, PUBLIC_LEN, "public key")
        sig = _check_bytes(sig_64, SIGNATURE_LEN, "signature")
        if isinstance(message, (bytearray, memoryview)):
            message = bytes(message)
        if not isinstance(message, bytes):
            return False
        validate_public(public)
        Ed25519PublicKey.from_public_bytes(public).verify(sig, message)
        return True
    except Exception:
        return False


# ----------------------------------------------------------------- domain separation

def tagged(tag: str, digest_32: bytes) -> bytes:
    """``tag.encode("ascii") + b"\\x00" + digest_32`` — the domain-separated preimage.

    ValueError for an empty tag, a tag containing NUL, a non-ASCII tag, or a digest
    that is not 32 bytes.
    """
    if not isinstance(tag, str):
        raise TypeError(f"tag must be str, got {type(tag).__name__}")
    if tag == "":
        raise ValueError("tag must not be empty")
    if "\x00" in tag:
        raise ValueError("tag must not contain NUL")
    try:
        tag_bytes = tag.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError("tag must be ASCII") from None
    digest = _check_bytes(digest_32, DIGEST_LEN, "digest")
    return tag_bytes + b"\x00" + digest


# ----------------------------------------------------------------- files

def _read_back(target, data: bytes) -> None:
    """Raise OSError unless ``target`` now holds exactly ``data``."""
    with open(target, "rb") as fh:
        written = fh.read()
    if written != data:
        raise OSError(f"write to {target!r} did not land: read back {len(written)} bytes")


def _write_bytes(path, data: bytes) -> None:
    """Write ``data`` in binary mode (no newline translation) and read it back."""
    target = os.fspath(path)
    with open(target, "wb") as fh:
        fh.write(data)
    _read_back(target, data)


def _one_line(exc: BaseException) -> str:
    """The exception's message collapsed onto one line; its type name if it has none."""
    return " ".join(str(exc).split()) or type(exc).__name__


def save_private_pem(private_seed_32: bytes, path, *, overwrite: bool = False) -> None:
    """Write the private key as unencrypted PKCS#8 PEM (LF line endings).

    The file is created by ``os.open(path, O_WRONLY | O_CREAT | O_EXCL, 0o600)``: on
    POSIX it is never readable by others, not even between creation and the write
    (``fchmod(0600)`` follows, best-effort, in case the umask took bits away).  On
    Windows the mode argument is not an ACL and no ACL is set — the file inherits the
    directory's ACL; protect the directory.

    An existing path (a file, a directory, a dangling symlink) is refused with
    FileExistsError unless ``overwrite=True``; then the file is opened with
    ``O_TRUNC``, re-protected, and only then written.
    """
    key = _private_key(private_seed_32)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).replace(b"\r\n", b"\n")
    target = os.fspath(path)
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_BINARY", 0)
    if overwrite:
        flags |= os.O_TRUNC
    else:
        if os.path.lexists(target):
            raise FileExistsError(
                f"refusing to overwrite existing key file {target!r}; pass overwrite=True"
            )
        flags |= os.O_EXCL
    fd = os.open(target, flags, 0o600)
    try:
        if hasattr(os, "fchmod"):
            try:
                os.fchmod(fd, 0o600)
            except OSError:
                pass
        fh = os.fdopen(fd, "wb")
    except BaseException:
        os.close(fd)
        raise
    with fh:
        fh.write(pem)
    _read_back(target, pem)


def load_private_pem(path) -> bytes:
    """Read a PKCS#8 PEM written by ``save_private_pem``; return the 32-byte seed.

    ValueError (one line, with the backend's reason) if the file is not an
    unencrypted Ed25519 private key, whatever exception the backend raised.
    """
    with open(os.fspath(path), "rb") as fh:
        data = fh.read()
    try:
        key = serialization.load_pem_private_key(data, password=None)
    except Exception as exc:
        raise ValueError(f"not an unencrypted PEM private key: {_one_line(exc)}") from None
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError(f"PEM holds a {type(key).__name__}, not an Ed25519 key")
    try:
        seed = key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
    except Exception as exc:
        raise ValueError(f"cannot extract the Ed25519 seed: {_one_line(exc)}") from None
    if len(seed) != SEED_LEN:
        raise ValueError(f"PEM yielded a {len(seed)}-byte seed, not {SEED_LEN}")
    return seed


def save_public(public_32: bytes, path) -> None:
    """Write ``encode_public(public_32) + "\\n"`` — ASCII, LF, no BOM."""
    _write_bytes(path, encode_public(public_32).encode("ascii") + b"\n")


def _strip_one_newline(data: bytes) -> bytes:
    """Remove exactly one trailing LF or CRLF; leave every other byte in place."""
    if data.endswith(b"\r\n"):
        return data[:-2]
    if data.endswith(b"\n"):
        return data[:-1]
    return data


def load_public(path) -> bytes:
    """Read a file written by ``save_public``; return the 32-byte public key.

    Framing rule: the file is the wire string followed by at most one newline, LF or
    CRLF, which is stripped.  Anything else around the string — leading whitespace, a
    bare CR, a second newline or line, a BOM, a comment — is a ValueError, as is a key
    ``validate_public`` refuses.
    """
    with open(os.fspath(path), "rb") as fh:
        data = fh.read()
    if data.startswith(b"\xef\xbb\xbf"):
        raise ValueError("public key file carries a BOM")
    try:
        text = _strip_one_newline(data).decode("ascii")
    except UnicodeDecodeError:
        raise ValueError("public key file is not ASCII") from None
    return decode_public(text)
