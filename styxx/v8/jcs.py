"""RFC 8785 (JCS) canonical bytes for styxx.v8.

Two backends produce the bytes: ``rfc8785.dumps`` when that package is importable, else
``styxx.attestation.jcs``.  The domain is validated here, before either backend runs, so the
exceptions a caller sees do not depend on which backend is loaded:

* ``TypeError``  -- a value that is not a JSON type (bytes, set, tuple, datetime, objects).
* ``ValueError`` -- NaN or an infinity, a non-``str`` object key, an ``int`` with ``|n| > 2**53``
  (not exactly representable as an IEEE-754 double, so the two backends could not agree on it),
  or text containing a lone surrogate (not encodable as UTF-8).

Known backend differences, handled here rather than hidden:

* ``rfc8785`` refuses ``|n| == 2**53`` although that value is exactly representable; the
  contract admits it, so such an ``int`` is coerced to the equal ``float`` before the call
  (``9007199254740992`` either way).
* ``styxx.attestation.jcs`` orders object keys by code point; RFC 8785 section 3.2.3 orders them
  by UTF-16 code unit.  The two orders differ only when sibling keys mix a non-BMP character
  with a BMP character at or above U+D800 in the same position.  The fallback backend refuses
  such an object with ``ValueError`` instead of emitting bytes that disagree with the RFC.
  ``tests/test_v8_jcs.py`` carries the receipt for this divergence.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

try:  # pragma: no cover - which branch runs depends on the environment
    import rfc8785 as _rfc8785
except ImportError:  # pragma: no cover
    _rfc8785 = None

from styxx.attestation import jcs as _styxx_jcs

__all__ = ["backend", "canonical_bytes", "digest", "sha256_hex", "MAX_SAFE_INT"]

MAX_SAFE_INT = 2**53


def backend() -> str:
    """Name of the serializer that ``canonical_bytes`` will call."""
    return "rfc8785" if _rfc8785 is not None else "styxx.attestation"


def _check_text(s: str, what: str) -> None:
    try:
        s.encode("utf-8")
    except UnicodeEncodeError as e:
        raise ValueError(f"{what} contains a lone surrogate and is not UTF-8 encodable") from e


def _normalize(obj: Any) -> Any:
    """Validate ``obj`` against the contract domain and return a plain-typed copy."""
    if obj is None or obj is True or obj is False:
        return obj
    if isinstance(obj, bool):  # pragma: no cover - bool cannot be subclassed
        return bool(obj)
    if isinstance(obj, str):
        _check_text(obj, "string")
        return str(obj)
    if isinstance(obj, int):
        n = int(obj)
        if n > MAX_SAFE_INT or n < -MAX_SAFE_INT:
            raise ValueError(
                f"int {n} is outside the exactly-representable double range (|n| <= 2**53)"
            )
        if n == MAX_SAFE_INT or n == -MAX_SAFE_INT:
            return float(n)
        return n
    if isinstance(obj, float):
        f = float(obj)
        if math.isnan(f) or math.isinf(f):
            raise ValueError(f"{f!r} has no JSON representation")
        return f
    if isinstance(obj, list):
        return [_normalize(x) for x in obj]
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValueError(f"object keys must be str, got {type(k).__name__}: {k!r}")
            _check_text(k, "object key")
            ks = str(k)
            if ks in out:
                raise ValueError(f"duplicate object key after normalization: {ks!r}")
            out[ks] = _normalize(v)
        return out
    raise TypeError(f"not a JSON type: {type(obj).__name__}")


def _utf16_units(k: str) -> bytes:
    return k.encode("utf-16-be")


def _refuse_utf16_divergent_keys(obj: Any) -> None:
    """Fallback guard: refuse objects whose key order differs between the two orderings."""
    if isinstance(obj, dict):
        keys = list(obj)
        if sorted(keys) != sorted(keys, key=_utf16_units):
            raise ValueError(
                "object keys whose code-point order differs from their UTF-16 order are "
                "refused by the styxx.attestation backend (RFC 8785 section 3.2.3)"
            )
        for v in obj.values():
            _refuse_utf16_divergent_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            _refuse_utf16_divergent_keys(v)


def canonical_bytes(obj: Any) -> bytes:
    """RFC 8785 canonical serialization of ``obj`` as UTF-8 bytes."""
    norm = _normalize(obj)
    if _rfc8785 is not None:
        return _rfc8785.dumps(norm)
    _refuse_utf16_divergent_keys(norm)
    return _styxx_jcs(norm).encode("utf-8")


def sha256_hex(b: bytes) -> str:
    """Lowercase hex SHA-256 of ``b`` (bytes, bytearray or memoryview; never text)."""
    if isinstance(b, str):
        raise TypeError("sha256_hex takes bytes, not str; encode the text before hashing")
    if not isinstance(b, (bytes, bytearray, memoryview)):
        raise TypeError(f"sha256_hex takes bytes, got {type(b).__name__}")
    return hashlib.sha256(bytes(b)).hexdigest()


def digest(obj: Any) -> str:
    """``sha256_hex(canonical_bytes(obj))`` -- 64 hex characters, no ``sha256:`` prefix."""
    return sha256_hex(canonical_bytes(obj))
