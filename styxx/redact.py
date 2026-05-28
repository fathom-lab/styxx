# -*- coding: utf-8 -*-
"""styxx.redact — Redactable Cognometric Attestation (selective disclosure).

Every proof in the attestation arc so far forces you to publish the underlying
text: to re-derive a digest or a score, a verifier must see the whole (prompt,
response). This module lets an agent disclose a CHOSEN SUBSET of attested facts
(a single vitals score, one claim verdict) and prove each is exactly the value
committed into the receipt's public ``digest.redactable.root`` — while the rest
of the response stays private.

Construction (pure hashing, any-language, zero-dependency — same ethos as the
portable verifier):

  * flatten the attested object to a canonical, pointer-sorted list of
    ``(pointer, value)`` leaves (JSON-pointer paths; values serialized with the
    RFC 8785 / JCS rule shared with ``digest.portable``);
  * salt each leaf with 256 bits of fresh randomness:
        leaf = SHA-256("styxx-redact-leaf:" + salt + ":" + jcs(ptr) + ":" + jcs(val))
  * roll the salted leaves into an RFC 6962-style Merkle tree (string node tag
    ``styxx-redact-node:``); the root is the public commitment.

The salts are the agent's secret and are NOT public. A *disclosure* reveals, per
chosen field, ``{pointer, value, salt, leaf_index, audit_path}``; the verifier
recomputes the salted leaf and checks its inclusion against the root. Undisclosed
fields appear only as opaque sibling hashes.

HONEST SCOPE (pre-registered): this is selective DISCLOSURE, not zero-knowledge.
It does NOT prove a predicate (range / threshold) over a HIDDEN value, and it
does NOT re-derive scores — a disclosed value is trusted as the *committed* value
(it inherits the commit-time / re-seal boundary, caught only via the transparency
log + an external witness). A disclosure leaks the field COUNT and the disclosed
pointers + values; it hides every undisclosed pointer and value. The 256-bit
per-leaf salt is load-bearing: without it a low-entropy field (a verdict, a 0–1
score) is brute-forceable from its leaf hash.
"""
from __future__ import annotations

import hashlib
import secrets
from typing import Any

from .attestation import _jcs

__all__ = [
    "redactable_commit",
    "disclose",
    "verify_disclosure",
    "flatten",
    "REDACT_VERSION",
    "REDACT_ALG",
]

REDACT_VERSION = "1.0"
REDACT_ALG = "sha256-redact"
_LEAF_TAG = "styxx-redact-leaf:"
_NODE_TAG = "styxx-redact-node:"
_SALT_BITS = 256
_EMPTY_ROOT = hashlib.sha256(b"").hexdigest()


# ---------------------------------------------------------------------------
# flatten + hashing
# ---------------------------------------------------------------------------
def flatten(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten a JSON value to pointer-sorted (pointer, scalar) leaves."""
    out: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k in obj:
            out += flatten(obj[k], f"{prefix}/{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += flatten(v, f"{prefix}/{i}" if prefix else str(i))
    else:
        out.append((prefix, obj))
    return sorted(out, key=lambda pv: pv[0]) if prefix == "" else out


def _leaf_hash(salt: str, pointer: str, value: Any) -> str:
    return hashlib.sha256(
        (_LEAF_TAG + salt + ":" + _jcs(pointer) + ":" + _jcs(value)).encode("utf-8")
    ).hexdigest()


def _node_hash(left: str, right: str) -> str:
    return hashlib.sha256((_NODE_TAG + left + ":" + right).encode("utf-8")).hexdigest()


def _k(n: int) -> int:
    k = 1
    while (k << 1) < n:
        k <<= 1
    return k


def _mth(leaves: list[str]) -> str:
    n = len(leaves)
    if n == 0:
        return _EMPTY_ROOT
    if n == 1:
        return leaves[0]
    k = _k(n)
    return _node_hash(_mth(leaves[:k]), _mth(leaves[k:]))


def _path(m: int, leaves: list[str]) -> list[str]:
    n = len(leaves)
    if n == 1:
        return []
    k = _k(n)
    if m < k:
        return _path(m, leaves[:k]) + [_mth(leaves[k:])]
    return _path(m - k, leaves[k:]) + [_mth(leaves[:k])]


# ---------------------------------------------------------------------------
# inclusion verification (index-driven, independent of the prover)
# ---------------------------------------------------------------------------
def _decomp_incl(index: int, size: int) -> tuple[int, int]:
    inner = (index ^ (size - 1)).bit_length()
    border = bin(index >> inner).count("1")
    return inner, border


def _chain_inner(seed: str, proof: list[str], index: int) -> str:
    for i, h in enumerate(proof):
        seed = _node_hash(seed, h) if (index >> i) & 1 == 0 else _node_hash(h, seed)
    return seed


def _chain_border_right(seed: str, proof: list[str]) -> str:
    for h in proof:
        seed = _node_hash(h, seed)
    return seed


def _verify_inclusion(index: int, size: int, leaf: str, path: list[str], root: str) -> bool:
    if not 0 <= index < size:
        return False
    inner, border = _decomp_incl(index, size)
    if len(path) != inner + border:
        return False
    res = _chain_border_right(_chain_inner(leaf, path[:inner], index), path[inner:])
    return res == root


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def redactable_commit(obj: Any) -> dict[str, Any]:
    """Commit to ``obj``'s fields under a salted Merkle root.

    Returns ``{"alg", "version", "root", "tree_size", "salts": {pointer: salt}}``.
    The ``salts`` map is SECRET — keep it to disclose later; never publish it.
    The public commitment is ``{alg, version, root, tree_size}``.
    """
    leaves = flatten(obj)
    salts = {ptr: secrets.token_hex(_SALT_BITS // 8) for ptr, _ in leaves}
    leaf_hashes = [_leaf_hash(salts[ptr], ptr, val) for ptr, val in leaves]
    return {
        "alg": REDACT_ALG,
        "version": REDACT_VERSION,
        "root": _mth(leaf_hashes),
        "tree_size": len(leaves),
        "salts": salts,
    }


def disclose(obj: Any, commitment: dict[str, Any], pointers: list[str]) -> dict[str, Any]:
    """Produce a disclosure revealing every leaf whose pointer equals, or is a
    descendant of, any entry in ``pointers``.

    ``commitment`` is the dict returned by :func:`redactable_commit` (it carries
    the secret salts). The disclosure reveals ONLY the selected fields.
    """
    leaves = flatten(obj)
    salts = commitment["salts"]
    leaf_hashes = [_leaf_hash(salts[ptr], ptr, val) for ptr, val in leaves]

    def selected(ptr: str) -> bool:
        return any(ptr == p or ptr.startswith(p + "/") for p in pointers)

    fields = []
    for i, (ptr, val) in enumerate(leaves):
        if selected(ptr):
            fields.append({
                "pointer": ptr,
                "value": val,
                "salt": salts[ptr],
                "leaf_index": i,
                "audit_path": _path(i, leaf_hashes),
            })
    return {
        "kind": "disclosure",
        "alg": REDACT_ALG,
        "version": REDACT_VERSION,
        "tree_size": len(leaves),
        "root": _mth(leaf_hashes),
        "fields": fields,
    }


def verify_disclosure(disclosure: dict[str, Any], root: str | None = None) -> bool:
    """Verify every disclosed field is bound to the redactable root.

    Pass ``root`` (e.g. from the public ``digest.redactable.root`` or a
    transparency-log leaf) to check against an independently obtained commitment.
    """
    try:
        size = int(disclosure["tree_size"])
        fields = list(disclosure["fields"])
    except (KeyError, TypeError, ValueError):
        return False
    expected = root if root is not None else disclosure.get("root")
    if expected is None or size < 1:
        return False
    for f in fields:
        try:
            ptr = f["pointer"]
            val = f["value"]
            salt = f["salt"]
            index = int(f["leaf_index"])
            path = list(f["audit_path"])
        except (KeyError, TypeError, ValueError):
            return False
        leaf = _leaf_hash(salt, ptr, val)
        if not _verify_inclusion(index, size, leaf, path, expected):
            return False
    return True
