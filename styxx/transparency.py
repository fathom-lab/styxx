# -*- coding: utf-8 -*-
"""styxx.transparency — Cognometric Transparency Log (RFC 6962 for attestations).

A trust gap the attestation arc does NOT close: an agent can attest only its
*flattering* runs and silently drop the bad ones. A receipt proves what it
says; it cannot prove that **nothing was suppressed**. This module closes that
gap the way Certificate Transparency (RFC 6962) does for TLS certificates —
with an append-only Merkle log over attestation digests and two checkable
proofs:

  * **inclusion proof** — "entry X is recorded at index i in the log whose
    root is R" (audit path, O(log n)).
  * **consistency proof** — "the log with root R_n (size n) is an append-only
    extension of the earlier log with root R_m (size m < n)" — i.e. no past
    leaf was edited, deleted, or reordered (RFC 6962 §2.1.2).

Anyone who has *witnessed* an earlier tree head {size, root} can later detect
if the operator rewrote history. Both proofs verify with hex-only SHA-256, so
they reproduce byte-for-byte in any language (see ``web/styxx_verify.js``).

Deliberate, documented deviation from RFC 6962: domain separation uses ASCII
string tags (``styxx-tlog-leaf:`` / ``styxx-tlog-node:``) instead of the
0x00 / 0x01 byte tags, so the same pure-JS string SHA-256 used by the portable
verifier works unchanged across languages. Functionally equivalent domain
separation, not the literal CT tree (a styxx log is not submittable to a CT
log and vice versa). Leaves are attestation ``digest.portable.value`` hex
strings, so the log inherits the cross-language content address.

Honest boundary (pre-registered, P4): the log proves append-only-ness
*relative to a witnessed tree head*. It does NOT by itself stop an operator
who never publishes a tree head from equivocating (showing different logs to
different parties) — that needs tree-head gossip / witnessing, exactly as in
CT. This module produces and content-addresses tree heads; the witnessing /
signing layer is out of scope and stated as such.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "TransparencyLog",
    "leaf_hash",
    "node_hash",
    "merkle_tree_hash",
    "inclusion_proof",
    "consistency_proof",
    "verify_inclusion",
    "verify_consistency",
    "TLOG_VERSION",
    "LEAF_TAG",
    "NODE_TAG",
]

TLOG_VERSION = "1.0"
LEAF_TAG = "styxx-tlog-leaf:"
NODE_TAG = "styxx-tlog-node:"

_EMPTY_ROOT = hashlib.sha256(b"").hexdigest()


# ---------------------------------------------------------------------------
# primitive hashes (ASCII string domain separation; hex in, hex out)
# ---------------------------------------------------------------------------
def leaf_hash(entry: str) -> str:
    """RFC 6962 leaf hash with a string tag: SHA-256("styxx-tlog-leaf:" + entry)."""
    return hashlib.sha256((LEAF_TAG + entry).encode("utf-8")).hexdigest()


def node_hash(left: str, right: str) -> str:
    """RFC 6962 interior node hash: SHA-256("styxx-tlog-node:" + left + ":" + right).

    Both operands are lowercase hex, so the ``:`` separator is unambiguous.
    """
    return hashlib.sha256((NODE_TAG + left + ":" + right).encode("utf-8")).hexdigest()


def _k(n: int) -> int:
    """Largest power of two strictly less than n (n > 1)."""
    k = 1
    while (k << 1) < n:
        k <<= 1
    return k


def merkle_tree_hash(leaf_hashes: list[str]) -> str:
    """Merkle Tree Hash (RFC 6962 §2.1) over a list of leaf hashes (hex)."""
    n = len(leaf_hashes)
    if n == 0:
        return _EMPTY_ROOT
    if n == 1:
        return leaf_hashes[0]
    k = _k(n)
    return node_hash(merkle_tree_hash(leaf_hashes[:k]), merkle_tree_hash(leaf_hashes[k:]))


# ---------------------------------------------------------------------------
# proof generation (recursive, straight from RFC 6962 §2.1.1 / §2.1.2)
# ---------------------------------------------------------------------------
def _path(m: int, leaf_hashes: list[str]) -> list[str]:
    n = len(leaf_hashes)
    if n == 1:
        return []
    k = _k(n)
    if m < k:
        return _path(m, leaf_hashes[:k]) + [merkle_tree_hash(leaf_hashes[k:])]
    return _path(m - k, leaf_hashes[k:]) + [merkle_tree_hash(leaf_hashes[:k])]


def _subproof(m: int, leaf_hashes: list[str], b: bool) -> list[str]:
    n = len(leaf_hashes)
    if m == n:
        return [] if b else [merkle_tree_hash(leaf_hashes)]
    k = _k(n)
    if m <= k:
        return _subproof(m, leaf_hashes[:k], b) + [merkle_tree_hash(leaf_hashes[k:])]
    return _subproof(m - k, leaf_hashes[k:], False) + [merkle_tree_hash(leaf_hashes[:k])]


def inclusion_proof(leaf_index: int, leaf_hashes: list[str]) -> dict[str, Any]:
    """Audit path proving the leaf at ``leaf_index`` is in the log."""
    n = len(leaf_hashes)
    if not 0 <= leaf_index < n:
        raise IndexError(f"leaf_index {leaf_index} out of range for size {n}")
    return {
        "tlog_version": TLOG_VERSION,
        "kind": "inclusion",
        "leaf_index": leaf_index,
        "tree_size": n,
        "leaf_hash": leaf_hashes[leaf_index],
        "audit_path": _path(leaf_index, leaf_hashes),
        "root": merkle_tree_hash(leaf_hashes),
    }


def consistency_proof(first_size: int, leaf_hashes: list[str]) -> dict[str, Any]:
    """Proof that the log of size ``len(leaf_hashes)`` append-only-extends the
    earlier log of size ``first_size``."""
    second = len(leaf_hashes)
    if not 0 <= first_size <= second:
        raise ValueError(f"first_size {first_size} out of range for size {second}")
    if first_size == 0 or first_size == second:
        proof: list[str] = []
    else:
        proof = _subproof(first_size, leaf_hashes, True)
    return {
        "tlog_version": TLOG_VERSION,
        "kind": "consistency",
        "first_size": first_size,
        "second_size": second,
        "first_root": merkle_tree_hash(leaf_hashes[:first_size]),
        "second_root": merkle_tree_hash(leaf_hashes),
        "proof": proof,
    }


# ---------------------------------------------------------------------------
# proof verification (iterative, index-driven — independent of the prover)
# ---------------------------------------------------------------------------
def _bit_length(x: int) -> int:
    return x.bit_length()


def _ones_count(x: int) -> int:
    return bin(x).count("1")


def _trailing_zeros(x: int) -> int:
    return (x & -x).bit_length() - 1


def _decomp_incl(index: int, size: int) -> tuple[int, int]:
    inner = _bit_length(index ^ (size - 1))
    border = _ones_count(index >> inner)
    return inner, border


def _chain_inner(seed: str, proof: list[str], index: int) -> str:
    for i, h in enumerate(proof):
        if (index >> i) & 1 == 0:
            seed = node_hash(seed, h)
        else:
            seed = node_hash(h, seed)
    return seed


def _chain_inner_right(seed: str, proof: list[str], index: int) -> str:
    for i, h in enumerate(proof):
        if (index >> i) & 1 == 1:
            seed = node_hash(h, seed)
    return seed


def _chain_border_right(seed: str, proof: list[str]) -> str:
    for h in proof:
        seed = node_hash(h, seed)
    return seed


def verify_inclusion(proof: dict[str, Any], root: str | None = None) -> bool:
    """Verify an inclusion proof. If ``root`` is given it overrides the proof's
    embedded root (use it to check against an independently witnessed root)."""
    try:
        index = int(proof["leaf_index"])
        size = int(proof["tree_size"])
        leaf = str(proof["leaf_hash"])
        path = list(proof["audit_path"])
    except (KeyError, TypeError, ValueError):
        return False
    expected = root if root is not None else proof.get("root")
    if expected is None or not 0 <= index < size:
        return False
    inner, border = _decomp_incl(index, size)
    if len(path) != inner + border:
        return False
    res = _chain_inner(leaf, path[:inner], index)
    res = _chain_border_right(res, path[inner:])
    return res == expected


def verify_consistency(
    proof: dict[str, Any],
    first_root: str | None = None,
    second_root: str | None = None,
) -> bool:
    """Verify a consistency proof. Optional ``first_root`` / ``second_root``
    override the embedded roots — pass a witnessed earlier root here to detect
    a rewrite of the first ``first_size`` leaves."""
    try:
        size1 = int(proof["first_size"])
        size2 = int(proof["second_size"])
        path = list(proof["proof"])
    except (KeyError, TypeError, ValueError):
        return False
    r1 = first_root if first_root is not None else proof.get("first_root")
    r2 = second_root if second_root is not None else proof.get("second_root")
    if r1 is None or r2 is None:
        return False
    if size1 > size2:
        return False
    if size1 == size2:
        return r1 == r2 and len(path) == 0
    if size1 == 0:
        return len(path) == 0
    inner, border = _decomp_incl(size1 - 1, size2)
    shift = _trailing_zeros(size1)
    inner -= shift
    if size1 == (1 << shift):
        seed, start = r1, 0
    else:
        if not path:
            return False
        seed, start = path[0], 1
    if len(path) != start + inner + border:
        return False
    path = path[start:]
    mask = (size1 - 1) >> shift
    hash1 = _chain_inner_right(seed, path[:inner], mask)
    hash1 = _chain_border_right(hash1, path[inner:])
    hash2 = _chain_inner(seed, path[:inner], mask)
    hash2 = _chain_border_right(hash2, path[inner:])
    return hash1 == r1 and hash2 == r2


# ---------------------------------------------------------------------------
# log object + content-addressed tree head
# ---------------------------------------------------------------------------
@dataclass
class TransparencyLog:
    """An append-only Merkle log over attestation entries (portable digests).

    Entries are arbitrary strings; for styxx use each attestation's
    ``digest.portable.value``. The log itself stores no secrets and is fully
    reconstructible from the ordered entry list.
    """

    entries: list[str] = field(default_factory=list)
    log_id: str = "styxx-cognometric-tlog-v1"

    def append(self, entry: str) -> int:
        """Append an entry; return its leaf index."""
        self.entries.append(str(entry))
        return len(self.entries) - 1

    @property
    def size(self) -> int:
        return len(self.entries)

    def leaf_hashes(self) -> list[str]:
        return [leaf_hash(e) for e in self.entries]

    def root(self) -> str:
        return merkle_tree_hash(self.leaf_hashes())

    def tree_head(self, timestamp: str | None = None) -> dict[str, Any]:
        """A content-addressed tree head {size, root, timestamp}.

        NOT cryptographically signed — signing / witnessing / gossip is the
        equivocation-defeating layer (pre-registered boundary P4) and out of
        scope here. ``digest`` content-addresses the head so a witness can pin
        and later compare it.
        """
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        core = {
            "tlog_version": TLOG_VERSION,
            "log_id": self.log_id,
            "size": self.size,
            "root": self.root(),
            "timestamp": ts,
        }
        canonical = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return {**core, "digest": {"alg": "sha256", "value": digest}}

    def inclusion_proof(self, leaf_index: int) -> dict[str, Any]:
        return inclusion_proof(leaf_index, self.leaf_hashes())

    def consistency_proof(self, first_size: int) -> dict[str, Any]:
        return consistency_proof(first_size, self.leaf_hashes())

    def to_dict(self) -> dict[str, Any]:
        return {
            "tlog_version": TLOG_VERSION,
            "log_id": self.log_id,
            "entries": list(self.entries),
            "tree_head": self.tree_head(),
        }
