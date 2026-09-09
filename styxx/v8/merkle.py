"""RFC 6962 Merkle tree, as restated in RFC 9162 section 2.1.

Pure functions over leaf HASHES (32 bytes each). Hashing a raw entry into a leaf is the
caller's job via ``leaf_hash``. ``root`` is MTH, ``inclusion_proof`` is PATH,
``consistency_proof`` is PROOF/SUBPROOF, and the two ``verify_*`` functions follow the
RFC 9162 section 2.1.3.2 and 2.1.4.2 algorithms step for step.

Hashes are accepted as any bytes-like object (``bytes``, ``bytearray``, ``memoryview``, a
``bytes`` subclass) and canonicalised to exact ``bytes`` from the underlying buffer, so a
subclass whose ``__len__`` or ``__bytes__`` lies is measured by what it holds, not by what it
claims. Leaf and proof containers must be ordered: ``set``, ``frozenset``, ``dict`` and any
other ``collections.abc.Set`` / ``Mapping`` are refused, as are ``str`` and bytes-like objects
standing in for a sequence of hashes.

Generators validate their arguments and raise ``ValueError`` (a non-iterable ``leaf_hashes``
included). Verifiers never raise on a malformed proof; they return ``False`` -- a stranger's
verifier must not crash on hostile input, so the proof is coerced inside a guard that turns
any exception from the proof object into ``False``, and its iteration is bounded so an endless
iterator cannot hang the verifier.

Edge cases the RFC leaves to the reader (RFC 9162 defines PROOF only for 0 < first <= second):
``consistency_proof`` returns ``[]`` when ``first == second`` or ``first == 0``, and
``verify_consistency`` accepts exactly that: an empty proof with equal roots when
``first == second``, and an empty proof with ``first_root == EMPTY_ROOT`` when ``first == 0``
(the size-0 tree has exactly one root; every tree extends it). The ``first == 0`` root check
is stricter than the transparency-dev Go verifier, which accepts any roots there.
"""
from __future__ import annotations

import collections.abc
import hashlib
import itertools
from typing import Iterable, List, Sequence

__all__ = [
    "EMPTY_ROOT",
    "HASH_LEN",
    "MerkleTree",
    "consistency_proof",
    "inclusion_proof",
    "leaf_hash",
    "node_hash",
    "root",
    "verify_consistency",
    "verify_inclusion",
]

HASH_LEN = 32

# MTH({}) = SHA-256() — the hash of the empty string.
EMPTY_ROOT: bytes = hashlib.sha256(b"").digest()

_LEAF_PREFIX = b"\x00"
_NODE_PREFIX = b"\x01"

_BYTES_LIKE = (bytes, bytearray, memoryview)
_UNORDERED = (collections.abc.Set, collections.abc.Mapping)


# --------------------------------------------------------------------------- canonical bytes


def _as_bytes(obj: object) -> "bytes | None":
    """Exact ``bytes`` holding the buffer of a bytes-like object, or None for anything else.

    Goes through ``memoryview`` so a ``bytes`` subclass is read from its real storage: an
    overridden ``__len__`` or ``__bytes__`` cannot change what comes back. Never raises.
    """
    try:
        if type(obj) is bytes:
            return obj
        if not isinstance(obj, _BYTES_LIKE):
            return None
        return bytes(memoryview(obj))
    except Exception:
        return None


def _as_hash(obj: object) -> "bytes | None":
    """``_as_bytes`` restricted to exactly HASH_LEN bytes. Never raises."""
    b = _as_bytes(obj)
    if b is None or len(b) != HASH_LEN:
        return None
    return b


def _is_unordered(obj: object) -> bool:
    """set / frozenset / dict / any Set or Mapping: iteration order is not a property of the value."""
    return isinstance(obj, _UNORDERED)


# --------------------------------------------------------------------------- hashing


def leaf_hash(entry: bytes) -> bytes:
    """SHA-256(0x00 || entry). ``entry`` is any bytes-like object."""
    data = _as_bytes(entry)
    if data is None:
        raise ValueError(f"leaf_hash: entry must be bytes, got {type(entry).__name__}")
    return hashlib.sha256(_LEAF_PREFIX + data).digest()


def node_hash(left: bytes, right: bytes) -> bytes:
    """SHA-256(0x01 || left || right)."""
    left_b = _check_hash(left, "left")
    right_b = _check_hash(right, "right")
    return hashlib.sha256(_NODE_PREFIX + left_b + right_b).digest()


# --------------------------------------------------------------------------- validation


def _check_hash(h: object, what: str) -> bytes:
    """Canonical 32-byte ``bytes`` for a bytes-like hash; ValueError otherwise."""
    b = _as_bytes(h)
    if b is None:
        raise ValueError(f"{what}: expected bytes, got {type(h).__name__}")
    if len(b) != HASH_LEN:
        raise ValueError(f"{what}: expected {HASH_LEN} bytes, got {len(b)}")
    return b


def _check_leaves(leaf_hashes: Iterable[bytes]) -> List[bytes]:
    """An ordered iterable of 32-byte bytes-like hashes, as a list of canonical ``bytes``.

    Raises ValueError for a non-iterable, for an unordered container (set / frozenset /
    dict / any Set or Mapping), for ``str`` or a bytes-like object standing in for the
    sequence, and for any element that is not a 32-byte bytes-like. An exception raised by
    the caller's own iterator while it is being consumed is the caller's and propagates.
    """
    if _is_unordered(leaf_hashes):
        raise ValueError(
            f"leaf_hashes: {type(leaf_hashes).__name__} has no defined order; pass a sequence"
        )
    if isinstance(leaf_hashes, (str, *_BYTES_LIKE)):
        raise ValueError(
            f"leaf_hashes: expected a sequence of {HASH_LEN}-byte hashes, "
            f"got a single {type(leaf_hashes).__name__}"
        )
    try:
        items = list(leaf_hashes)
    except TypeError as exc:
        raise ValueError(
            f"leaf_hashes: expected an iterable of {HASH_LEN}-byte hashes, "
            f"got {type(leaf_hashes).__name__}"
        ) from exc
    return [_check_hash(h, f"leaf_hashes[{i}]") for i, h in enumerate(items)]


def _check_int(n: object, what: str) -> int:
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError(f"{what}: expected int, got {type(n).__name__}")
    return n


def _resolve_size(leaves: Sequence[bytes], tree_size: "int | None", what: str) -> int:
    if tree_size is None:
        return len(leaves)
    n = _check_int(tree_size, what)
    if n < 0 or n > len(leaves):
        raise ValueError(f"{what}: {n} outside 0..{len(leaves)}")
    return n


def _largest_power_of_two_below(n: int) -> int:
    """k: the largest power of two strictly smaller than n (n >= 2)."""
    return 1 << (n - 1).bit_length() - 1


# --------------------------------------------------------------------------- MTH / PATH / PROOF


def _mth(leaves: Sequence[bytes], lo: int, hi: int) -> bytes:
    """MTH(D[lo:hi]) over already-hashed leaves."""
    n = hi - lo
    if n == 0:
        return EMPTY_ROOT
    if n == 1:
        return leaves[lo]
    k = _largest_power_of_two_below(n)
    return node_hash(_mth(leaves, lo, lo + k), _mth(leaves, lo + k, hi))


def root(leaf_hashes: Sequence[bytes]) -> bytes:
    """MTH over already-hashed leaves; MTH([]) = EMPTY_ROOT."""
    leaves = _check_leaves(leaf_hashes)
    return _mth(leaves, 0, len(leaves))


def _path(leaves: Sequence[bytes], m: int, lo: int, hi: int) -> List[bytes]:
    """PATH(m, D[lo:hi]) with m relative to lo."""
    n = hi - lo
    if n == 1:
        return []
    k = _largest_power_of_two_below(n)
    if m < k:
        return _path(leaves, m, lo, lo + k) + [_mth(leaves, lo + k, hi)]
    return _path(leaves, m - k, lo + k, hi) + [_mth(leaves, lo, lo + k)]


def inclusion_proof(
    leaf_hashes: Sequence[bytes], index: int, tree_size: "int | None" = None
) -> List[bytes]:
    """PATH(index, D[0:tree_size])."""
    leaves = _check_leaves(leaf_hashes)
    n = _resolve_size(leaves, tree_size, "tree_size")
    i = _check_int(index, "index")
    if n == 0:
        raise ValueError("inclusion_proof: empty tree has no leaves")
    if i < 0 or i >= n:
        raise ValueError(f"index: {i} outside 0..{n - 1}")
    return _path(leaves, i, 0, n)


def _subproof(leaves: Sequence[bytes], m: int, lo: int, hi: int, b: bool) -> List[bytes]:
    """SUBPROOF(m, D[lo:hi], b) with m relative to lo; 0 < m <= hi - lo."""
    n = hi - lo
    if m == n:
        return [] if b else [_mth(leaves, lo, hi)]
    k = _largest_power_of_two_below(n)
    if m <= k:
        return _subproof(leaves, m, lo, lo + k, b) + [_mth(leaves, lo + k, hi)]
    return _subproof(leaves, m - k, lo + k, hi, False) + [_mth(leaves, lo, lo + k)]


def consistency_proof(
    leaf_hashes: Sequence[bytes], first: int, second: "int | None" = None
) -> List[bytes]:
    """PROOF(first, D[0:second]); [] when first == second or first == 0."""
    leaves = _check_leaves(leaf_hashes)
    n = _resolve_size(leaves, second, "second")
    m = _check_int(first, "first")
    if m < 0 or m > n:
        raise ValueError(f"first: {m} outside 0..{n}")
    if m == 0 or m == n:
        return []
    return _subproof(leaves, m, 0, n, True)


# --------------------------------------------------------------------------- verification


def _shift_until_lsb_set_or_zero(fn: int, sn: int) -> "tuple[int, int]":
    """Right-shift both equally until LSB(fn) is set or fn is 0."""
    while fn != 0 and (fn & 1) == 0:
        fn >>= 1
        sn >>= 1
    return fn, sn


def _proof_ok(proof: object, limit: int) -> "List[bytes] | None":
    """The proof as a list of canonical 32-byte hashes, or None when malformed. Never raises.

    Refuses ``str``, a bytes-like object and any unordered container as the proof itself;
    every element must be a 32-byte bytes-like (canonicalised with ``_as_hash``). Anything
    the proof object raises -- from ``isinstance``, ``__iter__``, ``__len__`` or ``__next__``
    -- is a malformed proof. At most ``limit`` elements are consumed: the caller passes a
    bound above the longest proof its fold can accept, so the verdict is unchanged and an
    endless iterator cannot hang the verifier.
    """
    try:
        if isinstance(proof, (str, *_BYTES_LIKE)) or _is_unordered(proof):
            return None
        items = list(itertools.islice(proof, limit))  # type: ignore[call-overload]
        out: List[bytes] = []
        for p in items:
            b = _as_hash(p)
            if b is None:
                return None
            out.append(b)
        return out
    except Exception:
        return None


def _size_ok(n: object) -> bool:
    return isinstance(n, int) and not isinstance(n, bool)


def verify_inclusion(
    leaf_hash: bytes,
    index: int,
    tree_size: int,
    proof: Sequence[bytes],
    expected_root: bytes,
) -> bool:
    """RFC 9162 section 2.1.3.2, literally. False on any malformed input; never raises.

    ``leaf_hash``, ``expected_root`` and every proof element may be any bytes-like object of
    exactly 32 bytes; ``proof`` may be any ordered iterable of them.
    """
    try:
        leaf = _as_hash(leaf_hash)
        want = _as_hash(expected_root)
        if leaf is None or want is None:
            return False
        if not _size_ok(index) or not _size_ok(tree_size):
            return False
        if tree_size <= 0 or index < 0:
            return False
        # 1. Compare leaf_index against tree_size.
        if index >= tree_size:
            return False
        # PATH(index, D[n]) has at most (n - 1).bit_length() elements and the fold below
        # rejects any longer proof (sn reaches 0), so consuming bit_length + 2 elements
        # decides every proof exactly as consuming all of it would.
        path = _proof_ok(proof, tree_size.bit_length() + 2)
    except Exception:
        return False
    if path is None:
        return False
    # 2. fn = leaf_index, sn = tree_size - 1.
    fn, sn = index, tree_size - 1
    # 3. r = hash.
    r = leaf
    # 4. For each value p in the inclusion path:
    for p in path:
        if sn == 0:
            return False
        if (fn & 1) == 1 or fn == sn:
            r = node_hash(p, r)
            if (fn & 1) == 0:
                fn, sn = _shift_until_lsb_set_or_zero(fn, sn)
        else:
            r = node_hash(r, p)
        fn >>= 1
        sn >>= 1
    # 5. sn must be 0 and r must equal the root.
    return sn == 0 and r == want


def verify_consistency(
    first: int,
    second: int,
    first_root: bytes,
    second_root: bytes,
    proof: Sequence[bytes],
) -> bool:
    """RFC 9162 section 2.1.4.2, literally. False on any malformed input; never raises.

    Both roots and every proof element may be any bytes-like object of exactly 32 bytes;
    ``proof`` may be any ordered iterable of them.

    Edge cases outside the RFC's 0 < first < second, where the only valid proof is empty:

    * ``first == 0``: True iff the proof is empty AND ``first_root == EMPTY_ROOT``. The
      size-0 tree has exactly one root, MTH({}) = SHA-256(""), and every tree extends it,
      so ``second_root`` is not constrained when ``second > 0``; for ``second == 0`` it must
      be ``EMPTY_ROOT`` as well. The transparency-dev Go verifier accepts ANY roots here;
      this verifier does not.
    * ``first == second > 0``: True iff the proof is empty and the two roots are equal. Any
      equal pair is accepted: the verifier cannot know the root of a tree it has not seen.
    """
    try:
        r1 = _as_hash(first_root)
        r2 = _as_hash(second_root)
        if r1 is None or r2 is None:
            return False
        if not _size_ok(first) or not _size_ok(second):
            return False
        if first < 0 or second < 0 or first > second:
            return False
        # PROOF(first, D[second]) has at most (second - 1).bit_length() + 1 elements and the
        # fold below rejects any longer proof (sn reaches 0), so consuming bit_length + 2
        # elements decides every proof exactly as consuming all of it would.
        path = _proof_ok(proof, second.bit_length() + 2)
    except Exception:
        return False
    if path is None:
        return False
    if first == 0:
        # The size-0 tree has exactly one root; (0, 0) requires it on both sides.
        return len(path) == 0 and r1 == EMPTY_ROOT and (second > 0 or r2 == EMPTY_ROOT)
    if first == second:
        return len(path) == 0 and r1 == r2
    # 1. Empty path fails.
    if len(path) == 0:
        return False
    # 2. If first is an exact power of 2, prepend first_hash.
    if first & (first - 1) == 0:
        path = [r1] + path
    # 3. fn = first - 1, sn = second - 1.
    fn, sn = first - 1, second - 1
    # 4. If LSB(fn) is set, right-shift both until LSB(fn) is not set.
    while (fn & 1) == 1:
        fn >>= 1
        sn >>= 1
    # 5. fr = sr = path[0].
    fr = sr = path[0]
    # 6. For each subsequent value c:
    for c in path[1:]:
        if sn == 0:
            return False
        if (fn & 1) == 1 or fn == sn:
            fr = node_hash(c, fr)
            sr = node_hash(c, sr)
            if (fn & 1) == 0:
                fn, sn = _shift_until_lsb_set_or_zero(fn, sn)
        else:
            sr = node_hash(sr, c)
        fn >>= 1
        sn >>= 1
    # 7. fr == first_hash, sr == second_hash, sn == 0.
    return fr == r1 and sr == r2 and sn == 0


# --------------------------------------------------------------------------- append-only tree


class MerkleTree:
    """An append-only list of leaf hashes with MTH/PATH/PROOF over any prefix."""

    def __init__(self, leaf_hashes: Iterable[bytes] = ()):
        self._leaves: List[bytes] = _check_leaves(leaf_hashes)

    def append(self, leaf_hash: bytes) -> int:
        """Append one leaf hash (any 32-byte bytes-like, stored as ``bytes``); return its index."""
        self._leaves.append(_check_hash(leaf_hash, "leaf_hash"))
        return len(self._leaves) - 1

    @property
    def size(self) -> int:
        return len(self._leaves)

    def leaves(self) -> List[bytes]:
        """A copy of the leaf hashes in order."""
        return list(self._leaves)

    def root(self, tree_size: "int | None" = None) -> bytes:
        n = _resolve_size(self._leaves, tree_size, "tree_size")
        return _mth(self._leaves, 0, n)

    def inclusion_proof(self, index: int, tree_size: "int | None" = None) -> List[bytes]:
        return inclusion_proof(self._leaves, index, tree_size)

    def consistency_proof(self, first: int, second: "int | None" = None) -> List[bytes]:
        return consistency_proof(self._leaves, first, second)
