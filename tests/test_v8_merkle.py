"""tests for styxx.v8.merkle — RFC 6962 / RFC 9162 section 2.1.

Contract: styxx/v8/INTERFACES_foundations.md (frozen 2026-09-07), section merkle.

1. Published CT test vectors (roots 1..8, empty tree, two inclusion proofs, two consistency
   proofs). Every vector here was COMPUTED by the implementation and compared to the quoted
   value before being pinned; none was edited into agreement.
2. Properties over sizes 0..96 with random 32-byte leaves (hypothesis + one exhaustive pass).
3. Mutations: any single-byte flip, index change, element swap, truncation or extension makes
   verification return False; bad hash lengths return False from verify_* and raise ValueError
   from the generators. A size change is rejected against the honest root of the changed size
   (always) and against the ORIGINAL root only when the RFC fold's left/right shape changes --
   the exact property is stated on each size-change test, and the same-shape miss set is pinned.
4. Hostile pass (2026-09-08): bytes-like hashes are canonicalised everywhere, unordered
   containers are refused, a proof object that raises makes verify_* return False, generators
   raise ValueError on a non-iterable, and verify_consistency(0, n, ...) requires EMPTY_ROOT.
"""
from __future__ import annotations

import enum
import hashlib
import itertools
import os
import pathlib
import random
import subprocess
import sys

import pytest
from hypothesis import given, settings, strategies as st

from styxx.v8 import merkle as M
from styxx.v8.merkle import (
    EMPTY_ROOT,
    MerkleTree,
    consistency_proof,
    inclusion_proof,
    leaf_hash,
    node_hash,
    root,
    verify_consistency,
    verify_inclusion,
)

# --------------------------------------------------------------------------- vectors

# Raw entries from the certificate-transparency merkle test file; leaf-hashed before use.
CT_ENTRIES = [
    b"",
    b"\x00",
    b"\x10",
    b"\x20\x21",
    b"\x30\x31",
    b"\x40\x41\x42\x43",
    b"\x50\x51\x52\x53\x54\x55\x56\x57",
    b"\x60\x61\x62\x63\x64\x65\x66\x67\x68\x69\x6a\x6b\x6c\x6d\x6e\x6f",
]
CT_LEAVES = [leaf_hash(e) for e in CT_ENTRIES]

CT_ROOTS = {
    0: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    1: "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
    2: "fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125",
    3: "aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77",
    4: "d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7",
    5: "4e3bbb1f7b478dcfe71fb631631519a3bca12c9aefca1612bfce4c13a86264d4",
    6: "76e67dadbcdf1e10e1b74ddc608abd2f98dfb16fbce75277b5232a127f2087ef",
    7: "ddb89be403809e325750d3d263cd78929c2942b7942a34b77e122c9594a74c8c",
    8: "5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328",
}

# (index, tree_size) -> PATH
CT_INCLUSION = {
    (0, 8): [
        "96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7",
        "5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e",
        "6b47aaf29ee3c2af9af889bc1fb9254dabd31177f16232dd6aab035ca39bf6e4",
    ],
    (5, 8): [
        "bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b",
        "ca854ea128ed050b41b35ffc1b87b8eb2bde461e9e3b5596ece6b9d5975a0ae0",
        "d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7",
    ],
}

# (first, second) -> PROOF
CT_CONSISTENCY = {
    (2, 8): [
        "5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e",
        "6b47aaf29ee3c2af9af889bc1fb9254dabd31177f16232dd6aab035ca39bf6e4",
    ],
    (6, 8): [
        "0ebc5d3437fbe2db158b9f126a1d118e308181031d0a949f8dededebc558ef6a",
        "ca854ea128ed050b41b35ffc1b87b8eb2bde461e9e3b5596ece6b9d5975a0ae0",
        "d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7",
    ],
}


def _hx(items):
    return [b.hex() for b in items]


class TestHandDerivable:
    """Values derivable without any published file."""

    def test_empty_root_is_sha256_of_nothing(self):
        assert EMPTY_ROOT == hashlib.sha256(b"").digest()
        assert EMPTY_ROOT.hex() == CT_ROOTS[0]
        assert root([]).hex() == CT_ROOTS[0]

    def test_leaf_hash_is_prefixed_sha256(self):
        assert leaf_hash(b"") == hashlib.sha256(b"\x00").digest()
        assert leaf_hash(b"abc") == hashlib.sha256(b"\x00abc").digest()
        # size-1 root == the leaf hash of b"": hand-derivable, and the published size-1 vector.
        assert leaf_hash(b"").hex() == CT_ROOTS[1]
        assert root([leaf_hash(b"")]).hex() == CT_ROOTS[1]

    def test_node_hash_is_prefixed_sha256(self):
        a, b = leaf_hash(b"a"), leaf_hash(b"b")
        assert node_hash(a, b) == hashlib.sha256(b"\x01" + a + b).digest()
        assert node_hash(a, b) != node_hash(b, a)
        assert root([a, b]) == node_hash(a, b)

    def test_size_two_root_by_hand(self):
        # MTH(D[2]) = SHA-256(0x01 || d(0) || d(1)) with d = SHA-256(0x00 || entry).
        expect = hashlib.sha256(b"\x01" + CT_LEAVES[0] + CT_LEAVES[1]).digest()
        assert root(CT_LEAVES[:2]) == expect
        assert expect.hex() == CT_ROOTS[2]

    def test_size_three_root_by_hand(self):
        # k = 2 for n = 3: node(MTH(D[0:2]), MTH(D[2:3])) = node(node(d0, d1), d2).
        expect = node_hash(node_hash(CT_LEAVES[0], CT_LEAVES[1]), CT_LEAVES[2])
        assert root(CT_LEAVES[:3]) == expect
        assert expect.hex() == CT_ROOTS[3]

    def test_inclusion_proof_of_singleton_is_empty(self):
        assert inclusion_proof(CT_LEAVES[:1], 0) == []
        assert inclusion_proof(CT_LEAVES, 0, 1) == []
        assert verify_inclusion(CT_LEAVES[0], 0, 1, [], CT_LEAVES[0])

    def test_inclusion_proof_in_pair_is_the_sibling(self):
        assert inclusion_proof(CT_LEAVES, 0, 2) == [CT_LEAVES[1]]
        assert inclusion_proof(CT_LEAVES, 1, 2) == [CT_LEAVES[0]]

    def test_consistency_proof_from_one_equals_inclusion_path_of_leaf_zero(self):
        # PROOF(1, D[n]) = SUBPROOF(1, D[n], true); with m = 1 the recursion follows exactly
        # PATH(0, D[n]) until the singleton, where SUBPROOF(1, D[1], true) = {} = PATH(0, D[1]).
        for n in range(2, 9):
            assert consistency_proof(CT_LEAVES, 1, n) == inclusion_proof(CT_LEAVES, 0, n)


class TestPublishedVectors:
    @pytest.mark.parametrize("n", sorted(CT_ROOTS))
    def test_root(self, n):
        assert root(CT_LEAVES[:n]).hex() == CT_ROOTS[n]

    @pytest.mark.parametrize("n", sorted(CT_ROOTS))
    def test_root_via_tree(self, n):
        assert MerkleTree(CT_LEAVES).root(n).hex() == CT_ROOTS[n]
        assert MerkleTree(CT_LEAVES[:n]).root().hex() == CT_ROOTS[n]

    @pytest.mark.parametrize("key", sorted(CT_INCLUSION))
    def test_inclusion_proof(self, key):
        index, size = key
        proof = inclusion_proof(CT_LEAVES, index, size)
        assert _hx(proof) == CT_INCLUSION[key]
        assert MerkleTree(CT_LEAVES).inclusion_proof(index, size) == proof
        assert verify_inclusion(CT_LEAVES[index], index, size, proof, bytes.fromhex(CT_ROOTS[size]))

    @pytest.mark.parametrize("key", sorted(CT_CONSISTENCY))
    def test_consistency_proof(self, key):
        first, second = key
        proof = consistency_proof(CT_LEAVES, first, second)
        assert _hx(proof) == CT_CONSISTENCY[key]
        assert MerkleTree(CT_LEAVES).consistency_proof(first, second) == proof
        assert verify_consistency(
            first, second, bytes.fromhex(CT_ROOTS[first]), bytes.fromhex(CT_ROOTS[second]), proof
        )

    def test_every_ct_inclusion_verifies(self):
        for size in range(1, 9):
            r = bytes.fromhex(CT_ROOTS[size])
            for index in range(size):
                proof = inclusion_proof(CT_LEAVES, index, size)
                assert verify_inclusion(CT_LEAVES[index], index, size, proof, r), (index, size)

    def test_every_ct_consistency_verifies(self):
        for first in range(0, 9):
            for second in range(first, 9):
                proof = consistency_proof(CT_LEAVES, first, second)
                assert verify_consistency(
                    first,
                    second,
                    bytes.fromhex(CT_ROOTS[first]),
                    bytes.fromhex(CT_ROOTS[second]),
                    proof,
                ), (first, second)

    def test_consistency_proof_is_empty_when_first_is_zero_or_equals_second(self):
        for n in range(0, 9):
            assert consistency_proof(CT_LEAVES, 0, n) == []
            assert consistency_proof(CT_LEAVES, n, n) == []


# --------------------------------------------------------------------------- properties

hash32 = st.binary(min_size=32, max_size=32)
leaves_st = st.lists(hash32, min_size=0, max_size=96)


def _rand_leaves(n: int, seed: int = 20260907) -> list:
    rng = random.Random(seed)
    return [bytes(rng.getrandbits(8) for _ in range(32)) for _ in range(n)]


class TestProperties:
    def test_exhaustive_sizes_0_to_96(self):
        """Every inclusion proof and every consistency pair m <= n, sizes 0..96, seeded leaves."""
        leaves = _rand_leaves(96)
        roots = [root(leaves[:n]) for n in range(97)]
        tree = MerkleTree()
        assert tree.size == 0
        assert tree.root() == roots[0] == EMPTY_ROOT
        for n in range(1, 97):
            assert tree.append(leaves[n - 1]) == n - 1
            assert tree.size == n
            assert tree.root() == roots[n]
            for index in range(n):
                proof = inclusion_proof(leaves, index, n)
                assert tree.inclusion_proof(index) == proof
                assert verify_inclusion(leaves[index], index, n, proof, roots[n]), (index, n)
                assert not verify_inclusion(leaves[index], index, n, proof, roots[n - 1])
        for first in range(0, 97):
            for second in range(first, 97):
                proof = consistency_proof(leaves, first, second)
                assert tree.consistency_proof(first, second) == proof
                assert verify_consistency(first, second, roots[first], roots[second], proof), (
                    first,
                    second,
                )
                if first == 0 or first == second:
                    assert proof == []
                else:
                    assert proof

    @settings(deadline=None, max_examples=200)
    @given(leaves=leaves_st, data=st.data())
    def test_inclusion_verifies_at_its_size_and_not_the_next(self, leaves, data):
        n = len(leaves)
        if n == 0:
            with pytest.raises(ValueError):
                inclusion_proof(leaves, 0)
            return
        index = data.draw(st.integers(min_value=0, max_value=n - 1))
        proof = inclusion_proof(leaves, index, n)
        r = root(leaves)
        assert verify_inclusion(leaves[index], index, n, proof, r)
        extra = data.draw(hash32)
        r_next = root(leaves + [extra])
        assert r_next != r
        assert not verify_inclusion(leaves[index], index, n, proof, r_next)
        assert not verify_inclusion(leaves[index], index, n + 1, proof, r_next)

    @settings(deadline=None, max_examples=200)
    @given(leaves=leaves_st, data=st.data())
    def test_consistency_verifies_for_any_pair(self, leaves, data):
        n = len(leaves)
        first = data.draw(st.integers(min_value=0, max_value=n))
        second = data.draw(st.integers(min_value=first, max_value=n))
        proof = consistency_proof(leaves, first, second)
        r1, r2 = root(leaves[:first]), root(leaves[:second])
        assert verify_consistency(first, second, r1, r2, proof)
        if first == second:
            assert proof == []
            assert r1 == r2
        # Against the root of a longer tree the same proof must not verify -- except for
        # first == 0, where the contract fixes the proof at [] and the empty tree is a prefix
        # of every tree, so the empty proof verifies against any second root by definition.
        extra = data.draw(hash32)
        r2_next = root(leaves[:second] + [extra])
        if first == 0:
            assert proof == []
            assert verify_consistency(0, second + 1, EMPTY_ROOT, r2_next, [])
            # only the EMPTY proof is accepted for first == 0
            assert not verify_consistency(0, second + 1, EMPTY_ROOT, r2_next, [extra])
        else:
            assert not verify_consistency(first, second, r1, r2_next, proof)
            assert not verify_consistency(first, second + 1, r1, r2_next, proof)

    @settings(deadline=None, max_examples=100)
    @given(leaves=leaves_st)
    def test_incremental_tree_matches_functional_root_at_every_size(self, leaves):
        tree = MerkleTree()
        for i, h in enumerate(leaves):
            assert tree.append(h) == i
            assert tree.root() == root(leaves[: i + 1])
        assert tree.size == len(leaves)
        for n in range(len(leaves) + 1):
            assert tree.root(n) == root(leaves[:n])
        assert MerkleTree(leaves).root() == root(leaves)
        assert MerkleTree(leaves).leaves() == list(leaves)

    @settings(deadline=None, max_examples=100)
    @given(leaves=st.lists(hash32, min_size=1, max_size=96))
    def test_root_is_order_sensitive(self, leaves):
        if leaves[0] == leaves[-1]:
            return  # swapping equal ends is the identity permutation
        swapped = list(leaves)
        swapped[0], swapped[-1] = swapped[-1], swapped[0]
        assert root(swapped) != root(leaves)


# --------------------------------------------------------------------------- mutations


def _flip(b: bytes, i: int) -> bytes:
    return b[:i] + bytes([b[i] ^ 0x01]) + b[i + 1 :]


# Generator-side shapes. The RFC 9162 verifiers bind tree_size / second only through the
# left/right decision sequence of the fold; two sizes that give the same sequence for the same
# index are indistinguishable to the verifier when the caller hands it a STALE root. Against the
# honest root at the other size they always fail (asserted below). These helpers derive the
# decision sequence from the PATH / SUBPROOF recursion (RFC 9162 section 2.1.3.1 / 2.1.4.1),
# independently of the verifier code, so the pinned miss set is not the verifier grading itself.


def _k_of(n: int) -> int:
    return 1 << ((n - 1).bit_length() - 1)


def _path_shape(m: int, n: int) -> tuple:
    shape = []
    while n > 1:
        k = _k_of(n)
        if m < k:
            shape.append("L")
            n = k
        else:
            shape.append("R")
            m -= k
            n -= k
    return tuple(shape)


def _subproof_shape(m: int, n: int) -> tuple:
    shape = []
    b = True
    while m != n:
        k = _k_of(n)
        if m <= k:
            shape.append("L")
            n = k
        else:
            shape.append("R")
            m -= k
            n -= k
            b = False
    shape.append("T" if b else "F")
    return tuple(shape)


class TestInclusionMutations:
    LEAVES = _rand_leaves(20, seed=7)  # pool covers every `other` size probed below
    SIZES = (1, 2, 3, 5, 8, 13)

    def _cases(self):
        for n in self.SIZES:
            r = root(self.LEAVES[:n])
            for index in range(n):
                yield index, n, inclusion_proof(self.LEAVES, index, n), r

    def test_baseline_verifies(self):
        for index, n, proof, r in self._cases():
            assert verify_inclusion(self.LEAVES[index], index, n, proof, r)

    def test_flip_any_byte_of_any_proof_element(self):
        checked = 0
        for index, n, proof, r in self._cases():
            for k in range(len(proof)):
                for byte in range(32):
                    bad = list(proof)
                    bad[k] = _flip(proof[k], byte)
                    assert not verify_inclusion(self.LEAVES[index], index, n, bad, r), (index, n, k, byte)
                    checked += 1
        assert checked > 0

    def test_flip_any_byte_of_leaf_or_root(self):
        for index, n, proof, r in self._cases():
            for byte in range(32):
                assert not verify_inclusion(_flip(self.LEAVES[index], byte), index, n, proof, r)
                assert not verify_inclusion(self.LEAVES[index], index, n, proof, _flip(r, byte))

    def test_change_index(self):
        for index, n, proof, r in self._cases():
            for other in range(-2, n + 3):
                if other == index:
                    continue
                assert not verify_inclusion(self.LEAVES[index], other, n, proof, r), (index, other, n)

    # Stale-root size ambiguities for SIZES x other in -1..19, counted 2026-09-08. The number is
    # the finding: the RFC verifier reads tree_size only through the fold's shape.
    KNOWN_SIZE_MISSES = 96

    def test_change_tree_size_against_honest_root(self):
        # Property asserted (contract item 3, as it is actually true): a proof generated for
        # (index, n) never verifies as (index, other) against root(other), the honest root of
        # the changed size, for any other != n. Either the fold's shape differs (different
        # length or a different left/right sequence, so a different hash) or it is the same
        # and the fold reproduces root(n), which is not root(other).
        checked = 0
        for index, n, proof, r in self._cases():
            for other in range(-1, 20):
                if other == n:
                    continue
                # negative sizes have no tree; the verifier must reject them with any root
                r_other = root(self.LEAVES[:other]) if other >= 0 else r
                assert not verify_inclusion(self.LEAVES[index], index, other, proof, r_other), (
                    index,
                    n,
                    other,
                )
                checked += 1
        assert checked > 0

    def test_change_tree_size_against_stale_root_pins_the_miss_set(self):
        # Property asserted: against the ORIGINAL root root(n), a changed size `other` verifies
        # iff the RFC fold's left/right decision sequence for (index, other) equals the one for
        # (index, n) -- `_path_shape`, derived from the PATH recursion, not from the verifier.
        # The contract's unqualified "changing the tree size makes verification return False"
        # is not true here: the verifier binds tree_size only through that shape, so the miss
        # set is pinned by count and by predicate rather than asserted away.
        accepted = set()
        predicted = set()
        for index, n, proof, r in self._cases():
            for other in range(-1, 20):
                if other == n:
                    continue
                if verify_inclusion(self.LEAVES[index], index, other, proof, r):
                    accepted.add((index, n, other))
                if 0 <= index < other and _path_shape(index, n) == _path_shape(index, other):
                    predicted.add((index, n, other))
        assert accepted == predicted
        assert len(accepted) == self.KNOWN_SIZE_MISSES
        # every miss is a same-shape size; a different-shape size is always rejected
        assert (0, 3, 4) in accepted and (0, 3, 2) not in accepted and (2, 3, 4) not in accepted

    def test_swap_two_proof_elements(self):
        checked = 0
        for index, n, proof, r in self._cases():
            for a, b in itertools.combinations(range(len(proof)), 2):
                if proof[a] == proof[b]:
                    continue
                bad = list(proof)
                bad[a], bad[b] = bad[b], bad[a]
                assert not verify_inclusion(self.LEAVES[index], index, n, bad, r), (index, n, a, b)
                checked += 1
        assert checked > 0

    def test_truncated_or_extended_proof(self):
        for index, n, proof, r in self._cases():
            if proof:
                assert not verify_inclusion(self.LEAVES[index], index, n, proof[:-1], r)
                assert not verify_inclusion(self.LEAVES[index], index, n, proof[1:], r)
            for extra in (proof[0] if proof else r, EMPTY_ROOT):
                assert not verify_inclusion(self.LEAVES[index], index, n, list(proof) + [extra], r)
                assert not verify_inclusion(self.LEAVES[index], index, n, [extra] + list(proof), r)

    def test_wrong_byte_lengths_return_false(self):
        index, n, proof, r = next(c for c in self._cases() if c[1] == 8 and c[0] == 3)
        leaf = self.LEAVES[index]
        assert not verify_inclusion(leaf[:31], index, n, proof, r)
        assert not verify_inclusion(leaf + b"\x00", index, n, proof, r)
        assert not verify_inclusion(leaf, index, n, proof, r[:31])
        assert not verify_inclusion(leaf, index, n, proof, r + b"\x00")
        bad = list(proof)
        bad[1] = bad[1][:31]
        assert not verify_inclusion(leaf, index, n, bad, r)
        bad = list(proof)
        bad[1] = bad[1] + b"\x00"
        assert not verify_inclusion(leaf, index, n, bad, r)
        bad = list(proof)
        bad[0] = b""
        assert not verify_inclusion(leaf, index, n, bad, r)

    def test_hostile_types_return_false_never_raise(self):
        index, n, proof, r = next(c for c in self._cases() if c[1] == 8 and c[0] == 3)
        leaf = self.LEAVES[index]
        hostile = [
            (None, index, n, proof, r),
            (leaf.hex(), index, n, proof, r),
            (leaf, "3", n, proof, r),
            (leaf, index, "8", proof, r),
            (leaf, index, n, None, r),
            (leaf, index, n, proof[0], r),  # bytes, not a sequence of hashes
            (leaf, index, n, [p.hex() for p in proof], r),
            (leaf, index, n, [1, 2, 3], r),
            (leaf, index, n, proof, None),
            (leaf, index, n, proof, r.hex()),
            (leaf, True, n, proof, r),
            (leaf, index, True, proof, r),
            (leaf, index, 0, [], r),
            (leaf, 0, 0, [], r),
            (leaf, -1, n, proof, r),
            (leaf, index, -8, proof, r),
        ]
        for args in hostile:
            assert verify_inclusion(*args) is False, args


class TestConsistencyMutations:
    LEAVES = _rand_leaves(16, seed=11)  # pool covers every `other` size probed below
    N = 13

    def _cases(self):
        roots = [root(self.LEAVES[:n]) for n in range(self.N + 1)]
        for first in range(1, self.N + 1):
            for second in range(first + 1, self.N + 1):
                yield first, second, roots[first], roots[second], consistency_proof(
                    self.LEAVES, first, second
                )

    def test_baseline_verifies(self):
        for first, second, r1, r2, proof in self._cases():
            assert proof
            assert verify_consistency(first, second, r1, r2, proof)

    def test_flip_any_byte_of_any_proof_element(self):
        checked = 0
        for first, second, r1, r2, proof in self._cases():
            for k in range(len(proof)):
                for byte in range(32):
                    bad = list(proof)
                    bad[k] = _flip(proof[k], byte)
                    assert not verify_consistency(first, second, r1, r2, bad), (first, second, k, byte)
                    checked += 1
        assert checked > 0

    def test_flip_any_byte_of_either_root(self):
        for first, second, r1, r2, proof in self._cases():
            for byte in range(32):
                assert not verify_consistency(first, second, _flip(r1, byte), r2, proof)
                assert not verify_consistency(first, second, r1, _flip(r2, byte), proof)

    def test_change_first(self):
        for first, second, r1, r2, proof in self._cases():
            for other in range(-1, self.N + 3):
                if other == first:
                    continue
                assert not verify_consistency(other, second, r1, r2, proof), (first, other, second)

    # Stale-root size ambiguities for 1 <= first < second <= 13, other in -1..15, counted
    # 2026-09-08. Same finding as for inclusion: second is bound through the fold's shape.
    KNOWN_SIZE_MISSES = 308

    def test_change_second_against_honest_root(self):
        # Property asserted: PROOF(first, D[second]) never verifies as (first, other) against
        # root(other), the honest root of the changed size, for any other != second.
        roots = [root(self.LEAVES[:n]) for n in range(self.N + 3)]
        checked = 0
        for first, second, r1, r2, proof in self._cases():
            for other in range(-1, self.N + 3):
                if other == second:
                    continue
                # negative sizes have no tree; the verifier must reject them with any root
                r_other = roots[other] if other >= 0 else r2
                assert not verify_consistency(first, other, r1, r_other, proof), (first, second, other)
                checked += 1
        assert checked > 0

    def test_change_second_against_stale_root_pins_the_miss_set(self):
        # Property asserted: against the ORIGINAL second root, a changed `other` verifies iff
        # the SUBPROOF decision sequence for (first, other) equals the one for (first, second)
        # (`_subproof_shape`, from the RFC recursion). Same finding as for inclusion.
        accepted = set()
        predicted = set()
        for first, second, r1, r2, proof in self._cases():
            for other in range(-1, self.N + 3):
                if other == second:
                    continue
                if verify_consistency(first, other, r1, r2, proof):
                    accepted.add((first, second, other))
                if first < other and _subproof_shape(first, second) == _subproof_shape(first, other):
                    predicted.add((first, second, other))
        assert accepted == predicted
        assert len(accepted) == self.KNOWN_SIZE_MISSES
        assert (1, 3, 4) in accepted and (1, 3, 2) not in accepted and (3, 5, 4) not in accepted

    def test_swap_two_proof_elements(self):
        checked = 0
        for first, second, r1, r2, proof in self._cases():
            for a, b in itertools.combinations(range(len(proof)), 2):
                if proof[a] == proof[b]:
                    continue
                bad = list(proof)
                bad[a], bad[b] = bad[b], bad[a]
                assert not verify_consistency(first, second, r1, r2, bad), (first, second, a, b)
                checked += 1
        assert checked > 0

    def test_truncated_or_extended_proof(self):
        for first, second, r1, r2, proof in self._cases():
            assert not verify_consistency(first, second, r1, r2, proof[:-1])
            assert not verify_consistency(first, second, r1, r2, proof[1:])
            assert not verify_consistency(first, second, r1, r2, [])
            for extra in (proof[0], r1, r2, EMPTY_ROOT):
                assert not verify_consistency(first, second, r1, r2, list(proof) + [extra])
                assert not verify_consistency(first, second, r1, r2, [extra] + list(proof))

    def test_empty_proof_edge_cases(self):
        roots = [root(self.LEAVES[:n]) for n in range(self.N + 1)]
        for n in range(self.N + 1):
            assert verify_consistency(n, n, roots[n], roots[n], [])
            assert verify_consistency(0, n, EMPTY_ROOT, roots[n], [])
            # the size-0 tree has exactly one root: any other first_root is not consistent
            assert not verify_consistency(0, n, roots[3], roots[n], [])
            assert not verify_consistency(0, n, _flip(EMPTY_ROOT, 0), roots[n], [])
            if n:
                # equal sizes but different roots: not consistent
                assert not verify_consistency(n, n, roots[n], roots[n - 1], [])
                # a non-empty proof for the identity is malformed
                assert not verify_consistency(n, n, roots[n], roots[n], [roots[n]])
                assert not verify_consistency(0, n, EMPTY_ROOT, roots[n], [roots[n]])
        # first > second is never consistent
        assert not verify_consistency(5, 3, roots[5], roots[3], [])
        assert not verify_consistency(5, 3, roots[5], roots[3], consistency_proof(self.LEAVES, 3, 5))

    def test_wrong_byte_lengths_return_false(self):
        first, second, r1, r2, proof = next(c for c in self._cases() if c[0] == 6 and c[1] == 13)
        assert not verify_consistency(first, second, r1[:31], r2, proof)
        assert not verify_consistency(first, second, r1 + b"\x00", r2, proof)
        assert not verify_consistency(first, second, r1, r2[:31], proof)
        assert not verify_consistency(first, second, r1, r2 + b"\x00", proof)
        for k in range(len(proof)):
            bad = list(proof)
            bad[k] = bad[k][:31]
            assert not verify_consistency(first, second, r1, r2, bad)
            bad = list(proof)
            bad[k] = bad[k] + b"\x00"
            assert not verify_consistency(first, second, r1, r2, bad)

    def test_hostile_types_return_false_never_raise(self):
        first, second, r1, r2, proof = next(c for c in self._cases() if c[0] == 6 and c[1] == 13)
        hostile = [
            ("6", second, r1, r2, proof),
            (first, "13", r1, r2, proof),
            (True, second, r1, r2, proof),
            (first, True, r1, r2, proof),
            (first, second, None, r2, proof),
            (first, second, r1, None, proof),
            (first, second, r1.hex(), r2, proof),
            (first, second, r1, r2, None),
            (first, second, r1, r2, proof[0]),
            (first, second, r1, r2, [p.hex() for p in proof]),
            (first, second, r1, r2, [None] * len(proof)),
            (-1, second, r1, r2, proof),
            (first, -1, r1, r2, proof),
        ]
        for args in hostile:
            assert verify_consistency(*args) is False, args


class TestGeneratorValidation:
    LEAVES = _rand_leaves(8, seed=3)

    def test_non_32_byte_leaf_hashes_raise(self):
        short = self.LEAVES[:3] + [b"\x00" * 31]
        long_ = self.LEAVES[:3] + [b"\x00" * 33]
        short_ba = self.LEAVES[:3] + [bytearray(31)]
        long_mv = self.LEAVES[:3] + [memoryview(b"\x00" * 33)]
        for bad in (
            short,
            long_,
            short_ba,
            long_mv,
            self.LEAVES[:2] + [b""],
            self.LEAVES[:2] + ["x" * 32],
        ):
            with pytest.raises(ValueError):
                root(bad)
            with pytest.raises(ValueError):
                inclusion_proof(bad, 0)
            with pytest.raises(ValueError):
                consistency_proof(bad, 1, len(bad))
            with pytest.raises(ValueError):
                MerkleTree(bad)
        tree = MerkleTree(self.LEAVES)
        for bad_leaf in (b"\x00" * 31, b"\x00" * 33, b"", "a" * 32, None, bytearray(31), memoryview(b"\x00" * 33)):
            with pytest.raises(ValueError):
                tree.append(bad_leaf)
        assert tree.size == len(self.LEAVES)
        with pytest.raises(ValueError):
            node_hash(b"\x00" * 31, b"\x00" * 32)
        with pytest.raises(ValueError):
            node_hash(b"\x00" * 32, b"\x00" * 33)
        with pytest.raises(ValueError):
            leaf_hash("not bytes")

    def test_inclusion_index_and_size_bounds(self):
        n = len(self.LEAVES)
        with pytest.raises(ValueError):
            inclusion_proof(self.LEAVES, n)
        with pytest.raises(ValueError):
            inclusion_proof(self.LEAVES, -1)
        with pytest.raises(ValueError):
            inclusion_proof(self.LEAVES, 3, 3)  # index == tree_size
        with pytest.raises(ValueError):
            inclusion_proof(self.LEAVES, 0, n + 1)  # tree_size > len
        with pytest.raises(ValueError):
            inclusion_proof(self.LEAVES, 0, 0)
        with pytest.raises(ValueError):
            inclusion_proof(self.LEAVES, 0, -1)
        with pytest.raises(ValueError):
            inclusion_proof([], 0)
        with pytest.raises(ValueError):
            inclusion_proof(self.LEAVES, "0")
        with pytest.raises(ValueError):
            inclusion_proof(self.LEAVES, 0, "8")
        tree = MerkleTree(self.LEAVES)
        with pytest.raises(ValueError):
            tree.inclusion_proof(n)
        with pytest.raises(ValueError):
            tree.inclusion_proof(0, n + 1)
        with pytest.raises(ValueError):
            tree.root(n + 1)
        with pytest.raises(ValueError):
            tree.root(-1)

    def test_consistency_bounds(self):
        n = len(self.LEAVES)
        with pytest.raises(ValueError):
            consistency_proof(self.LEAVES, 5, 3)  # first > second
        with pytest.raises(ValueError):
            consistency_proof(self.LEAVES, -1, 3)
        with pytest.raises(ValueError):
            consistency_proof(self.LEAVES, 0, n + 1)  # second > len
        with pytest.raises(ValueError):
            consistency_proof(self.LEAVES, n + 1)  # first > size
        with pytest.raises(ValueError):
            consistency_proof(self.LEAVES, "1", 3)
        with pytest.raises(ValueError):
            consistency_proof(self.LEAVES, 1, "3")
        tree = MerkleTree(self.LEAVES)
        with pytest.raises(ValueError):
            tree.consistency_proof(5, 3)
        with pytest.raises(ValueError):
            tree.consistency_proof(1, n + 1)
        # in-range calls succeed and default `second` is the tree size
        assert tree.consistency_proof(3) == consistency_proof(self.LEAVES, 3, n)
        assert consistency_proof(self.LEAVES, 3) == consistency_proof(self.LEAVES, 3, n)

    def test_module_exports_match_contract(self):
        for name in (
            "EMPTY_ROOT",
            "leaf_hash",
            "node_hash",
            "root",
            "inclusion_proof",
            "verify_inclusion",
            "consistency_proof",
            "verify_consistency",
            "MerkleTree",
        ):
            assert hasattr(M, name), name
            assert name in M.__all__, name
        assert len(EMPTY_ROOT) == 32


# =========================================================================== hostile pass
#
# Added 2026-09-08 by the hostile tester. Nothing below is shared with the module under test:
#
#   * `_i*`  -- a hashlib-only MTH / PATH / SUBPROOF written from RFC 9162 section 2.1 (slices, not
#              indices; recursion, not a fold).
#   * `_go_*` -- a port of the transparency-dev Go verifier (decompInclProof / chainInner /
#              chainInnerRight / chainBorderRight), whose control structure differs from the
#              RFC 9162 section 2.1.3.2 / 2.1.4.2 fold the module follows.
#
# Where the module's behaviour is a finding rather than a defect, the test name starts with
# `test_known_`, the assertion pins the observed behaviour, and the docstring names the finding.
# Numbers pinned here were computed on this box on 2026-09-08 and cross-checked between the two
# verifiers; none was copied from the module or from the tests above.
#
# Five findings were repaired in the module the same day (see the module docstring); the tests
# that pinned them now assert the repair and lost the `test_known_` prefix. One divergence from
# the Go port was introduced by a repair -- verify_consistency(0, n, X, Y, []) requires
# X == EMPTY_ROOT, the Go verifier does not -- so the differential tests compare against
# `_go_ref_consistency` (the port plus that one documented check) and `test_size_zero_...`
# pins the divergence itself against the raw port.

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _H(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _ilh(e: bytes) -> bytes:
    return _H(b"\x00" + e)


def _inh(a: bytes, b: bytes) -> bytes:
    return _H(b"\x01" + a + b)


def _ik(n: int) -> int:
    k = 1
    while k * 2 < n:
        k *= 2
    return k


def _imth(d):
    if len(d) == 0:
        return _H(b"")
    if len(d) == 1:
        return d[0]
    k = _ik(len(d))
    return _inh(_imth(d[:k]), _imth(d[k:]))


def _ipath(m, d):
    if len(d) == 1:
        return []
    k = _ik(len(d))
    if m < k:
        return _ipath(m, d[:k]) + [_imth(d[k:])]
    return _ipath(m - k, d[k:]) + [_imth(d[:k])]


def _isubproof(m, d, b):
    if m == len(d):
        return [] if b else [_imth(d)]
    k = _ik(len(d))
    if m <= k:
        return _isubproof(m, d[:k], b) + [_imth(d[k:])]
    return _isubproof(m - k, d[k:], False) + [_imth(d[:k])]


def _iproof(m, d):
    return _isubproof(m, d, True)


# ---- Go port (transparency-dev/merkle proof/verify.go), bool-returning


def _go_decomp(index: int, size: int):
    inner = (index ^ (size - 1)).bit_length()
    border = bin(index >> inner).count("1")
    return inner, border


def _go_chain_inner(seed, pr, index):
    for i, h in enumerate(pr):
        seed = _inh(seed, h) if (index >> i) & 1 == 0 else _inh(h, seed)
    return seed


def _go_chain_inner_right(seed, pr, index):
    for i, h in enumerate(pr):
        if (index >> i) & 1 == 1:
            seed = _inh(h, seed)
    return seed


def _go_chain_border_right(seed, pr):
    for h in pr:
        seed = _inh(h, seed)
    return seed


def _go_verify_inclusion(leaf, index, size, pr, root_) -> bool:
    if index >= size:
        return False
    inner, border = _go_decomp(index, size)
    if len(pr) != inner + border:
        return False
    res = _go_chain_inner(leaf, pr[:inner], index)
    res = _go_chain_border_right(res, pr[inner:])
    return res == root_


def _go_verify_consistency(a, b, pr, root_a, root_b) -> bool:
    if b < a:
        return False
    if a == b:
        return root_a == root_b and len(pr) == 0
    if a == 0:
        return len(pr) == 0
    if len(pr) == 0:
        return False
    inner, border = _go_decomp(a - 1, b)
    shift = (a & -a).bit_length() - 1
    inner -= shift
    if a == 1 << shift:
        seed, start = root_a, 0
    else:
        seed, start = pr[0], 1
    if len(pr) != start + inner + border:
        return False
    pr = pr[start:]
    mask = (a - 1) >> shift
    h1 = _go_chain_border_right(_go_chain_inner_right(seed, pr[:inner], mask), pr[inner:])
    if h1 != root_a:
        return False
    h2 = _go_chain_border_right(_go_chain_inner(seed, pr[:inner], mask), pr[inner:])
    return h2 == root_b


def _go_ref_consistency(a, b, pr, root_a, root_b) -> bool:
    """The Go port plus the module's one documented divergence from it: for a == 0 the module
    also requires root_a == EMPTY_ROOT (the size-0 tree has exactly one root). When b == 0 as
    well, the port's own root_a == root_b check then forces root_b == EMPTY_ROOT too."""
    ok = _go_verify_consistency(a, b, pr, root_a, root_b)
    if a == 0:
        ok = ok and root_a == EMPTY_ROOT
    return ok


_POOL = _rand_leaves(40, seed=1)
_POOL_ROOTS = [root(_POOL[:n]) for n in range(41)]


class TestIndependentRecompute:
    """The contract's quoted vectors, recomputed with hashlib alone."""

    def test_ct_roots_hashlib_only(self):
        for n in range(9):
            mine = _imth([_ilh(e) for e in CT_ENTRIES[:n]])
            assert mine.hex() == CT_ROOTS[n], n
            assert mine == root(CT_LEAVES[:n]), n

    def test_ct_inclusion_vectors_hashlib_only(self):
        d = [_ilh(e) for e in CT_ENTRIES]
        for (index, size), want in CT_INCLUSION.items():
            mine = _ipath(index, d[:size])
            assert _hx(mine) == want
            assert mine == inclusion_proof(CT_LEAVES, index, size)

    def test_ct_consistency_vectors_hashlib_only(self):
        d = [_ilh(e) for e in CT_ENTRIES]
        for (a, b), want in CT_CONSISTENCY.items():
            mine = _iproof(a, d[:b])
            assert _hx(mine) == want
            assert mine == consistency_proof(CT_LEAVES, a, b)

    def test_generators_agree_with_independent_recursion_sizes_0_to_40(self):
        for n in range(0, 41):
            d = _POOL[:n]
            assert root(d) == _imth(d)
            for i in range(n):
                assert inclusion_proof(_POOL, i, n) == _ipath(i, d), (i, n)
            for a in range(1, n + 1):
                if a == n:
                    continue
                assert consistency_proof(_POOL, a, n) == _iproof(a, d), (a, n)


class TestGoPortDifferential:
    """The RFC 9162 fold (module) against the Go decomposition (port) on the same inputs."""

    # Stale-root size ambiguities on _POOL: (index, size, other) for 1 <= size <= 40, other in
    # 0..59, other != size, and (a, b, other) for 1 <= a < b <= 40. Both verifiers accept exactly
    # the same set, and the consistency set is the inclusion set shifted by one leaf.
    KNOWN_STALE_INCLUSION = 11672
    KNOWN_STALE_CONSISTENCY = 11672

    def test_valid_proofs_accepted_by_both(self):
        for n in range(1, 41):
            for i in range(n):
                pr = inclusion_proof(_POOL, i, n)
                assert verify_inclusion(_POOL[i], i, n, pr, _POOL_ROOTS[n])
                assert _go_verify_inclusion(_POOL[i], i, n, pr, _POOL_ROOTS[n])
        for a in range(0, 41):
            for b in range(a, 41):
                pr = consistency_proof(_POOL, a, b)
                assert verify_consistency(a, b, _POOL_ROOTS[a], _POOL_ROOTS[b], pr)
                assert _go_verify_consistency(a, b, pr, _POOL_ROOTS[a], _POOL_ROOTS[b])
                assert _go_ref_consistency(a, b, pr, _POOL_ROOTS[a], _POOL_ROOTS[b])

    def test_stale_root_size_misses_identical_and_pinned(self):
        incl_mod, incl_go = set(), set()
        for n in range(1, 41):
            for i in range(n):
                pr = inclusion_proof(_POOL, i, n)
                for other in range(0, 60):
                    if other == n:
                        continue
                    if verify_inclusion(_POOL[i], i, other, pr, _POOL_ROOTS[n]):
                        incl_mod.add((i, n, other))
                    if _go_verify_inclusion(_POOL[i], i, other, pr, _POOL_ROOTS[n]):
                        incl_go.add((i, n, other))
        assert incl_mod == incl_go
        assert len(incl_mod) == self.KNOWN_STALE_INCLUSION
        cons_mod, cons_go = set(), set()
        for a in range(1, 41):
            for b in range(a + 1, 41):
                pr = consistency_proof(_POOL, a, b)
                for other in range(0, 60):
                    if other == b:
                        continue
                    if verify_consistency(a, other, _POOL_ROOTS[a], _POOL_ROOTS[b], pr):
                        cons_mod.add((a, b, other))
                    if _go_verify_consistency(a, other, pr, _POOL_ROOTS[a], _POOL_ROOTS[b]):
                        cons_go.add((a, b, other))
        assert cons_mod == cons_go
        assert len(cons_mod) == self.KNOWN_STALE_CONSISTENCY
        # `second` is bound exactly as tree_size is bound for leaf index a - 1 ...
        assert {(a - 1, b, o) for (a, b, o) in cons_mod} == incl_mod
        # ... and the last leaf of any size is never ambiguous.
        assert not [t for t in incl_mod if t[0] == t[1] - 1]

    def test_mutated_inclusion_proofs_agree(self):
        checked = 0
        for n in range(1, 17):
            for i in range(n):
                pr = inclusion_proof(_POOL, i, n)
                variants = []
                for k in range(len(pr)):
                    for byte in (0, 15, 31):
                        v = list(pr)
                        v[k] = _flip(v[k], byte)
                        variants.append(v)
                if pr:
                    variants += [pr[:-1], pr[1:]]
                variants += [pr + [_POOL_ROOTS[3]], [_POOL_ROOTS[3]] + pr]
                for x, y in itertools.combinations(range(len(pr)), 2):
                    v = list(pr)
                    v[x], v[y] = v[y], v[x]
                    variants.append(v)
                for v in variants:
                    for other in range(0, 21):
                        m = verify_inclusion(_POOL[i], i, other, v, _POOL_ROOTS[n])
                        g = _go_verify_inclusion(_POOL[i], i, other, v, _POOL_ROOTS[n])
                        assert m == g, (i, n, other, _hx(v))
                        checked += 1
        assert checked > 10000

    def test_mutated_consistency_proofs_agree(self):
        checked = 0
        for a in range(1, 17):
            for b in range(a + 1, 17):
                pr = consistency_proof(_POOL, a, b)
                variants = [pr[:-1], pr[1:], [], pr + [_POOL_ROOTS[a]], [_POOL_ROOTS[a]] + pr]
                for k in range(len(pr)):
                    for byte in (0, 31):
                        v = list(pr)
                        v[k] = _flip(v[k], byte)
                        variants.append(v)
                for x, y in itertools.combinations(range(len(pr)), 2):
                    v = list(pr)
                    v[x], v[y] = v[y], v[x]
                    variants.append(v)
                for v in variants:
                    for other in range(0, 21):
                        m = verify_consistency(a, other, _POOL_ROOTS[a], _POOL_ROOTS[b], v)
                        g = _go_ref_consistency(a, other, v, _POOL_ROOTS[a], _POOL_ROOTS[b])
                        assert m == g, (a, b, other, _hx(v))
                        checked += 1
        assert checked > 5000

    def test_seeded_garbage_agrees(self):
        rng = random.Random(20260908)
        for _ in range(5000):
            pr = [rng.choice(_POOL) for _ in range(rng.randint(0, 8))]
            leaf, r = rng.choice(_POOL), rng.choice(_POOL_ROOTS)
            idx, sz = rng.randint(0, 45), rng.randint(0, 45)
            assert verify_inclusion(leaf, idx, sz, pr, r) == _go_verify_inclusion(leaf, idx, sz, pr, r)
            a, b = rng.randint(0, 45), rng.randint(0, 45)
            r1, r2 = rng.choice(_POOL_ROOTS), rng.choice(_POOL_ROOTS)
            assert verify_consistency(a, b, r1, r2, pr) == _go_ref_consistency(a, b, pr, r1, r2)

    @settings(deadline=None, max_examples=300)
    @given(data=st.data())
    def test_hypothesis_garbage_agrees(self, data):
        pool = st.sampled_from(_POOL + _POOL_ROOTS)
        pr = data.draw(st.lists(pool, max_size=9))
        leaf, r = data.draw(pool), data.draw(pool)
        idx = data.draw(st.integers(min_value=0, max_value=70))
        sz = data.draw(st.integers(min_value=0, max_value=70))
        assert verify_inclusion(leaf, idx, sz, pr, r) == _go_verify_inclusion(leaf, idx, sz, pr, r)
        a = data.draw(st.integers(min_value=0, max_value=70))
        b = data.draw(st.integers(min_value=0, max_value=70))
        assert verify_consistency(a, b, leaf, r, pr) == _go_ref_consistency(a, b, pr, leaf, r)

    def test_size_zero_divergence_from_go_is_exactly_the_empty_root_check(self):
        """Against the RAW Go port: for a == 0 the module accepts a strict subset of what the
        port accepts, and the subset is exactly {root_a == EMPTY_ROOT}. For a > 0 there is no
        divergence at all. Both halves are counted so neither can be vacuous."""
        rng = random.Random(20260908)
        diverged = agreed_zero = 0
        for _ in range(3000):
            pr = [rng.choice(_POOL) for _ in range(rng.choice([0, 0, 1, 2]))]
            a, b = rng.choice([0, 0, 0, 1, 2, 5]), rng.randint(0, 12)
            r1, r2 = rng.choice(_POOL_ROOTS[:3]), rng.choice(_POOL_ROOTS[:3])
            m = verify_consistency(a, b, r1, r2, pr)
            g = _go_verify_consistency(a, b, pr, r1, r2)
            if a != 0:
                assert m == g, (a, b, r1.hex(), r2.hex(), _hx(pr))
                continue
            assert m == (g and r1 == EMPTY_ROOT), (b, r1.hex(), r2.hex(), _hx(pr))
            if m != g:
                assert m is False and g is True and r1 != EMPTY_ROOT
                diverged += 1
            elif m is True:
                agreed_zero += 1
        assert diverged > 100
        assert agreed_zero > 100


class _Liar(bytes):
    """A bytes subclass whose len() lies: 32 whatever the payload."""

    def __len__(self):
        return 32


class _Sub(bytes):
    pass


class _RaisesOnIter:
    def __iter__(self):
        raise RuntimeError("boom")


class _RaisesOnLen:
    def __len__(self):
        raise ValueError("boom")

    def __iter__(self):
        return iter([])


class _RaisesOnClass:
    """Even `isinstance(x, bytes)` raises: `__class__` is consulted after the type check."""

    @property
    def __class__(self):
        raise RuntimeError("boom")

    def __iter__(self):
        return iter([])


def _yields_then_raises(item):
    yield item
    raise KeyError("boom")


class _Idx(enum.IntEnum):
    ONE = 1
    FOUR = 4


class TestHostileInputs:
    LEAF = _POOL[1]
    PROOF = inclusion_proof(_POOL, 1, 4)
    ROOT4 = _POOL_ROOTS[4]

    def test_verify_never_raises_on_exotic_arguments(self):
        L, P, R = self.LEAF, self.PROOF, self.ROOT4
        cases = [
            (L, 1, 4, P, R),
            (_Liar(b"\x00" * 31), 0, 2, [_POOL[1]], _POOL_ROOTS[2]),
            (_Liar(b"\x00" * 33), 0, 2, [_POOL[1]], _POOL_ROOTS[2]),
            (L, 1, 4, [_Liar(b"\x00" * 31)] + P[1:], R),
            (L, 1, 4, P, _Liar(b"\x00" * 33)),
            (bytearray(L), 1, 4, P, R),
            (memoryview(L), 1, 4, P, R),
            (L, 1, 4, [bytearray(p) for p in P], R),
            (L, 1, 4, [memoryview(p) for p in P], R),
            (L, 1, 4, P, bytearray(R)),
            (L, 1.0, 4, P, R),
            (L, 1, 4.0, P, R),
            (L, 1 + 0j, 4, P, R),
            (L, 1, 4, 3.5, R),
            (L, 1, 4, 7, R),
            (L, 1, 4, range(3), R),
            (L, 1, 4, [[p] for p in P], R),
            (L, 1, 4, [(p,) for p in P], R),
            (L, 1, 4, [P], R),
            (L, 1, 4, tuple(P), R),
            (L, 1, 4, iter(P), R),
            (L, 1, 4, b"".join(P), R),
            (L, 1, 4, "".join(p.hex() for p in P), R),
            (L, 1, 4, [True, False], R),
            (L, 1, 4, [None], R),
            (L, 1, 4, [b"\x00" * 32] * 64, R),
            (L, -(2 ** 70), 4, P, R),
            (L, 1, -(2 ** 70), P, R),
            (L, 2 ** 70, 2 ** 70 + 1, P, R),
            (L, 0, 2 ** 70, [], L),
            (b"", 0, 1, [], b""),
            (L, 0, 1, [], L),
        ]
        trues = 0
        for args in cases:
            out = verify_inclusion(*args)
            assert out is True or out is False, args
            trues += out is True
        # exactly the honest shapes are accepted: list, tuple, iterator, the singleton, and
        # bytes-like (bytearray / memoryview) hashes in any of the three hash positions
        assert trues == 9
        assert verify_inclusion(L, 1, 4, tuple(P), R) is True
        assert verify_inclusion(L, 1, 4, iter(P), R) is True
        assert verify_inclusion(L, 0, 1, [], L) is True
        assert verify_inclusion(L, 1, 4, P, R) is True
        assert verify_inclusion(bytearray(L), 1, 4, P, R) is True
        assert verify_inclusion(memoryview(L), 1, 4, P, R) is True
        assert verify_inclusion(L, 1, 4, [bytearray(p) for p in P], R) is True
        assert verify_inclusion(L, 1, 4, [memoryview(p) for p in P], R) is True
        assert verify_inclusion(L, 1, 4, P, bytearray(R)) is True
        # a bytes subclass is read from its buffer, not from what its __len__ claims
        assert verify_inclusion(_Liar(b"\x00" * 31), 0, 2, [_POOL[1]], _POOL_ROOTS[2]) is False
        assert verify_inclusion(L, 1, 4, P, _Liar(b"\x00" * 33)) is False
        # consistency: the same shapes
        r2, r4 = _POOL_ROOTS[2], _POOL_ROOTS[4]
        cp = consistency_proof(_POOL, 2, 4)
        ccases = [
            (2, 4, r2, r4, cp),
            (2.0, 4, r2, r4, cp),
            (2, 4.0, r2, r4, cp),
            (2, 4, bytearray(r2), r4, cp),
            (2, 4, r2, memoryview(r4), cp),
            (2, 4, r2, r4, 3.5),
            (2, 4, r2, r4, b"".join(cp)),
            (2, 4, r2, r4, [bytearray(p) for p in cp]),
            (2, 4, r2, r4, tuple(cp)),
            (2, 4, r2, r4, iter(cp)),
            (-(2 ** 70), 4, r2, r4, cp),
            (2, 2 ** 70, r2, r4, cp),
            (2 ** 70, 2 ** 70 + 1, r2, r4, [r2]),
            (2 ** 70, 2 ** 70, r2, r2, []),
            (_Liar(b"\x00" * 31), 4, r2, r4, cp),
        ]
        trues = 0
        for args in ccases:
            out = verify_consistency(*args)
            assert out is True or out is False, args
            trues += out is True
        # cp, tuple, iterator, the a == b identity, and the three bytes-like shapes
        assert trues == 7
        assert verify_consistency(2, 4, bytearray(r2), r4, cp) is True
        assert verify_consistency(2, 4, r2, memoryview(r4), cp) is True
        assert verify_consistency(2, 4, r2, r4, [bytearray(p) for p in cp]) is True

    def test_int_subclass_indices_behave_as_ints(self):
        assert inclusion_proof(_POOL, _Idx.ONE, _Idx.FOUR) == self.PROOF
        assert verify_inclusion(self.LEAF, _Idx.ONE, _Idx.FOUR, self.PROOF, self.ROOT4) is True
        assert consistency_proof(_POOL, _Idx.ONE, _Idx.FOUR) == consistency_proof(_POOL, 1, 4)
        assert MerkleTree(_POOL).root(_Idx.FOUR) == self.ROOT4

    def test_bytes_subclass_hashes_are_accepted_by_generators_and_verifiers(self):
        subs = [_Sub(h) for h in _POOL[:5]]
        assert root(subs) == _POOL_ROOTS[5]
        tree = MerkleTree()
        for h in subs:
            tree.append(h)
        assert tree.root() == _POOL_ROOTS[5]
        pr = [_Sub(p) for p in inclusion_proof(subs, 3)]
        assert verify_inclusion(_Sub(_POOL[3]), 3, 5, pr, _Sub(_POOL_ROOTS[5])) is True

    def test_bool_is_rejected_everywhere(self):
        for bad in (True, False):
            with pytest.raises(ValueError):
                inclusion_proof(_POOL, bad, 4)
            with pytest.raises(ValueError):
                inclusion_proof(_POOL, 0, bad)
            with pytest.raises(ValueError):
                consistency_proof(_POOL, bad, 4)
            with pytest.raises(ValueError):
                consistency_proof(_POOL, 1, bad)
            with pytest.raises(ValueError):
                MerkleTree(_POOL).root(bad)
            with pytest.raises(ValueError):
                leaf_hash(bad)
            assert verify_inclusion(self.LEAF, bad, 4, self.PROOF, self.ROOT4) is False
            assert verify_inclusion(self.LEAF, 1, bad, self.PROOF, self.ROOT4) is False
            assert verify_consistency(bad, 4, _POOL_ROOTS[2], self.ROOT4, consistency_proof(_POOL, 2, 4)) is False
            assert verify_consistency(2, bad, _POOL_ROOTS[2], self.ROOT4, consistency_proof(_POOL, 2, 4)) is False
        # True is not index 1 even where 1 would verify
        assert verify_inclusion(self.LEAF, True, 4, self.PROOF, self.ROOT4) is False

    def test_huge_integers_and_long_proofs_return_false_without_raising(self):
        big = 2 ** 100000
        assert verify_inclusion(self.LEAF, 0, big, [], self.LEAF) is False
        assert verify_inclusion(self.LEAF, big - 1, big, [], self.LEAF) is False
        assert verify_inclusion(self.LEAF, big, big + 1, [self.LEAF] * 3, self.LEAF) is False
        assert verify_consistency(big, big + 1, self.LEAF, self.LEAF, [self.LEAF]) is False
        assert verify_consistency(1, big, self.LEAF, self.LEAF, [self.LEAF] * 5) is False
        long_proof = [_POOL[2]] * 100000
        assert verify_inclusion(self.LEAF, 0, 2 ** 100000, long_proof, self.ROOT4) is False
        assert verify_consistency(3, 2 ** 100000, self.LEAF, self.ROOT4, long_proof) is False

    def test_generators_raise_valueerror_on_non_iterable_leaves(self):
        """Repair of finding 2: a non-iterable `leaf_hashes` (None, an int, a float, an
        object) is an argument-validation failure and raises ValueError from every generator,
        not the bare TypeError from list(). A `str` or a bytes-like standing in for the
        sequence is refused the same way (b"" is not an empty tree)."""
        for bad in (None, 5, 3.5, object(), "", "x" * 32, b"", b"\x00" * 32, bytearray(32), memoryview(b"\x00" * 32)):
            with pytest.raises(ValueError):
                root(bad)
            with pytest.raises(ValueError):
                MerkleTree(bad)
            with pytest.raises(ValueError):
                inclusion_proof(bad, 0)
            with pytest.raises(ValueError):
                consistency_proof(bad, 0)
        # the chain names the cause
        with pytest.raises(ValueError) as info:
            root(None)
        assert isinstance(info.value.__cause__, TypeError)
        # an exception raised by the caller's OWN iterator while it is consumed is the
        # caller's and propagates unchanged from a generator (only verifiers swallow)
        with pytest.raises(RuntimeError):
            root(_RaisesOnIter())
        with pytest.raises(KeyError):
            MerkleTree(_yields_then_raises(self.LEAF))

    def test_verifiers_return_false_when_the_proof_object_raises(self):
        """Repair of finding 1: whatever the proof object raises -- from __iter__, __len__,
        __next__ mid-way, or even from `isinstance` via a hostile __class__ -- verify_* returns
        False. The same holds for a hostile object in any hash position."""
        L, P, R = self.LEAF, self.PROOF, self.ROOT4
        # factories: a generator is one-shot, so every call gets a fresh hostile object
        hostile_proofs = [
            _RaisesOnIter,
            _RaisesOnClass,
            lambda: _yields_then_raises(P[0]),
            lambda: [_RaisesOnClass()] + P[1:],
            lambda: [_RaisesOnClass()],
        ]
        for make in hostile_proofs:
            assert verify_inclusion(L, 1, 4, make(), R) is False
            assert verify_inclusion(L, 0, 1, make(), L) is False
            assert verify_consistency(1, 2, _POOL_ROOTS[1], _POOL_ROOTS[2], make()) is False
            assert verify_consistency(2, 2, _POOL_ROOTS[2], _POOL_ROOTS[2], make()) is False
            assert verify_consistency(0, 2, EMPTY_ROOT, _POOL_ROOTS[2], make()) is False
        # `__len__` is never consulted (the proof is consumed through a bounded slice), so an
        # object whose __len__ raises but whose iteration is empty IS the empty proof: no raise,
        # False where a proof is needed, True exactly where [] is the honest proof
        assert verify_inclusion(L, 1, 4, _RaisesOnLen(), R) is False
        assert verify_inclusion(L, 0, 1, _RaisesOnLen(), L) is True
        assert verify_consistency(1, 2, _POOL_ROOTS[1], _POOL_ROOTS[2], _RaisesOnLen()) is False
        assert verify_consistency(2, 2, _POOL_ROOTS[2], _POOL_ROOTS[2], _RaisesOnLen()) is True
        assert verify_consistency(0, 2, EMPTY_ROOT, _POOL_ROOTS[2], _RaisesOnLen()) is True
        assert verify_inclusion(_RaisesOnClass(), 1, 4, P, R) is False
        assert verify_inclusion(L, _RaisesOnClass(), 4, P, R) is False
        assert verify_inclusion(L, 1, _RaisesOnClass(), P, R) is False
        assert verify_inclusion(L, 1, 4, P, _RaisesOnClass()) is False
        assert verify_consistency(_RaisesOnClass(), 4, _POOL_ROOTS[2], R, consistency_proof(_POOL, 2, 4)) is False
        assert verify_consistency(2, 4, _RaisesOnClass(), R, consistency_proof(_POOL, 2, 4)) is False
        assert verify_consistency(2, 4, _POOL_ROOTS[2], _RaisesOnClass(), consistency_proof(_POOL, 2, 4)) is False

    def test_endless_proof_iterators_return_false_without_hanging(self):
        """The proof is consumed under a bound above the longest proof the fold can accept, so
        an endless iterator is decided (False) instead of hanging list()."""
        L, P, R = self.LEAF, self.PROOF, self.ROOT4
        assert verify_inclusion(L, 1, 4, itertools.repeat(P[0]), R) is False
        assert verify_inclusion(L, 0, 1, itertools.repeat(L), L) is False
        assert verify_inclusion(L, 0, 2 ** 70, itertools.repeat(L), L) is False
        assert verify_inclusion(L, 1, 4, itertools.count(), R) is False
        assert verify_consistency(2, 4, _POOL_ROOTS[2], R, itertools.repeat(P[0])) is False
        assert verify_consistency(0, 4, EMPTY_ROOT, R, itertools.repeat(P[0])) is False
        assert verify_consistency(4, 4, R, R, itertools.repeat(P[0])) is False
        assert verify_consistency(3, 2 ** 70, _POOL_ROOTS[3], R, itertools.repeat(P[0])) is False
        # the bound changes no verdict: a finite proof one longer than the fold can consume is
        # rejected by the fold itself, and the honest proof still verifies through the slice
        assert verify_inclusion(L, 1, 4, iter(P + [P[0]] * 3), R) is False
        assert verify_inclusion(L, 1, 4, iter(P), R) is True

    def test_bytes_like_hashes_are_accepted_uniformly(self):
        """Repair of finding 5: node_hash, root, the generators, MerkleTree.append and both
        verifiers accept bytes / bytearray / memoryview / a bytes subclass alike, canonicalised
        to exact `bytes` from the buffer. A subclass whose __len__ lies is measured by its
        buffer: 31 real bytes is a bad hash everywhere."""
        L, P, R = self.LEAF, self.PROOF, self.ROOT4
        assert leaf_hash(bytearray(b"a")) == leaf_hash(memoryview(b"a")) == leaf_hash(b"a")
        ba, mv = bytearray(L), memoryview(L)
        assert node_hash(ba, L) == node_hash(L, ba) == node_hash(mv, mv) == node_hash(L, L)
        assert root([ba]) == root([mv]) == root([L]) == L
        assert root([bytearray(h) for h in _POOL[:5]]) == _POOL_ROOTS[5]
        assert root([memoryview(h) for h in _POOL[:5]]) == _POOL_ROOTS[5]
        assert inclusion_proof([bytearray(h) for h in _POOL[:4]], 1) == P
        assert consistency_proof([memoryview(h) for h in _POOL[:4]], 2) == consistency_proof(_POOL, 2, 4)
        tree = MerkleTree([bytearray(h) for h in _POOL[:2]])
        assert tree.append(memoryview(_POOL[2])) == 2
        assert tree.append(_Sub(_POOL[3])) == 3
        assert tree.root() == R
        assert tree.leaves() == _POOL[:4]
        assert all(type(h) is bytes for h in tree.leaves())
        assert verify_inclusion(ba, 1, 4, P, R) is True
        assert verify_inclusion(mv, 1, 4, [memoryview(p) for p in P], bytearray(R)) is True
        assert verify_consistency(2, 4, bytearray(_POOL_ROOTS[2]), memoryview(R), consistency_proof(_POOL, 2, 4)) is True
        # lying __len__: refused by generators, False from verifiers, and never hashed as-is
        liar = _Liar(b"\x00" * 31)
        for bad in (liar, bytearray(31), memoryview(b"\x00" * 33)):
            with pytest.raises(ValueError):
                node_hash(bad, L)
            with pytest.raises(ValueError):
                root([bad])
            with pytest.raises(ValueError):
                MerkleTree().append(bad)
            assert verify_inclusion(bad, 1, 4, P, R) is False
            assert verify_inclusion(L, 1, 4, [bad] + P[1:], R) is False
            assert verify_inclusion(L, 1, 4, P, bad) is False
        # a truthful subclass is a hash like any other, stored as exact bytes
        assert type(MerkleTree([_Liar(L)]).leaves()[0]) is bytes
        assert root([_Liar(L)]) == L

    def test_unordered_containers_are_refused(self):
        """Repair of finding 3: set / frozenset / dict (any Set or Mapping) have no order that
        is a property of the value, so generators raise ValueError and verifiers return False
        -- even for a dict, whose insertion order would have happened to work."""
        leaves = _POOL[:8]
        pr = inclusion_proof(leaves, 0)
        unordered = [
            set(leaves),
            frozenset(leaves),
            dict.fromkeys(leaves),
            dict.fromkeys(leaves).keys(),
            {i: h for i, h in enumerate(leaves)},
        ]
        for bad in unordered:
            with pytest.raises(ValueError):
                root(bad)
            with pytest.raises(ValueError):
                MerkleTree(bad)
            with pytest.raises(ValueError):
                inclusion_proof(bad, 0)
            with pytest.raises(ValueError):
                consistency_proof(bad, 1, 2)
        for bad in (set(pr), frozenset(pr), dict.fromkeys(pr), dict.fromkeys(pr).keys(), {}, set()):
            assert verify_inclusion(leaves[0], 0, 8, bad, _POOL_ROOTS[8]) is False
            assert verify_consistency(1, 8, leaves[0], _POOL_ROOTS[8], bad) is False
        # the empty unordered container is refused too, even where [] would verify
        assert verify_inclusion(leaves[0], 0, 1, set(), leaves[0]) is False
        assert verify_consistency(0, 8, EMPTY_ROOT, _POOL_ROOTS[8], frozenset()) is False
        assert verify_consistency(8, 8, _POOL_ROOTS[8], _POOL_ROOTS[8], {}) is False
        # ordered non-list containers are still fine, a dict's values view included
        assert root(tuple(leaves)) == _POOL_ROOTS[8]
        assert root(iter(leaves)) == _POOL_ROOTS[8]
        assert root(list(dict.fromkeys(leaves))) == _POOL_ROOTS[8]
        assert root({i: h for i, h in enumerate(leaves)}.values()) == _POOL_ROOTS[8]

    @staticmethod
    def _run(tmp_path, code: str, seed: "int | None" = None) -> str:
        env = dict(os.environ)
        env.pop("PYTHONHASHSEED", None)
        if seed is not None:
            env["PYTHONHASHSEED"] = str(seed)
        prelude = "import sys; sys.path.insert(0, %r); from styxx.v8 import merkle as M\n" % str(REPO_ROOT)
        script = tmp_path / ("probe_%s.py" % (seed if seed is not None else "x"))
        script.write_bytes((prelude + code).encode("utf-8"))
        out = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(REPO_ROOT),
            timeout=120,
        )
        assert out.returncode == 0, out.stderr
        return out.stdout.strip()

    def test_determinism_across_two_processes(self, tmp_path):
        code = (
            "import random\n"
            "rng = random.Random(20260908)\n"
            "L = [bytes(rng.getrandbits(8) for _ in range(32)) for _ in range(33)]\n"
            "t = M.MerkleTree(L)\n"
            "parts = [t.root(n).hex() for n in range(34)]\n"
            "parts += [p.hex() for n in range(1, 34) for i in range(n) for p in t.inclusion_proof(i, n)]\n"
            "parts += [p.hex() for a in range(34) for b in range(a, 34) for p in t.consistency_proof(a, b)]\n"
            "print(M.root([M.leaf_hash(e) for e in [b'', b'\\x00', b'\\x10']]).hex())\n"
            "import hashlib; print(hashlib.sha256(''.join(parts).encode()).hexdigest())\n"
        )
        one = self._run(tmp_path, code, seed=1)
        two = self._run(tmp_path, code, seed=2)
        assert one == two
        assert one.splitlines()[0] == CT_ROOTS[3]

    SET_ROOT_SEEDS = 6

    def test_set_leaves_are_refused_under_every_hash_seed(self, tmp_path):
        """Repair of finding 3, across processes: before the repair `root(set_of_hashes)` was a
        function of PYTHONHASHSEED (six seeds, eight leaves, more than one set root). Now the
        list root is identical under every seed and the set is refused under every seed, so
        no hash seed can reach a root through an unordered container."""
        code = (
            "L = [bytes([i]) * 32 for i in range(8)]\n"
            "print(M.root(L).hex())\n"
            "try:\n"
            "    M.root(set(L)); print('accepted')\n"
            "except ValueError:\n"
            "    print('refused')\n"
            "print('refused' if M.verify_inclusion(L[0], 0, 8, set(M.inclusion_proof(L, 0)), M.root(L)) is False else 'accepted')\n"
        )
        list_roots, verdicts = set(), set()
        for seed in range(self.SET_ROOT_SEEDS):
            a, b, c = self._run(tmp_path, code, seed=seed).splitlines()
            list_roots.add(a)
            verdicts.add(b)
            verdicts.add(c)
        assert len(list_roots) == 1
        assert verdicts == {"refused"}

    def test_consistency_from_size_zero_requires_the_empty_root(self):
        """Repair of finding 4: with an empty proof, verify_consistency(0, n, X, Y, []) is True
        iff X == EMPTY_ROOT -- the size-0 tree has exactly one root -- and Y is unconstrained
        for n > 0 (every tree extends the empty tree). (0, 0) needs EMPTY_ROOT on both sides.
        The raw Go port accepts any X; the divergence is pinned here on purpose."""
        for n in range(1, 10):
            assert verify_consistency(0, n, EMPTY_ROOT, _POOL_ROOTS[n], []) is True
            assert verify_consistency(0, n, EMPTY_ROOT, b"\x00" * 32, []) is True
            assert verify_consistency(0, n, b"\xff" * 32, b"\x00" * 32, []) is False
            assert verify_consistency(0, n, _POOL_ROOTS[3], _POOL_ROOTS[n], []) is False
            assert verify_consistency(0, n, _POOL_ROOTS[n], _POOL_ROOTS[n], []) is False
            for byte in range(32):
                assert verify_consistency(0, n, _flip(EMPTY_ROOT, byte), _POOL_ROOTS[n], []) is False
            assert verify_consistency(0, n, EMPTY_ROOT, _POOL_ROOTS[n], [EMPTY_ROOT]) is False
            assert verify_consistency(0, n, EMPTY_ROOT, _POOL_ROOTS[n], [_POOL_ROOTS[n]]) is False
            # the raw Go port accepts what the module now refuses
            assert _go_verify_consistency(0, n, [], b"\xff" * 32, b"\x00" * 32) is True
        assert verify_consistency(0, 0, EMPTY_ROOT, EMPTY_ROOT, []) is True
        assert verify_consistency(0, 0, b"\xff" * 32, b"\xff" * 32, []) is False
        assert verify_consistency(0, 0, EMPTY_ROOT, b"\xff" * 32, []) is False
        assert verify_consistency(0, 0, b"\xff" * 32, EMPTY_ROOT, []) is False
        assert verify_consistency(0, 0, EMPTY_ROOT, EMPTY_ROOT, [EMPTY_ROOT]) is False
        # with a == b > 0 any pair of EQUAL roots is still accepted, whatever they are: the
        # verifier cannot know the root of a tree it has not seen, so this is not a defect
        assert verify_consistency(7, 7, b"\xff" * 32, b"\xff" * 32, []) is True
        assert verify_consistency(7, 7, b"\xff" * 32, b"\xfe" * 32, []) is False

    def test_known_verifier_does_not_bind_tree_size_to_the_root(self):
        """FINDING (inherent to RFC 6962 verification; the contract does not say so): the root of a
        four-leaf tree accepts an inclusion proof that claims tree_size 2 for a 'leaf' that is
        really the internal node over leaves 0..1. Only a signed (size, root) tree head binds the
        size; verify_inclusion alone cannot. Callers must derive leaf_hash from the raw entry
        themselves and check tree_size against the tree head."""
        a, b, c, d = _POOL[:4]
        fake_leaf = node_hash(a, b)
        assert verify_inclusion(fake_leaf, 0, 2, [node_hash(c, d)], _POOL_ROOTS[4]) is True
        assert _go_verify_inclusion(fake_leaf, 0, 2, [node_hash(c, d)], _POOL_ROOTS[4]) is True
        # the same claim with the honest size is rejected: shape (2 elements) does not match
        assert verify_inclusion(fake_leaf, 0, 4, [node_hash(c, d)], _POOL_ROOTS[4]) is False

    def test_leaf_and_node_domains_are_separated(self):
        a, b = _POOL[:2]
        assert leaf_hash(a + b) != node_hash(a, b)
        assert leaf_hash(b"\x01" + a + b) != node_hash(a, b)
        assert leaf_hash(a) != a
        # a 64-byte entry that spells out two leaf hashes is not the pair's node
        assert verify_inclusion(leaf_hash(a + b), 0, 1, [], root([a, b])) is False
        # node_hash is not symmetric and not idempotent under swap of halves
        assert node_hash(a, b) != node_hash(b, a)
        assert node_hash(a, a) != leaf_hash(a + a)

    def test_tree_is_isolated_from_the_caller_list_and_leaves_returns_a_copy(self):
        src = list(_POOL[:3])
        tree = MerkleTree(src)
        src.append(_POOL[3])
        src[0] = _POOL[9]
        assert tree.size == 3
        assert tree.root() == _POOL_ROOTS[3]
        lv = tree.leaves()
        lv.append(_POOL[5])
        lv[0] = _POOL[7]
        assert tree.size == 3
        assert tree.root() == _POOL_ROOTS[3]
        assert tree.leaves() == _POOL[:3]

    def test_historical_size_proofs_do_not_depend_on_later_appends(self):
        full = MerkleTree(_POOL)
        for m in range(1, 41):
            prefix = MerkleTree(_POOL[:m])
            assert full.root(m) == prefix.root() == _POOL_ROOTS[m]
            for i in range(m):
                pr = full.inclusion_proof(i, m)
                assert pr == prefix.inclusion_proof(i)
                assert verify_inclusion(_POOL[i], i, m, pr, _POOL_ROOTS[m])
            for a in range(0, m + 1):
                assert full.consistency_proof(a, m) == prefix.consistency_proof(a)
        # an inclusion proof generated at size m is rejected by the honest root at every other size
        for m in range(1, 41):
            for i in range(m):
                pr = full.inclusion_proof(i, m)
                for other in range(max(1, i + 1), 41):
                    if other != m:
                        assert not verify_inclusion(_POOL[i], i, other, pr, _POOL_ROOTS[other]), (i, m, other)

    def test_index_equal_to_size_minus_one_every_size(self):
        """The fn == sn branch of the RFC fold, exercised for every size up to 40."""
        for n in range(1, 41):
            pr = inclusion_proof(_POOL, n - 1, n)
            # the last leaf's path has one element per set bit of n - 1
            assert len(pr) == bin(n - 1).count("1")
            assert verify_inclusion(_POOL[n - 1], n - 1, n, pr, _POOL_ROOTS[n])
            rev = list(reversed(pr))
            if rev != pr:
                assert not verify_inclusion(_POOL[n - 1], n - 1, n, rev, _POOL_ROOTS[n])

    def test_empty_proof_is_only_valid_for_a_singleton(self):
        for n in range(2, 41):
            for i in range(n):
                assert verify_inclusion(_POOL[i], i, n, [], _POOL_ROOTS[n]) is False
        assert verify_inclusion(_POOL[0], 0, 1, [], _POOL[0]) is True
        assert verify_inclusion(_POOL[0], 0, 1, [], _POOL_ROOTS[1]) is True
        assert verify_inclusion(_POOL[0], 0, 1, [_POOL[1]], _POOL_ROOTS[2]) is False
