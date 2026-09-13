# -*- coding: utf-8 -*-
"""styxx.beacon — the draw is deterministic, duplicate-free, beacon-sensitive, and language-free."""
from __future__ import annotations

import hashlib

import pytest

from styxx import beacon

B1 = "a" * 64
B2 = "a" * 63 + "b"


def test_pool_is_large_and_ids_are_unique():
    assert len(beacon.POOL) > 600
    assert len({c[0] for c in beacon.POOL}) == len(beacon.POOL)


def test_draw_is_deterministic_and_duplicate_free():
    a = beacon.select(B1, 48)
    b = beacon.select(B1, 48)
    assert a == b
    assert len({c[0] for c in a}) == 48


def test_one_changed_beacon_character_changes_the_draw():
    a = {c[0] for c in beacon.select(B1, 48)}
    b = {c[0] for c in beacon.select(B2, 48)}
    assert len(a & b) < 48


def test_draw_is_plain_sha256_arithmetic_a_stranger_can_replay():
    # replay the spec by hand, without importing select()
    pool = beacon.POOL
    head = (beacon.pool_sha256() + B1).encode()
    chosen, seen, i = [], set(), 0
    while len(chosen) < 5:
        idx = int(hashlib.sha256(head + str(i).encode()).hexdigest(), 16) % len(pool)
        if idx not in seen:
            seen.add(idx); chosen.append(pool[idx][0])
        i += 1
    assert chosen == [c[0] for c in beacon.select(B1, 5)]


def test_non_hex_beacon_is_refused():
    with pytest.raises(ValueError):
        beacon.select("not-a-hash", 5)
