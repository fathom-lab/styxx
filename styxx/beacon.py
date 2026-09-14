# -*- coding: utf-8 -*-
"""styxx.beacon — canaries nobody can study for.

The weakness DUE_DILIGENCE_2026_09_13 §3 named first: a public 48-item canary set can be tuned
for. This module removes the advance knowledge. The canary set is drawn from a large committed
POOL by a public random value — a Solana block hash — that did not exist when anyone could have
prepared for it:

    seed_i = sha256( pool_sha256 || beacon_hex || i )      i = 0, 1, 2, ...
    index  = int(seed_i) mod len(pool), duplicates skipped, until n items

The draw is pure sha256 arithmetic, so a browser, a shell script or a stranger's notebook
reproduces it exactly; no language's PRNG is involved. `draw()` returns the items AND a record
(`styxx.beacon/draw/v0`: pool_sha256, pool_size, beacon, n, canary_sha256 of the drawn items);
`checksum.Fingerprint.draw` carries that record, `checksum.cert()` digests it (schema
`compare/v2`), `checksum.distance()` refuses two fingerprints under different draws, and the
observatory writes it into every chained line. A record is never trusted: `checksum.check_draw_record`
re-runs the beacon against the committed pool and refuses a record whose beacon does not produce
exactly the items it claims (a record whose canary hash matched but whose beacon lied was accepted
for a few hours on 2026-09-13). `run_deploy_quant.py --beacon` draws from this module; the
deploy-scale PREREG of 2026-09-13 froze the hand-written 48 by design, so a beacon-drawn run must
carry `--tag` and is never that experiment.

Which beacon: the block hash of the slot in which the sealing transaction confirmed
(`anchors.jsonl` → `slot` → RPC `getBlock(slot).blockhash`), decoded from the chain's base58 to
its 32 bytes and written as 64 lowercase hex — `styxx.clock.blockhash_to_beacon` is the one
conversion, and `select` refuses anything that is not 64 hex characters. Not the transaction
signature — a signer can grind signatures by varying the memo or blockhash until a favourable one
appears; a block hash is produced by the network after the transaction is out of the signer's
hands. What a block hash does not remove: the signer chooses when to submit and may submit more
than once, recording the transaction whose slot drew the canaries it liked. The rule that closes
that — the beacon is the earliest confirmed memo carrying the digest from the creator wallet —
is not implemented by `styxx.clock.verify` and is owed.

The pool is 48 hand-written items plus template items with known answers. Template items are
deliberately dull: the point is fixedness and breadth, not difficulty.
"""
from __future__ import annotations

import hashlib
import json
from typing import Sequence

from .checksum import CANARIES as _HAND

_CAPITALS = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Germany", "Berlin"),
             ("Egypt", "Cairo"), ("Canada", "Ottawa"), ("Australia", "Canberra"), ("Brazil", "Brasília"), ("Peru", "Lima"),
             ("Kenya", "Nairobi"), ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Greece", "Athens"), ("Turkey", "Ankara"),
             ("Poland", "Warsaw"), ("Portugal", "Lisbon"), ("Ireland", "Dublin"), ("Austria", "Vienna"), ("Hungary", "Budapest"),
             ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"), ("Chile", "Santiago"), ("Argentina", "Buenos Aires"),
             ("Cuba", "Havana"), ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Nepal", "Kathmandu"), ("Denmark", "Copenhagen"),
             ("Finland", "Helsinki")]
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]
_OPPOSITES = [("hot", "cold"), ("up", "down"), ("big", "small"), ("fast", "slow"), ("light", "dark"), ("wet", "dry"),
              ("open", "closed"), ("early", "late"), ("full", "empty"), ("hard", "soft"), ("loud", "quiet"),
              ("rich", "poor"), ("thick", "thin"), ("young", "old"), ("high", "low"), ("near", "far"),
              ("push", "pull"), ("buy", "sell"), ("win", "lose"), ("begin", "end")]
_WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


def build_pool() -> list[tuple[str, str, str]]:
    pool: list[tuple[str, str, str]] = list(_HAND)
    for a in range(1, 21):
        for b in range(1, 21):
            pool.append((f"add-{a}-{b}", f"{a} plus {b} equals", f" {a + b}"))
    for a in range(2, 13):
        for b in range(2, 13):
            pool.append((f"mul-{a}-{b}", f"{a} times {b} is", f" {a * b}"))
    for a in range(5, 31):
        for b in range(1, 5):
            pool.append((f"sub-{a}-{b}", f"{a} minus {b} is", f" {a - b}"))
    for c, cap in _CAPITALS:
        pool.append((f"cap-{c.lower()}", f"The capital of {c} is", f" {cap}"))
    for i, d in enumerate(_DAYS):
        pool.append((f"day-next-{d.lower()}", f"The day after {d} is", f" {_DAYS[(i + 1) % 7]}"))
        pool.append((f"day-prev-{d.lower()}", f"The day before {d} is", f" {_DAYS[(i - 1) % 7]}"))
    for i, m in enumerate(_MONTHS):
        pool.append((f"month-next-{m.lower()}", f"The month after {m} is", f" {_MONTHS[(i + 1) % 12]}"))
    for a, b in _OPPOSITES:
        pool.append((f"opp-{a}", f"The opposite of {a} is", f" {b}"))
        pool.append((f"opp-{b}", f"The opposite of {b} is", f" {a}"))
    for i in range(len(_WORDS) - 3):
        pool.append((f"seq-{_WORDS[i]}", f"{_WORDS[i]}, {_WORDS[i + 1]}, {_WORDS[i + 2]},", f" {_WORDS[i + 3]}"))
    ids = [p[0] for p in pool]
    assert len(ids) == len(set(ids)), "duplicate canary ids in the pool"
    return pool


POOL: list[tuple[str, str, str]] = build_pool()


def pool_sha256(pool: Sequence[tuple[str, str, str]] = POOL) -> str:
    blob = json.dumps([list(c) for c in pool], ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def select(beacon_hex: str, n: int = 48, pool: Sequence[tuple[str, str, str]] = POOL) -> list[tuple[str, str, str]]:
    """Draw n distinct items from the pool by a public beacon. Pure sha256; no PRNG."""
    beacon_hex = beacon_hex.strip().lower()
    if len(beacon_hex) != 64 or any(ch not in "0123456789abcdef" for ch in beacon_hex):
        raise ValueError("beacon must be 64 hex characters: a 32-byte block hash as hex "
                         "(styxx.clock.blockhash_to_beacon converts the chain's base58)")
    if n > len(pool):
        raise ValueError("n exceeds the pool")
    head = (pool_sha256(pool) + beacon_hex).encode()
    chosen, seen, i = [], set(), 0
    while len(chosen) < n:
        h = hashlib.sha256(head + str(i).encode()).hexdigest()
        idx = int(h, 16) % len(pool)
        if idx not in seen:
            seen.add(idx)
            chosen.append(pool[idx])
        i += 1
    return chosen


def draw(beacon_hex: str, n: int = 48, pool: Sequence[tuple[str, str, str]] = POOL) -> tuple[list, dict]:
    """The draw AND its record: the items, and the dict a fingerprint and a cert carry so a stranger
    can re-derive which canaries were used from the pool hash and the beacon alone. The record's
    canary_sha256 is the hash of the drawn items, which is what `checksum.fingerprint` stamps; the
    two must agree, and `fingerprint` refuses a draw record that does not match its items."""
    from .checksum import canary_sha256
    items = select(beacon_hex, n, pool)
    record = {"schema": "styxx.beacon/draw/v0", "pool_sha256": pool_sha256(pool), "pool_size": len(pool),
              "beacon": beacon_hex.strip().lower(), "n": n, "canary_sha256": canary_sha256(items)}
    return items, record


def describe(beacon_hex: str, n: int = 48) -> dict:
    items, record = draw(beacon_hex, n)
    return {**record, "ids": [c[0] for c in items]}


if __name__ == "__main__":  # pragma: no cover
    import sys
    print(json.dumps(describe(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 48), indent=1))
