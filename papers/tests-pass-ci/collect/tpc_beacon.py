# -*- coding: utf-8 -*-
"""The beacon (PREREG section 3.1, P3.1): the hash of a Bitcoin block named before it exists.

`read_beacon(height)` reads the block id at that height and its 80-byte header from two Esplora
explorers (mempool.space and blockstream.info), requires them to agree, checks that the header
hashes to the id and meets its own proof-of-work target, and returns the record that
beacon.json holds. It runs only with the operator's word (OD-8): the CLI asks for --go. The
late rule (the lowest height whose header time is at least six hours after the pull request
carrying the PREREG opened) is the lead's to apply; this module checks whatever height it is
given and records the rule text the caller states.

A development beacon (`dev_beacon`) is a sha256 of a label, marked kind "dev"; the official
store refuses it.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import struct
import urllib.request
from typing import Callable, Optional

from tpc_const import UTC, USER_AGENT, iso

EXPLORERS = ("https://mempool.space/api", "https://blockstream.info/api")


class BeaconError(Exception):
    pass


def sha256d(b: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()


def parse_header(header_hex: str) -> dict:
    raw = bytes.fromhex(header_hex.strip())
    if len(raw) != 80:
        raise BeaconError(f"a header is 80 bytes, this one is {len(raw)}")
    version, prev, merkle, t, bits, nonce = struct.unpack("<I32s32sIII", raw)
    block_id = sha256d(raw)[::-1].hex()
    exp, mant = bits >> 24, bits & 0xFFFFFF
    target = mant * (1 << (8 * (exp - 3))) if exp >= 3 else mant >> (8 * (3 - exp))
    return {"id": block_id, "prev_id": prev[::-1].hex(), "merkle_root": merkle[::-1].hex(), "time": t,
            "time_iso": iso(dt.datetime.fromtimestamp(t, UTC)), "bits": f"{bits:08x}", "nonce": nonce,
            "pow_ok": target > 0 and int(block_id, 16) <= target}


def _get(url: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8").strip()


def read_beacon(height: int, rule: str, *, get: Callable[[str], str] = _get,
                explorers: tuple = EXPLORERS, min_confirmations: int = 6) -> dict:
    answers = []
    for base in explorers:
        bid = get(f"{base}/block-height/{height}").lower()
        header = get(f"{base}/block/{bid}/header").lower()
        tip = int(get(f"{base}/blocks/tip/height"))
        answers.append({"explorer": base, "block_id": bid, "header": header, "tip_height": tip})
    ids = {a["block_id"] for a in answers}
    if len(ids) != 1:
        raise BeaconError(f"explorers disagree at height {height}: {sorted(ids)}")
    bid = ids.pop()
    for a in answers:
        h = parse_header(a["header"])
        if h["id"] != bid:
            raise BeaconError(f"{a['explorer']}'s header hashes to {h['id']}, not {bid}")
        if not h["pow_ok"]:
            raise BeaconError("the header's hash is above its own target")
        if a["tip_height"] < height + min_confirmations - 1:
            raise BeaconError(f"{a['explorer']} tip {a['tip_height']}: fewer than {min_confirmations} confirmations")
    h = parse_header(answers[0]["header"])
    return {"kind": "bitcoin", "height": height, "beacon_hex": bid, "header": answers[0]["header"],
            "header_time": h["time_iso"], "rule": rule, "explorers": answers,
            "read_at": iso(dt.datetime.now(UTC))}


def dev_beacon(label: str) -> dict:
    return {"kind": "dev", "label": label,
            "beacon_hex": hashlib.sha256(("tests-pass-ci/dev-beacon/" + label).encode("utf-8")).hexdigest(),
            "rule": "development only: a sha256 of a label, not a block", "read_at": iso(dt.datetime.now(UTC))}


def validate(beacon: dict, *, official: bool) -> Optional[str]:
    """None when usable, else why not."""
    bh = beacon.get("beacon_hex") or ""
    if len(bh) != 64 or any(c not in "0123456789abcdef" for c in bh):
        return "beacon_hex is not 64 lowercase hex characters"
    if official and beacon.get("kind") != "bitcoin":
        return "the window store takes a Bitcoin beacon only"
    if official:
        h = parse_header(beacon["header"])
        if h["id"] != bh or not h["pow_ok"]:
            return "the stored header does not hash to the beacon or fails proof of work"
    return None


def dumps(beacon: dict) -> bytes:
    return (json.dumps(beacon, indent=1) + "\n").encode("utf-8")
