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

from tpc_const import BEACON_HEIGHT, UTC, USER_AGENT, iso, parse_utc

EXPLORERS = ("https://mempool.space/api", "https://blockstream.info/api")
# a dry run's stated stand-in (tpc_packet), never a beacon: every check below refuses it
STAND_IN_HEX = hashlib.sha256(b"tests-pass-ci/stand-in: the hash of a dry run, never a beacon").hexdigest()


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


def named_problems(b: Optional[dict]) -> list:
    """Why a beacon record may not draw rows or shuffle P1 and P2 (section 3.1); [] when it may: a Bitcoin block at the
    named height 975000, or a higher one under the late rule (named in the rule text, with the time the pull request
    opened and a header time at least six hours after it), read from both explorers, which agree, each header hashing
    to the block and meeting its own target, with six confirmations; never the dry run's stand-in hash."""
    if not b:
        return ["no beacon: block 975000 is not read yet (OD-8)"]
    probs = []
    bh = b.get("beacon_hex") or ""
    if b.get("kind") != "bitcoin":
        probs.append(f"the beacon is a {b.get('kind')!r} record, not a Bitcoin block")
    if bh == STAND_IN_HEX:
        probs.append("the beacon is the dry run's stand-in hash")
    if len(bh) != 64 or any(c not in "0123456789abcdef" for c in bh):
        probs.append("beacon_hex is not 64 lowercase hex characters")
    h = b.get("height")
    rule = (b.get("rule") or "").lower()
    if h != BEACON_HEIGHT:
        if not (isinstance(h, int) and h > BEACON_HEIGHT and "late rule" in rule and b.get("pr_opened_at")):
            probs.append(f"height {h!r}: the named height is {BEACON_HEIGHT}; another height needs the late rule, "
                         "named in the rule text with the time the pull request opened (pr_opened_at)")
        else:
            try:
                if parse_utc(b.get("header_time") or "") < parse_utc(b["pr_opened_at"]) + dt.timedelta(hours=6):
                    probs.append("under the late rule the block's header time is at least six hours after the pull "
                                 "request opened")
            except ValueError:
                probs.append("header_time or pr_opened_at unreadable")
    answers = b.get("explorers") or []
    seen = {a.get("explorer") for a in answers}
    if not set(EXPLORERS) <= seen:
        probs.append(f"the block is read from both explorers {EXPLORERS}; this record names {sorted(seen, key=str)}")
    for a in answers:
        if a.get("block_id") != bh:
            probs.append(f"{a.get('explorer')} answers {a.get('block_id')}, not {bh}")
            continue
        try:
            hd = parse_header(a.get("header") or "")
        except (ValueError, BeaconError) as e:
            probs.append(f"{a.get('explorer')}: header unreadable ({e})")
            continue
        if hd["id"] != bh or not hd["pow_ok"]:
            probs.append(f"{a.get('explorer')}: the header does not hash to the block or fails its target")
        if isinstance(h, int) and (a.get("tip_height") or 0) < h + 5:
            probs.append(f"{a.get('explorer')}: fewer than 6 confirmations")
    return probs


def dumps(beacon: dict) -> bytes:
    return (json.dumps(beacon, indent=1) + "\n").encode("utf-8")
