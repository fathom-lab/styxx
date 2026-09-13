# -*- coding: utf-8 -*-
"""styxx.epoch — private canaries, committed now, revealed later.

Beacon canaries stop anyone preparing for a set in advance; they do not stop a provider from
reading the pool. An epoch closes that too: the lab draws a private item set for the epoch,
publishes only its commitment — sha256(salt || canonical items) — and anchors that commitment.
Every fingerprint taken in the epoch carries canary_sha256(items). At the end of the epoch the
lab reveals salt and items; anyone checks that the commitment matches, that the certs' canary
hash matches the revealed items, and re-runs whatever they like. A commitment that never gets
revealed is a broken promise visible on chain.

    c = epoch.commit(items, salt)                      -> {"commitment", "n", "canary_sha256"}
    epoch.reveal_ok(c["commitment"], items, salt)      -> True/False
"""
from __future__ import annotations

import hashlib
import json
import secrets
from typing import Sequence

from .checksum import canary_sha256


def new_salt() -> str:
    return secrets.token_hex(32)


def _canonical(items: Sequence[tuple[str, str, str]]) -> bytes:
    return json.dumps([list(c) for c in items], ensure_ascii=False, separators=(",", ":")).encode()


def commit(items: Sequence[tuple[str, str, str]], salt: str) -> dict:
    if len(salt) < 32:
        raise ValueError("salt must be at least 32 hex characters")
    commitment = hashlib.sha256(bytes.fromhex(salt) + _canonical(items)).hexdigest()
    return {"commitment": commitment, "n": len(items), "canary_sha256": canary_sha256(items)}


def reveal_ok(commitment: str, items: Sequence[tuple[str, str, str]], salt: str) -> bool:
    return commit(items, salt)["commitment"] == commitment.strip().lower()
