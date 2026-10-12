# -*- coding: utf-8 -*-
"""draw — n accusations drawn by the hash of a Bitcoin block whose height the plan names in advance.

    python draw.py --params plan_params.json --accusations DIR/accusations.json \\
        --header <160 hex> --tip-height <int> [--coinbase-hex <hex> --merkle-branch <h1,h2,...>] \\
        [--anchor-time 2026-10-24T00:00:00Z]
    python draw.py --params plan_params.json --accusations DIR/accusations.json --fetch   # explorers

The arithmetic is styxx.beacon's, imported, not copied (`styxx.beacon.select`):

    seed_i = sha256( pool_sha256 || beacon_hex || str(i) ),  index = int(seed_i) mod len(pool)

with the accusation set as the pool (accuse.accusation_items: 3-tuples of strings, ordered by
event id), `pool_sha256` = styxx.beacon.pool_sha256 of that set (accuse.py's accusations_sha256)
and `beacon_hex` = the block's id as explorers print it (sha256d of the 80-byte header, byte
order reversed). Given the set and the block, the draw is fixed; the record names both digests,
the size, the height and the drawn ids, and the LF sha256 of styxx/beacon.py that drew them.

What draw refuses, each by name:

  HEIGHT_NOT_THE_PLANS     the height is not params.draw.bitcoin_height
  HEADER_INVALID           the header is not 80 bytes, does not hash to the block id given, or its
                           hash is above the target its own bits field sets (proof of work)
  HEIGHT_UNPROVEN          a coinbase and merkle branch were given and do not prove the height
                           (BIP34: the coinbase's script starts with the height; the branch takes
                           the coinbase's txid to the header's merkle root)
  TOO_FEW_CONFIRMATIONS    the chain tip is below height + min_confirmations - 1
  BLOCK_NOT_AFTER_WINDOW   the header's time is not after t1 (miners set it, within about two
                           hours; a plan names a height expected days after t1, so this only
                           catches a wrong height)
  SET_NOT_ANCHORED_BEFORE  an anchor time for the accusation set's digest was given and is not at
                           least two hours before the header's time
  SET_DIGEST_MISMATCH      the items do not hash to the accusation file's own digest
  SIZING_FAILURE           fewer accusations than n: no draw, published as a sizing failure

The order the plan needs is: the window closes (t1), the accusation set is derived and its digest
anchored, then the named block is mined. If the block came earlier, whoever derived the set could
have seen the hash before fixing the set; the anchor-time check exists for that. Without an anchor
time the record says the order was not checked.

The explorer fetch (`--fetch`) reads https://mempool.space/api and https://blockstream.info/api
(the Esplora API both serve), requires the two to agree on the block id at the height, and checks
the BIP34 height from the coinbase and its merkle branch. Those hosts are not in the network grant
of 2026-10-12; the fetch needs the operator's word before it runs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import struct
import sys
import urllib.request
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
DRAW_SCHEMA = "styxx.sealed-ahead/draw/v0"
EXPLORERS = ("https://mempool.space/api", "https://blockstream.info/api")
USER_AGENT = "styxx-sealed-ahead/0 (research; read-only)"
ANCHOR_SLACK = timedelta(hours=2)


def _sibling(name):
    key = "sealed_ahead_" + name
    mod = sys.modules.get(key)
    if mod is None:
        spec = importlib.util.spec_from_file_location(key, os.path.join(HERE, name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return mod


common = _sibling("common")


class Refused(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


def sha256d(b: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()


# ---------------------------------------------------------------- the block

def parse_header(header_hex: str) -> dict:
    raw = bytes.fromhex(header_hex.strip())
    if len(raw) != 80:
        raise Refused("HEADER_INVALID", f"a header is 80 bytes, this one is {len(raw)}")
    version, prev, merkle, t, bits, nonce = struct.unpack("<I32s32sIII", raw)
    block_id = sha256d(raw)[::-1].hex()
    exp, mant = bits >> 24, bits & 0xFFFFFF
    target = mant * (1 << (8 * (exp - 3))) if exp >= 3 else mant >> (8 * (3 - exp))
    return {"id": block_id, "version": version, "prev_id": prev[::-1].hex(), "merkle_root": merkle[::-1].hex(),
            "time": t, "bits": f"{bits:08x}", "nonce": nonce, "pow_ok": int(block_id, 16) <= target and target > 0}


def verify_header(header_hex: str, block_id: str | None = None) -> dict:
    h = parse_header(header_hex)
    if block_id is not None and h["id"] != block_id.strip().lower():
        raise Refused("HEADER_INVALID", f"the header hashes to {h['id']}, not {block_id}")
    if not h["pow_ok"]:
        raise Refused("HEADER_INVALID", "the header's hash is above the target its bits set")
    return h


def verify_chain(headers_hex: list) -> list:
    """Each header's prev field is the id of the one before it. Returns the ids."""
    ids, prev = [], None
    for hx in headers_hex:
        h = verify_header(hx)
        if prev is not None and h["prev_id"] != prev:
            raise Refused("HEADER_INVALID", f"{h['id']} does not follow {prev}")
        ids.append(h["id"])
        prev = h["id"]
    return ids


def _varint(b: bytes, i: int) -> tuple:
    v = b[i]
    if v < 0xFD:
        return v, i + 1
    size = {0xFD: 2, 0xFE: 4, 0xFF: 8}[v]
    return int.from_bytes(b[i + 1:i + 1 + size], "little"), i + 1 + size


def parse_coinbase(tx_hex: str) -> dict:
    """A coinbase transaction -> its txid (witness stripped) and the BIP34 height in its script."""
    b = bytes.fromhex(tx_hex.strip())
    i = 4
    segwit = b[4] == 0 and b[5] == 1
    if segwit:
        i = 6
    body_start = i
    n_in, i = _varint(b, i)
    if n_in != 1:
        raise Refused("HEIGHT_UNPROVEN", "a coinbase has exactly one input")
    prev_out = b[i:i + 36]
    if prev_out != b"\x00" * 32 + b"\xff\xff\xff\xff":
        raise Refused("HEIGHT_UNPROVEN", "the input does not spend the null outpoint: not a coinbase")
    i += 36
    slen, i = _varint(b, i)
    script = b[i:i + slen]
    i += slen + 4
    n_out, i = _varint(b, i)
    for _ in range(n_out):
        i += 8
        olen, i = _varint(b, i)
        i += olen
    body_end = i
    if segwit:
        for _ in range(n_in):
            items, i = _varint(b, i)
            for _ in range(items):
                ln, i = _varint(b, i)
                i += ln
    locktime = b[i:i + 4]
    if len(locktime) != 4 or i + 4 != len(b):
        raise Refused("HEIGHT_UNPROVEN", "the transaction does not parse to its end")
    stripped = b[:4] + b[body_start:body_end] + locktime
    if not script:
        raise Refused("HEIGHT_UNPROVEN", "empty coinbase script")
    op = script[0]
    if 1 <= op <= 8:
        height = int.from_bytes(script[1:1 + op], "little")
    elif 0x51 <= op <= 0x60:
        height = op - 0x50
    else:
        raise Refused("HEIGHT_UNPROVEN", "the coinbase script does not start with a height push (BIP34)")
    return {"txid": sha256d(stripped)[::-1].hex(), "height": height, "segwit": segwit}


def merkle_root_from_branch(txid: str, branch: list, pos: int = 0) -> str:
    h = bytes.fromhex(txid)[::-1]
    for sib in branch:
        s = bytes.fromhex(sib)[::-1]
        h = sha256d(s + h) if pos & 1 else sha256d(h + s)
        pos >>= 1
    return h[::-1].hex()


def verify_height(header_hex: str, coinbase_hex: str, branch: list, height: int) -> dict:
    h = parse_header(header_hex)
    cb = parse_coinbase(coinbase_hex)
    if cb["height"] != height:
        raise Refused("HEIGHT_UNPROVEN", f"the coinbase says height {cb['height']}, not {height}")
    root = merkle_root_from_branch(cb["txid"], branch, 0)
    if root != h["merkle_root"]:
        raise Refused("HEIGHT_UNPROVEN", "the branch does not take the coinbase to the header's merkle root")
    return {"height": height, "coinbase_txid": cb["txid"], "proof": "bip34+merkle"}


# ---------------------------------------------------------------- the draw

def draw(items: list, accusations_sha256: str, block_id: str, n: int) -> dict:
    """The drawn ids by styxx.beacon.select. Pure given styxx.beacon."""
    from styxx import beacon
    items = [tuple(i) for i in items]
    if beacon.pool_sha256(items) != accusations_sha256:
        raise Refused("SET_DIGEST_MISMATCH", "the items do not hash to the accusation set's digest")
    rec = {"schema": DRAW_SCHEMA, "pool_sha256": accusations_sha256, "pool_size": len(items),
           "beacon": block_id.strip().lower(), "n": n}
    with open(beacon.__file__, "rb") as fh:
        rec["beacon_module_lf_sha256"] = common.lf_sha256(fh.read())
    if len(items) < n:
        rec.update({"status": "SIZING_FAILURE", "drawn": [], "drawn_sha256": None})
        return rec
    chosen = beacon.select(block_id, n, items)
    ids = [c[0] for c in chosen]
    rec.update({"status": "DRAWN", "drawn": ids, "drawn_items": [list(c) for c in chosen],
                "drawn_sha256": common.canon_sha256(ids)})
    return rec


def checked_draw(params: dict, accusations: dict, header_hex: str, height: int, tip_height: int,
                 block_id: str | None = None, coinbase_hex: str | None = None, branch=None,
                 anchor_time: str | None = None) -> dict:
    """Every check the plan needs, then the draw. Raises Refused naming the failed check."""
    plan_height = params["draw"]["bitcoin_height"]
    if height != plan_height:
        raise Refused("HEIGHT_NOT_THE_PLANS", f"{height} is not the plan's height {plan_height}")
    h = verify_header(header_hex, block_id)
    proof = {"proof": "none: the height rests on the source of the header"}
    if coinbase_hex is not None:
        proof = verify_height(header_hex, coinbase_hex, branch or [], height)
    if tip_height < height + params["draw"]["min_confirmations"] - 1:
        raise Refused("TOO_FEW_CONFIRMATIONS", f"tip {tip_height} < {height} + "
                                               f"{params['draw']['min_confirmations']} - 1")
    t1 = common.parse_utc(params["t1"])
    block_time = datetime.fromtimestamp(h["time"], tz=common.UTC)
    if block_time <= t1:
        raise Refused("BLOCK_NOT_AFTER_WINDOW", f"block time {common.fmt_utc(block_time)} is not after t1 {params['t1']}")
    order = "not checked: no anchor time was given for the accusation set's digest"
    if anchor_time is not None:
        at = common.parse_utc(anchor_time)
        if at > block_time - ANCHOR_SLACK:
            raise Refused("SET_NOT_ANCHORED_BEFORE", f"anchor {anchor_time} is not two hours before the block's "
                                                     f"time {common.fmt_utc(block_time)}")
        order = f"anchored {anchor_time}, block {common.fmt_utc(block_time)}"
    rec = draw(accusations["items"], accusations["accusations_sha256"], h["id"], params["draw"]["n"])
    rec.update({"bitcoin_height": height, "block_time": common.fmt_utc(block_time), "height_proof": proof,
                "tip_height": tip_height, "order": order, "params_sha256": common.params_sha256(params),
                "accusations_status": accusations.get("status")})
    if accusations.get("status") != "COMPLETE":
        rec["warning"] = "the accusation set is not COMPLETE; this draw is not the plan's"
    return rec


# ---------------------------------------------------------------- explorers (network; the operator's word before use)

def _get(url: str, opener=None) -> str:
    opener = opener or urllib.request.urlopen
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with opener(req, timeout=30) as r:
        return r.read().decode("utf-8").strip()


def fetch_block(height: int, opener=None) -> dict:
    ids = {base: _get(f"{base}/block-height/{height}", opener) for base in EXPLORERS}
    if len(set(ids.values())) != 1:
        raise Refused("HEADER_INVALID", f"the explorers disagree on the block at {height}: {ids}")
    block_id = next(iter(ids.values()))
    base = EXPLORERS[0]
    header = _get(f"{base}/block/{block_id}/header", opener)
    if _get(f"{EXPLORERS[1]}/block/{block_id}/header", opener) != header:
        raise Refused("HEADER_INVALID", "the explorers serve different header bytes")
    txid = _get(f"{base}/block/{block_id}/txid/0", opener)
    coinbase = _get(f"{base}/tx/{txid}/hex", opener)
    proof = json.loads(_get(f"{base}/tx/{txid}/merkle-proof", opener))
    if proof.get("pos") != 0:
        raise Refused("HEIGHT_UNPROVEN", "the coinbase is not at position 0")
    tip = min(int(_get(f"{b}/blocks/tip/height", opener)) for b in EXPLORERS)
    return {"block_id": block_id, "header": header, "coinbase_hex": coinbase, "branch": proof["merkle"],
            "tip_height": tip}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--params", required=True)
    ap.add_argument("--accusations", required=True)
    ap.add_argument("--header")
    ap.add_argument("--block-id")
    ap.add_argument("--height", type=int)
    ap.add_argument("--tip-height", type=int)
    ap.add_argument("--coinbase-hex")
    ap.add_argument("--merkle-branch", default="")
    ap.add_argument("--anchor-time")
    ap.add_argument("--fetch", action="store_true", help="read the block from two explorers (needs the operator's word)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    params = common.load_params(a.params)
    with open(a.accusations, "rb") as fh:
        acc = json.loads(fh.read().decode("utf-8"))
    height = a.height if a.height is not None else params["draw"]["bitcoin_height"]
    try:
        if a.fetch:
            b = fetch_block(height)
            rec = checked_draw(params, acc, b["header"], height, b["tip_height"], b["block_id"], b["coinbase_hex"],
                               b["branch"], a.anchor_time)
        else:
            if not (a.header and a.tip_height is not None):
                ap.error("--header and --tip-height are required without --fetch")
            branch = [x for x in a.merkle_branch.split(",") if x]
            rec = checked_draw(params, acc, a.header, height, a.tip_height, a.block_id, a.coinbase_hex, branch,
                               a.anchor_time)
    except Refused as e:
        print(json.dumps({"status": "REFUSED", "code": e.code, "detail": e.detail}, indent=1))
        return 2
    text = json.dumps(rec, indent=1, ensure_ascii=False, sort_keys=True) + "\n"
    if a.out:
        with open(a.out, "wb") as fh:
            fh.write(text.encode("utf-8"))
    print(text)
    return 0 if rec["status"] == "DRAWN" else 1


if __name__ == "__main__":
    sys.exit(main())
