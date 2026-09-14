# -*- coding: utf-8 -*-
"""styxx.clock — the clock, checkable.

An anchor is a memo on a $STYXX transfer from the creator wallet to itself, carrying the digest
of a claim. This module builds the exact memo (so no anchor is ever mistyped) and re-verifies
every recorded anchor against the chain.

    python -m styxx.clock memo sworn-receipt <digest>          # prints the memo and the command
    python -m styxx.clock verify papers/charon/anchors.jsonl    # re-checks every line on chain

What verify() checks, per line, in this order — the first failure is the line's status, and the
line's `checks` map records every check that was computed, so a reader sees all of them:

    MALFORMED_LINE      the line lacks kind/digest/tx, the kind is unknown, the digest is not 64
                        lowercase hex, or the line's own recorded memo is not the one its kind and
                        digest imply
    RPC_ERROR           no endpoint answered getTransaction
    NOT_FOUND           the chain has no such transaction
    FAILED_TX           the transaction is on chain but failed (meta.err is not null)
    MEMO_MISMATCH       no memo instruction in the transaction equals the memo the line implies
    NOT_CREATOR         the fee payer (the first account key, always a signer) is not the creator
    NO_TRANSFER         the transaction carries no spl-token transfer of the $STYXX mint
    TIME_UNAVAILABLE    the endpoint returned no block time for the slot (never reported as 1970)
    SLOT_MISMATCH       the line records a slot and it is not the transaction's
    TIME_MISMATCH       the line records a block time and it is not the chain's
    BEACON_UNAVAILABLE  (seals only) getBlock failed or returned no blockhash
    BEACON_MALFORMED    (seals only) the blockhash did not decode to 32 bytes
    EARLIEST_UNKNOWN    (seals only) the wallet's history could not be scanned to its end, or lists
                        no confirmed memo carrying the digest though the transaction resolved
    EARLIER_MEMO_EXISTS (seals only) an earlier confirmed memo from the wallet carries the same
                        digest: the beacon is THAT slot's, and the recorded transaction is not the seal
    ANCHORED            everything above held

The earliest-memo rule (added 2026-09-13, night): a signer can anchor one digest several times and
record whichever transaction's slot drew the canaries they liked. The rule that closes that is
that the seal of a digest is the EARLIEST confirmed memo carrying it from the creator wallet, and
verify() enforces it by scanning the wallet's signature listing (each entry carries its memo, so
the scan costs one request per thousand transactions). A memo from another wallet is not a seal
(NOT_CREATOR) and is not in this wallet's listing; a later duplicate from the same wallet is
caught here. What remains: the scan trusts the same endpoint as everything else, and it stops at
fifty pages — a longer history reads EARLIEST_UNKNOWN, never ANCHORED.

What it does not check, stated so nobody reads more into ANCHORED than it says. It trusts the
first RPC endpoint that answers: two are tried, no cross-endpoint agreement is required, and the
answering endpoint is written into the line (`rpc`) so a reader can ask another one. A seal
bounds the digest's existence from above by the block time; it orders nothing else relative to
that time, and in particular it cannot show that a run started after the seal.

The beacon for a seal is the slot's blockhash. The chain returns it as base58; it is decoded to
its 32 bytes and reported as 64 lowercase hex in `beacon`, beside the base58 in `beacon_b58`,
because `styxx.beacon.select` takes hex. One decoding, written down here, so a stranger and the
lab draw the same canaries from the same block.

The log verifies without the chain. The chain only proves when. verify() never touches a
document; it only compares what the chain returns with what the file recorded.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

MINT = "93ihpGjLVnhghXciSeFovXwKF762rwcigW12kSQBpump"
CREATOR = "74gZN4yQjGCMpHqmWWSEYkNKwrb1TUybXWhAD6vMoMQ2"
KINDS = ("sworn-receipt", "sealed-prereg", "sealed-canaries", "charon-head")
SEAL_KINDS = ("sealed-prereg", "sealed-canaries")
MEMO_PROGRAMS = {"MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr", "Memo1UhkJRfHyvLMcVucJwxXeuD728EqVDDwQDxFMNo"}
TOKEN_PROGRAMS = {"TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA", "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"}
RPCS = ["https://api.mainnet-beta.solana.com", "https://solana-rpc.publicnode.com"]
_HEX = "0123456789abcdef"
_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def memo(kind: str, digest: str) -> str:
    """The exact memo text. The digest is taken as written: 64 lowercase hex, no folding, no trimming,
    so the memo on chain is byte-for-byte the digest a document names."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in _HEX for c in digest):
        raise ValueError("digest must be exactly 64 lowercase hex characters, as written "
                         "(no case folding, no surrounding whitespace)")
    return f"styxx {kind} {digest}"


def command(kind: str, digest: str, wallet: str = CREATOR) -> str:
    return f'spl-token transfer {MINT} 1 {wallet} --with-memo "{memo(kind, digest)}"'


def _b58decode(s: str) -> bytes:
    n = 0
    for ch in s:
        i = _B58.find(ch)
        if i < 0:
            raise ValueError(f"not base58: {ch!r}")
        n = n * 58 + i
    out = n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b""
    pad = len(s) - len(s.lstrip("1"))
    return b"\x00" * pad + out


def _b58encode(b: bytes) -> str:
    n = int.from_bytes(b, "big")
    s = ""
    while n:
        n, r = divmod(n, 58)
        s = _B58[r] + s
    pad = len(b) - len(b.lstrip(b"\x00"))
    return "1" * pad + s


def blockhash_to_beacon(blockhash_b58: str) -> str:
    """The chain's base58 blockhash as the 64-hex beacon `styxx.beacon.select` takes."""
    raw = _b58decode(blockhash_b58.strip())
    if len(raw) != 32:
        raise ValueError(f"a blockhash decodes to 32 bytes; this one decodes to {len(raw)}")
    return raw.hex()


def _instructions(tx: dict) -> list:
    msg = (tx.get("transaction") or {}).get("message") or {}
    meta = tx.get("meta") or {}
    inner = [i for g in (meta.get("innerInstructions") or []) for i in (g.get("instructions") or [])]
    return list(msg.get("instructions") or []) + inner


def parse_memos(tx: dict) -> list[str]:
    """Every memo text in a jsonParsed transaction, top-level and inner, in order. An instruction
    whose bytes do not decode is skipped, never allowed to hide the next one."""
    out: list[str] = []
    for ins in _instructions(tx):
        if ins.get("program") == "spl-memo" or ins.get("programId") in MEMO_PROGRAMS:
            parsed = ins.get("parsed")
            if isinstance(parsed, str):
                out.append(parsed)
                continue
            if isinstance(parsed, dict) and isinstance(parsed.get("info"), str):
                out.append(parsed["info"])
                continue
            data = ins.get("data")
            if isinstance(data, str):
                try:
                    out.append(_b58decode(data).decode("utf-8"))
                except Exception:
                    continue
    return out


def parse_memo(tx: dict) -> str | None:
    """The first memo text, or None. Kept for callers of v0; verification uses parse_memos."""
    ms = parse_memos(tx)
    return ms[0] if ms else None


def fee_payer(tx: dict) -> str | None:
    """The first account key of the message: on Solana it is always the fee payer and a signer."""
    keys = ((tx.get("transaction") or {}).get("message") or {}).get("accountKeys") or []
    if not keys:
        return None
    k = keys[0]
    return k.get("pubkey") if isinstance(k, dict) else k


def transfers_mint(tx: dict, mint: str = MINT, owner: str = CREATOR) -> bool:
    """True if the transaction carries an spl-token transfer of `mint`. `transferChecked` names the
    mint in its parsed info; a plain `transfer` does not, so for that shape the mint is read from
    the token-balance deltas the chain records for `owner`."""
    meta = tx.get("meta") or {}
    for ins in _instructions(tx):
        if ins.get("program") == "spl-token" or ins.get("programId") in TOKEN_PROGRAMS:
            parsed = ins.get("parsed")
            if not (isinstance(parsed, dict) and parsed.get("type") in ("transfer", "transferChecked")):
                continue
            info = parsed.get("info") or {}
            if info.get("mint") == mint:
                return True
            if "mint" not in info:
                for tb in list(meta.get("preTokenBalances") or []) + list(meta.get("postTokenBalances") or []):
                    if tb.get("mint") == mint and tb.get("owner") == owner:
                        return True
    return False


def _rpc(method: str, params: list, rpcs=RPCS, tries: int = 2):
    last = None
    for _ in range(tries):
        for u in rpcs:
            try:
                req = urllib.request.Request(u, data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": method,
                                                                  "params": params}).encode(),
                                             headers={"Content-Type": "application/json"})
                r = json.load(urllib.request.urlopen(req, timeout=30))
                if "result" in r:
                    _rpc.last_endpoint = u  # type: ignore[attr-defined]
                    return r["result"]
                last = r.get("error")
            except Exception as e:  # pragma: no cover
                last = str(e)
            time.sleep(0.4)
    raise RuntimeError(f"rpc {method} failed: {last}")


def wallet_memos(wallet: str = CREATOR, fetch=_rpc, rpcs=RPCS, limit: int = 1000, max_pages: int = 50) -> list[dict]:
    """Every signature the chain lists for the wallet, newest first, through getSignaturesForAddress
    pages — each entry carries the transaction's memo text, so no per-transaction fetch is needed.
    Stops at max_pages; a wallet with more history than that is reported as scanned in part."""
    out, before = [], None
    for _ in range(max_pages):
        opts = {"limit": limit}
        if before:
            opts["before"] = before
        page = fetch("getSignaturesForAddress", [wallet, opts], rpcs) or []
        out.extend(page)
        if len(page) < limit:
            return out
        before = page[-1]["signature"]
    out.append({"_truncated": True})
    return out


def earliest_memo(digest: str, wallet: str = CREATOR, fetch=_rpc, rpcs=RPCS) -> dict:
    """The EARLIEST confirmed signature of the wallet whose memo carries the digest — the rule that
    closes slot selection: a signer who anchors the same digest more than once and records the
    transaction whose slot drew the canaries they liked is caught, because the beacon is defined as
    the earliest one's slot. Returns {"earliest": entry-or-None, "n_carrying": count, "complete": bool}.
    Solana lists a memo as "[len] text"; the digest is searched as a substring."""
    entries = wallet_memos(wallet, fetch, rpcs)
    complete = not any(e.get("_truncated") for e in entries)
    hits = [e for e in entries if not e.get("_truncated") and e.get("err") is None
            and isinstance(e.get("memo"), str) and digest in e["memo"]]
    earliest = min(hits, key=lambda e: (e.get("slot") or 0, e["signature"])) if hits else None
    return {"earliest": earliest, "n_carrying": len(hits), "complete": complete}


def _iso(block_time) -> str | None:
    if isinstance(block_time, bool) or not isinstance(block_time, int):
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(block_time))


def check_line(line: dict, rpcs=RPCS, fetch=_rpc, wallet: str = CREATOR, mint: str = MINT, scan: bool = True) -> dict:
    """Re-verify one recorded anchor against the chain. Pure comparison; no document is read.
    The first failing check in the documented order is the status; `checks` holds all of them.
    With scan=True (the default) a seal is also checked against the wallet's history: the recorded
    transaction must be the EARLIEST confirmed memo carrying the digest, or the beacon belongs to
    another transaction (EARLIER_MEMO_EXISTS); a history the scan could not finish is
    EARLIEST_UNKNOWN, never a pass."""
    out: dict = {"n": line.get("n"), "kind": line.get("kind"), "digest": line.get("digest"), "tx": line.get("tx"),
                 "checks": {}}

    def fail(status: str, **more) -> dict:
        out.update(more)
        out["status"] = status
        return out

    for k in ("kind", "digest", "tx"):
        if not isinstance(line.get(k), str) or not line.get(k):
            return fail("MALFORMED_LINE", detail=f"missing or empty {k!r}")
    if line["kind"] not in KINDS:
        return fail("MALFORMED_LINE", detail=f"unknown kind {line['kind']!r}")
    try:
        expected = memo(line["kind"], line["digest"])
    except ValueError as e:
        return fail("MALFORMED_LINE", detail=str(e))
    out["memo_expected"] = expected
    if "memo" in line and line["memo"] != expected:
        return fail("MALFORMED_LINE", detail="the line's recorded memo is not the memo its kind and digest imply")

    try:
        tx = fetch("getTransaction", [line["tx"], {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}], rpcs)
    except RuntimeError as e:
        return fail("RPC_ERROR", detail=str(e))
    out["rpc"] = getattr(fetch, "last_endpoint", None)
    if tx is None:
        return fail("NOT_FOUND")

    meta = tx.get("meta") or {}
    out["slot"] = tx.get("slot")
    out["block_time"] = _iso(tx.get("blockTime"))
    memos = parse_memos(tx)
    out["memos_on_chain"] = memos
    out["memo_on_chain"] = expected if expected in memos else (memos[0] if memos else None)
    out["signer"] = fee_payer(tx)
    c = out["checks"]
    c["succeeded"] = meta.get("err") is None
    c["memo"] = expected in memos
    c["creator"] = out["signer"] == wallet
    c["transfer"] = transfers_mint(tx, mint, wallet)
    c["time"] = out["block_time"] is not None
    c["slot_recorded"] = ("slot" not in line) or line["slot"] == out["slot"]
    c["time_recorded"] = ("block_time" not in line) or line["block_time"] == out["block_time"]
    if not c["succeeded"]:
        return fail("FAILED_TX", detail=json.dumps(meta.get("err")))
    if not c["memo"]:
        return fail("MEMO_MISMATCH")
    if not c["creator"]:
        return fail("NOT_CREATOR", detail=f"fee payer {out['signer']} is not {wallet}")
    if not c["transfer"]:
        return fail("NO_TRANSFER", detail=f"no spl-token transfer of {mint} in the transaction")
    if not c["time"]:
        return fail("TIME_UNAVAILABLE", detail="the endpoint returned no blockTime; ask another")
    if not c["slot_recorded"]:
        return fail("SLOT_MISMATCH", detail=f"line records slot {line['slot']!r}, chain says {out['slot']!r}")
    if not c["time_recorded"]:
        return fail("TIME_MISMATCH", detail=f"line records {line['block_time']!r}, chain says {out['block_time']!r}")

    if line["kind"] in SEAL_KINDS:
        try:
            blk = fetch("getBlock", [tx["slot"], {"encoding": "json", "transactionDetails": "none", "rewards": False,
                                                  "maxSupportedTransactionVersion": 0}], rpcs)
        except RuntimeError as e:
            return fail("BEACON_UNAVAILABLE", detail=str(e))
        b58 = (blk or {}).get("blockhash") if isinstance(blk, dict) else None
        if not isinstance(b58, str) or not b58:
            return fail("BEACON_UNAVAILABLE", detail="getBlock returned no blockhash")
        out["beacon_b58"] = b58
        try:
            out["beacon"] = blockhash_to_beacon(b58)
        except ValueError as e:
            return fail("BEACON_MALFORMED", detail=str(e))
        c["beacon"] = True
        if scan:
            try:
                em = earliest_memo(line["digest"], wallet, fetch, rpcs)
            except RuntimeError as e:
                return fail("EARLIEST_UNKNOWN", detail=f"the wallet history could not be scanned: {e}")
            out["memos_carrying_digest"] = em["n_carrying"]
            out["earliest_tx"] = (em["earliest"] or {}).get("signature")
            out["earliest_slot"] = (em["earliest"] or {}).get("slot")
            c["earliest"] = em["earliest"] is not None and em["earliest"]["signature"] == line["tx"]
            if not em["complete"]:
                return fail("EARLIEST_UNKNOWN", detail="the wallet's history is longer than the scan read; the earliest memo is unknown")
            if em["earliest"] is None:
                return fail("EARLIEST_UNKNOWN", detail="the wallet's history lists no confirmed memo carrying this digest, "
                                                       "though the transaction resolved; the listing and the transaction disagree")
            if not c["earliest"]:
                return fail("EARLIER_MEMO_EXISTS", detail=f"the earliest confirmed memo carrying this digest is {out['earliest_tx']} "
                                                          f"at slot {out['earliest_slot']}; the beacon is that slot's, not this one's")
    out["status"] = "ANCHORED"
    return out


def verify(path: str, rpcs=RPCS, fetch=_rpc) -> list[dict]:
    """Every line of an anchors file, checked; a line that fails is a status, never an abort."""
    results = []
    for raw in open(path, encoding="utf-8"):
        raw = raw.strip()
        if raw:
            results.append(check_line(json.loads(raw), rpcs, fetch))
    return results


def main(argv=None) -> int:  # pragma: no cover
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 3 and argv[0] == "memo":
        print(memo(argv[1], argv[2])); print(command(argv[1], argv[2])); return 0
    if len(argv) >= 2 and argv[0] == "verify":
        rs = verify(argv[1])
        for r in rs:
            line = (f"#{r.get('n')} {str(r.get('kind')):14s} {r['status']:18s} {r.get('block_time') or '-':20s} "
                    f"slot={r.get('slot')} rpc={r.get('rpc')}")
            if r.get("beacon"):
                line += f" beacon={r['beacon']} (b58 {r.get('beacon_b58')})"
            if r.get("detail"):
                line += f"  — {r['detail']}"
            print(line)
        return 0 if rs and all(r["status"] == "ANCHORED" for r in rs) else 1
    print(__doc__); return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
