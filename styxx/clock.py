# -*- coding: utf-8 -*-
"""styxx.clock — the clock, checkable.

An anchor is a memo on a $STYXX transfer carrying the digest of a claim. This module does two
things: builds the exact memo (so no anchor is ever mistyped), and re-verifies every recorded
anchor against the chain: the memo in the confirmed transaction equals the recorded memo, the
digest is inside it, the block time is read from the chain, and — for a seal — the block hash of
the slot is returned as the beacon for `styxx.beacon`.

    python -m styxx.clock memo sworn-receipt <digest>          # prints the memo and the command
    python -m styxx.clock verify papers/charon/anchors.jsonl    # re-checks every line on chain

The log verifies without the chain. The chain only proves when. Both sentences are enforced here:
verify() never touches a document; it only compares strings the chain returns with strings the
file recorded.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

MINT = "93ihpGjLVnhghXciSeFovXwKF762rwcigW12kSQBpump"
CREATOR = "74gZN4yQjGCMpHqmWWSEYkNKwrb1TUybXWhAD6vMoMQ2"
KINDS = ("sworn-receipt", "sealed-prereg", "sealed-canaries", "charon-head")
MEMO_PROGRAMS = {"MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr", "Memo1UhkJRfHyvLMcVucJwxXeuD728EqVDDwQDxFMNo"}
RPCS = ["https://api.mainnet-beta.solana.com", "https://solana-rpc.publicnode.com"]


def memo(kind: str, digest: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    digest = digest.strip().lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("digest must be 64 hex characters")
    return f"styxx {kind} {digest}"


def command(kind: str, digest: str, wallet: str = CREATOR) -> str:
    return f'spl-token transfer {MINT} 1 {wallet} --with-memo "{memo(kind, digest)}"'


def parse_memo(tx: dict) -> str | None:
    """The memo text of a jsonParsed transaction, or None."""
    msg = tx.get("transaction", {}).get("message", {})
    inner = [i for g in (tx.get("meta", {}) or {}).get("innerInstructions", []) or [] for i in g.get("instructions", [])]
    for ins in list(msg.get("instructions", [])) + inner:
        if ins.get("program") == "spl-memo" or ins.get("programId") in MEMO_PROGRAMS:
            parsed = ins.get("parsed")
            if isinstance(parsed, str):
                return parsed
            if isinstance(parsed, dict) and isinstance(parsed.get("info"), str):
                return parsed["info"]
            data = ins.get("data")
            if isinstance(data, str):
                try:
                    return _b58decode(data).decode("utf-8")
                except Exception:
                    return None
    return None


def _b58decode(s: str) -> bytes:
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    n = 0
    for ch in s:
        n = n * 58 + alphabet.index(ch)
    out = n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b""
    pad = len(s) - len(s.lstrip("1"))
    return b"\x00" * pad + out


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
                    return r["result"]
                last = r.get("error")
            except Exception as e:  # pragma: no cover
                last = str(e)
            time.sleep(0.4)
    raise RuntimeError(f"rpc {method} failed: {last}")


def check_line(line: dict, rpcs=RPCS, fetch=_rpc) -> dict:
    """Re-verify one recorded anchor against the chain. Pure comparison; no document is read."""
    tx = fetch("getTransaction", [line["tx"], {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}], rpcs)
    if tx is None:
        return {**line, "status": "NOT_FOUND"}
    on_chain = parse_memo(tx)
    expected = memo(line["kind"], line["digest"])
    ok = on_chain == expected == line.get("memo", expected)
    out = {"n": line.get("n"), "kind": line["kind"], "digest": line["digest"], "tx": line["tx"],
           "slot": tx.get("slot"),
           "block_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(tx.get("blockTime") or 0)),
           "memo_on_chain": on_chain, "status": "ANCHORED" if ok else "MEMO_MISMATCH"}
    if line["kind"] == "sealed-prereg" and ok:
        blk = fetch("getBlock", [tx["slot"], {"encoding": "json", "transactionDetails": "none", "rewards": False,
                                              "maxSupportedTransactionVersion": 0}], rpcs)
        out["beacon"] = blk.get("blockhash") if blk else None
    return out


def verify(path: str, rpcs=RPCS, fetch=_rpc) -> list[dict]:
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
            print(f"#{r.get('n')} {r['kind']:14s} {r['status']:14s} {r.get('block_time', '')} slot={r.get('slot')} "
                  f"{('beacon=' + str(r.get('beacon'))) if r.get('beacon') else ''}")
        return 0 if rs and all(r["status"] == "ANCHORED" for r in rs) else 1
    print(__doc__); return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
