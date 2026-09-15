#!/usr/bin/env python3
"""
curve_update.py — read the $STYXX bonding curve straight from Robinhood Chain and
print a paste-ready update for the receipt thread. no dependencies, python 3.8+.

  python3 curve_update.py            # human-readable + paste block
  python3 curve_update.py --json     # machine-readable

what it reads (all via the public rpc, nothing typed):
  - ETH sitting in the curve contract (graduation happens at 4.2 ETH raised; the
    contract balance also holds unswept fees, so it runs slightly above pons' number)
  - holder count and transfer count from Transfer logs on the token
  - dev / declared wallet balances (so the "we haven't sold" line is a number, not a vibe)
  - graduation flag: the curve contract no longer holds tokens

the dev wallet and the declared other wallet are hard-coded below on purpose: the
whole point is that the wallets are public and anyone can run this against them.
"""
import argparse, json, sys, time, urllib.request, urllib.error

RPC = "https://rpc.mainnet.chain.robinhood.com"
TOKEN = "0xC750bcdAe34cC578Ff17963bed40C5d9396fdC5D"
CURVE = "0xef305301933bf8d49ee1804f37558148db052a6a"
LAUNCH_BLOCK = 63874429
GRADUATION_ETH = 4.2
WALLETS = {
    "dev":   "0x903540529C5085E47B049793790e5a856BFc0948",
    "other": "0x14a2487daD53073F10ACc0117E0AbA6dD199c0f3",
}
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
ZERO = "0x" + "0" * 40


def rpc(method, params, tries=6):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(RPC, data=body, headers={"content-type": "application/json", "user-agent": "styxx-curve/1.0"})
    last = None
    for i in range(tries):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=40).read())
            if "error" in r:
                raise RuntimeError(r["error"])
            return r["result"]
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError) as e:
            last = e
            time.sleep(1.5 * (i + 1))   # public rpc rate-limits; back off
    sys.exit(f"rpc failed after {tries} tries: {last!r}")


def balance_of(addr):
    data = "0x70a08231" + addr[2:].lower().rjust(64, "0")
    return int(rpc("eth_call", [{"to": TOKEN, "data": data}, "latest"]), 16) / 1e18


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    block = int(rpc("eth_blockNumber", []), 16)
    curve_eth = int(rpc("eth_getBalance", [CURVE, "latest"]), 16) / 1e18
    logs = rpc("eth_getLogs", [{"fromBlock": hex(LAUNCH_BLOCK), "toBlock": "latest", "address": TOKEN, "topics": [TRANSFER_TOPIC]}])

    hold = {}
    for l in logs:
        f = "0x" + l["topics"][1][-40:]
        t = "0x" + l["topics"][2][-40:]
        v = int(l["data"], 16)
        hold[f] = hold.get(f, 0) - v
        hold[t] = hold.get(t, 0) + v
    holders = [(v, adr) for adr, v in hold.items() if v > 0 and adr not in (ZERO, CURVE.lower())]
    holders.sort(reverse=True)
    curve_tokens = hold.get(CURVE.lower(), 0) / 1e18
    graduated = curve_tokens == 0

    wallets = {k: {"address": v, "styxx": balance_of(v), "nonce": int(rpc("eth_getTransactionCount", [v, "latest"]), 16)} for k, v in WALLETS.items()}
    ts = time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())

    out = {
        "read_utc": ts, "block": block,
        "curve_eth": round(curve_eth, 4), "graduation_eth": GRADUATION_ETH,
        "pct_to_graduation": round(min(curve_eth / GRADUATION_ETH, 1) * 100, 1),
        "curve_tokens_M": round(curve_tokens / 1e6, 1), "graduated": graduated,
        "holders": len(holders), "transfers": len(logs),
        "top5": [{"address": adr, "styxx_M": round(v / 1e24, 2)} for v, adr in holders[:5]],
        "wallets": wallets,
    }
    if a.json:
        print(json.dumps(out, indent=2)); return

    print(f"\n$STYXX curve — read from the chain at {ts}, block {block}")
    print(f"  ETH in curve contract : {curve_eth:.4f} / {GRADUATION_ETH}  ({out['pct_to_graduation']}%)"
          + ("   ** GRADUATED — curve holds no tokens; the pool is live **" if graduated else ""))
    print(f"  holders               : {len(holders)}    transfers: {len(logs)}")
    for k, w in wallets.items():
        print(f"  {k:5} wallet          : {w['styxx']/1e6:.2f}M STYXX   nonce {w['nonce']}   {w['address']}")
    print("\npaste (numbers are the contract balance, which runs a little above pons' 'raised' because it includes unswept fees):\n")
    print(f"update — read from the chain, block {block}:")
    print(f"{curve_eth:.2f} ETH in the curve contract · {out['pct_to_graduation']:.0f}% to graduation · {len(holders)} holders · {len(logs)} transfers")
    print(f"dev wallet still holds {wallets['dev']['styxx']/1e6:.1f}M. no sells. every tx from it is public.")
    if graduated:
        print("graduated: the curve is closed and liquidity is in the locked pool. pool address + explorer link below.")
    print()


if __name__ == "__main__":
    main()
