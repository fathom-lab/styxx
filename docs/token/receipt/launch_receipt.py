#!/usr/bin/env python3
"""
styxx launch receipt — sworn record of the $STYXX launch on Robinhood Chain.

Reads the launch straight from the chain (no trust in this script's inputs),
writes launch_receipt.json, prints its sha256, and prints the calldata you
paste into a 0-ETH self-transfer to anchor the hash on-chain.

usage:
  python3 launch_receipt.py --token 0x... --launch-tx 0x... [--dev-buy-tx 0x...] \
      [--creator-wallet 0x...] [--creator-tax-bps 0] [--holder-fee-sharing yes] \
      [--logo ../logo/styxx_token_512.png] [--rpc https://rpc.mainnet.chain.robinhood.com]

verify (anyone, any machine — this is the point):
  python3 launch_receipt.py verify launch_receipt.json [--anchor-tx 0x...]
      re-reads every chain fact in the file from the chain and diffs it; recomputes the
      file's sha256; if --anchor-tx is given, checks that tx is a 0-ETH self-send from the
      receipt's creator wallet whose calldata is exactly that sha256. exit 0 = everything
      holds, exit 1 = something doesn't, and it tells you what.

no dependencies beyond python 3.8+.
"""
import argparse, json, hashlib, sys, time, urllib.request, urllib.error, datetime as dt

RPC_DEFAULT = "https://rpc.mainnet.chain.robinhood.com"
CHAIN_ID = 4663
PONS_V2 = {
    "factory":              "0x7eD598BcEf8bd9Edd8C97A195C6d13f40801EC7e",
    "launch_and_buy_router":"0xe33E9E479dF8802cb0866d5d05258bEc4cF62948",
    "launch_deployer":      "0x3711ceA4feaDE896C913C68F01Eda97Cb06D1A42",
    "fee_escrow":           "0xd3AFEB2a57f70eF218Aa82451c51B2fb0416Ac9e",
    "launch_locker":        "0x267444D099b10fB5Ed7c3Cc7B7c767AdcA574952",
    "weth":                 "0x0Bd7D308f8E1639FAb988df18A8011f41EAcAD73",
}
# 4-byte selectors (keccak256 of the signature), precomputed so there is no keccak dependency
SEL = {"name()": "06fdde03", "symbol()": "95d89b41", "totalSupply()": "18160ddd", "decimals()": "313ce567"}

def rpc(url, method, params, _tries=6):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json", "user-agent": "styxx-receipt/1.0"})
    for i in range(_tries):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=40).read())
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and i < _tries - 1:
                time.sleep(1.5 * (i + 1)); continue   # public rpc rate-limits; back off and retry
            raise
    if "error" in r:
        raise RuntimeError(f"{method}: {r['error']}")
    time.sleep(0.25)
    return r["result"]

def call(url, to, sel):
    return rpc(url, "eth_call", [{"to": to, "data": "0x" + sel}, "latest"])

def abi_string(hexdata):
    b = bytes.fromhex(hexdata[2:])
    if len(b) < 64: return b.rstrip(b"\x00").decode(errors="replace")
    off = int.from_bytes(b[0:32], "big"); ln = int.from_bytes(b[off:off+32], "big")
    return b[off+32:off+32+ln].decode(errors="replace")

def hexint(h): return int(h, 16)

def tx_record(url, h):
    tx = rpc(url, "eth_getTransactionByHash", [h])
    rc = rpc(url, "eth_getTransactionReceipt", [h])
    if not tx or not rc: raise RuntimeError(f"tx not found on chain: {h}")
    blk = rpc(url, "eth_getBlockByNumber", [rc["blockNumber"], False])
    return {
        "hash": h,
        "from": tx["from"], "to": tx["to"],
        "value_wei": str(hexint(tx["value"])), "value_eth": hexint(tx["value"]) / 1e18,
        "block": hexint(rc["blockNumber"]),
        "block_timestamp_utc": dt.datetime.utcfromtimestamp(hexint(blk["timestamp"])).isoformat() + "Z",
        "status": "success" if hexint(rc["status"]) == 1 else "FAILED",
        "log_count": len(rc["logs"]),
    }

def verify(argv):
    ap = argparse.ArgumentParser(prog="launch_receipt.py verify")
    ap.add_argument("receipt", help="launch_receipt.json to check")
    ap.add_argument("--anchor-tx", default=None, help="hash of the 0-ETH self-send that anchors the file's sha256")
    ap.add_argument("--rpc", default=None, help="override the rpc recorded in the receipt")
    a = ap.parse_args(argv)

    with open(a.receipt, "rb") as f: body = f.read()
    digest = hashlib.sha256(body).hexdigest()
    r = json.loads(body)
    url = a.rpc or r["chain"]["rpc"]
    checks = []   # (ok, label, expected, actual)
    def chk(label, expected, actual):
        checks.append((str(expected).lower() == str(actual).lower(), label, expected, actual))

    cid = hexint(rpc(url, "eth_chainId", []))
    chk("chain id", r["chain"]["chain_id"], cid)

    t = r["token"]
    chk("token name",     t["name"],             abi_string(call(url, t["address"], SEL["name()"])))
    chk("token symbol",   t["symbol"],           abi_string(call(url, t["address"], SEL["symbol()"])))
    chk("token decimals", t["decimals"],         hexint(call(url, t["address"], SEL["decimals()"])))
    chk("total supply",   t["total_supply_raw"], hexint(call(url, t["address"], SEL["totalSupply()"])))

    for key in ("launch_tx", "dev_buy_tx"):
        rec = r.get(key)
        if not isinstance(rec, dict): continue
        live = tx_record(url, rec["hash"])
        for f_ in ("from", "to", "value_wei", "block", "block_timestamp_utc", "status", "log_count"):
            chk(f"{key}.{f_}", rec[f_], live[f_])

    d = r["declared"]
    chk("dev wallet == launch tx sender", d["dev_wallet"], r["launch_tx"]["from"])

    anchor = None
    if a.anchor_tx:
        tx = rpc(url, "eth_getTransactionByHash", [a.anchor_tx])
        rc = rpc(url, "eth_getTransactionReceipt", [a.anchor_tx])
        if not tx or not rc:
            checks.append((False, "anchor tx", "found on chain", "NOT FOUND"))
        else:
            blk = rpc(url, "eth_getBlockByNumber", [rc["blockNumber"], False])
            anchor = {"block": hexint(rc["blockNumber"]),
                      "timestamp_utc": dt.datetime.utcfromtimestamp(hexint(blk["timestamp"])).isoformat() + "Z"}
            chk("anchor tx status", "success", "success" if hexint(rc["status"]) == 1 else "FAILED")
            chk("anchor tx from == creator wallet", d["creator_wallet"], tx["from"])
            chk("anchor tx to == from (self-send)", tx["from"], tx["to"])
            chk("anchor tx value", 0, hexint(tx["value"]))
            chk("anchor tx calldata == sha256(file)", "0x" + digest, tx["input"])
            chk("anchor is after launch", True, anchor["block"] > r["launch_tx"]["block"])

    width = max(len(c[1]) for c in checks) + 2
    print("\n" + "=" * 72 + f"\nSTYXX RECEIPT VERIFY   {a.receipt}\n" + "=" * 72)
    print(f"sha256(file)   {digest}")
    if anchor: print(f"anchor         block {anchor['block']}  {anchor['timestamp_utc']}")
    print("-" * 72)
    bad = 0
    for ok, label, exp, act in checks:
        mark = "ok  " if ok else "FAIL"
        line = f"{mark} {label.ljust(width)}"
        if not ok:
            bad += 1
            line += f" expected {exp!r}  got {act!r}"
        print(line)
    print("-" * 72)
    if bad:
        print(f"{bad} check(s) FAILED. the file and the chain disagree — do not trust the receipt.")
        print("=" * 72 + "\n"); sys.exit(1)
    print(f"all {len(checks)} checks hold." + ("" if anchor else "  (no --anchor-tx given: the hash is verified, its on-chain anchor is not.)"))
    print("=" * 72 + "\n")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        return verify(sys.argv[2:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True)
    ap.add_argument("--launch-tx", required=True)
    ap.add_argument("--dev-buy-tx", default=None, help="omit if the dev buy was inside the launch tx")
    ap.add_argument("--creator-wallet", default=None)
    ap.add_argument("--team-wallet", action="append", default=[], help="any other wallet you or your team bought from. repeatable. every one goes in the receipt.")
    ap.add_argument("--creator-tax-bps", type=int, default=0)
    ap.add_argument("--holder-fee-sharing", default="yes")
    ap.add_argument("--logo", default=None)
    ap.add_argument("--rpc", default=RPC_DEFAULT)
    ap.add_argument("--out", default="launch_receipt.json")
    a = ap.parse_args()

    url = a.rpc
    cid = hexint(rpc(url, "eth_chainId", []))
    if cid != CHAIN_ID:
        sys.exit(f"wrong chain: rpc reports chainId {cid}, expected {CHAIN_ID} (Robinhood Chain)")

    code = rpc(url, "eth_getCode", [a.token, "latest"])
    if code in ("0x", "0x0"): sys.exit(f"no contract at {a.token} — check the address")

    token = {
        "address": a.token,
        "name": abi_string(call(url, a.token, SEL["name()"])),
        "symbol": abi_string(call(url, a.token, SEL["symbol()"])),
        "decimals": hexint(call(url, a.token, SEL["decimals()"])),
        "total_supply_raw": str(hexint(call(url, a.token, SEL["totalSupply()"]))),
    }
    token["total_supply"] = int(token["total_supply_raw"]) / (10 ** token["decimals"])

    launch = tx_record(url, a.launch_tx)
    devbuy = tx_record(url, a.dev_buy_tx) if a.dev_buy_tx else None

    logo = None
    if a.logo:
        with open(a.logo, "rb") as f: logo = {"file": a.logo.split("/")[-1], "sha256": hashlib.sha256(f.read()).hexdigest()}

    receipt = {
        "receipt_type": "styxx.launch.v1",
        "statement": "This is the complete record of the $STYXX launch on Robinhood Chain. "
                     "It was read from the chain, not typed. Anyone can regenerate it with launch_receipt.py and compare the hash.",
        "generated_utc": dt.datetime.utcnow().isoformat() + "Z",
        "chain": {"name": "Robinhood Chain", "chain_id": CHAIN_ID, "rpc": url, "explorer": "https://robinhoodchain.blockscout.com"},
        "launchpad": {"name": "pons v2", "site": "https://www.ponsfamily.com", "contracts": PONS_V2},
        "token": token,
        "launch_tx": launch,
        "dev_buy_tx": devbuy if devbuy else "included in launch_tx",
        "declared": {
            "creator_wallet": a.creator_wallet or launch["from"],
            "dev_wallet": launch["from"],
            "team_wallets": a.team_wallet,
            "wallets_statement": "The dev wallet and every team wallet that holds or bought $STYXX are listed here. There are no others.",
            "creator_tax_bps": a.creator_tax_bps,
            "holder_fee_sharing": a.holder_fee_sharing,
            "fixed_supply": True, "mint_function": False, "freeze_or_blacklist": False,
            "liquidity": "locked at graduation (pons v2)",
        },
        "logo": logo,
        "prior_tokens_disclosure": "The Solana $STYXX mint 93ihpGjLVnhghXciSeFovXwKF762rwcigW12kSQBpump is discontinued. This token is separate. There is no migration and no claim.",
    }
    body = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    with open(a.out, "wb") as f: f.write(body)
    digest = hashlib.sha256(body).hexdigest()   # == `sha256sum launch_receipt.json`

    print("\n" + "=" * 72)
    print("STYXX LAUNCH RECEIPT")
    print("=" * 72)
    print(f"token      {token['name']} (${token['symbol']})  {token['address']}")
    print(f"supply     {token['total_supply']:,.0f}")
    print(f"launch tx  {launch['hash']}  block {launch['block']}  {launch['block_timestamp_utc']}  {launch['status']}")
    if devbuy: print(f"dev buy    {devbuy['hash']}  {devbuy['value_eth']:.4f} ETH  {devbuy['status']}")
    else:      print(f"dev buy    inside launch tx, value {launch['value_eth']:.4f} ETH (includes 0.0005 launch fee)")
    print(f"creator    {receipt['declared']['creator_wallet']}   tax {a.creator_tax_bps} bps   holder fee sharing: {a.holder_fee_sharing}")
    print(f"wallets    dev {launch['from']}" + (f"   team {', '.join(a.team_wallet)}" if a.team_wallet else "   (no other wallets)"))
    if logo: print(f"logo       {logo['file']}  sha256 {logo['sha256']}")
    print(f"\nwrote      {a.out}")
    print(f"sha256     {digest}")
    print("\nANCHOR IT: send 0 ETH from the creator wallet to itself on Robinhood Chain with this hex data:")
    print(f"           0x{digest}")
    print("           (metamask: settings → advanced → 'show hex data' → paste into the data field)")
    print("\nPOST IT:   receipt sha256 " + digest[:16] + "…  full file + anchor tx in the pinned post.")
    print("=" * 72 + "\n")

if __name__ == "__main__":
    main()
