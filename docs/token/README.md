# $STYXX Token — Overview

$STYXX is the unit that styxx receipts and bounties settle in. Bounties for reproduction
failures are paid in $STYXX; receipt hashes are anchored on-chain from the wallet that
launched it. It is not required to use the library. It is not equity and not a promise of
anything.

## The contract (the only one)

| | |
|---|---|
| **Chain** | Robinhood Chain (chain id `4663`) |
| **Contract** | `0xC750bcdAe34cC578Ff17963bed40C5d9396fdC5D` |
| **Launched** | 2026-09-15 18:40:03 UTC, block 63874429, via pons v2 |
| **Launch tx** | `0x4996f313ef099e5297c767a3a816db49b231756b3faea67e3662696a521b5ce2` |
| **Anchor tx** | `0x4c5d0b1fc4240fa8640b6c4354b98bfcc2298f4fc6f5c9e95f1e8146dd3fc82a` — 0 ETH, dev wallet to itself, block 63935251, 2026-09-15 20:25:10 UTC; calldata = sha256 of the receipt file |
| **Supply** | fixed 1,000,000,000 — no mint function, no freeze, no blacklist |
| **Creator tax** | 0 |
| **Creator fees** | routed to holders pro-rata through the pons distributor. There is no creator claim. |
| **Liquidity** | locks in a Uniswap v4 pool at graduation (4.2 ETH raised on the curve) |
| **Dev wallet** | `0x903540529C5085E47B049793790e5a856BFc0948` — 0.07 ETH dev buy in the launch tx |
| **Other wallet** | `0x14a2487daD53073F10ACc0117E0AbA6dD199c0f3` — the only other wallet we hold in |
| **Trade** | https://www.ponsfamily.com/launchpad/0xC750bcdAe34cC578Ff17963bed40C5d9396fdC5D |
| **Explorer** | https://robinhoodchain.blockscout.com/token/0xC750bcdAe34cC578Ff17963bed40C5d9396fdC5D |

Names and symbols can be copied. The address cannot. Anything else called styxx is not us.

## The launch receipt

Every number above was read from the chain, not typed, and the whole record is in
[`receipt/launch_receipt.json`](receipt/launch_receipt.json) — sha256
`5ff50c30f02c8122cf69c867a66e2355b21e0c346973ebaa5cb2e175af7bd12a`. Regenerate it yourself:

```
python3 receipt/launch_receipt.py \
  --token 0xC750bcdAe34cC578Ff17963bed40C5d9396fdC5D \
  --launch-tx 0x4996f313ef099e5297c767a3a816db49b231756b3faea67e3662696a521b5ce2 \
  --team-wallet 0x14a2487daD53073F10ACc0117E0AbA6dD199c0f3 \
  --creator-tax-bps 0 --holder-fee-sharing yes
sha256sum launch_receipt.json
```

No dependencies. It talks to the public RPC and prints the hash. If your hash differs from
ours, one of us is wrong, and it is checkable which.

Or don't regenerate — audit the file we published against the chain directly:

```
python3 receipt/launch_receipt.py verify receipt/launch_receipt.json
```

That re-reads every chain fact in the file (token name, symbol, supply, launch tx sender,
value, block, timestamp, status) and diffs it line by line. Exit 0 means the file and the
chain agree; exit 1 prints exactly which line doesn't. Add the anchor and it also checks
that tx is a 0-ETH self-send from the creator wallet whose calldata is exactly the file's
sha256:

```
python3 receipt/launch_receipt.py verify receipt/launch_receipt.json \
  --anchor-tx 0x4c5d0b1fc4240fa8640b6c4354b98bfcc2298f4fc6f5c9e95f1e8146dd3fc82a
# all 19 checks hold.
```

Live curve numbers, the same way (no dashboard, no trust):

```
python3 receipt/curve_update.py
```

ETH in the curve contract vs the 4.2 ETH graduation line, holder and transfer counts from
Transfer logs, and the dev / declared wallet balances — printed as a paste-ready update.

## Prior mints — discontinued

Two earlier $STYXX tokens were launched on Solana via pump.fun. Both are discontinued.
There is no migration and no claim from either.

| | mint | status |
|---|---|---|
| v1 (Apr 2026) | `Dxw3u4KxN32KpSdHSq4TkwjfMPJTPeosa22JXN15pump` | retired |
| v2 (Aug 2026) | `93ihpGjLVnhghXciSeFovXwKF762rwcigW12kSQBpump` | discontinued 2026-09-15 |

## What it is not

- Not a security. Not an investment contract. Not a promise of yield.
- Not required to use `styxx`. The library is and will remain open source, MIT.
- Not a governance token for anything that doesn't yet exist.

## Files in this folder

- [`receipt/`](receipt/) — the launch receipt and the script that regenerates it.
- [`economic-architecture.md`](economic-architecture.md) — design intent (written for the
  Solana era; the contract facts there are superseded by this file).
- [`utility-tiers.md`](utility-tiers.md), [`validator-spec.md`](validator-spec.md) — design
  documents, not deployed mechanics.

## For enterprise readers

The styxx library is usable without ever touching $STYXX. The token coordinates the
*public* side — bounties and anchored receipts. It does not gate the technology.
