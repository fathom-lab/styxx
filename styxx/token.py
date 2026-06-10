"""styxx.token — read-only $STYXX holder-tier lookup.

Maps a Solana wallet's $STYXX balance to a utility tier (see ``docs/token/utility-tiers.md``).

**READ-ONLY.** This module queries a *public* Solana RPC for a token balance and returns a tier. It
holds no keys, signs nothing, sends no transactions, and moves no funds. Tiers are **hold-based** —
you are never asked to spend, lock, or burn anything; the lookup is a point-in-time balance read.

    from styxx.token import get_tier, tier_for_balance
    tier_for_balance(120_000).name      # 'Validator'   (pure, offline)
    get_tier("<wallet-address>").name   # live balance -> tier via public RPC

    # CLI:  python -m styxx.token <wallet-address>

The token NEVER gates the open-source library — it coordinates the *public* network only. Pure stdlib;
no third-party dependencies.
"""
from __future__ import annotations

import json
import urllib.request
from typing import NamedTuple

STYXX_MINT = "Dxw3u4KxN32KpSdHSq4TkwjfMPJTPeosa22JXN15pump"
DEFAULT_RPC = "https://api.mainnet-beta.solana.com"


class Tier(NamedTuple):
    """A $STYXX utility tier. ``balance >= min_styxx`` grants it (hold-based, not spend-based)."""
    level: int
    name: str
    min_styxx: int
    unlocks: str


# Hold-based tiers — docs/token/utility-tiers.md (denominated in $STYXX units, not USD).
TIERS = (
    Tier(0, "Public",    0,          "styxx.verify() OSS library + hosted free tier (rate-limited)"),
    Tier(1, "Supporter", 10_000,     "priority verify queue, supporter listing"),
    Tier(2, "Validator", 100_000,    "validator node eligibility, co-sign provenance certs"),
    Tier(3, "Governor",  1_000_000,  "weighted vote on atlas / benchmark parameters"),
    Tier(4, "Core",      10_000_000, "veto-capable governance, co-maintainer eligibility"),
)


def tier_for_balance(balance: float) -> Tier:
    """Return the highest tier whose threshold ``balance`` meets. Pure — no network, fully testable."""
    out = TIERS[0]
    for t in TIERS:
        if balance >= t.min_styxx:
            out = t
    return out


def get_balance(wallet: str, *, mint: str = STYXX_MINT, rpc_url: str = DEFAULT_RPC,
                timeout: float = 10.0) -> float:
    """Read-only: sum the wallet's $STYXX balance across its token accounts via public Solana RPC.

    Uses ``getTokenAccountsByOwner`` (jsonParsed). No keys, no signing, no transactions. Raises on
    network / RPC error — callers decide how to handle it (e.g. treat a failed lookup as Tier 0).
    """
    body = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "getTokenAccountsByOwner",
        "params": [wallet, {"mint": mint}, {"encoding": "jsonParsed"}],
    }).encode("utf-8")
    req = urllib.request.Request(rpc_url, data=body, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if "error" in data:
        raise RuntimeError(f"RPC error: {data['error']}")
    accounts = (data.get("result") or {}).get("value") or []
    total = 0.0
    for acc in accounts:
        amt = acc["account"]["data"]["parsed"]["info"]["tokenAmount"]
        total += float(amt.get("uiAmount") or 0.0)
    return total


def get_tier(wallet: str, *, mint: str = STYXX_MINT, rpc_url: str = DEFAULT_RPC) -> Tier:
    """Read-only live tier lookup: public-RPC balance -> Tier. No keys, no spend, no lock."""
    return tier_for_balance(get_balance(wallet, mint=mint, rpc_url=rpc_url))


def _main(argv=None) -> int:
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m styxx.token <wallet-address>")
        return 2
    try:
        t = get_tier(argv[0])
    except Exception as e:  # read-only lookup failed -> default to Public, say so
        print(f"lookup failed ({type(e).__name__}: {e}); defaulting to Tier 0 (Public)")
        t = TIERS[0]
    print(f"tier {t.level} - {t.name}: {t.unlocks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
