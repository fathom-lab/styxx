"""SWALLOW-11's receipt with each change counted once -- the correction the RESULT states.

The population rule kept SWALLOW-10's pull requests that SWALLOW-9 did not hold (by head), but did
not deduplicate within SWALLOW-10, and two of its people's pull requests -- ruvnet/ruv-FANN #44
and #48 -- are one change: one base, one head, the same six checks. This reads the same receipt,
keeps the first of every (repository, base, head), and scores P1-P8 by the scorer's definitions.
Exploratory: `swallow11_scored.json` is the rule as frozen, and stays as scored.

    python papers/harness/swallow11_once.py     # reads swallow11_receipt.json(.gz); writes swallow11_once.json
"""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECEIPT, RECEIPT_GZ, OUT = HERE / "swallow11_receipt.json", HERE / "swallow11_receipt.json.gz", HERE / "swallow11_once.json"
AGENTS = ("OpenAI_Codex", "Copilot", "Devin", "Cursor", "Claude_Code", "Google_Jules")


def once(pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    """The pairs with each (repository, base, head) kept once, and what was repeated."""
    seen: dict = {}
    kept, repeated = [], []
    for p in pairs:
        k = (p["repo"], p.get("base"), p["head"])
        if k in seen:
            repeated.append({"repo": p["repo"], "numbers": [seen[k].get("number"), p.get("number")], "source": p["source"], "group": p["group"],
                             "base": str(p.get("base"))[:12], "head": p["head"][:12], "checks": len(p.get("checks") or [])})
            continue
        seen[k] = p
        kept.append(p)
    return kept, repeated


def score(pairs: list[dict]) -> dict:
    ok = [p for p in pairs if "error" not in p]
    checks = [dict(c, group=p["group"], source=p["source"], repo=p["repo"]) for p in ok for c in p["checks"]]

    def share(sel) -> list[int]:
        xs = [c for c in checks if sel(c)]
        return [sum(1 for c in xs if c["one_click"]), len(xs)]

    located = [c for c in checks if c["located"]]
    rebuilt = [c for c in checks if c["rebuilt"]]
    oc = [c for c in checks if c["one_click"]]
    b, a = share(lambda c: c["kind"] == "born hidden"), share(lambda c: c["kind"] == "acquired")
    prs, main = share(lambda c: c["source"] in ("swallow9", "swallow10")), share(lambda c: c["source"] == "swallow7")
    people, agents = share(lambda c: c["group"] == "human"), share(lambda c: c["group"] in AGENTS)
    small = sum(1 for c in oc if c.get("size", 99) <= 3)
    vis = sum(1 for c in located if c["visible"])
    P = {"P1": {"hit": len(located) >= 0.98 * len(checks), "located": [len(located), len(checks)], "target": dict(Counter(c.get("target_what") for c in located))},
         "P2": {"hit": vis >= 0.80 * len(located), "visible": [vis, len(located)]},
         "P3": {"hit": len(oc) >= 0.60 * len(checks), "one_click": [len(oc), len(checks)], "why_not": dict(Counter(c.get("why_not") for c in checks if not c["one_click"]))},
         "P4": {"hit": sum(1 for c in rebuilt if c["one_click"]) >= 0.80 * len(rebuilt), "one_click_of_rebuilt": [sum(1 for c in rebuilt if c["one_click"]), len(rebuilt)],
                "reproduced": [sum(1 for c in rebuilt if c["reproduces"]), len(rebuilt)]},
         "P5": {"hit": small >= 0.75 * len(oc), "at_most_3": [small, len(oc)], "one_line": sum(1 for c in oc if c.get("size") == 1)},
         "P6": {"hit": b[0] * a[1] > a[0] * b[1], "born_hidden": b, "acquired": a},
         "P7": {"hit": prs[0] * main[1] >= main[0] * prs[1], "pull_requests": prs, "mainline": main},
         "P8": {"hit": people[0] * agents[1] >= agents[0] * people[1], "people": people, "agents": agents, "agent_signed": share(lambda c: c["group"] == "agent-signed")}}
    return {"pairs": len(pairs), "re_read": len(ok), "reproduced": sum(1 for p in ok if p.get("reproduced")), "checks": len(checks),
            "repos": len({p["repo"] for p in ok}), "repos_with_a_one_click_fix": len({c["repo"] for c in oc}), "predictions": P,
            "hits": sum(1 for v in P.values() if v["hit"]),
            "by_repair": {rp: share(lambda c, rp=rp: c.get("repair") == rp) for rp in sorted({str(c.get("repair")) for c in checks if c.get("repair")})},
            "sizes": dict(sorted(Counter(c.get("size") for c in oc).items()))}


def main() -> int:
    raw = RECEIPT.read_bytes() if RECEIPT.exists() else gzip.decompress(RECEIPT_GZ.read_bytes())
    r = json.loads(raw.decode("utf-8"))
    kept, repeated = once(r["pairs"])
    out = {"receipt_sha256": hashlib.sha256(raw).hexdigest(), "rule": "each (repository, base, head) once", "repeated": repeated,
           "as_frozen": score(r["pairs"]), "each_change_once": score(kept)}
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    f, o = out["as_frozen"], out["each_change_once"]
    print(f"repeated: {repeated}")
    print(f"as frozen: {f['pairs']} pairs, {f['checks']} checks, {f['hits']}/8 HIT; each change once: {o['pairs']} changes, {o['checks']} checks, {o['hits']}/8 HIT")
    for k in o["predictions"]:
        print(f"  {k}: {'HIT' if o['predictions'][k]['hit'] else 'MISS'} (as frozen {'HIT' if f['predictions'][k]['hit'] else 'MISS'})  "
              f"{json.dumps({x: v for x, v in o['predictions'][k].items() if x != 'hit'})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
