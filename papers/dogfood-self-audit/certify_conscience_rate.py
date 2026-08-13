"""Dogfood the falsifiability receipt on a number this lab actually publishes.

The conscience fire rate is a headline instrument reading: what fraction of an agent's
drafts trip the register gates. It has been quoted in findings, and `MEMORY.md` already
carries one correction to a rate of exactly this kind -- the 'sensitivity 1.0' strike,
where a gate that fired on 6/6 attacks AND 6/6 benign texts had its two numbers counted
once and named twice.

That correction was found by argument. This asks the apparatus directly: while computing
this rate over this population, could the gates behind it have decided otherwise?

Population: darkflobi's own logged drafts, which is the traffic the rate is meant to
describe. Not a synthetic battery -- a synthetic population would let the receipt
certify a number nobody reports.

    python certify_conscience_rate.py --log <turn log jsonl> --json receipt.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from falsifiability_receipt import install_for, scope        # noqa: E402


def load_drafts(path, limit=400):
    """Real drafts, in the order they were produced."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            d = rec.get("draft") or rec.get("shipped") or rec.get("text")
            if isinstance(d, str) and d.strip():
                out.append(d)
            if len(out) >= limit:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--json")
    ap.add_argument("--limit", type=int, default=400)
    a = ap.parse_args()

    # Instrumentation must be installed BEFORE styxx is imported, or the gates run
    # uninstrumented and the receipt certifies a path it never observed -- which would
    # produce a confident REFUSED__nothing_adjudicable and look like a finding.
    install_for(["styxx"])
    from styxx.conscience import review                      # noqa: PLC0415

    drafts = load_drafts(a.log, a.limit)
    if not drafts:
        print(f"no drafts in {a.log} — refusing to certify an empty population")
        return 2

    with scope("conscience_fire_rate") as sc:
        fired = 0
        for d in drafts:
            v = review(d)
            if bool(getattr(v, "fired", False)
                    or getattr(v, "needs_revision", False)):
                fired += 1
        rate = fired / len(drafts)

    rec = sc.receipt(value=round(rate, 4))
    rec["population"] = {"source": os.path.basename(a.log), "n_drafts": len(drafts),
                         "n_fired": fired}

    print(f"  conscience fire rate : {rate:.4f}  ({fired}/{len(drafts)})")
    print(f"  terms on path        : {rec['n_terms_on_path']}")
    print(f"  live / constant      : {rec['n_live']} / {rec['n_constant']}"
          f"   (underpowered {rec['n_underpowered']})")
    print(f"  VERDICT              : {rec['verdict']}")
    print(f"  {rec['why']}")
    if rec["pinned_terms"]:
        print("\n  pinned throughout this measurement:")
        for p in rec["pinned_terms"][:10]:
            print(f"    {p['verdict']:14s} n={p['n']:5d} {p['module']}:{p['func']} "
                  f"L{p['line']} [{p['op']}] {str(p['src'])[:52]}")
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
