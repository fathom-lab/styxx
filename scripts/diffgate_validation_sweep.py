# -*- coding: utf-8 -*-
"""Re-runnable validation for diffgate's headline claim.

`action.yml` tells every prospective user:

    "Validated on real agent-authored PRs with zero false accusations before
     release (see the styxx CHANGELOG for the receipts)."

The receipts are real — 7.29.1 swept 80 commits, 7.29.2 swept 24 agent-authored
public PRs, and both found and fixed false-accusation classes before claiming
zero. But that was **15 releases ago**, the sweep was run ad hoc, and no harness
was committed. An unreproducible receipt is a claim, not evidence, and the
product states it in the present tense.

So: this script. It sweeps two corpora and prints the numbers that back (or
retract) the sentence on the product page.

    local    this repository's own recent commits: message vs its own diff
    market   real public PRs whose body carries the Claude Code marker,
             gated against the diff GitHub serves for them

**A CONTRADICTED verdict is not automatically a false accusation, and not
automatically a caught lie.** Every one is printed in full with its evidence for
adjudication. The headline number is only claimable after each has been read.

Usage:
    python scripts/diffgate_validation_sweep.py            # local only
    python scripts/diffgate_validation_sweep.py --market   # adds the PR sweep
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from styxx import __version__
from styxx.diffgate import gate_diff_text

REPO = Path(__file__).resolve().parent.parent
MARKER = "Generated with [Claude Code]"


def token() -> str | None:
    p = Path("C:/Users/heyzo/clawd/secrets/fathomlab-github.txt")
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return os.environ.get("GH_TOKEN")


def api(url: str, accept: str = "application/vnd.github+json"):
    hdr = {"Accept": accept, "User-Agent": "styxx-diffgate-sweep",
           "X-GitHub-Api-Version": "2022-11-28"}
    t = token()
    if t:
        hdr["Authorization"] = f"Bearer {t}"
    req = urllib.request.Request(url, headers=hdr)
    with urllib.request.urlopen(req, timeout=45) as r:
        raw = r.read().decode("utf-8", errors="replace")
    return raw if accept.endswith("diff") else json.loads(raw)


# ── corpus 1: this repository's own commits ────────────────────────────────

def sweep_local(n: int):
    shas = subprocess.run(["git", "log", "-n", str(n), "--format=%H"], cwd=REPO,
                          capture_output=True, text=True).stdout.split()
    rows = []
    for sha in shas:
        msg = subprocess.run(["git", "log", "-1", "--format=%B", sha], cwd=REPO,
                             capture_output=True, text=True,
                             encoding="utf-8", errors="replace").stdout
        diff = subprocess.run(["git", "show", "--format=", sha], cwd=REPO,
                              capture_output=True, text=True,
                              encoding="utf-8", errors="replace").stdout
        if not msg.strip() or not diff.strip():
            continue
        g = gate_diff_text(msg, diff)
        rows.append((f"{sha[:9]}", g, msg.splitlines()[0][:60]))
    return rows


# ── corpus 2: real agent-authored public PRs ───────────────────────────────

def sweep_market(want: int):
    q = urllib.parse.quote(f'"{MARKER}" in:body is:pr is:public')
    rows, page = [], 1
    while len(rows) < want and page <= 5:
        try:
            res = api(f"https://api.github.com/search/issues?q={q}"
                      f"&per_page=30&page={page}")
        except urllib.error.HTTPError as e:
            print(f"  search failed ({e}) — market corpus unavailable", file=sys.stderr)
            return rows
        items = res.get("items") or []
        if not items:
            break
        for it in items:
            if len(rows) >= want:
                break
            body = it.get("body") or ""
            pr_url = (it.get("pull_request") or {}).get("url")
            if not pr_url or not body.strip():
                continue
            try:
                diff = api(pr_url, "application/vnd.github.diff")
            except Exception:
                continue
            g = gate_diff_text(body, diff)
            rows.append((it.get("html_url", "?"), g, (it.get("title") or "")[:60]))
        page += 1
    return rows


def report(name: str, rows) -> Counter:
    c: Counter = Counter()
    contradicted = []
    for ident, g, title in rows:
        c["prs" if name == "market" else "commits"] += 1
        if not g.measured:
            c["unmeasured"] += 1
            continue
        if g.claims:
            c["claim_bearing"] += 1
        for cl in g.claims:
            c[cl.verdict] += 1
            if cl.verdict == "CONTRADICTED":
                contradicted.append((ident, title, cl))
    total_claims = c["VERIFIED"] + c["CONTRADICTED"] + c["UNCHECKABLE"]
    print(f"\n== {name} corpus")
    print(f"   items {c['prs'] + c['commits']}, claim-bearing {c['claim_bearing']}, "
          f"unmeasured {c['unmeasured']}")
    print(f"   claims {total_claims}  |  VERIFIED {c['VERIFIED']}  "
          f"CONTRADICTED {c['CONTRADICTED']}  UNCHECKABLE {c['UNCHECKABLE']}")
    if contradicted:
        print(f"\n   {len(contradicted)} CONTRADICTED — EACH NEEDS ADJUDICATION.")
        print("   A contradiction on real data is either a caught lie or a false")
        print("   accusation, and the headline number depends on which:")
        for ident, title, cl in contradicted:
            print(f"     {ident}")
            print(f"       title : {title}")
            print(f"       claim : {cl.text[:100]}")
            print(f"       why   : {cl.why[:110]}")
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", action="store_true", help="also sweep public agent PRs")
    ap.add_argument("--commits", type=int, default=120)
    ap.add_argument("--prs", type=int, default=30)
    a = ap.parse_args()

    print(f"diffgate validation sweep — styxx {__version__}")
    print(f"re-running the claim printed in action.yml, on the CURRENT version")

    total = report("local", sweep_local(a.commits))
    if a.market:
        total += report("market", sweep_market(a.prs))

    print("\n" + "=" * 66)
    n_contra = total["CONTRADICTED"]
    n_claims = total["VERIFIED"] + n_contra + total["UNCHECKABLE"]
    print(f"  {n_claims} claims across both corpora, {n_contra} contradicted")
    if n_contra == 0:
        print("  No contradiction, therefore no false accusation, on this sweep.")
        print("  NOTE: zero contradictions is also what a gate that extracts")
        print("  nothing produces. Coverage is the number that keeps this honest:")
        print(f"    claim-bearing items: {total['claim_bearing']}")
    else:
        print("  The zero-false-accusation claim is NOT re-established until every")
        print("  contradiction above is adjudicated and found to be a real lie.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
