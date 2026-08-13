"""Point the falsifiability receipt at the gate shipped hours earlier the same day.

`execution_receipt_gate.py` was extended on 2026-08-13 after it returned zero claims on
a status report containing a fabricated work item. Its validation set went from four
cases to six and all six pass. That establishes it CAN fire and CAN stay quiet on
hand-built examples.

It does not establish that its decision path is live on real traffic, which is a
different question and the one that actually matters: a gate can pass a validation set
built from four imagined phrasings and still, on the messages it will really see, run
entirely through terms that never vary. That was the original defect -- the gate was
validated against phrasings its author could think of.

So this asks the apparatus instead of the author. Population: darkflobi's own logged
drafts.

    python certify_receipt_gate.py --log <turn log jsonl> --json receipt.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CLAWD_SCRIPTS = r"C:\Users\heyzo\clawd\scripts"
CLAWD_REPO = r"C:\Users\heyzo\clawd"
for p in (HERE, CLAWD_SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from certify_conscience_rate import load_drafts             # noqa: E402
from falsifiability_receipt import install_for, scope       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--json")
    ap.add_argument("--limit", type=int, default=400)
    a = ap.parse_args()

    install_for(["execution_receipt_gate"])
    import execution_receipt_gate as erg                     # noqa: PLC0415

    drafts = load_drafts(a.log, a.limit)
    if not drafts:
        print(f"no drafts in {a.log} — refusing to certify an empty population")
        return 2

    with scope("receipt_gate_fire_rate") as sc:
        fired = 0
        for d in drafts:
            res = erg.review(d, repo=CLAWD_REPO, since_s=86400)
            hit = bool(res.get("fires"))
            fired += hit
            sc.mark_item(hit)
        rate = fired / len(drafts)

    rec = sc.receipt(value=round(rate, 4))
    rec["population"] = {"source": os.path.basename(a.log), "n_drafts": len(drafts),
                         "n_fired": fired}
    # PIN THE SUBJECT. The first gate receipt had to be withdrawn as unauditable: the
    # source was edited 22 minutes after the receipt was cut, five commits followed, and
    # the line numbers it cited matched no version on disk. A certificate that cannot be
    # checked against the thing it certifies is not a certificate.
    try:
        rec["subject_commit"] = subprocess.run(
            ["git", "-C", CLAWD_REPO, "log", "-1", "--format=%H",
             "--", "scripts/execution_receipt_gate.py"],
            capture_output=True, text=True, timeout=30, shell=True).stdout.strip()
        rec["subject_dirty"] = bool(subprocess.run(
            ["git", "-C", CLAWD_REPO, "status", "--porcelain",
             "--", "scripts/execution_receipt_gate.py"],
            capture_output=True, text=True, timeout=30, shell=True).stdout.strip())
    except Exception as e:                                   # noqa: BLE001
        rec["subject_commit"] = f"unavailable: {type(e).__name__}"
    rec["subject"] = ("execution_receipt_gate.review — pinned at subject_commit; "
                      "a dirty tree invalidates this receipt")

    print(f"  receipt-gate fire rate : {rate:.4f}  ({fired}/{len(drafts)})")
    print(f"  terms on path          : {rec['n_terms_on_path']}")
    print(f"  live / constant        : {rec['n_live']} / {rec['n_constant']}"
          f"   (underpowered {rec['n_underpowered']})")
    print(f"  VERDICT                : {rec['verdict']}")
    print(f"  {rec['why']}")
    if rec["pinned_terms"]:
        print("\n  pinned throughout this measurement:")
        for p in rec["pinned_terms"][:12]:
            print(f"    {p['verdict']:14s} n={p['n']:5d} {p['func']} "
                  f"L{p['line']} [{p['op']}] {str(p['src'])[:56]}")
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
