"""Find certificate-cited receipts that no committed script can regenerate.

    python papers/scan_orphan_receipts.py

E1 (cycle 159) found that `c5_effective_df_addendum.json` — the receipt beneath a sealed
finding's central number — had no generator script, and five standard methods all failed to
reproduce it (closest differed by 7.4805). A number nobody can reproduce is worse than a wrong
one, because a wrong one can be checked. The rule is now explicit: **every receipt must be
regenerable by a committed script.** This scan is the work list for enforcing it retroactively.

**The count is an UPPER BOUND on true orphans, not a defect count.** Detection is a filename/stem
heuristic (a runner that constructs its output name from variables the heuristic cannot see will
be flagged falsely), and absence of a mention is not proof of irreproducibility — only a flag
that reproducibility is unverified. Each flagged receipt needs a human to either point at its
generator or write the backfill.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "orphan_receipt_scan.json"

code = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in ROOT.rglob("*.py"))

cited = set()
for c in ROOT.rglob("*.certificate.json"):
    try:
        d = json.loads(c.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    rs = d.get("receipts_sha256")
    if isinstance(rs, dict):
        cited.update(Path(r).name for r in rs)


def has_generator(name: str) -> bool:
    stem = name[:-5]
    frags = {name, stem, stem.replace("_result", ""), stem.replace("_smoke", "")}
    return any(f in code for f in frags if len(f) >= 6)


orphans = sorted(n for n in cited if not has_generator(n))
addenda = [n for n in orphans if "addendum" in n]

res = {"n_receipts_cited_by_certificates": len(cited),
       "n_orphans_upper_bound": len(orphans),
       "n_orphan_addenda": len(addenda),
       "WARNING": ("UPPER BOUND from a filename/stem heuristic — dynamic output names create "
                   "false positives, and a mention is not proof of regenerability. This is a "
                   "work list, not a defect count; quoting n_orphans_upper_bound as a count of "
                   "irreproducible receipts would be an overclaim."),
       "pattern_note": (f"{len(addenda)} of {len(orphans)} flagged receipts are addenda — the "
                        "C5 pattern: a number computed ad hoc in a session and committed "
                        "without its computation."),
       "confirmed_irreproducible": ["c5_effective_df_addendum.json  (E1: five standard methods "
                                    "all fail to reproduce it; closest differs by 7.4805)"],
       "orphans": orphans}
OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
print(f"wrote {OUT}")
print(f"{len(cited)} cited receipts, {len(orphans)} flagged (upper bound), "
      f"{len(addenda)} of them addenda")
