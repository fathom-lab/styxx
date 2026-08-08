"""Find frozen gates structurally vulnerable to the E1 composition defect.

    python papers/scan_extremum_gates.py

E1 (2026-08-08) failed because `G1` judged the *minimum error over all candidates* while `G2`
disqualified one of those candidates in the same run — so G1 passed on a value belonging to a
candidate G2 had already excluded. Every component was individually correct; the composition was
not. `styxx.protocol` has no check on relationships between gates.

**This scan reports a PRECONDITION, not a defect.** A gate judging an extremum over a set is
vulnerable *only if* another gate in the same prereg restricts that set. Confirming that requires
reading the prereg; the scan cannot do it, and a count from this script must never be quoted as a
count of defects. It exists to bound the reading list.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXTREMUM = re.compile(r'^(max|min|best|worst|strongest|weakest)_|_(max|min|best)$')
OUT = ROOT / "extremum_gate_scan.json"

rows = []
for p in sorted(ROOT.rglob("PREREG_*.md")):
    m = re.search(r'```gates\s*(\{.*?\})\s*```', p.read_text(encoding="utf-8"), re.S)
    if not m:
        continue
    try:
        gates = json.loads(m.group(1)).get("gates", {})
    except json.JSONDecodeError:
        continue
    ext = {n: d.get("metric") for n, d in gates.items()
           if isinstance(d, dict) and EXTREMUM.search(str(d.get("metric", "")))}
    if ext and len(gates) > 1:
        rows.append({"prereg": p.name, "n_gates": len(gates), "extremum_gates": ext})

res = {"n_preregs_with_precondition": len(rows),
       "confirmed_defects": ["PREREG_e1_effective_n_bakeoff_2026_08_08.md"],
       "WARNING": ("n_preregs_with_precondition is a COUNT OF PRECONDITIONS, not of defects. A "
                   "gate judging an extremum over a set is vulnerable only if another gate "
                   "restricts that set, which this scan cannot determine. Quoting this number as "
                   "a defect count would be exactly the kind of overclaim this program exists to "
                   "refuse."),
       "preregs": rows}
OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
print(f"wrote {OUT}")
print(f"{len(rows)} preregs carry the precondition; 1 confirmed defect (E1)")
for r in rows:
    print(f"  {r['prereg']} ({r['n_gates']} gates): {', '.join(r['extremum_gates'])}")
