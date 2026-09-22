"""SWALLOW-13, after the run: stage 3 again, with the one change made after scoring (a hoisted grep's,
diff's, `git diff --exit-code`'s or `jq -e`'s status 1 kept, `|| [ $? -eq 1 ]`), on every check of the
receipt's stage-3 population, read from the run's own clones at the same tips. Not a result: what the
change moves, next to what was scored.

    python papers/harness/swallow13_after_fix.py --clones <the run's clones>     # writes swallow13_after_fix.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from styxx.ciaudit import engine                    # noqa: E402
from styxx.ciaudit import repair_frontier as F      # noqa: E402

RECEIPT = HERE / "swallow13_receipt.json.gz"
OUT = HERE / "swallow13_after_fix.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clones", type=Path, required=True)
    a = ap.parse_args(argv)
    raw = RECEIPT.read_bytes()
    r = json.loads(gzip.decompress(raw).decode("utf-8"))
    rows = []
    for rep in r["repos"]:
        for t in rep.get("targets", []):
            if not (t["tried_stage3"] and t["baseline"] is not None):
                continue
            tree = a.clones / rep["repo"].replace("/", "__")
            head = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
            text = (tree / ".github" / "workflows" / t["workflow"]).read_text(encoding="utf-8", errors="replace")
            again = F.try_frontier(text, t["workflow"], t["job"], t["index"], engine.Runner())
            was = t["verified_repair"] if t["stage"] == "swallow-13" else None
            was_diff = next((c.get("diff") for c in t["candidates"] if c["repair"] == was), None) if was else None
            now_diff = next((c.get("diff") for c in again["candidates"] if c["repair"] == again["verified_repair"]), None)
            rows.append({"repo": rep["repo"], "tip_checked_out": head == rep["tip"], "workflow": t["workflow"], "job": t["job"],
                         "index": t["index"], "name": t["name"], "scored": was, "after": again["verified_repair"],
                         "same_repair": was == again["verified_repair"], "same_diff": was_diff == now_diff,
                         "diff_after": now_diff if was_diff != now_diff else None})
    out = {"receipt_sha256": hashlib.sha256(raw).hexdigest(),
           "repair_frontier_sha256": hashlib.sha256((ROOT / "styxx" / "ciaudit" / "repair_frontier.py").read_bytes()).hexdigest(),
           "checks": len(rows), "all_tips_checked_out": all(x["tip_checked_out"] for x in rows),
           "verified_scored": sum(1 for x in rows if x["scored"]), "verified_after": sum(1 for x in rows if x["after"]),
           "repair_changed": [x for x in rows if not x["same_repair"]], "diff_changed": [x for x in rows if x["same_repair"] and not x["same_diff"]],
           "rows": rows}
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
