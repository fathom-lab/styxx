# -*- coding: utf-8 -*-
"""handedness_accusations_v2.py -- frozen by PREREG_handedness_v2_header_bound_2026_09_02.

Join the 366 panel-judged accusations to the re-certified ledger (the verifier's own epistemics)
and read the HEADER cell against the LINE cell. No new judgement anywhere.

  python handedness_accusations_v2.py [--smoke]
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from styxx.protocol import Experiment      # noqa: E402

SMOKE = "--smoke" in sys.argv
PREREG = HERE / "PREREG_handedness_v2_header_bound_2026_09_02.md"


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def cell(rows):
    k = sum(1 for r in rows if r["panel"] == "CLAIM")
    return {"n": len(rows), "genuine": k, "genuine_share": round(k / len(rows), 4) if rows else None,
            "wilson95": wilson(k, len(rows)), "top_repos": collections.Counter(r["repo"] for r in rows).most_common(3)}


def main() -> int:
    adj = json.loads((HERE / "oath_adjudication_result.json").read_text(encoding="utf-8"))
    ledger = [json.loads(l) for l in (HERE / "oath_external_epistemics_ledger.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_key = collections.defaultdict(list)
    for r in ledger:
        by_key[(r["repo"], r["line"], str(r["token"]))].append(r)
    accused = adj["per_arm_detail"]["UNGROUNDED"]
    if SMOKE:
        accused = accused[:40]
    rows, unresolved = [], 0
    for it in accused:
        cands = by_key.get((it["repo"], it["line"], str(it["token"])), [])
        if len(cands) != 1:
            unresolved += 1
            continue
        c = cands[0]
        src = c.get("obligation_source")
        if src == "vocabulary":
            cellname = "header" if c.get("header_bound") else "line"
        else:
            cellname = src or "unobligated_now"
        rows.append({"id": it["id"], "repo": it["repo"], "line": it["line"], "token": it["token"],
                     "panel": it["verdict"], "source": src, "cell": cellname, "status_now": c["status"],
                     "header_bound": bool(c.get("header_bound"))})
    n = len(accused)
    cells = {name: cell([r for r in rows if r["cell"] == name])
             for name in sorted({r["cell"] for r in rows} | {"header", "line"})}
    h, ln = cells["header"]["genuine_share"], cells["line"]["genuine_share"]
    delta = (h - ln) if (h is not None and ln is not None) else None
    metrics = {"unresolved_share": round(unresolved / n, 4) if n else 1.0,
               "min_cell_n": min(cells["header"]["n"], cells["line"]["n"]),
               "delta_header_minus_line": round(delta, 4) if delta is not None else -1.0,
               "delta_line_minus_header": round(-delta, 4) if delta is not None else -1.0}
    v = Experiment(PREREG, repo_root=ROOT).score(metrics, smoke=SMOKE)
    res = {"prereg": PREREG.name, "smoke": SMOKE, "accusations": n, "joined": len(rows), "unresolved": unresolved,
           "still_accused_now": sum(1 for r in rows if r["status_now"] == "UNGROUNDED"),
           "cells": cells, "metrics": metrics, "verdict": v.verdict, "gates": v.gates, "rows": rows,
           "what_this_is_not": "no token re-judged; sources are the current verifier's own epistemics on the pinned documents; the panel verdicts are the 2026-08-27 receipt's"}
    out = HERE / ("handedness_v2_smoke.json" if SMOKE else "handedness_v2_result.json")
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("cells:", {k: (c["n"], c["genuine_share"], c["wilson95"]) for k, c in cells.items()})
    print("metrics:", metrics, "| still accused now:", res["still_accused_now"], "/", len(rows))
    print(f"\n===== VERDICT: {res['verdict']} =====\nwrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
