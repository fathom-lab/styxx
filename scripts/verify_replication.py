"""Verify a replication submission against the canonical receipt (REPLICATIONS.md flow).

Usage: python scripts/verify_replication.py <target> <submitted_result.json>
Targets: b2-adaptive | b2-static | parity-control | e1

Checks, per REPLICATIONS.md: (1) the frozen verdict string matches; (2) each decisive per-cell
read is within TOLERANCE of the canonical value; prints a full comparison either way. Exit 0 on
verdict match (even with disclosed deltas), exit 1 on verdict mismatch or malformed submission —
divergent replications are merged manually with the replication-divergent label, never silently.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOLERANCE = 0.05

TARGETS = {
    "b2-adaptive": {
        "canonical": "papers/calib-poison-general/b2_adaptive_erasure_result.json",
        "verdict_key": "verdict",
        "cells_key": "points",
        "cell_id": ("seed", "alpha"),
        "decisive": ["private13_auroc", "eval_knowledge", "frozen18_read"],
    },
    "b2-static": {
        "canonical": "papers/calib-poison-general/b2_subspace_erasure_result.json",
        "verdict_key": "verdict",
        "cells_key": "points",
        "cell_id": ("seed", "alpha"),
        "decisive": ["private13_auroc", "eval_knowledge", "frozen18_read"],
    },
    "parity-control": {
        "canonical": "papers/calib-poison-general/honesty_parity_control_result.json",
        "verdict_key": "verdict",
        "cells_key": "points",
        "cell_id": ("seed",),
        "decisive": ["private13_auroc", "naive_matched13_auroc", "parity_gap"],
    },
    "e1": {
        "canonical": "papers/read-neq-write/e1_result.json",
        "verdict_key": "verdict",
        "cells_key": "points",
        "cell_id": ("seed",),
        "decisive": [],  # e1's schema differs per family; verdict-level only
    },
}


def cell_key(cell: dict, fields: tuple) -> tuple:
    return tuple(cell.get(f) for f in fields)


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in TARGETS:
        print(__doc__)
        return 1
    spec = TARGETS[sys.argv[1]]
    canonical = json.loads((ROOT / spec["canonical"]).read_text(encoding="utf-8"))
    try:
        submitted = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    except Exception as e:
        print(f"MALFORMED submission: {type(e).__name__}: {e}")
        return 1

    v_can = canonical.get(spec["verdict_key"])
    v_sub = submitted.get(spec["verdict_key"])
    verdict_match = v_can == v_sub
    print(f"verdict canonical : {v_can}")
    print(f"verdict submitted : {v_sub}")
    print(f"VERDICT MATCH     : {verdict_match}")

    max_delta = 0.0
    n_compared = 0
    if spec["decisive"] and spec["cells_key"] in canonical and spec["cells_key"] in submitted:
        can_cells = {cell_key(c, spec["cell_id"]): c for c in canonical[spec["cells_key"]]}
        for sub_cell in submitted[spec["cells_key"]]:
            k = cell_key(sub_cell, spec["cell_id"])
            if k not in can_cells:
                print(f"  cell {k}: NOT IN CANONICAL (extra cell in submission)")
                continue
            for field in spec["decisive"]:
                a, b = can_cells[k].get(field), sub_cell.get(field)
                if a is None or b is None:
                    print(f"  cell {k} {field}: MISSING ({a} vs {b})")
                    continue
                d = abs(float(a) - float(b))
                max_delta = max(max_delta, d)
                n_compared += 1
                flag = "ok" if d <= TOLERANCE else "OVER-TOLERANCE"
                print(f"  cell {k} {field}: canonical={a} submitted={b} delta={d:.4f} [{flag}]")
    print(f"\ncompared={n_compared} max_delta={max_delta:.4f} tolerance={TOLERANCE}")
    if not verdict_match:
        print("RESULT: DIVERGENT (verdict mismatch) -- label replication-divergent, do not merge silently")
        return 1
    print("RESULT: VERDICT REPLICATED" + ("" if max_delta <= TOLERANCE else " (with over-tolerance cells disclosed above)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
