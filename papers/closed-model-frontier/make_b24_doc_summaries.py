"""Persist the B24 finding's grid-cell cites as summary receipt fields (OATH v0.3 repair).

The corpus oath caught FINDING_b24 citing two values that live only inside the bulk 74-cell ramp of
`b24_headtohead_result.json` (bulk arrays are not a claimable truth surface — coincidence): the
logit-lens AUROC at the winning white-box cell (POS-A, L29) and the strictly-pre-commit POS-B best
group-invariant AUROC. This derives both deterministically from the frozen receipt and adds them to
`b24_controls_addendum.json` under `doc_cited_summaries` (idempotent re-run).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    head = json.loads((HERE / "b24_headtohead_result.json").read_text(encoding="utf-8"))
    ramp = head["ramp_pos_layer_strat_group_lens"]   # rows: [pos, layer, strat, group, lens]
    a29 = next(r for r in ramp if r[0] == "A" and r[1] == 29)
    b_best_group = max((r for r in ramp if r[0] == "B"), key=lambda r: r[3])
    add_path = HERE / "b24_controls_addendum.json"
    add = json.loads(add_path.read_text(encoding="utf-8"))
    add["doc_cited_summaries"] = {
        "derived_from": "b24_headtohead_result.json ramp_pos_layer_strat_group_lens",
        "lens_auroc_at_WB_cell_A29": a29[4],
        "posB_best_groupinv_auroc": b_best_group[3],
        "posB_best_groupinv_layer": b_best_group[1],
    }
    add_path.write_text(json.dumps(add, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(add["doc_cited_summaries"], indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
