"""Recompute B50's derived quantities FROM the committed receipt, into their own receipt.

The finding quotes three numbers that are arithmetic on `b50_result.json` rather than fields in
it: the legibility-over-chance multiple, the largest internal gap in the member means, and the
number of null draws the G3 tail bound permits. OATH abstains on derived arithmetic (a known
verifier debt), so quoting them against the primary receipt alone is UNGROUNDED — correctly.

The lab rule for this case is an addendum receipt recomputed from committed data, never a
rewording. Nothing here is typed by hand; re-run it and diff.
"""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "b50_result.json"
r = json.loads(SRC.read_text(encoding="utf-8"))

means = np.array(sorted(r["member_mean_legibility"].values(), reverse=True))
gaps = means[:-1] - means[1:]

out = {
    "derived_from": "b50_result.json",
    "source_verdict": r["verdict"],
    "source_prereg_commit": r["prereg_commit"],
    "note": "every field below is arithmetic on the source receipt; no new computation was run",

    "max_pair_legibility_over_chance": round(r["max_pair_legibility"] / r["chance"], 4),
    "max_pair_legibility_over_chance_formula": "max_pair_legibility / chance",

    "member_mean_range": round(float(means[0] - means[-1]), 4),
    "member_mean_largest_internal_gap": round(float(gaps.max()), 4),
    "member_mean_largest_gap_formula": "max of successive differences, members sorted descending",

    "g3_tail_bound_permitted_draws": int(math.floor(0.10 * r["n_pairs"])),
    "g3_tail_bound_permitted_draws_formula": "floor(0.10 * n_pairs), the G3 bar as a draw count",
    "g3_tail_bound_observed_draws": r["n_nulls_above_5x_chance"],
}

DST = HERE / "b50_derived.json"
DST.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(f"wrote {DST}")
for k, v in out.items():
    if not k.endswith(("_formula", "note", "derived_from")):
        print(f"  {k} = {v}")
