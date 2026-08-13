"""Emit the C6 ceiling table as a receipt, instead of deriving it in prose.

Found by auditing FINDING_c6 with the hardened styxx.claim_audit: the implied pairwise
correlations (0.102, 0.130, 0.160, 0.194) and the licensing rates (0.095, 0.143) appeared only
in the finding's prose. They are correct arithmetic, but a number that exists only in a sentence
is exactly what the grounding auditor is for. This writes them to a receipt so the claims are
checkable.

Nothing here is a new analysis: every input is already committed in c6_basis_v2.json or
c6_result.json.
"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
basis = json.loads((HERE / "c6_basis_v2.json").read_text(encoding="utf-8"))
result = json.loads((HERE / "c6_result.json").read_text(encoding="utf-8"))

BAR = f">={result['bar_k']}/7"
rows = {}
for key, cell in basis["cells"].items():
    if not key.startswith("knee"):
        continue
    c = float(key.split("c=")[1])
    rows[f"c={c:.2f}"] = {
        "planted_coupling_c": c,
        "pairwise_r_implied": round(c * c, 4),
        "P_bar_cleared": cell["bar_table"][BAR]["rate"],
        "mean_licensed_fraction": cell["mean_licensed_fraction"],
    }

cleared95 = next((v for v in rows.values() if v["P_bar_cleared"] >= 0.95), None)
cleared100 = next((v for v in rows.values() if v["P_bar_cleared"] >= 1.0), None)

n_sub = result["n_sub"]
k = result["cohort_licensed_count"]

out = {
    "note": "ceiling table for FINDING_c6; all inputs already committed in "
            "c6_basis_v2.json and c6_result.json",
    "bar": BAR,
    "rho": 0.8054,
    "rows": rows,
    "first_c_with_95pct_power": cleared95["planted_coupling_c"] if cleared95 else None,
    "first_c_with_100pct_power": cleared100["planted_coupling_c"] if cleared100 else None,
    "ceiling_pairwise_r": cleared95["pairwise_r_implied"] if cleared95 else None,
    "c6_cohort_license_rate": round(k / n_sub, 4),
    "c5_pair_license_rate": round(2 / 21, 4),
    "c5_reference": "FINDING_c5: 2 of 21 pairs licensed",
}
(HERE / "c6_ceiling.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out, indent=2))
