"""How many source paths could each 'grounded' claim have matched?

`_match` returns the FIRST path in dict order whose value is within tolerance. It never looks at
the sentence. If a claim's value matches many paths, the reported `source` is arbitrary — an
artifact of dict ordering, not evidence about the claim.

This measures the ambiguity directly on my own C6 audit before any fix is designed.
"""
from __future__ import annotations
import json
import pathlib
import sys
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from styxx.claim_audit import audit_grounding, _load_sources, _decimals  # noqa: E402

FA = HERE.parent / "first-afference"
SOURCES = {"basis_v2": json.loads((FA / "c6_basis_v2.json").read_text(encoding="utf-8")),
           "power": json.loads((FA / "c6_power.json").read_text(encoding="utf-8"))}
PREREG = (FA / "PREREG_c6_derived_bar_2026_08_13.md").read_text(encoding="utf-8")

vals = _load_sources([SOURCES])
rep = audit_grounding(PREREG, SOURCES)
items = rep.items if hasattr(rep, "items") else rep.__dict__["items"]

def _cands_independent(it):
    """Re-derived here on purpose: an independent second opinion on the module's own count.

    It must handle percents in BOTH spaces (a "20%" claim can legitimately match a rate 0.20 or a
    count 20). The first version of this script checked only the as-is space and therefore
    reported 13 ambiguous where the module said 14 — the module was right and this script was
    incomplete. Kept as a worked example: when two implementations disagree, find out which is
    wrong before quoting either.
    """
    d = _decimals(it.raw)
    tol = 0.5 * 10 ** (-d)
    out = []
    for sv, p in vals.items():
        if it.kind == "percent":
            if abs(sv * 100 - it.value) <= tol or abs(sv - it.value) <= tol:
                out.append(p)
        elif abs(sv - it.value) <= tol or round(sv, d) == round(it.value, d):
            out.append(p)
    return out


rows = []
for it in items:
    if it.status != "grounded":
        continue
    d = _decimals(it.raw)
    cands = _cands_independent(it)
    assert len(cands) == it.n_candidates, (
        f"independent count {len(cands)} != module {it.n_candidates} for {it.raw!r}")
    rows.append((it.raw, d, len(cands), it.source, cands))

n = len(rows)
amb = [r for r in rows if r[2] > 1]
print(f"grounded claims examined: {n}")
print(f"claims whose value matches MORE THAN ONE source path: {len(amb)}  "
      f"({len(amb)/max(n,1):.1%})")
cnt = Counter(r[2] for r in rows)
print("\ncandidate-count distribution:")
for k in sorted(cnt):
    print(f"  {k:>3} candidate path(s): {cnt[k]:>3} claim(s)")

worst = sorted(rows, key=lambda r: -r[2])[:8]
print("\nmost ambiguous claims (reported source is dict-order arbitrary):")
for raw, d, k, src, cands in worst:
    print(f"  {raw!r:>10}  {k:>3} candidates   reported -> {src}")
    for c in cands[:3]:
        print(f"                   also: {c}")

by_prec = {}
for raw, d, k, src, cands in rows:
    by_prec.setdefault(d, []).append(k)
print("\nmean candidate count by precision:")
for d in sorted(by_prec):
    ks = by_prec[d]
    print(f"  {d} dec: mean {sum(ks)/len(ks):.2f} candidates over {len(ks)} claim(s)")

out = {"n_grounded_examined": n, "n_ambiguous": len(amb),
       "ambiguous_fraction": round(len(amb) / max(n, 1), 4),
       "candidate_count_distribution": {str(k): v for k, v in sorted(cnt.items())},
       "mean_candidates_by_precision": {str(d): round(sum(v)/len(v), 3)
                                        for d, v in sorted(by_prec.items())}}
(HERE / "match_ambiguity.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print("\nwrote match_ambiguity.json")
