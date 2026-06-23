"""Audit the SHIPPED competence_cliff (7.18.0) tier assignments for statistical robustness.

Each domain is tagged SAFE (committed_precision >= 0.90) / REVIEW (0.60-0.90) / DO-NOT-DEPLOY (< 0.60)
from a single-run POINT estimate. But committed_precision = committed_correct / committed_n, and
committed_n is tiny for many domains. This computes the Wilson 95% CI per domain and asks: is each tier
assignment ROBUST (the whole CI on the tagged side of the threshold), or FRAGILE (CI crosses the
threshold)? Reads the package-data JSON; no GPU, no network.
"""
from __future__ import annotations
import json, math
from pathlib import Path

DATA = Path(r"C:\Users\heyzo\clawd\styxx\styxx\_data\competence_cliff_truthfulqa_gpt4omini_v1.json")
SAFE_T, REVIEW_T = 0.90, 0.60

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / d
    return (max(0.0, c-h), min(1.0, c+h))

def tier(p):
    if p != p: return "NO-COMMIT"
    if p >= SAFE_T: return "SAFE"
    if p >= REVIEW_T: return "REVIEW"
    return "DO-NOT-DEPLOY"

def main():
    d = json.load(open(DATA, encoding="utf-8"))
    cats = d["categories"]
    rows = []
    for name, v in cats.items():
        cn = v.get("committed_n", 0)
        cp = v.get("committed_precision")
        if isinstance(cp, float) and cp != cp: cp = None
        k = round(cp * cn) if (cp is not None and cn) else None
        lo, hi = wilson(k, cn) if k is not None else (float("nan"), float("nan"))
        t = tier(cp if cp is not None else float("nan"))
        # robust iff the whole CI sits on the tagged side of BOTH relevant thresholds
        if t == "SAFE":
            robust = lo >= SAFE_T
        elif t == "REVIEW":
            robust = lo >= REVIEW_T and hi < SAFE_T
        elif t == "DO-NOT-DEPLOY":
            robust = hi < REVIEW_T
        else:
            robust = False
        rows.append({"domain": name, "committed_n": cn, "committed_precision": cp,
                     "ci95": [round(lo,3), round(hi,3)] if k is not None else None,
                     "tier": t, "robust": bool(robust)})
    rows.sort(key=lambda r: (r["tier"], -(r["committed_n"])))

    from collections import Counter
    tc = Counter(r["tier"] for r in rows)
    safe = [r for r in rows if r["tier"] == "SAFE"]
    safe_robust = [r for r in safe if r["robust"]]
    print("="*78)
    print("SHIPPED competence_cliff (7.18.0) — per-domain tier robustness (Wilson 95% CI)")
    print("="*78)
    print(f"tiers: {dict(tc)}")
    print(f"SAFE domains: {len(safe)} | statistically robust (CI_lower >= 0.90): {len(safe_robust)}")
    print(f"committed_n for SAFE domains: min {min(r['committed_n'] for r in safe)}, "
          f"median {sorted(r['committed_n'] for r in safe)[len(safe)//2]}, max {max(r['committed_n'] for r in safe)}")
    print("\nSAFE-tagged domains and whether 1.00/0.9x precision robustly clears 0.90:")
    for r in safe:
        ci = r["ci95"]
        print(f"  {r['domain']:28s} cp={r['committed_precision']:.2f}  cn={r['committed_n']:<3}  "
              f"CI[{ci[0]:.2f},{ci[1]:.2f}]  {'ROBUST' if r['robust'] else 'FRAGILE (CI dips below 0.90)'}")
    dnb = [r for r in rows if r["tier"] == "DO-NOT-DEPLOY"]
    print("\nDO-NOT-DEPLOY domains (the actionable warnings) — robust?")
    for r in dnb:
        ci = r["ci95"]
        print(f"  {r['domain']:28s} cp={r['committed_precision']:.2f}  cn={r['committed_n']:<3}  "
              f"CI[{ci[0]:.2f},{ci[1]:.2f}]  {'ROBUST' if r['robust'] else 'FRAGILE'}")

    # how large must committed_n be for a 1.00 observation to clear 0.90 lower bound?
    need = next(n for n in range(2, 500) if wilson(n, n)[0] >= 0.90)
    print(f"\nFor an observed precision of 1.00, committed_n >= {need} is needed for Wilson lower >= 0.90.")
    out = {"safe_total": len(safe), "safe_robust": len(safe_robust),
           "dnb_total": len(dnb), "dnb_robust": sum(1 for r in dnb if r["robust"]),
           "min_committed_n_for_safe_at_p1": need, "rows": rows}
    (DATA.parent.parent / "papers" / "grounded-honesty-axis" / "shipped_cliff_ci_result.json")
    Path(r"C:\Users\heyzo\clawd\styxx\papers\grounded-honesty-axis\shipped_cliff_ci_result.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\nwritten: shipped_cliff_ci_result.json")

if __name__ == "__main__":
    main()
