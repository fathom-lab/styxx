"""The chance floor of `audit_grounding`, resolved by decimal precision.

The qualification run refuted my pre-stated prediction: the judge DOES discriminate (real 0.868
vs fabricated 0.303). But it also showed a random 2-decimal number grounds 0.574 of the time
against this receipt. Both facts are true, and together they say something neither says alone:

    a grounded RATE is uninterpretable without the chance floor for the PRECISION of the claims,
    because tolerance in `_match` is 0.5 * 10^-decimals — so the floor collapses as claims carry
    more decimals, and rises as the source receipt grows.

This script measures that floor directly: for d in 1..4 decimals, draw random values in [0,1] and
ask what fraction ground against the real receipt. Then it restates my own C6 audit as an
excess-over-chance figure per precision class, which is the honest form of the number.
"""
from __future__ import annotations
import json
import pathlib
import random
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from styxx.claim_audit import audit_grounding, _flatten, _decimals  # noqa: E402

FA = HERE.parent / "first-afference"
SOURCES = {"basis_v2": json.loads((FA / "c6_basis_v2.json").read_text(encoding="utf-8")),
           "power": json.loads((FA / "c6_power.json").read_text(encoding="utf-8"))}
PREREG = (FA / "PREREG_c6_derived_bar_2026_08_13.md").read_text(encoding="utf-8")

flat = _flatten(SOURCES, "", {})
vals = list(flat)


def chance_floor(d, trials=20000, seed=7):
    """P(a uniformly random d-decimal number in [0,1] grounds against this receipt)."""
    rng = random.Random(seed + d)
    tol = 0.5 * 10 ** (-d)
    hits = 0
    for _ in range(trials):
        q = round(rng.uniform(0, 1), d)
        if any(abs(v - q) <= tol or round(v, d) == q for v in vals):
            hits += 1
    return hits / trials


def main():
    print("=" * 74)
    print("CHANCE FLOOR of audit_grounding, by claim precision")
    print(f"receipt has {len(vals)} distinct leaf values")
    print("=" * 74)
    floors = {}
    for d in (1, 2, 3, 4):
        f = chance_floor(d)
        floors[d] = f
        print(f"  {d} decimal(s): tol=±{0.5*10**-d:<8.5f}  chance floor = {f:.4f}")

    rep = audit_grounding(PREREG, SOURCES)
    items = rep.items if hasattr(rep, "items") else rep.__dict__["items"]
    byprec = defaultdict(lambda: [0, 0])
    for it in items:
        d = _decimals(it.raw)
        byprec[d][1] += 1
        if it.status in ("grounded", "derived"):
            byprec[d][0] += 1

    print("\nMY OWN C6 PREREG, restated against the floor for its precision:")
    print("  prec   grounded/total   observed   floor    excess")
    tot_g = tot_n = 0
    weighted_floor = 0.0
    for d in sorted(byprec):
        g, n = byprec[d]
        obs = g / n
        fl = floors.get(d, floors[4] if d > 4 else floors[1])
        tot_g += g
        tot_n += n
        weighted_floor += fl * n
        print(f"  {d:>4}   {g:>3}/{n:<3}          {obs:.3f}      {fl:.4f}   {obs-fl:+.3f}")
    overall = tot_g / tot_n
    wf = weighted_floor / tot_n
    print(f"\n  OVERALL: {tot_g}/{tot_n} = {overall:.3f} observed")
    print(f"  precision-weighted chance floor = {wf:.3f}")
    print(f"  EXCESS OVER CHANCE = {overall - wf:+.3f}")
    denom = 1 - wf
    norm = (overall - wf) / denom if denom > 1e-9 else float("nan")
    print(f"  normalised (excess / headroom) = {norm:.3f}")

    print("\nREADING: the headline '65/76 grounded' is not wrong, but on its own it is not the")
    print("claim a reader will take from it. Most of my prereg's numbers carry 2-3 decimals,")
    print("where a randomly chosen value already grounds a substantial fraction of the time")
    print("against a receipt this size. The excess figure is the part that is about my honesty.")

    out = {"n_source_leaf_values": len(vals),
           "chance_floor_by_decimals": {str(k): round(v, 5) for k, v in floors.items()},
           "prereg_by_precision": {str(d): {"grounded": byprec[d][0], "total": byprec[d][1]}
                                   for d in sorted(byprec)},
           "observed_overall": round(overall, 4),
           "precision_weighted_chance_floor": round(wf, 4),
           "excess_over_chance": round(overall - wf, 4),
           "normalised_excess": round(norm, 4)}
    (HERE / "grounding_chance_floor.json").write_text(json.dumps(out, indent=2) + "\n",
                                                      encoding="utf-8")
    print("\nwrote grounding_chance_floor.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
