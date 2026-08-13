"""Red-team of claim_audit.py's self-audit fixes (commits 1fb1de5, 4de77d1, af62490).

Commissioned by the module's author, who asked specifically for the paths he did not
visit. Run standalone; imports the shipped module READ-ONLY and modifies nothing.

    python redteam_claim_audit.py            # prints receipts, exit 0
    python redteam_claim_audit.py --json     # machine-readable

LANE 1 — chance floor reference distribution (_chance_floor)
LANE 2 — context resolver confident-wrong (_resolve_by_context)
LANE 3 — accounting identity n_ambiguous == n_context_resolved + n_arbitrary

Every lane prints what it found OR states plainly that it found nothing.
"""
import argparse
import json
import random
import sys

from styxx.claim_audit import audit_grounding
from styxx.claim_audit import _chance_floor, _resolve_by_context, _tokens  # noqa

RESULTS = {}


# --------------------------------------------------------------------------- lane 2
def lane2_resolver():
    """The resolver scores overlap / len(path_tokens). The denominator is the PATH,
    so a short generic path needs one lucky word to score 1.0 while a long, specific
    path is penalised for every token the prose did not happen to repeat.

    Attack: a receipt that carries BOTH a short summary key and the specific nested
    key the sentence actually names. Realistic — styxx receipts routinely carry a
    flat summary alongside per-cell detail.
    """
    cases = []

    # Case A: sentence names the specific cell almost completely; a bare summary key wins.
    src_a = {"rate": 0.1, "cells": {"blockconf_ge3_of_7": {"cave_rate": 0.1}}}
    txt_a = ("The blockconf arm at >=3/7 shows a cave rate of 0.100 on the frozen "
             "protocol.")
    cases.append(("A_summary_key_beats_named_cell", txt_a, src_a,
                  "cells.blockconf_ge3_of_7.cave_rate"))

    # Case B: two nested cells, one named in prose, but the named one has MORE tokens.
    src_b = {"knee": {"rate": 0.25},
             "blockconf_high_confidence_arm": {"cave_rate": 0.25}}
    txt_b = ("For the blockconf high confidence arm the cave rate was 0.250, "
             "materially above the knee.")
    cases.append(("B_longer_named_path_loses_to_shorter", txt_b, src_b,
                  "blockconf_high_confidence_arm.cave_rate"))

    # Case C: control — the two candidates are the same length. Expect a decline
    # (arbitrary) or the correct pick; a WRONG "context" label here would be worse.
    src_c = {"arm_one": {"cave_rate": 0.5}, "arm_two": {"cave_rate": 0.5}}
    txt_c = "In arm two the cave rate reached 0.500."
    cases.append(("C_control_equal_length", txt_c, src_c, "arm_two.cave_rate"))

    out = []
    for name, txt, src, expected in cases:
        rep = audit_grounding(txt, src)
        for it in rep.items:
            if it.status != "grounded" or it.n_candidates <= 1:
                continue
            out.append({
                "case": name, "claim": it.raw, "expected_path": expected,
                "resolved_path": it.source, "label": it.resolved_by,
                "score": it.context_score,
                "confident_wrong": bool(it.resolved_by == "context"
                                        and it.source != expected),
            })
    wrong = [o for o in out if o["confident_wrong"]]
    RESULTS["lane2"] = {"cases": out, "n_confident_wrong": len(wrong),
                        "found_defect": bool(wrong)}
    return out


# --------------------------------------------------------------------------- lane 1
def lane1_chance_floor():
    """The floor samples [0, 1] for any claim with >=1 decimal place. That is the
    right ORDER of magnitude but the wrong BAND when the audited claims cluster in a
    narrow sub-range where the source values also cluster — the uniform draw spends
    most of its mass in empty territory, so the floor reads lower than the luck a
    real claim actually enjoys. Lower floor = the grounding rate looks better than
    chance explains. Flattering direction.

    Reference used for comparison: the same Monte Carlo restricted to the band the
    document's own claims occupy (min..max of the claimed values).
    """
    # A receipt whose leaves are all small rates — the shape of every cave-rate
    # receipt in this repo.
    rng = random.Random(11)
    vals = {round(rng.uniform(0.0, 0.25), 3): (f"cells.c{i}.cave_rate", None)
            for i in range(120)}
    decimals = 3
    # POST-FIX (f44c8f4): _chance_floor grew a `band` parameter and audit_grounding
    # now passes the claims' own band. Calling it WITHOUT a band exercises the legacy
    # fallback, not the shipped path — on the first re-run this script did exactly
    # that and reported a defect that no longer exists. Recorded because the rule is
    # symmetric: my false alarm belongs in the receipt beside his defects.
    legacy = _chance_floor(vals, decimals)                    # no band: old behaviour
    try:
        shipped = _chance_floor(vals, decimals, band=(0.0, 0.25))
        band_param_available = True
    except TypeError:                                          # pre-fix module
        shipped, band_param_available = legacy, False

    # Band-matched reference: draw from the claims' own range instead of [0,1].
    xs = list(vals)
    lo, hi = 0.0, 0.25
    tol = 0.5 * 10 ** (-decimals)
    r3 = random.Random(7 + decimals)  # same seed rule the shipped floor uses
    hits = 0
    for _ in range(4000):
        q = round(r3.uniform(lo, hi), decimals)
        if any(abs(v - q) <= tol or round(v, decimals) == q for v in xs):
            hits += 1
    band_matched = hits / 4000

    RESULTS["lane1"] = {
        "n_source_leaves": len(vals), "decimals": decimals,
        "claim_band": [lo, hi],
        "band_param_available": band_param_available,
        "legacy_floor_no_band": legacy,
        "shipped_floor": shipped, "band_matched_floor": band_matched,
        "gap": round(band_matched - shipped, 4),
        "gap_before_fix": round(band_matched - legacy, 4),
        "flattering": band_matched > shipped,
        "found_defect": bool(band_matched - shipped >= 0.05),
    }
    return RESULTS["lane1"]


# --------------------------------------------------------------------------- lane 3
def lane3_identity():
    """Try to violate n_ambiguous == n_context_resolved + n_arbitrary, and to make a
    value be counted in two categories at once (a raw claim that is also a derived
    ratio of two other source values)."""
    # Collision receipt: 0.5 is a leaf AND 1/2 of two other leaves; 50.0 is a percent
    # of the same pair; several leaves share values to force ambiguity.
    src = {"a": {"x": 0.5}, "b": {"x": 0.5}, "c": {"n": 2.0}, "d": {"n": 4.0},
           "e": {"ratio": 0.5}, "f": {"pct": 50.0}}
    txt = ("The x value was 0.500 in both arms, a ratio of 0.500 and 50.0 percent "
           "of the total, with n of 2.0 and 4.0 respectively.")
    rep = audit_grounding(txt, src)
    identity_ok = (rep.n_ambiguous == rep.n_context_resolved + rep.n_arbitrary)
    total_ok = (rep.n_total == rep.n_grounded + rep.n_derived + rep.n_unsourced)
    RESULTS["lane3"] = {
        "n_total": rep.n_total, "n_grounded": rep.n_grounded,
        "n_derived": rep.n_derived, "n_unsourced": rep.n_unsourced,
        "n_ambiguous": rep.n_ambiguous,
        "n_context_resolved": rep.n_context_resolved,
        "n_arbitrary": rep.n_arbitrary,
        "identity_holds": identity_ok, "category_total_holds": total_ok,
        "found_defect": not (identity_ok and total_ok),
    }
    return RESULTS["lane3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    lane2_resolver()
    lane1_chance_floor()
    lane3_identity()

    if args.json:
        print(json.dumps(RESULTS, indent=1))
        return

    print("=" * 70)
    print("LANE 2 — context resolver (the author's own priority bet)")
    print("=" * 70)
    for c in RESULTS["lane2"]["cases"]:
        flag = "  <-- CONFIDENT-WRONG" if c["confident_wrong"] else ""
        print(f"  {c['case']}: claim {c['claim']} -> {c['resolved_path']!r} "
              f"[{c['label']} {c['score']}] expected {c['expected_path']!r}{flag}")
    print(f"  confident-wrong cases: {RESULTS['lane2']['n_confident_wrong']}")

    print("=" * 70)
    print("LANE 1 — chance floor reference distribution")
    print("=" * 70)
    l1 = RESULTS["lane1"]
    print(f"  {l1['n_source_leaves']} leaves in {l1['claim_band']}, "
          f"{l1['decimals']} decimals")
    print(f"  shipped floor       : {l1['shipped_floor']}")
    print(f"  band-matched floor  : {l1['band_matched_floor']}")
    print(f"  gap (flattering if >0): {l1['gap']}")

    print("=" * 70)
    print("LANE 3 — accounting identity")
    print("=" * 70)
    l3 = RESULTS["lane3"]
    print(f"  n_ambiguous={l3['n_ambiguous']} == context {l3['n_context_resolved']} "
          f"+ arbitrary {l3['n_arbitrary']} -> {l3['identity_holds']}")
    print(f"  n_total={l3['n_total']} == grounded {l3['n_grounded']} + derived "
          f"{l3['n_derived']} + unsourced {l3['n_unsourced']} -> "
          f"{l3['category_total_holds']}")

    found = [k for k, v in RESULTS.items() if v.get("found_defect")]
    print("\nLANES WITH A DEFECT:", ", ".join(sorted(found)) if found else "none")


if __name__ == "__main__":
    sys.exit(main())
