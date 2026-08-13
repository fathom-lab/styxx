"""Compare PROBE E runs across repositories without repeating the census's mistake.

`EXTERNAL_CENSUS_2026_08_13.md` put styxx's static rate next to nine libraries' and
called us the worst on the list. That comparison was invalid twice over: once because
broad and strict modes count different denominators, and again -- discovered by
execution -- because the static screen misses ~69% of real dead terms anyway. Both
errors had the same shape: a number placed beside another number that was not measuring
the same thing.

So this table refuses to print a bare dead rate. Every row carries the **exercised
fraction** alongside it, because the two are not independent: a suite that drives 40% of
its package's decision terms to the observation floor is answering a different question
than one that drives 3%, and the dead rate is computed over the exercised minority in
both cases. A repository whose suite barely runs will show a dead rate calculated from
almost nothing, and without the coverage column that reads as a finding.

Chunks whose suite did not execute are excluded from the population and counted, so a
library whose tests need a Fortran compiler is not silently scored as if they ran.

    python compare_probe_e.py --runs probe_e_styxx_joined.json probe_e_numpy_v2.json ...
"""
from __future__ import annotations

import argparse
import json
import os

# Two runs whose exercised fractions differ by more than this factor are not compared at
# all. 1.5 is a judgement call and is stated rather than buried: styxx at 37.1% against
# numpy at 78.5% is a factor of 2.1, which is exactly the pair that prompted the guard.
MAX_COVERAGE_SPREAD = 1.5


def summarise(path):
    with open(path, encoding="utf-8") as f:
        r = json.load(f)
    c = r.get("counts", {})
    pop = r.get("population", {})
    total = r.get("n_terms_instrumented", 0)
    powered = r.get("n_powered", 0)
    dead = r.get("n_dead_of_powered", 0)
    cj = r.get("census_join", {})
    return {
        "run": os.path.basename(path),
        "terms": total,
        "powered": powered,
        "exercised_frac": round(powered / total, 4) if total else None,
        "live": c.get("LIVE", 0),
        "constant": c.get("CONSTANT_TRUE", 0) + c.get("CONSTANT_FALSE", 0),
        "underpowered": c.get("UNDERPOWERED", 0),
        "never_reached": c.get("NEVER_REACHED", 0),
        "dead_rate_of_powered": r.get("dead_rate_of_powered"),
        "dead_rate_adjudicative": r.get("dead_rate_adjudicative"),
        "adjudicative_powered": r.get("n_adjudicative_powered"),
        "adjudicative_dead": r.get("n_adjudicative_dead"),
        "value_dead": r.get("n_value_position_dead"),
        "dead_at_process_multiple": r.get("n_dead_at_process_multiple"),
        "n_test_files": pop.get("n_test_files"),
        "files_no_population": pop.get("n_files_failed_to_run"),
        "census_candidates": cj.get("n_static_candidate_functions"),
        "census_confirmed": cj.get("n_confirmed_dead_by_execution"),
        "census_refuted": cj.get("n_refuted_all_terms_live"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()

    rows = [summarise(p) for p in a.runs if os.path.exists(p)]
    if not rows:
        print("no readable runs")
        return 2

    print(f"{'run':30s} {'terms':>7s} {'exercised':>10s} {'dead/pow':>9s} "
          f"{'DEAD/adj':>9s} {'val-dead':>9s} {'no-pop':>8s}")
    for r in rows:
        ef = f"{r['exercised_frac']:.1%}" if r["exercised_frac"] is not None else "-"
        dr = (f"{r['dead_rate_of_powered']:.1%}"
              if r["dead_rate_of_powered"] is not None else "-")
        da = (f"{r['dead_rate_adjudicative']:.1%}"
              if r.get("dead_rate_adjudicative") is not None else "-")
        npf = (f"{r['files_no_population']}/{r['n_test_files']}"
               if r["n_test_files"] else "-")
        print(f"{r['run'][:30]:30s} {r['terms']:7d} {ef:>10s} {dr:>9s} "
              f"{da:>9s} {str(r.get('value_dead', '-')):>9s} {npf:>8s}")
    print("  (DEAD/adj is the headline: decision terms only. dead/pow pools in "
          "value-position\n   operands, 21.9% of which were not decisions at all "
          "when this was audited.)")

    # A WARNING IS NOT A GUARD. The first version printed a paragraph asking the reader
    # not to compare rows whose coverage differs -- which is the same move as labelling
    # a chunk "no population" and merging its rows anyway, and it was flagged in review
    # for exactly that reason. If two rows are not comparable the tool refuses to
    # present them as a comparison, rather than presenting one and appealing to the
    # reader's restraint.
    fracs = [r["exercised_frac"] for r in rows if r["exercised_frac"]]
    if len(fracs) >= 2:
        spread = max(fracs) / min(fracs)
        if spread > MAX_COVERAGE_SPREAD:
            print(f"\n  !! REFUSING TO COMPARE. Exercised fractions span "
                  f"{min(fracs):.1%}-{max(fracs):.1%}, a factor of {spread:.2f} "
                  f"(limit {MAX_COVERAGE_SPREAD}). `dead_rate_of_powered` is computed "
                  f"over each suite's exercised minority, so these rows answer "
                  f"different questions. The per-row numbers above stand on their own; "
                  f"the pairwise reading does not.")
            return 3

    print("\n  Rows are within the coverage spread limit, which makes them arguably")
    print("  comparable and does NOT make them a quality ranking: dead_rate_of_powered")
    print("  is a joint property of code composition, suite design, and observations")
    print("  per term. A suite that parametrises heavily over the very parameters its")
    print("  gates read will move those terms by construction.")

    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"rows": rows,
                       "caveat": ("dead_rate_of_powered is computed over the exercised "
                                  "minority; rows with small exercised_frac are not "
                                  "comparable to rows with large ones.")}, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
