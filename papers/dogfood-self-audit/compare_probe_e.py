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

    print(f"{'run':34s} {'terms':>7s} {'exercised':>10s} {'dead/powered':>13s} "
          f"{'live':>6s} {'const':>6s} {'no-pop files':>13s}")
    for r in rows:
        ef = f"{r['exercised_frac']:.1%}" if r["exercised_frac"] is not None else "-"
        dr = (f"{r['dead_rate_of_powered']:.1%}"
              if r["dead_rate_of_powered"] is not None else "-")
        npf = (f"{r['files_no_population']}/{r['n_test_files']}"
               if r["n_test_files"] else "-")
        print(f"{r['run'][:34]:34s} {r['terms']:7d} {ef:>10s} {dr:>13s} "
              f"{r['live']:6d} {r['constant']:6d} {npf:>13s}")

    print("\n  READ WITH THE COVERAGE COLUMN. The dead rate is computed over terms the")
    print("  suite exercised at least 8 times; where 'exercised' is small, the rate is")
    print("  a statement about a sliver of the package and must not be quoted as a")
    print("  property of the codebase. Comparing dead rates across rows with very")
    print("  different exercised fractions repeats the error this file exists to avoid.")

    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"rows": rows,
                       "caveat": ("dead_rate_of_powered is computed over the exercised "
                                  "minority; rows with small exercised_frac are not "
                                  "comparable to rows with large ones.")}, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
