"""How much does the logged plan constrain the choice class-two member 2 describes?

Member 2 is "the choice of which run becomes the baseline" -- the reference run whose values
become the subject's fingerprint, and against which the other four are differenced.

The roster calls it reachable by nothing. But this system already owns a mechanism for taking a
choice away from the party who benefits from it: preregistration. The plan is logged BEFORE the
runs, and log order is one of the five properties THE_BOUNDARY itself credits a self-written
record with establishing. The question is therefore not "can any check reach this" but "does the
plan already fix it, and does anything compare them" -- the exact test that reclassified the
environment member and the repository-and-revision member.

This counts, from the published prereg's own bytes, how many run schedules satisfy it.
"""
import itertools
import json
import pathlib

BASE = pathlib.Path(r"C:\Users\heyzo\clawd\wt\v8\papers\v8\first_verdict_2026_09_09\log\entries")

plan = json.loads((BASE / "000000" / "00000001.json").read_text(encoding="utf-8"))
body = plan["body"]
factors = {f["factor"]: f["values"] for f in body["nuisance"]}
n_runs = body["runs"]
covers = body["covers"]

print("THE LOGGED PLAN, entry 00000001")
print("  runs   :", n_runs)
print("  covers :", covers)
for k, v in factors.items():
    print(f"  {k}: {v}")
print("  keys it carries:", ", ".join(sorted(body)))
print("  does it name which assignment each run takes?",
      any(k in body for k in ("schedule", "assignments", "runs_spec", "reference")))

cells = list(itertools.product(*(factors[f] for f in sorted(factors))))
print(f"\n  the factorial has {len(cells)} cells: " +
      ", ".join("(" + ",".join(c) + ")" for c in cells))

total = len(cells) ** n_runs
print(f"\n  ordered schedules of {n_runs} runs over {len(cells)} cells: "
      f"{len(cells)}^{n_runs} = {total:,}")

# The log does enforce one thing, from the vacuous-floor repair: every covered factor must
# actually vary across the runs. Count the schedules that survive it.
def varies_all(sched):
    for pos, f in enumerate(sorted(factors)):
        if f in covers and len({s[pos] for s in sched}) < 2:
            return False
    return True

surviving = sum(1 for s in itertools.product(cells, repeat=n_runs) if varies_all(s))
print(f"  of those, schedules in which every covered factor varies: {surviving:,}")
print(f"  the reference is whichever the issuer puts first, so distinct reference cells "
      f"available: {len(cells)}")

# What the published run actually used.
runs = {}
for p in sorted(BASE.glob("*/*[0-9].json")):
    c = json.loads(p.read_text(encoding="utf-8"))
    if c.get("type") == "fingerprint" and "run_index" in c["body"]:
        n = c["body"]["nuisance"]
        runs[c["body"]["run_index"]] = (str(n["batch_size"]), n["item_order"])
print("\n  the schedule actually used, in run_index order:")
for i in sorted(runs):
    mark = "   <- the reference, and the fingerprint the subject gets" if i == 0 else ""
    print(f"    run {i}: batch_size={runs[i][0]:>2}  item_order={runs[i][1]}{mark}")

print(f"""
FINDING
  The plan fixes the factors, their levels, and the count. It does not fix the schedule, so
  {surviving:,} schedules satisfy it and the plan names none of them. The reference run --
  the one whose values become the subject's fingerprint -- is therefore chosen after the plan
  is logged, by the party the fingerprint is about, from {len(cells)} available cells.

  That is not a field for which no corroborating byte can exist. It is a field the plan is
  already the right place to carry and does not. Put the schedule in the plan and the predicate
  "the runs' assignments are the plan's schedule, in order" is a comparison between two logged
  certificates, which is exactly the shape of the predicate that reclassified member 3.

  The residue is the same one member 3 leaves: a party who commits to a favourable schedule
  before running is not caught by any comparison, because nothing contradicts them. That is
  suppression, not contradiction, and this lab filed member 3 under class one carrying an
  identical caveat.""")
