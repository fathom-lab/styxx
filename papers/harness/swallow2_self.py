"""G-S2-3 of SWALLOW-2: the fault instrument on this repository, and on the #137 shape in both
its states. Writes swallow2_self.json.

    python papers/harness/swallow2_self.py --before /path/to/gauntlet-pr.yml@main --after /path/to/gauntlet-pr.yml@137

Three things are recorded:
  1. the per-step "alone" verdicts of this repository's workflows equal SWALLOW-1's self-census
     (33 PROPAGATES; the 5 TOOLLESS steps are not fault sites and do not appear);
  2. gauntlet-pr.yml as it stands on `main` (`sort -u || true`): with the discover step's tools
     failing, the two steps gated on its output are skipped and the job is green -- the mechanism
     of #137, seen by simulation rather than by 138 runs;
  3. the same file as #137 leaves it: the same fault is RED.
The verdict the instrument gives the `main` fault is also recorded: by the check rule the
gauntlet step is not a recognised check, so the instrument says NO_CHECK, and the RESULT says so.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.harness_mutation import faults as F  # noqa: E402

OUT = HERE / "swallow2_self.json"
SELF_CENSUS = HERE / "swallow1_self_census.json"


def discover_fault(path: Path) -> dict:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    job = "verify-submissions"
    idx = next(i for i, st in enumerate(doc["jobs"][job]["steps"]) if st.get("id") == "discover")
    runner = F.Runner()
    plus = F.simulate(doc, runner, None, "x")
    minus = F.simulate(doc, runner, (job, idx), "x", artifacts=F._artifacts(plus))
    def steps(sim):
        return [{"index": s["index"], "name": s["name"], "ran": s["ran"], "why": s["why"], "exit": s["exit"]} for s in sim[job]["steps"]]
    alone = {(job, i): F.alone_verdict(st["run"], runner) for i, st in enumerate(doc["jobs"][job]["steps"]) if isinstance(st.get("run"), str)}
    faults, _ = F.analyse_workflow(doc, path.name, runner, alone)
    verdict = next(f for f in faults if f["job"] == job and f["index"] == idx)
    return {"file": str(path), "discover_index": idx, "discover_alone": alone[(job, idx)],
            "plus": {"result": plus[job]["result"], "steps": steps(plus)},
            "minus_discover": {"result": minus[job]["result"], "steps": steps(minus)},
            "instrument_verdict": verdict["verdict"], "dropped": verdict["dropped"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, help="gauntlet-pr.yml as on main (sort -u || true)")
    ap.add_argument("--after", required=True, help="gauntlet-pr.yml as #137 leaves it")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        rec_a = F.analyse_tree(ROOT)
        rec_b = F.analyse_tree(ROOT)
    same = json.dumps(rec_a["faults"], sort_keys=True) == json.dumps(rec_b["faults"], sort_keys=True)
    alone = {(f["workflow"], f["job"], f["index"]): f["alone"] for f in rec_a["faults"]}
    census = json.loads(SELF_CENSUS.read_text(encoding="utf-8"))
    old = {(s["workflow"], s["job"], s["index"]): s["verdict"] for s in census["steps"]}
    differ = sorted(f"{k[0]}::{k[1]}::{k[2]} {old[k]} -> {alone[k]}" for k in alone if k in old and old[k] != alone[k])
    missing = sorted(f"{k[0]}::{k[1]}::{k[2]} ({old[k]})" for k in old if k not in alone and old[k] in ("PROPAGATES", "SWALLOWS"))
    before, after = discover_fault(Path(a.before)), discover_fault(Path(a.after))
    gated = [s for s in before["minus_discover"]["steps"] if s["index"] in (before["discover_index"] + 1, before["discover_index"] + 2)]
    payload = {
        "what": "the fault instrument on this repository and on gauntlet-pr.yml before and after #137",
        "self": {"fault_sites": len(rec_a["faults"]), "by_verdict": F._count(f["verdict"] for f in rec_a["faults"]),
                 "alone_by_verdict": F._count(alone.values()), "deterministic": same,
                 "alone_differs_from_swallow1": differ, "swallow1_sites_missing": missing},
        "gauntlet_before_137": before, "gauntlet_after_137": after,
        "gate": {"alone_reproduces_swallow1": not differ and not missing, "deterministic": same,
                 "before_137_gated_steps_skipped_and_green": before["minus_discover"]["result"] == "success" and all(not s["ran"] and s["why"] == "if" for s in gated),
                 "after_137_discover_is_red": after["instrument_verdict"] == "RED"},
    }
    payload["pass"] = all(payload["gate"].values())
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"gate": payload["gate"], "pass": payload["pass"], "self": payload["self"],
                      "before_verdict": before["instrument_verdict"], "after_verdict": after["instrument_verdict"]}, indent=1))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
