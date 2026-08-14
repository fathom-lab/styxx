"""Freeze today's dead-term rates so the retraction question becomes answerable later.

The pre-registered test of "do dead instruments predict retracted claims" returned a
null (p = 0.248), and its own limits section named the only design that could settle it:

    A prospective design -- freeze the dead rates, then wait to see which claims get
    retracted -- is the only version that separates [the attention confound], and it is
    the obvious successor.

That successor was named and not built, which is the difference between a limitation and
an excuse. This builds it. It costs one run today and makes a question answerable in six
months that is otherwise unanswerable forever, because the rates will have moved and
nobody will be able to say what they were when the claims were still standing.

What it records, per module: the adjudicative dead rate, the exercised fraction, the term
counts, the git HEAD of both the subject and the instrument, and the UTC timestamp. The
exposure variable does not exist yet -- that is the point. Retractions that happen AFTER
this snapshot are the ones that count, and they cannot be selected to fit the rates
because the rates are already written down and pushed.

    python freeze_prospective_baseline.py --probe probe_e_styxx_v2.json \
        --out PROSPECTIVE_BASELINE_2026_08_14.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import defaultdict


def git_head(repo, path=None):
    cmd = ["git", "-C", repo, "log", "-1", "--format=%H"]
    if path:
        cmd += ["--", path]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                              shell=True).stdout.strip() or None
    except Exception:                                        # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--repo", default=r"C:\Users\heyzo\clawd\styxx")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(a.probe, encoding="utf-8") as f:
        rep = json.load(f)

    by = defaultdict(lambda: {"adj_powered": 0, "adj_dead": 0, "terms": 0,
                              "powered": 0})
    for r in rep.get("rows", []):
        d = by[r.get("module") or "?"]
        d["terms"] += 1
        powered = r["verdict"] in ("LIVE", "CONSTANT_TRUE", "CONSTANT_FALSE")
        d["powered"] += powered
        if r.get("pos") == "adjudicative" and powered:
            d["adj_powered"] += 1
            d["adj_dead"] += r["verdict"] != "LIVE"

    modules = {m: {"dead_rate_adjudicative": (d["adj_dead"] / d["adj_powered"]
                                              if d["adj_powered"] else None),
                   "adj_powered": d["adj_powered"], "adj_dead": d["adj_dead"],
                   "exercised_frac": d["powered"] / d["terms"] if d["terms"] else None,
                   "terms": d["terms"]}
               for m, d in by.items()}

    snap = {
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": (
            "PROSPECTIVE baseline for the retraction hypothesis. The retrospective test "
            "(FINDING_prereg_retraction_null_2026_08_13.md) found no association and "
            "could not rule out the attention confound: retracted claims were "
            "investigated, which is how they came to be retracted, and investigation "
            "may drive both the retraction and the discovery of dead gates nearby. "
            "Only claims retracted AFTER this timestamp count toward the prospective "
            "test."),
        "how_to_use": (
            "When a claim is retracted, struck, voided or withdrawn after frozen_utc, "
            "record it and the modules on its causal path. Once MIN_MODULES=5 such "
            "modules accumulate, run the SAME analysis "
            "(analyze_retraction_falsifiability.py) against THIS file rather than a "
            "fresh probe run. Using a fresh run would reintroduce the confound: the "
            "rates would already reflect whatever attention the retraction caused."),
        "subject_repo_head": git_head(a.repo),
        "instrument_head": git_head(a.repo,
                                    "papers/dogfood-self-audit/probe_e_runtime.py"),
        "probe_source": os.path.basename(a.probe),
        "n_modules": len(modules),
        "n_modules_with_powered_adjudicative": sum(
            1 for v in modules.values() if v["adj_powered"]),
        "repo_totals": {k: rep.get(k) for k in
                        ("n_terms_instrumented", "n_powered",
                         "n_adjudicative_powered", "n_adjudicative_dead",
                         "dead_rate_adjudicative")},
        "preregistered_prediction": (
            "H1: modules whose dead rate is HIGH in this snapshot will be "
            "over-represented among the causal paths of claims retracted after "
            "frozen_utc. The retrospective test found the opposite direction "
            "(exposed median 0.333 vs 0.369); the author's expectation is that the "
            "prospective test will also find no association, and this prediction is "
            "recorded now so that it cannot be revised when the data arrives."),
        "modules": modules,
    }
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(snap, f, indent=1)

    print(f"  frozen at {snap['frozen_utc']}")
    print(f"  modules                       : {snap['n_modules']}")
    print(f"  with powered adjudicative terms: "
          f"{snap['n_modules_with_powered_adjudicative']}")
    print(f"  subject HEAD    : {str(snap['subject_repo_head'])[:12]}")
    print(f"  instrument HEAD : {str(snap['instrument_head'])[:12]}")
    print(f"  repo dead rate  : {snap['repo_totals']['dead_rate_adjudicative']}")
    print(f"\n  wrote {a.out} — only retractions AFTER this timestamp count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
