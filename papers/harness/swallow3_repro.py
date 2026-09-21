# -*- coding: utf-8 -*-
"""G-S3-3 failed: eight of 30,642 SWALLOW-2 records did not reproduce from a re-clone at the same
HEADs. This script demonstrates each cause on the step that showed it, and writes the evidence to
`swallow3_repro.json`. All three causes are in the frozen instrument's contact with the world, not
in the catalogue; none is a timeout, so the frozen gate does not excuse them. A fourth mechanism
(real /tmp state leaking between runs) is shown on the same step it was found on, and did not by
itself change a record.

    python papers/harness/swallow3_repro.py --clones <clones>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


def _x(run: str) -> str:
    return re.sub(r"\$\{\{.*?\}\}", "x", run)


def hash_seed_trials(step_run: str, seeds=range(8)) -> list[dict]:
    """The same step under different PYTHONHASHSEEDs: the stub order is a set's iteration order
    when two stubs tie on depth, and the alone verdict follows it."""
    prog = f"""
import sys, re, json; sys.path.insert(0, {str(ROOT)!r})
from benchmarks.harness_mutation import faults
run = {step_run!r}
text = re.sub(r"\\$\\{{\\{{.*?\\}}\\}}", "x", run)
order = sorted(faults._relative_commands(text), key=lambda r: -r.count("/"))
r = faults.Runner()
print(json.dumps({{"stub_order": order, "alone": faults.alone_verdict(run, r), "n_reached": r.run(text, "fail", {{}})["n_reached"]}}))
"""
    out = []
    for seed in seeds:
        env = dict(os.environ, PYTHONHASHSEED=str(seed))
        p = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=str(ROOT), timeout=120)
        out.append({"seed": seed, **(json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 and p.stdout.strip() else {"error": p.stderr[-300:]})})
    return out


def main(argv=None) -> int:
    import yaml
    from benchmarks.harness_mutation import faults
    ap = argparse.ArgumentParser()
    ap.add_argument("--clones", required=True)
    ap.add_argument("--out", default=str(HERE / "swallow3_repro.json"))
    a = ap.parse_args(argv)
    clones = Path(a.clones)
    ev: dict = {"schema": "styxx.swallow3-repro/v1", "instrument": "benchmarks/harness_mutation/faults.py (frozen, d26a407c…)"}

    # 1. a tie in the stub order, resolved by set iteration order (browser-use/browser-use test.yaml › tests › 9)
    d = yaml.safe_load((clones / "browser-use__browser-use" / ".github" / "workflows" / "test.yaml").read_text(encoding="utf-8"))
    st = d["jobs"]["tests"]["steps"][9]
    trials = hash_seed_trials(st["run"])
    ev["stub_order_tie"] = {"step": "browser-use/browser-use test.yaml › tests › 9 'Check if test file exists'",
                            "stubs": sorted(faults._relative_commands(_x(st["run"]))),
                            "trials": trials, "outcomes": sorted({t.get("alone") for t in trials if "alone" in t}),
                            "reading": "two stubs of equal depth (tests/ci/ from the sed pattern, tests/ci/x.py from the assignment); created in set order; "
                                       "when the file comes first the step finds it and reaches nothing (TOOLLESS: not a fault site), when the directory "
                                       "comes first the file is not created and the step reaches find/sed/sort (PROPAGATES: a fault site)"}

    # 2. the real `date` (airbytehq/airbyte ai-ready-command.yml › validate › 3 'Check for weekend freeze')
    d = yaml.safe_load((clones / "airbytehq__airbyte" / ".github" / "workflows" / "ai-ready-command.yml").read_text(encoding="utf-8"))
    st = d["jobs"]["validate"]["steps"][3]
    r = faults.Runner()
    now = r.run(_x(st["run"]), "x", {})
    git = subprocess.run(["git", "log", "-1", "--format=%cI", "--", "papers/harness/swallow2_receipt.json.gz"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", cwd=str(ROOT))
    ev["wall_clock"] = {"step": "airbytehq/airbyte ai-ready-command.yml › validate › 3 'Check for weekend freeze'",
                        "script_head": st["run"].strip().splitlines()[1][:80],
                        "now_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "now_pacific_dow": subprocess.run(["date", "+%u %A"], capture_output=True, text=True, env=dict(os.environ, TZ="America/Los_Angeles")).stdout.strip(),
                        "blocked_now": "blocked=true" in now["tail"], "step_outputs_now": now["outputs"],
                        "swallow2_receipt_committed": git.stdout.strip(),
                        "reading": "`date` is real in the healthy worlds; the step decides by the Pacific day of week; SWALLOW-2's receipt was made on a "
                                   "Sunday evening Pacific and its 'Apply ready label' step was skipped (BASELINE_SKIPPED); this run was made on the "
                                   "Monday and the same step runs (RED)"}

    # 3. a background subshell racing the stub log (langfuse/langfuse pipeline.yml › tests-web › 1)
    d = yaml.safe_load((clones / "langfuse__langfuse" / ".github" / "workflows" / "pipeline.yml").read_text(encoding="utf-8"))
    st = d["jobs"]["tests-web"]["steps"][1]
    verdicts = []
    for _ in range(12):
        r = faults.Runner()
        x = r.run(_x(st["run"]), "fail", {})
        verdicts.append((faults.alone_verdict(st["run"], r), x["n_reached"]))
    ev["background_race"] = {"step": "langfuse/langfuse pipeline.yml › tests-web › 1 'Install golang-migrate for Clickhouse migrations in background'",
                             "script_shape": "( set -e; curl …; tar …; sudo mv …; touch … ) > /tmp/migrate-install.log 2>&1 &",
                             "trials_now": verdicts, "in_the_run": "SWALLOWS (a fault site) in all three jobs; absent from SWALLOW-2's receipt",
                             "reading": "the subshell runs in the background; whether its first command has written to the stub log when the script "
                                        "exits and the log is read depends on scheduling"}

    # 4. real /tmp state leaking across executions and runs (githubnext/gh-aw visual-regression-checker.lock.yml › agent › 12)
    d = yaml.safe_load((clones / "githubnext__gh-aw" / ".github" / "workflows" / "visual-regression-checker.lock.yml").read_text(encoding="utf-8"))
    st = d["jobs"]["agent"]["steps"][12]
    tmp = Path("/tmp/gh-aw/agent")
    present = tmp.exists()
    r = faults.Runner()
    with_dir = (faults.alone_verdict(st["run"], r), r.run(_x(st["run"]), "fail", {})["n_reached"]) if present else None
    moved = None
    if present:
        moved = Path("/tmp/gh-aw-moved-by-swallow3-repro")
        os.rename("/tmp/gh-aw", moved)
    try:
        r = faults.Runner()
        without = (faults.alone_verdict(st["run"], r), r.run(_x(st["run"]), "fail", {})["n_reached"])
    finally:
        if moved is not None:
            os.rename(moved, "/tmp/gh-aw")
    ev["background_race_2"] = {"step": "githubnext/gh-aw visual-regression-checker.lock.yml › agent › 12 'Start docs server'",
                               "script_head": st["run"].strip().splitlines()[0][:100],
                               "in_the_run": "absent (not a fault site); SWALLOWS in SWALLOW-2's receipt",
                               "trials_now_with_dir": with_dir,
                               "reading": "`nohup npm … &` is a background command; in the fail world the not-found handler that logs `nohup` runs in "
                                          "the background subshell, and whether it has written to the stub log when the script exits and the log is "
                                          "read depends on scheduling: reached (SWALLOWS, a fault site) or not (TOOLLESS, absent). Same cause as "
                                          "langfuse's; the run was under load, the trials here are not"}
    ev["tmp_leak_observed"] = {"step": "the same step", "/tmp/gh-aw/agent_present_before": present,
                               "alone_with_dir": with_dir, "alone_without_dir": without,
                               "reading": "a second mechanism seen on the same step, not the cause of this record: the script redirects into "
                                          "/tmp/gh-aw/agent/, an absolute path the sandbox does not cover, and `mkdir -p` is real in the healthy "
                                          "worlds, so an earlier step of an earlier run creates it on the machine; with the directory npm is reached "
                                          "(SWALLOWS), without it the redirect fails and the -e script stops (PROPAGATES) -- a fault site either way"}
    Path(a.out).write_text(json.dumps(ev, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: (v if k == "schema" else {kk: vv for kk, vv in v.items() if kk in ("outcomes", "blocked_now", "now_pacific_dow", "trials_now", "trials_now_with_dir", "alone_with_dir", "alone_without_dir")})
                      for k, v in ev.items() if k != "instrument"}, indent=1, default=str)[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
