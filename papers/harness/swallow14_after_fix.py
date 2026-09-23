"""SWALLOW-14, after the run: the stage with the change made after scoring (wait-list reads a script
whose backslash-continued line comes before a second `< <(` list in place, instead of raising; an
edit that cannot read a script does not apply to it, and the audit goes on; the wait line at the
statement's indent), run again on what the scored run read. Not a result: what the change moves,
next to what was scored.

  - every check of the receipt's stage-3 population: stage 3 again, from the same tip
  - every repository the run could not audit because the stage raised: the instrument's
    `audit_one`, unchanged, with the changed stage -- what those repositories hold

Each repository is fetched again at the receipt's tip, in a process of its own. That process runs
the workflow's steps' shell with their tools stubbed, and what the shell does -- a redirect, a
`mkdir`, an `rm` -- acts on the machine it runs on (RESULT_swallow14 §0): run it where that is safe.
The scored run's processes each ran as an unprivileged user of their own (`swallow14_sandbox.sh`).

    python papers/harness/swallow14_after_fix.py --work <dir>          # writes swallow14_after_fix.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

RECEIPT = HERE / "swallow14_receipt.json.gz"
OUT = HERE / "swallow14_after_fix.json"
RAISED = ("IndexError: string index out of range", "IndexError: list index out of range")   # wait-list's, read by traceback


def one(repo: str, tip: str, work: Path, checks: list[dict], whole: bool) -> dict:
    from benchmarks.harness_mutation import empty_list as EL
    from styxx.ciaudit import engine
    from styxx.ciaudit import repair_frontier as F
    tree, info = EL.fetch(repo, tip, work)
    res: dict = {"repo": repo, "tip": tip, "fetch": info}
    if tree is None:
        return res
    try:
        rows = []
        for t in checks:
            text = (tree / ".github" / "workflows" / t["workflow"]).read_text(encoding="utf-8", errors="replace")
            again = F.try_frontier(text, t["workflow"], t["job"], t["index"], engine.Runner())
            rows.append({"workflow": t["workflow"], "job": t["job"], "index": t["index"], "verified_repair": again["verified_repair"],
                         "candidates": [{k: c.get(k) for k in ("repair", "applies", "why", "loud", "unchanged", "verified", "verdict_after",
                                                                "lines_changed", "diff")} for c in again["candidates"]]})
        res["checks"] = rows
        if whole:
            res["audit"] = EL.audit_one(tree, repo)
    except Exception as e:  # noqa: BLE001 -- recorded
        res["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    shutil.rmtree(tree, ignore_errors=True)
    return res


def _run_one(repo: str, tip: str, checks: list[dict], whole: bool, work: Path, out_dir: Path) -> dict:
    path = out_dir / (repo.replace("/", "__") + ".json")
    if not path.exists():
        spec = work / (repo.replace("/", "__") + ".spec.json")
        spec.write_text(json.dumps({"checks": checks, "whole": whole}), encoding="utf-8")
        p = subprocess.run([sys.executable, "-m", "papers.harness.swallow14_after_fix", "--one", repo, tip, "--work", str(work / "clones"),
                            "--json", str(path), "--spec", str(spec)], cwd=str(ROOT), capture_output=True, text=True, timeout=1800)
        if p.returncode != 0 or not path.exists():
            path.write_text(json.dumps({"repo": repo, "tip": tip, "error": "process: " + ((p.stderr or "").strip().splitlines() or ["?"])[-1][:200]}),
                            encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path)
    ap.add_argument("--one", nargs=2)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--spec", type=Path)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    if a.one:
        spec = json.loads(a.spec.read_text(encoding="utf-8"))
        a.json.write_text(json.dumps(one(a.one[0], a.one[1], a.work, spec["checks"], spec["whole"])), encoding="utf-8")
        return 0
    raw = RECEIPT.read_bytes()
    r = json.loads(gzip.decompress(raw).decode("utf-8"))
    work = a.work.resolve()
    for d in (work / "clones", work / "after"):         # writable by each repository's process, whichever user it runs as
        d.mkdir(parents=True, exist_ok=True)
        d.chmod(0o1777)
    jobs = []
    for rep in r["repos"]:
        pop3 = [t for t in rep.get("targets", []) if t["tried_stage3"] and t["baseline"] is not None]
        raised = rep.get("error") in RAISED
        if pop3 or raised:
            jobs.append((rep, pop3, raised))
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        done = list(ex.map(lambda j: _run_one(j[0]["repo"], j[0]["tip"], [{k: t[k] for k in ("workflow", "job", "index")} for t in j[1]],
                                              j[2], work, work / "after"), jobs))
    rows, raised_rows = [], []
    for (rep, pop3, raised), got in zip(jobs, done):
        again = {(c["workflow"], c["job"], c["index"]): c for c in got.get("checks", [])}
        for t in pop3:
            g = again.get((t["workflow"], t["job"], t["index"]))
            was = t["verified_repair"] if t["stage"] == "swallow-13" else None
            n12 = len(t["candidates"]) - len(g["candidates"]) if g else None
            was_diff = next((c.get("diff") for c in t["candidates"] if c["repair"] == was), None) if was else None
            now = g["verified_repair"] if g else None
            now_diff = next((c.get("diff") for c in g["candidates"] if c["repair"] == now), None) if g and now else None
            moved = [c["repair"] for c, d in zip(t["candidates"][n12:], g["candidates"])
                     if (c.get("applies"), c.get("verified")) != (d.get("applies"), d.get("verified"))] if g else None
            rows.append({"repo": rep["repo"], "workflow": t["workflow"], "job": t["job"], "index": t["index"], "name": t["name"],
                         "read_again": g is not None, "scored": was, "after": now, "same_repair": g is not None and was == now,
                         "same_diff": g is not None and was_diff == now_diff, "candidates_moved": moved,
                         "diff_after": now_diff if g is not None and was_diff != now_diff else None})
        if raised:
            au = got.get("audit") or {}
            raised_rows.append({"repo": rep["repo"], "tip": rep["tip"], "error_after": got.get("error"), "fetch": got.get("fetch"),
                                "workflows": au.get("workflows"), "targets": [{k: x.get(k) for k in ("workflow", "job", "index", "name", "verdict",
                                                                                                   "baseline", "verified_repair", "stage",
                                                                                                   "tried_stage3")} for x in au.get("targets", [])]})
    out = {"receipt_sha256": hashlib.sha256(raw).hexdigest(),
           "repair_frontier_sha256": hashlib.sha256((ROOT / "styxx" / "ciaudit" / "repair_frontier.py").read_bytes()).hexdigest(),
           "checks": len(rows), "read_again": sum(1 for x in rows if x["read_again"]),
           "verified_scored": sum(1 for x in rows if x["scored"]), "verified_after": sum(1 for x in rows if x["after"]),
           "repair_changed": [x for x in rows if x["read_again"] and not x["same_repair"]],
           "diff_changed": [x for x in rows if x["same_repair"] and not x["same_diff"]],
           "candidates_moved": [x for x in rows if x["candidates_moved"]],
           "raised": raised_rows, "rows": rows}
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in out.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
