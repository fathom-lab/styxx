# -*- coding: utf-8 -*-
"""SWALLOW-13 -- the frontier: the hidden checks the first two repair stages leave, read on
repositories no repair stage was designed on.

`styxx ci-audit --repair` verifies a repair in two stages: SWALLOW-4's (remove the step's
`continue-on-error`; make its shell strict) and SWALLOW-5's (a guard whose failing tool is not its
green path; a query without its `|| echo` default). SWALLOW-13 adds a third, designed on the 49
checks of SWALLOW-11 that neither verifies (`styxx/ciaudit/repair_frontier.py`): a `$(...)` whose
status its line throws away hoisted onto a line of its own, a background job whose early death
nobody waits for checked once, `|| exit 0` removed, a fallback hidden by a continued line, each
alone or with the `continue-on-error` removed; and, for what none repairs, two readings of the
script -- routed (a flag of the failure written to GITHUB_ENV or GITHUB_OUTPUT that a later step
reads) and declared (the script says a failure is not fatal).

This instrument runs the product's `--repair` path, unchanged, on a held-out population:
SWALLOW-9's repositories at the default-branch tip SWALLOW-9 recorded, minus every repository an
earlier cycle's repairs or this cycle's development set read (SWALLOW-1's hundred, which SWALLOW-2
to -7 read; SWALLOW-11's sixty, which hold the development set). For each repository:

  1. `.github/workflows` at the recorded tip: a sparse, blob-less fetch of that one commit
  2. the product's audit, `styxx.ciaudit.engine.analyse_tree` (actions counted), with a deadline
  3. every hand-written finding (SWALLOWED, FAIL_OPEN) through `styxx.ciaudit.repair.repair_faults`
     -- stage 1, then 2, then 3, each verified: loud under the same fault, the healthy run
     unchanged; for what none verifies, the two readings
  4. stage 3 once more, with a fresh runner, on every check it was tried on (the determinism gate)

Each repository runs in its own process under a hard time limit; a repository that exceeds it is
recorded, not retried. The receipt keeps, per check, the step's place, verdict, first line, the
sha256 of its script, every candidate of every stage with its verdict and diff, the stage that
verified, the readings, and the re-run's agreement; for a check no stage repairs, the script.

    python -m benchmarks.harness_mutation.frontier --build-population --out papers/harness/swallow13_population.json
    python -m benchmarks.harness_mutation.frontier --population papers/harness/swallow13_population.json \\
        --work <dir> --out papers/harness/swallow13_receipt.json.gz --workers 2
    python -m benchmarks.harness_mutation.frontier --tree <checkout>          # one checkout, as it is
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
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from styxx.ciaudit import engine                        # noqa: E402  -- the product, as it ships
from styxx.ciaudit import repair as R                   # noqa: E402
from styxx.ciaudit import repair_frontier as F          # noqa: E402
from styxx.ciaudit import repair_structural as RS       # noqa: E402

SCHEMA = "styxx.harness-frontier/v1"
DEADLINE = 900.0          # seconds of analysis per repository; a capped repository says so
TIMEOUT = 3600            # seconds per repository, all of it; past it the repository is recorded as timed out
HARNESS = ROOT / "papers" / "harness"
PRODUCT = ("styxx/ciaudit/engine.py", "styxx/ciaudit/actions.py", "styxx/ciaudit/repair.py", "styxx/ciaudit/repair_structural.py",
           "styxx/ciaudit/repair_frontier.py")
FAMILY = {"hoist-substitution": "hoist", "no-coe+hoist-substitution": "hoist", "background-liveness": "background",
          "no-coe+background-liveness": "background", "no-exit-zero": "exit-zero", "no-coe+no-exit-zero": "exit-zero",
          "no-default-joined": "default-joined", "no-coe+no-default-joined": "default-joined", "no-coe+no-default": "no-coe+no-default",
          "no-coe+guard-status": "no-coe+guard-status"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def instrument_sha256() -> str:
    return _sha(Path(__file__).read_bytes())


def product_sha256() -> dict:
    return {p: _sha((ROOT / p).read_bytes()) for p in PRODUCT}


def _load(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads((gzip.decompress(raw) if path.suffix == ".gz" else raw).decode("utf-8")), _sha(raw)


# ----------------------------------------------------------------------------- the population

def build_population() -> dict:
    """SWALLOW-9's repositories with the default-branch tip it recorded, minus SWALLOW-1's hundred
    and SWALLOW-11's sixty. Sorted by name."""
    r9, h9 = _load(HARNESS / "swallow9_receipt.json.gz")
    s1, h1 = _load(HARNESS / "swallow1_population.json")
    r11, h11 = _load(HARNESS / "swallow11_receipt.json.gz")
    out1 = {x["repo"] for x in s1}
    out11 = {p["repo"] for p in r11["pairs"]}
    tips = {r["repo"]: r["clone"]["tip"] for r in r9["repos"] if (r.get("clone") or {}).get("tip")}
    repos = [{"repo": k, "tip": tips[k]} for k in sorted(tips) if k not in out1 and k not in out11]
    return {"schema": SCHEMA + "#population", "rule": "SWALLOW-9's repositories at the default-branch tip SWALLOW-9 recorded, minus "
            "SWALLOW-1's hundred (read by SWALLOW-2 to -7) and SWALLOW-11's sixty (which hold SWALLOW-13's development set)",
            "sources_sha256": {"swallow9_receipt.json.gz": h9, "swallow1_population.json": h1, "swallow11_receipt.json.gz": h11},
            "swallow9_repos": len(tips), "excluded_swallow1": sum(1 for k in tips if k in out1),
            "excluded_swallow11": sum(1 for k in tips if k in out11 and k not in out1), "count": len(repos), "repos": repos}


# ----------------------------------------------------------------------------- one repository

def _git(args: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def fetch(repo: str, tip: str, work: Path) -> tuple[Path | None, dict]:
    """`.github/workflows` at one commit: init, sparse, fetch that commit at depth 1 without blobs
    outside the sparse set, check it out."""
    dest = work / repo.replace("/", "__")
    info: dict = {"tip": tip}
    t0 = time.time()
    if dest.exists():
        shutil.rmtree(dest)
    try:
        for args in (["init", "-q", str(dest)], ["-C", str(dest), "remote", "add", "origin", f"https://github.com/{repo}.git"],
                     ["-C", str(dest), "sparse-checkout", "set", ".github/workflows"],
                     ["-C", str(dest), "fetch", "-q", "--depth", "1", "--filter=blob:none", "--no-tags", "origin", tip],
                     ["-C", str(dest), "-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD"]):
            p = _git(args)
            if p.returncode != 0:
                info["error"] = f"{args[2] if args[0] == '-C' else args[0]}: " + (p.stderr.strip().splitlines() or [f"exit {p.returncode}"])[-1][:200]
                return None, info
    except subprocess.TimeoutExpired:
        info["error"] = "timeout"
        return None, info
    head = _git(["-C", str(dest), "rev-parse", "HEAD"]).stdout.strip()
    if head != tip:
        info["error"] = f"checked out {head[:12]}, not the tip"
        return None, info
    info["seconds"] = round(time.time() - t0, 1)
    return dest, info


def _stage(name: str | None) -> str | None:
    if name is None:
        return None
    if name in R.REPAIRS:
        return "swallow-4"
    if name in RS.REPAIRS:
        return "swallow-5"
    if name in F.REPAIRS:
        return "swallow-13"
    return "?"


def _cand(c: dict) -> dict:
    keep = ("repair", "applies", "why", "loud", "unchanged", "verified", "verdict_after", "by_flavour_after", "lines_changed", "diff")
    return {k: c[k] for k in keep if k in c}


def _outcome(cands: list[dict]) -> list:
    return [(c["repair"], c.get("applies"), c.get("loud"), c.get("unchanged"), c.get("verified"), c.get("verdict_after"), c.get("diff")) for c in cands]


def audit_one(tree: Path, repo: str | None = None) -> dict:
    """The product's audit and `--repair` path on one checkout, hand-written findings only."""
    t0 = time.time()
    rec = engine.analyse_tree(tree, repo, deadline=t0 + DEADLINE, actions=True)
    found = [f for f in rec["faults"] if f["verdict"] in R.TARGET_VERDICTS]
    hand = [f for f in found if not f.get("generated")]
    out = {"workflows": rec["workflows"], "capped": rec["capped"], "unparseable": len(rec["unparseable"]),
           "fault_sites": len(rec["faults"]), "findings": len(found), "findings_generated": len(found) - len(hand),
           "seconds_audit": round(time.time() - t0, 1), "targets": []}
    t1 = time.time()
    texts: dict = {}
    for t in R.repair_faults(tree, hand):
        wf = t["workflow"]
        if wf not in texts:
            texts[wf] = (tree / ".github" / "workflows" / wf).read_text(encoding="utf-8", errors="replace")
        text = texts[wf]
        pos = R.locate(text, t["job"], t["index"]) or {}
        run = (pos.get("run") or {}).get("value")
        tried3 = len(t["candidates"]) > len(R.REPAIRS) + len(RS.REPAIRS)
        rec_t = {"workflow": wf, "job": t["job"], "index": t["index"], "name": t.get("name"), "verdict": t["verdict"],
                 "baseline": t["baseline"].get("verdict"), "continue_on_error": t.get("continue_on_error"), "run_head": t.get("run_head"),
                 "run_sha256": _sha(run.encode("utf-8")) if run is not None else None,
                 "verified_repair": t["verified_repair"], "stage": _stage(t["verified_repair"]), "tried_stage3": tried3,
                 "candidates": [_cand(c) for c in t["candidates"]], "readings": t.get("readings")}
        if tried3:
            again = F.try_frontier(text, wf, t["job"], t["index"], engine.Runner())
            first3 = t["candidates"][len(R.REPAIRS) + len(RS.REPAIRS):]
            rec_t["stage3_again_equal"] = (_outcome(again["candidates"]) == _outcome(first3)
                                           and again["verified_repair"] == (t["verified_repair"] if rec_t["stage"] == "swallow-13" else None))
            if t["verified_repair"] is None:
                rec_t["run"] = run
        out["targets"].append(rec_t)
    out["seconds_repair"] = round(time.time() - t1, 1)
    return out


def one(repo: str, tip: str, work: Path) -> dict:
    t0 = time.time()
    tree, info = fetch(repo, tip, work)
    if tree is None:                                   # the network, once more, after a pause
        time.sleep(5)
        tree, again = fetch(repo, tip, work)
        info = dict(again, first_error=info.get("error"))
    res = {"repo": repo, "tip": tip, "fetch": info}
    if tree is not None:
        try:
            res.update(audit_one(tree, repo))
        except Exception as e:  # noqa: BLE001 -- recorded, never a silent pass
            res["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ----------------------------------------------------------------------------- the population, in parallel

def _run_one(repo: str, tip: str, work: Path, res_dir: Path) -> dict:
    path = res_dir / (repo.replace("/", "__") + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-m", "benchmarks.harness_mutation.frontier", "--one", repo, tip, "--work", str(work),
                            "--json", str(path)], cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=TIMEOUT)
        if p.returncode != 0 or not path.exists():
            res = {"repo": repo, "tip": tip, "error": "process: " + ((p.stderr or "").strip().splitlines() or [f"exit {p.returncode}"])[-1][:200],
                   "seconds": round(time.time() - t0, 1)}
            path.write_text(json.dumps(res), encoding="utf-8")
    except subprocess.TimeoutExpired:
        res = {"repo": repo, "tip": tip, "error": "timeout", "seconds": TIMEOUT}
        path.write_text(json.dumps(res), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def run(population: Path, work: Path, out: Path, workers: int = 2) -> dict:
    pop, pop_sha = _load(population)
    work.mkdir(parents=True, exist_ok=True)
    res_dir = work / "results"
    res_dir.mkdir(exist_ok=True)
    t0 = time.time()
    results: dict = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one, r["repo"], r["tip"], work / "clones", res_dir): r["repo"] for r in pop["repos"]}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            r = fut.result()
            results[futs[fut]] = r
            nt = len(r.get("targets", []))
            print(f"[{n}/{len(futs)}] {r['repo']} targets={nt} {r.get('error') or ''} {r.get('seconds', 0)}s total={time.time() - t0:.0f}s",
                  file=sys.stderr, flush=True)
    repos = [results[r["repo"]] for r in pop["repos"]]
    receipt = {"schema": SCHEMA, "instrument": "benchmarks/harness_mutation/frontier.py", "instrument_sha256": instrument_sha256(),
               "product_sha256": product_sha256(), "population_file": str(population.relative_to(ROOT)) if population.is_relative_to(ROOT) else str(population),
               "population_sha256": pop_sha, "deadline_per_repo": DEADLINE, "timeout_per_repo": TIMEOUT, "workers": workers,
               "seconds": round(time.time() - t0, 1), "repos": repos, "summary": summary(repos)}
    raw = (json.dumps(receipt, indent=1, sort_keys=True) + "\n").encode("utf-8")
    out.write_bytes(gzip.compress(raw, mtime=0) if out.suffix == ".gz" else raw)
    return receipt


# ----------------------------------------------------------------------------- the counts

def _count(it) -> dict:
    out: dict = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], str(kv[0]))))


def summary(repos: list[dict]) -> dict:
    fetched = [r for r in repos if "targets" in r]
    ts = [dict(t, repo=r["repo"]) for r in fetched for t in r["targets"]]
    read = [t for t in ts if t["baseline"] is not None]
    s1 = [t for t in read if t["stage"] == "swallow-4"]
    s2 = [t for t in read if t["stage"] == "swallow-5"]
    pop3 = [t for t in read if t["tried_stage3"]]
    s3 = [t for t in pop3 if t["stage"] == "swallow-13"]
    residue = [t for t in pop3 if t["verified_repair"] is None]
    routed = [t for t in residue if (t.get("readings") or {}).get("routed")]
    declared = [t for t in residue if (t.get("readings") or {}).get("declared")]
    either = [t for t in residue if t.get("readings")]
    n_stage3 = len(R.REPAIRS) + len(RS.REPAIRS)
    loud_changed = [c for t in pop3 for c in t["candidates"][n_stage3:] if c.get("applies") and c.get("loud") and c.get("unchanged") is False]

    def distinct(sel):
        return len({(t["repo"], t["workflow"], t["run_sha256"], bool(t["continue_on_error"])) for t in sel})

    lines = sorted(next(c["lines_changed"] for c in t["candidates"] if c["repair"] == t["verified_repair"]) for t in s3)
    return {
        "repos": len(repos), "fetched": len(fetched), "fetch_failed": sum(1 for r in repos if r.get("fetch", {}).get("error")),
        "errors": sum(1 for r in repos if r.get("error")), "capped": sum(1 for r in fetched if r.get("capped")),
        "repos_with_targets": sum(1 for r in fetched if r["targets"]), "workflows": sum(r.get("workflows", 0) for r in fetched),
        "targets": len(ts), "not_a_fault_site_on_repair": len(ts) - len(read), "read": len(read),
        "stage1_verified": len(s1), "stage2_verified": len(s2), "stage3_population": len(pop3), "stage3_verified": len(s3),
        "stage3_by_repair": _count(t["verified_repair"] for t in s3), "stage3_by_family": _count(FAMILY[t["verified_repair"]] for t in s3),
        "stage3_repos": len({t["repo"] for t in s3}), "stage3_lines_changed": lines,
        "stage3_candidates_loud_but_change_the_healthy_run": len(loud_changed),
        "all_stages_verified": len(s1) + len(s2) + len(s3),
        "residue": len(residue), "residue_routed": len(routed), "residue_routed_a_reader_can_fail": sum(1 for t in routed if t["readings"]["routed"]["a_reader_can_fail"]),
        "residue_declared": len(declared), "residue_read": len(either), "residue_unexplained": len(residue) - len(either),
        "stage3_again_equal": sum(1 for t in pop3 if t.get("stage3_again_equal")), "stage3_again_differs": sum(1 for t in pop3 if t.get("stage3_again_equal") is False),
        "distinct_scripts": {"read": distinct(read), "stage3_population": distinct(pop3), "stage3_verified": distinct(s3), "residue": distinct(residue),
                             "residue_read": distinct(either)},
        "by_verdict": {"read": _count(t["verdict"] for t in read), "stage3_population": _count(t["verdict"] for t in pop3)},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build-population", action="store_true")
    ap.add_argument("--population", type=Path)
    ap.add_argument("--tree", type=Path, help="one checkout, as it is: no fetch")
    ap.add_argument("--one", nargs=2, metavar=("REPO", "TIP"))
    ap.add_argument("--work", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    if a.build_population:
        pop = build_population()
        a.out.write_text(json.dumps(pop, indent=1) + "\n", encoding="utf-8")
        print(f"{pop['count']} repositories", file=sys.stderr)
        return 0
    if a.tree:
        res = audit_one(a.tree.resolve(), None)
        print(json.dumps(res, indent=1))
        return 0
    if a.one:
        res = one(a.one[0], a.one[1], a.work)
        a.json.write_text(json.dumps(res), encoding="utf-8")
        return 0
    receipt = run(a.population.resolve(), a.work.resolve(), a.out.resolve(), a.workers)
    print(json.dumps(receipt["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
