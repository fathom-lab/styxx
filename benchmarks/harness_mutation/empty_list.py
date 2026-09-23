# -*- coding: utf-8 -*-
"""SWALLOW-14 -- the empty list, and a hoist that stays local.

SWALLOW-13's third repair stage verified a repair for 13 of the 44 hidden checks the first two
stages left, and most of the 31 it left pass on an empty list: a loop fed by `< <(find …)` or
`mapfile -t a < <(…)` runs zero times when the command that makes its list fails, and bash drops a
process substitution's status, so neither `set -e` nor pipefail sees it. SWALLOW-14 adds two edits
to the stage (`styxx/ciaudit/repair_frontier.py`), designed on SWALLOW-13's 44:

  wait-list    after the statement that reads the list, `wait $! || exit $?` (bash 4.4+: `$!` is
               the process substitution), pipefail inside it when it is a pipeline
  hoist-local  SWALLOW-13's hoist with its strictness kept on its own line -- `__subN="$(…)" ||
               exit $?` -- instead of `set -eo pipefail` for the whole script, which also takes every
               `|| true` elsewhere away; tried before the global hoist

and their no-coe+ forms. This instrument runs the product's `--repair` path, unchanged, on a
population none of the stage was designed on: the repositories of the AIDev dataset's `repository`
table (repositories with more than 100 stars that received agents' pull requests), minus
SWALLOW-1's hundred and SWALLOW-9's 635 -- every repository an earlier cycle read -- each at the
default-branch tip `git ls-remote` gave when the population was frozen. Per repository, as
SWALLOW-13's instrument: the workflows at the tip (sparse, blob-less, depth 1), the audit, every
hand-written finding through stages 1, 2 and 3 with all fourteen of the stage's candidates tried,
the readings, and stage 3 once more with a fresh runner.

    python -m benchmarks.harness_mutation.empty_list --build-population --parquet repository.parquet --out papers/harness/swallow14_population.json.gz
    python -m benchmarks.harness_mutation.empty_list --population papers/harness/swallow14_population.json.gz \\
        --work <dir> --out papers/harness/swallow14_receipt.json.gz --workers 3
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import hashlib
import json
import shutil
import statistics
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

SCHEMA = "styxx.harness-empty-list/v1"
DEADLINE = 600.0          # seconds of analysis per repository; a capped repository says so
TIMEOUT = 1800            # seconds per repository, all of it
HARNESS = ROOT / "papers" / "harness"
PRODUCT = ("styxx/ciaudit/engine.py", "styxx/ciaudit/actions.py", "styxx/ciaudit/repair.py", "styxx/ciaudit/repair_structural.py",
           "styxx/ciaudit/repair_frontier.py")
NEW = ("hoist-local", "wait-list", "no-coe+hoist-local", "no-coe+wait-list")
WAIT = ("wait-list", "no-coe+wait-list")
LOCAL = ("hoist-local", "no-coe+hoist-local")
GLOBAL = ("hoist-substitution", "no-coe+hoist-substitution")


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

def _git(args: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def _tip(repo: str) -> tuple[str, str | None, str | None]:
    try:
        p = _git(["ls-remote", f"https://github.com/{repo}.git", "HEAD"], timeout=60)
    except subprocess.TimeoutExpired:
        return repo, None, "timeout"
    line = (p.stdout.strip().splitlines() or [""])[0].split("\t")
    if p.returncode != 0 or len(line[0]) != 40:
        return repo, None, ((p.stderr.strip().splitlines() or [f"exit {p.returncode}"])[-1])[:160]
    return repo, line[0], None


def excluded() -> tuple[set, dict]:
    """Every repository an earlier cycle read: SWALLOW-1's hundred, SWALLOW-9's population (which
    holds SWALLOW-10's, -11's and -13's)."""
    s1, h1 = _load(HARNESS / "swallow1_population.json")
    p9, h9 = _load(HARNESS / "swallow9_population.json.gz")
    names = {x["repo"].lower() for x in s1} | {pr["repo"].lower() for pr in p9["prs"]}
    return names, {"swallow1_population.json": h1, "swallow9_population.json.gz": h9}


def build_population(parquet: Path, workers: int = 16) -> dict:
    import pandas as pd
    raw = parquet.read_bytes()
    df = pd.read_parquet(parquet)
    names = sorted(set(df["full_name"].dropna()))
    out, sources = excluded()
    cand = [n for n in names if n.lower() not in out]
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        tips = list(ex.map(_tip, cand))
    return {"schema": SCHEMA + "#population",
            "rule": "the AIDev dataset's repository table (hao-li/AIDev, repository.parquet: repositories with more than 100 stars "
                    "that received agents' pull requests), minus SWALLOW-1's hundred and SWALLOW-9's 635, each at the default-branch "
                    "tip `git ls-remote HEAD` gave at the freeze; a repository ls-remote could not read is left out, and listed",
            "source": "https://huggingface.co/datasets/hao-li/AIDev/resolve/main/repository.parquet",
            "sources_sha256": dict(sources, **{"repository.parquet": _sha(raw)}),
            "table_repos": len(names), "excluded_earlier": len(names) - len(cand),
            "unreachable": [{"repo": r, "error": e} for r, t, e in tips if t is None],
            "count": sum(1 for _, t, _ in tips if t), "repos": [{"repo": r, "tip": t} for r, t, _ in tips if t]}


# ----------------------------------------------------------------------------- one repository

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
    n12 = len(R.REPAIRS) + len(RS.REPAIRS)
    for t in R.repair_faults(tree, hand):
        wf = t["workflow"]
        if wf not in texts:
            texts[wf] = (tree / ".github" / "workflows" / wf).read_text(encoding="utf-8", errors="replace")
        text = texts[wf]
        pos = R.locate(text, t["job"], t["index"]) or {}
        run = (pos.get("run") or {}).get("value")
        tried3 = len(t["candidates"]) > n12
        rec_t = {"workflow": wf, "job": t["job"], "index": t["index"], "name": t.get("name"), "verdict": t["verdict"],
                 "baseline": t["baseline"].get("verdict"), "continue_on_error": t.get("continue_on_error"), "run_head": t.get("run_head"),
                 "run_sha256": _sha(run.encode("utf-8")) if run is not None else None,
                 "verified_repair": t["verified_repair"], "stage": _stage(t["verified_repair"]), "tried_stage3": tried3,
                 "candidates": [_cand(c) for c in t["candidates"]], "readings": t.get("readings")}
        if tried3:
            again = F.try_frontier(text, wf, t["job"], t["index"], engine.Runner())
            rec_t["stage3_again_equal"] = (_outcome(again["candidates"]) == _outcome(t["candidates"][n12:])
                                           and again["verified_repair"] == (t["verified_repair"] if rec_t["stage"] == "swallow-13" else None))
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
        shutil.rmtree(tree, ignore_errors=True)        # the tip is in the receipt; thousands of clones are not kept
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ----------------------------------------------------------------------------- the population, in parallel

def _run_one(repo: str, tip: str, work: Path, res_dir: Path) -> dict:
    path = res_dir / (repo.replace("/", "__") + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-m", "benchmarks.harness_mutation.empty_list", "--one", repo, tip, "--work", str(work),
                            "--json", str(path)], cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=TIMEOUT)
        if p.returncode != 0 or not path.exists():
            res = {"repo": repo, "tip": tip, "error": "process: " + ((p.stderr or "").strip().splitlines() or [f"exit {p.returncode}"])[-1][:200],
                   "seconds": round(time.time() - t0, 1)}
            path.write_text(json.dumps(res), encoding="utf-8")
    except subprocess.TimeoutExpired:
        res = {"repo": repo, "tip": tip, "error": "timeout", "seconds": TIMEOUT}
        path.write_text(json.dumps(res), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def run(population: Path, work: Path, out: Path, workers: int = 3) -> dict:
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
            print(f"[{n}/{len(futs)}] {r['repo']} {r.get('error') or ''} {r.get('seconds', 0)}s total={time.time() - t0:.0f}s",
                  file=sys.stderr, flush=True)
    repos = [results[r["repo"]] for r in pop["repos"]]
    receipt = {"schema": SCHEMA, "instrument": "benchmarks/harness_mutation/empty_list.py", "instrument_sha256": instrument_sha256(),
               "product_sha256": product_sha256(), "population_sha256": pop_sha, "deadline_per_repo": DEADLINE, "timeout_per_repo": TIMEOUT,
               "workers": workers, "seconds": round(time.time() - t0, 1), "repos": repos, "summary": summary(repos)}
    raw = (json.dumps(receipt, indent=1, sort_keys=True) + "\n").encode("utf-8")
    out.write_bytes(gzip.compress(raw, mtime=0) if out.suffix == ".gz" else raw)
    return receipt


# ----------------------------------------------------------------------------- the counts

def _count(it) -> dict:
    out: dict = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], str(kv[0]))))


def _ver(t: dict, names) -> bool:
    return any(c.get("verified") for c in t["candidates"] if c["repair"] in names)


def _applies(t: dict, names) -> bool:
    return any(c.get("applies") for c in t["candidates"] if c["repair"] in names)


def summary(repos: list[dict]) -> dict:
    fetched = [r for r in repos if "targets" in r]
    ts = [dict(t, repo=r["repo"]) for r in fetched for t in r["targets"]]
    read = [t for t in ts if t["baseline"] is not None]
    s1 = [t for t in read if t["stage"] == "swallow-4"]
    s2 = [t for t in read if t["stage"] == "swallow-5"]
    pop3 = [t for t in read if t["tried_stage3"]]
    s13 = [t for t in pop3 if _ver(t, F.S13_REPAIRS)]                      # the stage as SWALLOW-13 ran it
    s13_residue = [t for t in pop3 if not _ver(t, F.S13_REPAIRS)]
    new_reach = [t for t in s13_residue if _ver(t, NEW)]
    wait_class = [t for t in s13_residue if _applies(t, WAIT)]
    wait_ver = [t for t in pop3 if _ver(t, WAIT)]
    glob = [t for t in pop3 if _ver(t, GLOBAL)]
    loc = [t for t in pop3 if _ver(t, LOCAL)]
    chosen3 = [t for t in pop3 if t["stage"] == "swallow-13"]
    residue = [t for t in pop3 if t["verified_repair"] is None]

    def lines(sel):
        return sorted(next(c["lines_changed"] for c in t["candidates"] if c["repair"] == t["verified_repair"]) for t in sel)

    new_chosen = [t for t in chosen3 if t["verified_repair"] in NEW]
    lc = lines(new_chosen)
    wc = lines([t for t in chosen3 if t["verified_repair"] in WAIT])
    return {
        "repos": len(repos), "fetched": len(fetched), "fetch_failed": sum(1 for r in repos if r.get("fetch", {}).get("error")),
        "errors": sum(1 for r in repos if r.get("error")), "capped": sum(1 for r in fetched if r.get("capped")),
        "repos_with_targets": sum(1 for r in fetched if r["targets"]), "workflows": sum(r.get("workflows", 0) for r in fetched),
        "targets": len(ts), "read": len(read), "stage1_verified": len(s1), "stage2_verified": len(s2),
        "stage3_population": len(pop3), "s13_verified": len(s13), "s13_residue": len(s13_residue),
        "new_reach": len(new_reach), "new_reach_by_edit": _count(next(c["repair"] for c in t["candidates"] if c["repair"] in NEW and c.get("verified"))
                                                                    for t in new_reach),
        "new_reach_repos": len({t["repo"] for t in new_reach}),
        "wait_class": len(wait_class), "wait_class_verified": sum(1 for t in wait_class if _ver(t, WAIT)),
        "wait_verified": len(wait_ver), "wait_repos": len({t["repo"] for t in wait_ver}),
        "hoist_global_verified": len(glob), "hoist_local_verified": len(loc),
        "global_not_local": len([t for t in glob if not _ver(t, LOCAL)]), "local_not_global": len([t for t in loc if not _ver(t, GLOBAL)]),
        "stage3_chosen": _count(t["verified_repair"] for t in chosen3),
        "all_stages_verified": len(s1) + len(s2) + len(chosen3), "residue": len(residue),
        "residue_read": sum(1 for t in residue if t.get("readings")),
        "new_chosen_lines": lc, "new_chosen_lines_median": statistics.median(lc) if lc else None,
        "wait_chosen_lines": wc, "wait_chosen_lines_median": statistics.median(wc) if wc else None,
        "stage3_again_equal": sum(1 for t in pop3 if t.get("stage3_again_equal")),
        "stage3_again_differs": sum(1 for t in pop3 if t.get("stage3_again_equal") is False),
        "by_verdict": {"read": _count(t["verdict"] for t in read), "stage3_population": _count(t["verdict"] for t in pop3)},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build-population", action="store_true")
    ap.add_argument("--parquet", type=Path)
    ap.add_argument("--population", type=Path)
    ap.add_argument("--one", nargs=2, metavar=("REPO", "TIP"))
    ap.add_argument("--work", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    if a.build_population:
        pop = build_population(a.parquet)
        raw = (json.dumps(pop, indent=1) + "\n").encode("utf-8")
        a.out.write_bytes(gzip.compress(raw, mtime=0) if a.out.suffix == ".gz" else raw)
        print(f"{pop['count']} repositories, {len(pop['unreachable'])} unreachable", file=sys.stderr)
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
