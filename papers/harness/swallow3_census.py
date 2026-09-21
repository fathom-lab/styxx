# -*- coding: utf-8 -*-
"""SWALLOW-3, step one: the census of `uses:` across SWALLOW-2's population, at SWALLOW-2's HEADs.

Names only. No verdict is computed here; the catalogue of checking actions is classified from
this census, by the rule the preregistration states, before the instrument is run.

    python papers/harness/swallow3_census.py --work /path/to/clones
        --out papers/harness/swallow3_actions_census.json

Clones are blob-less sparse checkouts of `.github/workflows` at the HEAD the SWALLOW-2 receipt
records for each repository, so the trees are the ones that receipt was made from; a HEAD that
can no longer be fetched is replaced by the branch head and recorded as `head_moved`. The clones
are kept, for the run.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


sys.path.insert(0, str(ROOT))
from benchmarks.harness_mutation.action_checks import clone_at  # noqa: E402  the pinned-HEAD clone


def action_name(uses) -> tuple[str, str] | None:
    """('owner/repo[/path]', kind) for a `uses:` value; kind is 'action', 'local' (./...), 'docker'
    (docker://...) or 'workflow' (a reusable workflow, .yml/.yaml in the path). The @ref is dropped."""
    if not isinstance(uses, str) or not uses.strip():
        return None
    u = uses.strip()
    if u.startswith("./") or u == ".":
        return (u, "local")
    if u.startswith("docker://"):
        return (u.split("@")[0], "docker")
    name = u.split("@")[0]
    if name.endswith((".yml", ".yaml")):
        return (name, "workflow")
    return (name, "action")


def census_tree(tree: Path) -> dict:
    import yaml
    wdir = tree / ".github" / "workflows"
    out = {"workflows": 0, "unparseable": 0, "steps_uses": {}, "steps_uses_generated": {}, "local": 0, "docker": 0,
           "jobs_reusable": {}, "with_inputs": {}, "steps_run": 0, "steps_uses_total": 0}
    if not wdir.exists():
        return out
    for wf in sorted(list(wdir.glob("*.yml")) + list(wdir.glob("*.yaml"))):
        try:
            doc = yaml.safe_load(wf.read_text(encoding="utf-8", errors="replace")) or {}
        except Exception:  # noqa: BLE001
            out["unparseable"] += 1
            continue
        if not isinstance(doc, dict) or not isinstance(doc.get("jobs"), dict):
            continue
        out["workflows"] += 1
        generated = wf.name.endswith(".lock.yml")
        bucket = out["steps_uses_generated"] if generated else out["steps_uses"]
        for jid, job in doc["jobs"].items():
            if not isinstance(job, dict):
                continue
            if isinstance(job.get("uses"), str):
                an = action_name(job["uses"])
                if an:
                    out["jobs_reusable"][an[0]] = out["jobs_reusable"].get(an[0], 0) + 1
                continue
            for st in job.get("steps") or []:
                if not isinstance(st, dict):
                    continue
                if isinstance(st.get("run"), str):
                    out["steps_run"] += 1
                    continue
                an = action_name(st.get("uses"))
                if not an:
                    continue
                out["steps_uses_total"] += 1
                if an[1] == "local":
                    out["local"] += 1
                    continue
                if an[1] == "docker":
                    out["docker"] += 1
                    continue
                bucket[an[0]] = bucket.get(an[0], 0) + 1
                w = st.get("with")
                if isinstance(w, dict) and not generated:
                    keys = out["with_inputs"].setdefault(an[0], {})
                    for k in w:
                        keys[str(k)] = keys.get(str(k), 0) + 1
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default=str(HERE / "swallow1_population.json"))
    ap.add_argument("--heads", default=str(HERE / "swallow2_receipt.json.gz"), help="the receipt whose per-repository HEADs to clone at")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default=str(HERE / "swallow3_actions_census.json"))
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    t0 = time.time()
    pop = json.loads(Path(a.population).read_text(encoding="utf-8"))
    if a.limit:
        pop = pop[: a.limit]
    heads = {}
    if a.heads:
        opener = gzip.open if a.heads.endswith(".gz") else open
        with opener(a.heads, "rt", encoding="utf-8") as f:
            rec = json.load(f)
        heads = {r["repo"]: r.get("head") for r in rec["repos"] if r.get("head")}
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    repos = []
    totals: dict = {}
    totals_gen: dict = {}
    reusable: dict = {}
    with_inputs: dict = {}
    present: dict = {}
    for k, r in enumerate(pop):
        name = r["repo"] if isinstance(r, dict) else r
        dest, err, info = clone_at(name, heads.get(name), work)
        if dest is None:
            repos.append({"repo": name, "clone_error": err})
            sys.stderr.write(f"{k+1}/{len(pop)} {name}: clone failed: {err}\n")
            continue
        c = census_tree(dest)
        c.update(repo=name, **info)
        repos.append(c)
        for n, v in c["steps_uses"].items():
            totals[n] = totals.get(n, 0) + v
            present.setdefault(n, set()).add(name)
        for n, v in c["steps_uses_generated"].items():
            totals_gen[n] = totals_gen.get(n, 0) + v
        for n, v in c["jobs_reusable"].items():
            reusable[n] = reusable.get(n, 0) + v
        for n, keys in c["with_inputs"].items():
            wk = with_inputs.setdefault(n, {})
            for kk, vv in keys.items():
                wk[kk] = wk.get(kk, 0) + vv
        sys.stderr.write(f"{k+1}/{len(pop)} {name}: {c['workflows']} workflows, {sum(c['steps_uses'].values())} action steps hand-written"
                         f"{' (HEAD moved)' if info.get('head_moved') else ''}\n")
    order = sorted(totals, key=lambda n: (-totals[n], n))
    out = {
        "schema": "styxx.swallow3-actions-census/v1",
        "population_file": a.population, "population_sha256": hashlib.sha256(Path(a.population).read_bytes()).hexdigest(),
        "heads_from": a.heads, "repos": repos,
        "distinct_actions_hand_written": len(totals),
        "action_steps_hand_written": sum(totals.values()),
        "action_steps_generated": sum(totals_gen.values()),
        "steps_run_total": sum(c.get("steps_run", 0) for c in repos),
        "local_action_steps": sum(c.get("local", 0) for c in repos),
        "docker_action_steps": sum(c.get("docker", 0) for c in repos),
        "reusable_workflow_jobs": reusable,
        "actions_hand_written": [{"name": n, "steps": totals[n], "repos": len(present[n])} for n in order],
        "actions_generated": dict(sorted(totals_gen.items(), key=lambda kv: (-kv[1], kv[0]))),
        "with_inputs": with_inputs,
        "seconds": round(time.time() - t0, 1),
    }
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("distinct_actions_hand_written", "action_steps_hand_written", "action_steps_generated",
                                          "steps_run_total", "local_action_steps", "docker_action_steps", "seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
