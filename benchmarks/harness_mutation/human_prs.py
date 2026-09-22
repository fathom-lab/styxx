# -*- coding: utf-8 -*-
"""SWALLOW-10 -- the baseline: the same gate, the same repositories, the same months, on the pull
requests people opened.

SWALLOW-9 read the pull requests five coding agents opened and found the gate firing on 0.8% of
them. Compared with what? AIDev has no human pull requests, but GitHub keeps every pull request's
head under refs/pull/N/head and numbers them in order. So, for every repository SWALLOW-9 read:
the pull-request numbers between the repository's first and last agent pull request in the
dataset's window are the same months; the ones the dataset does not attribute to an agent are the
rest of the repository's pull requests in those months. This module samples them (a seeded sample
of up to CAP_HUMAN per repository), fetches their heads, and puts every pull request -- the agents'
and the others' -- through one pipeline:

  group   the dataset's agent for a pull request it lists; otherwise, by SWALLOW-8's rule on the
          head commit's author, subject and body: `agent-signed` (a signature the dataset did not
          attribute: excluded from the baseline, a floor), `automation` (a bot), `human`
  BASE    the merge commit's first parent when a merge commit on any branch merged HEAD, else the
          newest commit of HEAD's ancestry that some branch not containing HEAD reaches (the fork
          point from the closest branch); the same rule for every group
  touch   git's three-dot diff between BASE and HEAD names a hand-written workflow
  merged  a merge commit merging HEAD; or a commit on any branch whose subject GitHub's squash
          merge writes -- "... (#N)" -- or "Merge pull request #N"; or a commit on any branch with
          HEAD's own subject and author date (a rebase merge keeps both); a floor, checked against
          the dataset's truth on the agents' pull requests

The clones are shallow (commits since 2024-06-01); when that leaves a pull request without a base,
every branch is deepened once, to 2022-01-01, and the pull requests still without one are skipped.

The gate is SWALLOW-7's `audit_pair` on each changed workflow. The receipt carries repository,
number, group, shas, the reading and the merge signal -- no author, no name, no address, no text.

    python -m benchmarks.harness_mutation.human_prs --aidev <dir> --receipt9 papers/harness/swallow9_receipt.json.gz \
        --population9 papers/harness/swallow9_population.json.gz --sample-out papers/harness/swallow10_sample.json
    python -m benchmarks.harness_mutation.human_prs --sample papers/harness/swallow10_sample.json.gz --work <clones> \
        --workers 3 --out papers/harness/swallow10_receipt.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import agent_prs as P
from . import authorship as A
from . import differential as D
from . import history as H

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "styxx.harness-human-prs/v1"
SAMPLE_SCHEMA = "styxx.harness-human-prs.sample/v1"
SINCE = P.SINCE
CAP_HUMAN = 150            # non-agent pull requests sampled per repository, from the numbers between its first and last agent pull request
CAP_AGENT_EXTRA = 50       # agent pull requests the dataset does not list as touching a workflow, sampled per repository (the selection's floor)
SEED = 10
WINDOW = ("2024-12-24", "2025-07-30")
SQUASH = re.compile(r"\(#(\d+)\)\s*$")
MERGE_SUBJECT = re.compile(r"^Merge pull request #(\d+)\b")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instrument_sha256() -> str:
    return _sha(Path(__file__))


def _load(path: Path) -> dict:
    raw = path.read_bytes()
    return json.loads((gzip.decompress(raw) if path.suffix == ".gz" else raw).decode("utf-8"))


# ----------------------------------------------------------------------------- the sample, from the dataset and SWALLOW-9

def build_sample(aidev: Path, receipt9: Path, population9: Path) -> dict:
    """For every repository SWALLOW-9 cloned: its agent pull requests in the window (all of them,
    from the dataset's full table), the span of numbers they cover, which of them SWALLOW-9 read,
    and a seeded choice of CAP_AGENT_EXTRA others. The non-agent numbers are sampled at run time
    from what refs/pull/*/head lists inside the span."""
    import pyarrow.parquet as pq  # an offline step

    r9 = _load(receipt9)
    pop9 = _load(population9)
    repos = {x["repo"] for x in r9["repos"]}
    in9 = {(p["repo"], p["number"]) for p in pop9["prs"]}
    rows = pq.read_table(aidev / "all_pull_request.parquet", columns=["number", "agent", "repo_url", "created_at", "merged_at", "state"]).to_pylist()
    per: dict = {}
    for row in rows:
        repo = (row["repo_url"] or "").replace("https://api.github.com/repos/", "")
        if repo not in repos or not (WINDOW[0] <= (row["created_at"] or "")[:10] <= WINDOW[1]):
            continue
        per.setdefault(repo, {})[int(row["number"])] = {"agent": row["agent"], "merged": row["merged_at"] is not None, "state": row["state"],
                                                        "created_at": row["created_at"], "in_population9": (repo, int(row["number"])) in in9}
    out = []
    for repo in sorted(per):
        agents = per[repo]
        nums = sorted(agents)
        extra_pool = [n for n in nums if not agents[n]["in_population9"]]
        rng = random.Random(f"{SEED}:{repo}")
        extra = sorted(rng.sample(extra_pool, min(CAP_AGENT_EXTRA, len(extra_pool))))
        out.append({"repo": repo, "span": [nums[0], nums[-1]], "agent_prs": {str(n): agents[n] for n in nums},
                    "agent_fetch": sorted({n for n in nums if agents[n]["in_population9"]} | set(extra)), "cap_human": CAP_HUMAN, "seed": SEED})
    return {"schema": SAMPLE_SCHEMA, "source": "AIDev all_pull_request.parquet; the SWALLOW-9 receipt and population", "window": list(WINDOW),
            "source_sha256": {"all_pull_request.parquet": _sha(aidev / "all_pull_request.parquet"), "receipt9": _sha(receipt9), "population9": _sha(population9)},
            "cap_human": CAP_HUMAN, "cap_agent_extra": CAP_AGENT_EXTRA, "seed": SEED, "repos": out, "count": len(out)}


def human_sample(repo: str, remote_numbers: list[int], span: list[int], agent_numbers: set[int], cap: int = CAP_HUMAN, seed: int = SEED) -> list[int]:
    """The non-agent pull-request numbers inside the span, sampled without replacement, seeded by the repository."""
    pool = sorted(n for n in remote_numbers if span[0] <= n <= span[1] and n not in agent_numbers)
    rng = random.Random(f"{seed}:{repo}:human")
    return sorted(rng.sample(pool, min(cap, len(pool))))


# ----------------------------------------------------------------------------- git

def fetch_branches(clone: Path, since: str = SINCE) -> list[str]:
    """Every branch of the remote, commits and trees since `since`, with the clone's own filter."""
    H._git(clone, ["fetch", "--quiet", "--filter=blob:none", f"--shallow-since={since}", "--no-tags", "--no-write-fetch-head", "origin",
                   "+refs/heads/*:refs/remotes/origin/*"], check=False, timeout=3000)
    return [ln.split()[1] for ln in H._git(clone, ["for-each-ref", "--format=%(objectname) %(refname)", "refs/remotes/origin/"], check=False).splitlines()
            if len(ln.split()) == 2 and not ln.endswith("/HEAD")]


def remote_pull_numbers(clone: Path) -> list[int]:
    out = []
    for line in H._git(clone, ["ls-remote", "origin", "refs/pull/*/head"], check=False).splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and parts[1].startswith("refs/pull/") and parts[1].endswith("/head"):
            try:
                out.append(int(parts[1].split("/")[2]))
            except ValueError:
                continue
    return sorted(out)


def fetch_heads(clone: Path, numbers: list[int]) -> dict[int, str | None]:
    """refs/pull/N/head into refs/pr/N, in batches, with the clone's own filter and no depth option."""
    have = sorted(set(numbers))
    for i in range(0, len(have), P.FETCH_BATCH):
        batch = have[i:i + P.FETCH_BATCH]
        p = subprocess.run(["git", "-C", str(clone), "fetch", "--quiet", "--filter=blob:none", "--no-tags", "--no-write-fetch-head", "origin"]
                           + [f"+refs/pull/{n}/head:refs/pr/{n}" for n in batch], capture_output=True, text=True, timeout=1800)
        if p.returncode != 0 and len(batch) > 1:
            for n in batch:
                subprocess.run(["git", "-C", str(clone), "fetch", "--quiet", "--filter=blob:none", "--no-tags", "--no-write-fetch-head", "origin",
                                f"+refs/pull/{n}/head:refs/pr/{n}"], capture_output=True, text=True, timeout=1800)
    heads: dict[int, str | None] = {}
    for n in have:
        sha = H._git(clone, ["rev-parse", "--verify", "--quiet", f"refs/pr/{n}^{{commit}}"], check=False).strip()
        heads[n] = sha or None
    return heads


def branch_maps(clone: Path, branches: list[str]) -> dict:
    """Over every commit any branch reaches: the merge commit each non-first parent was merged by,
    first parents, the reachable set, and the pull-request numbers the subjects say were merged."""
    merged_by: dict[str, str] = {}
    first_parent: dict[str, str] = {}
    reachable: set[str] = set()
    merged_numbers: dict[int, str] = {}
    rebased: dict[tuple[str, str], list[str]] = {}
    for line in H._git(clone, ["log", "--format=%H %P%x00%at%x00%s", "--stdin"], data="\n".join(branches) + "\n", check=False, timeout=3000).splitlines():
        if line.count("\x00") < 2:
            continue
        shas, atime, subject = line.split("\x00", 2)
        parts = shas.split()
        if not parts:
            continue
        sha, parents = parts[0], parts[1:]
        reachable.add(sha)
        if parents:
            first_parent[sha] = parents[0]
        for q in parents[1:]:
            merged_by.setdefault(q, sha)
        m = SQUASH.search(subject) or MERGE_SUBJECT.match(subject)
        if m:
            merged_numbers.setdefault(int(m.group(1)), sha)
        rebased.setdefault((subject, atime), []).append(sha)
    return {"merged_by": merged_by, "first_parent": first_parent, "reachable": reachable, "merged_numbers": merged_numbers, "rebased": rebased}


def _containing(clone: Path, head: str) -> list[str]:
    return [ln.strip() for ln in H._git(clone, ["for-each-ref", "--format=%(refname)", "--contains", head, "refs/remotes/origin/"], check=False).splitlines() if ln.strip()]


def resolve_base(clone: Path, head: str, branches: list[str], maps: dict, cache: dict) -> dict:
    """BASE by the closest-branch rule; the merge commit's first parent when one merged HEAD.
    When no branch reaches HEAD (a closed, squashed or rebased pull request), every commit of
    HEAD's ancestry that a branch reaches is on another branch, and the newest is the base; when
    one does (a fast-forward, or the pull request's branch still there), the branches containing
    HEAD are found and left out of the union first."""
    m = maps["merged_by"].get(head)
    if m is not None:
        target = maps["first_parent"].get(m)
        base = H._git(clone, ["merge-base", head, target], check=False).strip() if target else ""
        return {"base": base or None, "method": "merge-commit", "merge_commit": m}
    ancestry = H._git(clone, ["rev-list", "--date-order", head], check=False, timeout=3000).split()
    if head not in maps["reachable"]:
        union, excluded = maps["reachable"], 0
    else:
        containing = tuple(sorted(_containing(clone, head)))
        if containing not in cache:
            others = [b for b in branches if b not in set(containing)]
            cache[containing] = set(H._git(clone, ["rev-list", "--stdin"], data="\n".join(others) + "\n", check=False, timeout=3000).split()) if others else set()
        union, excluded = cache[containing], len(containing)
    for c in ancestry:
        if c != head and c in union:
            return {"base": c, "method": "closest-branch", "merge_commit": None, "excluded_branches": excluded}
    return {"base": None, "method": "closest-branch", "merge_commit": None, "excluded_branches": excluded}


def head_class(clone: Path, head: str) -> tuple[str, str | None, int, tuple[str, str]]:
    """SWALLOW-8's class of the head commit, the signal's label, the commit time, and the
    (subject, author time) pair a rebase merge preserves. No name is kept."""
    out = H._git(clone, ["show", "-s", "--format=%an%x00%ae%x00%ct%x00%at%x00%s%x00%b", head], check=False)
    parts = out.split("\x00", 5)
    if len(parts) < 6:
        return "human", None, 0, ("", "")
    cls, label = A.classify(parts[0], parts[1], parts[4], parts[5])
    return cls, label, int(parts[2] or 0), (parts[4], parts[3])


# ----------------------------------------------------------------------------- the gate, per pull request

def audit(clone: Path, base: str, head: str, readers: dict) -> dict:
    changed = D.changed_workflows(clone, base, head)
    out = {"touching": bool(changed), "workflows_changed": len(changed), "fires": False, "new_hidden": 0, "removed_hidden": 0, "still_hidden": 0,
           "hidden_after_unread": 0, "detail": []}
    if not changed:
        return out
    P.prefetch(clone, [(base, ch["from"]) for ch in changed if ch["status"] != "A"] + [(head, ch["path"]) for ch in changed if ch["status"] != "D"])
    for ch in changed:
        path, before = ch["path"], ch["from"]
        wf_name = path.rsplit("/", 1)[-1]
        reader = readers.setdefault(path, H.Reader())
        bt = None if ch["status"] == "A" else D._text(clone, base, before)
        ht = None if ch["status"] == "D" else D._text(clone, head, path)
        w = D.audit_pair(bt, ht, wf_name, reader, fix=True)
        for k in ("new_hidden", "removed_hidden", "still_hidden", "hidden_after_unread"):
            out[k] += len(w[k])
        if w["new_hidden"] or w["removed_hidden"] or w["hidden_after_unread"]:
            out["detail"].append({"workflow": w["workflow"], "path": path, "status": ch["status"], "new_hidden": w["new_hidden"], "removed_hidden": w["removed_hidden"],
                                  "hidden_after_unread": w["hidden_after_unread"], "base_unparseable": w["base_unparseable"], "head_unparseable": w["head_unparseable"]})
    out["fires"] = out["new_hidden"] > 0
    return out


def repo_prs(clone: Path, spec: dict, info: dict, receipt9_prs: dict[int, dict] | None = None, deadline: float | None = None) -> dict:
    """Every sampled pull request of one repository, through one pipeline."""
    t0 = time.time()
    repo = spec["repo"]
    branches = fetch_branches(clone)
    numbers_remote = remote_pull_numbers(clone)
    agent_numbers = {int(n) for n in spec["agent_prs"]}
    humans = human_sample(repo, numbers_remote, spec["span"], agent_numbers, cap=int(spec.get("cap_human", CAP_HUMAN)), seed=int(spec.get("seed", SEED)))
    to_fetch = sorted(set(spec["agent_fetch"]) | set(humans))
    heads = fetch_heads(clone, to_fetch)
    maps = branch_maps(clone, branches)
    out = {"repo": repo, "tip": info["tip"], "clone": info, "branches": len(branches), "remote_pull_requests": len(numbers_remote), "deepened": False,
           "span": spec["span"], "non_agent_in_span": sum(1 for n in numbers_remote if spec["span"][0] <= n <= spec["span"][1] and n not in agent_numbers),
           "human_sampled": len(humans), "agent_fetched": len(spec["agent_fetch"]), "prs": [], "capped": False}
    readers: dict = {}
    cache: dict = {}
    pending: list[dict] = []
    for n in to_fetch:
        rec: dict = {"number": n, "head": heads.get(n)}
        a9 = spec["agent_prs"].get(str(n))
        if a9:
            rec.update(group=a9["agent"], dataset_merged=a9["merged"], dataset_state=a9["state"], in_population9=a9["in_population9"])
        if rec["head"] is None:
            rec["skip"] = "missing head"
            out["prs"].append(rec)
            continue
        cls, label, ctime, key = head_class(clone, rec["head"])
        rec["head_time"] = ctime
        if not a9:
            rec["group"] = {"agent": "agent-signed", "automation": "automation", "human": "human"}[cls]
            rec["signal"] = label
        else:
            rec["head_signal"] = label if cls == "agent" else None
        rec["_key"] = key
        out["prs"].append(rec)
        pending.append(rec)

    def merge_signals(rec: dict) -> None:
        by_merge = rec["head"] in maps["merged_by"]
        by_subject = rec["number"] in maps["merged_numbers"]
        by_rebase = any(sha != rec["head"] for sha in maps["rebased"].get(rec["_key"], []))
        rec["merged_by_merge_commit"], rec["merged_by_subject"], rec["merged_by_rebase"] = by_merge, by_subject, by_rebase
        rec["merged_heuristic"] = by_merge or by_subject or by_rebase

    for attempt in (0, 1):
        for rec in pending:
            if "base" in rec and rec["base"] is not None:
                continue
            merge_signals(rec)
            b = resolve_base(clone, rec["head"], branches, maps, cache)
            rec.update(base=b["base"], base_method=b["method"], merge_commit=b.get("merge_commit"), excluded_branches=b.get("excluded_branches", 0))
        if attempt == 0 and any(rec["base"] is None for rec in pending) and P.shallow_set(clone):
            H._git(clone, ["fetch", "--quiet", "--filter=blob:none", f"--shallow-since={P.DEEPEN}", "--no-tags", "--no-write-fetch-head", "origin",
                           "+refs/heads/*:refs/remotes/origin/*"], check=False, timeout=3000)
            out["deepened"] = True
            maps = branch_maps(clone, branches)
            cache = {}
            continue
        break
    for rec in pending:
        rec.pop("_key", None)
        if rec["base"] is None:
            rec["skip"] = "no base"
            continue
        if deadline and time.time() > deadline:
            rec["skip"] = "capped"
            out["capped"] = True
            continue
        if receipt9_prs and rec["number"] in receipt9_prs:
            r9 = receipt9_prs[rec["number"]]
            rec["base9"] = r9.get("base")
            rec["base_agrees9"] = r9.get("base") == rec["base"] if r9.get("base") else None
            rec["fires9"] = r9.get("fires") if r9.get("audited") else None
        t1 = time.time()
        rec.update(audit(clone, rec["base"], rec["head"], readers))
        rec["seconds"] = round(time.time() - t1, 2)
        rec["audited"] = True
    for rec in out["prs"]:
        rec.setdefault("audited", False)
    out["seconds"] = round(time.time() - t0, 1)
    out["summary"] = repo_summary(out)
    return out


def repo_summary(r: dict) -> dict:
    prs = r["prs"]
    aud = [p for p in prs if p.get("audited")]
    touch = [p for p in aud if p.get("touching")]
    return {"prs": len(prs), "audited": len(aud), "touching": len(touch), "firing": sum(1 for p in touch if p["fires"]),
            "human_touching": sum(1 for p in touch if p["group"] == "human"), "human_firing": sum(1 for p in touch if p["group"] == "human" and p["fires"]),
            "missing_head": sum(1 for p in prs if p.get("skip") == "missing head"), "no_base": sum(1 for p in prs if p.get("skip") == "no base")}


# ----------------------------------------------------------------------------- the population

def _one(spec: dict, work: Path, keep: bool, receipt9_prs: dict | None, deadline_per_repo: float | None) -> dict:
    clone, info = P.clone_repo(spec["repo"], work)
    if clone is None:
        return {"repo": spec["repo"], "clone_failure": info}
    try:
        return repo_prs(clone, spec, info, receipt9_prs, deadline=(time.time() + deadline_per_repo) if deadline_per_repo else None)
    except Exception as e:  # noqa: BLE001
        return {"repo": spec["repo"], "clone_failure": dict(info, error=f"audit: {str(e)[:160]}")}
    finally:
        if not keep:
            shutil.rmtree(clone, ignore_errors=True)


def population(sample: dict, work: Path, out_path: Path | None = None, workers: int = 1, limit: int | None = None, keep: bool = False,
               only: list[str] | None = None, receipt9: dict | None = None, deadline_per_repo: float | None = None, sample_sha256: str | None = None) -> dict:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.time()
    specs = [s for s in sample["repos"] if not only or s["repo"] in only]
    specs = specs[:limit] if limit else specs
    order = {s["repo"]: n for n, s in enumerate(specs)}
    r9 = {x["repo"]: {p["number"]: p for p in x["prs"]} for x in (receipt9 or {"repos": []})["repos"]}
    rec = {"schema": SCHEMA, "instrument": "benchmarks/harness_mutation/human_prs.py", "instrument_sha256": instrument_sha256(),
           "agent_prs_sha256": P.instrument_sha256(), "authorship_sha256": A.instrument_sha256(), "differential_sha256": D.instrument_sha256(),
           "history_sha256": H.instrument_sha256(), "faults_sha256": H.faults_sha256(), "sample_file_sha256": sample_sha256,
           "receipt9_sha256": (receipt9 or {}).get("_sha256"), "window": sample.get("window"), "cap_human": sample.get("cap_human"),
           "cap_agent_extra": sample.get("cap_agent_extra"), "seed": sample.get("seed"), "since": SINCE, "workers": workers, "repos": [], "clone_failures": []}
    work.mkdir(parents=True, exist_ok=True)

    def take(r: dict) -> None:
        if "clone_failure" in r:
            rec["clone_failures"].append(r["clone_failure"])
            print(f"  {r['repo']}: {r['clone_failure'].get('error')}", file=sys.stderr, flush=True)
            return
        rec["repos"].append(r)
        s = r["summary"]
        print(f"  {r['repo']}: {s['prs']} PRs ({r['human_sampled']} human of {r['non_agent_in_span']} in span), {s['audited']} audited, {s['touching']} touching, "
              f"fires {s['firing']} (human {s['human_firing']}/{s['human_touching']}), missing {s['missing_head']}, no base {s['no_base']}, "
              f"{r['branches']} branches, {r['seconds']}s{' deepened' if r.get('deepened') else ''}{' CAPPED' if r.get('capped') else ''}", file=sys.stderr, flush=True)

    if workers <= 1:
        for s in specs:
            take(_one(s, work, keep, r9.get(s["repo"]), deadline_per_repo))
            if out_path:
                _write(rec, out_path, t0)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_one, s, work, keep, r9.get(s["repo"]), deadline_per_repo) for s in specs]
            for fut in as_completed(futs):
                take(fut.result())
                if out_path:
                    _write(rec, out_path, t0)
    rec["repos"].sort(key=lambda r: order.get(r["repo"], 10**6))
    rec["seconds"] = round(time.time() - t0, 1)
    rec["summary"] = summary(rec)
    if out_path:
        _write(rec, out_path, t0)
    return rec


GROUPS = ("human", "automation", "agent-signed", "OpenAI_Codex", "Copilot", "Devin", "Cursor", "Claude_Code", "Google_Jules")


def summary(rec: dict) -> dict:
    prs = [dict(p, repo=x["repo"]) for x in rec["repos"] for p in x["prs"]]
    aud = [p for p in prs if p.get("audited")]
    touch = [p for p in aud if p.get("touching")]
    by: dict = {}
    for p in aud:
        g = by.setdefault(p["group"], {"audited": 0, "touching": 0, "firing": 0, "new_hidden_checks": 0, "merged_heuristic_touching": 0})
        g["audited"] += 1
        if p.get("touching"):
            g["touching"] += 1
            g["merged_heuristic_touching"] += int(bool(p.get("merged_heuristic")))
            if p["fires"]:
                g["firing"] += 1
                g["new_hidden_checks"] += p["new_hidden"]
    agents9 = [p for p in aud if p.get("in_population9")]
    truth = [p for p in aud if "dataset_merged" in p]
    return {"repos": len(rec["repos"]), "clone_failures": len(rec["clone_failures"]), "prs": len(prs), "audited": len(aud), "touching": len(touch),
            "firing": sum(1 for p in touch if p["fires"]), "missing_head": sum(1 for p in prs if p.get("skip") == "missing head"),
            "no_base": sum(1 for p in prs if p.get("skip") == "no base"), "capped": sum(1 for p in prs if p.get("skip") == "capped"),
            "human_sampled": sum(x.get("human_sampled", 0) for x in rec["repos"]), "non_agent_in_span": sum(x.get("non_agent_in_span", 0) for x in rec["repos"]),
            "by_group": {g: by[g] for g in list(GROUPS) + sorted(set(by) - set(GROUPS)) if g in by},
            "base_agreement9": {"compared": sum(1 for p in agents9 if p.get("base_agrees9") is not None), "agree": sum(1 for p in agents9 if p.get("base_agrees9"))},
            "fires_agreement9": {"compared": sum(1 for p in agents9 if p.get("fires9") is not None), "agree": sum(1 for p in agents9 if p.get("fires9") is not None and p["fires9"] == p.get("fires"))},
            "merged_heuristic_on_dataset_truth": {"merged": sum(1 for p in truth if p["dataset_merged"]), "recalled": sum(1 for p in truth if p["dataset_merged"] and p.get("merged_heuristic")),
                                                  "not_merged": sum(1 for p in truth if not p["dataset_merged"]), "false_positive": sum(1 for p in truth if not p["dataset_merged"] and p.get("merged_heuristic")),
                                                  "recalled_by": {k: sum(1 for p in truth if p["dataset_merged"] and p.get(k)) for k in ("merged_by_merge_commit", "merged_by_subject", "merged_by_rebase")}},
            "repos_deepened": sum(1 for x in rec["repos"] if x.get("deepened"))}


def _write(rec: dict, out_path: Path, t0: float) -> None:
    rec["seconds"] = round(time.time() - t0, 1)
    rec["summary"] = summary(rec)
    out_path.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aidev")
    ap.add_argument("--receipt9")
    ap.add_argument("--population9")
    ap.add_argument("--sample-out")
    ap.add_argument("--sample")
    ap.add_argument("--work", default="s10prs")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--deadline-per-repo", type=float, default=None)
    a = ap.parse_args(argv)
    if a.aidev:
        s = build_sample(Path(a.aidev), Path(a.receipt9), Path(a.population9))
        text = json.dumps(s, indent=1, sort_keys=True) + "\n"
        if a.sample_out:
            Path(a.sample_out).write_text(text, encoding="utf-8")
        print(json.dumps({k: s[k] for k in ("count", "source_sha256", "cap_human", "cap_agent_extra", "seed")}, indent=1))
        return 0
    if a.sample:
        raw = Path(a.sample).read_bytes()
        sample = json.loads((gzip.decompress(raw) if a.sample.endswith(".gz") else raw).decode("utf-8"))
        r9 = None
        if a.receipt9:
            raw9 = Path(a.receipt9).read_bytes()
            r9 = json.loads((gzip.decompress(raw9) if a.receipt9.endswith(".gz") else raw9).decode("utf-8"))
            r9["_sha256"] = hashlib.sha256(raw9).hexdigest()
        rec = population(sample, Path(a.work), Path(a.out) if a.out else None, a.workers, a.limit, a.keep, a.only, r9, a.deadline_per_repo,
                         hashlib.sha256(raw).hexdigest())
        print(json.dumps(rec["summary"], indent=1))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
