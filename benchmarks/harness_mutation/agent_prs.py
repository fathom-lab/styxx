# -*- coding: utf-8 -*-
"""SWALLOW-9 -- the agent's pull request at the gate.

AIDev (Zenodo record 16919272) lists the pull requests that five coding agents -- OpenAI Codex,
GitHub Copilot, Devin, Cursor, Claude Code -- opened on public GitHub repositories between
December 2024 and July 2025, with the agent named by GitHub's own attribution and whether a person
merged the pull request. SWALLOW-7's gate is a function of a BASE and a HEAD. This module runs it
on every one of those pull requests whose own commits touch a hand-written workflow:

  HEAD   refs/pull/N/head as GitHub keeps it, and it must be one of the commits the dataset lists
         for the pull request (the list is GitHub's base...head set, capped at 30)
  BASE   what the dataset's commit list implies: the parents of the pull request's own commits that
         are not themselves its commits are the base branch as the pull request last saw it -- the
         fork point, or the last commit of the base merged into the branch; a candidate that is an
         ancestor of another is dropped; if several remain (a merge of an unrelated branch), the
         newest that the default branch reaches, else the newest, and the record says AMBIGUOUS.
         This needs no branch: a pull request into `dev` gets `dev`'s commit, whatever the default
         branch is. The clones are shallow (commits since 2024-06-01); when a pull request's own
         commit sits on that boundary, its parents are hidden and the default branch is deepened
         once, to 2022-01-01; what is still without a base is skipped
  files  only the workflows the dataset says the pull request's commits changed, and the pull
         request is SUSPECT when git's diff of one of them is larger than the dataset's count for
         it -- the sign that BASE is not the branch the pull request was written against

For each pull request the gate says whether a check that hides its own failure arrives (new hidden:
born, acquired, or rewritten into one), with the repair SWALLOW-4 then SWALLOW-5 verifies on HEAD's
text; for a merged pull request that fires, whether each such check is still hidden at the default
branch's tip today. The receipt carries the repository, the pull request's number, agent, state and
dates, its shas, and the gate's records -- no author, no name, no address, no text of the pull
request beyond the acknowledgement word the SWALLOW-6 regex matched in its title or body.

    python -m benchmarks.harness_mutation.agent_prs --aidev <dir> --population-out papers/harness/swallow9_population.json
    python -m benchmarks.harness_mutation.agent_prs --population papers/harness/swallow9_population.json \
        --work <clones> --workers 4 --out papers/harness/swallow9_receipt.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import differential as D
from . import history as H

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "styxx.harness-agent-prs/v1"
POPULATION_SCHEMA = "styxx.harness-agent-prs.population/v1"
SINCE = "2024-06-01"         # the shallow boundary of every clone: AIDev's earliest pull request is dated 2024-12-24
DEEPEN = "2022-01-01"        # the second try when a pull request's own commit sits on the boundary
FETCH_BATCH = 50             # pull-request heads fetched per round trip
WORKFLOWS = ".github/workflows/"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instrument_sha256() -> str:
    return _sha(Path(__file__))


# ----------------------------------------------------------------------------- the population, from the dataset

def ack_words(title: str | None, body: str | None) -> dict:
    """The first SWALLOW-6 acknowledgement word in the title and in the body; None where none."""
    out = {}
    for k, text in (("title", title), ("body", body)):
        hit = H.ACK.search(text or "")
        out[k] = hit.group(0).lower() if hit else None
    return out


def build_population(aidev: Path) -> dict:
    """Every AIDev pull request one of whose commits changed a file under .github/workflows, with
    the dataset's own counts for those files, its commit shas, and the acknowledgement words of its
    title and body. Reads pull_request.parquet, pr_commits.parquet and pr_commit_details.parquet."""
    import pyarrow.parquet as pq  # an offline step; pyarrow is not a dependency of the instrument's run

    files = {"pull_request.parquet": _sha(aidev / "pull_request.parquet"), "pr_commits.parquet": _sha(aidev / "pr_commits.parquet"),
             "pr_commit_details.parquet": _sha(aidev / "pr_commit_details.parquet")}
    det = pq.read_table(aidev / "pr_commit_details.parquet", columns=["sha", "pr_id", "filename", "status", "additions", "deletions"]).to_pylist()
    per_pr: dict = {}
    for row in det:
        fn = row["filename"] or ""
        if not fn.startswith(WORKFLOWS) or not H._is_workflow(fn) or fn.endswith(".lock.yml"):   # hand-written workflows only, as SWALLOW-7 read them
            continue
        f = per_pr.setdefault(row["pr_id"], {}).setdefault(fn, {"additions": 0, "deletions": 0, "status": []})
        f["additions"] += int(row["additions"] or 0)
        f["deletions"] += int(row["deletions"] or 0)
        if row["status"] not in f["status"]:
            f["status"].append(row["status"])
    commits: dict = {}
    for row in pq.read_table(aidev / "pr_commits.parquet", columns=["sha", "pr_id"]).to_pylist():
        if row["pr_id"] in per_pr:
            commits.setdefault(row["pr_id"], []).append(row["sha"])
    prs = []
    cols = ["id", "number", "title", "body", "agent", "state", "created_at", "closed_at", "merged_at", "repo_url"]
    for row in pq.read_table(aidev / "pull_request.parquet", columns=cols).to_pylist():
        if row["id"] not in per_pr:
            continue
        repo = (row["repo_url"] or "").replace("https://api.github.com/repos/", "")
        prs.append({"repo": repo, "number": int(row["number"]), "agent": row["agent"], "state": row["state"], "merged": row["merged_at"] is not None,
                    "created_at": row["created_at"], "closed_at": row["closed_at"], "merged_at": row["merged_at"],
                    "commits": sorted(commits.get(row["id"], [])), "files": dict(sorted(per_pr[row["id"]].items())),
                    "ack": ack_words(row["title"], row["body"])})
    prs.sort(key=lambda p: (p["repo"], p["number"]))
    return {"schema": POPULATION_SCHEMA, "source": "AIDev, Zenodo record 16919272", "source_sha256": files, "selection":
            "pull requests with at least one commit-detail row whose filename is a hand-written workflow (.github/workflows/*.yml|yaml, not *.lock.yml)",
            "since": SINCE,
            "prs": prs, "count": len(prs), "repos": len({p["repo"] for p in prs})}


# ----------------------------------------------------------------------------- git

def clone_repo(repo: str, work: Path, since: str = SINCE) -> tuple[Path | None, dict]:
    """A blobless, checkout-less clone of the default branch since `since`."""
    dest = work / repo.replace("/", "__")
    info: dict = {"repo": repo, "since": since}
    t0 = time.time()
    if not (dest / ".git").exists():
        p = subprocess.run(["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", f"--shallow-since={since}", "--single-branch",
                            f"https://github.com/{repo}.git", str(dest)], capture_output=True, text=True, timeout=1800,
                           env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
        if p.returncode != 0:
            info["error"] = "clone: " + p.stderr.strip()[:200]
            return None, info
    H._git(dest, ["config", "gc.auto", "0"], check=False)                 # every fetch below adds a pack; auto-gc would repack after each
    try:
        info["tip"] = H._git(dest, ["rev-parse", "HEAD"]).strip()
    except RuntimeError as e:
        info["error"] = f"no default tip: {str(e)[:120]}"
        return None, info
    info["seconds_clone"] = round(time.time() - t0, 1)
    return dest, info


def pull_heads(clone: Path, numbers: list[int]) -> dict[int, str | None]:
    """refs/pull/N/head for each number, fetched into refs/pr/N; None where GitHub no longer has it.
    The fetch carries no depth option: one would move the clone's shallow boundary onto the pull
    request's own commits and cut the mainline behind them. Its filter is the clone's own: a
    different one makes git fetch every tree behind the new commits one at a time."""
    out = H._git(clone, ["ls-remote", "origin", "refs/pull/*/head"], check=False)
    remote: dict[int, str] = {}
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and parts[1].startswith("refs/pull/") and parts[1].endswith("/head"):
            try:
                remote[int(parts[1].split("/")[2])] = parts[0]
            except ValueError:
                continue
    have = [n for n in numbers if n in remote]

    def fetch(batch: list[int]) -> bool:
        p = subprocess.run(["git", "-C", str(clone), "fetch", "--quiet", "--filter=blob:none", "--no-tags", "--no-write-fetch-head", "origin"]
                           + [f"+refs/pull/{n}/head:refs/pr/{n}" for n in batch], capture_output=True, text=True, timeout=1800,
                           env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
        return p.returncode == 0

    for i in range(0, len(have), FETCH_BATCH):
        batch = have[i:i + FETCH_BATCH]
        if not fetch(batch) and len(batch) > 1:        # one bad ref fails the batch: the rest one at a time
            for n in batch:
                fetch([n])
    heads: dict[int, str | None] = {}
    for n in numbers:
        sha = H._git(clone, ["rev-parse", "--verify", "--quiet", f"refs/pr/{n}^{{commit}}"], check=False).strip() if n in remote else ""
        heads[n] = sha or None
    return heads


def shallow_set(clone: Path) -> set[str]:
    """The commits the clone treats as parentless."""
    f = clone / ".git" / "shallow"
    return set(f.read_text().split()) if f.exists() else set()


def deepen(clone: Path, since: str = DEEPEN) -> None:
    """Move the default branch's shallow boundary back to `since` (commits only)."""
    branch = H._git(clone, ["symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"], check=False).strip()
    ref = f"refs/heads/{branch.split('/', 1)[1]}" if "/" in branch else "HEAD"
    H._git(clone, ["fetch", "--quiet", "--filter=blob:none", f"--shallow-since={since}", "--no-tags", "--no-write-fetch-head", "origin", ref], check=False)


def _is_ancestor(clone: Path, a: str, b: str) -> bool:
    return subprocess.run(["git", "-C", str(clone), "merge-base", "--is-ancestor", a, b], capture_output=True).returncode == 0


def _parents_of(clone: Path, shas: list[str]) -> dict[str, list[str]]:
    """The parents each commit object names (a shallow boundary commit names parents the clone may not have)."""
    out: dict[str, list[str]] = {}
    if not shas:
        return out
    data = subprocess.run(["git", "-C", str(clone), "cat-file", "--batch"], input="".join(f"{s}\n" for s in shas).encode(), capture_output=True, timeout=1800).stdout
    pos = 0
    for sha in shas:
        nl = data.index(b"\n", pos)
        header = data[pos:nl].decode("utf-8", "replace")
        pos = nl + 1
        if header.endswith(" missing") or len(header.split()) < 3:
            continue
        size = int(header.split()[2])
        body = data[pos:pos + size].decode("utf-8", "replace")
        pos += size + 1
        out[sha] = [ln.split()[1] for ln in body.split("\n\n", 1)[0].splitlines() if ln.startswith("parent ")]
    return out


def _present(clone: Path, sha: str) -> bool:
    return subprocess.run(["git", "-C", str(clone), "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True).returncode == 0


def _commit_time(clone: Path, sha: str) -> int:
    return int(H._git(clone, ["show", "-s", "--format=%ct", sha], check=False).strip() or 0)


def resolve_base(clone: Path, head: str, tip: str, pr_commits: set[str], shallow: set[str]) -> dict:
    """BASE for one pull request from its own commit list: the parents of its commits that are not
    its commits, reduced to the ones no other candidate descends from."""
    out: dict = {"base": None, "method": "pr-commits", "candidates": 0, "ambiguous": False, "at_boundary": False}
    parents = _parents_of(clone, sorted(pr_commits))
    cands: list[str] = []
    seen: set[str] = set()
    stack = [head]
    while stack:                                        # over the pull request's own commits only
        c = stack.pop()
        if c in seen or c not in pr_commits:
            continue
        seen.add(c)
        for q in parents.get(c, []):
            if q in pr_commits:
                stack.append(q)
            elif q not in cands:
                cands.append(q)
    missing = [c for c in cands if not _present(clone, c)]
    if missing:                                         # a parent behind the boundary: ask for it by name
        H._git(clone, ["fetch", "--quiet", "--filter=blob:none", "--no-tags", "--no-write-fetch-head", "origin"] + missing, check=False)
        missing = [c for c in cands if not _present(clone, c)]
    out["at_boundary"] = bool(missing) or any(c in shallow for c in seen)
    cands = [c for c in cands if c not in missing]
    if not cands or missing:
        return out
    keep = [c for c in cands if not any(o != c and _is_ancestor(clone, c, o) for o in cands)]
    out["candidates"] = len(keep)
    if len(keep) == 1:
        out["base"] = keep[0]
        return out
    out["ambiguous"] = True
    on_tip = [c for c in keep if _is_ancestor(clone, c, tip)]
    pool = on_tip or keep
    out["base"] = max(pool, key=lambda c: (_commit_time(clone, c), c))
    return out


def resolve_all(clone: Path, tip: str, prs: list[dict], heads: dict[int, str | None]) -> tuple[dict[int, dict], bool]:
    """BASE for every pull request; the default branch deepened once when a boundary hides the parents."""
    deepened = False
    for attempt in (0, 1):
        shallow = shallow_set(clone)
        out: dict[int, dict] = {}
        blocked = 0
        for p in prs:
            head = heads.get(p["number"])
            own = set(p.get("commits") or [])
            if head is None or head not in own:
                continue
            out[p["number"]] = resolve_base(clone, head, tip, own, shallow)
            blocked += int(out[p["number"]]["base"] is None and out[p["number"]]["at_boundary"])
        if not blocked or attempt == 1:
            return out, deepened
        deepen(clone)
        deepened = True
    return out, deepened


def prefetch(clone: Path, wants: list[tuple[str, str]]) -> int:
    """Fetch the missing blobs of every (commit, path) in one round trip."""
    ids = []
    for sha, path in wants:
        oid = H._git(clone, ["rev-parse", "--verify", "--quiet", f"{sha}:{path}"], check=False).strip()
        if oid:
            ids.append(oid)
    if not ids:
        return 0
    out = H._git(clone, ["cat-file", "--batch-check"], data="\n".join(ids) + "\n", check=False)
    missing = [line.split()[0] for line in out.splitlines() if line.endswith(" missing")]
    if missing:
        H._git(clone, ["-c", "fetch.negotiationAlgorithm=noop", "fetch", "--quiet", "origin", "--no-tags", "--no-write-fetch-head",
                       "--recurse-submodules=no", "--filter=blob:none", "--stdin"], data="\n".join(missing) + "\n", check=False)
    return len(missing)


def numstat(clone: Path, base: str, head: str) -> dict[str, tuple[int, int]]:
    """git's additions and deletions per workflow path between BASE and HEAD, renames followed."""
    out: dict = {}
    for line in H._git(clone, ["diff", "--numstat", "-M", base, head, "--", ".github/workflows"], check=False).splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        add, dele = (int(x) if x.isdigit() else 0 for x in parts[:2])
        path = parts[-1]
        if "{" in path and " => " in path:              # a/{b => c}/d
            pre, rest = path.split("{", 1)
            inner, post = rest.split("}", 1)
            path = pre + inner.split(" => ")[1] + post
        elif " => " in path:
            path = path.split(" => ")[1]
        out[path] = (add, dele)
    return out


# ----------------------------------------------------------------------------- the gate, per pull request

def audit_pr(clone: Path, pr: dict, head: str, base: str, readers: dict, tip: str | None = None) -> dict:
    """The gate on one pull request: only the workflows the dataset says its commits changed."""
    t0 = time.time()
    expected = pr.get("files") or {}
    changed = [ch for ch in D.changed_workflows(clone, base, head) if ch["path"] in expected or ch["from"] in expected]
    ns = numstat(clone, base, head)
    out = {"workflows": [], "fires": False, "new_hidden": 0, "removed_hidden": 0, "still_hidden": 0, "hidden_after_unread": 0,
           "changed_in_dataset": len(expected), "changed_in_git": len(changed), "suspect": False, "suspect_paths": []}
    for ch in changed:
        path, before = ch["path"], ch["from"]
        exp = expected.get(path) or expected.get(before) or {}
        add, dele = ns.get(path, (0, 0))
        if add > exp.get("additions", 0) or dele > exp.get("deletions", 0):
            out["suspect"] = True
            out["suspect_paths"].append(path)
        wf_name = path.rsplit("/", 1)[-1]
        reader = readers.setdefault(path, H.Reader())
        bt = None if ch["status"] == "A" else D._text(clone, base, before)
        ht = None if ch["status"] == "D" else D._text(clone, head, path)
        w = D.audit_pair(bt, ht, wf_name, reader, fix=True)
        w.update(path=path, status=ch["status"], git_additions=add, git_deletions=dele,
                 dataset_additions=exp.get("additions"), dataset_deletions=exp.get("deletions"))
        if tip and pr.get("merged") and w["new_hidden"] and ht is not None:
            w["at_tip"] = at_tip(clone, tip, path, ht, w["new_hidden"], wf_name, reader)
        out["workflows"].append(w)
        for k in ("new_hidden", "removed_hidden", "still_hidden", "hidden_after_unread"):
            out[k] += len(w[k])
    out["fires"] = out["new_hidden"] > 0
    out["seconds"] = round(time.time() - t0, 2)
    return out


def at_tip(clone: Path, tip: str, path: str, head_text: str, new_hidden: list[dict], wf_name: str, reader: H.Reader) -> list[dict]:
    """For each check the pull request brought hidden: what it is at the default branch's tip today."""
    tt = D._text(clone, tip, path)
    if tt is None:
        return [{"job": r["job"], "step": r["step"], "now": "file gone"} for r in new_hidden]
    w = D.audit_pair(head_text, tt, wf_name, reader, fix=False)
    still = {(r["job"], r["step"]) for r in w["still_hidden"]}
    removed = {(r["job"], r["step"]): r["kind"] for r in w["removed_hidden"]}
    unread = {(r["job"], r["step"]) for r in w["unread_after_hidden"]}
    out = []
    for r in new_hidden:
        k = (r["job"], r["step"])
        now = "still hidden" if k in still else removed.get(k, "unread" if k in unread else ("unparseable" if w["head_unparseable"] else "unmatched"))
        out.append({"job": r["job"], "step": r["step"], "now": now})
    return out


def repo_agent_prs(clone: Path, repo: str, prs: list[dict], info: dict, deadline: float | None = None) -> dict:
    """Every pull request of one repository, through the gate."""
    t0 = time.time()
    tip = info["tip"]
    heads = pull_heads(clone, [p["number"] for p in prs])
    bases, deepened = resolve_all(clone, tip, prs, heads)
    out = {"repo": repo, "tip": tip, "clone": info, "deepened": deepened, "prs": [], "capped": False}
    resolved = []
    for p in prs:
        rec = {k: p.get(k) for k in ("number", "agent", "state", "merged", "created_at", "closed_at", "merged_at", "ack")}
        rec["head"] = heads.get(p["number"])
        rec["dataset_commits"] = len(p.get("commits") or [])
        if rec["head"] is None:
            rec["skip"] = "missing head"
            out["prs"].append(rec)
            continue
        rec["head_in_dataset"] = rec["head"] in set(p.get("commits") or [])
        if not rec["head_in_dataset"]:
            rec["skip"] = "commit list capped" if rec["dataset_commits"] >= 30 else "head moved"
            out["prs"].append(rec)
            continue
        b = bases[p["number"]]
        rec.update(base=b["base"], base_method=b["method"], candidates=b["candidates"], ambiguous=b["ambiguous"], at_boundary=b["at_boundary"])
        if b["base"] is None:
            rec["skip"] = "no base"
            out["prs"].append(rec)
            continue
        resolved.append((p, rec))
        out["prs"].append(rec)
    wants = []
    for p, rec in resolved:
        for path in (p.get("files") or {}):
            wants.append((rec["base"], path))
            wants.append((rec["head"], path))
            if p.get("merged"):
                wants.append((tip, path))
    out["blobs_prefetched"] = prefetch(clone, wants)
    readers: dict = {}
    for p, rec in resolved:
        if deadline and time.time() > deadline:
            out["capped"] = True
            rec["skip"] = "capped"
            continue
        a = audit_pr(clone, p, rec["head"], rec["base"], readers, tip=tip)
        rec.update({k: a[k] for k in ("fires", "new_hidden", "removed_hidden", "still_hidden", "hidden_after_unread", "changed_in_dataset", "changed_in_git",
                                       "suspect", "suspect_paths", "seconds")})
        rec["head_in_tip"] = _is_ancestor(clone, rec["head"], tip)
        rec["audited"] = not a["suspect"]
        if a["fires"] or a["removed_hidden"] or a["hidden_after_unread"] or a["suspect"]:
            rec["detail"] = [{"workflow": w["workflow"], "path": w["path"], "status": w["status"], "new_hidden": w["new_hidden"],
                              "removed_hidden": w["removed_hidden"], "hidden_after_unread": w["hidden_after_unread"], "at_tip": w.get("at_tip"),
                              "git": [w["git_additions"], w["git_deletions"]], "dataset": [w["dataset_additions"], w["dataset_deletions"]],
                              "base_unparseable": w["base_unparseable"], "head_unparseable": w["head_unparseable"]}
                             for w in a["workflows"] if w["new_hidden"] or w["removed_hidden"] or w["hidden_after_unread"] or w["path"] in a["suspect_paths"]]
    for rec in out["prs"]:
        rec.setdefault("audited", False)
    out["seconds"] = round(time.time() - t0, 1)
    out["summary"] = repo_summary(out)
    return out


def repo_summary(r: dict) -> dict:
    prs = r["prs"]
    aud = [p for p in prs if p.get("audited")]
    return {"prs": len(prs), "audited": len(aud), "firing": sum(1 for p in aud if p.get("fires")), "new_hidden_checks": sum(p.get("new_hidden", 0) for p in aud),
            "missing_head": sum(1 for p in prs if p.get("skip") == "missing head"), "no_base": sum(1 for p in prs if p.get("skip") == "no base"),
            "suspect": sum(1 for p in prs if p.get("suspect")), "head_moved": sum(1 for p in prs if p.get("skip") in ("head moved", "commit list capped"))}


# ----------------------------------------------------------------------------- the population

def _one(repo: str, prs: list[dict], work: Path, keep: bool, deadline_per_repo: float | None) -> dict:
    clone, info = clone_repo(repo, work)
    if clone is None:
        return {"repo": repo, "clone_failure": info, "prs_lost": len(prs)}
    try:
        r = repo_agent_prs(clone, repo, prs, info, deadline=(time.time() + deadline_per_repo) if deadline_per_repo else None)
    except Exception as e:  # noqa: BLE001
        return {"repo": repo, "clone_failure": dict(info, error=f"audit: {str(e)[:160]}"), "prs_lost": len(prs)}
    finally:
        if not keep:
            shutil.rmtree(clone, ignore_errors=True)
    return r


def population(pop: dict, work: Path, out_path: Path | None = None, workers: int = 1, limit: int | None = None, keep: bool = False,
               only: list[str] | None = None, deadline_per_repo: float | None = None, population_sha256: str | None = None) -> dict:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.time()
    by_repo: dict[str, list[dict]] = {}
    for p in pop["prs"]:
        if only and p["repo"] not in only:
            continue
        by_repo.setdefault(p["repo"], []).append(p)
    repos = sorted(by_repo)
    repos = repos[:limit] if limit else repos
    order = {r: n for n, r in enumerate(repos)}
    rec = {"schema": SCHEMA, "instrument": "benchmarks/harness_mutation/agent_prs.py", "instrument_sha256": instrument_sha256(),
           "differential_sha256": D.instrument_sha256(), "history_sha256": H.instrument_sha256(), "faults_sha256": H.faults_sha256(),
           "action_checks_sha256": H._sha(ROOT / "benchmarks/harness_mutation/action_checks.py"),
           "repair_sha256": H._sha(ROOT / "benchmarks/harness_mutation/repair.py"), "repair_structural_sha256": H._sha(ROOT / "benchmarks/harness_mutation/repair_structural.py"),
           "population_file_sha256": population_sha256, "population_prs": sum(len(v) for v in by_repo.values()), "population_repos": len(repos),
           "since": SINCE, "deepen": DEEPEN, "workers": workers, "repos": [], "clone_failures": []}
    work.mkdir(parents=True, exist_ok=True)

    def take(r: dict) -> None:
        if "clone_failure" in r:
            rec["clone_failures"].append(dict(r["clone_failure"], prs_lost=r["prs_lost"]))
            print(f"  {r['repo']}: {r['clone_failure'].get('error')} ({r['prs_lost']} PRs)", file=sys.stderr, flush=True)
            return
        rec["repos"].append(r)
        s = r["summary"]
        print(f"  {r['repo']}: {s['prs']} PRs, {s['audited']} audited, fires {s['firing']} ({s['new_hidden_checks']} new hidden), "
              f"missing {s['missing_head']}, no base {s['no_base']}, suspect {s['suspect']}, moved {s['head_moved']}, {r['seconds']}s"
              f"{' deepened' if r.get('deepened') else ''}", file=sys.stderr, flush=True)

    if workers <= 1:
        for repo in repos:
            take(_one(repo, by_repo[repo], work, keep, deadline_per_repo))
            if out_path:
                _write(rec, out_path, t0)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_one, repo, by_repo[repo], work, keep, deadline_per_repo) for repo in repos]
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


def summary(rec: dict) -> dict:
    prs = [p for r in rec["repos"] for p in r["prs"]]
    aud = [p for p in prs if p.get("audited")]
    fires = [p for p in aud if p.get("fires")]
    by_agent: dict = {}
    for p in aud:
        a = by_agent.setdefault(p["agent"], {"audited": 0, "firing": 0, "new_hidden_checks": 0, "merged": 0, "firing_merged": 0})
        a["audited"] += 1
        a["merged"] += int(bool(p.get("merged")))
        if p.get("fires"):
            a["firing"] += 1
            a["new_hidden_checks"] += p["new_hidden"]
            a["firing_merged"] += int(bool(p.get("merged")))
    lost = sum(c.get("prs_lost", 0) for c in rec["clone_failures"])
    return {"repos": len(rec["repos"]), "clone_failures": len(rec["clone_failures"]), "prs_in_population": rec.get("population_prs"),
            "prs_seen": len(prs), "prs_lost_to_clone_failures": lost, "audited": len(aud), "firing": len(fires),
            "new_hidden_checks": sum(p["new_hidden"] for p in fires), "fire_rate": round(len(fires) / len(aud), 4) if aud else None,
            "missing_head": sum(1 for p in prs if p.get("skip") == "missing head"), "no_base": sum(1 for p in prs if p.get("skip") == "no base"),
            "commit_list_capped": sum(1 for p in prs if p.get("skip") == "commit list capped"), "capped": sum(1 for p in prs if p.get("skip") == "capped"),
            "repos_deepened": sum(1 for r in rec["repos"] if r.get("deepened")),
            "suspect": sum(1 for p in prs if p.get("suspect")), "head_moved": sum(1 for p in prs if p.get("skip") == "head moved"),
            "ambiguous_base": sum(1 for p in aud if p.get("ambiguous")), "head_in_tip": sum(1 for p in aud if p.get("head_in_tip")),
            "by_agent": dict(sorted(by_agent.items())),
            "merged_audited": sum(1 for p in aud if p.get("merged")), "merged_firing": sum(1 for p in fires if p.get("merged"))}


def _write(rec: dict, out_path: Path, t0: float) -> None:
    rec["seconds"] = round(time.time() - t0, 1)
    rec["summary"] = summary(rec)
    out_path.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aidev", help="the AIDev directory (pull_request.parquet, pr_commits.parquet, pr_commit_details.parquet): build the population")
    ap.add_argument("--population-out")
    ap.add_argument("--population", help="the population file (.json or .json.gz)")
    ap.add_argument("--work", default="s9prs")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, help="the first N repositories in sorted order")
    ap.add_argument("--only", nargs="*", help="repositories to run, owner/name")
    ap.add_argument("--keep", action="store_true", help="keep the clones")
    ap.add_argument("--deadline-per-repo", type=float, default=None)
    a = ap.parse_args(argv)
    if a.aidev:
        pop = build_population(Path(a.aidev))
        text = json.dumps(pop, indent=1, sort_keys=True) + "\n"
        if a.population_out:
            Path(a.population_out).write_text(text, encoding="utf-8")
        print(json.dumps({k: pop[k] for k in ("count", "repos", "source_sha256")}, indent=1))
        return 0
    if a.population:
        raw = Path(a.population).read_bytes()
        pop = json.loads((gzip.decompress(raw) if a.population.endswith(".gz") else raw).decode("utf-8"))
        rec = population(pop, Path(a.work), Path(a.out) if a.out else None, a.workers, a.limit, a.keep, a.only, a.deadline_per_repo,
                         hashlib.sha256(raw).hexdigest())
        print(json.dumps(rec["summary"], indent=1))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
