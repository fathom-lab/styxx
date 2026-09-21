# -*- coding: utf-8 -*-
"""SWALLOW-7 -- the differential audit: the check that arrives hidden, caught at the commit.

SWALLOW-6 read the population's history and found that hidden checks are mostly written hidden,
rarely made loud, and old. The place to catch one is the pull request that brings it. This module
is that gate: given a BASE and a HEAD, it reads only the workflows that changed between them, with
the frozen SWALLOW-3 reading, matches every step across the two revisions (the same job and step
name -- else id, else first line; a step whose script survived a rename is matched by its script),
and reports:

  new hidden          hidden at HEAD, and at BASE loud, not a check, or not there at all (the gate
                      FIRES); for each, the mechanism when the step existed before, and the repair
                      SWALLOW-4 then SWALLOW-5 verifies on HEAD's text, with its diff
  hidden after unread hidden at HEAD, uninterpretable at BASE: reported, not a firing
  removed hidden      hidden at BASE, and at HEAD loud (a repair), not a check, or gone (a death)
  still hidden        hidden at both

The population reading runs the gate at every mainline commit of the SWALLOW-6 clones that touches
a hand-written workflow -- BASE its first parent, HEAD the commit -- and records what the gate would
have said, and how long it took on a stated sample with nothing memoised. Generated workflows
(`*.lock.yml`) are not read.

    python -m benchmarks.harness_mutation.differential --tree . --base origin/main      # this checkout against a base
    python -m benchmarks.harness_mutation.differential --receipt papers/harness/swallow6_receipt.json.gz \
        --work <clones> --workers 2 --out papers/harness/swallow7_receipt.json          # the population's
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

from . import history as H
from . import repair
from . import repair_structural as rs

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "styxx.harness-differential/v1"
FIRES = ("loud", "other")           # what a check was at BASE for its hiding at HEAD to be a firing
SAMPLE_EVERY = 100                  # the timing sample: every Nth mainline commit, read with nothing memoised


# ----------------------------------------------------------------------------- git

def changed_workflows(clone: Path, base: str, head: str) -> list[dict]:
    """The workflow paths that differ between two commits, renames followed; hand-written only."""
    out = H._git(clone, ["diff", "--name-status", "-M", base, head, "--", ".github/workflows"], check=False)
    changed = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0][0]
        if status == "R" and len(parts) >= 3:
            rec = {"status": "R", "path": parts[2], "from": parts[1]}
        else:
            rec = {"status": status, "path": parts[-1], "from": parts[-1]}
        if H._is_workflow(rec["path"]) and not rec["path"].endswith(".lock.yml"):
            changed.append(rec)
    return changed


def _text(clone: Path, sha: str, path: str) -> str | None:
    return H.texts_at(clone, [(sha, path)]).get((sha, path))


# ----------------------------------------------------------------------------- the gate

def _sides(fl: dict | None, snap: dict) -> dict:
    """Every step of one revision keyed by (job, key): its state, verdict, fault record and snapshot."""
    out = {}
    for (jid, i), s in snap.items():
        k = (jid, s["key"])
        if k in out:                                   # two steps share a key: the first is followed
            continue
        f = (fl or {}).get((jid, i))
        out[k] = {"state": H.state_of(f["verdict"]) if f else "other", "verdict": f["verdict"] if f else None, "fault": f,
                  "snap": s, "index": i, "run_sha": s["run_sha"]}
    return out


def audit_pair(base_text: str | None, head_text: str | None, wf_name: str, reader: H.Reader, fix: bool = True) -> dict:
    """The gate on one workflow: what changed between its BASE text and its HEAD text."""
    out = {"workflow": wf_name, "new_hidden": [], "hidden_after_unread": [], "removed_hidden": [], "still_hidden": [],
           "unread_after_hidden": [], "base_unparseable": None, "head_unparseable": None, "reads": 0}
    b = h = {}
    if base_text is not None:
        fl, snap, err = reader.read(base_text, wf_name)
        out["base_unparseable"] = err
        b = _sides(fl, snap) if err is None else {}
    if head_text is not None:
        fl, snap, err = reader.read(head_text, wf_name)
        out["head_unparseable"] = err
        h = _sides(fl, snap) if err is None else {}
    out["reads"] = reader.reads
    # renames: a key only at BASE and a key only at HEAD, same job, same script
    only_b = {k for k in b if k not in h}
    only_h = {k for k in h if k not in b}
    renamed = {}
    for kb in sorted(only_b):
        if b[kb]["run_sha"] is None:
            continue
        match = next((kh for kh in sorted(only_h) if kh[0] == kb[0] and h[kh]["run_sha"] == b[kb]["run_sha"] and kh not in renamed.values()), None)
        if match is not None:
            renamed[kb] = match
    pairs = [(k, k) for k in b if k in h] + sorted(renamed.items())
    born = [k for k in h if k in only_h and k not in renamed.values()]          # document order
    gone = [k for k in b if k in only_b and k not in renamed]

    def where(side, k):
        s = side[k]
        st = s["snap"]["step"]
        return {"job": k[0], "step": k[1], "index": s["index"], "name": st.get("name") if isinstance(st.get("name"), str) else None,
                "run_head": (st["run"].strip().splitlines() or [""])[0][:160] if isinstance(st.get("run"), str) else "",
                "continue_on_error": bool(s["fault"] and s["fault"].get("continue_on_error"))}

    for kb, kh in pairs:
        sb, sh = b[kb]["state"], h[kh]["state"]
        if sh == "hidden" and sb in FIRES:
            rec = where(h, kh)
            rec.update(kind="acquired" if sb == "loud" else "became a check, hidden", verdict=h[kh]["verdict"], verdict_before=b[kb]["verdict"],
                       mechanism=H.mechanism({"step": b[kb]["snap"]["step"], "job": b[kb]["snap"]["job"]}, {"step": h[kh]["snap"]["step"], "job": h[kh]["snap"]["job"]}),
                       renamed_from=kb[1] if kb != kh else None)
            out["new_hidden"].append(rec)
        elif sh == "hidden" and sb == "unread":
            rec = where(h, kh)
            rec.update(kind="hidden after unread", verdict=h[kh]["verdict"], verdict_before=b[kb]["verdict"])
            out["hidden_after_unread"].append(rec)
        elif sb == "hidden" and sh in FIRES:
            rec = where(h, kh)
            rec.update(kind="repaired" if sh == "loud" else "no longer a check", verdict=h[kh]["verdict"], verdict_before=b[kb]["verdict"],
                       mechanism=H.mechanism({"step": b[kb]["snap"]["step"], "job": b[kb]["snap"]["job"]}, {"step": h[kh]["snap"]["step"], "job": h[kh]["snap"]["job"]}))
            out["removed_hidden"].append(rec)
        elif sb == "hidden" and sh == "hidden":
            rec = where(h, kh)
            rec.update(verdict=h[kh]["verdict"], changed=b[kb]["run_sha"] != h[kh]["run_sha"])
            out["still_hidden"].append(rec)
        elif sb == "hidden" and sh == "unread":
            rec = where(h, kh)
            rec.update(kind="unread after hidden", verdict=h[kh]["verdict"], verdict_before=b[kb]["verdict"])
            out["unread_after_hidden"].append(rec)
    for k in born:
        if h[k]["state"] == "hidden":
            rec = where(h, k)
            rec.update(kind="born hidden", verdict=h[k]["verdict"], verdict_before=None, mechanism=None, renamed_from=None)
            out["new_hidden"].append(rec)
    for k in gone:
        if b[k]["state"] == "hidden":
            rec = where(b, k)
            rec.update(kind="removed", verdict=None, verdict_before=b[k]["verdict"], mechanism=None)
            out["removed_hidden"].append(rec)
    if fix and head_text is not None:
        for rec in out["new_hidden"]:
            rec["fix"] = fix_for(head_text, wf_name, rec["job"], rec["index"], reader.runner)
    out["fires"] = bool(out["new_hidden"])
    return out


def fix_for(text: str, wf_name: str, jid: str, i: int, runner) -> dict:
    """SWALLOW-4's repairs, then SWALLOW-5's, on HEAD's text: the first verified one with its diff."""
    r = repair.try_repairs(text, wf_name, jid, i, runner)
    stage = "swallow-4"
    if r.get("verified_repair") is None:
        r2 = rs.try_structural(text, wf_name, jid, i, runner)
        if r2.get("verified_repair") is not None:
            r, stage = r2, "swallow-5"
    v = r.get("verified_repair")
    c = next((c for c in r.get("candidates", []) if c["repair"] == v), None) if v else None
    return {"verified_repair": v, "stage": stage if v else None, "lines_changed": c["lines_changed"] if c else None,
            "diff": c["diff"] if c else None,
            "why_not": None if v else "; ".join(f"{c['repair']}: {c.get('why', '')}" for c in r.get("candidates", []) if c.get("why"))[:200]}


def audit_commit(clone: Path, base: str, head: str, readers: dict | None = None, fix: bool = True) -> dict:
    """The gate at one commit: every changed hand-written workflow between BASE and HEAD."""
    t0 = time.time()
    readers = readers if readers is not None else {}
    out = {"base": base, "head": head, "workflows": [], "fires": False, "new_hidden": 0, "removed_hidden": 0, "still_hidden": 0,
           "hidden_after_unread": 0}
    for ch in changed_workflows(clone, base, head):
        path, before = ch["path"], ch["from"]
        wf_name = path.rsplit("/", 1)[-1]
        reader = readers.setdefault(path, H.Reader())
        bt = None if ch["status"] == "A" else _text(clone, base, before)
        ht = None if ch["status"] == "D" else _text(clone, head, path)
        w = audit_pair(bt, ht, wf_name, reader, fix=fix)
        w.update(path=path, status=ch["status"])
        out["workflows"].append(w)
        for k in ("new_hidden", "removed_hidden", "still_hidden", "hidden_after_unread"):
            out[k] += len(w[k])
    out["fires"] = out["new_hidden"] > 0
    out["seconds"] = round(time.time() - t0, 2)
    return out


# ----------------------------------------------------------------------------- the population

def repo_differential(clone: Path, tip: str, repo: str | None = None, deadline: float | None = None, sample_every: int = SAMPLE_EVERY) -> dict:
    """The gate at every mainline commit of `clone` up to `tip` that touches a hand-written workflow."""
    t0 = time.time()
    out = {"repo": repo, "tip": tip, "commits": [], "capped": False, "sample": []}
    commits = [c for c in H.mainline(clone, tip) if any(not ch["path"].endswith(".lock.yml") for ch in c["changed"])]
    out["mainline_commits"] = len(commits)
    readers: dict = {}
    for n, c in enumerate(commits):
        if deadline and time.time() > deadline:
            out["capped"] = True
            break
        parent = c["parents"][0] if c["parents"] else None
        if parent is None:                                 # a root (or the shallow boundary): everything is born here
            base = None
        else:
            base = parent
        a = audit_commit_or_root(clone, base, c["sha"], readers)
        rec = {"sha": c["sha"], "time": c["time"], "subject": c["subject"][:120], "root": base is None, "fires": a["fires"],
               "new_hidden": a["new_hidden"], "removed_hidden": a["removed_hidden"], "still_hidden": a["still_hidden"],
               "hidden_after_unread": a["hidden_after_unread"], "workflows_changed": len(a["workflows"]),
               "seconds_memoised": a["seconds"]}
        if a["fires"] or a["removed_hidden"] or a["hidden_after_unread"]:
            rec["detail"] = [{"workflow": w["workflow"], "status": w["status"], "new_hidden": w["new_hidden"], "removed_hidden": w["removed_hidden"],
                              "hidden_after_unread": w["hidden_after_unread"]} for w in a["workflows"]
                             if w["new_hidden"] or w["removed_hidden"] or w["hidden_after_unread"]]
        out["commits"].append(rec)
        if sample_every and n % sample_every == 0 and base is not None:
            t1 = time.time()
            fresh = audit_commit(clone, base, c["sha"], readers={}, fix=False)
            out["sample"].append({"sha": c["sha"], "seconds": round(time.time() - t1, 2), "workflows_changed": len(fresh["workflows"]),
                                  "reads": sum(w["reads"] for w in fresh["workflows"]), "fires": fresh["fires"]})
    out["seconds"] = round(time.time() - t0, 1)
    return out


def audit_commit_or_root(clone: Path, base: str | None, head: str, readers: dict) -> dict:
    if base is not None:
        return audit_commit(clone, base, head, readers)
    # a root commit: every workflow in it is born
    t0 = time.time()
    out = {"base": None, "head": head, "workflows": [], "fires": False, "new_hidden": 0, "removed_hidden": 0, "still_hidden": 0, "hidden_after_unread": 0}
    paths = [ln.split("\t")[-1] for ln in H._git(clone, ["ls-tree", "-r", "--name-only", head, "--", ".github/workflows"], check=False).splitlines()]
    for path in paths:
        if not H._is_workflow(path) or path.endswith(".lock.yml"):
            continue
        wf_name = path.rsplit("/", 1)[-1]
        reader = readers.setdefault(path, H.Reader())
        w = audit_pair(None, _text(clone, head, path), wf_name, reader)
        w.update(path=path, status="A")
        out["workflows"].append(w)
        for k in ("new_hidden", "removed_hidden", "still_hidden", "hidden_after_unread"):
            out[k] += len(w[k])
    out["fires"] = out["new_hidden"] > 0
    out["seconds"] = round(time.time() - t0, 2)
    return out


def summary(rec: dict) -> dict:
    cs = rec["commits"]
    fires = [c for c in cs if c["fires"]]
    new = sum(c["new_hidden"] for c in cs)
    sample = [s["seconds"] for s in rec.get("sample", [])]
    sample.sort()
    return {"mainline_commits": len(cs), "firing_commits": len(fires), "new_hidden_checks": new,
            "removed_hidden_checks": sum(c["removed_hidden"] for c in cs), "hidden_after_unread": sum(c["hidden_after_unread"] for c in cs),
            "batch_firings_3_or_more": sum(1 for c in fires if c["new_hidden"] >= 3),
            "sample_n": len(sample), "sample_median_seconds": sample[len(sample) // 2] if sample else None}


def _one(repo: str, head: str | None, work: Path, deadline_per_repo: float | None) -> dict:
    clone, info = H.clone_history(repo, head, work)
    if clone is None:
        return {"repo": repo, "clone_failure": info}
    try:
        r = repo_differential(clone, info["tip"], repo, deadline=(time.time() + deadline_per_repo) if deadline_per_repo else None)
    except Exception as e:  # noqa: BLE001
        return {"repo": repo, "clone_failure": dict(info, error=f"differential: {str(e)[:160]}")}
    r["clone"] = info
    r["summary"] = summary(r)
    return r


def population(receipt6: dict, work: Path, out_path: Path | None = None, deadline_per_repo: float | None = None, limit: int | None = None,
               workers: int = 1, source_sha256: str | None = None) -> dict:
    """The gate at every mainline commit of every repository the SWALLOW-6 receipt read."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.time()
    repos = [{"repo": h["repo"], "head": h["tip"]} for h in receipt6["repos"]]
    repos = repos[:limit] if limit else repos
    order = {x["repo"]: n for n, x in enumerate(repos)}
    rec = {"schema": SCHEMA, "instrument": "benchmarks/harness_mutation/differential.py", "instrument_sha256": instrument_sha256(),
           "history_sha256": H.instrument_sha256(), "faults_sha256": H.faults_sha256(),
           "action_checks_sha256": H._sha(ROOT / "benchmarks/harness_mutation/action_checks.py"),
           "repair_sha256": H._sha(ROOT / "benchmarks/harness_mutation/repair.py"), "repair_structural_sha256": H._sha(ROOT / "benchmarks/harness_mutation/repair_structural.py"),
           "source_receipt": "papers/harness/swallow6_receipt.json.gz", "source_receipt_sha256": source_sha256, "population_size": len(repos),
           "sample_every": SAMPLE_EVERY, "workers": workers, "deadline_per_repo": deadline_per_repo, "repos": [], "clone_failures": []}
    work.mkdir(parents=True, exist_ok=True)

    def take(r: dict) -> None:
        if "clone_failure" in r:
            rec["clone_failures"].append(r["clone_failure"])
            print(f"  {r['repo']}: {r['clone_failure'].get('error')}", file=sys.stderr, flush=True)
            return
        rec["repos"].append(r)
        s = r["summary"]
        print(f"  {r['repo']}: {s['mainline_commits']} commits, fires {s['firing_commits']} ({s['new_hidden_checks']} new hidden), removed {s['removed_hidden_checks']}, "
              f"sample median {s['sample_median_seconds']}s, {r['seconds']}s{' CAPPED' if r.get('capped') else ''}", file=sys.stderr, flush=True)

    if workers <= 1:
        for x in repos:
            take(_one(x["repo"], x["head"], work, deadline_per_repo))
            if out_path:
                _write(rec, out_path, t0)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_one, x["repo"], x["head"], work, deadline_per_repo) for x in repos]
            for fut in as_completed(futs):
                take(fut.result())
                if out_path:
                    _write(rec, out_path, t0)
    rec["repos"].sort(key=lambda r: order.get(r["repo"], 10**6))
    if out_path:
        _write(rec, out_path, t0)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["summary"] = merged_summary(rec)
    return rec


def merged_summary(rec: dict) -> dict:
    keys = ("mainline_commits", "firing_commits", "new_hidden_checks", "removed_hidden_checks", "hidden_after_unread", "batch_firings_3_or_more")
    out = {k: 0 for k in keys}
    sample = []
    for r in rec["repos"]:
        s = r.get("summary") or summary(r)
        for k in keys:
            out[k] += s[k]
        sample.extend(x["seconds"] for x in r.get("sample", []))
    sample.sort()
    out["repos"] = len(rec["repos"])
    out["repos_capped"] = sum(1 for r in rec["repos"] if r.get("capped"))
    out["sample_n"] = len(sample)
    out["sample_median_seconds"] = sample[len(sample) // 2] if sample else None
    out["sample_p90_seconds"] = sample[int(len(sample) * 0.9)] if sample else None
    out["alert_rate"] = round(out["firing_commits"] / out["mainline_commits"], 4) if out["mainline_commits"] else None
    return out


def _write(rec: dict, out_path: Path, t0: float) -> None:
    rec["seconds"] = round(time.time() - t0, 1)
    rec["summary"] = merged_summary(rec)
    out_path.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def tree_differential(tree: Path, base: str, fix: bool = True) -> dict:
    """This checkout's HEAD against `base` (any revision git resolves)."""
    head = H._git(tree, ["rev-parse", "HEAD"]).strip()
    base_sha = H._git(tree, ["rev-parse", "--verify", f"{base}^{{commit}}"]).strip()
    merge_base = H._git(tree, ["merge-base", base_sha, head], check=False).strip() or base_sha
    out = audit_commit(tree, merge_base, head, readers={}, fix=fix)
    out.update(base_ref=base, merge_base=merge_base)
    return out


def instrument_sha256() -> str:
    return H._sha(Path(__file__))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tree")
    ap.add_argument("--base", help="with --tree: the revision to compare HEAD against (its merge-base with HEAD is used)")
    ap.add_argument("--receipt", help="the SWALLOW-6 receipt (.json or .json.gz): the population and its pinned HEADs")
    ap.add_argument("--work", default="s6hist")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--deadline-per-repo", type=float, default=None)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    if a.tree:
        rec = tree_differential(Path(a.tree), a.base or "origin/main")
        print(json.dumps({k: rec[k] for k in ("base_ref", "merge_base", "head", "fires", "new_hidden", "removed_hidden", "still_hidden", "seconds")}, indent=1))
        if a.out:
            Path(a.out).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return 1 if rec["fires"] else 0
    if a.receipt:
        raw = Path(a.receipt).read_bytes()
        receipt6 = json.loads((gzip.decompress(raw) if a.receipt.endswith(".gz") else raw).decode("utf-8"))
        rec = population(receipt6, Path(a.work), Path(a.out) if a.out else None, a.deadline_per_repo, a.limit, a.workers, hashlib.sha256(raw).hexdigest())
        print(json.dumps(rec["summary"], indent=1))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
