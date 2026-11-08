# -*- coding: utf-8 -*-
"""Reading a drawn row (PREREG P4): body and head by GraphQL, the head commit's CI by REST, logs and
test-report artifacts for the rows that need them.

Per row, in one pass (P4.2):
- GraphQL: body, lastEditedAt, createdAt, state, isDraft, merged, headRefOid, headRefName,
  author { __typename login }, commits(last: 1) { commit { oid committedDate pushedDate } }
  (plus databaseId, to check the row is the one the frame named). Rows are batched, 20 a query.
- Exclusions (P4.4), each recorded: LOST, NO_HEAD, EMPTY_BODY, POSTFILTER_FAIL, BOT_AUTHOR. An
  excluded row stops here: no CI listing is read for it.
- The screen (P5) on the body.
- At headRefOid, every listing paged to its end: check-runs?filter=latest; the combined status;
  Actions runs?head_sha=, each run's jobs?filter=latest (with steps) and artifacts; annotations of
  every check run whose conclusion is failure, timed_out or action_required. A github-actions check
  run whose job is not in the listings is read by /actions/jobs/{id}, as testreceipt does.
- The CI test result (P6).
- Logs and artifacts, for eligible claimers with a CI test result (any set) and for the rows that
  can become AU1 audit rows (eligible, in M, neither claimer nor implied-only, tests failed): the
  log of every completed job whose conclusion is failure or timed_out or whose name or a step is
  test-like, cut at LOG_MAX, gzipped, hashed; every report-like artifact (P18). Audit rows are the
  lowest-u rows of their kind over the whole window, so which rows they are is known only at the
  close; logs expire sooner, so every row that can become one gets its logs now.
- STALE (P8.3), LATE_READ (P4.1), rule R (P17).

Every response is stored gzipped under a2/raw/<pr_id>/<attempt>/ with its sha256 in the manifest.
The row record (a2/rows/<pr_id>.json) is written last; its presence is what makes a read done.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import time
from collections import Counter
from typing import Callable, Optional

from tpc_const import (ANNOTATED_CONCLUSIONS, ARTIFACT_MAX, BOT_EXCLUDED_AGENTS, DISK_FLOOR, DISK_PATH,
                       DISK_RESUME, LATE_READ_AFTER, LOG_MAX, LOGGED_JOB_CONCLUSIONS, POSTFILTER,
                       READ_WINDOW_END, REPORT_CHECK_NAME, iso, parse_utc)
from tpc_reports import inspect_zip, report_like_name
from tpc_rule import clean_lines, evaluate as rule_r, summary_hits
from tpc_screen import build_ci, reconcile_verdict, screen, test_like_job
from tpc_store import Store, dumps, sha256_bytes

REPORT_CHECK_RE = re.compile(REPORT_CHECK_NAME)

PR_FRAGMENT = """fragment PRF on PullRequest {
  databaseId body lastEditedAt createdAt state isDraft merged headRefOid headRefName
  author { __typename login }
  commits(last: 1) { nodes { commit { oid committedDate pushedDate } } }
}"""
BATCH = 20


def gql_literal(s: str) -> str:
    return json.dumps(s)


def batch_query(rows: list) -> str:
    parts = []
    for i, r in enumerate(rows):
        owner, name = r["repo"].split("/", 1)
        parts.append(f"  p{i}: repository(owner: {gql_literal(owner)}, name: {gql_literal(name)}) "
                     f"{{ pullRequest(number: {int(r['number'])}) {{ ...PRF }} }}")
    return "query { rateLimit { cost remaining resetAt }\n" + "\n".join(parts) + "\n}\n" + PR_FRAGMENT


def fetch_pr_nodes(gh, store: Store, guard, rows: list, attempt: str) -> dict:
    """{pr_id: (node or None, error text or None)} for a batch of rows. The seal guard is asked for
    every row before the query is sent; the response's own createdAt is checked again after."""
    for r in rows:
        guard.check(r.get("created_at"), f"body read of {r.get('repo')}#{r.get('number')}")
    out = {}
    named = []
    for r in rows:  # a frame node with a null repository cannot be read back: LOST, no query
        if r.get("repo") and r.get("number") is not None:
            named.append(r)
        else:
            out[r["pr_id"]] = (None, "no repository or number in the frame")
    if not named:
        return out
    rows = named
    q = batch_query(rows)
    payload = gh.graphql(q, {}, kind="graphql", what="pr_batch:" + ",".join(str(r["pr_id"]) for r in rows))
    raw = dumps(payload)
    store.write_gz(f"a2/raw/_graphql/{attempt}-{rows[0]['pr_id']}.json", raw)
    errors = payload.get("errors") or []
    data = payload.get("data") or {}
    for i, r in enumerate(rows):
        node = (data.get(f"p{i}") or {}).get("pullRequest") if data.get(f"p{i}") else None
        errs = [e.get("message") for e in errors if (e.get("path") or [None])[0] == f"p{i}"]
        if node is not None:
            guard.check(node.get("createdAt"), f"body read of {r.get('repo')}#{r.get('number')} (response)")
        out[r["pr_id"]] = (node, "; ".join(m for m in errs if m) or None)
    return out


class DiskGate:
    """P4.3: under 4 GB free stop logs and artifacts; resume above 6 GB."""

    def __init__(self, path: str = DISK_PATH, floor: int = DISK_FLOOR, resume: int = DISK_RESUME,
                 free: Callable[[str], int] = lambda p: shutil.disk_usage(p).free) -> None:
        self.path, self.floor, self.resume, self._free = path, floor, resume, free
        self.paused = False
        self.check()

    def check(self) -> bool:
        f = self._free(self.path)
        if self.paused and f > self.resume:
            self.paused = False
        elif not self.paused and f < self.floor:
            self.paused = True
        self.last_free = f
        return not self.paused


def _paged(gh, store: Store, rawdir: str, label: str, path: str, key: Optional[str]) -> tuple:
    items, pages, complete = gh.rest_paged(path, key)
    for i, r in enumerate(pages, 1):
        store.write_gz(f"{rawdir}/{label}.p{i}.json", r.raw or b"", note={"http": r.status})
    status = pages[-1].status if pages else None
    return items, complete, status


def exclusions(row: dict, node: Optional[dict]) -> list:
    if node is None:
        return ["LOST"]
    out = []
    body = node.get("body") or ""
    if not node.get("headRefOid"):
        out.append("NO_HEAD")
    if not body.strip():
        out.append("EMPTY_BODY")
    need = POSTFILTER.get(row["agent"])
    if need and need not in body.lower():
        out.append("POSTFILTER_FAIL")
    author = node.get("author") or {}
    if row["agent"] in BOT_EXCLUDED_AGENTS and author.get("__typename") == "Bot":
        out.append("BOT_AUTHOR")
    return out


def stale_of(node: dict) -> Optional[bool]:
    """P8.3: lastEditedAt (createdAt when never edited) before the head commit's committedDate."""
    commits = ((node.get("commits") or {}).get("nodes") or [])
    commit = (commits[-1] or {}).get("commit") if commits else None
    if not commit or not commit.get("committedDate"):
        return None
    edited = node.get("lastEditedAt") or node.get("createdAt")
    if not edited:
        return None
    return parse_utc(edited) < parse_utc(commit["committedDate"])


def read_row(gh, store: Store, guard, row: dict, node: Optional[dict], node_err: Optional[str],
             now: dt.datetime, attempt: str, disk: DiskGate, *, backlog: bool,
             evidence: Optional[Callable] = None) -> dict:
    t0 = time.time()
    calls0 = dict(gh.c.calls)
    guard.check(row.get("created_at"), f"CI read of {row.get('repo')}#{row.get('number')}")
    rawdir = f"a2/raw/{row['pr_id']}/{attempt}"
    created = parse_utc(row["created_at"])
    lag = now - created
    rec: dict = {
        "pr_id": row["pr_id"], "repo": row["repo"], "number": row["number"], "agent": row["agent"],
        "created_at": row["created_at"], "day": row["day"], "u": row["u"], "v": row["v"],
        "in_D": row["in_D"], "in_M": row["in_M"], "frame_flags": row.get("flags") or [],
        "read_at": iso(now), "read_lag_hours": round(lag.total_seconds() / 3600, 2), "backlog": backlog,
        "raw_dir": rawdir, "flags": [], "exclusions": [],
    }
    if lag > LATE_READ_AFTER:
        rec["flags"].append("LATE_READ")
    if not backlog and lag > READ_WINDOW_END:
        rec["flags"].append("READ_AFTER_84H")
    store.write_gz(f"{rawdir}/pr_node.json", dumps(node), note={"graphql_error": node_err})
    rec["exclusions"] = exclusions(row, node)
    if node is not None:
        commits = ((node.get("commits") or {}).get("nodes") or [])
        commit = (commits[-1] or {}).get("commit") if commits else {}
        body = node.get("body") or ""
        rec.update({
            "node_pr_id": node.get("databaseId"), "node_created_at": node.get("createdAt"),
            "state": node.get("state"), "is_draft": node.get("isDraft"), "merged": node.get("merged"),
            "head_sha": node.get("headRefOid"), "head_ref": node.get("headRefName"),
            "last_edited_at": node.get("lastEditedAt"),
            "head_commit_oid": (commit or {}).get("oid"), "head_committed_at": (commit or {}).get("committedDate"),
            "author": (node.get("author") or {}).get("login"),
            "author_type": (node.get("author") or {}).get("__typename"),
            "body_sha256": sha256_bytes(body.encode("utf-8")), "body_chars": len(body),
        })
        if node.get("databaseId") not in (None, row["pr_id"]):
            rec["flags"].append("PR_ID_MISMATCH")
        if node.get("createdAt") and node.get("createdAt") != row["created_at"]:
            rec["flags"].append("CREATED_AT_MISMATCH")
        rec["stale"] = stale_of(node)
        if rec["stale"]:
            rec["flags"].append("STALE")
    else:
        rec["graphql_error"] = node_err
    rec["eligible"] = not rec["exclusions"]
    if not rec["eligible"]:
        rec["finished"] = True
        rec["requests"] = {k: gh.c.calls[k] - calls0.get(k, 0) for k in gh.c.calls}
        rec["secs"] = round(time.time() - t0, 2)
        return rec

    # ---- P5, the screen ----------------------------------------------------------------------
    sc = screen(body)
    rec["screen"] = sc["screen"]
    rec["fail_claim"] = sc["fail_claim"]
    rec["claims"] = sc["claims"]

    # ---- P4.2, the CI listings at the head commit --------------------------------------------
    repo, head = row["repo"], node["headRefOid"]
    crs, crs_ok, crs_st = _paged(gh, store, rawdir, "check_runs", f"/repos/{repo}/commits/{head}/check-runs?filter=latest", "check_runs")
    sts, sts_ok, sts_st = _paged(gh, store, rawdir, "status", f"/repos/{repo}/commits/{head}/status", "statuses")
    runs, runs_ok, runs_st = _paged(gh, store, rawdir, "runs", f"/repos/{repo}/actions/runs?head_sha={head}", "workflow_runs")
    jobs: list = []
    arts: list = []
    jobs_ok = arts_ok = True
    for w in runs:
        j, ok, _ = _paged(gh, store, rawdir, f"jobs.{w['id']}", f"/repos/{repo}/actions/runs/{w['id']}/jobs?filter=latest", "jobs")
        jobs += j
        jobs_ok = jobs_ok and ok
        a, ok, _ = _paged(gh, store, rawdir, f"artifacts.{w['id']}", f"/repos/{repo}/actions/runs/{w['id']}/artifacts", "artifacts")
        for x in a:
            x["_run_id"] = w["id"]
        arts += a
        arts_ok = arts_ok and ok
    jobs_by_id = {j["id"]: j for j in jobs}
    fallback = 0
    for c in crs:
        app = (c.get("app") or {}).get("slug")
        concl = "failure" if c.get("conclusion") == "timed_out" else c.get("conclusion")
        if app == "github-actions" and concl in ("success", "failure") and c.get("id") not in jobs_by_id:
            r = gh.rest(f"/repos/{repo}/actions/jobs/{c['id']}")
            store.write_gz(f"{rawdir}/job.{c['id']}.json", r.raw or b"", note={"http": r.status})
            fallback += 1
            if r.status == 200 and isinstance(r.data, dict):
                jobs_by_id[c["id"]] = r.data
                r.data["_fallback"] = True
    annotations: list = []
    for c in crs:
        if c.get("conclusion") in ANNOTATED_CONCLUSIONS:
            an, _, _ = _paged(gh, store, rawdir, f"annotations.{c['id']}", f"/repos/{repo}/check-runs/{c['id']}/annotations", None)
            for x in an:
                annotations.append({"check_run_id": c["id"], "level": x.get("annotation_level"),
                                    "message": (x.get("message") or "")[:300], "path": x.get("path")})

    # ---- P6, the CI test result --------------------------------------------------------------
    result, steps_source = build_ci(head, crs, jobs_by_id, sts)
    outcome, evidence_name = result.tests()
    has_ci = outcome in ("tests failed", "tests passed")
    all_jobs = list(jobs_by_id.values())
    failed_jobs = [j for j in all_jobs if j.get("conclusion") in LOGGED_JOB_CONCLUSIONS]
    failed_crs = [c for c in crs if c.get("conclusion") in ("failure", "timed_out")]
    failed_sts = [s for s in sts if s.get("state") in ("failure", "error")]
    rec["ci"] = {
        "check_runs": len(crs), "statuses": len(sts), "actions_runs": len(runs), "jobs": len(all_jobs),
        "jobs_fallback": fallback, "artifacts": len(arts), "annotations": len(annotations),
        "complete": {"check_runs": crs_ok, "status": sts_ok, "runs": runs_ok, "jobs": jobs_ok, "artifacts": arts_ok},
        "http": {"check_runs": crs_st, "status": sts_st, "runs": runs_st},
        "apps": dict(Counter((c.get("app") or {}).get("slug") for c in crs)),
        "outcome": outcome, "evidence": evidence_name, "has_ci_test_result": has_ci,
        "steps_missing_for": [k for k, v in steps_source.items() if v == "missing"],
        "failed_jobs": [{"id": j["id"], "name": j.get("name"), "conclusion": j.get("conclusion")} for j in failed_jobs],
        "failed_check_runs": [{"id": c["id"], "name": c.get("name"), "app": (c.get("app") or {}).get("slug"),
                               "conclusion": c.get("conclusion")} for c in failed_crs],
        "failed_statuses": [{"context": s.get("context"), "state": s.get("state")} for s in failed_sts],
        "any_failed_job_or_check": bool(failed_jobs or failed_crs),
        "report_check_runs": [c.get("name") for c in crs
                              if REPORT_CHECK_RE.search(c.get("name") or "")
                              and ((c.get("output") or {}).get("summary") or "").strip()],
        "check_list": [{"id": c.get("id"), "name": c.get("name"), "app": (c.get("app") or {}).get("slug"),
                        "status": c.get("status"), "conclusion": c.get("conclusion"),
                        "title": ((c.get("output") or {}).get("title") or "")[:200]} for c in crs],
        "status_list": [{"context": s.get("context"), "state": s.get("state")} for s in sts],
        "annotation_list": annotations[:40],
    }
    rec["reconcile_verdict"] = reconcile_verdict(body, outcome, evidence_name)

    # ---- logs and artifacts --------------------------------------------------------------------
    au1_eligible = row["in_M"] and sc["screen"] == "none" and outcome == "tests failed"
    wants = (sc["screen"] == "claimer" and has_ci) or au1_eligible
    rec["logs_wanted"] = bool(wants)
    rec["logs_reason"] = ("claimer_with_ci" if sc["screen"] == "claimer" and has_ci
                          else "au1_eligible" if au1_eligible else None)
    rec["logs"] = []
    rec["artifacts"] = []
    log_hits: dict = {}
    if wants and not disk.check():
        rec["flags"].append("LOGS_SKIPPED")
        wants = False
    if wants:
        for j in sorted(all_jobs, key=lambda x: x["id"]):
            failed = j.get("conclusion") in LOGGED_JOB_CONCLUSIONS
            testy = test_like_job(j)
            if not (failed or testy):
                continue
            entry = {"job_id": j["id"], "name": j.get("name"), "conclusion": j.get("conclusion"),
                     "status": j.get("status"), "why": "both" if failed and testy else "failed" if failed else "test_like"}
            if j.get("status") != "completed":
                entry["fetched"] = False
                entry["skip"] = "not_completed"
                rec["logs"].append(entry)
                continue
            if j.get("conclusion") == "skipped":  # a skipped job ran nothing: GitHub serves no log (404)
                entry["fetched"] = False
                entry["skip"] = "skipped_job"
                rec["logs"].append(entry)
                continue
            if not disk.check():
                entry["fetched"] = False
                entry["skip"] = "disk_floor"
                rec["logs"].append(entry)
                if "LOGS_SKIPPED" not in rec["flags"]:
                    rec["flags"].append("LOGS_SKIPPED")
                continue
            api_st, blob_st, raw, trunc = gh.follow(f"/repos/{repo}/actions/jobs/{j['id']}/logs",
                                                    f"joblog:{repo}#{row['number']}:{j['id']}", max_bytes=LOG_MAX)
            entry.update({"api_status": api_st, "blob_status": blob_st, "fetched": bool(raw)})
            if raw:
                w = store.write_gz(f"{rawdir}/logs/{j['id']}.log", raw)
                lines = clean_lines(raw)
                hits = summary_hits(lines)
                log_hits[j["id"]] = hits
                entry.update({"path": w["path"], "raw_bytes": len(raw), "gz_bytes": w["bytes"],
                              "raw_sha256": w["raw_sha256"], "truncated_at_log_max": trunc,
                              "summary_hits": len(hits), "summary_first": [h[2] for h in hits[:3]]})
            rec["logs"].append(entry)
        for a in sorted(arts, key=lambda x: x["id"]):
            if not report_like_name(a.get("name") or ""):
                continue
            entry = {"id": a["id"], "name": a.get("name"), "bytes": a.get("size_in_bytes"),
                     "expired": a.get("expired"), "created_at": a.get("created_at"),
                     "expires_at": a.get("expires_at"), "run_id": a.get("_run_id")}
            if a.get("expired"):
                entry["skip"] = "expired"
            elif (a.get("size_in_bytes") or 0) > ARTIFACT_MAX:
                entry["skip"] = "over_artifact_max"
            elif not disk.check():
                entry["skip"] = "disk_floor"
            else:
                api_st, blob_st, raw, _ = gh.follow(f"/repos/{repo}/actions/artifacts/{a['id']}/zip",
                                                    f"artifact:{repo}#{row['number']}:{a['id']}",
                                                    max_bytes=ARTIFACT_MAX + 1)
                entry.update({"api_status": api_st, "blob_status": blob_st})
                if raw:
                    w = store.write(f"{rawdir}/artifacts/{a['id']}.zip", raw)
                    entry.update({"path": w["path"], "zip_sha256": w["sha256"]})
                    found = inspect_zip(raw)
                    entry["bad_zip"] = found["bad_zip"]
                    entry["members"] = found["members"]
                    counted = []
                    for k, m in enumerate(found["counted"]):
                        safe = re.sub(r"[^\w.\-]+", "_", m["name"])[-120:]
                        wm = store.write(f"{rawdir}/artifacts/{a['id']}/{k:03d}-{safe}", m["_data"])
                        item = {"name": m["name"], "kind": m["kind"], "bytes": m["bytes"], "sha256": wm["sha256"]}
                        if evidence is not None:
                            try:
                                item["evidence"] = evidence([store.path(wm["path"])], head)
                            except Exception as e:  # noqa: BLE001
                                item["evidence"] = {"verdict": "READ_ERROR", "reason": f"{type(e).__name__}: {e}"[:300]}
                        counted.append(item)
                    entry["counted"] = counted
            rec["artifacts"].append(entry)
    rec["report_artifact_counted"] = any(a.get("counted") for a in rec["artifacts"])

    # ---- P17, rule R ---------------------------------------------------------------------------
    rec["rule_r"] = rule_r(sc["claims"], sc["fail_claim"], rec.get("stale"), all_jobs, log_hits)
    rec["finished"] = True
    rec["requests"] = {k: gh.c.calls[k] - calls0.get(k, 0) for k in gh.c.calls}
    rec["secs"] = round(time.time() - t0, 2)
    return rec
