# -*- coding: utf-8 -*-
"""Packet items (PREREG P8): the description a seat reads in step 1 and the CI record it reads in step 2.

An item is two parts (P8.1): DESCRIPTION, shown in step 1, and CI AT THE HEAD COMMIT, shown only in step 2. No item
carries a URL, repository name, pull request number, author, agent name, date or state.

- `clean_description` (P8.2): the body as read, line endings made LF; HTML comments removed (an unclosed comment to
  the end, as GitHub renders it); images removed (markdown and <img>); agent footer lines (P2.1's signature texts,
  Codex task links, Devin session links) replaced by "[agent footer removed]"; every URL replaced by "<url>"; cut at
  DESC_MAX with "[cut]". Repository words inside the prose stay as they are.
- `render_ci` (P8.3): per check run `check: <name> (<app slug>) -> <conclusion>`; per status
  `status: <context> -> <state>`; per Actions job that failed or is test-like `job: <name> -> <conclusion>` and its
  steps; up to 20 annotations of failed check runs; log excerpts of failing test-like jobs (at most 3, job-id order);
  then the one fixed line. URLs in the block are replaced by "<url>" and the repository's own owner/name by "<repo>"
  (P8.1 says no item carries either; P8.3 does not say how, so this is the reading the code makes).
- `log_excerpt` (P8.3): the log decoded as UTF-8 with replacement, ANSI escapes and each line's leading timestamp
  removed. For each failing step of the job (failure or timed_out), in step-number order, a span of lines:
  * by the step's times: it starts at the earliest line that begins `##[group]Run ` and is stamped from one second
    before the step's started_at to one second after its completed_at (a composite step's several Run groups are one
    step); it ends before the earliest later line stamped more than one second after completed_at (an unstamped line
    takes the latest stamp before it) or beginning "Post job cleanup." (the line that opens the post steps),
    whichever comes sooner; and when the span holds `##[error]` lines, it ends just after the last of them (a failed
    step's output closes with one, "Process completed with exit code N.");
  * with no span found by any failing step's times: from the last `##[group]Run ` line before the log's last
    `##[error]` line to that line;
  * failing that, the whole log.
  Of each span, the last 150 lines; the spans joined in step order; 12,000 characters per job and 30,000 per item,
  "[cut]" included (the tail kept). The boundaries are step times and the runner's own markers; no pattern chooses a
  line. Which way was used is recorded beside the item, never shown to a seat. Why the two markers: GitHub stamps
  steps to the second, so the post steps that begin in the second the failing step ended fall inside its time range,
  and their checkout cleanup would otherwise fill the last 150 lines and push the failure out (seen on pre-window
  logs while P2's decoy material was read).

The step-1 text is the cleaned description; the step-2 item text is
"DESCRIPTION\\n" + description + "\\n\\nCI AT THE HEAD COMMIT\\n" + the CI block (P12.3 sends it after the prompt and
the seat's own step-1 answer). An item's sha256 is that of its step-2 item text in UTF-8.

Adapters build the one CI record shape `render_ci` reads from: a collector row and its raw store (A2, `ci_from_row`),
the A1 snapshot of testreceipt's items (`ci_from_snapshot`), and testreceipt's transcription of failing checks for
the 2025 items GitHub no longer serves (`ci_from_transcription`).
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Optional

from tpc_const import (ANNOTATED_CONCLUSIONS, ANNOTATION_CHARS, ANNOTATIONS_MAX, DESC_MAX, LOG_EXCERPT_ITEM_CHARS,
                       LOG_EXCERPT_JOB_CHARS, LOG_EXCERPT_JOBS, LOG_EXCERPT_LINES, parse_utc)
from tpc_rule import ANSI
from tpc_screen import _map, is_test_step, test_like_job

DESC_HEAD = "DESCRIPTION"
CI_HEAD = "CI AT THE HEAD COMMIT"
FIXED_LINE = ("These checks are the latest attempt of each check at the head commit, read some days after the pull "
              "request was opened.")
NAMES_ONLY_LINE = "Only the names and conclusions of the failing checks survive for this pull request."
URL_TOKEN = "<url>"
REPO_TOKEN = "<repo>"
FOOTER_TOKEN = "[agent footer removed]"
CUT_TOKEN = "[cut]"

HTML_COMMENT = re.compile(r"<!--.*?(?:-->|\Z)", re.S)
MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)\n]*\)")
HTML_IMG = re.compile(r"<img\b[^>]*>", re.I)
URL = re.compile(r"(?i)\b(?:https?|ftp)://[^\s<>()\[\]\"'`]+")
URL_TRAIL = ".,;:!?*_~"
# P8.2: P2.1's signature texts (Claude Code's footer, Cursor's agent link and its post-filter text), Codex task links,
# Devin session links
FOOTER = re.compile(r"(?i)generated with \[?claude code\b|cursor\.com/|chatgpt\.com/codex/tasks|app\.devin\.ai/sessions")
LOG_TIMESTAMP = re.compile(r"^﻿?(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.\d+)?Z ?")
RUN_GROUP = "##[group]Run "
ERROR_MARK = "##[error]"
POST_STEPS = "Post job cleanup."
ONE_SECOND = dt.timedelta(seconds=1)


# ---- P8.2, the description ----------------------------------------------------------------------------------
def _replace_urls(text: str) -> str:
    def sub(m: "re.Match[str]") -> str:
        s = m.group(0)
        tail = ""
        while s and s[-1] in URL_TRAIL:
            tail = s[-1] + tail
            s = s[:-1]
        return URL_TOKEN + tail
    return URL.sub(sub, text)


def clean_description(body: Optional[str]) -> str:
    text = (body or "").replace("\r\n", "\n").replace("\r", "\n")
    text = HTML_COMMENT.sub("", text)
    text = MD_IMAGE.sub("", text)
    text = HTML_IMG.sub("", text)
    lines = [FOOTER_TOKEN if FOOTER.search(ln) else ln for ln in text.split("\n")]
    text = _replace_urls("\n".join(lines)).strip()
    if len(text) > DESC_MAX:
        text = text[:DESC_MAX] + "\n" + CUT_TOKEN
    return text


# ---- P8.3, the log excerpt ----------------------------------------------------------------------------------
def _parse_log(raw: "bytes | str") -> list:
    """[(the line's own timestamp or None, the cleaned line, the latest timestamp seen at or before it)]."""
    text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else raw
    out = []
    last = None
    for ln in text.splitlines():
        ln = ANSI.sub("", ln)
        m = LOG_TIMESTAMP.match(ln)
        own = None
        if m:
            try:
                own = last = parse_utc(m.group(1) + "Z")
            except ValueError:
                own = None
            ln = ln[m.end():]
        out.append((own, ln, last))
    return out


def _step_span(lines: list, step: dict) -> Optional[tuple]:
    """(start, end) line indices of a step's lines by its start and end times, or None."""
    try:
        t0 = parse_utc(step["started_at"]) - ONE_SECOND
        t1 = parse_utc(step["completed_at"]) + ONE_SECOND
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    start = None
    for i, (own, ln, inh) in enumerate(lines):
        if own is not None and ln.startswith(RUN_GROUP) and t0 <= own <= t1:
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for i in range(start + 1, len(lines)):
        t = lines[i][2]
        if (t is not None and t > t1) or lines[i][1].startswith(POST_STEPS):
            end = i
            break
    return start, _end_at_error(lines, start, end)


def _end_at_error(lines: list, start: int, end: int) -> int:
    """A failed step's output closes with its ##[error] line; the lines after the span's last one belong to the steps
    that follow in the same second."""
    errs = [i for i in range(start, end) if lines[i][1].startswith(ERROR_MARK)]
    return errs[-1] + 1 if errs else end


def _error_span(lines: list) -> Optional[tuple]:
    """The Run group holding the log's last ##[error] line: from its Run line to that ##[error] line."""
    errs = [i for i, (_, ln, _) in enumerate(lines) if ln.startswith(ERROR_MARK)]
    runs = [i for i, (_, ln, _) in enumerate(lines) if ln.startswith(RUN_GROUP)]
    if not errs or not runs:
        return None
    e = errs[-1]
    before = [r for r in runs if r <= e]
    if not before:
        return None
    start = before[-1]
    after = [r for r in runs if r > start]
    return start, _end_at_error(lines, start, after[0] if after else len(lines))


def log_excerpt(raw: "bytes | str", job: dict) -> tuple:
    """(excerpt text, how) for one failing job's log (P8.3)."""
    lines = _parse_log(raw)
    failing = [s for s in sorted(job.get("steps") or [], key=lambda s: (s.get("number") or 0))
               if _map(s.get("conclusion")) == "failure"]
    spans = []
    for s in failing:
        sp = _step_span(lines, s)
        if sp is not None and sp not in spans:
            spans.append(sp)
    how = "step_times"
    if not spans:
        sp = _error_span(lines)
        if sp is not None:
            spans, how = [sp], "error_marker"
        else:
            spans, how = [(0, len(lines))], "log_tail"
    out = []
    for a, b in spans:
        seg = [ln for _, ln, _ in lines[a:b]]
        out += seg[-LOG_EXCERPT_LINES:]
    return _tail("\n".join(out), LOG_EXCERPT_JOB_CHARS), how


def _tail(text: str, cap: int) -> str:
    """The text's last characters, at most `cap` in all: "[cut]" and a line feed lead when anything was cut."""
    if len(text) <= cap:
        return text
    keep = max(0, cap - len(CUT_TOKEN) - 1)
    return CUT_TOKEN + "\n" + (text[-keep:] if keep else "")


# ---- P8.3, the CI block ---------------------------------------------------------------------------------------
def _redact(text: str, repo: Optional[str]) -> str:
    text = _replace_urls(text)
    if repo:
        text = re.sub(re.escape(repo), REPO_TOKEN, text, flags=re.I)
    return text


def _one_line(s: Optional[str]) -> str:
    return " ".join((s or "").split())


def failing_test_like_jobs(ci: dict) -> list:
    return [j for j in sorted(ci.get("jobs") or [], key=lambda j: j["id"])
            if _map(j.get("conclusion")) == "failure" and test_like_job(j)]


def render_ci(ci: dict, repo: Optional[str] = None) -> tuple:
    """(CI block text, build info). `ci`: {check_runs, statuses, jobs, annotations, logs, names_only}; logs maps a job
    id (str) to its raw log (bytes or str)."""
    lines = []
    if ci.get("names_only"):
        lines.append(NAMES_ONLY_LINE)
    for c in sorted(ci.get("check_runs") or [], key=lambda c: ((c.get("name") or "").lower(), c.get("name") or "",
                                                                 c.get("id") or 0)):
        concl = c.get("conclusion") or c.get("status") or "none"
        lines.append(f"check: {_one_line(c.get('name'))} ({c.get('app') or 'none'}) -> {concl}")
    for s in sorted(ci.get("statuses") or [], key=lambda s: ((s.get("context") or "").lower(), s.get("context") or "")):
        lines.append(f"status: {_one_line(s.get('context'))} -> {s.get('state')}")
    for j in sorted(ci.get("jobs") or [], key=lambda j: j["id"]):
        if not (_map(j.get("conclusion")) == "failure" or test_like_job(j)):
            continue
        lines.append(f"job: {_one_line(j.get('name'))} -> {j.get('conclusion') or j.get('status') or 'none'}")
        for st in sorted(j.get("steps") or [], key=lambda s: (s.get("number") or 0)):
            lines.append(f"step: {_one_line(st.get('name'))} -> {st.get('conclusion') or st.get('status') or 'none'}")
    failed_ids = {c.get("id") for c in ci.get("check_runs") or [] if c.get("conclusion") in ANNOTATED_CONCLUSIONS}
    ann = [a for a in ci.get("annotations") or [] if a.get("check_run_id") in failed_ids]
    for a in ann[:ANNOTATIONS_MAX]:
        lines.append(f"annotation: {a.get('level') or 'none'}: {_one_line(a.get('message'))[:ANNOTATION_CHARS]}")
    info = {"logs": [], "annotations_shown": min(len(ann), ANNOTATIONS_MAX), "annotations_listed": len(ann)}
    budget = LOG_EXCERPT_ITEM_CHARS
    logs = ci.get("logs") or {}
    for j in failing_test_like_jobs(ci)[:LOG_EXCERPT_JOBS]:
        raw = logs.get(str(j["id"]))
        if raw is None:
            info["logs"].append({"job_id": j["id"], "how": "no_log"})
            continue
        if budget <= 0:
            info["logs"].append({"job_id": j["id"], "how": "item_budget_spent"})
            continue
        text, how = log_excerpt(raw, j)
        text = _tail(text, budget)
        budget -= len(text)
        lines.append("")
        lines.append(f"log: {_one_line(j.get('name'))}")
        lines.append(text)
        info["logs"].append({"job_id": j["id"], "how": how, "chars": len(text)})
    lines.append("")
    lines.append(FIXED_LINE)
    return _redact("\n".join(lines), repo), info


# ---- the two texts a seat reads ----------------------------------------------------------------------------
def item_text(description: str, ci_block: str) -> str:
    return f"{DESC_HEAD}\n{description}\n\n{CI_HEAD}\n{ci_block}"


def item_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_item(body: Optional[str], ci: Optional[dict], repo: Optional[str] = None, *, steps: int = 2) -> dict:
    """{"steps", "step1_text", "item_text" (None for a step-1-only item), "item_sha256", "build"}."""
    desc = clean_description(body)
    if steps == 1:
        return {"steps": 1, "step1_text": desc, "item_text": None,
                "item_sha256": item_sha256(item_text(desc, "")), "build": {}}
    block, info = render_ci(ci or {}, repo)
    text = item_text(desc, block)
    return {"steps": 2, "step1_text": desc, "item_text": text, "item_sha256": item_sha256(text), "build": info}


# ---- adapters ---------------------------------------------------------------------------------------------
def _gz_json(p: Path):
    return json.loads(gzip.decompress(p.read_bytes()).decode("utf-8"))


def _pages(d: Path, label: str) -> list:
    """The pages <label>.p<i>.json.gz in page order."""
    found = []
    for p in d.glob(f"{label}.p*.json.gz"):
        m = re.match(re.escape(label) + r"\.p(\d+)\.json\.gz$", p.name)
        if m:
            found.append((int(m.group(1)), p))
    return [p for _, p in sorted(found)]


def _norm_check(c: dict) -> dict:
    app = c.get("app")
    return {"id": c.get("id"), "name": c.get("name"), "app": app.get("slug") if isinstance(app, dict) else app,
            "status": c.get("status"), "conclusion": c.get("conclusion")}


def _norm_job(j: dict) -> dict:
    return {"id": j.get("id"), "name": j.get("name"), "status": j.get("status"), "conclusion": j.get("conclusion"),
            "steps": [{"number": s.get("number"), "name": s.get("name"), "status": s.get("status"),
                       "conclusion": s.get("conclusion"), "started_at": s.get("started_at"),
                       "completed_at": s.get("completed_at")} for s in (j.get("steps") or [])]}


def ci_from_row(store_root: Path, rec: dict) -> dict:
    """The CI record of a collector row (P4.2's listings under the row's raw directory)."""
    d = Path(store_root) / rec["raw_dir"]
    crs = []
    for p in _pages(d, "check_runs"):
        crs += (_gz_json(p) or {}).get("check_runs") or []
    sts = []
    for p in _pages(d, "status"):
        sts += (_gz_json(p) or {}).get("statuses") or []
    jobs = {}
    for p in sorted(d.glob("jobs.*.p*.json.gz")):
        m = re.match(r"jobs\.(\d+)\.p(\d+)\.json\.gz$", p.name)
        if not m:
            continue
        for j in (_gz_json(p) or {}).get("jobs") or []:
            jobs[j["id"]] = j
    for p in sorted(d.glob("job.*.json.gz")):
        j = _gz_json(p)
        if isinstance(j, dict) and j.get("id") is not None and j["id"] not in jobs:
            jobs[j["id"]] = j
    ann = []
    for c in crs:
        if c.get("conclusion") in ANNOTATED_CONCLUSIONS:
            for p in _pages(d, f"annotations.{c['id']}"):
                for a in _gz_json(p) or []:
                    ann.append({"check_run_id": c["id"], "level": a.get("annotation_level"),
                                "message": a.get("message")})
    ci = {"check_runs": [_norm_check(c) for c in crs], "statuses": [{"context": s.get("context"),
                                                                     "state": s.get("state")} for s in sts],
          "jobs": [_norm_job(j) for j in jobs.values()], "annotations": ann, "logs": {}}
    for j in failing_test_like_jobs(ci):
        lp = d / "logs" / f"{j['id']}.log.gz"
        if lp.exists():
            ci["logs"][str(j["id"])] = gzip.decompress(lp.read_bytes())
    return ci


def body_from_row(store_root: Path, rec: dict) -> str:
    node = _gz_json(Path(store_root) / rec["raw_dir"] / "pr_node.json.gz") or {}
    return node.get("body") or ""


def ci_from_snapshot(item: dict, logs_dir: Optional[Path] = None) -> dict:
    """The CI record of one of testreceipt's items in the lab's snapshot (section 14; P8.4): check runs, the combined
    status, and for failed GitHub Actions check runs their steps, annotations and saved logs. In GitHub Actions a job's
    id is its check run's id."""
    crs = item.get("check_runs") or []
    jobs, ann, logs = [], [], {}
    for c in crs:
        if c.get("steps") is not None:
            jobs.append(_norm_job({"id": c.get("id"), "name": c.get("name"), "status": c.get("status"),
                                   "conclusion": c.get("conclusion"), "steps": c.get("steps")}))
        for a in c.get("annotations") or []:
            ann.append({"check_run_id": c.get("id"), "level": a.get("annotation_level"), "message": a.get("message")})
        if logs_dir is not None and c.get("log_file"):
            lp = Path(logs_dir) / c["log_file"]
            if lp.exists():
                logs[str(c.get("id"))] = lp.read_bytes()
    return {"check_runs": [_norm_check(c) for c in crs],
            "statuses": [{"context": s.get("context"), "state": s.get("state")} for s in item.get("statuses") or []],
            "jobs": jobs, "annotations": ann, "logs": logs}


def ci_from_transcription(failing: str) -> dict:
    """testreceipt's "all failing checks" line ("status <context>: <state>; <check name>: <conclusion>; ...") for the
    2025 items GitHub no longer serves (P8.4): names and conclusions only, under NAMES_ONLY_LINE."""
    crs, sts = [], []
    for i, part in enumerate(p.strip() for p in (failing or "").split("; ")):
        if not part or ": " not in part:
            continue
        name, _, concl = part.rpartition(": ")
        if name.startswith("status "):
            sts.append({"context": name[len("status "):], "state": concl})
        else:
            crs.append({"id": -(i + 1), "name": name, "app": "none", "status": "completed", "conclusion": concl})
    return {"check_runs": crs, "statuses": sts, "jobs": [], "annotations": [], "logs": {}, "names_only": True}


def is_test_like_name(name: str) -> bool:
    return bool(is_test_step(name or ""))
