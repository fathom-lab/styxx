# -*- coding: utf-8 -*-
"""Rule R, the candidate accusing rule (PREREG P17), measured report-only and posted nowhere; and
the log cleaning P8.3 and P17.2 share.

Cleaning (P8.3): decode UTF-8 with replacement, remove ANSI escapes and each line's leading
timestamp. P17.2's twelve summary patterns are matched per cleaned line, case-sensitive; their
bytes (the lines joined by LF, no final LF) must hash to the PREREG's lock-table digest or this
module refuses to load.

Rule R fires on a row when all four hold (P17.1):
 1. testreceipt's claims() returns an explicit pass claim of scope "full" whose line is not a
    ticked checkbox and names no filtered command, test path or local run;
 2. a failing test-like Actions step (P6) at the head commit sits in a job whose log was read and
    holds a line matching P17.2;
 3. the description is not STALE (P8.3);
 4. testreceipt's reader finds no fail claim in the description.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable, Optional

from tpc_screen import STEPS_FOR, _map, failing_test_steps

SUMMARY_LINES = (
    r"=+ .*\b\d+ failed\b",
    r"^Tests?:\s+\d+ failed",
    r"test result: FAILED",
    r"^--- FAIL:",
    r"^FAIL\s",
    r"\b\d+ failing\b",
    r"\b\d+ examples?, [1-9]\d* failures?\b",
    r"Tests run: \d+, Failures: [1-9]",
    r"Tests run: \d+, Failures: \d+, Errors: [1-9]",
    r"Failed!\s+-\s+Failed:\s+[1-9]",
    r"^FAILURES!",
    r"tests completed, \d+ failed",
)
SUMMARY_SHA256 = "205631ad1590299de48e7801652882d51e1151bd9e561f626dde587c2fce7bcb"
_got = hashlib.sha256("\n".join(SUMMARY_LINES).encode("utf-8")).hexdigest()
if _got != SUMMARY_SHA256:
    raise RuntimeError(f"rule R's failure summaries differ from the PREREG's bytes ({_got})")
SUMMARY = [re.compile(p) for p in SUMMARY_LINES]

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07")
TIMESTAMP = re.compile(r"^﻿?\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z ?")

TICKED_BOX = re.compile(r"^\s*[-*+]\s*\[[xX]\]")
# P17.1 item 1: a claim line naming a local run is a part claim under the rubric (P11.2: an
# environment CI does not run), so rule R never fires on it.
LOCAL = re.compile(r"(?i)\blocal(?:ly)?\b|\bon my (?:machine|computer|laptop)\b")
FILTERED = re.compile(
    r"\s-k\s|\s-p\s|--package|\s-pl\s|--filter|\s-t\s|--testNamePattern|--grep|\s-run\s|::"
    r"|(?:test\.py|_test\.go|\.test\.ts|\.test\.js|\.spec\.ts|\.spec\.js)(?![\w.])"
    r"|\btest_[\w-]*\.py(?![\w.])"
)


def clean_lines(raw: bytes) -> list:
    text = raw.decode("utf-8", "replace")
    out = []
    for line in text.splitlines():
        line = ANSI.sub("", line)
        line = TIMESTAMP.sub("", line)
        out.append(line)
    return out


def summary_hits(lines: Iterable[str], limit: Optional[int] = None) -> list:
    """[(line index, pattern index, line)] of cleaned lines matching P17.2."""
    hits = []
    for i, line in enumerate(lines):
        for j, pat in enumerate(SUMMARY):
            if pat.search(line):
                hits.append((i, j, line[:300]))
                break
        if limit is not None and len(hits) >= limit:
            break
    return hits


def whole_claim_lines(claims: list) -> list:
    """Item 1: explicit pass claims of scope full, not a ticked box, no filtered command, test path or local run."""
    out = []
    for c in claims:
        if c.get("kind") != "pass" or not c.get("explicit") or c.get("scope") != "full":
            continue
        text = c.get("text") or ""
        if TICKED_BOX.search(text) or FILTERED.search(" " + text + " ") or LOCAL.search(text):
            continue
        out.append(text)
    return out


def evaluate(claims: list, fail_claim: bool, stale: Optional[bool], jobs: list, log_hits: dict) -> dict:
    """claims: the screen's claim dicts; jobs: the head commit's Actions jobs (latest attempt);
    log_hits: {job_id: [hits]} for jobs whose log was read (absent = not read)."""
    item1 = whole_claim_lines(claims)
    item2 = []
    for job in jobs:
        if _map(job.get("conclusion")) not in STEPS_FOR:
            continue  # P6 attaches steps to success/failure (and timed_out) jobs only
        steps = failing_test_steps(job)
        if not steps:
            continue
        jid = job.get("id")
        if jid in log_hits and log_hits[jid]:
            item2.append({"job_id": jid, "job": job.get("name"), "steps": steps[:5],
                          "line": log_hits[jid][0][2]})
    item3 = stale is False
    item4 = not fail_claim
    fires = bool(item1) and bool(item2) and item3 and item4
    return {"fires": fires, "item1_whole_claim": bool(item1), "item2_failing_step_with_summary": bool(item2),
            "item3_not_stale": item3, "item4_no_fail_claim": item4,
            "claim_line": item1[0][:300] if item1 else None, "job": item2[0] if item2 else None}
