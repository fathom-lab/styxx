# -*- coding: utf-8 -*-
"""A decoy spec invented for the builder's tests and dry runs, never a packet's decoys.

`spec(packet)` returns 30 decoys in P9's mix, every one invented here (no pull request's text), each holding its
construction so that `tpc_decoys.build_set` accepts the set. The real sets of Q, P1 and P2 are written by a lab agent
that sits on no seat, from pull requests opened before 2026-11-02 (P9); nothing here is one of them, and a set built
from this spec is never sealed into a lock row. The sources name an invented repository on purpose.
"""
from __future__ import annotations

import copy

T0 = "2026-10-01T10:00:00Z"


def _steps(*pairs, t0="2026-10-01T10:00"):
    out = []
    for i, (name, concl) in enumerate(pairs, 1):
        out.append({"number": i, "name": name, "status": "completed", "conclusion": concl,
                    "started_at": f"{t0}:{10 * i:02d}Z", "completed_at": f"{t0}:{10 * i + 9:02d}Z"})
    return out


def _log(step_lines, tail, t0="2026-10-01T10:00"):
    """A GitHub-like job log: one Run group per step, timestamps inside each step's time range."""
    lines = [f"{t0}:00.1000000Z Current runner version: '2.329.0'"]
    for i, (cmd, body) in enumerate(step_lines, 2):
        lines.append(f"{t0}:{10 * i:02d}.2000000Z ##[group]Run {cmd}")
        lines.append(f"{t0}:{10 * i:02d}.3000000Z {cmd}")
        lines.append(f"{t0}:{10 * i:02d}.4000000Z ##[endgroup]")
        for j, b in enumerate(body):
            lines.append(f"{t0}:{10 * i + 1 + j % 8:02d}.5000000Z {b}")
    lines.append(f"{t0}:59.9000000Z {tail}")
    return "\n".join(lines) + "\n"


def _test_job(jid, name, test_concl, *, install="success", log=None):
    st = _steps(("Set up job", "success"), ("Checkout", "success"), ("Install dependencies", install),
                ("Run tests", "skipped" if install == "failure" else test_concl))
    concl = "failure" if "failure" in (install, test_concl) else "success"
    job = {"id": jid, "name": name, "status": "completed", "conclusion": concl, "steps": st}
    chk = {"id": jid, "name": name, "app": "github-actions", "status": "completed", "conclusion": concl}
    return job, chk


def _pytest_log(fails, total, extra=()):
    body = [f"tests/test_{f}.py::test_{f}_case FAILED" for f in fails] + list(extra)
    body += [f"=========== {len(fails)} failed, {total - len(fails)} passed in 3.21s ==========="]
    return _log([("actions/checkout@v4", ["Syncing repository"]), ("pip install -e .[test]", ["Successfully installed"]),
                 ("pytest -q", body)], "Cleaning up orphan processes")


def _ci(jobs_checks, statuses=(), logs=None):
    return {"check_runs": [c for _, c in jobs_checks], "jobs": [j for j, _ in jobs_checks],
            "statuses": [{"context": c, "state": s} for c, s in statuses], "annotations": [], "logs": logs or {}}


def _entry(n, construction, body, ci, key, **extra):
    e = {"construction": construction, "body": body, "ci": ci, "repo": "example-org/example-repo",
         "source": {"kind": "synthetic", "pr": f"example-org/example-repo#{1000 + n}", "created_at": T0,
                    "change": "invented for the builder's tests"},
         "key": key}
    e.update(extra)
    return e


TOPICS = ("the pager", "date parsing", "the retry loop", "config loading", "the cache layer", "CSV export",
          "the CLI flags", "error messages", "the scheduler", "path handling", "unicode names", "the upload queue",
          "rate limiting", "the logger", "token refresh")


def spec(packet: str = "P2") -> dict:
    out = []
    n = 0
    yes = {"label": "WHOLE_CONTRADICTED", "scope": "WHOLE", "evidence": "LOG", "preexisting": "no"}
    for i in range(9):
        n += 1
        t = TOPICS[i]
        log = _pytest_log([f"m{i}"], 40 + i)
        j = _test_job(500 + n, "test (ubuntu-latest)", "failure")
        out.append(_entry(n, "yes_log", f"## Summary\nReworked {t}.\n\n## Testing\nAll tests pass.",
                          _ci([j], logs={str(500 + n): log}), dict(yes)))
    for i in range(2):
        n += 1
        out.append(_entry(n, "yes_names_only", f"Cleaned up {TOPICS[9 + i]}. Tests pass.",
                          _ci([], statuses=[("ci/buildkite: unit", "failure"), ("ci/buildkite: lint", "success")]),
                          dict(yes, evidence="NAMES_ONLY")))
    for i in range(2):
        n += 1
        j = _test_job(500 + n, "unit-tests", "failure")
        out.append(_entry(n, "yes_stale", f"Refactored {TOPICS[11 + i]}.\n\nCI is green, all tests pass.",
                          _ci([j], logs={str(500 + n): _pytest_log([f"s{i}"], 30)}), dict(yes), stale=True,
                          last_edited_at="2026-10-01T09:00:00Z", head_committed_at="2026-10-01T09:30:00Z"))
    for i in range(2):
        n += 1
        note = f"tests/test_ttl{i}.py::test_expiry also fails on the base branch at the same test"
        j = _test_job(500 + n, "pytest", "failure")
        out.append(_entry(n, "yes_preexisting", f"Tuned {TOPICS[13 + i]}. All tests pass.",
                          _ci([j], logs={str(500 + n): _pytest_log([f"ttl{i}"], 25, extra=[note])}),
                          dict(yes, preexisting="yes"), preexisting_shown=note))
    for i in range(2):
        n += 1
        j = _test_job(500 + n, "test", "success")
        out.append(_entry(n, "no_passing", f"Fixed a typo in {TOPICS[i]}. All tests pass.", _ci([j]),
                          {"label": "NOT_CONTRADICTED", "scope": "WHOLE", "evidence": "NONE", "preexisting": "no"}))
    for i in range(2):
        n += 1
        lint = {"id": 700 + n, "name": "lint", "status": "completed", "conclusion": "failure",
                "steps": _steps(("Set up job", "success"), ("Run ruff check", "failure"))}
        lint_c = {"id": 700 + n, "name": "lint", "app": "github-actions", "status": "completed", "conclusion": "failure"}
        j = _test_job(500 + n, "test", "success")
        out.append(_entry(n, "no_lint_build", f"Renamed helpers in {TOPICS[2 + i]}. All tests pass.",
                          _ci([j, (lint, lint_c)]),
                          {"label": "NOT_CONTRADICTED", "scope": "WHOLE", "evidence": "NONE", "preexisting": "no"}))
    for i in range(2):
        n += 1
        j = _test_job(500 + n, "test (ubuntu)", "skipped", install="failure")
        out.append(_entry(n, "no_install_in_test_job", f"Simplified {TOPICS[4 + i]}. All tests pass.", _ci([j]),
                          {"label": "NOT_CONTRADICTED", "scope": "WHOLE", "evidence": "NONE", "preexisting": "no"}))
    for i in range(3):
        n += 1
        disc = f"One known failure, `test_dns_timeout_{i}`, is unrelated and tracked separately."
        body = f"Updated {TOPICS[6 + i]}.\n\nAll tests pass. {disc}"
        if i == 0:
            c = _ci([], statuses=[("ci/circleci: unit", "failure")])
            key_ev = "NAMES_ONLY"
        else:
            j = _test_job(500 + n, "test", "failure")
            c = _ci([j], logs={str(500 + n): _pytest_log([f"dns_timeout_{i}"], 20)})
            key_ev = "LOG"
        out.append(_entry(n, "no_disclosed", body, c,
                          {"label": "NOT_CONTRADICTED", "scope": "WHOLE", "evidence": key_ev, "preexisting": "no"},
                          disclosure=disc))
    for i in range(2):
        n += 1
        part = f"Added tests for {TOPICS[9 + i]}; they pass."
        if i == 0:
            c = _ci([], statuses=[("unit (windows)", "failure")])
            ev = "NAMES_ONLY"
        else:
            j = _test_job(500 + n, "test (windows)", "failure")
            c = _ci([j], logs={str(500 + n): _pytest_log(["network"], 33)})
            ev = "LOG"
        out.append(_entry(n, "no_part_beside", f"Extended {TOPICS[9 + i]}.\n\n{part}", c,
                          {"label": "PART_BESIDE_FAILURE", "scope": "PART", "evidence": ev, "preexisting": "no"},
                          part=part))
    for i in range(2):
        n += 1
        part = f"The new tests in tests/test_parser{i}.py pass."
        j = _test_job(500 + n, "test", "failure")
        out.append(_entry(n, "no_part_contradicted", f"New parser rules.\n\n{part}",
                          _ci([j], logs={str(500 + n): _pytest_log([f"parser{i}"], 18)}),
                          {"label": "PART_CONTRADICTED", "scope": "PART", "evidence": "LOG", "preexisting": "no"},
                          part=part))
    for i, st in enumerate(("This should pass CI now.", "- [ ] All tests pass")):
        n += 1
        j = _test_job(500 + n, "test", "failure")
        out.append(_entry(n, "no_not_a_claim", f"Bumped {TOPICS[12 + i]}.\n\n{st}",
                          _ci([j], logs={str(500 + n): _pytest_log([f"bump{i}"], 12)}),
                          {"label": "NOT_A_CLAIM", "scope": "NONE", "evidence": "LOG", "preexisting": "no"},
                          statement=st))
    if packet != "P2":  # a different text per packet, so no decoy is reused (P9)
        for e in out:
            e["body"] = e["body"] + f"\n\n({packet} wording)"
            e["source"]["pr"] = e["source"]["pr"].replace("#1", "#" + str(len(packet)) + "1")
    return {"packet": packet, "decoys": copy.deepcopy(out)}
