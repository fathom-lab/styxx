# -*- coding: utf-8 -*-
"""Reading a row (P4) and the catch-up run end to end, against a fake GitHub, in a temporary store.

The tests-pass-ci collector (papers/tests-pass-ci/collect/), offline, against fakes (tests/_tests_pass_ci_fakes.py).
"""
from __future__ import annotations

import datetime as dt
import json

import pytest
from tests._tests_pass_ci_fakes import FakeGitHub, OpenGuard

import tpc_collect as collect
import tpc_read
from tpc_const import UTC
from tpc_seal import SealGuard, SealRefused
from tpc_store import Store

NOW = dt.datetime(2026, 10, 29, 12, tzinfo=UTC) + dt.timedelta(days=4)
HEAD = "c" * 40


def row(pr_id=1, agent="Claude_Code", in_m=True, created="2026-10-29T10:00:00Z", repo="o/r", number=7):
    return {"pr_id": pr_id, "repo": repo, "number": number, "agent": agent, "created_at": created,
            "day": created[:10], "u": 0.01, "v": 0.5, "in_D": True, "in_M": in_m, "flags": []}


def node(body, author_type="User", head=HEAD, created="2026-10-29T10:00:00Z", edited=None, committed="2026-10-29T09:00:00Z"):
    return {"databaseId": 1, "body": body, "lastEditedAt": edited, "createdAt": created, "state": "OPEN",
            "isDraft": False, "merged": False, "headRefOid": head, "headRefName": "feat",
            "author": {"__typename": author_type, "login": "a"},
            "commits": {"nodes": [{"commit": {"oid": head, "committedDate": committed, "pushedDate": None}}]}}


CC_BODY = "Adds the parser.\n\nAll tests pass.\n\nGenerated with Claude Code"


def ci_map(test_concl="failure", extra_jobs=()):
    jobs = [{"id": 11, "name": "build", "status": "completed", "conclusion": test_concl,
             "steps": [{"name": "Checkout", "conclusion": "success"}, {"name": "Run tests", "conclusion": test_concl}]},
            {"id": 12, "name": "lint", "status": "completed", "conclusion": "success",
             "steps": [{"name": "ruff", "conclusion": "success"}]}] + list(extra_jobs)
    return {
        f"/repos/o/r/commits/{HEAD}/check-runs?filter=latest": [{"total_count": 2, "check_runs": [
            {"id": 11, "name": "build", "status": "completed", "conclusion": test_concl, "app": {"slug": "github-actions"}, "output": {}},
            {"id": 12, "name": "lint", "status": "completed", "conclusion": "success", "app": {"slug": "github-actions"}, "output": {}}]}],
        f"/repos/o/r/commits/{HEAD}/status": [{"total_count": 0, "statuses": []}],
        f"/repos/o/r/actions/runs?head_sha={HEAD}": [{"total_count": 1, "workflow_runs": [{"id": 900}]}],
        "/repos/o/r/actions/runs/900/jobs?filter=latest": [{"total_count": len(jobs), "jobs": jobs}],
        "/repos/o/r/actions/runs/900/artifacts": [{"total_count": 1, "artifacts": [
            {"id": 77, "name": "test-results", "size_in_bytes": 300, "expired": False}]}],
        "/repos/o/r/check-runs/11/annotations": [[{"annotation_level": "failure", "message": "boom"}]],
        "/repos/o/r/actions/jobs/11/logs": (302, "https://blob.example.net/11"),
        "/repos/o/r/actions/jobs/12/logs": (302, "https://blob.example.net/12"),
        "/repos/o/r/actions/artifacts/77/zip": (302, "https://blob.example.net/a77"),
    }


def zip_junit():
    import io
    import zipfile
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("junit.xml", '<?xml version="1.0"?><testsuite tests="1" failures="1"><testcase name="t"><failure/></testcase></testsuite>')
    return b.getvalue()


BLOBS = {"https://blob.example.net/11": b"2026-10-29T10:00:00.0Z ##[group]Run pytest\n"
                                         b"2026-10-29T10:00:01.0Z ===== 2 failed, 40 passed in 3.2s =====\n",
         "https://blob.example.net/12": b"lint ok\n",
         "https://blob.example.net/a77": zip_junit()}


class Disk:
    def __init__(self, ok=True):
        self.ok = ok

    def check(self):
        return self.ok


def read(tmp_path, r, n, gh=None, disk=None, guard=None):
    gh = gh or FakeGitHub(rest=ci_map(), blobs=BLOBS)
    st = Store(tmp_path / "s")
    rec = tpc_read.read_row(gh, st, guard or OpenGuard(), r, n, None, NOW, "t1", disk or Disk(), backlog=False)
    return rec, gh, st


def test_claimer_with_failing_tests_gets_logs_artifacts_and_rule_r(tmp_path):
    rec, gh, st = read(tmp_path, row(), node(CC_BODY))
    assert rec["eligible"] and rec["screen"] == "claimer"
    assert rec["ci"]["outcome"] == "tests failed" and rec["ci"]["has_ci_test_result"]
    logs = {x["job_id"]: x for x in rec["logs"]}
    assert set(logs) == {11}  # the failing test job; lint is neither failed nor test-like
    assert logs[11]["fetched"] and logs[11]["summary_hits"] == 1
    assert rec["artifacts"][0]["counted"][0]["kind"] == "junit_xml"
    assert rec["report_artifact_counted"]
    assert rec["rule_r"]["fires"] is True
    assert rec["stale"] is False
    assert rec["ci"]["annotation_list"][0]["message"] == "boom"
    assert st.path(logs[11]["path"]).exists()
    paths = [r["path"] for r in st.read_jsonl("manifest.jsonl")]
    assert any(p.endswith("logs/11.log.gz") for p in paths)
    assert any(p.endswith("check_runs.p1.json.gz") for p in paths)


@pytest.mark.parametrize("agent,body,author_type,head,want", [
    ("Claude_Code", "All tests pass.", "User", HEAD, ["POSTFILTER_FAIL"]),
    ("Claude_Code", CC_BODY, "Bot", HEAD, ["BOT_AUTHOR"]),
    ("Copilot", CC_BODY, "Bot", HEAD, []),  # Copilot's own signature is a bot author
    ("Devin", "   ", "Bot", HEAD, ["EMPTY_BODY"]),
    ("Devin", "text", "User", None, ["NO_HEAD"]),
])
def test_exclusions_stop_before_any_ci_read(tmp_path, agent, body, author_type, head, want):
    gh = FakeGitHub(rest=ci_map(), blobs=BLOBS)
    rec, gh, _ = read(tmp_path, row(agent=agent), node(body, author_type=author_type, head=head), gh=gh)
    assert rec["exclusions"] == want
    if want:
        assert not rec["eligible"] and gh.c.calls["rest"] == 0
    else:
        assert rec["eligible"]


def test_lost_row(tmp_path):
    rec, gh, _ = read(tmp_path, row(), None)
    assert rec["exclusions"] == ["LOST"] and gh.c.calls["rest"] == 0


def test_au1_eligible_non_claimer_in_m_gets_logs_but_not_outside_m(tmp_path):
    body = "Refactors the parser.\n\nGenerated with Claude Code"
    rec, _, _ = read(tmp_path, row(in_m=True), node(body))
    assert rec["screen"] == "none" and rec["logs_reason"] == "au1_eligible" and rec["logs"]
    rec2, _, _ = read(tmp_path / "b", row(in_m=False), node(body))
    assert rec2["logs_wanted"] is False and rec2["logs"] == []


def test_claimer_with_passing_tests_gets_test_like_logs(tmp_path):
    gh = FakeGitHub(rest=ci_map("success"), blobs=BLOBS)
    rec, _, _ = read(tmp_path, row(), node(CC_BODY), gh=gh)
    assert rec["ci"]["outcome"] == "tests passed"
    assert [x["job_id"] for x in rec["logs"]] == [11] and rec["logs"][0]["why"] == "test_like"
    assert rec["rule_r"]["fires"] is False


def test_disk_floor_skips_logs_and_says_so(tmp_path):
    rec, gh, _ = read(tmp_path, row(), node(CC_BODY), disk=Disk(False))
    assert "LOGS_SKIPPED" in rec["flags"] and rec["logs"] == [] and gh.c.calls["blob"] == 0


def test_stale_description(tmp_path):
    rec, _, _ = read(tmp_path, row(), node(CC_BODY, committed="2026-10-30T00:00:00Z"))
    assert rec["stale"] is True and "STALE" in rec["flags"] and not rec["rule_r"]["fires"]


def test_late_read_flag(tmp_path):
    r = row(created="2026-10-20T10:00:00Z")
    rec, _, _ = read(tmp_path, r, node(CC_BODY, created="2026-10-20T10:00:00Z"))
    assert "LATE_READ" in rec["flags"]


def test_seal_guard_refuses_a_window_row_before_any_call(tmp_path):
    g = SealGuard(sealed_file=tmp_path / "SEALED", repo=tmp_path)
    gh = FakeGitHub(rest=ci_map(), blobs=BLOBS)
    st = Store(tmp_path / "s")
    with pytest.raises(SealRefused):
        tpc_read.fetch_pr_nodes(gh, st, g, [row(created="2026-11-03T00:00:00Z")], "t")
    with pytest.raises(SealRefused):
        tpc_read.read_row(gh, st, g, row(created="2026-11-03T00:00:00Z"), node(CC_BODY), None, NOW, "t",
                          Disk(), backlog=False)
    assert sum(gh.c.calls.values()) == 0


# ---- the catch-up run --------------------------------------------------------------------------
def frame_search(day_nodes):
    """A fake search over a fixed universe of nodes, by agent qualifier and created range."""
    from tpc_const import QUALIFIER, parse_utc

    def search(q, after, query):
        agent = next(a for a, s in QUALIFIER.items() if s in q)
        rng = q.split("created:", 1)[1]
        lo, hi = (parse_utc(x) for x in rng.split(".."))
        hits = [n for n in day_nodes.get(agent, []) if lo <= parse_utc(n["createdAt"]) <= hi]
        if "{ issueCount } }" in query:
            return {"issueCount": len(hits)}
        start = int(after or 0)
        return {"issueCount": len(hits), "pageInfo": {"hasNextPage": start + 100 < len(hits), "endCursor": str(start + 100)},
                "nodes": hits[start:start + 100]}

    return search


def universe(day="2026-10-29"):
    out = {}
    pid = 1000
    for agent, n in (("Copilot", 30), ("Claude_Code", 2000), ("Cursor", 40), ("Devin", 30), ("Google_Jules", 3), ("OpenAI_Codex", 12)):
        nodes = []
        for i in range(n):
            pid += 1
            t = dt.datetime.fromisoformat(day + "T00:00:00+00:00") + dt.timedelta(seconds=int(86400 * i / n))
            nodes.append({"databaseId": pid, "number": pid, "createdAt": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                          "author": {"__typename": "Bot" if agent == "Copilot" else "User", "login": "a"},
                          "repository": {"nameWithOwner": f"org/repo{pid % 7}", "stargazerCount": 50 + pid % 300,
                                         "isFork": pid % 11 == 0}})
        out[agent] = nodes
    return out


def setup_run(monkeypatch, tmp_path, now, gh):
    monkeypatch.setattr(collect, "NOW", lambda: now)
    monkeypatch.setattr(collect, "GitHub", lambda log, **kw: gh)
    monkeypatch.setattr(collect, "SealGuard", type("G", (SealGuard,), {
        "__init__": lambda self: SealGuard.__init__(self, sealed_file=tmp_path / "SEALED", repo=tmp_path)}))
    monkeypatch.setattr(collect, "DiskGate", lambda: Disk())
    return tmp_path / "store"


def test_dev_catch_up_is_idempotent(monkeypatch, tmp_path):
    uni = universe()
    prs = {}
    rest = {}
    for agent, nodes in uni.items():
        for n in nodes:
            body = CC_BODY if agent == "Claude_Code" else "All tests pass."
            prs[f"{n['repository']['nameWithOwner']}#{n['number']}"] = dict(node(body), databaseId=n["databaseId"],
                                                                            createdAt=n["createdAt"])
    gh = FakeGitHub(search=frame_search(uni), prs=prs, rest=rest)
    root = setup_run(monkeypatch, tmp_path, dt.datetime(2026, 11, 2, 15, tzinfo=UTC), gh)
    argv = ["run", "--dev", "--root", str(root), "--days", "2026-10-29", "--dev-beacon", "t", "--fetch-ci",
            "--workers", "1"]
    assert collect.main(argv) == 0
    st = Store(root)
    frame = st.read_jsonl("frames/frame_2026-10-29.jsonl")
    assert frame and all(tuple(r) == collect.FRAME_KEYS for r in frame)
    cc = [r for r in frame if r["agent"] == "Claude_Code"]
    assert 0 < len(cc) < 2000  # only the 48 enumerated slices
    drawn = st.read_jsonl("draw/draw_2026-10-29.jsonl")
    assert drawn
    rows = st.read_jsonl("a2/rows_index.jsonl")
    assert len(rows) == len(drawn)
    calls_first = sum(gh.c.calls.values())
    assert collect.main(argv) == 0  # second run: nothing new to fetch
    assert sum(gh.c.calls.values()) == calls_first
    assert len(st.read_jsonl("a2/rows_index.jsonl")) == len(drawn)
    assert st.verify_manifest()["differ"] == []
    runs = st.read_jsonl("runs.jsonl")
    assert len(runs) == 2 and runs[0]["code"]["combined"] == runs[1]["code"]["combined"]
    import argparse
    s = collect.status(collect.Ctx(argparse.Namespace(root=str(root), dev=True)))
    assert s["read"] == len(drawn)


def test_window_rows_are_refused_without_a_sealed_file(monkeypatch, tmp_path):
    from tests.test_tests_pass_ci_collector_screen_rule_reports import GENESIS, GENESIS_ID
    uni = universe("2026-11-02")
    gh = FakeGitHub(search=frame_search(uni), prs={}, rest={})
    root = setup_run(monkeypatch, tmp_path, dt.datetime(2026, 11, 7, 12, tzinfo=UTC), gh)
    st = Store(root)
    st.write("draw/beacon.json", json.dumps({"kind": "bitcoin", "height": 0, "beacon_hex": GENESIS_ID, "header": GENESIS,
                                             "read_at": "2026-11-07T10:00:00Z"}).encode())
    assert collect.main(["run", "--root", str(root), "--fetch-ci", "--workers", "1"]) == 0
    assert st.exists("frames/frame_2026-11-02.jsonl")  # the frame stage runs before the freeze (section 0.3)
    assert st.read_jsonl("draw/draw_2026-11-02.jsonl")
    ref = st.read_jsonl("a2/seal_refusals.jsonl")
    assert ref and ref[0]["code"] == "SEAL_GUARD" and ref[0]["rows"] > 0
    assert gh.c.calls["rest"] == 0 and gh.c.calls["graphql"] == 0  # no body, no CI
    assert not st.exists("a2/rows_index.jsonl")


def test_dry_run_writes_counts_and_no_rows(monkeypatch, tmp_path):
    uni = universe("2026-11-02")
    gh = FakeGitHub(search=frame_search(uni))
    root = setup_run(monkeypatch, tmp_path, dt.datetime(2026, 11, 2, 15, tzinfo=UTC), gh)
    assert collect.main(["run", "--dry-run", "--root", str(root), "--workers", "1"]) == 0
    st = Store(root)
    files = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
    assert not any(f.startswith(("frames/", "draw/", "a2/")) for f in files)
    out = st.read_json(next(f for f in files if f.startswith("dryrun/")))
    d = out["days"]["2026-11-02"]
    assert d["search_total"]["OpenAI_Codex"] > 0 and "per_agent" in d
    assert "databaseId" not in out["fields"] and "isFork" not in out["fields"] and "body" not in out["fields"]
    assert gh.c.calls["rest"] == 0 and gh.c.calls["graphql"] == 0
    # only ranges that have started by 15:00Z - 10 min are enumerated
    for kind, q in gh.calls:
        if "created:" in (q or ""):
            assert q.split("created:")[1][:13] <= "2026-11-02T14"


def test_cli_refuses_mixed_modes(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        collect.main(["run", "--dry-run", "--fetch-ci"])
    with pytest.raises(SystemExit):
        collect.main(["run", "--days", "2026-10-29"])
    with pytest.raises(SystemExit):
        collect.main(["run", "--dev", "--root", str(tmp_path / "x"), "--days", "2026-11-03", "--dev-beacon", "x"])


def test_capped_keeps_three_lowest_v_per_repository():
    rows = [{"repo": "A/x", "u": u, "v": u} for u in (0.5, 0.1, 0.3, 0.2)] + [{"repo": "a/X", "u": 0.05, "v": 0.05},
                                                                              {"repo": "b/y", "u": 0.9, "v": 0.9}]
    got = collect.capped(rows)
    assert sorted(r["u"] for r in got if r["repo"].lower() == "a/x") == [0.05, 0.1, 0.2]
    assert len(got) == 4


def test_cap_in_the_main_sample_orders_by_v_not_u():
    """Section 9.2 (statistics review): in M a Claude Code row has u < 0.120 and any other agent's
    u < 0.020, so a cap by u would keep other agents' rows six to one; v = u / p_M is uniform for
    every agent, so the cap by v keeps the rows with the lowest v whatever their agent."""
    cc = [{"repo": "o/r", "agent": "Claude_Code", "u": v * 0.120, "v": v} for v in (0.10, 0.20, 0.30)]
    other = [{"repo": "o/r", "agent": "Devin", "u": v * 0.020, "v": v} for v in (0.40, 0.45, 0.50)]
    got = collect.capped(cc + other)
    assert sorted(r["agent"] for r in got) == ["Claude_Code"] * 3
    by_u = sorted(cc + other, key=lambda r: r["u"])[:3]
    assert sorted(r["agent"] for r in by_u) == ["Devin"] * 3  # what a cap by u would have kept


def test_null_repository_row_is_lost_without_a_query(tmp_path):
    gh = FakeGitHub(prs={"o/r#7": node(CC_BODY)})
    st = Store(tmp_path / "s")
    out = tpc_read.fetch_pr_nodes(gh, st, OpenGuard(), [row(pr_id=5, repo=None), row(pr_id=6)], "t")
    assert out[5] == (None, "no repository or number in the frame")
    assert out[6][0] is not None and gh.c.calls["graphql"] == 1
    gh2 = FakeGitHub()
    assert tpc_read.fetch_pr_nodes(gh2, st, OpenGuard(), [row(pr_id=5, repo=None)], "t2")[5][0] is None
    assert gh2.c.calls["graphql"] == 0


def test_a_failed_batch_is_recorded_and_the_run_goes_on(monkeypatch, tmp_path):
    import tpc_gh
    uni = universe()
    calls = {"n": 0}

    class Flaky(FakeGitHub):
        def graphql(self, query, variables=None, *, kind="graphql", what=""):
            if kind == "graphql":
                calls["n"] += 1
                if calls["n"] == 1:
                    self.c.calls[kind] += 1
                    raise tpc_gh.GitHubError("graphql errors: boom")
            return super().graphql(query, variables, kind=kind, what=what)

    prs = {f"{n['repository']['nameWithOwner']}#{n['number']}": dict(node("x"), databaseId=n["databaseId"], createdAt=n["createdAt"])
           for a, ns in uni.items() for n in ns}
    gh = Flaky(search=frame_search(uni), prs=prs)
    root = setup_run(monkeypatch, tmp_path, dt.datetime(2026, 11, 2, 15, tzinfo=UTC), gh)
    argv = ["run", "--dev", "--root", str(root), "--days", "2026-10-29", "--dev-beacon", "t", "--fetch-ci", "--workers", "1"]
    assert collect.main(argv) == 0
    st = Store(root)
    errs = st.read_jsonl("a2/read_errors.jsonl")
    assert errs and "boom" in errs[0]["error"]
    drawn = st.read_jsonl("draw/draw_2026-10-29.jsonl")
    after_one = len(st.read_jsonl("a2/rows_index.jsonl"))
    assert 0 < after_one < len(drawn)
    assert collect.main(argv) == 0  # the next run reads the failed batch
    assert len(st.read_jsonl("a2/rows_index.jsonl")) == len(drawn)


def test_lock_refuses_a_live_run_and_takes_over_a_dead_one(tmp_path):
    import os
    import time as _t
    p = tmp_path / "run.lock"
    with collect.Lock(p) as lk:
        with pytest.raises(SystemExit):
            collect.Lock(p).__enter__()
        lk._last_touch = 0.0
        old = _t.time() - 3 * 3600
        os.utime(p, (old, old))
        lk.touch()  # the heartbeat brings it back
        assert _t.time() - p.stat().st_mtime < 60
    assert not p.exists()
    p.write_bytes(b"123 old\n")
    old = _t.time() - 3 * 3600
    os.utime(p, (old, old))
    with collect.Lock(p):  # a lock untouched for over two hours belonged to a dead run
        assert p.exists()
    assert collect.MAX_WAIT_CAP < collect.LOCK_STALE.total_seconds()


def test_a_skipped_test_job_is_listed_but_its_log_is_not_asked_for(tmp_path):
    skipped = {"id": 13, "name": "e2e tests", "status": "completed", "conclusion": "skipped", "steps": []}
    gh = FakeGitHub(rest=ci_map(extra_jobs=[skipped]), blobs=BLOBS)
    rec, gh, _ = read(tmp_path, row(), node(CC_BODY), gh=gh)
    e = {x["job_id"]: x for x in rec["logs"]}[13]
    assert e["fetched"] is False and e["skip"] == "skipped_job"
    assert not any("/actions/jobs/13/logs" in (c[1] or "") for c in gh.calls)
