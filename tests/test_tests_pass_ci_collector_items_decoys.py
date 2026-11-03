# -*- coding: utf-8 -*-
"""Packet items (P8) and decoys (P9) of the tests-pass-ci builder, offline, on invented items only.

The description's cleaning, the CI block, the log excerpt (by the failing step's times, by the error marker, the log's
tail), the adapters from a collector row's raw store, from the A1 snapshot and from testreceipt's transcription; every
decoy construction checked by machine, the set's mix, the names-only decoys P9 asks for, markers, sources, the seal and
its refusals, and the transforms for synthetic decoys. Nothing here reads a pull request or a network.
"""
from __future__ import annotations

import copy
import gzip
import json
import sys

import pytest

from tests import _tests_pass_ci_fakes  # noqa: F401  (puts the collector on sys.path)
from tests._tests_pass_ci_fakes import ROOT

# the invented decoy fixture is read from the repository's own collector, never from a mutated copy
sys.path.insert(0, str(ROOT / "papers" / "tests-pass-ci" / "collect" / "checks"))

import synthetic_decoys  # noqa: E402
import tpc_decoys  # noqa: E402
import tpc_items  # noqa: E402


# ---- P8.2 --------------------------------------------------------------------------------------------------
def test_clean_description_removes_comments_images_footers_urls():
    body = ("<!-- CURSOR_AGENT_PR_BODY_BEGIN -->Fixes the pager.\r\n\r\n![shot](https://img.example/x.png)"
            "<img src=\"https://img.example/y.png\">\r\nSee https://example.com/docs/page.html, then retry.\r\n"
            "All tests pass.\r\n\r\n\U0001f916 Generated with [Claude Code](https://claude.com/claude-code)\r\n"
            "Link to Devin run: https://app.devin.ai/sessions/abc\r\n"
            "https://chatgpt.com/codex/tasks/task_e_1\r\n<a href=\"https://cursor.com/background-agent?bcId=1\">x</a>")
    out = tpc_items.clean_description(body)
    assert "\r" not in out and "<!--" not in out and "![" not in out and "<img" not in out
    assert "See <url>, then retry." in out
    assert out.count(tpc_items.FOOTER_TOKEN) == 4
    assert "Claude Code" not in out and "devin" not in out.lower() and "cursor" not in out.lower()
    assert "http" not in out
    assert "All tests pass." in out


def test_clean_description_unclosed_comment_hides_the_rest_and_cut_marks_long_bodies():
    assert tpc_items.clean_description("Tests pass.<!-- hidden\nAll tests fail") == "Tests pass."
    long = "x" * (tpc_items.DESC_MAX + 50)
    out = tpc_items.clean_description(long)
    assert out.endswith("\n" + tpc_items.CUT_TOKEN) and len(out) == tpc_items.DESC_MAX + 1 + len(tpc_items.CUT_TOKEN)


def test_clean_description_keeps_repository_words_in_prose():
    out = tpc_items.clean_description("The kody-hq/kody parser now passes all tests.")
    assert "kody-hq/kody" in out


# ---- P8.1, dates ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["2026-10-29", "2026-10-29T12:00:00.5Z", "2026-10-29 12:00:00+02:00", "2026/10/29",
                                  "2026_10_29", "29/10/2026", "10/29/2026", "29.10.2026", "29-10-2026", "10/29/26",
                                  "20261029", "Oct 29", "oct 29th", "October 29, 2026", "29 Oct 2026", "Sept 3",
                                  "November 2026",
                                  # the review of 2026-11-02 found these survived the mask
                                  "1793232000", "1793232000123", "20261029T120000Z", "20261029T1200", "29-Oct-2026",
                                  "29-OCT-26", "Oct-29", "2026-10", "2026/11", "29.10.26", "October", "december", "Oct",
                                  "NOV", "Sept."])
def test_every_date_form_of_p8_1_is_masked(text):
    assert tpc_items.mask_dates(f"at {text} ok") == "at <date> ok"


def test_clock_times_versions_and_counts_stay():
    for text in ("12:34:56", "v1.2.3", "node 20.10.0", "3 failed, 120 passed in 2.10s", "sha a20261029b", "1/2 done",
                 "python 3.12.10", "10/29 passed", "this May fail", "oct(8) and dec(9)", "id 1700000000",
                 "2026-2027", "print(x, sep='')"):
        assert tpc_items.mask_dates(text) == text


def test_no_date_reaches_an_item():
    job = _job(5, "test 2026-10-29", "failure", [("Run tests", "failure")])
    ci = {"check_runs": [{"id": 1, "name": "nightly 2026/10/29", "app": "buildkite", "conclusion": "failure"}],
          "statuses": [{"context": "ci: Oct 29 run", "state": "failure"}],
          "jobs": [job],
          "annotations": [{"check_run_id": 1, "level": "failure", "message": "expired on 29 Oct 2026"}],
          "logs": {"5": "##[group]Run pytest\nDate: Thu, 29 Oct 2026 12:00:00 GMT\nsee https://x.example/2026-10-29/r\n"
                         "FAILED test_expiry - assert 2026-10-29 == 2026-11-03\n##[error]exit 1\n"}}
    it = tpc_items.build_item("Fixed the October 2026 report (10/29/2026). All tests pass.", ci)
    for text in (it["step1_text"], it["item_text"]):
        assert not tpc_items.DATES.search(text)
        for s in ("2026", "Oct 29", "October"):
            assert s not in text
    assert it["item_text"].count(tpc_items.DATE_TOKEN) >= 8 and "12:00:00 GMT" in it["item_text"]
    assert "see <url>" in it["item_text"]  # a date inside a URL goes with the URL


def test_a_decoy_that_carries_a_date_is_refused():
    e = synthetic_decoys.spec("P2")["decoys"][0]
    built = tpc_items.build_item(e["body"], e["ci"], e["repo"])
    assert not any("carries a date" in p for p in tpc_decoys.check_decoy(copy.deepcopy(e), built))
    dated = dict(built, item_text=built["item_text"] + "\nrun of 2026-10-29")
    assert any("carries a date" in p for p in tpc_decoys.check_decoy(copy.deepcopy(e), dated))


@pytest.mark.parametrize("left,kind", [("started 1793232000 ok", "epoch"), ("at 1793232000123 ms", "epoch"),
                                       ("built october", "month"), ("Nov build", "month"), ("on 10/29 at noon", "mm_dd"),
                                       ("09/30", "mm_dd")])
def test_the_decoy_date_check_is_written_apart_from_the_mask_and_fires_on_what_a_mask_would_miss(left, kind):
    """The review of 2026-11-02: the decoy check ran DATES over text DATES had masked, so it could never fire. Its
    detector reads Unix times of 2026-09 to 2026-12, the names of those months and their MM/DD, by patterns of its
    own: text the mask left, as if the mask had missed it, is refused."""
    assert kind in tpc_items.date_traces(left)
    e = synthetic_decoys.spec("P2")["decoys"][0]
    built = tpc_items.build_item(e["body"], e["ci"], e["repo"])
    left_in = dict(built, item_text=built["item_text"] + "\n" + left)
    probs = tpc_decoys.check_decoy(copy.deepcopy(e), left_in)
    assert any("date trace" in p and kind in p for p in probs), probs
    for quiet in ("id 1700000000", "1/2 done", "oct(8)", "3 failed, 120 passed", "11/12 passed"):
        assert tpc_items.date_traces(quiet) == []
    # a log line that carries one of these after masking refuses the decoy built from it
    e2 = copy.deepcopy(e)
    jid = next(iter(e2["ci"]["logs"]))
    e2["ci"]["logs"][jid] = e2["ci"]["logs"][jid].replace("Syncing repository", "x").replace(
        "=========== 1 failed", "seen on 10/28, =========== 1 failed")
    built2 = tpc_items.build_item(e2["body"], e2["ci"], e2["repo"])
    assert "10/28" in built2["item_text"]
    assert any("date trace" in p for p in tpc_decoys.check_decoy(e2, built2))


# ---- P8.3 --------------------------------------------------------------------------------------------------
def _job(jid, name, concl, steps):
    return {"id": jid, "name": name, "status": "completed", "conclusion": concl,
            "steps": [{"number": i, "name": n, "conclusion": c, "status": "completed",
                       "started_at": f"2026-10-01T10:00:{10 * i:02d}Z", "completed_at": f"2026-10-01T10:00:{10 * i + 9:02d}Z"}
                      for i, (n, c) in enumerate(steps, 1)]}


def _log_with_post():
    return ("2026-10-01T10:00:00.1Z Current runner version\n"
            "2026-10-01T10:00:20.1Z ##[group]Run pytest -q\n"
            "2026-10-01T10:00:20.2Z ##[endgroup]\n"
            + "".join(f"2026-10-01T10:00:2{i % 9}.5Z line {i}\n" for i in range(300))
            + "2026-10-01T10:00:28.0Z =========== 2 failed, 310 passed in 1.0s ===========\n"
            "2026-10-01T10:00:28.1Z ##[error]Process completed with exit code 1.\n"
            "2026-10-01T10:00:40.0Z Post job cleanup.\n2026-10-01T10:00:41.0Z Cleaning up orphan processes\n")


def test_log_excerpt_by_the_failing_steps_times_leaves_out_post_steps():
    job = _job(9, "test", "failure", [("Set up job", "success"), ("Run tests", "failure")])
    text, how = tpc_items.log_excerpt(_log_with_post().encode("utf-8"), job)
    lines = text.split("\n")
    assert how == "step_times"
    assert len(lines) == tpc_items.LOG_EXCERPT_LINES
    assert lines[-1] == "##[error]Process completed with exit code 1."
    assert "2 failed, 310 passed" in text and "Cleaning up orphan" not in text and "Post job" not in text
    assert not any(ln.startswith("2026-") for ln in lines)  # timestamps removed


def test_log_excerpt_ends_at_the_failure_when_post_steps_share_its_last_second():
    # GitHub stamps steps to the second: the post steps begin in the second the failing step ended, inside the
    # step's own time range, and a long checkout cleanup would fill the last 150 lines and push the failure out
    # every line below is invented for this test (no pull request's log)
    job = _job(9, "test", "failure", [("Set up job", "success"), ("Run tests", "failure")])
    log = ("2026-10-01T10:00:20.1Z ##[group]Run make check\n2026-10-01T10:00:20.2Z ##[endgroup]\n"
           "2026-10-01T10:00:29.1Z FAIL ledger/rounding_test.go:41: TestHalfEvenRounding: got 0.13, want 0.12\n"
           "2026-10-01T10:00:29.2Z 1 of 57 checks failed in package ledger\n"
           "2026-10-01T10:00:29.3Z ##[error]Process completed with exit code 1.\n"
           "2026-10-01T10:00:29.4Z Post job cleanup.\n"
           + "".join(f"2026-10-01T10:00:29.5Z cleanup line {i} of the checkout\n" for i in range(400)))
    text, how = tpc_items.log_excerpt(log, job)
    assert how == "step_times"
    assert text.split("\n")[-1] == "##[error]Process completed with exit code 1."
    assert "1 of 57 checks failed" in text and "Post job" not in text and "cleanup line" not in text
    # without the post steps' marker, the span still ends at its last ##[error] line
    text, _ = tpc_items.log_excerpt(log.replace("Post job cleanup.", "Run actions/upload-artifact"), job)
    assert text.split("\n")[-1] == "##[error]Process completed with exit code 1." and "cleanup line" not in text
    # without an ##[error] line, the post steps' marker ends it
    text, _ = tpc_items.log_excerpt(log.replace("##[error]Process completed with exit code 1.", "exit 1"), job)
    assert text.split("\n")[-1] == "exit 1" and "cleanup line" not in text
    # and the error-marker fallback ends there too
    nojob = {"id": 1, "name": "test", "conclusion": "failure", "steps": [{"number": 1, "name": "Run tests",
                                                                           "conclusion": "failure"}]}
    text, how = tpc_items.log_excerpt(log, nojob)
    assert how == "error_marker" and text.split("\n")[-1] == "##[error]Process completed with exit code 1."


def test_log_excerpt_falls_back_to_the_error_marker_then_the_tail():
    job = {"id": 1, "name": "test", "conclusion": "failure", "steps": [{"number": 1, "name": "Run tests",
                                                                         "conclusion": "failure"}]}
    log = "##[group]Run make lint\nok\n##[group]Run make test\nFAIL x\n##[error]exit 2\n##[group]Run upload\ndone\n"
    text, how = tpc_items.log_excerpt(log, job)
    assert how == "error_marker" and text.split("\n")[0] == "##[group]Run make test" and "upload" not in text
    text, how = tpc_items.log_excerpt("a\nb\nc\n", job)
    assert how == "log_tail" and text == "a\nb\nc"


def test_log_excerpt_strips_ansi_and_caps_a_job_at_12000_characters():
    job = _job(9, "test", "failure", [("Set up job", "success"), ("Run tests", "failure")])
    log = ("2026-10-01T10:00:20.1Z ##[group]Run pytest\n"
           + "".join(f"2026-10-01T10:00:25.0Z \x1b[31m{'y' * 200}\x1b[0m\n" for _ in range(149)))
    text, _ = tpc_items.log_excerpt(log, job)
    assert "\x1b" not in text
    assert text.startswith(tpc_items.CUT_TOKEN + "\n") and len(text) == 12000


def test_render_ci_lines_order_and_redaction():
    ci = {"check_runs": [{"id": 2, "name": "lint", "app": "github-actions", "conclusion": "success"},
                         {"id": 1, "name": "build https://ci.example/x", "app": "buildkite", "conclusion": "failure"}],
          "statuses": [{"context": "ci/circleci: unit", "state": "failure"}],
          "jobs": [_job(5, "test", "failure", [("Run tests", "failure")]), _job(6, "docs", "success", [("Build", "success")])],
          "annotations": [{"check_run_id": 1, "level": "failure", "message": "boom\nin acme/widgets " + "z" * 400}],
          "logs": {"5": "##[group]Run pytest\nE assert 1 == 2 in acme/widgets/tests\n##[error]exit 1\n"}}
    block, info = tpc_items.render_ci(ci, repo="Acme/Widgets")
    lines = block.split("\n")
    assert lines[0] == "check: build <url> (buildkite) -> failure" and lines[1] == "check: lint (github-actions) -> success"
    assert "status: ci/circleci: unit -> failure" in lines
    assert "job: test -> failure" in lines and "step: Run tests -> failure" in lines
    assert "job: docs -> success" not in lines  # neither failed nor test-like
    ann = [ln for ln in lines if ln.startswith("annotation: ")][0]
    assert len(ann) <= len("annotation: failure: ") + 300 and "\n" not in ann and "<repo>" in ann  # cut, then redacted
    assert "log: test" in lines and "acme/widgets" not in block.lower()
    assert lines[-1] == tpc_items.FIXED_LINE
    assert info["logs"][0]["how"] == "error_marker"


def test_annotations_follow_the_check_runs_order_whatever_the_records_list_order():
    """P8.3: annotations are shown in the check runs' order, each run's own in its order, so a step flip that appends
    its donor's annotations at the end of the list renders as the real failing run does (the distinguishability audit
    of 2026-11-02: re-making real failing jobs through the flip changed 18 of 44 items, by annotation order)."""
    runs = [{"id": 7, "name": "build", "app": "github-actions", "conclusion": "failure"},
            {"id": 3, "name": "test (ubuntu)", "app": "github-actions", "conclusion": "failure"},
            {"id": 9, "name": "Lint", "app": "github-actions", "conclusion": "failure"}]
    ann = [{"check_run_id": 3, "level": "failure", "message": "t1"}, {"check_run_id": 7, "level": "failure", "message": "b1"},
           {"check_run_id": 3, "level": "failure", "message": "t2"}, {"check_run_id": 9, "level": "warning", "message": "l1"}]
    base = {"check_runs": runs, "statuses": [], "jobs": [], "logs": {}}
    a, _ = tpc_items.render_ci(dict(base, annotations=ann))
    b, _ = tpc_items.render_ci(dict(base, annotations=[ann[1], ann[3], ann[0], ann[2]]))
    assert a == b
    shown = [ln.rsplit(": ", 1)[1] for ln in a.split("\n") if ln.startswith("annotation: ")]
    assert shown == ["b1", "l1", "t1", "t2"]  # build, Lint, test (ubuntu): the check lines' order


def test_the_ci_block_carries_no_repository_name_in_runner_paths_tokens_or_a_donor_log():
    """The review of 2026-11-02: the runner's working path named the repository in 6 of 26 real two-step-like items,
    and the short name appeared in 15 of 26 CI blocks; a synthetic flip's donor log named another repository."""
    log = ("##[group]Run pytest\ncd /home/runner/work/widgets/widgets\nD:\\a\\widgets\\widgets\\src\\x.py\n"
           "/__w/widgets/widgets/t.py /Users/runner/work/widgets/widgets/u.py\n"
           "FAILED tests/test_widgets.py::test_one - acme raised\nfrom donorco/gizmo-kit import x\n"
           "/home/runner/work/gizmo-kit/gizmo-kit/a.py\n/home/runner/work/_temp/abc.sh\n##[error]exit 1\n")
    ci = {"check_runs": [{"id": 1, "name": "widgets test", "app": "github-actions", "conclusion": "failure"}],
          "jobs": [_job(1, "widgets test", "failure", [("Run tests", "failure")])],
          "annotations": [{"check_run_id": 1, "level": "failure", "message": "Process completed with exit code 1."}],
          "logs": {"1": log}}
    block, _ = tpc_items.render_ci(ci, repo="acme/widgets", also=("donorco/gizmo-kit",))
    low = block.lower()
    for name in ("widgets", "gizmo-kit", "donorco", "acme"):
        assert name not in low, (name, block)
    assert "/home/runner/work/<repo>/<repo>" in block and "D:\\a\\<repo>\\<repo>\\src" in block
    assert "tests/test_<repo>.py" in block and "/home/runner/work/_temp/abc.sh" in block
    # a short name that is a common word stays a word; a token inside another word stays
    b2, _ = tpc_items.render_ci({"jobs": [_job(2, "test", "failure", [("Run tests", "failure")])],
                                 "logs": {"2": "##[group]Run x\nthe test of org/test failed\nwidgetsmith ok\n"
                                                 "##[error]exit 1\n"}}, repo="org/test")
    assert "the test of <repo> failed" in b2 and "org/test" not in b2
    b3, _ = tpc_items.render_ci({"jobs": [_job(3, "test", "failure", [("Run tests", "failure")])],
                                 "logs": {"3": "##[group]Run x\nwidgetsmith ok\n##[error]exit 1\n"}}, repo="a/widgets")
    assert "widgetsmith ok" in b3
    # a short name that is a common word is not redacted as a token, but a runner's working path still names the
    # repository, and is redacted (the mutation run of this repair found no test for the path rule alone)
    b4, _ = tpc_items.render_ci({"jobs": [_job(4, "test", "failure", [("Run tests", "failure")])],
                                 "logs": {"4": "##[group]Run x\ncd /home/runner/work/core/core && make\n"
                                                 "D:\\a\\core\\core\\t.py\n##[error]exit 1\n"}}, repo="acme/core")
    assert "/home/runner/work/<repo>/<repo> && make" in b4 and "D:\\a\\<repo>\\<repo>\\t.py" in b4
    assert "work/core" not in b4 and "\\core\\" not in b4


def test_render_ci_at_most_three_logs_and_30000_characters():
    jobs = [_job(i, f"test {i}", "failure", [("Run tests", "failure")]) for i in range(1, 6)]
    logs = {str(i): "".join(f"line {k} {'q' * 100}\n" for k in range(140)) for i in range(1, 6)}
    block, info = tpc_items.render_ci({"jobs": jobs, "logs": logs})
    assert sum(1 for ln in block.split("\n") if ln.startswith("log: ")) == 3
    assert [x["job_id"] for x in info["logs"]] == [1, 2, 3]
    assert sum(x.get("chars", 0) for x in info["logs"]) <= 30000


def test_build_item_texts_and_digest():
    it = tpc_items.build_item("All tests pass.", {"jobs": [_job(1, "test", "success", [("Run tests", "success")])]})
    assert it["steps"] == 2 and it["step1_text"] == "All tests pass."
    assert it["item_text"].startswith("DESCRIPTION\nAll tests pass.\n\nCI AT THE HEAD COMMIT\n")
    assert it["item_sha256"] == tpc_items.item_sha256(it["item_text"])
    one = tpc_items.build_item("All tests pass.", None, steps=1)
    assert one["item_text"] is None and one["steps"] == 1


def test_ci_from_row_reads_the_raw_store(tmp_path):
    raw = tmp_path / "a2" / "raw" / "7" / "att"
    raw.mkdir(parents=True)

    def gz(name, obj):
        (raw / name).write_bytes(gzip.compress(json.dumps(obj).encode()))
    gz("pr_node.json.gz", {"body": "All tests pass."})
    gz("check_runs.p1.json.gz", {"check_runs": [{"id": 11, "name": "test", "app": {"slug": "github-actions"},
                                                 "conclusion": "failure", "status": "completed"}]})
    gz("check_runs.p2.json.gz", {"check_runs": [{"id": 12, "name": "lint", "app": {"slug": "github-actions"},
                                                 "conclusion": "success", "status": "completed"}]})
    gz("status.p1.json.gz", {"statuses": [{"context": "ci/x", "state": "success"}]})
    gz("jobs.99.p1.json.gz", {"jobs": [_job(11, "test", "failure", [("Run tests", "failure")])]})
    gz("annotations.11.p1.json.gz", [{"annotation_level": "failure", "message": "assert"}])
    (raw / "logs").mkdir()
    (raw / "logs" / "11.log.gz").write_bytes(gzip.compress(b"##[group]Run pytest\n1 failed\n##[error]x\n"))
    rec = {"raw_dir": "a2/raw/7/att"}
    ci = tpc_items.ci_from_row(tmp_path, rec)
    assert [c["id"] for c in ci["check_runs"]] == [11, 12] and ci["check_runs"][0]["app"] == "github-actions"
    assert ci["statuses"] == [{"context": "ci/x", "state": "success"}] and ci["annotations"][0]["message"] == "assert"
    assert set(ci["logs"]) == {"11"}
    assert tpc_items.body_from_row(tmp_path, rec) == "All tests pass."


def test_ci_from_snapshot_and_from_transcription(tmp_path):
    (tmp_path / "2025_1_5.log").write_bytes(b"##[group]Run test\nFAIL\n##[error]x\n")
    item = {"check_runs": [{"id": 5, "name": "test", "app": "github-actions", "status": "completed",
                            "conclusion": "failure", "steps": [{"name": "Run tests", "conclusion": "failure", "number": 1}],
                            "annotations": [{"annotation_level": "failure", "message": "m"}], "log_file": "2025_1_5.log"}],
            "statuses": [{"context": "ci/a", "state": "failure"}]}
    ci = tpc_items.ci_from_snapshot(item, tmp_path)
    assert ci["jobs"][0]["id"] == 5 and ci["logs"]["5"].startswith(b"##[group]") and ci["annotations"][0]["check_run_id"] == 5
    t = tpc_items.ci_from_transcription("status buildkite/bun: failure; test (ubuntu): failure; Run: x: failure")
    assert t["names_only"] and t["statuses"] == [{"context": "buildkite/bun", "state": "failure"}]
    assert [c["name"] for c in t["check_runs"]] == ["test (ubuntu)", "Run: x"]
    block, _ = tpc_items.render_ci(t)
    assert block.split("\n")[0] == tpc_items.NAMES_ONLY_LINE


# ---- P9 ----------------------------------------------------------------------------------------------------
def test_the_constructions_are_p9s_mix():
    c = tpc_decoys.CONSTRUCTIONS
    assert sum(v[1] for v in c.values() if v[0] == "YES") == 15 and sum(v[1] for v in c.values() if v[0] == "NO") == 15
    assert all(v[2] == "WHOLE_CONTRADICTED" for v in c.values() if v[0] == "YES")
    assert all(v[2] != "WHOLE_CONTRADICTED" and v[1] >= 2 for v in c.values() if v[0] == "NO")
    assert c["no_disclosed"][1] == 3 and c["yes_log"][1] == 9
    for packet in ("Q", "P1", "P2"):
        m = tpc_decoys.mix(packet)
        assert sum(n for k, n in m.items() if c[k][0] == "YES") == 15 and sum(n for k, n in m.items() if c[k][0] == "NO") == 15
        assert all(n >= 2 for k, n in m.items() if c[k][0] == "NO" and n)
    # P2: no decoy shows nothing failing (no real two-step item of P2 does); with OD-22 taken (2026-11-03), the shapes
    # no pre-window source showed are filled by real-sourced decoys of the same side: the preexisting and names-only
    # YES decoys by yes_log, the installation failures by no_lint_build, and no decoy of P2 is names-only
    assert tpc_decoys.mix("P2") == dict(tpc_decoys.mix("P1"), yes_log=13, yes_names_only=0, yes_preexisting=0,
                                        no_passing=0, no_lint_build=6, no_install_in_test_job=0)
    assert {k: n for k, n in tpc_decoys.mix("P2").items() if n} == {
        "yes_log": 13, "yes_stale": 2, "no_lint_build": 6, "no_disclosed": 3, "no_part_beside": 2,
        "no_part_contradicted": 2, "no_not_a_claim": 2}
    assert tpc_decoys.names_only_mix("P2") == {"no_disclosed": 0, "no_part_beside": 0}
    assert tpc_decoys.names_only_mix("P1") == {"no_disclosed": 1, "no_part_beside": 1}
    assert tpc_decoys.mix("P1") == tpc_decoys.mix("Q") == {n: v[1] for n, v in c.items()}
    assert synthetic_decoys.MIX_P2 == {k: tpc_decoys.mix("P2")[k] for k in synthetic_decoys.MIX_P2}
    assert synthetic_decoys.MIX == {k: tpc_decoys.mix("P1")[k] for k in synthetic_decoys.MIX}
    assert synthetic_decoys.NAMES_ONLY_P2 == tpc_decoys.names_only_mix("P2")


def test_the_fixture_set_holds_every_construction_and_seals():
    set_obj, probs = tpc_decoys.build_set(synthetic_decoys.spec("P2"))
    assert probs == []
    assert len(set_obj["decoys"]) == 30
    assert [d["item_sha256"] for d in set_obj["decoys"]] == sorted(d["item_sha256"] for d in set_obj["decoys"])
    sealed = tpc_decoys.seal_set(set_obj, salt=b"\x01" * 32)
    assert tpc_decoys.open_set(sealed["set_bytes"], b"\x01" * 32, sealed["digest"]) == set_obj
    with pytest.raises(tpc_decoys.DecoyError):
        tpc_decoys.open_set(sealed["set_bytes"], b"\x02" * 32, sealed["digest"])
    with pytest.raises(tpc_decoys.DecoyError):
        tpc_decoys.open_set(sealed["set_bytes"] + b" ", b"\x01" * 32, sealed["digest"])
    with pytest.raises(tpc_decoys.DecoyError):
        tpc_decoys.seal_set(set_obj, salt=b"short")


def _break(name, fn, packet="P2"):
    spec = synthetic_decoys.spec(packet)
    e = next(x for x in spec["decoys"] if x["construction"] == name)
    fn(e)
    return tpc_decoys.build_set(spec)[1]


@pytest.mark.parametrize("name,fn,want,packet", [
    ("yes_log", lambda e: e.update(ci=dict(tpc_decoys.names_only(e["ci"]), annotations=e["ci"]["annotations"])),
     "a log excerpt is shown", "P2"),
    ("yes_log", lambda e: e["key"].update(label="NOT_CONTRADICTED"), "key label", "P2"),
    ("yes_names_only", lambda e: e["ci"].update(statuses=[{"context": "ci/x: unit", "state": "success"}]), "test failure",
     "P1"),
    ("yes_stale", lambda e: e.update(last_edited_at="2026-10-01T10:00:00Z"), "STALE", "P2"),
    ("yes_preexisting", lambda e: e.update(preexisting_shown="not in the item"), "preexisting_shown", "P1"),
    ("no_lint_build", lambda e: e["ci"]["jobs"][1]["steps"][1].update(name="Run unit tests"), "nothing that fails", "P2"),
    ("no_install_in_test_job", lambda e: e["ci"]["jobs"][0]["steps"][3].update(conclusion="failure"), "installation",
     "P1"),
    ("no_disclosed", lambda e: e.update(disclosure="a sentence the body lacks"), "disclosure", "P2"),
    ("no_part_beside", lambda e: e.update(part=""), "part quotes", "P2"),
    ("no_not_a_claim", lambda e: e["key"].update(scope="WHOLE"), "key scope", "P2"),
])
def test_each_construction_is_checked_by_machine(name, fn, want, packet):
    """Each construction broken and caught, in a packet that holds it (with OD-22 taken, P2 holds no yes_names_only,
    yes_preexisting or no_install_in_test_job decoy; Q and P1 hold two of each)."""
    assert tpc_decoys.mix(packet)[name] > 0
    probs = _break(name, fn, packet)
    assert any(want in p for p in probs), probs


@pytest.mark.parametrize("name", ["yes_names_only", "yes_preexisting", "no_install_in_test_job"])
def test_p2_holds_none_of_the_shapes_od22_replaced(name):
    """OD-22 (taken 2026-11-03): a P2 decoy of a shape no pre-window source showed in a form the builder can key is
    refused by P2's mix, and the set the fixture builds in P2's mix holds none."""
    assert tpc_decoys.mix("P2")[name] == 0
    assert not any(e["construction"] == name for e in synthetic_decoys.spec("P2")["decoys"])
    spec = synthetic_decoys.spec("P2")
    spec["decoys"][0]["construction"] = name
    probs = tpc_decoys.build_set(spec)[1]
    assert any(p.startswith(f"{name}: 1 decoys, P9 asks for 0 in P2") for p in probs), probs
    assert any(p.startswith("yes_log: 12 decoys, P9 asks for 13 in P2") for p in probs), probs


def test_no_passing_is_checked_by_machine_where_its_packet_holds_it():
    probs = _break("no_passing", lambda e: e["ci"]["statuses"].append({"context": "deploy", "state": "failure"}), "P1")
    assert any("nothing fails" in p for p in probs), probs


def test_the_set_refuses_a_wrong_mix_a_missing_names_only_and_reuse():
    spec = synthetic_decoys.spec("P1")
    spec["decoys"][0]["construction"] = "no_passing"
    probs = tpc_decoys.build_set(spec)[1]
    assert any("yes_log: 8 decoys" in p for p in probs) and any("no_passing: 3 decoys" in p for p in probs)
    spec = synthetic_decoys.spec("P2")
    spec["decoys"][0]["construction"] = "no_passing"
    probs = tpc_decoys.build_set(spec)[1]
    assert any("yes_log: 12 decoys" in p for p in probs) and any("no_passing: 1 decoys, P9 asks for 0" in p
                                                                for p in probs)
    spec = synthetic_decoys.spec("P1")
    d = next(x for x in spec["decoys"] if x["construction"] == "no_disclosed" and not x["ci"]["jobs"])
    d["ci"] = synthetic_decoys.spec("P1")["decoys"][0]["ci"]
    probs = tpc_decoys.build_set(spec)[1]
    assert any("no_disclosed: 0 names-only decoys, P9 asks for 1" in p for p in probs), probs
    # OD-22: P2's disclosed failures all show a log; a names-only one is refused
    spec = synthetic_decoys.spec("P2")
    d = next(x for x in spec["decoys"] if x["construction"] == "no_disclosed")
    d["ci"] = {"check_runs": [], "jobs": [], "statuses": [{"context": "ci/circleci: unit", "state": "failure"}],
               "annotations": [], "logs": {}}
    probs = tpc_decoys.build_set(spec)[1]
    assert any("no_disclosed: 1 names-only decoys, P9 asks for 0" in p for p in probs), probs
    set_p2, _ = tpc_decoys.build_set(synthetic_decoys.spec("P2"))
    probs = tpc_decoys.build_set(synthetic_decoys.spec("P2"), other_sets=[set_p2["decoys"][0]["item_sha256"]])[1]
    assert any("another packet's" in p for p in probs)
    p2_texts = [d["step1_text"] for d in set_p2["decoys"]]
    p2_sources = [e["source"]["pr"] for e in synthetic_decoys.spec("P2")["decoys"]]
    assert tpc_decoys.build_set(synthetic_decoys.spec("P1"), other_sets=[d["item_sha256"] for d in set_p2["decoys"]],
                                other_sources=p2_sources, other_texts=p2_texts)[1] == []


def test_no_source_pull_request_or_near_duplicate_description_is_shared_across_packets():
    """The review of 2026-11-02: check_set compared item bytes only, so a source reused in another packet, or a
    description changed by one line, passed; and packet Q's real items could be P2's decoy sources."""
    spec = synthetic_decoys.spec("P2")
    set_p2, _ = tpc_decoys.build_set(spec)
    variant = synthetic_decoys.spec("P2")
    for e in variant["decoys"]:
        e["body"] += "\n\n(one more line)"
        e["source"]["pr"] = e["source"]["pr"].replace("#1", "#9")
    probs = tpc_decoys.build_set(variant, other_texts=[d["step1_text"] for d in set_p2["decoys"]])[1]
    assert any("30 decoys' descriptions near-duplicate" in p for p in probs), probs
    probs = tpc_decoys.build_set(spec, other_sources=[spec["decoys"][3]["source"]["pr"].upper(),
                                                      "https://github.com/" + spec["decoys"][4]["repo"] + "/pull/"
                                                      + spec["decoys"][4]["source"]["pr"].split("#")[1]])[1]
    assert any("2 decoys come from a pull request another packet uses" in p for p in probs), probs
    assert tpc_decoys.near_duplicate("Fixed the pager. All tests pass.", "fixed the PAGER -- all tests pass!")
    assert not tpc_decoys.near_duplicate("Fixed the pager. All tests pass.", "Reworked date parsing. Tests pass.")


def test_one_decoy_per_source_and_two_per_repository():
    spec = synthetic_decoys.spec("P2")
    spec["decoys"][5]["source"]["pr"] = spec["decoys"][4]["source"]["pr"]
    spec["decoys"][5]["repo"] = spec["decoys"][4]["repo"]
    probs = tpc_decoys.build_set(spec)[1]
    assert any("carry more than one decoy" in p for p in probs), probs
    spec = synthetic_decoys.spec("P2")
    for e in spec["decoys"][:3]:
        e["repo"] = "example-org/one-place"
        e["source"]["pr"] = "example-org/one-place#" + e["source"]["pr"].split("#")[1]
    probs = tpc_decoys.build_set(spec)[1]
    assert any("repositories carry more than 2 decoys" in p for p in probs), probs


def test_an_entry_whose_repository_is_not_its_sources_is_refused():
    spec = synthetic_decoys.spec("P2")
    spec["decoys"][0]["repo"] = "other-org/other-repo"
    probs = tpc_decoys.build_set(spec)[1]
    assert any("is not its source's" in p for p in probs), probs
    spec = synthetic_decoys.spec("P2")
    del spec["decoys"][0]["repo"]
    assert any("is not its source's" in p for p in tpc_decoys.build_set(spec)[1])


# ---- P2: only the shapes P2's real two-step items show (the review of 2026-11-02) ------------------------------
def _p2_entry(name):
    spec = synthetic_decoys.spec("P2")
    return spec, next(x for x in spec["decoys"] if x["construction"] == name)


def test_p2_refuses_a_decoy_with_nothing_failing():
    spec, e = _p2_entry("no_lint_build")
    e["ci"] = synthetic_decoys.spec("P1")["decoys"][15]["ci"]  # P1's no_passing: tests pass, nothing fails
    assert synthetic_decoys.spec("P1")["decoys"][15]["construction"] == "no_passing"
    probs = tpc_decoys.build_set(spec)[1]
    assert any("'claimer' beside 'tests passed' (any job or check failed: False)" in p for p in probs), probs


@pytest.mark.parametrize("statement,kind", [("The tests should pass.", "none"), ("Tests pass locally.", "claimer"),
                                            ("## Testing\n- `npm test`", "implied_only")])
def test_p2_reads_the_claim_beside_failing_tests(statement, kind):
    """Beside failing tests P2's real two-step items are claimers (candidates) or read no claim at all (AU1); an
    implied-only description beside failing tests is in no stratum of P2."""
    from tpc_screen import screen
    spec, e = _p2_entry("no_not_a_claim")
    e["body"] = "Bumped the pager.\n\n" + statement
    e["statement"] = statement
    assert screen(e["body"])["screen"] == kind
    probs = [p for p in tpc_decoys.build_set(spec)[1] if "screen reads" in p]
    assert (probs == []) == (kind != "implied_only"), probs


def test_p2_refuses_an_implied_only_claim_and_a_ci_with_no_test_result():
    from tpc_screen import screen
    implied = "## Testing\n- `pytest -q`"  # a test command listed under Testing: testreceipt's implied claim
    spec, e = _p2_entry("no_not_a_claim")
    e["body"] = "Bumped the pager.\n\n" + implied
    e["statement"] = implied
    assert screen(e["body"])["screen"] == "implied_only"
    probs = tpc_decoys.build_set(spec)[1]
    assert any("'implied_only' beside 'tests failed'" in p for p in probs), probs
    spec, e = _p2_entry("no_lint_build")
    e["ci"]["jobs"] = [e["ci"]["jobs"][1]]  # only the failing lint job: no test result at all
    e["ci"]["check_runs"] = [e["ci"]["check_runs"][1]]
    probs = tpc_decoys.build_set(spec)[1]
    assert any("'no test result'" in p for p in probs), probs


def test_p2_refuses_names_only_failures_inside_actions_and_logs_missing_or_found_by_markers():
    """The review of 2026-11-02: 94 of 94 failing Actions check runs of the development store carry an annotation,
    the collector reads every candidate's and AU1 row's logs, and every real excerpt was found by the step's times."""
    spec, e = _p2_entry("yes_log")
    e["ci"] = tpc_decoys.names_only(e["ci"])
    probs = tpc_decoys.build_set(spec)[1]
    assert any("carries no annotation" in p for p in probs), probs
    assert any("names-only decoy of P2" in p for p in probs), probs
    assert any("without its log excerpt" in p for p in probs), probs
    spec, e = _p2_entry("yes_log")
    jid = next(iter(e["ci"]["logs"]))
    e["ci"]["jobs"][0]["steps"][3]["started_at"] = "2026-10-01T11:00:00Z"  # the step's times find no span
    e["ci"]["jobs"][0]["steps"][3]["completed_at"] = "2026-10-01T11:00:09Z"
    e["ci"]["logs"][jid] = e["ci"]["logs"][jid].replace("Cleaning up orphan processes", "##[error]exit 1")
    probs = tpc_decoys.build_set(spec)[1]
    assert any("not by the failing step's times" in p for p in probs), probs
    # the same shapes pass where the packet is not P2 (names-only CI blocks are P1's 2025 items' form)
    spec = synthetic_decoys.spec("P1")
    e = next(x for x in spec["decoys"] if x["construction"] == "yes_log")
    e["ci"]["jobs"][0]["steps"][3]["started_at"] = "2026-10-01T11:00:00Z"
    e["ci"]["jobs"][0]["steps"][3]["completed_at"] = "2026-10-01T11:00:09Z"
    assert tpc_decoys.build_set(spec)[1] == []


def test_every_packet_refuses_a_failing_actions_check_run_with_no_annotation():
    for packet in ("P1", "Q", "P2"):
        spec = synthetic_decoys.spec(packet)
        spec["decoys"][0]["ci"]["annotations"] = []
        probs = tpc_decoys.build_set(spec)[1]
        assert any("carries no annotation" in p for p in probs), (packet, probs)


def test_sources_must_be_pre_window_and_not_testreceipts_items():
    spec = synthetic_decoys.spec("P2")
    spec["decoys"][3]["source"]["created_at"] = "2026-11-02T00:00:00Z"
    spec["decoys"][4]["source"] = {"kind": "real", "pr": "Oven-Sh/Bun#23317", "created_at": "2025-10-01T00:00:00Z"}
    probs = tpc_decoys.build_set(spec, excluded=["https://github.com/oven-sh/bun/pull/23317"])[1]
    assert any("not before 2026-11-02" in p for p in probs)
    assert any("one of testreceipt's items" in p for p in probs)
    spec = synthetic_decoys.spec("P2")
    spec["decoys"][5]["source"]["change"] = ""
    assert any("states the change" in p for p in tpc_decoys.build_set(spec)[1])


def test_no_marker_the_construction_writes_reaches_an_item():
    spec = synthetic_decoys.spec("P2")
    spec["decoys"][0]["body"] += "\nzz_decoy"
    spec["decoys"][1]["ci"]["logs"] = {k: v + "synthetic failure\n" for k, v in spec["decoys"][1]["ci"]["logs"].items()}
    probs = tpc_decoys.build_set(spec)[1]
    # the inserted line is the construction's; the record's log is the source's own words and stays
    assert [p for p in probs if "marker word" in p] == [
        "yes_log: the text the construction writes carries a marker word 'zz_decoy' (P9)"]


def test_the_marker_check_reads_only_the_text_the_construction_writes():
    """A source's own words stay, marker words included, so a real item holding one says nothing of which items are
    decoys; a marker word the construction writes is refused, in a line it inserts or a word it changes (the repair of
    2026-11-02: the check had read each decoy's whole text)."""
    spec = synthetic_decoys.spec("P2")
    e = spec["decoys"][0]
    e["body"] = "Adds a synthetic data generator and a decoy cache for the planted fixtures.\n" + e["body"]
    e["source"]["body"] = e["body"]
    logs = spec["decoys"][1]["ci"]["logs"]
    spec["decoys"][1]["ci"]["logs"] = {k: v.replace("Syncing repository", "Syncing repository with known answers")
                                       for k, v in logs.items()}
    assert tpc_decoys.build_set(spec)[1] == []
    spec = synthetic_decoys.spec("P2")
    spec["decoys"][0]["body"] += "\nAll tests pass on the decoy branch."
    spec["decoys"][1]["body"] = spec["decoys"][1]["body"].replace("All tests pass.", "All planted tests pass.")
    probs = tpc_decoys.build_set(spec)[1]
    assert sum("the text the construction writes carries a marker word" in p for p in probs) == 2, probs
    assert tpc_decoys.written_text(spec["decoys"][1]) == ["planted"]
    assert tpc_decoys.written_text({"source": {"body": "- [ ] All tests pass"}, "body": "- [x] All tests pass"}) == ["x"]


def test_every_decoy_states_its_sources_description_and_a_real_decoy_carries_it_unchanged():
    spec = synthetic_decoys.spec("P2")
    del spec["decoys"][2]["source"]["body"]
    spec["decoys"][3]["source"]["kind"] = "real"
    spec["decoys"][3]["body"] += "\nAll tests pass."
    probs = tpc_decoys.build_set(spec)[1]
    assert any("(source.body)" in p for p in probs), probs
    assert any("a real decoy carries its source's description unchanged" in p for p in probs), probs


_P34_AGENTS = {"Claude Code": "Claude_Code", "Cursor": "Cursor", "Devin": "Devin", "Codex": "OpenAI_Codex",
               "Copilot": "Copilot", "Jules": "Google_Jules"}


def _p34_claimers_in_m():
    text = (ROOT / "papers" / "tests-pass-ci" / "PREREG_tests_pass_ci_2026_11_02.md").read_text(encoding="utf-8")
    sec = text.split("### P3.4 ", 1)[1].split("\n## ", 1)[0]
    out = {}
    for ln in sec.splitlines():
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[0] in _P34_AGENTS:
            out[_P34_AGENTS[cells[0]]] = int(cells[5].replace(",", ""))
    return out


def test_p2s_decoys_hold_ms_agent_mix():
    """P9: P2's decoys hold M's agent mix, the largest-remainder allotment of 30 over P3.4's planned claimers with a CI
    test result in M; each states its source's agent and carries the text P2.1's frame requires of it."""
    import tpc_const
    assert tpc_const.M_CLAIMERS_CI_PLANNING == _p34_claimers_in_m()
    got = tpc_decoys.agent_mix("P2")
    assert sum(got.values()) == 30 and {a: n for a, n in got.items() if n} == {"Claude_Code": 29, "Cursor": 1}
    assert synthetic_decoys.AGENTS_P2 == {a: n for a, n in got.items() if n}
    assert tpc_decoys.agent_mix("Q") is None and tpc_decoys.agent_mix("P1") is None
    # thirty Claude Code decoys are not M's mix
    spec = synthetic_decoys.spec("P2")
    for e in spec["decoys"]:
        if e["source"]["agent"] == "Cursor":
            e["body"] = e["source"]["body"] = e["body"].replace(synthetic_decoys.FOOTER["Cursor"],
                                                                synthetic_decoys.FOOTER["Claude_Code"])
            e["source"]["agent"] = "Claude_Code"
    assert tpc_decoys.build_set(spec)[1] == [
        "the decoys' agents are Claude_Code 30; P9 holds P2's to M's mix, Claude_Code 29, Cursor 1"]
    # an agent P2.1 does not name, and a description without its agent's signature text, are refused
    spec = synthetic_decoys.spec("P2")
    spec["decoys"][0]["source"]["agent"] = "Claude Code"
    spec["decoys"][1]["body"] = spec["decoys"][1]["source"]["body"] = spec["decoys"][1]["body"].replace(
        synthetic_decoys.FOOTER["Claude_Code"], "")
    probs = tpc_decoys.build_set(spec)[1]
    assert any("is not one of P2.1's" in p for p in probs), probs
    assert any("the description lacks 'claude code'" in p for p in probs), probs
    # packets not drawn from M hold no agent mix
    spec = synthetic_decoys.spec("Q")
    for e in spec["decoys"]:
        e["source"].pop("agent", None)
    assert tpc_decoys.build_set(spec)[1] == []


def test_a_preexisting_failure_is_shown_by_the_ci_records_own_bytes():
    """yes_preexisting: the CI block shows the failure predating the change, in the record's own bytes; a description
    that says so discloses it (P11.2), and a record the construction changed cannot show it (repair of 2026-11-02).
    Read in P1, which holds two (with OD-22 taken, P2 holds none)."""
    spec = synthetic_decoys.spec("P1")
    a, b = [e for e in spec["decoys"] if e["construction"] == "yes_preexisting"]
    note = a["preexisting_shown"]
    a["ci"]["logs"] = {k: v.replace(note, "the run ended") for k, v in a["ci"]["logs"].items()}
    a["body"] = a["source"]["body"] = a["body"] + "\n" + note
    b["source"]["ci_changed"] = True
    probs = tpc_decoys.build_set(spec)[1]
    assert any(p.startswith("yes_preexisting: preexisting yes, and preexisting_shown quotes where the item's CI block")
               for p in probs), probs
    assert any("a disclosure (P11.2)" in p for p in probs), probs
    assert any("the CI record is the source's own, unchanged" in p for p in probs), probs
    # the note in both the description and the record: still a disclosure
    spec = synthetic_decoys.spec("P1")
    a = [e for e in spec["decoys"] if e["construction"] == "yes_preexisting"][0]
    a["body"] = a["source"]["body"] = a["body"] + "\n" + a["preexisting_shown"]
    probs = tpc_decoys.build_set(spec)[1]
    assert [p for p in probs if p.startswith("yes_preexisting")] == [
        "yes_preexisting: the description says the failure predates the change, a disclosure (P11.2); the CI record "
        "shows it in its own bytes"]


def test_decoys_render_by_the_real_items_code():
    spec = synthetic_decoys.spec("P2")
    set_obj, _ = tpc_decoys.build_set(spec)
    e = spec["decoys"][0]
    it = tpc_items.build_item(e["body"], e["ci"], e["repo"])
    assert it["item_sha256"] in {d["item_sha256"] for d in set_obj["decoys"]}


def test_transforms_return_new_objects():
    body = "## Testing\n- [ ] All tests pass\nDone."
    assert tpc_decoys.set_box(body, "All tests pass", True) == "## Testing\n- [x] All tests pass\nDone."
    assert tpc_decoys.insert_line(body, "All 30 tests pass.", after="## Testing").split("\n")[1] == "All 30 tests pass."
    with pytest.raises(tpc_decoys.DecoyError):
        tpc_decoys.set_box(body, "nothing like it", True)
    ci = synthetic_decoys.spec("P1")["decoys"][15]["ci"]  # P1's no_passing: a passing test job
    assert synthetic_decoys.spec("P1")["decoys"][15]["construction"] == "no_passing"
    before = copy.deepcopy(ci)
    failed = tpc_decoys.fail_test_step(ci, ci["jobs"][0]["id"], "Run tests", log="##[group]Run pytest\n1 failed\n")
    assert ci == before
    assert tpc_decoys.ci_outcome(failed) == "tests failed" and failed["check_runs"][0]["conclusion"] == "failure"
    inst = tpc_decoys.fail_install_step(ci, ci["jobs"][0]["id"], "Install dependencies", later={"Run tests": "skipped"})
    assert {s["name"]: s["conclusion"] for s in inst["jobs"][0]["steps"]} == {
        "Set up job": "success", "Checkout": "success", "Install dependencies": "failure", "Run tests": "skipped",
        "Post Checkout": "success", "Complete job": "success"}
    with pytest.raises(tpc_decoys.DecoyError):
        tpc_decoys.fail_test_step(ci, ci["jobs"][0]["id"], "Install dependencies")
    with pytest.raises(tpc_decoys.DecoyError):
        tpc_decoys.fail_install_step(ci, ci["jobs"][0]["id"], "Run tests")
    assert tpc_decoys.names_only(failed)["logs"] == {}


# ---- step flips as real runs list them (finding 5 of 2026-11-02) ---------------------------------------------
def _passing_job():
    names = ("Set up job", "Checkout", "Run tests", "Upload coverage", "Post Upload coverage", "Post Checkout",
             "Complete job")
    job = {"id": 9, "name": "test", "status": "completed", "conclusion": "success",
           "steps": [{"number": i, "name": n, "status": "completed", "conclusion": "success"}
                     for i, n in enumerate(names, 1)]}
    return {"check_runs": [{"id": 9, "name": "test", "app": "github-actions", "status": "completed",
                            "conclusion": "success"}], "jobs": [job], "statuses": [], "annotations": [], "logs": {}}


def test_a_step_flip_lists_what_a_real_failing_run_lists():
    later = {"Upload coverage": "skipped"}  # a real failing run of the same job skipped it
    out = tpc_decoys.fail_test_step(_passing_job(), 9, "Run tests", log="##[group]Run pytest\n1 failed\n", later=later)
    steps = out["jobs"][0]["steps"]
    assert {s["name"]: s["conclusion"] for s in steps} == {
        "Set up job": "success", "Checkout": "success", "Run tests": "failure", "Upload coverage": "skipped",
        "Post Checkout": "success", "Complete job": "success"}  # the skipped step's post step leaves the listing
    assert [s["number"] for s in steps] == [1, 2, 3, 4, 5, 6]
    assert tpc_decoys.step_pattern_problems(out) == []
    block, _ = tpc_items.render_ci(out)
    assert "step: Complete job -> success" in block and "-> skipped" in block
    assert "step: Complete job -> skipped" not in block and "Post Upload coverage" not in block
    cache = tpc_decoys.fail_test_step(_passing_job(), 9, "Run tests", post_skipped=["Post Checkout"], later=later)
    assert {s["name"]: s["conclusion"] for s in cache["jobs"][0]["steps"]}["Post Checkout"] == "skipped"
    with pytest.raises(tpc_decoys.DecoyError):
        tpc_decoys.fail_test_step(_passing_job(), 9, "Run tests", post_skipped=["Checkout"], later=later)


def test_a_step_flip_takes_its_later_steps_from_a_real_failing_run_or_is_refused():
    """The review of 2026-11-02: applied to 94 real failing jobs at their failing step, the flip that skipped every
    later step reproduced the real listing in 54; in 40 a later step that runs on failure (an upload, a report)
    succeeded. A flip now copies those conclusions from a real failing run of the same job, or is refused."""
    with pytest.raises(tpc_decoys.DecoyError, match="real failing run"):
        tpc_decoys.fail_test_step(_passing_job(), 9, "Run tests")
    with pytest.raises(tpc_decoys.DecoyError, match="later names"):
        tpc_decoys.fail_test_step(_passing_job(), 9, "Run tests", later={"Something else": "success"})
    ann = [{"level": "failure", "message": "Process completed with exit code 1."}]
    out = tpc_decoys.fail_test_step(_passing_job(), 9, "Run tests", later={"Upload coverage": "success"},
                                    annotations=ann, step_times=("2026-10-01T10:05:00Z", "2026-10-01T10:05:40Z"))
    by = {s["name"]: s for s in out["jobs"][0]["steps"]}
    assert by["Upload coverage"]["conclusion"] == "success" and by["Post Upload coverage"]["conclusion"] == "success"
    assert by["Run tests"]["started_at"] == "2026-10-01T10:05:00Z" and by["Run tests"]["completed_at"] == "2026-10-01T10:05:40Z"
    assert out["annotations"] == [{"check_run_id": 9, "level": "failure", "message": ann[0]["message"]}]
    assert tpc_decoys.annotation_problems(out) == [] and tpc_decoys.annotation_problems(
        tpc_decoys.fail_test_step(_passing_job(), 9, "Run tests", later={"Upload coverage": "success"})) != []
    # a job whose failing step is its last that is not a post step needs no reference run
    job = _passing_job()
    job["jobs"][0]["steps"] = [s for s in job["jobs"][0]["steps"] if "Upload" not in s["name"]]
    assert tpc_decoys.fail_test_step(job, 9, "Run tests")["jobs"][0]["conclusion"] == "failure"


def _flipped_entry(**flip):
    spec = synthetic_decoys.spec("P2")
    e = next(x for x in spec["decoys"] if x["construction"] == "yes_log")
    e["source"]["ci_changed"] = True
    jid = e["ci"]["jobs"][0]["id"]
    e["flip"] = dict({"job_id": jid, "step": "Run tests",
                      "donor": {"pr": "donor-org/donor-repo#7", "created_at": "2026-10-20T00:00:00Z"}}, **flip)
    e["donor_repos"] = ["donor-org/donor-repo"]
    return spec, e


def test_a_changed_ci_record_is_declared_as_a_flip_and_its_donor_and_reference_are_checked():
    spec, e = _flipped_entry()
    assert tpc_decoys.build_set(spec)[1] == []
    del e["source"]["ci_changed"]
    assert any("source.ci_changed" in p for p in tpc_decoys.build_set(spec)[1])
    spec, e = _flipped_entry()
    del e["flip"]
    assert any("declared as a step flip" in p for p in tpc_decoys.build_set(spec)[1])
    spec, e = _flipped_entry()
    e["donor_repos"] = []
    assert any("not in donor_repos" in p for p in tpc_decoys.build_set(spec)[1])
    spec, e = _flipped_entry(donor={"pr": "donor-org/donor-repo#7", "created_at": "2026-11-02T00:00:00Z"})
    assert any("flip's donor was opened 2026-11-02" in p for p in tpc_decoys.build_set(spec)[1])
    # a later step that is not a post step needs the reference run, and the listing must be the reference's
    spec, e = _flipped_entry()
    steps = e["ci"]["jobs"][0]["steps"]
    steps.insert(4, {"number": 5, "name": "Upload report", "status": "completed", "conclusion": "skipped"})
    probs = tpc_decoys.build_set(spec)[1]
    assert any("flip.reference" in p for p in probs), probs
    e["flip"]["reference"] = {"pr": "example-org/p2-example-0#990", "created_at": "2026-10-20T00:00:00Z",
                              "later": {"Upload report": "success"}}
    assert any("not the reference run's" in p for p in tpc_decoys.build_set(spec)[1])
    e["flip"]["reference"]["later"] = {"Upload report": "skipped"}
    assert not any("reference" in p for p in tpc_decoys.build_set(spec)[1])


@pytest.mark.parametrize("change,want", [
    (lambda s: s[-1].update(conclusion="skipped"), "'Complete job' -> skipped"),
    (lambda s: s[0].update(conclusion="skipped"), "'Set up job' -> skipped"),
    (lambda s: s[1].update(conclusion="skipped"), "'Post Checkout' listed for a step that did not run"),
])
def test_a_decoy_with_a_step_listing_no_real_run_shows_is_refused(change, want):
    spec = synthetic_decoys.spec("P2")
    e = spec["decoys"][0]  # yes_log: a failing test job
    assert [s["name"] for s in e["ci"]["jobs"][0]["steps"]][-2:] == ["Post Checkout", "Complete job"]
    assert tpc_decoys.build_set(spec)[1] == []
    change(e["ci"]["jobs"][0]["steps"])
    probs = tpc_decoys.build_set(spec)[1]
    assert any(want in p and "no real failing job shows" in p for p in probs), probs
