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
                                  "November 2026"])
def test_every_date_form_of_p8_1_is_masked(text):
    assert tpc_items.mask_dates(f"at {text} ok") == "at <date> ok"


def test_clock_times_versions_and_counts_stay():
    for text in ("12:34:56", "v1.2.3", "node 20.10.0", "3 failed, 120 passed in 2.10s", "sha a20261029b", "1/2 done"):
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


def _break(name, fn):
    spec = synthetic_decoys.spec("P2")
    e = next(x for x in spec["decoys"] if x["construction"] == name)
    fn(e)
    return tpc_decoys.build_set(spec)[1]


@pytest.mark.parametrize("name,fn,want", [
    ("yes_log", lambda e: e.update(ci=tpc_decoys.names_only(e["ci"])), "a log excerpt is shown"),
    ("yes_log", lambda e: e["key"].update(label="NOT_CONTRADICTED"), "key label"),
    ("yes_names_only", lambda e: e["ci"].update(statuses=[{"context": "ci/x: unit", "state": "success"}]), "test failure"),
    ("yes_stale", lambda e: e.update(last_edited_at="2026-10-01T10:00:00Z"), "STALE"),
    ("yes_preexisting", lambda e: e.update(preexisting_shown="not in the item"), "preexisting_shown"),
    ("no_passing", lambda e: e["ci"]["statuses"].append({"context": "deploy", "state": "failure"}), "nothing fails"),
    ("no_lint_build", lambda e: e["ci"]["jobs"][1]["steps"][1].update(name="Run unit tests"), "nothing that fails"),
    ("no_install_in_test_job", lambda e: e["ci"]["jobs"][0]["steps"][3].update(conclusion="failure"), "installation"),
    ("no_disclosed", lambda e: e.update(disclosure="a sentence the body lacks"), "disclosure"),
    ("no_part_beside", lambda e: e.update(part=""), "part quotes"),
    ("no_not_a_claim", lambda e: e["key"].update(scope="WHOLE"), "key scope"),
])
def test_each_construction_is_checked_by_machine(name, fn, want):
    probs = _break(name, fn)
    assert any(want in p for p in probs), probs


def test_the_set_refuses_a_wrong_mix_a_missing_names_only_and_reuse():
    spec = synthetic_decoys.spec("P2")
    spec["decoys"][0]["construction"] = "no_passing"
    probs = tpc_decoys.build_set(spec)[1]
    assert any("yes_log: 8 decoys" in p for p in probs) and any("no_passing: 3 decoys" in p for p in probs)
    spec = synthetic_decoys.spec("P2")
    d = next(x for x in spec["decoys"] if x["construction"] == "no_disclosed" and not x["ci"]["jobs"])
    d["ci"] = synthetic_decoys.spec("P2")["decoys"][0]["ci"]
    probs = tpc_decoys.build_set(spec)[1]
    assert any("no_disclosed: 0 names-only" in p for p in probs)
    set_p2, _ = tpc_decoys.build_set(synthetic_decoys.spec("P2"))
    probs = tpc_decoys.build_set(synthetic_decoys.spec("P2"), other_sets=[set_p2["decoys"][0]["item_sha256"]])[1]
    assert any("another packet's" in p for p in probs)
    assert tpc_decoys.build_set(synthetic_decoys.spec("P1"),
                                other_sets=[d["item_sha256"] for d in set_p2["decoys"]])[1] == []


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


def test_no_marker_reaches_an_item():
    spec = synthetic_decoys.spec("P2")
    spec["decoys"][0]["body"] += "\nzz_decoy"
    spec["decoys"][1]["ci"]["logs"] = {k: v + "synthetic failure\n" for k, v in spec["decoys"][1]["ci"]["logs"].items()}
    probs = tpc_decoys.build_set(spec)[1]
    assert sum("marker word" in p for p in probs) >= 1


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
    ci = synthetic_decoys.spec("P2")["decoys"][13]["ci"]  # a passing test job
    before = copy.deepcopy(ci)
    failed = tpc_decoys.fail_test_step(ci, ci["jobs"][0]["id"], "Run tests", log="##[group]Run pytest\n1 failed\n")
    assert ci == before
    assert tpc_decoys.ci_outcome(failed) == "tests failed" and failed["check_runs"][0]["conclusion"] == "failure"
    inst = tpc_decoys.fail_install_step(ci, ci["jobs"][0]["id"], "Install dependencies")
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
    out = tpc_decoys.fail_test_step(_passing_job(), 9, "Run tests", log="##[group]Run pytest\n1 failed\n")
    steps = out["jobs"][0]["steps"]
    assert {s["name"]: s["conclusion"] for s in steps} == {
        "Set up job": "success", "Checkout": "success", "Run tests": "failure", "Upload coverage": "skipped",
        "Post Checkout": "success", "Complete job": "success"}  # the skipped step's post step leaves the listing
    assert [s["number"] for s in steps] == [1, 2, 3, 4, 5, 6]
    assert tpc_decoys.step_pattern_problems(out) == []
    block, _ = tpc_items.render_ci(out)
    assert "step: Complete job -> success" in block and "-> skipped" in block
    assert "step: Complete job -> skipped" not in block and "Post Upload coverage" not in block
    cache = tpc_decoys.fail_test_step(_passing_job(), 9, "Run tests", post_skipped=["Post Checkout"])
    assert {s["name"]: s["conclusion"] for s in cache["jobs"][0]["steps"]}["Post Checkout"] == "skipped"
    with pytest.raises(tpc_decoys.DecoyError):
        tpc_decoys.fail_test_step(_passing_job(), 9, "Run tests", post_skipped=["Checkout"])


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
