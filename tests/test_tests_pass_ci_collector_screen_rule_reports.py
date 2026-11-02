# -*- coding: utf-8 -*-
"""The screen (P5), the CI test result (P6), rule R and the log cleaning (P17, P8.3), the report
reader (P18), the beacon check (P3.1), and the vendored code's pins (section 14).

The tests-pass-ci collector (papers/tests-pass-ci/collect/), offline, against fakes (tests/_tests_pass_ci_fakes.py).
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import zipfile

import pytest

from tests import _tests_pass_ci_fakes  # noqa: F401,E402  (puts the collector on sys.path)
import tpc_beacon
import tpc_reports
import tpc_rule
import tpc_screen

from tests._tests_pass_ci_fakes import COLLECTOR  # noqa: E402


# ---- section 14 pins and py3.9 -------------------------------------------------------------------
def test_vendored_testreceipt_is_byte_identical_to_the_pins():
    assert tpc_screen.vendor_digests() == tpc_screen.VENDOR_PINS


def test_every_module_parses_as_python_3_9():
    for f in list(COLLECTOR.glob("*.py")) + list((COLLECTOR / "vendor" / "testreceipt").glob("*.py")):
        ast.parse(f.read_bytes().decode("utf-8"), filename=str(f), feature_version=(3, 9))


# ---- P5 ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("body,kind", [
    ("Summary\n\nAll 312 tests pass.", "claimer"),
    ("## Testing\n- `pytest -q`\n", "implied_only"),
    ("- [ ] All tests pass", "none"),
    ("This should pass CI now.", "none"),
    ("", "none"),
])
def test_screen_kinds(body, kind):
    assert tpc_screen.screen(body)["screen"] == kind


def test_screen_records_fail_claims_and_scope():
    s = tpc_screen.screen("Tests: 3 failed, 10 passed\n\nAll tests pass on Windows.")
    assert s["fail_claim"] is True
    assert any(c["kind"] == "pass" and c["scope"] == "part" for c in s["claims"])


# ---- P6 ----------------------------------------------------------------------------------------
def cr(i, name, concl, app="github-actions"):
    return {"id": i, "name": name, "conclusion": concl, "app": {"slug": app}}


def job(i, name, concl, steps):
    return {"id": i, "name": name, "conclusion": concl, "status": "completed",
            "steps": [{"name": n, "conclusion": c} for n, c in steps]}


def outcome(crs, jobs=(), statuses=()):
    res, _ = tpc_screen.build_ci("h", list(crs), {j["id"]: j for j in jobs}, list(statuses))
    return res.tests()[0]


def test_ci_steps_decide_for_actions_jobs():
    j = job(1, "build", "failure", [("Checkout", "success"), ("Run tests", "failure")])
    assert outcome([cr(1, "build", "failure")], [j]) == "tests failed"
    j2 = job(2, "build", "failure", [("Compile", "failure")])
    assert outcome([cr(2, "build", "failure")], [j2]) == "no test result"


def test_timed_out_reads_as_failure_the_prereg_change():
    assert outcome([cr(1, "test (3.11)", "timed_out", app="circleci-checks")]) == "tests failed"
    j = job(3, "unit", "timed_out", [("Run unit tests", "timed_out")])
    assert outcome([cr(3, "unit", "timed_out")], [j]) == "tests failed"


def test_cancelled_skipped_neutral_are_no_result_and_status_error_is_failure():
    for c in ("cancelled", "skipped", "neutral", "stale", None):
        assert outcome([cr(1, "test", c, app="other")]) == "no test result"
    assert outcome([], statuses=[{"context": "ci/circleci: unit", "state": "error"}]) == "tests failed"
    assert outcome([], statuses=[{"context": "ci/circleci: unit", "state": "success"}]) == "tests passed"
    assert outcome([]) == "no CI"


def test_steps_attach_only_to_success_failure_timed_out_jobs_as_testreceipt_does():
    j = job(4, "test", "cancelled", [("Run tests", "success"), ("Upload", "cancelled")])
    # a cancelled job keeps no steps: the name decides, and cancelled is no result
    assert outcome([cr(4, "test", "cancelled")], [j]) == "no test result"


def test_test_like_job_by_name_or_step():
    assert tpc_screen.test_like_job(job(1, "pytest (3.12)", "success", []))
    assert tpc_screen.test_like_job(job(2, "build", "success", [("Run unit tests", "success")]))
    assert not tpc_screen.test_like_job(job(3, "lint", "failure", [("ruff", "failure")]))


# ---- P17 / P8.3 ----------------------------------------------------------------------------------
def test_summary_patterns_are_the_prereg_bytes():
    assert hashlib.sha256("\n".join(tpc_rule.SUMMARY_LINES).encode("utf-8")).hexdigest() == tpc_rule.SUMMARY_SHA256
    assert len(tpc_rule.SUMMARY) == 12


@pytest.mark.parametrize("line,hit", [
    ("========================= 1 failed, 1 passed in 27.09s =========================", True),
    ("Tests:       2 failed, 40 passed, 42 total", True),
    ("test result: FAILED. 10 passed; 1 failed", True),
    ("--- FAIL: TestParse (0.00s)", True),
    ("FAIL\tgithub.com/x/y\t0.012s", True),
    ("  3 failing", True),
    ("12 examples, 2 failures", True),
    ("Tests run: 10, Failures: 1, Errors: 0, Skipped: 0", True),
    ("Tests run: 10, Failures: 0, Errors: 2, Skipped: 0", True),
    ("Failed!  - Failed:     1, Passed:    99", True),
    ("FAILURES!", True),
    ("12 tests completed, 1 failed", True),
    ("========================= 40 passed in 2.1s =========================", False),
    ("Tests run: 10, Failures: 0, Errors: 0", False),
    ("12 examples, 0 failures", False),
    ("ok  \tgithub.com/x/y\t0.01s", False),
])
def test_each_summary_pattern(line, hit):
    assert bool(tpc_rule.summary_hits([line])) is hit


def test_clean_lines_strips_timestamps_and_ansi():
    raw = ("﻿2026-10-29T20:57:01.1234567Z \x1b[31m--- FAIL: TestX (0.00s)\x1b[0m\n"
           "2026-10-29T20:57:02.0000000Z FAIL\tpkg\t0.1s\n").encode("utf-8")
    lines = tpc_rule.clean_lines(raw)
    assert lines == ["--- FAIL: TestX (0.00s)", "FAIL\tpkg\t0.1s"]
    assert len(tpc_rule.summary_hits(lines)) == 2


@pytest.mark.parametrize("text,ok", [
    ("All tests pass.", True),
    ("- [x] All tests pass", False),
    ("All tests pass (`pytest -k parser`)", False),
    ("All tests pass with `go test ./... -run TestX`", False),
    ("Ran `pytest tests/api_test.py`; all tests pass", False),
    # P17.1's path list names pytest's test_<name>.py too (added by the reviser, 2026-11-02)
    ("Ran `pytest tests/test_api.py`; all tests pass", False),
    # P17.1 item 1: a local run is a part claim under the rubric, so R never fires on it
    ("Tests pass locally.", False),
    ("All tests pass on my machine", False),
    ("Ran the full suite locally, all green; all tests pass", False),
    ("All tests pass (`jest src/a.test.ts`)", False),
    ("`cargo test -p mycrate` all tests pass", False),
    ("All tests pass: `pytest tests/unit.py::test_one`", False),
])
def test_rule_item1_whole_claim_lines(text, ok):
    c = [{"kind": "pass", "explicit": True, "scope": "full", "text": text}]
    assert bool(tpc_rule.whole_claim_lines(c)) is ok


def test_rule_r_needs_all_four_items():
    claims = [{"kind": "pass", "explicit": True, "scope": "full", "text": "All tests pass."}]
    jobs = [job(9, "build", "failure", [("Run tests", "failure")])]
    hits = {9: [(10, 0, "=== 2 failed, 3 passed in 1s ===")]}
    assert tpc_rule.evaluate(claims, False, False, jobs, hits)["fires"] is True
    assert not tpc_rule.evaluate(claims, True, False, jobs, hits)["fires"]  # a fail claim
    assert not tpc_rule.evaluate(claims, False, True, jobs, hits)["fires"]  # STALE
    assert not tpc_rule.evaluate(claims, False, None, jobs, hits)["fires"]  # staleness unknown
    assert not tpc_rule.evaluate(claims, False, False, jobs, {9: []})["fires"]  # no summary line
    assert not tpc_rule.evaluate(claims, False, False, jobs, {})["fires"]  # log not read
    part = [dict(claims[0], scope="part")]
    assert not tpc_rule.evaluate(part, False, False, jobs, hits)["fires"]
    cancelled = [job(9, "build", "cancelled", [("Run tests", "failure")])]
    assert not tpc_rule.evaluate(claims, False, False, cancelled, hits)["fires"]


# ---- P18 ---------------------------------------------------------------------------------------
def zbytes(files):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for n, d in files.items():
            z.writestr(n, d)
    return b.getvalue()


def test_inspect_zip_counts_junit_trx_and_intoto_only():
    z = zbytes({
        "a/results.xml": b'\xef\xbb\xbf<?xml version="1.0"?>\n<!-- x -->\n<testsuites><testsuite/></testsuites>',
        "b/other.xml": b"<coverage/>",
        "c/run.trx": b"<TestRun/>",
        "d/att.json": json.dumps({"predicateType": "https://in-toto.io/attestation/test-result/v0.1"}).encode(),
        "e/notes.json": b'{"a": 1}',
        "f/ns.xml": b'<?xml version="1.0"?><ns:testsuite xmlns:ns="x"/>',
    })
    got = tpc_reports.inspect_zip(z)
    kinds = sorted((c["name"], c["kind"]) for c in got["counted"])
    assert kinds == [("a/results.xml", "junit_xml"), ("c/run.trx", "trx"), ("d/att.json", "intoto_test_result"),
                     ("f/ns.xml", "junit_xml")]
    assert tpc_reports.inspect_zip(b"not a zip")["bad_zip"]


def test_report_artifact_names():
    assert tpc_reports.report_like_name("integration-results-19")
    assert tpc_reports.report_like_name("JUnit")
    assert not tpc_reports.report_like_name("dist-wheels")


# ---- P3.1 --------------------------------------------------------------------------------------
GENESIS = ("0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e"
           "67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d1dac2b7c")
GENESIS_ID = "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"


def test_header_check_on_the_genesis_block():
    h = tpc_beacon.parse_header(GENESIS)
    assert h["id"] == GENESIS_ID and h["pow_ok"]
    bad = GENESIS[:-2] + "00"
    assert not tpc_beacon.parse_header(bad)["pow_ok"]


def test_read_beacon_needs_two_agreeing_explorers():
    def get_ok(url):
        if url.endswith("/block-height/0"):
            return GENESIS_ID
        if url.endswith("/header"):
            return GENESIS
        return "975010"

    b = tpc_beacon.read_beacon(0, "test", get=get_ok)
    assert b["beacon_hex"] == GENESIS_ID and b["kind"] == "bitcoin"
    assert tpc_beacon.validate(b, official=True) is None

    def get_split(url):
        if "mempool" in url and url.endswith("/block-height/0"):
            return "11" * 32
        return get_ok(url)

    with pytest.raises(tpc_beacon.BeaconError):
        tpc_beacon.read_beacon(0, "test", get=get_split)
    dev = tpc_beacon.dev_beacon("x")
    assert tpc_beacon.validate(dev, official=False) is None
    assert "Bitcoin" in tpc_beacon.validate(dev, official=True)
