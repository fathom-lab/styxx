# -*- coding: utf-8 -*-
"""The seal guard (P21), the raw store and its manifest (P4.3), and the GitHub client's limits.

The tests-pass-ci collector (papers/tests-pass-ci/collect/), offline, against fakes (tests/_tests_pass_ci_fakes.py).
"""
from __future__ import annotations

import datetime as dt
import gzip
import json

import pytest

from tests import _tests_pass_ci_fakes  # noqa: F401,E402  (puts the collector on sys.path)
import tpc_gh
import tpc_seal
from tpc_const import UTC
from tpc_seal import SealGuard, SealRefused
from tpc_store import Store

COMMIT = "a" * 40
CODE = "c" * 64
ROW = ("| the collector: frame, draw, read, screen wrapper, rule R | `papers/tests-pass-ci/collect/` | "
       + CODE + " | frozen at the signing commit |\n")
SIGNED = ("# PREREG - tests-pass-ci\n\n**Signature.** Signed by the operator on 2026-11-05: \"go\".\n\n" + ROW).encode()
UNSIGNED = SIGNED.replace(b"**Signature.** Signed by the operator on 2026-11-05: \"go\".",
                          b"**Signature.** Unsigned. At the signing commit...")
GOOD = "prereg_commit: " + COMMIT + "\nfreeze_anchor: 2026-11-06T12:00:00Z\n"


def guard(tmp_path, text=None, blob=SIGNED, now=dt.datetime(2026, 11, 7, tzinfo=UTC), on_main=True, code=CODE):
    f = tmp_path / "SEALED"
    if text is not None:
        f.write_bytes(text.encode("utf-8"))
    fetched = []
    g = SealGuard(sealed_file=f, repo=tmp_path, now=lambda: now,
                  blob_reader=lambda repo, c, p: blob if c == COMMIT else None,
                  fetch=lambda repo: fetched.append(repo) or "ok",
                  on_main=lambda repo, c: on_main and c == COMMIT,
                  running_code=lambda: code)
    g.fetched = fetched
    return g


def test_no_sealed_file_refuses_window_and_allows_pre_window(tmp_path):
    g = guard(tmp_path)
    assert not g.open
    assert g.fetched == []  # no SEALED file: no network either
    with pytest.raises(SealRefused) as e:
        g.check("2026-11-02T00:00:00Z", "CI read")
    assert "SEAL_GUARD" in str(e.value)
    with pytest.raises(SealRefused):
        g.check("2026-11-15T10:00:00Z", "body read")
    with pytest.raises(SealRefused):
        g.check(None, "body read")  # unknown time counts as window
    g.check("2026-11-01T23:59:59Z", "CI read")  # pre-window: allowed
    g.check("2026-10-29T12:00:00Z", "CI read")


# Every case the integrity review's probe opened (review/integrity/seal/probe_seal_output.txt), and the
# checks the review asked for, each refused by name.
@pytest.mark.parametrize("name,text,blob,kw,why", [
    ("commit not 40 hex", "prereg_commit: nothex\nfreeze_anchor: 2026-11-06T12:00:00Z\n", SIGNED, {}, "40-hex"),
    ("no freeze anchor", "prereg_commit: " + COMMIT + "\n", SIGNED, {}, "no freeze_anchor"),
    ("anchor unreadable", "prereg_commit: " + COMMIT + "\nfreeze_anchor: soon\n", SIGNED, {}, "unreadable"),
    ("anchor in the future", "prereg_commit: " + COMMIT + "\nfreeze_anchor: 2026-11-08T00:00:00Z\n", SIGNED, {},
     "before the freeze anchor"),
    ("commit on a side branch", GOOD, SIGNED, {"on_main": False}, "not an ancestor of origin/main"),
    ("commit unknown", "prereg_commit: " + "b" * 40 + "\nfreeze_anchor: 2026-11-06T12:00:00Z\n", SIGNED, {},
     "not an ancestor"),
    ("junk at the path", GOOD, b"not a prereg at all\n", {}, "not a PREREG"),
    ("the unsigned draft", GOOD, UNSIGNED, {}, "no positive"),
    ("'Unsigned (draft...)'", GOOD, SIGNED.replace(b"Signed by the operator", b"Unsigned (draft, operator to sign)"),
     {}, "no positive"),
    ("re-wrapped signature line", GOOD, SIGNED.replace(b"**Signature.** Signed", b"**Signature.**\nSigned"), {},
     "no positive"),
    ("paragraph moved away", GOOD, SIGNED.replace(b"**Signature.** Signed by the operator on 2026-11-05: \"go\".",
                                                  b"(signature paragraph moved to the end)"), {}, "no positive"),
    ("signed line beside an unsigned one", GOOD, SIGNED + b"**Signature.** Unsigned.\n", {}, "no positive"),
    ("signed, lock table still TO-FILL", GOOD, SIGNED + b"| decoy key digest P2 | TO-FILL at the signing commit |\n",
     {}, "TO-FILL at the signing commit"),
    ("collector row without a digest", GOOD, SIGNED.replace(CODE.encode(), b"TO-FILL"), {}, "holds 0 sha256"),
    ("collector row with two digests", GOOD, SIGNED.replace(CODE.encode(), (CODE + " " + "d" * 64).encode()), {},
     "holds 2 sha256"),
    ("running code is not the sealed code", GOOD, SIGNED, {"code": "e" * 64}, "is not the sealed one"),
])
def test_the_seal_stays_shut_unless_every_check_holds(tmp_path, name, text, blob, kw, why):
    g = guard(tmp_path, text, blob, **kw)
    assert not g.open, name
    assert why in g.state["why"], (name, g.state["why"])
    with pytest.raises(SealRefused):
        g.check("2026-11-03T00:00:00Z", "CI read")
    g.check("2026-10-29T12:00:00Z", "CI read")  # pre-window reads never depend on the seal


def test_a_signed_commit_on_main_with_the_running_code_opens_the_seal(tmp_path):
    g = guard(tmp_path, GOOD)
    assert g.open, g.state
    assert g.fetched == [tmp_path]  # origin fetched before the ancestry check
    assert len(g.state["prereg_blob_sha256"]) == 64
    assert g.state["collector_sealed"] == [CODE] and g.state["collector_running"] == CODE
    g.check("2026-11-03T00:00:00Z", "CI read")


def test_prose_that_quotes_the_markers_mid_line_does_not_shut_the_seal(tmp_path):
    """P21 and the PREREG's header quote the markers in prose; only the signature line and lock-table rows count."""
    prose = (b"The guard needs a line that begins \"**Signature.** Signed\", no \"**Signature.** Unsigned\" and no\n"
             b"lock row \"TO-FILL at the signing commit\"; the rows marked TO-FILL at the signing commit are filled.\n")
    g = guard(tmp_path, GOOD, SIGNED + prose)
    assert g.open, g.state
    assert tpc_seal.signature_lines((SIGNED + prose).decode()) == [
        "**Signature.** Signed by the operator on 2026-11-05: \"go\"."]


def test_code_digests_cover_the_code_and_not_the_tests(tmp_path):
    (tmp_path / "a.py").write_bytes(b"x = 1\n")
    (tmp_path / "vendor" / "testreceipt").mkdir(parents=True)
    (tmp_path / "vendor" / "testreceipt" / "claims.py").write_bytes(b"y = 2\n")
    (tmp_path / "vendor" / "testreceipt" / "LICENSE").write_bytes(b"MIT\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_bytes(b"z = 3\n")
    (tmp_path / "README.md").write_bytes(b"readme\n")
    d = tpc_seal.code_digests(tmp_path)
    assert sorted(d["files"]) == ["a.py", "vendor/testreceipt/LICENSE", "vendor/testreceipt/claims.py"]
    before = d["combined"]
    (tmp_path / "tests" / "test_a.py").write_bytes(b"z = 4\n")
    assert tpc_seal.code_digests(tmp_path)["combined"] == before
    (tmp_path / "a.py").write_bytes(b"x = 2\n")
    assert tpc_seal.code_digests(tmp_path)["combined"] != before


def test_the_collector_row_is_read_from_the_lock_table_line_only():
    dry = "| the collector's window dry run | 387 searches | sha256 `" + "0" * 64 + "` |\n"  # section 0.2's row
    text = dry + "| the beacon | x | " + "f" * 64 + " |\n" + ROW + "| `styxx.mind` | y | " + "9" * 64 + " |\n"
    assert tpc_seal.collector_row_digests(text) == [CODE]
    assert tpc_seal.collector_row_digests("no table here") == []
    assert tpc_seal.collector_row_digests(ROW + ROW) == []  # two collector rows: no digest is read



def test_store_refuses_overwrite_and_records_digests(tmp_path):
    st = Store(tmp_path / "s")
    w = st.write("a/b.json", b"{}\n")
    with pytest.raises(FileExistsError):
        st.write("a/b.json", b"{}\n")
    g1 = st.write_gz("a/c.json", b"hello")
    g2 = st.write_gz("a/d.json", b"hello")
    assert g1["sha256"] == g2["sha256"]  # mtime=0: same bytes, same gzip
    assert gzip.decompress(st.path("a/c.json.gz").read_bytes()) == b"hello"
    assert g1["raw_sha256"] != g1["sha256"]
    rows = st.read_jsonl("manifest.jsonl")
    assert [r["path"] for r in rows] == ["a/b.json", "a/c.json.gz", "a/d.json.gz"]
    assert rows[0]["sha256"] == w["sha256"]
    with pytest.raises(ValueError):
        st.write("../outside.txt", b"x")


def test_verify_manifest_finds_changed_and_missing_files(tmp_path):
    st = Store(tmp_path / "s")
    st.write("x.txt", b"one")
    st.write("y.txt", b"two")
    st.append_jsonl("log.jsonl", {"a": 1})
    st.seal_log("log.jsonl")
    st.append_jsonl("log.jsonl", {"a": 2})  # growing after the seal is fine
    assert st.verify_manifest()["differ"] == []
    st.path("x.txt").write_bytes(b"ONE")
    st.path("y.txt").unlink()
    res = st.verify_manifest()
    assert res["differ"] == ["x.txt"] and res["missing"] == ["y.txt"]


class Transport:
    def __init__(self, answers):
        self.answers = list(answers)
        self.seen = []

    def __call__(self, method, url, headers, body, timeout):
        self.seen.append((method, url, dict(headers)))
        return self.answers.pop(0)


def client(answers, **kw):
    log = []
    clock = [1000.0]
    slept = []

    def sleep(s):
        slept.append(s)
        clock[0] += s

    t = Transport(answers)
    gh = tpc_gh.GitHub(log.append, token="TESTTOKEN", transport=t, sleep=sleep, clock=lambda: clock[0], **kw)
    return gh, t, log, slept


def test_rest_waits_out_retry_after_and_logs_every_call():
    gh, t, log, slept = client([(403, {"Retry-After": "30"}, b'{"message":"You have exceeded a secondary rate limit"}'),
                                (200, {"X-RateLimit-Remaining": "4000", "X-RateLimit-Reset": "5000"}, b'{"ok":1}')])
    r = gh.rest("/repos/o/r/pulls/1")
    assert r.status == 200 and r.data == {"ok": 1}
    assert 31 in slept
    assert [x["status"] for x in log if x["kind"] == "rest"] == [403, 200]
    assert all({"at", "kind", "what", "status", "bytes", "secs"} <= set(x) for x in log)
    assert "TESTTOKEN" not in json.dumps(log)


def test_rest_sleeps_to_reset_under_the_reserve_and_stops_past_max_wait():
    gh, t, log, slept = client([(200, {"X-RateLimit-Remaining": "10", "X-RateLimit-Reset": "1600"}, b"{}"),
                                (200, {"X-RateLimit-Remaining": "4999", "X-RateLimit-Reset": "9000"}, b"{}")],
                               core_reserve=150)
    gh.rest("/a")
    gh.rest("/b")
    assert any(abs(s - 602) < 1 for s in slept)
    gh2, _, _, _ = client([(200, {"X-RateLimit-Remaining": "1", "X-RateLimit-Reset": "999999"}, b"{}")], max_wait=60)
    gh2.rest("/a")
    with pytest.raises(tpc_gh.BudgetStop):
        gh2.rest("/b")


def test_redirects_are_not_followed_and_the_token_never_leaves_the_api():
    gh, t, log, slept = client([(302, {"Location": "https://blob.example.net/log?sig=1"}, b""),
                                (200, {}, b"line 1\nline 2\n")])
    api, st, raw, trunc = gh.follow("/repos/o/r/actions/jobs/1/logs", "joblog", max_bytes=8)
    assert api == 302 and st == 200 and raw == b"line 1\nl" and trunc
    assert "Authorization" in t.seen[0][2]
    assert "Authorization" not in t.seen[1][2]
    with pytest.raises(ValueError):
        gh.rest("https://blob.example.net/x")
    with pytest.raises(ValueError):
        gh.blob("https://api.github.com/x", "x")


def test_call_budget_stops_the_run():
    gh, t, log, slept = client([(200, {}, b"{}")] * 5, max_calls=2)
    gh.rest("/a")
    gh.rest("/b")
    with pytest.raises(tpc_gh.BudgetStop):
        gh.rest("/c")


def test_graphql_rate_limited_error_waits_and_retries_and_paces_searches():
    ok = json.dumps({"data": {"rateLimit": {"cost": 1, "remaining": 4000, "resetAt": "2026-11-02T15:00:00Z"},
                              "search": {"issueCount": 0}}}).encode()
    limited = json.dumps({"errors": [{"type": "RATE_LIMITED", "message": "slow down"}], "data": None}).encode()
    gh, t, log, slept = client([(200, {}, limited), (200, {}, ok), (200, {}, ok)], search_pace=2.2)
    gh.graphql("q", {"q": "x"}, kind="search")
    gh.graphql("q", {"q": "x"}, kind="search")
    assert 60 in slept
    assert gh.c.calls["search"] == 3


def test_a_search_403_cools_every_search_down_and_slows_the_pace():
    ok = json.dumps({"data": {"rateLimit": {"cost": 1, "remaining": 4000, "resetAt": "2026-11-02T15:00:00Z"},
                              "search": {"issueCount": 0}}}).encode()
    gh, t, log, slept = client([(403, {"Retry-After": "60"}, b'{"message":"secondary rate limit"}'), (200, {}, ok), (200, {}, ok)],
                               search_pace=2.2)
    gh.graphql("q", {"q": "x"}, kind="search")
    assert any(x["kind"] == "wait" and x["what"] == "search cooldown" for x in log)
    assert gh.search_pace == 2.7
    gh.graphql("q", {"q": "y"}, kind="search")
    assert gh.c.calls["search"] == 3
