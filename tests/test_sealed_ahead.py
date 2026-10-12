# -*- coding: utf-8 -*-
"""papers/sealed-ahead: population, pool, accuse, draw and stranger on synthetic GH Archive lines.

No network: every fetch here is a fake that serves bytes built in the test. The gate is the
checkout's styxx.diffgate, named by its own LF sha256, so these tests exercise the real gate.
"""
from __future__ import annotations

import gzip
import importlib.util
import io
import json
import sys
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SA = ROOT / "papers" / "sealed-ahead"
sys.path.insert(0, str(ROOT))


def _load(name):
    key = "sealed_ahead_" + name
    mod = sys.modules.get(key)
    if mod is None:
        spec = importlib.util.spec_from_file_location(key, SA / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return mod


common = _load("common")
population = _load("population")
pool = _load("pool")
import styxx.diffgate as _dg  # noqa: E402  -- the checkout's gate, named by its own LF sha256

GATE_SHA = common.lf_sha256(Path(_dg.__file__).read_bytes())

UTC = timezone.utc
T0 = "2026-10-15T00:00:00Z"
T1 = "2026-10-15T03:00:00Z"
TABLE = population.load_identity_table()

# The operative rules of identity_table.json, pinned: editing a rule must be a visible act.
RULES_SHA256_PIN = "2fce0c15763231ef7a32a59e1e93091c286258136dd7782ed79dc84da6c6ad66"


def params(**over):
    p = {
        "schema": common.PARAMS_SCHEMA,
        "t0": T0,
        "t1": T1,
        "identity_rules_sha256": population.rules_sha256(TABLE),
        "young_days": 30,
        "ceiling_hour": common.ceiling_hour_for(common.parse_utc(T0), 30),
        "account_id_ceiling": 5000,
        "repo_id_ceiling": 9000,
        "excluded_owners": ["fathom-lab"],
        "excluded_actor_logins": ["lab-operator"],
        "max_prs_per_repo": None,
        "max_prs_per_actor": None,
        "gap_grace_hours": 24,
        "gate": {"module": "styxx.diffgate", "sha256": GATE_SHA, "styxx_version": "7.49.0", "run": None,
                 "strict": False},
        "fetch": {"body_source": "payload_else_api", "diff_source": "pull", "rate_limit_per_hour": 50,
                  "require_unmoved_head": True},
        "draw": {"n": 3, "bitcoin_height": 970500, "min_confirmations": 6},
    }
    p.update(over)
    return p


def ev(eid, *, login="someone", actor_id=100, repo_id=200, repo="acme/app", number=1, ref="feature/x",
       action="opened", created="2026-10-15T00:10:00Z", head_sha="a" * 40, base_sha="b" * 40,
       typ="PullRequestEvent", full=None):
    """A GH Archive event shaped like the trimmed 2026 payload; `full` adds a pre-trim pull_request object."""
    pr = {"base": {"ref": "main", "repo": {"id": repo_id, "name": repo, "url": "u"}, "sha": base_sha},
          "head": {"ref": ref, "repo": {"id": repo_id, "name": repo, "url": "u"}, "sha": head_sha},
          "id": 10 ** 9 + int(eid), "number": number, "url": "u"}
    if full is not None:
        pr.update(full)
    return {"actor": {"id": actor_id, "login": login, "display_login": login, "gravatar_id": "", "url": "u",
                      "avatar_url": "u"},
            "created_at": created, "id": str(eid),
            "payload": {"action": action, "number": number, "pull_request": pr},
            "public": True, "repo": {"id": repo_id, "name": repo, "url": "u"}, "type": typ}


def lines_of(events) -> bytes:
    return b"".join(json.dumps(e).encode("utf-8") + b"\n" for e in events)


def gz(data: bytes, members: int = 1) -> bytes:
    if members == 1:
        return gzip.compress(data, mtime=0)            # mtime fixed: gzip stamps the clock otherwise
    cut = len(data) // 2
    return gzip.compress(data[:cut], mtime=0) + gzip.compress(data[cut:], mtime=0)


# ================================================================ common

def test_hour_names_are_not_padded_and_padding_is_refused():
    t = datetime(2026, 10, 5, 3, tzinfo=UTC)
    assert common.hour_name(t) == "2026-10-05-3"
    assert common.hour_url("2026-10-05-3") == "https://data.gharchive.org/2026-10-05-3.json.gz"
    assert common.hour_name(datetime(2026, 10, 5, 12, tzinfo=UTC)) == "2026-10-05-12"
    assert common.parse_hour_name("2026-10-05-3") == t
    with pytest.raises(ValueError, match="pads the hour"):
        common.parse_hour_name("2026-10-05-03")
    with pytest.raises(ValueError):
        common.parse_hour_name("2026-10-05-24")
    with pytest.raises(ValueError):
        common.hour_name(t + timedelta(minutes=5))


def test_hours_between_is_half_open_and_ordered():
    hs = common.hours_between(common.parse_utc("2026-10-15T22:00:00Z"), common.parse_utc("2026-10-16T02:00:00Z"))
    assert hs == ["2026-10-15-22", "2026-10-15-23", "2026-10-16-0", "2026-10-16-1"]
    with pytest.raises(ValueError):
        common.hours_between(common.parse_utc(T1), common.parse_utc(T0))


def test_params_validate_and_refuse():
    assert common.validate_params(params()) == []
    bad = params(ceiling_hour="2026-09-15-00")
    assert any("ceiling_hour" in p for p in common.validate_params(bad))
    assert any("unknown keys" in p for p in common.validate_params({**params(), "snapshot": "x"}))
    f = dict(params()["fetch"], rate_limit_per_hour=51)
    assert any("rate_limit" in p for p in common.validate_params(params(fetch=f)))
    f = dict(params()["fetch"], require_unmoved_head=False)
    assert any("require_unmoved_head" in p for p in common.validate_params(params(fetch=f)))


def test_lf_sha256_folds_crlf_only():
    assert common.lf_sha256(b"a\r\nb\n") == common.sha256_hex(b"a\nb\n")
    assert common.lf_sha256(b"a\rb") == common.sha256_hex(b"a\rb")


# ================================================================ population

def test_identity_rules_are_pinned():
    assert population.rules_sha256(TABLE) == RULES_SHA256_PIN
    for r in TABLE["admitted"]:
        assert r["evidence"], r["rule"]


def test_identity_matching():
    rules = population.rules_of(TABLE)
    ids = lambda login, ref: [r["rule"] for r in population.match_identity(login, ref, rules)]
    assert ids("Copilot", "copilot/fix-1") == ["login:copilot"]
    assert ids("copilot", "main") == ["login:copilot"]                  # logins compare without ASCII case
    assert ids("someone", "claude/fix-db") == ["prefix:claude/"]
    assert ids("someone", "Claude/fix-db") == []                       # prefixes compare exactly
    assert ids("someone", "feature/claude/x") == []
    assert ids("devin-ai-integration[bot]", "codex/x") == ["login:devin", "prefix:codex/"]
    assert ids("dependabot[bot]", "dependabot/npm/x") == []


def ctx(**over):
    return population.Context(params(**over), TABLE)


def test_classify_admits_a_trimmed_agent_pr_and_records_what_it_needs():
    kind, rec = population.classify(ev(7, login="Copilot", actor_id=42, ref="copilot/fix"), ctx(), "2026-10-15-0")
    assert kind == "admit"
    assert rec["event_id"] == "7" and rec["repo_id"] == 200 and rec["number"] == 1
    assert rec["head_sha"] == "a" * 40 and rec["base_sha"] == "b" * 40
    assert rec["opened_at"] == "2026-10-15T00:10:00Z" and rec["hour"] == "2026-10-15-0"
    assert rec["family"] == "copilot" and rec["rules"] == ["login:copilot"]
    assert rec["as_opened"] is None                                     # a 2026 payload carries no body


def test_classify_keeps_a_pre_trim_body_as_opened():
    e = ev(8, ref="cursor/x", full={"title": "Fix", "body": "3 files changed"})
    kind, rec = population.classify(e, ctx(), "2026-10-15-0")
    assert kind == "admit" and rec["as_opened"] == {"title": "Fix", "body": "3 files changed"}


@pytest.mark.parametrize("event,expect", [
    (ev(1, action="merged", ref="claude/x"), ("skip", "not_opened")),
    (ev(1, typ="PushEvent"), ("skip", "not_pr_event")),
    (ev(1, ref="feature/x"), ("skip", "not_agent")),
    (ev(1, ref="claude/x", head_sha="zz"), ("exclude", "missing_fields:head_sha")),
    (ev(1, ref="claude/x", repo="fathom-lab/styxx"), ("exclude", "excluded_owner")),
    (ev(1, ref="claude/x", login="Lab-Operator"), ("exclude", "excluded_actor")),
    (ev(1, ref="claude/x", actor_id=5001), ("exclude", "account_too_young")),
    (ev(1, ref="claude/x", repo_id=9001), ("exclude", "repo_too_young")),
    (ev(1, ref="claude/x", created="2026-10-15T03:00:00Z"), ("exclude", "created_at_outside_window")),
    (ev(1, ref="claude/x", created="2026-10-14T23:59:59Z"), ("exclude", "created_at_outside_window")),
])
def test_classify_reasons(event, expect):
    assert population.classify(event, ctx(), "2026-10-15-0") == expect


def test_ceilings_take_the_largest_ids_of_any_event():
    evs = [ev(1, actor_id=10, repo_id=30, typ="PushEvent"), ev(2, actor_id=99, repo_id=5), ev(3, actor_id=7)]
    assert population.id_ceilings(evs) == {"events": 3, "account_id_ceiling": 99, "repo_id_ceiling": 200}


def test_fields_census_names_keysets_not_actors():
    c = population.fields_census([ev(1, ref="claude/x"), ev(2, action="merged"), ev(3, typ="PushEvent")])
    assert c["pr_actions"] == {"merged": 1, "opened": 1}
    assert c["pr_body"] == {"no_body": 2}
    assert c["keysets"]["pull_request"] == [('["base", "head", "id", "number", "url"]', 2)]
    assert "someone" not in json.dumps(c)


def test_bad_lines_are_counted_and_hashed_not_kept():
    raw = [b'{"type": "PushEvent", "id": "1"}\n', b"{not json\n", b"[1,2]\n", b"\n"]
    out = population.scan_lines(raw, ctx(), "2026-10-15-0")
    assert out["events"] == 1 and out["bad_json_lines"] == 2
    assert out["bad_json_lines_sha256"] == common.sha256_hex(b"{not json\n[1,2]")


def test_context_refuses_a_table_the_params_do_not_name():
    with pytest.raises(ValueError, match="rules digest"):
        population.Context(params(identity_rules_sha256="f" * 64), TABLE)


# ================================================================ pool

class FakeArchive:
    """Serves synthetic hours as gzip bytes through pool.stream_gzip_lines; 404s and failures on demand."""

    def __init__(self, hours: dict, members: int = 1):
        self.hours, self.members, self.calls = hours, members, []

    def fetch(self, hour, consume):
        self.calls.append(hour)
        v = self.hours.get(hour, 404)
        if v == 404:
            return {"status": "404", "attempts": 1, "url": common.hour_url(hour)}
        if v == "fail":
            return {"status": "failed", "attempts": 4, "url": common.hour_url(hour), "error": "HTTP 503"}
        result, sha, n = pool.stream_gzip_lines(io.BytesIO(gz(lines_of(v), self.members)), consume)
        return {"status": "ok", "result": result, "compressed_sha256": sha, "compressed_bytes": n, "attempts": 1,
                "url": common.hour_url(hour)}


LATE = lambda: datetime(2026, 10, 20, tzinfo=UTC)          # long after the window: a 404 is a gap


def synthetic_hours():
    return {
        "2026-10-15-0": [ev(11, ref="claude/a", number=1, created="2026-10-15T00:01:00Z"),
                         ev(9, ref="codex/b", number=2, repo_id=201, repo="acme/two", created="2026-10-15T00:02:00Z"),
                         ev(12, ref="feature/x", number=3), ev(13, typ="PushEvent")],
        "2026-10-15-1": [ev(105, ref="claude/a", number=1, created="2026-10-15T01:00:00Z"),   # same PR: dedup
                         ev(100, login="Copilot", ref="copilot/c", number=4, created="2026-10-15T01:05:00Z")],
        "2026-10-15-2": [ev(1000, ref="cursor/d", number=5, created="2026-10-15T02:00:00Z")],
    }


def test_pool_dedups_orders_by_event_id_and_digests():
    d = pool.derive(params(), TABLE, FakeArchive(synthetic_hours()), now=LATE, log=lambda *_: None)
    recs, meta = d["records"], d["meta"]
    assert [r["event_id"] for r in recs] == ["9", "11", "100", "1000"]           # integer order, not text
    assert meta["duplicates"] == 1 and meta["records"] == 4 and meta["status"] == "COMPLETE"
    assert meta["pool_sha256"] == common.sha256_hex(b"".join(common.canon(r) + b"\n" for r in recs))
    assert meta["families_for_sizing"] == {"claude": 1, "codex": 1, "copilot": 1, "cursor": 1}
    assert [h["admitted"] for h in meta["hours"]] == [2, 2, 1]


def test_pool_is_the_same_whatever_the_line_order_or_gzip_members():
    a = pool.derive(params(), TABLE, FakeArchive(synthetic_hours()), now=LATE, log=lambda *_: None)
    rev = {h: list(reversed(v)) for h, v in synthetic_hours().items()}
    b = pool.derive(params(), TABLE, FakeArchive(rev, members=2), now=LATE, log=lambda *_: None)
    assert a["meta"]["pool_sha256"] == b["meta"]["pool_sha256"]


def test_pool_lists_a_404_as_a_gap_only_after_the_grace():
    hours = synthetic_hours()
    del hours["2026-10-15-1"]
    late = pool.derive(params(), TABLE, FakeArchive(hours), now=LATE, log=lambda *_: None)["meta"]
    assert late["status"] == "COMPLETE" and late["gaps"] == ["2026-10-15-1"]
    soon = pool.derive(params(), TABLE, FakeArchive(hours), now=lambda: datetime(2026, 10, 15, 6, tzinfo=UTC),
                       log=lambda *_: None)["meta"]
    assert soon["status"] == "INCOMPLETE" and soon["open_hours"] == [{"hour": "2026-10-15-1", "status": "not_yet"}]


def test_pool_is_incomplete_when_an_hour_fails():
    hours = synthetic_hours()
    hours["2026-10-15-2"] = "fail"
    meta = pool.derive(params(), TABLE, FakeArchive(hours), now=LATE, log=lambda *_: None)["meta"]
    assert meta["status"] == "INCOMPLETE" and meta["open_hours"] == [{"hour": "2026-10-15-2", "status": "failed"}]


def test_pool_caps_per_repository_then_per_account():
    hours = {"2026-10-15-0": [ev(i, ref="claude/x", number=i, created="2026-10-15T00:30:00Z") for i in range(1, 6)]
             + [ev(50, ref="claude/y", number=1, repo_id=300, repo="acme/three", created="2026-10-15T00:40:00Z")]}
    d = pool.derive(params(max_prs_per_repo=2), TABLE, FakeArchive(hours), now=LATE, log=lambda *_: None)
    assert [r["event_id"] for r in d["records"]] == ["1", "2", "50"] and d["meta"]["capped_by_repo"] == 3
    d = pool.derive(params(max_prs_per_actor=1), TABLE, FakeArchive(hours), now=LATE, log=lambda *_: None)
    assert [r["event_id"] for r in d["records"]] == ["1"] and d["meta"]["capped_by_actor"] == 5


def test_pool_cache_resumes_without_refetching_final_hours(tmp_path):
    hours = synthetic_hours()
    del hours["2026-10-15-2"]
    arch = FakeArchive(hours)
    early = pool.derive(params(), TABLE, arch, now=lambda: datetime(2026, 10, 15, 4, tzinfo=UTC),
                        cache_dir=str(tmp_path), log=lambda *_: None)
    assert early["meta"]["status"] == "INCOMPLETE"
    arch2 = FakeArchive(synthetic_hours())
    second = pool.derive(params(), TABLE, arch2, now=LATE, cache_dir=str(tmp_path), log=lambda *_: None)
    assert arch2.calls == ["2026-10-15-2"] and second["meta"]["status"] == "COMPLETE"
    fresh = pool.derive(params(), TABLE, FakeArchive(synthetic_hours()), now=LATE, log=lambda *_: None)
    assert second["meta"]["pool_sha256"] == fresh["meta"]["pool_sha256"]
    meta = pool.write_pool(str(tmp_path / "out"), second)
    assert pool.read_pool(str(tmp_path / "out" / "pool.jsonl")) == second["records"]
    assert meta["derivation_sha256"] == fresh["meta"]["derivation_sha256"]


def test_pool_cache_refuses_an_edited_hour(tmp_path):
    pool.derive(params(), TABLE, FakeArchive(synthetic_hours()), now=LATE, cache_dir=str(tmp_path),
                log=lambda *_: None)
    p = tmp_path / "hours" / "2026-10-15-0.json"
    e = json.loads(p.read_text(encoding="utf-8"))
    e["records"] = e["records"][1:]
    p.write_text(json.dumps(e), encoding="utf-8")
    with pytest.raises(ValueError, match="do not hash"):
        pool.derive(params(), TABLE, FakeArchive(synthetic_hours()), now=LATE, cache_dir=str(tmp_path),
                    log=lambda *_: None)


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_http_archive_streams_retries_and_reads_404():
    body = gz(lines_of(synthetic_hours()["2026-10-15-0"]))
    calls, sleeps = [], []

    def opener(req, timeout):
        calls.append(req.full_url)
        assert req.get_header("User-agent") == pool.USER_AGENT and timeout == 30.0
        assert "Authorization" not in req.headers
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, None)
        return _Resp(body)
    arch = pool.HttpArchive(opener=opener, sleep=sleeps.append, log=lambda *_: None)
    got = arch.fetch("2026-10-15-0", lambda lines: sum(1 for _ in lines))
    assert got["status"] == "ok" and got["result"] == 4 and got["attempts"] == 2 and sleeps == [10]
    assert calls[0] == "https://data.gharchive.org/2026-10-15-0.json.gz"
    assert got["compressed_sha256"] == common.sha256_hex(body) and got["compressed_bytes"] == len(body)

    def gone(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 404, "nope", {}, None)
    assert pool.HttpArchive(opener=gone, sleep=lambda s: None).fetch("2026-10-15-0", list)["status"] == "404"

    def cut(req, timeout):
        return _Resp(body[: len(body) // 2])
    got = pool.HttpArchive(opener=cut, sleep=lambda s: None, log=lambda *_: None).fetch(
        "2026-10-15-0", lambda lines: sum(1 for _ in lines))
    assert got["status"] == "failed" and got["attempts"] == 4


# ================================================================ accuse

accuse = _load("accuse")
GATE = accuse.load_gate(GATE_SHA)
FETCH_PULL = params()["fetch"]
FETCH_COMPARE = dict(FETCH_PULL, diff_source="compare")

ONE_FILE_DIFF = (b"diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n"
                 b"@@ -1,2 +1,3 @@\n x = 1\n+y = 2\n")


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t, self.slept = t, []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


class FakeWeb:
    """github.com and api.github.com as a dict of URL -> bytes | HTTP status | list of those."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, req, timeout):
        url = req.full_url
        self.calls.append(url)
        assert "Authorization" not in req.headers and req.get_header("User-agent") == accuse.USER_AGENT
        v = self.routes.get(url, 404)
        if isinstance(v, list):
            v = v.pop(0)
        if isinstance(v, tuple):
            code, headers = v
            raise urllib.error.HTTPError(url, code, "x", headers, None)
        if isinstance(v, int):
            raise urllib.error.HTTPError(url, v, "x", {}, None)
        return _Resp(v)


def api_pr(number=1, head="a" * 40, title="Fix the parser", body="3 files changed in this PR.",
           full_name="acme/app", updated_at="2000-01-01T00:00:00Z"):
    return json.dumps({"number": number, "title": title, "body": body, "head": {"sha": head},
                       "base": {"repo": {"full_name": full_name}}, "updated_at": updated_at}).encode("utf-8")


def rec_of(eid=7, number=1, **kw):
    kind, rec = population.classify(ev(eid, ref="claude/x", number=number, **kw), ctx(), "2026-10-15-0")
    assert kind == "admit"
    return rec


def github(routes, per_hour=50, clock=None):
    clock = clock or Clock()
    web = FakeWeb(routes)
    lim = accuse.RateLimiter(per_hour, clock=clock, sleep=clock.sleep)
    return accuse.GitHub(lim, opener=web, sleep=clock.sleep, log=lambda *_: None, clock=clock), web


API1 = "https://api.github.com/repositories/200/pulls/1"
PULL1 = "https://github.com/acme/app/pull/1.diff"


def test_gate_digest_matching():
    assert accuse.match_digest(b"a\nb\n", common.sha256_hex(b"a\nb\n")) == "bytes"
    assert accuse.match_digest(b"a\r\nb\r\n", common.sha256_hex(b"a\nb\n")) == "lf"
    assert accuse.match_digest(b"a\nb\n", "0" * 64) is None
    with pytest.raises(accuse.Refused, match="not the frozen gate"):
        accuse.load_gate("0" * 64)
    assert GATE.identity["sha256"] == GATE_SHA and GATE.identity["unidata_version"]


def test_accuse_gates_an_unmoved_head_and_names_the_accusation():
    gh, web = github({API1: api_pr(), PULL1: ONE_FILE_DIFF})
    o, inputs = accuse.accuse_one(rec_of(), gh, GATE, FETCH_PULL)
    assert web.calls == [API1, PULL1]
    assert o["outcome"] == "GATED" and o["inputs"]["body_source"] == "api_now"
    assert o["inputs"]["head_moved"] is False and o["inputs"]["diff_sha256"] == common.sha256_hex(ONE_FILE_DIFF)
    [c] = o["claims"]
    assert (c["kind"], c["verdict"]) == ("files_changed_count", "CONTRADICTED")
    assert o["accusations"] == [c["accusation_id"]]
    assert inputs["body"] == "3 files changed in this PR." and inputs["diff_sha256"] == o["inputs"]["diff_sha256"]


def test_accuse_never_judges_a_moved_head_and_spends_one_request():
    gh, web = github({API1: api_pr(head="c" * 40), PULL1: ONE_FILE_DIFF})
    o, inputs = accuse.accuse_one(rec_of(), gh, GATE, FETCH_PULL)
    assert o["outcome"] == "HEAD_MOVED" and web.calls == [API1] and inputs is None


@pytest.mark.parametrize("routes,expect,n_calls", [
    ({API1: 404}, "PR_GONE", 1),
    ({API1: 451}, "PR_UNAVAILABLE", 1),
    ({API1: api_pr(body="   ")}, "EMPTY_BODY", 1),
    ({API1: api_pr(body=None)}, "EMPTY_BODY", 1),
    ({API1: api_pr()}, "DIFF_GONE", 2),
    ({API1: api_pr(), PULL1: 406}, "DIFF_UNAVAILABLE", 2),
    ({API1: b"<html>not json</html>"}, "PR_UNAVAILABLE", 1),
])
def test_accuse_outcomes(routes, expect, n_calls):
    gh, web = github(routes)
    o, _ = accuse.accuse_one(rec_of(), gh, GATE, FETCH_PULL)
    assert o["outcome"] == expect and len(web.calls) == n_calls


def test_accuse_uses_a_pre_trim_body_as_opened_and_the_compare_route():
    rec = rec_of(full={"title": "Tests", "body": "Added 2 tests for the parser."})
    url = f"https://github.com/acme/app/compare/{'b' * 40}...{'a' * 40}.diff"
    gh, web = github({url: ONE_FILE_DIFF})
    o, _ = accuse.accuse_one(rec, gh, GATE, FETCH_COMPARE)
    assert web.calls == [url] and o["inputs"]["body_source"] == "payload_as_opened"
    assert [(c["kind"], c["verdict"]) for c in o["claims"]] == [("tests_added", "CONTRADICTED")]


def test_accuse_follows_a_renamed_repository_by_id():
    gh, web = github({API1: api_pr(full_name="acme/renamed"),
                      "https://github.com/acme/renamed/pull/1.diff": ONE_FILE_DIFF})
    o, _ = accuse.accuse_one(rec_of(), gh, GATE, FETCH_PULL)
    assert o["outcome"] == "GATED" and o["inputs"]["repo_now"] == "acme/renamed"


def test_github_retries_5xx_waits_out_a_rate_limit_and_refuses_other_hosts():
    clock = Clock()
    limited = (403, {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(int(clock.t) + 100)})
    gh, web = github({API1: [503, limited, api_pr()]}, clock=clock)
    r = gh.get(API1, "application/vnd.github+json")
    assert r["status"] == 200 and r["attempts"] == 3 and len(web.calls) == 3
    assert clock.slept[0] == 30 and 60 <= clock.slept[1] <= 100
    with pytest.raises(accuse.Refused, match="not a host"):
        gh.get("https://example.com/x.diff", "text/plain")
    gh, _ = github({API1: [503, 503, 503, 503]})
    assert gh.get(API1, "x")["status"] == "failed"


def test_rate_limiter_holds_the_hourly_budget_across_restarts(tmp_path):
    clock = Clock()
    state = str(tmp_path / "stamps.txt")
    lim = accuse.RateLimiter(3, clock=clock, sleep=clock.sleep, state_path=state)
    for _ in range(3):
        lim.acquire()
        clock.t += 1
    assert clock.slept == []
    again = accuse.RateLimiter(3, clock=clock, sleep=clock.sleep, state_path=state)   # a restart
    again.acquire()
    assert len(clock.slept) == 1 and 3590 < clock.slept[0] <= 3600.5
    times = [float(x) for x in Path(state).read_text().split()]
    assert len(times) == 4 and times[3] - times[0] >= 3600


def _outcome_with_claims(gate_claims, eid="7", number=1):
    class G:
        def run(self, summary, diff):
            return _dg.DiffGate(verdict="FAIL", base="x", head="y", claims=gate_claims, sentences_total=3)
    rec = rec_of(eid=int(eid), number=number)
    return dict(accuse.gate_outcome(rec, G(), "t", "b", b""), pool_key=f"200#{number}", event_id=eid,
                schema=accuse.OUTCOME_SCHEMA, inputs={}, fetch=[])


def test_overlay_withholding_is_recorded_and_never_an_accusation():
    withheld = _dg.DiffClaim(kind="files_changed_count", text="2 files changed", detail={"n": 2},
                             verdict="UNCHECKABLE",
                             why="CONTRADICTED withheld by PATH-2a (#121): two changed paths differ. main's reading: x")
    accused = _dg.DiffClaim(kind="tests_added", text="added 2 tests", detail={"n": 2}, verdict="CONTRADICTED",
                            why="diff adds 0")
    o = _outcome_with_claims([withheld, accused])
    assert [c["withheld_from"] for c in o["claims"]] == ["CONTRADICTED", None]
    assert len(o["accusations"]) == 1 and o["claims"][1]["accusation_id"] == o["accusations"][0]


def test_accusation_items_are_ordered_by_event_id_and_digested_by_the_beacon():
    from styxx.beacon import pool_sha256
    mk = lambda kind: _dg.DiffClaim(kind=kind, text=kind, detail={}, verdict="CONTRADICTED", why="w")
    a = _outcome_with_claims([mk("tests_added"), mk("symbol_added")], eid="100", number=1)
    b = _outcome_with_claims([mk("only_touches")], eid="9", number=2)
    items = accuse.accusation_items([a, b])
    assert [i[2] for i in items] == ["only_touches", "tests_added", "symbol_added"]
    assert all(isinstance(x, str) for i in items for x in i)
    s = accuse.summarize([rec_of(eid=100, number=1), rec_of(eid=9, number=2)], {"200#1": a, "200#2": b})
    assert s["status"] == "COMPLETE" and s["accusations"] == 3
    assert s["accusations_sha256"] == pool_sha256(items)
    a["claims"][0]["text"] = "edited"
    with pytest.raises(ValueError, match="does not re-derive"):
        accuse.accusation_items([a, b])


def test_run_resumes_after_a_failed_fetch_and_never_drops_a_record(tmp_path):
    recs = [rec_of(eid=7, number=1), rec_of(eid=8, number=2)]
    api2 = "https://api.github.com/repositories/200/pulls/2"
    gh, web = github({API1: api_pr(), PULL1: ONE_FILE_DIFF, api2: [503, 503, 503, 503]})
    s = accuse.run(params(), recs, str(tmp_path), GATE, gh, log=lambda *_: None)
    assert s["status"] == "INCOMPLETE" and s["by_outcome"] == {"FETCH_FAILED": 1, "GATED": 1}
    assert (tmp_path / "inputs" / "200_1.json").exists()
    gh2, web2 = github({api2: api_pr(number=2, body="Only touches docs/ here."),
                        "https://github.com/acme/app/pull/2.diff": ONE_FILE_DIFF})
    s2 = accuse.run(params(), recs, str(tmp_path), GATE, gh2, log=lambda *_: None)
    assert web2.calls == [api2, "https://github.com/acme/app/pull/2.diff"]       # record 1 is not fetched again
    assert s2["status"] == "COMPLETE" and s2["accusations"] == 2       # a count in #1, only_touches in #2
    assert s2["accusations_by_kind"] == {"files_changed_count": 1, "only_touches": 1}
    text = (tmp_path / "accusations.json").read_text(encoding="utf-8")
    assert "family" not in text and "acme" not in text


def test_stable_outcomes_ignore_fetch_times_only():
    gh, _ = github({API1: api_pr(), PULL1: ONE_FILE_DIFF})
    o1, _ = accuse.accuse_one(rec_of(), gh, GATE, FETCH_PULL)
    gh, _ = github({API1: api_pr(), PULL1: ONE_FILE_DIFF}, clock=Clock(1_900_000_000.0))
    o2, _ = accuse.accuse_one(rec_of(), gh, GATE, FETCH_PULL)
    o2["inputs"]["body_fetched_at"] = "2030-01-01T00:00:00Z"
    assert common.canon(accuse.stable(o1)) == common.canon(accuse.stable(o2))
    gh, _ = github({API1: api_pr(body="4 files changed in this PR."), PULL1: ONE_FILE_DIFF})
    o3, _ = accuse.accuse_one(rec_of(), gh, GATE, FETCH_PULL)
    assert common.canon(accuse.stable(o1)) != common.canon(accuse.stable(o3))


# ================================================================ draw

draw = _load("draw")
import hashlib  # noqa: E402
import struct  # noqa: E402

# Two real mainnet headers, heights 0 and 1, and their ids as every explorer prints them.
GENESIS = ("0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f"
           "617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d1dac2b7c")
GENESIS_ID = "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"
BLOCK1 = ("010000006fe28c0ab6f1b372c1a6a246ae63f74f931e8365e15a089c68d6190000000000982051fd1e4ba744bbbe680e1fee1467"
          "7ba1a3c3540bf7b1cdb606e857233e0e61bc6649ffff001d01e36299")
BLOCK1_ID = "00000000839a8e6886ab5951d76f411475428afc90947ee320161bbf18eb6048"


def _d(b):
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()


def test_real_headers_verify_and_chain():
    h = draw.verify_header(GENESIS, GENESIS_ID)
    assert h["pow_ok"] and h["time"] == 1231006505 and h["bits"] == "1d00ffff"
    assert draw.verify_chain([GENESIS, BLOCK1]) == [GENESIS_ID, BLOCK1_ID]
    with pytest.raises(draw.Refused) as e:
        draw.verify_header(GENESIS, BLOCK1_ID)
    assert e.value.code == "HEADER_INVALID"
    tampered = GENESIS[:-8] + "00000000"                        # another nonce: the work no longer holds
    with pytest.raises(draw.Refused, match="above the target"):
        draw.verify_header(tampered)
    with pytest.raises(draw.Refused, match="does not follow"):
        draw.verify_chain([BLOCK1, GENESIS])


def synthetic_block(height=970500, when=None, extra_txids=3, script_override=None):
    """A segwit coinbase carrying `height` (BIP34), a merkle tree over it and fake txids, and an 80-byte
    header over that root with the easiest target, ground to a valid nonce. Synthetic, not a mainnet block."""
    script = script_override if script_override is not None else (
        bytes([3]) + height.to_bytes(3, "little") + b"\x08sealed-ahead")
    out_script = bytes.fromhex("0014") + b"\x11" * 20
    tx = (b"\x02\x00\x00\x00" + b"\x00\x01" + b"\x01" + b"\x00" * 32 + b"\xff\xff\xff\xff"
          + bytes([len(script)]) + script + b"\xff\xff\xff\xff"
          + b"\x01" + (625_000_000).to_bytes(8, "little") + bytes([len(out_script)]) + out_script
          + b"\x01\x20" + b"\x00" * 32 + b"\x00\x00\x00\x00")
    stripped = tx[:4] + tx[6:-38] + tx[-4:]
    leaves = [_d(stripped)] + [hashlib.sha256(bytes([i])).digest() for i in range(extra_txids)]
    level, branch, pos = leaves, [], 0
    while len(level) > 1:
        if len(level) % 2:
            level = level + [level[-1]]
        branch.append(level[pos ^ 1])
        level = [_d(level[i] + level[i + 1]) for i in range(0, len(level), 2)]
        pos //= 2
    when = when if when is not None else int(common.parse_utc(T1).timestamp()) + 86400
    for nonce in range(1000):
        hdr = struct.pack("<I32s32sIII", 0x20000000, b"\x07" * 32, level[0], when, 0x207FFFFF, nonce)
        if draw.parse_header(hdr.hex())["pow_ok"]:
            break
    return {"header": hdr.hex(), "id": _d(hdr)[::-1].hex(), "coinbase": tx.hex(),
            "branch": [b[::-1].hex() for b in branch], "txid": _d(stripped)[::-1].hex()}


def test_bip34_height_is_proved_from_the_coinbase_and_its_branch():
    b = synthetic_block()
    cb = draw.parse_coinbase(b["coinbase"])
    assert cb == {"txid": b["txid"], "height": 970500, "segwit": True}
    assert draw.verify_height(b["header"], b["coinbase"], b["branch"], 970500)["proof"] == "bip34+merkle"
    with pytest.raises(draw.Refused, match="says height 970500"):
        draw.verify_height(b["header"], b["coinbase"], b["branch"], 970501)
    with pytest.raises(draw.Refused, match="merkle root"):
        draw.verify_height(b["header"], b["coinbase"], list(reversed(b["branch"])), 970500)
    bad = synthetic_block(script_override=b"\x6a\x01\x00")
    with pytest.raises(draw.Refused, match="BIP34"):
        draw.parse_coinbase(bad["coinbase"])


def _acc(n_items):
    from styxx.beacon import pool_sha256
    items = [[f"{i:064x}", f"200#{i}", "tests_added"] for i in range(n_items)]
    return {"status": "COMPLETE", "items": items, "accusations_sha256": pool_sha256([tuple(i) for i in items])}


def test_draw_is_styxx_beacon_select_and_nothing_else():
    from styxx import beacon
    acc, b = _acc(10), synthetic_block()
    rec = draw.draw(acc["items"], acc["accusations_sha256"], b["id"], 4)
    expect = beacon.select(b["id"], 4, [tuple(i) for i in acc["items"]])
    assert rec["status"] == "DRAWN" and rec["drawn"] == [e[0] for e in expect]
    assert rec["drawn_sha256"] == common.canon_sha256(rec["drawn"])
    assert rec == draw.draw(acc["items"], acc["accusations_sha256"], b["id"].upper(), 4)
    assert rec["beacon_module_lf_sha256"] == common.lf_sha256(Path(beacon.__file__).read_bytes())
    assert draw.draw(acc["items"], acc["accusations_sha256"], b["id"], 11)["status"] == "SIZING_FAILURE"
    with pytest.raises(draw.Refused) as e:
        draw.draw(acc["items"][1:], acc["accusations_sha256"], b["id"], 4)
    assert e.value.code == "SET_DIGEST_MISMATCH"


def test_checked_draw_refuses_each_broken_precondition_by_name():
    p, acc, b = params(), _acc(5), synthetic_block()
    ok = draw.checked_draw(p, acc, b["header"], 970500, 970505, b["id"], b["coinbase"], b["branch"],
                           anchor_time="2026-10-15T12:00:00Z")
    assert ok["status"] == "DRAWN" and len(ok["drawn"]) == 3 and ok["height_proof"]["proof"] == "bip34+merkle"
    cases = [
        (dict(height=970499), "HEIGHT_NOT_THE_PLANS"),
        (dict(tip_height=970504), "TOO_FEW_CONFIRMATIONS"),
        (dict(anchor_time="2026-10-16T02:00:00Z"), "SET_NOT_ANCHORED_BEFORE"),
        (dict(block_id=GENESIS_ID), "HEADER_INVALID"),
    ]
    for over, code in cases:
        kw = dict(height=970500, tip_height=970505, block_id=b["id"], anchor_time="2026-10-15T12:00:00Z")
        kw.update(over)
        with pytest.raises(draw.Refused) as e:
            draw.checked_draw(p, acc, b["header"], kw["height"], kw["tip_height"], kw["block_id"],
                              anchor_time=kw["anchor_time"])
        assert e.value.code == code
    early = synthetic_block(when=int(common.parse_utc(T1).timestamp()) - 60)
    with pytest.raises(draw.Refused) as e:
        draw.checked_draw(p, acc, early["header"], 970500, 970505)
    assert e.value.code == "BLOCK_NOT_AFTER_WINDOW"
    unordered = draw.checked_draw(p, acc, b["header"], 970500, 970505)
    assert unordered["order"].startswith("not checked")


# ================================================================ stranger

stranger = _load("stranger")
QUIET = lambda *_: None
P2 = params(draw=dict(params()["draw"], n=2))


def routes(**over):
    """The web as the lab saw it: two accusing records, one moved head, one gated without claims."""
    r = {
        "https://api.github.com/repositories/201/pulls/2": api_pr(number=2, full_name="acme/two",
                                                                  body="Added 2 tests for the parser."),
        "https://github.com/acme/two/pull/2.diff": ONE_FILE_DIFF,
        API1: api_pr(), PULL1: ONE_FILE_DIFF,
        "https://api.github.com/repositories/200/pulls/4": api_pr(number=4, head="c" * 40),
        "https://api.github.com/repositories/200/pulls/5": api_pr(number=5, body="Refactor the parser."),
        "https://github.com/acme/app/pull/5.diff": ONE_FILE_DIFF,
    }
    r.update(over)
    return r


def lab_world(tmp_path, hours=None):
    d = pool.derive(P2, TABLE, FakeArchive(hours or synthetic_hours()), now=LATE, log=QUIET)
    gh, _ = github(routes())
    s = accuse.run(P2, d["records"], str(tmp_path / "lab"), GATE, gh, log=QUIET)
    digests = json.loads((tmp_path / "lab" / "outcome_digests.json").read_text(encoding="utf-8"))
    b = synthetic_block()
    drawn = draw.checked_draw(P2, s, b["header"], 970500, 970505, b["id"])
    return d, s, digests, b, drawn


def test_stranger_agrees_end_to_end_from_public_bytes_only(tmp_path):
    d, s, digests, b, drawn = lab_world(tmp_path)
    assert s["by_outcome"] == {"GATED": 3, "HEAD_MOVED": 1} and s["accusations"] == 2 and drawn["status"] == "DRAWN"
    assert "accusations_by_kind" not in json.dumps(digests) and "CONTRADICTED" not in json.dumps(digests)
    sec = stranger.check_pool(P2, TABLE, d["meta"], FakeArchive(synthetic_hours()), now=LATE, log=QUIET)
    assert sec["verdict"] == "AGREES" and sec["pool_sha256"] == d["meta"]["pool_sha256"]
    gh, _ = github(routes())
    acc = stranger.check_accuse(P2, sec["_records"], digests, GATE, gh, str(tmp_path / "mine"), False, log=QUIET)
    assert acc["verdict"] == "AGREES" and acc["categories"] == {"MATCH": 4}
    assert acc["accusations_sha256"] == s["accusations_sha256"]
    dr = stranger.check_draw(P2, acc["_summary"], drawn, b["header"], 970505)
    assert dr["verdict"] == "AGREES" and dr["drawn_sha256"] == drawn["drawn_sha256"]
    forged = dict(drawn, drawn=list(reversed(drawn["drawn"])))
    assert stranger.check_draw(P2, acc["_summary"], forged, b["header"], 970505)["verdict"] == "MISMATCH"
    assert stranger._exit_code("AGREES") == 0


def test_stranger_pool_refuses_a_forged_digest_and_names_a_differing_hour(tmp_path):
    d = pool.derive(P2, TABLE, FakeArchive(synthetic_hours()), now=LATE, log=QUIET)
    forged = dict(d["meta"], pool_sha256="f" * 64)
    assert stranger.check_pool(P2, TABLE, forged, FakeArchive(synthetic_hours()), now=LATE,
                               log=QUIET)["verdict"] == "MISMATCH"
    more = synthetic_hours()
    more["2026-10-15-1"].append(ev(150, ref="codex/z", number=9, created="2026-10-15T01:30:00Z"))
    sec = stranger.check_pool(P2, TABLE, d["meta"], FakeArchive(more), now=LATE, log=QUIET)
    assert sec["verdict"] == "MISMATCH" and [h["hour"] for h in sec["hours_differ"]] == ["2026-10-15-1"]


def test_stranger_pool_replays_published_gaps_and_reports_one_that_answers_now():
    hours = synthetic_hours()
    del hours["2026-10-15-1"]
    lab = pool.derive(P2, TABLE, FakeArchive(hours), now=LATE, log=QUIET)
    assert lab["meta"]["gaps"] == ["2026-10-15-1"]
    full = pool.derive(P2, TABLE, FakeArchive(synthetic_hours()), now=LATE, log=QUIET)
    sec = stranger.check_pool(P2, TABLE, lab["meta"], FakeArchive(synthetic_hours(), members=2), now=LATE,
                              log=QUIET)
    assert sec["verdict"] == "AGREES"
    assert sec["gaps_now_available"] == [{"hour": "2026-10-15-1", "admitted": 2,
                                          "admitted_sha256": full["meta"]["hours"][1]["admitted_sha256"],
                                          "pool_sha256_with_it": full["meta"]["pool_sha256"]}]
    assert sec["compressed_bytes_differ"] == ["2026-10-15-0", "2026-10-15-2"]   # rebuilt files, same records


def _stranger_accuse(tmp_path, digests, d, **over):
    gh, _ = github(routes(**over))
    return stranger.check_accuse(P2, d["records"], digests, GATE, gh, str(tmp_path / "mine"), False, log=QUIET)


def test_stranger_accuse_tells_drift_from_a_contradicted_record(tmp_path):
    d, s, digests, _, _ = lab_world(tmp_path)
    later = _stranger_accuse(tmp_path / "a", digests, d,
                             **{API1: api_pr(body="2 files changed.", updated_at="2099-01-01T00:00:00Z")})
    assert later["verdict"] == "CANNOT_CONFIRM" and later["categories"] == {"DRIFT": 1, "MATCH": 3}
    assert stranger._exit_code(later["verdict"]) == 1
    quiet_edit = _stranger_accuse(tmp_path / "b", digests, d, **{API1: api_pr(body="2 files changed.")})
    assert quiet_edit["verdict"] == "MISMATCH" and quiet_edit["categories"]["INCONSISTENT"] == 1
    moved_lie = _stranger_accuse(tmp_path / "c", digests, d,
                                 **{"https://api.github.com/repositories/200/pulls/4": api_pr(number=4),
                                    "https://github.com/acme/app/pull/4.diff": ONE_FILE_DIFF})
    [row] = moved_lie["listed"]
    assert moved_lie["verdict"] == "MISMATCH" and row["lab"] == "HEAD_MOVED" and row["category"] == "INCONSISTENT"
    assert stranger._exit_code(moved_lie["verdict"]) == 2


def test_stranger_accuse_catches_another_gate_and_a_forged_set_digest(tmp_path):
    d, s, digests, _, _ = lab_world(tmp_path)

    class Silent:
        identity = GATE.identity

        def run(self, summary, diff):
            g = GATE.run(summary, diff)
            g.claims = []
            return g
    gh, _ = github(routes())
    sec = stranger.check_accuse(P2, d["records"], digests, Silent(), gh, str(tmp_path / "g"), False, log=QUIET)
    assert sec["verdict"] == "MISMATCH" and {r["why"] for r in sec["listed"]} == {
        "the same title, body and diff gave other claims: not the plan's gate"}
    forged = dict(digests, accusations_sha256="e" * 64)
    sec = _stranger_accuse(tmp_path / "f", forged, d)
    assert sec["verdict"] == "MISMATCH" and sec["checks"]["accusations_sha256"] is False
    pruned = dict(digests, rows=digests["rows"][1:])
    assert _stranger_accuse(tmp_path / "p", pruned, d)["verdict"] == "MISMATCH"


@pytest.mark.parametrize("lab,mine,cat", [
    ({"outcome": "EMPTY_BODY"}, {"outcome": "GATED", "body_sha256": "x", "pr_updated_at": "2099-01-01T00:00:00Z"},
     "DRIFT"),
    ({"outcome": "EMPTY_BODY"}, {"outcome": "GATED", "body_sha256": "x", "pr_updated_at": "2000-01-01T00:00:00Z"},
     "INCONSISTENT"),
    ({"outcome": "PR_GONE", "body_fetched_at": None}, {"outcome": "GATED", "body_sha256": "x"}, "INCONSISTENT"),
    ({"outcome": "GATED"}, {"outcome": "PR_GONE"}, "UNREPRODUCIBLE"),
    ({"outcome": "GATED"}, {"outcome": "HEAD_MOVED"}, "UNREPRODUCIBLE"),
    ({"outcome": "HEAD_MOVED"}, {"outcome": "EMPTY_BODY", "body_sha256": "y",
                                 "pr_updated_at": "2099-01-01T00:00:00Z"}, "DRIFT"),
    ({"outcome": "HEAD_MOVED"}, {"outcome": "HEAD_MOVED", "body_sha256": "z"}, "MATCH"),
    ({"outcome": "GATED", "diff_sha256": "d1"}, {"outcome": "GATED", "diff_sha256": "d2", "body_sha256": "q"},
     "INCONSISTENT"),
])
def test_classify_row(lab, mine, cat):
    base = {"title_sha256": "t", "body_sha256": "b", "diff_sha256": "d", "claims_sha256": "c",
            "body_fetched_at": "2026-10-16T00:00:00Z", "pr_updated_at": None}
    assert stranger.classify_row({**base, **lab}, {**base, **mine})[0] == cat


def test_bundle_writes_the_digests_and_a_manifest(tmp_path):
    d, s, digests, b, drawn = lab_world(tmp_path)
    pool.write_pool(str(tmp_path / "pooldir"), d)
    (tmp_path / "params.json").write_text(json.dumps(P2), encoding="utf-8")
    (tmp_path / "draw.json").write_text(json.dumps(drawn), encoding="utf-8")
    m = stranger.bundle(str(tmp_path / "params.json"), str(tmp_path / "pooldir"), str(tmp_path / "lab"),
                        str(tmp_path / "draw.json"), str(tmp_path / "pub"))
    assert set(m["files"]) == {"plan_params.json", "pool_meta.json", "outcome_digests.json", "draw.json"}
    assert m["params_sha256"] == common.params_sha256(P2)
    for name, sha in m["files"].items():
        assert common.sha256_hex((tmp_path / "pub" / name).read_bytes()) == sha
    assert not (tmp_path / "pub" / "outcomes.jsonl").exists() and not (tmp_path / "pub" / "inputs").exists()


# ================================================================ the words

def test_no_banned_word_in_this_directory_or_this_file():
    banned = ["fir" + "st", "nov" + "el", "revolution" + "ary", "ground" + "breaking", "break" + "through",
              "tamper" + "-proof", "self" + "-verifying", "hallucination" + " detector"]
    import re as _re
    pat = _re.compile(r"\b(" + "|".join(_re.escape(w) for w in banned) + r")", _re.I)
    hits = []
    for p in sorted(SA.iterdir()) + [Path(__file__)]:
        if p.is_file():
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if pat.search(line):
                    hits.append(f"{p.name}:{i}")
    assert hits == []
