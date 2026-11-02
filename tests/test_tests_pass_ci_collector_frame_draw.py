# -*- coding: utf-8 -*-
"""The frame (P2) and the draw (P3) against the PREREG's formulas, offline.

The tests-pass-ci collector (papers/tests-pass-ci/collect/), offline, against fakes (tests/_tests_pass_ci_fakes.py).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import random
from fractions import Fraction

import pytest
from tests._tests_pass_ci_fakes import FakeGitHub

import tpc_draw
import tpc_frame
from tpc_const import AGENT_ORDER, FRAME_FIELDS, P0, P_AGENT, UTC, parse_utc

D = dt.date(2026, 11, 2)


def test_o_day_is_the_prereg_formula():
    for d in [D + dt.timedelta(days=i) for i in range(28)]:
        want = int(hashlib.sha256(("tests-pass-ci/slice/" + d.isoformat()).encode("utf-8")).hexdigest(), 16) % 6
        assert tpc_frame.o_day(d) == want


def test_cc_slices_48_a_day_every_30_minutes():
    for d in [D, dt.date(2026, 10, 29), dt.date(2026, 11, 29)]:
        sl = tpc_frame.cc_slices(d)
        assert len(sl) == 48
        o = tpc_frame.o_day(d)
        assert all((k + o) % 6 == 0 for k, _, _ in sl)
        starts = [a for _, a, _ in sl]
        assert all((b - a) == dt.timedelta(minutes=30) for a, b in zip(starts, starts[1:]))
        assert all((e - a) == dt.timedelta(minutes=5) for _, a, e in sl)
        assert sl[0][1].date() == d and sl[-1][2] <= dt.datetime(d.year, d.month, d.day, tzinfo=UTC) + dt.timedelta(days=1)


def test_whole_ranges_four_hours():
    rs = tpc_frame.whole_ranges(D)
    assert len(rs) == 6
    assert rs[0][1] == dt.datetime(2026, 11, 2, tzinfo=UTC)
    assert rs[-1][2] == dt.datetime(2026, 11, 3, tzinfo=UTC)


def test_range_query_is_inclusive_and_verbatim():
    a = dt.datetime(2026, 11, 2, 4, tzinfo=UTC)
    q = tpc_frame.range_q("Cursor", a, a + dt.timedelta(hours=4))
    assert q == 'is:pr "cursor.com/background-agent" in:body created:2026-11-02T04:00:00Z..2026-11-02T07:59:59Z'
    assert tpc_frame.range_q("OpenAI_Codex", a, a + dt.timedelta(minutes=5)).startswith("is:pr label:codex created:")


def test_frame_query_requests_only_p22_fields():
    q = tpc_frame.search_query(FRAME_FIELDS)
    assert "body" not in q and "title" not in q and "label" not in q.replace("label:codex", "")
    for f in ("databaseId", "number", "createdAt", "__typename", "login", "nameWithOwner", "stargazerCount", "isFork"):
        assert f in q


def test_query_bytes_are_unchanged_by_storing_the_paging_argument_reversed():
    """The reviser stored GitHub's paging argument reversed (the word lint); the bytes sent are the draft's."""
    q = tpc_frame.search_query(FRAME_FIELDS).encode("utf-8")
    assert hashlib.sha256(q).hexdigest() == "bd8226c71df71fdcaf371544f4de1ed65e18d341e6b10b632bafcbcea783f09f"
    c = tpc_frame.COUNT_QUERY.encode("utf-8")
    assert hashlib.sha256(c).hexdigest() == "4648980f4d1eb79fc7e911d9f2004a475f0bff0be6061f2fbee137e4890db735"


def _nodes_between(a, b, n, start_id):
    span = (b - a).total_seconds()
    out = []
    for i in range(n):
        t = a + dt.timedelta(seconds=int(span * i / n))
        out.append({"databaseId": start_id + i, "number": i + 1, "createdAt": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "author": {"__typename": "User", "login": "x"},
                    "repository": {"nameWithOwner": "o/r", "stargazerCount": 150, "isFork": False}})
    return out


def test_enumerate_range_halves_above_1000_and_pages_to_the_end():
    a = dt.datetime(2026, 11, 2, 0, tzinfo=UTC)
    b = a + dt.timedelta(hours=4)
    universe = _nodes_between(a, b, 2500, 1)

    def search(q, after, query):
        rng = q.split("created:", 1)[1]
        lo, hi = (parse_utc(x) for x in rng.split(".."))
        hits = [n for n in universe if lo <= parse_utc(n["createdAt"]) <= hi]
        start = int(after or 0)
        page = hits[start:start + 100] if len(hits) <= 1000 else hits[:100]
        nxt = start + 100 < min(len(hits), 1000)
        return {"issueCount": len(hits), "pageInfo": {"hasNextPage": nxt, "endCursor": str(start + 100)}, "nodes": page}

    gh = FakeGitHub(search=search)
    queries = []
    nodes = tpc_frame.enumerate_range(gh, "Cursor", a, b, FRAME_FIELDS, queries)
    assert len(nodes) == 2500
    assert len({n["databaseId"] for n in nodes}) == 2500
    assert any(q["split"] for q in queries)
    assert all(q["issueCount"] <= 1000 for q in queries if not q["split"])
    row = tpc_frame.frame_row(nodes[0], "Cursor")
    assert tuple(row) == tpc_frame.FRAME_KEYS


def test_kept_and_dedupe_priority():
    base = {"pr_id": 1, "repo": "o/r", "number": 1, "stars": 100, "fork": False, "author": "a",
            "author_type": "User", "created_at": "2026-11-02T00:00:00Z", "slice": "s"}
    assert tpc_frame.kept(dict(base, agent="Copilot"))
    assert not tpc_frame.kept(dict(base, stars=99, agent="Copilot"))
    assert not tpc_frame.kept(dict(base, fork=True, agent="Copilot"))
    assert tpc_frame.kept(dict(base, repo=None, stars=None, fork=None, agent="Copilot"))  # kept, flagged later
    rows = [dict(base, agent="Cursor"), dict(base, agent="Claude_Code"), dict(base, agent="Copilot"),
            dict(base, pr_id=2, agent="OpenAI_Codex"), dict(base, pr_id=2, agent="Devin")]
    uniq, doubles = tpc_frame.dedupe(rows, key=lambda r: r["pr_id"])
    by = {r["pr_id"]: r["agent"] for r in uniq}
    assert by == {1: "Copilot", 2: "Devin"}
    assert doubles == {1: ["Copilot", "Cursor", "Claude_Code"], 2: ["Devin", "OpenAI_Codex"]}
    assert AGENT_ORDER == ("Copilot", "Claude_Code", "Cursor", "Devin", "Google_Jules", "OpenAI_Codex")


def test_a_double_match_with_claude_code_goes_to_the_other_agent():
    """P2.1 as revised: Claude Code is read last, so a pull request also matched by a whole-enumerated signature
    is drawn at that agent's probabilities and sits in M with probability 0.020, not 1/6 x 0.120 + 5/6 x 0.020."""
    base = {"pr_id": 7, "repo": "o/r", "number": 7, "stars": 100, "fork": False, "author": "a",
            "author_type": "User", "created_at": "2026-11-02T00:00:00Z", "slice": "s"}
    for other in ("Cursor", "Devin", "Google_Jules", "OpenAI_Codex", "Copilot"):
        uniq, doubles = tpc_frame.dedupe([dict(base, agent="Claude_Code"), dict(base, agent=other)],
                                         key=lambda r: r["pr_id"])
        assert [r["agent"] for r in uniq] == [other]
        assert doubles == {7: [other, "Claude_Code"]}


def test_draw_value_is_the_prereg_formula():
    beacon = "00" * 32
    for pr in (1, 5017028180, 123456789012):
        h = int(hashlib.sha256((beacon + ":tests-pass-ci:" + str(pr)).encode("utf-8")).hexdigest(), 16)
        assert tpc_draw.draw_int(beacon, pr) == h
        m = tpc_draw.membership(beacon, pr, "Cursor")
        assert m["u"] == h / 2 ** 256
    with pytest.raises(ValueError):
        tpc_draw.draw_int("XYZ", 1)


def test_below_is_exact_at_the_boundary():
    p = Fraction("0.020")
    edge = (p.numerator * 2 ** 256) // p.denominator  # floor(0.02 * 2**256)
    exact = (p.numerator * 2 ** 256) % p.denominator == 0
    assert tpc_draw.below(edge - 1, p)
    assert tpc_draw.below(edge, p) is (not exact)
    assert not tpc_draw.below(edge + 1, p)


def test_sets_follow_section_3_3():
    beacon = hashlib.sha256(b"test beacon").hexdigest()
    rnd = random.Random(7)
    ids = rnd.sample(range(10 ** 9, 10 ** 10), 60000)
    for agent in AGENT_ORDER:
        n_d = n_m = 0
        for pr in ids:
            m = tpc_draw.membership(beacon, pr, agent)
            u = m["u"]
            assert m["in_D"] == (u < float(P_AGENT[agent]))
            if agent == "Claude_Code":
                assert m["in_M"] == m["in_D"]  # every drawn Claude Code row (inside its slices)
            else:
                assert m["in_M"] == (u < float(P0))
            if m["in_M"]:
                assert 0.0 <= m["v"] < 1.0
            n_d += m["in_D"]
            n_m += m["in_M"]
        # M is 0.020 of every non-Claude agent's kept rows, 0.120 of Claude Code's slice rows
        want_m = float(P_AGENT["Claude_Code"]) if agent == "Claude_Code" else float(P0)
        assert abs(n_m / len(ids) - want_m) < 5 * (want_m * (1 - want_m) / len(ids)) ** 0.5
        assert abs(n_d / len(ids) - float(P_AGENT[agent])) < 0.01


def test_claude_code_overall_main_probability_is_0_020():
    # inside a slice with probability 1/6, then u < 0.120: 0.020 overall (section 3.3)
    assert Fraction(1, 6) * P_AGENT["Claude_Code"] == P0
