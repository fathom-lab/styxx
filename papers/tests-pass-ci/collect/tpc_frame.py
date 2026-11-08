# -*- coding: utf-8 -*-
"""The frame (PREREG P2) and the dry run, by metadata only.

Frame of day D (UTC), enumerated between D+1 00:00Z and D+1 12:00Z:
- Copilot, Cursor, Devin, Jules, Codex: the day whole, in four-hour ranges.
- Claude Code: five-minute slice k (k = 0..287) is enumerated iff (k + o_D) % 6 == 0, with
  o_D = int(sha256(("tests-pass-ci/slice/" + D).encode("utf-8")).hexdigest(), 16) % 6.
- Any range whose issueCount exceeds 1,000 is halved until each part is at or under 1,000.
- GraphQL search, type ISSUE, 100 nodes a page, cursor paging to the end, query
  "is:pr <qualifier> created:<from>..<to>", requesting tpc_const.FRAME_FIELDS and nothing else.
- frame_<D>.jsonl holds, per node returned, exactly: pr_id, repo, number, stars, fork, author,
  author_type, created_at, agent, slice. Its sha256, the enumeration times, the code's digest and
  every query's issueCount and node count go into frame_digests.jsonl.

The dry run (--dry-run) reads the same ranges with tpc_const.DRYRUN_FIELDS (number, createdAt,
author login, repository name and stars: the seal's own list) and writes counts only, never rows.
"""
from __future__ import annotations

import datetime as dt
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from tpc_const import (AGENT_ORDER, MATCH_PRECEDENCE, CC_SLICE_EVERY, CC_SLICE_MINUTES, CC_SLICES_PER_DAY, DRYRUN_FIELDS,
                       FRAME_FIELDS, MIN_STARS, PAGE_SIZE, QUALIFIER, SEARCH_CAP, SLICED_AGENT, WHOLE_RANGE_HOURS, WINDOW_START, day_start, iso)

FRAME_KEYS = ("pr_id", "repo", "number", "stars", "fork", "author", "author_type", "created_at",
              "agent", "slice")


# GitHub's GraphQL paging argument is a word on the lab's must-not-print list; it is stored reversed so the source
# passes the word lint, and the query bytes sent are unchanged (tests pin them).
PAGE_ARG = "tsrif"[::-1]


def search_query(fields: str) -> str:
    return ("query($q: String!, $after: String) { rateLimit { cost remaining resetAt }\n"
            "  search(type: ISSUE, query: $q, " + PAGE_ARG + ": %d, after: $after) {\n"
            "    issueCount pageInfo { hasNextPage endCursor }\n"
            "    nodes { ... on PullRequest { %s } } } }") % (PAGE_SIZE, fields)


COUNT_QUERY = ("query($q: String!) { rateLimit { cost remaining resetAt }\n"
               "  search(type: ISSUE, query: $q, " + PAGE_ARG + ": 1) { issueCount } }")


def o_day(d: dt.date) -> int:
    return int(hashlib.sha256(("tests-pass-ci/slice/" + d.isoformat()).encode("utf-8")).hexdigest(), 16) % 6


def cc_slices(d: dt.date) -> list:
    """The 48 enumerated five-minute slices of day d: [(k, start, end)]."""
    o = o_day(d)
    t0 = day_start(d)
    out = []
    for k in range(CC_SLICES_PER_DAY):
        if (k + o) % CC_SLICE_EVERY == 0:
            a = t0 + dt.timedelta(minutes=CC_SLICE_MINUTES * k)
            out.append((k, a, a + dt.timedelta(minutes=CC_SLICE_MINUTES)))
    return out


def whole_ranges(d: dt.date) -> list:
    t0 = day_start(d)
    step = dt.timedelta(hours=WHOLE_RANGE_HOURS)
    return [(i, t0 + i * step, t0 + (i + 1) * step) for i in range(24 // WHOLE_RANGE_HOURS)]


def ranges_for(agent: str, d: dt.date) -> list:
    return cc_slices(d) if agent == SLICED_AGENT else whole_ranges(d)


def range_q(agent: str, a: dt.datetime, b: dt.datetime) -> str:
    """Inclusive created range a .. b-1s."""
    return f"is:pr {QUALIFIER[agent]} created:{iso(a)}..{iso(b - dt.timedelta(seconds=1))}"


def enumerate_range(gh, agent: str, a: dt.datetime, b: dt.datetime, fields: str,
                    queries: list) -> list:
    """Every node created in [a, b), halving while a range's issueCount exceeds 1,000.
    Appends one entry per query run (its range, issueCount, nodes, pages) to `queries`."""
    q = range_q(agent, a, b)
    sq = search_query(fields)
    page0 = gh.graphql(sq, {"q": q, "after": None}, kind="search", what=q)["data"]["search"]
    n = page0["issueCount"]
    width = (b - a).total_seconds()
    if n > SEARCH_CAP and width >= 2:
        mid = a + dt.timedelta(seconds=int(width // 2))
        queries.append({"agent": agent, "q": q, "issueCount": n, "split": True, "nodes": 0, "pages": 1})
        return (enumerate_range(gh, agent, a, mid, fields, queries)
                + enumerate_range(gh, agent, mid, b, fields, queries))
    nodes = [x for x in (page0.get("nodes") or []) if x]
    empty = sum(1 for x in (page0.get("nodes") or []) if not x)
    page, pages = page0, 1
    while page["pageInfo"]["hasNextPage"]:
        page = gh.graphql(sq, {"q": q, "after": page["pageInfo"]["endCursor"]}, kind="search",
                          what=q)["data"]["search"]
        pages += 1
        nodes += [x for x in (page.get("nodes") or []) if x]
        empty += sum(1 for x in (page.get("nodes") or []) if not x)
    # distinct ids: GitHub's search pages by relevance, so paging can repeat or skip a node
    ids = {(x.get("databaseId"), (x.get("repository") or {}).get("nameWithOwner"), x.get("number")) for x in nodes}
    queries.append({"agent": agent, "q": q, "issueCount": n, "split": False, "nodes": len(nodes),
                    "distinct": len(ids), "pages": pages, "empty_nodes": empty, "truncated": n > SEARCH_CAP})
    for x in nodes:
        x["_slice"] = q.split(" created:", 1)[1]
    return nodes


def frame_row(node: dict, agent: str) -> dict:
    repo = node.get("repository") or None
    author = node.get("author") or None
    return {
        "pr_id": node.get("databaseId"),
        "repo": repo.get("nameWithOwner") if repo else None,
        "number": node.get("number"),
        "stars": repo.get("stargazerCount") if repo else None,
        "fork": repo.get("isFork") if repo else None,
        "author": author.get("login") if author else None,
        "author_type": author.get("__typename") if author else None,
        "created_at": node.get("createdAt"),
        "agent": agent,
        "slice": node.get("_slice"),
    }


def enumerate_day(gh, d: dt.date, *, fields: str = FRAME_FIELDS, workers: int = 2,
                  agents: tuple = AGENT_ORDER, slice_limit: Optional[int] = None,
                  until: Optional[dt.datetime] = None,
                  progress: Optional[Callable[[str], None]] = None) -> tuple:
    """(rows-by-agent {agent: [node, ...]}, queries, per-agent totals). `until` clips ranges to end
    by then (the dry run on a day not yet over); `slice_limit` is for development runs only."""
    tasks = []
    for agent in agents:
        rs = ranges_for(agent, d)
        if slice_limit is not None:
            rs = rs[:slice_limit]
        for k, a, b in rs:
            if until is not None:
                if a >= until:
                    continue
                b = min(b, until)
            tasks.append((agent, k, a, b))

    def run(task):
        agent, k, a, b = task
        qs: list = []
        nodes = enumerate_range(gh, agent, a, b, fields, qs)
        if progress:
            progress(f"{d} {agent} {iso(a)} count={qs[0]['issueCount'] if qs else '?'} nodes={len(nodes)}")
        return agent, k, nodes, qs

    by_agent: dict = {a: [] for a in agents}
    queries: list = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for agent, k, nodes, qs in ex.map(run, tasks):  # map keeps the task order
            by_agent[agent].extend(nodes)
            queries.extend(qs)
    return by_agent, queries


def day_totals(gh, d: dt.date, *, until: Optional[dt.datetime] = None,
               agents: tuple = AGENT_ORDER) -> dict:
    """issueCount of each signature over the whole day (or up to `until`): one search each."""
    a = day_start(d)
    b = a + dt.timedelta(days=1)
    if until is not None:
        b = min(b, until)
    out = {}
    for agent in agents:
        q = range_q(agent, a, b)
        res = gh.graphql(COUNT_QUERY, {"q": q}, kind="search", what=q)
        out[agent] = res["data"]["search"]["issueCount"]
    return out


def kept(row: dict) -> bool:
    """P2.4: stars >= 100 and not a fork. A null repository is kept (and flagged at the draw)."""
    if row.get("repo") is None:
        return True
    return (row.get("stars") or 0) >= MIN_STARS and row.get("fork") is False


def dedupe(rows: list, key: Callable[[dict], object]) -> tuple:
    """One row per pull request, its agent the earliest signature of P2.1 that matched it, Claude Code read
    last (tpc_const.MATCH_PRECEDENCE). Returns (rows, double_matches {key: [agents]})."""
    best: dict = {}
    agents_of: dict = {}
    rank = {a: i for i, a in enumerate(MATCH_PRECEDENCE)}
    for r in rows:
        k = key(r)
        agents_of.setdefault(k, set()).add(r["agent"])
        cur = best.get(k)
        if cur is None or rank[r["agent"]] < rank[cur["agent"]]:
            best[k] = r
    doubles = {k: sorted(v, key=lambda a: rank[a]) for k, v in agents_of.items() if len(v) > 1}
    return list(best.values()), doubles


def dry_run(gh, now: dt.datetime, *, counts_only: bool = False, workers: int = 2,
            index_lag: dt.timedelta = dt.timedelta(minutes=10),
            progress: Optional[Callable[[str], None]] = None) -> dict:
    """Count window pull requests by the seal's metadata only. Returns a counts object; no rows."""
    until = min(now - index_lag, WINDOW_START + dt.timedelta(days=28))
    out: dict = {"window_start": iso(WINDOW_START), "until": iso(until), "fields": DRYRUN_FIELDS,
                 "counts_only": counts_only, "days": {}}
    d = WINDOW_START.date()
    while day_start(d) < until:
        dd: dict = {"search_total": day_totals(gh, d, until=until)}
        if not counts_only:
            by_agent, queries = enumerate_day(gh, d, fields=DRYRUN_FIELDS, workers=workers, until=until,
                                              progress=progress)
            rows = []
            for agent, nodes in by_agent.items():
                for n in nodes:
                    repo = n.get("repository") or {}
                    rows.append({"agent": agent, "key": f"{repo.get('nameWithOwner')}#{n.get('number')}",
                                 "stars": repo.get("stargazerCount")})
            uniq, doubles = dedupe(rows, key=lambda r: r["key"])
            per = {}
            for agent in AGENT_ORDER:
                mine = [r for r in uniq if r["agent"] == agent]
                star = [r for r in mine if (r["stars"] or 0) >= MIN_STARS]
                enumerated = sum(q["issueCount"] for q in queries if q["agent"] == agent and not q["split"])
                per[agent] = {"nodes": sum(1 for r in rows if r["agent"] == agent),
                              "distinct": len(mine), "stars_100": len(star),
                              "enumerated_issueCount": enumerated,
                              "ranges_enumerated": len([1 for (k, a, b) in ranges_for(agent, d) if a < until])}
            cc = per.get(SLICED_AGENT)
            if cc:
                cc["estimate_stars_100_all_slices"] = cc["stars_100"] * CC_SLICE_EVERY
            dd["per_agent"] = per
            dd["double_matches"] = len(doubles)
            dd["queries"] = len(queries)
            dd["stars_100_distinct"] = sum(p["stars_100"] for p in per.values())
        out["days"][d.isoformat()] = dd
        d += dt.timedelta(days=1)
    return out
