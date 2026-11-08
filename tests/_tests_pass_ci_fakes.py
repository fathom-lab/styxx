# -*- coding: utf-8 -*-
"""Shared fakes for the tests-pass-ci collector's tests (tests/test_tests_pass_ci_collector_*.py).

Importing this module puts the collector, papers/tests-pass-ci/collect/, on sys.path, so its modules import by their
own names (tpc_*); TPC_COLLECT_DIR names another copy instead (the mutation check, collect/checks/mutation_check.py).
Nothing here touches the network or the real stores; every pull request the tests read was invented for them.
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLLECTOR = Path(os.environ.get("TPC_COLLECT_DIR") or (ROOT / "papers" / "tests-pass-ci" / "collect")).resolve()
if str(COLLECTOR) not in sys.path:
    sys.path.insert(0, str(COLLECTOR))

from tpc_gh import Counters, Resp  # noqa: E402


class FakeGitHub:
    """Answers GraphQL searches from a function, PR batches from a dict, REST paths from a dict.

    rest: {path_without_paging: [page1_obj, page2_obj, ...]} or {path: (status, obj)}; a path is
    looked up with its per_page/page parameters stripped, and pages are served in order.
    """

    def __init__(self, search=None, prs=None, rest=None, blobs=None):
        self.search_fn = search
        self.prs = prs or {}
        self.rest_map = rest or {}
        self.blobs = blobs or {}
        self.c = Counters()
        self.calls = []

    def graphql(self, query, variables=None, *, kind="graphql", what=""):
        self.c.calls[kind] += 1
        self.calls.append((kind, what or (variables or {}).get("q")))
        if kind == "search":
            return {"data": {"search": self.search_fn(variables["q"], variables.get("after"), query)}}
        data = {}
        import re
        for alias, owner, name, number in re.findall(
                r'(p\d+): repository\(owner: "([^"]+)", name: "([^"]+)"\) \{ pullRequest\(number: (\d+)\)', query):
            key = f"{owner}/{name}#{number}"
            node = self.prs.get(key)
            data[alias] = None if node is None else {"pullRequest": node}
        errors = [{"path": [a], "message": "Could not resolve"} for a, v in data.items() if v is None]
        out = {"data": data}
        if errors:
            out["errors"] = errors
        return out

    @staticmethod
    def _strip(path):
        base, _, q = path.partition("?")
        keep = [p for p in q.split("&") if p and not p.startswith(("per_page=", "page="))]
        return base + ("?" + "&".join(keep) if keep else "")

    def rest(self, path, *, accept=None):
        self.c.calls["rest"] += 1
        self.calls.append(("rest", path))
        key = self._strip(path)
        page = 1
        for p in path.split("?", 1)[-1].split("&"):
            if p.startswith("page="):
                page = int(p[5:])
        v = self.rest_map.get(key)
        if v is None:
            return Resp(404, {"message": "Not Found"}, {}, b'{"message":"Not Found"}')
        if isinstance(v, tuple):
            status, obj = v
            hdrs = {"Location": obj} if status in (301, 302, 303, 307, 308) else {}
            raw = b"" if status in (301, 302, 303, 307, 308) else json.dumps(obj).encode()
            return Resp(status, None if status in (301, 302) else obj, hdrs, raw)
        obj = v[page - 1] if page - 1 < len(v) else ([] if isinstance(v[0], list) else {k: [] for k in v[0]})
        return Resp(200, obj, {}, json.dumps(obj).encode())

    def rest_paged(self, path, key, *, per_page=100, max_pages=1000):
        items, pages = [], []
        sep = "&" if "?" in path else "?"
        for page in range(1, max_pages + 1):
            r = self.rest(f"{path}{sep}per_page={per_page}&page={page}")
            pages.append(r)
            if r.status != 200 or r.data is None:
                return items, pages, False
            batch = r.data if key is None else (r.data.get(key) or [])
            items += batch
            if len(batch) < per_page:
                return items, pages, True
        return items, pages, False

    def blob(self, url, what, *, max_bytes=None):
        self.c.calls["blob"] += 1
        raw = self.blobs.get(url, b"")
        trunc = max_bytes is not None and len(raw) > max_bytes
        return (200 if raw else 404), (raw[:max_bytes] if trunc else raw), trunc

    def follow(self, path, what, *, max_bytes=None):
        r = self.rest(path)
        if r.status in (301, 302, 303, 307, 308) and r.location:
            st, raw, trunc = self.blob(r.location, what, max_bytes=max_bytes)
            return r.status, st, raw, trunc
        return r.status, None, b"", False


class OpenGuard:
    open = True
    state = {"open": True, "why": "test"}

    def check(self, created_at, what):
        return None


def gunzip(p: Path) -> bytes:
    return gzip.decompress(p.read_bytes())
