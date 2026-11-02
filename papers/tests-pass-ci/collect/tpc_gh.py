# -*- coding: utf-8 -*-
"""A read-only GitHub client that logs every call and waits out GitHub's limits.

- The token is `gh auth token`, run with GH_TOKEN and GITHUB_TOKEN unset (the machine's own login,
  OD-5). It is never printed, logged or written, and it is sent to api.github.com only: signed
  redirect targets (job logs, artifact zips) are fetched without it.
- Every call is one line in the read log (P21): time, kind, endpoint, query or path, status, bytes,
  seconds, and the remaining quota GitHub reported.
- Limits: REST and GraphQL report what remains; under a reserve the client sleeps to the reset. A
  403 or 429 with Retry-After, or with nothing left, sleeps as told; a secondary limit without a
  header waits a minute and more. Searches are paced (at most about 27 a minute). A wait longer
  than `max_wait` raises `BudgetStop`, so a catch-up run ends cleanly and the next run continues.
- GET and the GraphQL POST only. Nothing here writes to GitHub.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from tpc_const import UTC, USER_AGENT, iso, parse_utc

API = "https://api.github.com"
API_HOST = "api.github.com"


class BudgetStop(Exception):
    """The run must stop: a limit would need a wait longer than allowed, or the request budget is spent."""


class GitHubError(RuntimeError):
    pass


@dataclass
class Resp:
    status: int
    data: Any
    headers: dict
    raw: bytes
    secs: float = 0.0

    @property
    def location(self) -> Optional[str]:
        for k, v in self.headers.items():
            if k.lower() == "location":
                return v
        return None


def gh_token() -> str:
    env = dict(os.environ)
    env.pop("GH_TOKEN", None)
    env.pop("GITHUB_TOKEN", None)
    out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True, env=env)
    tok = out.stdout.strip()
    if not tok:
        raise GitHubError("gh auth token returned nothing")
    return tok


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


def urllib_transport(method: str, url: str, headers: dict, body: Optional[bytes], timeout: float):
    """(status, headers, bytes). Redirects come back as responses, never followed."""
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with opener.open(req, timeout=timeout) as r:
            return r.status, dict(r.headers.items()), r.read()
    except urllib.error.HTTPError as e:
        hdrs = dict(e.headers.items()) if e.headers else {}
        try:
            raw = e.read()
        except Exception:  # noqa: BLE001
            raw = b""
        return e.code, hdrs, raw


def _hget(headers: dict, name: str) -> Optional[str]:
    for k, v in headers.items():
        if k.lower() == name.lower():
            return v
    return None


@dataclass
class Counters:
    calls: dict = field(default_factory=lambda: {"rest": 0, "graphql": 0, "search": 0, "blob": 0})
    bytes: dict = field(default_factory=lambda: {"rest": 0, "graphql": 0, "search": 0, "blob": 0})
    waited_secs: float = 0.0


class GitHub:
    def __init__(self, log: Callable[[dict], None], *, token: Optional[str] = None,
                 transport: Callable = urllib_transport, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.time, search_pace: float = 2.2,
                 core_reserve: int = 150, graphql_reserve: int = 150, max_wait: float = 3700.0,
                 max_calls: Optional[int] = None, timeout: float = 90.0) -> None:
        self._token = token if token is not None else gh_token()
        self._log = log
        self._transport = transport
        self._sleep = sleep
        self._clock = clock
        self.search_pace = search_pace
        self.core_reserve = core_reserve
        self.graphql_reserve = graphql_reserve
        self.max_wait = max_wait
        self.max_calls = max_calls
        self.timeout = timeout
        self.c = Counters()
        self._search_lock = threading.Lock()
        self._last_search = 0.0
        self._cooldown_until = 0.0
        self._quota_lock = threading.Lock()
        self.quota: dict = {}  # resource -> (remaining, reset epoch)

    # ---- plumbing ------------------------------------------------------------------------------
    def _headers(self, url: str, accept: str = "application/vnd.github+json") -> dict:
        h = {"Accept": accept, "User-Agent": USER_AGENT}
        if urllib.parse.urlsplit(url).hostname == API_HOST:
            h["Authorization"] = f"Bearer {self._token}"
            h["X-GitHub-Api-Version"] = "2022-11-28"
        return h

    def _write_log(self, kind: str, what: str, status: Any, nbytes: int, secs: float, **extra: Any) -> None:
        row = {"at": iso(dt.datetime.now(UTC)), "kind": kind, "what": what, "status": status,
               "bytes": nbytes, "secs": round(secs, 3)}
        row.update({k: v for k, v in extra.items() if v is not None})
        self._log(row)

    def _count(self, kind: str) -> None:
        total = sum(self.c.calls.values())
        if self.max_calls is not None and total >= self.max_calls:
            raise BudgetStop(f"request budget of {self.max_calls} calls spent")
        self.c.calls[kind] += 1

    def _wait(self, secs: float, why: str) -> None:
        if secs <= 0:
            return
        if secs > self.max_wait:
            raise BudgetStop(f"{why}: would wait {secs:.0f}s, more than {self.max_wait:.0f}s")
        self._write_log("wait", why, "sleep", 0, secs)
        self.c.waited_secs += secs
        self._sleep(secs)

    def _note_quota(self, resource: str, remaining: Optional[int], reset: Optional[float]) -> None:
        if remaining is None:
            return
        with self._quota_lock:
            self.quota[resource] = (remaining, reset)

    def _respect_quota(self, resource: str, reserve: int) -> None:
        with self._quota_lock:
            q = self.quota.get(resource)
        if not q:
            return
        remaining, reset = q
        if remaining is not None and remaining <= reserve and reset:
            self._wait(reset - self._clock() + 2, f"{resource} quota at {remaining}")
            with self._quota_lock:
                self.quota.pop(resource, None)

    def _pace_search(self) -> None:
        """Searches start at least `search_pace` seconds apart, across threads, and none starts
        while a secondary-limit cooldown set by any thread is running."""
        with self._search_lock:
            cool = self._cooldown_until - self._clock()
            if cool > 0:
                self._wait(cool, "search cooldown")
            wait = self._last_search + self.search_pace - self._clock()
            if wait > 0:
                self._sleep(wait)
            self._last_search = self._clock()

    def _search_backoff(self, hdrs: dict, attempt: int) -> None:
        """A search met a 403 or 429: every search thread cools down, and the pace slows (to 4 s)."""
        retry = _hget(hdrs, "Retry-After")
        secs = float(retry) + 1 if retry else 60.0 * (attempt + 1)
        with self._search_lock:
            self._cooldown_until = max(self._cooldown_until, self._clock() + secs)
            self.search_pace = min(4.0, self.search_pace + 0.5)

    # ---- GraphQL -------------------------------------------------------------------------------
    def graphql(self, query: str, variables: Optional[dict] = None, *, kind: str = "graphql",
                what: str = "") -> dict:
        """POST /graphql. kind 'search' is paced. Returns the payload; raises on a payload with
        errors and no data. Partial data with errors is returned, errors included."""
        body = json.dumps({"query": query, "variables": variables or {}}).encode("utf-8")
        label = what or (variables or {}).get("q") or "graphql"
        for attempt in range(8):
            self._respect_quota("graphql", self.graphql_reserve)
            if kind == "search":
                self._pace_search()
            self._count(kind)
            t0 = self._clock()
            try:
                status, hdrs, raw = self._transport("POST", API + "/graphql", self._headers(API + "/graphql"),
                                                    body, self.timeout)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                self._write_log(kind, label, f"neterr:{type(e).__name__}", 0, self._clock() - t0)
                self._sleep(5 * (attempt + 1))
                continue
            secs = self._clock() - t0
            self.c.bytes[kind] += len(raw)
            payload = None
            try:
                payload = json.loads(raw) if raw else None
            except ValueError:
                payload = None
            rl = ((payload or {}).get("data") or {}).get("rateLimit") or {} if isinstance(payload, dict) else {}
            self._write_log(kind, label, status, len(raw), secs, cost=rl.get("cost"),
                            remaining=rl.get("remaining"))
            if status == 200 and isinstance(payload, dict):
                if rl.get("remaining") is not None and rl.get("resetAt"):
                    self._note_quota("graphql", int(rl["remaining"]), parse_utc(rl["resetAt"]).timestamp())
                errs = payload.get("errors") or []
                if errs and any((e or {}).get("type") == "RATE_LIMITED" for e in errs):
                    self._wait(60 * (attempt + 1), "graphql RATE_LIMITED")
                    continue
                if errs and not payload.get("data"):
                    raise GitHubError("graphql errors: " + json.dumps(errs)[:400])
                return payload
            if status in (403, 429):
                if kind == "search":
                    self._search_backoff(hdrs, attempt)  # the next _pace_search waits it out
                else:
                    self._limit_wait(hdrs, attempt, "graphql")
                continue
            if status >= 500 or status == 0:
                self._sleep(min(120, 5 * 2 ** attempt))
                continue
            raise GitHubError(f"graphql HTTP {status}: {raw[:200]!r}")
        raise GitHubError(f"graphql failed after retries: {label}")

    def _limit_wait(self, hdrs: dict, attempt: int, what: str) -> None:
        retry = _hget(hdrs, "Retry-After")
        remaining = _hget(hdrs, "X-RateLimit-Remaining")
        reset = _hget(hdrs, "X-RateLimit-Reset")
        if retry:
            self._wait(float(retry) + 1, f"{what} Retry-After")
        elif remaining == "0" and reset:
            self._wait(float(reset) - self._clock() + 2, f"{what} limit reached")
        else:  # a secondary limit without headers: wait at least a minute, longer each time
            self._wait(60.0 * (attempt + 1), f"{what} secondary limit")

    # ---- REST ----------------------------------------------------------------------------------
    def rest(self, path: str, *, accept: str = "application/vnd.github+json") -> Resp:
        """GET a REST path (or a full api.github.com URL). Redirects come back unfollowed (302 with
        Location). 404/410/422 come back as responses; limits are waited out; 5xx retried."""
        url = path if path.startswith("http") else API + path
        if urllib.parse.urlsplit(url).hostname != API_HOST:
            raise ValueError("rest() is for api.github.com only; use blob() for signed URLs")
        for attempt in range(7):
            self._respect_quota("core", self.core_reserve)
            self._count("rest")
            t0 = self._clock()
            try:
                status, hdrs, raw = self._transport("GET", url, self._headers(url, accept), None, self.timeout)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                self._write_log("rest", path, f"neterr:{type(e).__name__}", 0, self._clock() - t0)
                self._sleep(min(120, 2 ** attempt * 2))
                continue
            secs = self._clock() - t0
            self.c.bytes["rest"] += len(raw)
            rem = _hget(hdrs, "X-RateLimit-Remaining")
            reset = _hget(hdrs, "X-RateLimit-Reset")
            resource = _hget(hdrs, "X-RateLimit-Resource") or "core"
            self._write_log("rest", path, status, len(raw), secs, remaining=rem, resource=resource)
            if rem is not None:
                try:
                    self._note_quota(resource, int(rem), float(reset) if reset else None)
                except ValueError:
                    pass
            if status in (403, 429) and (_hget(hdrs, "Retry-After") or rem == "0"
                                         or b"secondary rate limit" in raw.lower()
                                         or b"rate limit" in raw.lower()):
                self._limit_wait(hdrs, attempt, "rest")
                continue
            if status >= 500:
                self._sleep(min(120, 2 ** attempt * 2))
                continue
            data = None
            if raw:
                try:
                    data = json.loads(raw)
                except ValueError:
                    data = None
            return Resp(status, data, hdrs, raw, secs)
        raise GitHubError(f"rest failed after retries: {path}")

    def rest_paged(self, path: str, key: Optional[str], *, per_page: int = 100,
                   max_pages: int = 1000) -> tuple:
        """Every page of a listing (to its end). Returns (items, [Resp per page], complete).
        `key` is the list's field in the object (None when the response is a bare list)."""
        items: list = []
        pages: list = []
        sep = "&" if "?" in path else "?"
        total = None
        for page in range(1, max_pages + 1):
            r = self.rest(f"{path}{sep}per_page={per_page}&page={page}")
            pages.append(r)
            if r.status != 200 or r.data is None:
                return items, pages, False
            batch = r.data if key is None else (r.data.get(key) or [])
            if key is not None and isinstance(r.data, dict) and r.data.get("total_count") is not None:
                total = r.data["total_count"]
            items += batch
            if len(batch) < per_page:
                return items, pages, True
            if total is not None and len(items) >= total:
                return items, pages, True
        return items, pages, False

    # ---- signed blobs (logs, artifact zips) --------------------------------------------------
    def blob(self, url: str, what: str, *, max_bytes: Optional[int] = None) -> tuple:
        """GET a signed URL without the GitHub token. (status, bytes, truncated)."""
        if urllib.parse.urlsplit(url).hostname == API_HOST:
            raise ValueError("blob() is for signed redirect targets, not the API")
        for attempt in range(4):
            self._count("blob")
            t0 = self._clock()
            try:
                status, hdrs, raw = self._transport("GET", url, self._headers(url, "*/*"), None, 300.0)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                self._write_log("blob", what, f"neterr:{type(e).__name__}", 0, self._clock() - t0)
                self._sleep(min(60, 2 ** attempt * 2))
                continue
            secs = self._clock() - t0
            self.c.bytes["blob"] += len(raw)
            truncated = False
            if max_bytes is not None and len(raw) > max_bytes:
                raw, truncated = raw[:max_bytes], True
            self._write_log("blob", what, status, len(raw), secs, truncated=truncated or None)
            if status >= 500:
                self._sleep(min(60, 2 ** attempt * 2))
                continue
            return status, (raw if status == 200 else b""), truncated
        return -1, b"", False

    def follow(self, path: str, what: str, *, max_bytes: Optional[int] = None) -> tuple:
        """An API path that answers 302 to a signed URL (job logs, artifact zips): (api status,
        blob status, bytes, truncated)."""
        r = self.rest(path)
        if r.status in (301, 302, 303, 307, 308) and r.location:
            bst, raw, trunc = self.blob(r.location, what, max_bytes=max_bytes)
            return r.status, bst, raw, trunc
        if r.status == 200 and r.raw:  # some endpoints answer the bytes directly
            raw = r.raw
            trunc = max_bytes is not None and len(raw) > max_bytes
            return r.status, 200, (raw[:max_bytes] if trunc else raw), trunc
        return r.status, None, b"", False
