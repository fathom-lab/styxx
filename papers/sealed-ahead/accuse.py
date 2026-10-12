# -*- coding: utf-8 -*-
"""accuse — run the frozen gate over every pool record; the step a stranger re-runs.

    python accuse.py run --params plan_params.json --pool DIR/pool.jsonl --out DIR [--styxx-path PATH]
    python accuse.py summarize --params plan_params.json --pool DIR/pool.jsonl --out DIR

For each pool record, in event-id order, this fetches the description and the diff from public
URLs without credentials, runs `styxx.diffgate.gate_diff_text(title + "\\n\\n" + body, diff,
run=None, strict=False)` (EXTERNAL-1's call), and writes one outcome line. Every record ends in
exactly one outcome; nothing is dropped silently. A CONTRADICTED claim is an accusation; a claim
the PATH-2a overlay withheld is UNCHECKABLE and is recorded with the verdict it withheld.

The frozen gate. `load_gate` imports styxx.diffgate and refuses to run unless its file hashes to
the plan's gate.sha256, either as bytes or with CRLF folded to LF (a Windows checkout made with
core.autocrlf holds the same blob with CRLF line ends; for styxx 7.49.0 the blob is
09867056957082f129985591f5ed048021e85ed48627e73854494b4a1e6e2b3a and that checkout's file
hashes to d1fca413...). The record names the interpreter, unicodedata.unidata_version, and the
LF sha256 of every styxx module loaded by the end of the run (declare, claimdetect and evidence
among them), because a verdict depends on more than one file.

The description AS OPENED. A 2026 GH Archive payload carries no title and no body. Where the
event's own payload carries them (a pre-2025-10 hour), they are used, and they are the description
as opened. Otherwise the only unauthenticated source is the REST API's pull request now
(api.github.com/repositories/<repo_id>/pulls/<number>), which gives the title and body at the
moment of the fetch, not at opening. GitHub keeps a body's edit history, but only GraphQL
(userContentEdits) exposes it, and GraphQL needs a token; the grant of 2026-10-12 allows none.
So for every 2026 record the description judged is the description at fetch time, `body_source`
says so on every outcome, and the fetch time is recorded.

The change being judged. A description read now must not be judged against the diff as it stood
at opening: an agent that opens a pull request with a placeholder commit and a plan, then pushes
the work and rewrites the description, would be accused of every count and file its finished
description names. So a record is gated only when the pull request's head now is still the
event's head (`require_unmoved_head`); otherwise its outcome is HEAD_MOVED and its diff is not
fetched. With the head unmoved, the description now and the diff at the event's head describe the
same change. The diff comes from one of two routes, named by the plan:

  pull     https://github.com/<owner>/<name>/pull/<number>.diff, the pull request's diff now,
           which is the diff at the event's head because the head has not moved;
  compare  https://github.com/<owner>/<name>/compare/<base_sha>...<head_sha>.diff, the diff
           between the event's own commits, which stays the same after the pull request moves.
           A stranger re-checking an outcome after the head moved needs this route. The grant of
           2026-10-12 names pull-request .diff URLs; the compare URL is the same host and rate
           and needs the operator's word before it is used.

The budget. The grant allows 50 requests an hour to github.com and api.github.com together, and
GitHub allows an unauthenticated caller 60 an hour on its API. A record costs two requests (the
pull request, then its diff) or one (HEAD_MOVED, EMPTY_BODY, PR_GONE). At 50 an hour that is at
most about 600 records a day. `RateLimiter` enforces the plan's rate over a sliding hour and keeps
its stamps in the output directory, so a restart does not reset it.

Outcomes: GATED, EMPTY_BODY, HEAD_MOVED, PR_GONE, PR_UNAVAILABLE, DIFF_GONE, DIFF_UNAVAILABLE,
GATE_ERROR (final) and FETCH_FAILED (not final: fetched again next run). The run is COMPLETE when
every pool record has a final outcome.
"""
from __future__ import annotations

import argparse
import collections
import http.client
import importlib
import importlib.util
import json
import os
import platform
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
OUTCOME_SCHEMA = "styxx.sealed-ahead/outcome/v0"
SUMMARY_SCHEMA = "styxx.sealed-ahead/accusations/v0"
USER_AGENT = "styxx-sealed-ahead/0 (research; read-only; unauthenticated; <=50 requests/hour)"
API = "https://api.github.com"
WEB = "https://github.com"
FINAL = ("GATED", "EMPTY_BODY", "HEAD_MOVED", "PR_GONE", "PR_UNAVAILABLE", "DIFF_GONE", "DIFF_UNAVAILABLE",
         "GATE_ERROR")
_WITHHELD = re.compile(r"^(VERIFIED|CONTRADICTED) withheld by PATH-2a \(")
_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def _sibling(name):
    key = "sealed_ahead_" + name
    mod = sys.modules.get(key)
    if mod is None:
        spec = importlib.util.spec_from_file_location(key, os.path.join(HERE, name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return mod


common = _sibling("common")


class Refused(Exception):
    """A precondition of the plan does not hold; nothing was gated."""


# ---------------------------------------------------------------- the frozen gate

class Gate:
    def __init__(self, module, identity: dict):
        self.dg, self.identity = module, identity

    def run(self, summary: str, diff_text: str):
        return self.dg.gate_diff_text(summary, diff_text, run=None, strict=False)


def match_digest(raw: bytes, expected_sha256: str):
    """'bytes' when the file hashes to the digest as it is, 'lf' when it does with CRLF folded, else None."""
    if common.sha256_hex(raw) == expected_sha256:
        return "bytes"
    if common.lf_sha256(raw) == expected_sha256:
        return "lf"
    return None


def load_gate(expected_sha256: str, styxx_path: str | None = None) -> Gate:
    if styxx_path:
        sys.path.insert(0, os.path.abspath(styxx_path))
    dg = importlib.import_module("styxx.diffgate")
    with open(dg.__file__, "rb") as fh:
        raw = fh.read()
    match = match_digest(raw, expected_sha256)
    if match is None:
        raise Refused(f"styxx.diffgate at {dg.__file__} is not the frozen gate: bytes {common.sha256_hex(raw)}, "
                      f"LF {common.lf_sha256(raw)}, the plan names {expected_sha256}")
    styxx = sys.modules.get("styxx")
    identity = {"module": "styxx.diffgate", "sha256": expected_sha256, "matched_as": match,
                "styxx_version": getattr(styxx, "__version__", None),
                "python": platform.python_version(), "implementation": platform.python_implementation(),
                "unidata_version": unicodedata.unidata_version, "platform": sys.platform}
    return Gate(dg, identity)


def loaded_styxx_closure() -> dict:
    """The LF sha256 of every styxx module loaded now, by module name."""
    out = {}
    for name, mod in sorted(sys.modules.items()):
        if not (name == "styxx" or name.startswith("styxx.")):
            continue
        f = getattr(mod, "__file__", None)
        if f and os.path.isfile(f):
            with open(f, "rb") as fh:
                out[name] = common.lf_sha256(fh.read())
    return out


# ---------------------------------------------------------------- the rate and the fetch

class RateLimiter:
    """At most `per_hour` requests in any sliding hour, stamps kept in `state_path` across runs."""

    def __init__(self, per_hour: int, clock=time.time, sleep=time.sleep, state_path: str | None = None):
        self.per_hour, self.clock, self.sleep, self.state_path = per_hour, clock, sleep, state_path
        self.stamps = collections.deque()
        if state_path and os.path.exists(state_path):
            now = clock()
            with open(state_path, "rb") as fh:
                for line in fh.read().split(b"\n"):
                    if line.strip():
                        t = float(line)
                        if t > now - 3600:
                            self.stamps.append(t)
        self.waited = 0.0

    def acquire(self) -> None:
        while True:
            now = self.clock()
            while self.stamps and self.stamps[0] <= now - 3600:
                self.stamps.popleft()
            if len(self.stamps) < self.per_hour:
                break
            wait = self.stamps[0] + 3600 - now + 0.5
            self.waited += wait
            self.sleep(wait)
        t = self.clock()
        self.stamps.append(t)
        if self.state_path:
            with open(self.state_path, "ab") as fh:
                fh.write(f"{t:.3f}\n".encode("ascii"))


class GitHub:
    """Unauthenticated GETs to github.com and api.github.com, every one through the limiter."""

    def __init__(self, limiter: RateLimiter, opener=None, sleep=time.sleep, timeout: float = 30.0,
                 backoff=(30, 120, 600), max_rate_wait: float = 3700.0, log=print, clock=time.time):
        self.limiter, self.sleep, self.timeout, self.backoff = limiter, sleep, timeout, tuple(backoff)
        self.max_rate_wait, self.log, self.clock = max_rate_wait, log, clock
        self.opener = opener or urllib.request.urlopen
        self.requests = 0

    def get(self, url: str, accept: str) -> dict:
        if not (url.startswith(API + "/") or url.startswith(WEB + "/")):
            raise Refused(f"not a host the grant names: {url}")
        attempts, last = 0, ""
        for wait in (0,) + self.backoff:
            if wait:
                self.sleep(wait)
            self.limiter.acquire()
            attempts += 1
            self.requests += 1
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
            try:
                with self.opener(req, timeout=self.timeout) as resp:
                    body = resp.read()
                    return {"status": 200, "body": body, "attempts": attempts, "url": url}
            except urllib.error.HTTPError as e:
                hdr = e.headers or {}
                if e.code in (403, 429) and (e.code == 429 or str(hdr.get("X-RateLimit-Remaining")) == "0"):
                    reset = hdr.get("X-RateLimit-Reset")
                    pause = max(1.0, float(reset) - self.clock()) if reset else float(hdr.get("Retry-After") or 60)
                    if pause > self.max_rate_wait:
                        return {"status": "rate_limited", "attempts": attempts, "url": url}
                    self.log(f"  GitHub rate limit; waiting {pause:.0f}s")
                    self.sleep(pause)
                    last = f"HTTP {e.code} rate limited"
                    continue
                if e.code >= 500:
                    last = f"HTTP {e.code}"
                    continue
                return {"status": e.code, "attempts": attempts, "url": url}
            except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
                last = f"{type(e).__name__}: {e}"[:200]
            self.log(f"  {url}: attempt {attempts} failed ({last})")
        return {"status": "failed", "attempts": attempts, "url": url, "error": last}


def api_url(rec: dict) -> str:
    return f"{API}/repositories/{rec['repo_id']}/pulls/{rec['number']}"


def pull_diff_url(full_name: str, number: int) -> str:
    return f"{WEB}/{full_name}/pull/{number}.diff"


def compare_diff_url(full_name: str, base_sha: str, head_sha: str) -> str:
    return f"{WEB}/{full_name}/compare/{base_sha}...{head_sha}.diff"


# ---------------------------------------------------------------- one record

def _text_sha(s) -> str | None:
    return None if s is None else common.sha256_hex(s.encode("utf-8"))


def accusation_id(pool_key: str, event_id: str, index: int, claim: dict) -> str:
    return common.canon_sha256({"pool_key": pool_key, "event_id": event_id, "claim_index": index,
                                "kind": claim["kind"], "text": claim["text"], "detail": claim["detail"]})


def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return repr(v)


def gate_outcome(rec: dict, gate: Gate, title, body: str, diff_bytes: bytes) -> dict:
    """The gate over one description and one diff. Pure given the gate."""
    try:
        diff_text, strict_utf8 = diff_bytes.decode("utf-8"), True
    except UnicodeDecodeError:
        diff_text, strict_utf8 = diff_bytes.decode("utf-8", errors="replace"), False
    summary = f"{title or ''}\n\n{body}"
    try:
        g = gate.run(summary, diff_text)
    except Exception as e:  # noqa: BLE001  -- never a silent pass
        return {"outcome": "GATE_ERROR", "gate_error": type(e).__name__, "diff_utf8_strict": strict_utf8}
    key = f"{rec['repo_id']}#{rec['number']}"
    claims, accusations = [], []
    for i, c in enumerate(g.claims):
        m = _WITHHELD.match(c.why or "")
        cl = {"index": i, "kind": c.kind, "text": c.text, "detail": {k: _jsonable(v) for k, v in c.detail.items()},
              "verdict": c.verdict, "why": c.why, "withheld_from": m.group(1) if m else None,
              "accusation_id": None}
        if c.verdict == "CONTRADICTED":
            cl["accusation_id"] = accusation_id(key, rec["event_id"], i, cl)
            accusations.append(cl["accusation_id"])
        claims.append(cl)
    return {"outcome": "GATED", "diff_utf8_strict": strict_utf8,
            "gate_result": {"verdict": g.verdict, "n_claims": len(g.claims), "sentences_total": g.sentences_total,
                            "uncovered_sentences": g.uncovered_sentences},
            "claims": claims, "accusations": accusations}


def accuse_one(rec: dict, gh: GitHub, gate: Gate, fetch: dict, require_unmoved_head=None) -> tuple:
    """(outcome line, inputs or None). The inputs (title, body, diff) are returned only when the
    record was gated with at least one claim, for the panel and for a stranger's check."""
    unmoved = fetch["require_unmoved_head"] if require_unmoved_head is None else require_unmoved_head
    key = f"{rec['repo_id']}#{rec['number']}"
    out = {"schema": OUTCOME_SCHEMA, "pool_key": key, "event_id": rec["event_id"], "opened_at": rec["opened_at"],
           "head_sha": rec["head_sha"], "inputs": {}, "fetch": []}
    inp = out["inputs"]
    full_name, head_now = rec["repo"], None
    payload = rec.get("as_opened")
    need_api = payload is None or (unmoved and fetch["diff_source"] == "pull")
    title = body = None
    if need_api:
        r = gh.get(api_url(rec), "application/vnd.github+json")
        out["fetch"].append({"url": r["url"], "status": r["status"], "attempts": r["attempts"],
                             "at": common.fmt_utc(_now())})
        if r["status"] in (404, 410):
            out["outcome"] = "PR_GONE"
            return out, None
        if r["status"] in ("failed", "rate_limited"):
            out["outcome"] = "FETCH_FAILED"
            return out, None
        if r["status"] != 200:
            out["outcome"] = "PR_UNAVAILABLE"
            out["http_status"] = r["status"]
            return out, None
        try:
            pr = json.loads(r["body"].decode("utf-8"))
            head_now = pr["head"]["sha"]
            full_name = pr["base"]["repo"]["full_name"]
            api_title, api_body = pr.get("title"), pr.get("body")
            if pr.get("number") != rec["number"] or not _SHA.match(head_now or ""):
                raise ValueError("unexpected pull request object")
        except (ValueError, KeyError, TypeError):
            out["outcome"] = "PR_UNAVAILABLE"
            out["http_status"] = "malformed"
            return out, None
        inp["head_at_fetch"] = head_now
        inp["head_moved"] = head_now != rec["head_sha"]
        inp["repo_now"] = full_name
        if payload is None:
            title, body = api_title, api_body
            inp["body_source"] = "api_now"
            inp["body_fetched_at"] = out["fetch"][-1]["at"]
    if payload is not None:
        title, body = payload.get("title"), payload.get("body")
        inp["body_source"] = "payload_as_opened"
    inp["title_sha256"] = _text_sha(title)
    inp["body_sha256"] = _text_sha(body)
    if not (body or "").strip():
        out["outcome"] = "EMPTY_BODY"
        return out, None
    if unmoved and inp.get("head_moved"):
        out["outcome"] = "HEAD_MOVED"
        return out, None
    route = fetch["diff_source"]
    url = (pull_diff_url(full_name, rec["number"]) if route == "pull"
           else compare_diff_url(full_name, rec["base_sha"], rec["head_sha"]))
    r = gh.get(url, "text/plain")
    out["fetch"].append({"url": r["url"], "status": r["status"], "attempts": r["attempts"],
                         "at": common.fmt_utc(_now())})
    inp["diff_route"] = route
    if r["status"] in (404, 410):
        out["outcome"] = "DIFF_GONE"
        return out, None
    if r["status"] in ("failed", "rate_limited"):
        out["outcome"] = "FETCH_FAILED"
        return out, None
    if r["status"] != 200:
        out["outcome"] = "DIFF_UNAVAILABLE"
        out["http_status"] = r["status"]
        return out, None
    diff = r["body"]
    inp["diff_sha256"] = common.sha256_hex(diff)
    inp["diff_bytes"] = len(diff)
    out.update(gate_outcome(rec, gate, title, body, diff))
    inputs = None
    if out.get("claims"):
        inputs = {"pool_key": key, "title": title, "body": body, "diff": diff.decode("utf-8", errors="replace"),
                  "diff_sha256": inp["diff_sha256"]}
    return out, inputs


def _now():
    return datetime.now(tz=common.UTC)


# ---------------------------------------------------------------- the set

STABLE_DROP = ("fetch",)


def stable(outcome: dict) -> dict:
    """What a stranger's re-run must reproduce: everything but fetch times and attempts."""
    s = {k: v for k, v in outcome.items() if k not in STABLE_DROP}
    s["inputs"] = {k: v for k, v in outcome.get("inputs", {}).items() if k != "body_fetched_at"}
    return s


def accusation_items(outcomes: list) -> list:
    """The draw's pool: one 3-tuple of strings per accusation (accusation_id, pool_key, kind),
    ordered by event id as an integer and then by the claim's index in its record."""
    rows = []
    for o in outcomes:
        if o.get("outcome") != "GATED":
            continue
        for c in o["claims"]:
            if c["verdict"] != "CONTRADICTED":
                continue
            aid = accusation_id(o["pool_key"], o["event_id"], c["index"], c)
            if aid != c["accusation_id"] or aid not in o["accusations"]:
                raise ValueError(f"{o['pool_key']}: claim {c['index']}'s accusation id does not re-derive")
            rows.append((int(o["event_id"]), c["index"], (aid, o["pool_key"], c["kind"])))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [r[2] for r in rows]


def summarize(pool_records: list, outcomes_by_key: dict, gate_identity=None, closure=None) -> dict:
    """Counts by outcome and by claim kind (never by agent family), the digests, and COMPLETE or not."""
    from styxx.beacon import pool_sha256
    order = sorted(pool_records, key=lambda r: int(r["event_id"]))
    keys = [f"{r['repo_id']}#{r['number']}" for r in order]
    missing = [k for k in keys if k not in outcomes_by_key]
    outs = [outcomes_by_key[k] for k in keys if k in outcomes_by_key]
    open_ = [o["pool_key"] for o in outs if o["outcome"] not in FINAL]
    by_outcome = collections.Counter(o["outcome"] for o in outs)
    claims, withheld, acc_kind = collections.Counter(), collections.Counter(), collections.Counter()
    for o in outs:
        for c in o.get("claims", []):
            claims[f"{c['kind']}:{c['verdict']}"] += 1
            if c["withheld_from"]:
                withheld[f"{c['kind']}:{c['withheld_from']}"] += 1
            if c["verdict"] == "CONTRADICTED":
                acc_kind[c["kind"]] += 1
    lines = b"".join(common.canon(stable(o)) + b"\n" for o in outs)
    items = accusation_items(outs)
    s = {
        "schema": SUMMARY_SCHEMA,
        "status": "COMPLETE" if not missing and not open_ else "INCOMPLETE",
        "pool_records": len(keys), "outcomes": len(outs), "missing": len(missing), "open": len(open_),
        "by_outcome": dict(sorted(by_outcome.items())),
        "claims_by_kind_and_verdict": dict(sorted(claims.items())),
        "withheld_by_overlay": dict(sorted(withheld.items())),
        "accusations": len(items), "accusations_by_kind": dict(sorted(acc_kind.items())),
        "outcomes_sha256": common.sha256_hex(lines),
        "accusations_sha256": pool_sha256(items),
        "items": [list(i) for i in items],
    }
    if gate_identity is not None:
        s["gate"] = gate_identity
    if closure is not None:
        s["styxx_closure"] = closure
        s["styxx_closure_sha256"] = common.canon_sha256(closure)
    return s


# ---------------------------------------------------------------- files

def read_outcomes(path: str) -> dict:
    """The latest outcome per pool key from an append-only ledger."""
    out = {}
    if os.path.exists(path):
        with open(path, "rb") as fh:
            for line in fh.read().split(b"\n"):
                if line.strip():
                    o = json.loads(line.decode("utf-8"))
                    out[o["pool_key"]] = o
    return out


def _append(path: str, obj) -> None:
    with open(path, "ab") as fh:
        fh.write(common.canon(obj) + b"\n")


def run(params: dict, pool_records: list, out_dir: str, gate: Gate, gh: GitHub, limit=None, log=print,
        require_unmoved_head=None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    ledger = os.path.join(out_dir, "outcomes.jsonl")
    inputs_dir = os.path.join(out_dir, "inputs")
    done = read_outcomes(ledger)
    todo = [r for r in sorted(pool_records, key=lambda r: int(r["event_id"]))
            if done.get(f"{r['repo_id']}#{r['number']}", {}).get("outcome") not in FINAL]
    if limit is not None:
        todo = todo[:limit]
    for i, rec in enumerate(todo, 1):
        o, inputs = accuse_one(rec, gh, gate, params["fetch"], require_unmoved_head)
        _append(ledger, o)
        done[o["pool_key"]] = o
        if inputs is not None and o.get("accusations"):
            os.makedirs(inputs_dir, exist_ok=True)
            name = o["pool_key"].replace("#", "_") + ".json"
            with open(os.path.join(inputs_dir, name), "wb") as fh:
                fh.write(common.canon(inputs) + b"\n")
        if i % 25 == 0 or i == len(todo):
            log(f"  {i}/{len(todo)} this run; {gh.requests} requests; waited {gh.limiter.waited:.0f}s for the rate")
    s = summarize(pool_records, done, gate.identity, loaded_styxx_closure())
    with open(os.path.join(out_dir, "accusations.json"), "wb") as fh:
        fh.write((json.dumps(s, indent=1, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "summarize"):
        p = sub.add_parser(name)
        p.add_argument("--params", required=True)
        p.add_argument("--pool", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--styxx-path", default=None, help="directory holding the styxx package to import")
        if name == "run":
            p.add_argument("--limit", type=int, default=None, help="stop after this many records (resumable)")
    a = ap.parse_args(argv)
    params = common.load_params(a.params)
    pool_mod = _sibling("pool")
    records = pool_mod.read_pool(a.pool)
    try:
        gate = load_gate(params["gate"]["sha256"], a.styxx_path)
    except Refused as e:
        print(f"REFUSED: {e}")
        return 2
    if a.cmd == "summarize":
        s = summarize(records, read_outcomes(os.path.join(a.out, "outcomes.jsonl")), gate.identity,
                      loaded_styxx_closure())
    else:
        lim = RateLimiter(params["fetch"]["rate_limit_per_hour"], state_path=os.path.join(a.out, "rate_stamps.txt"))
        os.makedirs(a.out, exist_ok=True)
        s = run(params, records, a.out, gate, GitHub(lim), limit=a.limit)
    print(json.dumps({k: s[k] for k in ("status", "pool_records", "outcomes", "open", "by_outcome", "accusations",
                                        "accusations_by_kind", "outcomes_sha256", "accusations_sha256")}, indent=1))
    return 0 if s["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    sys.exit(main())
