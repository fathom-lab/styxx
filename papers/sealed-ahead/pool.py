# -*- coding: utf-8 -*-
"""pool — the pool as a pure function of the GH Archive hours [t0, t1) and the plan's parameters.

    python pool.py ceilings --t0 2026-10-15T00:00:00Z --young-days 30     # one calibration hour
    python pool.py fields --hour 2026-10-08-15                             # what one hour carries
    python pool.py derive --params plan_params.json --out DIR              # the pool, resumable

There is no argument that chooses hours, records or a snapshot. `derive` walks every hour of the
window in time order; an hour is streamed from https://data.gharchive.org/<hour>.json.gz through
gzip line by line, one request at a time, and only derived records are kept (the admitted pull
requests and the hour's counts, a few kilobytes); the raw hour never touches the disk.

An hour ends in one of four states:

  ok        streamed to its end; its compressed bytes' sha256 and length are recorded
  gap       answered 404 at a fetch made at least gap_grace_hours after the hour ended; listed
            in the meta and never filled, and a stranger who can fetch it later is told so
  not_yet   answered 404 sooner than that (GH Archive publishes an hour about an hour late)
  failed    anything else after the retries (a 5xx, a timeout, a stream cut short)

The pool is COMPLETE only when every hour is ok or gap; otherwise it is INCOMPLETE, its digest is
printed as provisional, and the command exits 1. Hours already ok or gap are kept in DIR/hours/ as
derived records and are not fetched again; not_yet and failed hours are fetched again next run.

Assembly, a pure function (`assemble`):

  1. the admitted records of every ok hour;
  2. one record per pull request, keyed (repo_id, number): the repository's numeric id, not its
     name, because a repository renamed inside the window keeps its id; where a key repeats, the
     record with the smallest event id is kept (an 'opened' event repeated across hours is the
     same event);
  3. ordered by event id as an integer (event ids are digit strings, so '9' sorts before '10');
  4. if the plan sets them, at most max_prs_per_repo records per repository and then at most
     max_prs_per_actor per opening account, the earliest by event id kept: one holder of an old
     account and an old repository can add at most that many;
  5. pool_sha256 = sha256 of the records' canonical JSON lines, each followed by one LF.

Each hour's own admitted records are digested too (admitted_sha256), so two derivations that
disagree name the hour where they part.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import http.client
import importlib.util
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zlib
from collections import Counter
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
META_SCHEMA = "styxx.sealed-ahead/pool-meta/v0"
HOUR_SCHEMA = "styxx.sealed-ahead/pool-hour/v0"
USER_AGENT = "styxx-sealed-ahead/0 (research; read-only; one request at a time)"
FINAL = ("ok", "gap")


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
population = _sibling("population")


# ---------------------------------------------------------------- streaming one hour

class _HashingReader(io.RawIOBase):
    """Passes bytes through, hashing and counting every byte read from the response."""

    def __init__(self, f):
        self.f, self.h, self.n = f, hashlib.sha256(), 0

    def readable(self):
        return True

    def readinto(self, b):
        data = self.f.read(len(b))
        n = len(data)
        b[:n] = data
        self.h.update(data)
        self.n += n
        return n


def stream_gzip_lines(fileobj, consume):
    """Feed consume() the decompressed lines of a gzip stream (multi-member allowed); return
    (consume's result, compressed sha256, compressed byte count). Nothing is written anywhere."""
    hr = _HashingReader(fileobj)
    buf = io.BufferedReader(hr, 1 << 20)
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        result = consume(iter(gz))
        for _ in iter(lambda: gz.read(1 << 20), b""):   # consume() stopped early: finish the stream
            pass
    while buf.read(1 << 20):                              # bytes after the last member are hashed too
        pass
    return result, hr.h.hexdigest(), hr.n


class HttpArchive:
    """GH Archive over HTTP: one request at a time, retried with backoff, no credentials."""

    def __init__(self, timeout: float = 30.0, backoff=(10, 60, 300), sleep=time.sleep, opener=None,
                 log=print, recheck_404_after: float = 60.0):
        self.timeout, self.backoff, self.sleep, self.log = timeout, tuple(backoff), sleep, log
        self.opener = opener or urllib.request.urlopen
        self.recheck_404_after = recheck_404_after

    def fetch(self, hour: str, consume) -> dict:
        """One hour. A 404 is asked again once after a pause before it is reported, so a passing
        CDN miss does not become a gap; any other failure is retried with the backoff."""
        url = common.hour_url(hour)
        attempts, last, seen_404 = 0, "", False
        waits = list((0,) + self.backoff)
        while waits:
            wait = waits.pop(0)
            if wait:
                self.sleep(wait)
            attempts += 1
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with self.opener(req, timeout=self.timeout) as resp:
                    result, sha, n = stream_gzip_lines(resp, consume)
                return {"status": "ok", "result": result, "compressed_sha256": sha, "compressed_bytes": n,
                        "attempts": attempts, "url": url}
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    if seen_404:
                        return {"status": "404", "attempts": attempts, "url": url}
                    seen_404 = True
                    waits = [self.recheck_404_after]
                    continue
                last = f"HTTP {e.code}"
            except (urllib.error.URLError, http.client.HTTPException, OSError, EOFError, zlib.error,
                    ValueError) as e:
                last = f"{type(e).__name__}: {e}"[:200]
            self.log(f"  {hour}: attempt {attempts} failed ({last})")
        return {"status": "failed", "attempts": attempts, "url": url, "error": last}


# ---------------------------------------------------------------- the pool

def hour_state(fetched: dict, hour: str, fetched_at: datetime, grace_hours: int) -> str:
    if fetched["status"] == "ok":
        return "ok"
    if fetched["status"] == "404":
        end = common.parse_hour_name(hour) + timedelta(hours=1)
        return "gap" if fetched_at >= end + timedelta(hours=grace_hours) else "not_yet"
    return "failed"


def records_sha256(records) -> str:
    h = hashlib.sha256()
    for r in records:
        h.update(common.canon(r) + b"\n")
    return h.hexdigest()


def pool_key(r: dict) -> str:
    return f"{r['repo_id']}#{r['number']}"


def hour_entry(hour: str, state: str, fetched: dict, scanned) -> dict:
    """The derived, deterministic record of one hour (no fetch times: those go to the fetch log)."""
    e = {"schema": HOUR_SCHEMA, "hour": hour, "status": state}
    if state == "ok":
        recs = scanned["records"]
        e.update({"compressed_sha256": fetched["compressed_sha256"],
                  "compressed_bytes": fetched["compressed_bytes"],
                  "events": scanned["events"], "bad_json_lines": scanned["bad_json_lines"],
                  "bad_json_lines_sha256": scanned["bad_json_lines_sha256"],
                  "skipped": scanned["skipped"], "excluded": scanned["excluded"],
                  "admitted": len(recs), "admitted_by_family": scanned["admitted_by_family"],
                  "admitted_sha256": records_sha256(recs), "records": recs})
    return e


def assemble(hours: list, params: dict) -> tuple:
    """(records, summary) from the hour entries, in window order. Pure."""
    admitted = [r for h in hours if h["status"] == "ok" for r in h["records"]]
    best = {}
    for r in admitted:
        k = (r["repo_id"], r["number"])
        if k not in best or int(r["event_id"]) < int(best[k]["event_id"]):
            best[k] = r
    recs = sorted(best.values(), key=lambda r: int(r["event_id"]))
    capped = Counter()
    for cap, field in ((params.get("max_prs_per_repo"), "repo_id"), (params.get("max_prs_per_actor"), "actor_id")):
        if cap is None:
            continue
        seen, kept = Counter(), []
        for r in recs:
            seen[r[field]] += 1
            if seen[r[field]] <= cap:
                kept.append(r)
            else:
                capped[field] += 1
        recs = kept
    families = Counter(r["family"] for r in recs)
    summary = {"admitted_before_dedup": len(admitted), "duplicates": len(admitted) - len(best),
               "capped_by_repo": capped["repo_id"], "capped_by_actor": capped["actor_id"],
               "records": len(recs), "families_for_sizing": dict(sorted(families.items())),
               "pool_sha256": records_sha256(recs)}
    return recs, summary


def merge_census(parts: list) -> dict:
    keysets = {k: Counter() for k in population.Census.PARTS}
    actions, body, n = Counter(), Counter(), 0
    for p in parts:
        n += p["events"]
        actions.update(p["pr_actions"])
        body.update(p["pr_body"])
        for k, rows in p["keysets"].items():
            for ks, c in rows:
                keysets[k][ks] += c
    return {"events": n, "pr_actions": dict(sorted(actions.items())), "pr_body": dict(sorted(body.items())),
            "keysets": {k: sorted(v.items(), key=lambda kv: (-kv[1], kv[0])) for k, v in keysets.items()}}


def derive(params: dict, table: dict, archive, now=None, cache_dir=None, skip_hours=(), log=print) -> dict:
    """Walk every hour of the window and assemble the pool. `skip_hours` is only for a stranger
    replaying a published gap list (stranger.py checks each such hour itself); the lab passes none."""
    now = now or (lambda: datetime.now(tz=common.UTC))
    ctx = population.Context(params, table)
    hours = common.hours_between(ctx.t0, ctx.t1)
    entries, fetch_log, census = [], [], []
    for h in hours:
        cached = _load_hour(cache_dir, h) if cache_dir else None
        if cached is not None:
            entries.append(cached)
            continue
        if h in skip_hours:
            entries.append({"schema": HOUR_SCHEMA, "hour": h, "status": "gap"})
            fetch_log.append({"hour": h, "state": "gap", "replayed_from_published_gaps": True})
            continue

        def consume(lines, h=h):            # a fresh census per attempt: a retried hour counts once
            obs = population.Census()
            out = population.scan_lines(lines, ctx, h, [obs])
            out["census"] = obs.result()
            return out
        t_start = time.time()
        fetched = archive.fetch(h, consume)
        at = now()
        state = hour_state(fetched, h, at, params["gap_grace_hours"])
        entry = hour_entry(h, state, fetched, fetched.get("result"))
        if state == "ok":
            entry["census"] = fetched["result"]["census"]
        entries.append(entry)
        fetch_log.append({"hour": h, "url": fetched.get("url"), "fetch_status": fetched["status"], "state": state,
                          "attempts": fetched.get("attempts"), "error": fetched.get("error"),
                          "fetched_at": common.fmt_utc(at), "seconds": round(time.time() - t_start, 1)})
        log(f"{h}: {state}" + (f"  admitted {entry['admitted']}" if state == "ok" else ""))
        if cache_dir and state in FINAL:
            _save_hour(cache_dir, entry)
    for e in entries:
        if e["status"] == "ok" and "census" in e:
            census.append(e["census"])
    records, summary = assemble(entries, params)
    open_hours = [{"hour": e["hour"], "status": e["status"]} for e in entries if e["status"] not in FINAL]
    meta = {
        "schema": META_SCHEMA,
        "status": "COMPLETE" if not open_hours else "INCOMPLETE",
        "params_sha256": common.params_sha256(params),
        "identity_rules_sha256": params["identity_rules_sha256"],
        "window": {"t0": params["t0"], "t1": params["t1"], "hours": len(hours)},
        "gaps": [e["hour"] for e in entries if e["status"] == "gap"],
        "open_hours": open_hours,
        "hours": [{k: v for k, v in e.items() if k not in ("records", "census")} for e in entries],
        **summary,
        "fields_census": merge_census(census),
    }
    meta["derivation_sha256"] = common.canon_sha256({k: v for k, v in meta.items() if k != "fields_census"})
    return {"records": records, "meta": meta, "fetch_log": fetch_log, "entries": entries}


def _hour_path(cache_dir: str, hour: str) -> str:
    return os.path.join(cache_dir, "hours", f"{hour}.json")


def _load_hour(cache_dir: str, hour: str):
    p = _hour_path(cache_dir, hour)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as fh:
        e = json.loads(fh.read().decode("utf-8"))
    if e.get("schema") != HOUR_SCHEMA or e.get("hour") != hour or e.get("status") not in FINAL:
        raise ValueError(f"cached hour {p} is not a final derived record; delete it to fetch the hour again")
    if e["status"] == "ok" and records_sha256(e["records"]) != e["admitted_sha256"]:
        raise ValueError(f"cached hour {p}: its records do not hash to its own admitted_sha256")
    return e


def _save_hour(cache_dir: str, entry: dict) -> None:
    os.makedirs(os.path.join(cache_dir, "hours"), exist_ok=True)
    _write_json(_hour_path(cache_dir, entry["hour"]), entry)


def _write_json(path: str, obj) -> None:
    data = (json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)
    with open(path, "rb") as fh:
        if fh.read() != data:
            raise OSError(f"write did not land: {path}")


def write_pool(out_dir: str, derived: dict) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    data = b"".join(common.canon(r) + b"\n" for r in derived["records"])
    if common.sha256_hex(data) != derived["meta"]["pool_sha256"]:
        raise AssertionError("pool bytes do not hash to the meta's pool_sha256")
    with open(os.path.join(out_dir, "pool.jsonl"), "wb") as fh:
        fh.write(data)
    _write_json(os.path.join(out_dir, "pool_meta.json"), derived["meta"])
    with open(os.path.join(out_dir, "fetch_log.jsonl"), "ab") as fh:
        for line in derived["fetch_log"]:
            fh.write(common.canon(line) + b"\n")
    with open(os.path.join(out_dir, "pool.jsonl"), "rb") as fh:
        if common.sha256_hex(fh.read()) != derived["meta"]["pool_sha256"]:
            raise OSError("pool.jsonl on disk does not hash to pool_sha256")
    return derived["meta"]


def read_pool(path: str) -> list:
    with open(path, "rb") as fh:
        return [json.loads(l.decode("utf-8")) for l in fh.read().split(b"\n") if l]


# ---------------------------------------------------------------- command line (network)

def _cmd_ceilings(a) -> int:
    t0 = common.parse_utc(a.t0)
    hour = common.ceiling_hour_for(t0, a.young_days)

    def consume(lines):
        obs = population.Ceilings()
        for e in population.iter_events(lines, population.BadLines()):
            obs.add(e)
        return obs.result()
    got = HttpArchive().fetch(hour, consume)
    if got["status"] != "ok":
        print(json.dumps({"ceiling_hour": hour, **{k: v for k, v in got.items() if k != "result"}}, indent=1))
        return 1
    print(json.dumps({"ceiling_hour": hour, "compressed_sha256": got["compressed_sha256"],
                      "compressed_bytes": got["compressed_bytes"], **got["result"]}, indent=1))
    return 0


def _cmd_fields(a) -> int:
    def consume(lines):
        obs = population.Census()
        bad = population.BadLines()
        for e in population.iter_events(lines, bad):
            obs.add(e)
        return {**obs.result(), **bad.result()}
    got = HttpArchive().fetch(a.hour, consume)
    print(json.dumps({"hour": a.hour, **got}, indent=1, ensure_ascii=False))
    return 0 if got["status"] == "ok" else 1


def _cmd_derive(a) -> int:
    params = common.load_params(a.params)
    table = population.load_identity_table(a.table)
    derived = derive(params, table, HttpArchive(), cache_dir=a.out)
    meta = write_pool(a.out, derived)
    print(json.dumps({k: meta[k] for k in ("status", "records", "pool_sha256", "derivation_sha256", "gaps",
                                           "open_hours")}, indent=1))
    if meta["status"] != "COMPLETE":
        print("INCOMPLETE: the digest above is provisional; run again after the open hours settle")
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("ceilings")
    c.add_argument("--t0", required=True)
    c.add_argument("--young-days", type=int, required=True)
    f = sub.add_parser("fields")
    f.add_argument("--hour", required=True)
    d = sub.add_parser("derive")
    d.add_argument("--params", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--table", default=population.TABLE_PATH)
    a = ap.parse_args(argv)
    return {"ceilings": _cmd_ceilings, "fields": _cmd_fields, "derive": _cmd_derive}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
