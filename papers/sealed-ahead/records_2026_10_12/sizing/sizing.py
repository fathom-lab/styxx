# -*- coding: utf-8 -*-
"""sizing — the sealed-ahead sizing week, run through the builder's own scripts.

    py -3.12 -B sizing.py ceilings    # the calibration hours named by sizing_plan_2026_09_28.json
    py -3.12 -B sizing.py derive      # the 168 hours through pool.derive (resumable)
    py -3.12 -B sizing.py sample      # the seeded sample of at most 300 pool records
    py -3.12 -B sizing.py accuse      # the sample through accuse.accuse_one and the 7.49.0 gate (resumable)
    py -3.12 -B sizing.py report      # ../sizing_week_2026_09_28.json

Every choice lives in sizing_plan_2026_09_28.json, committed before this ran. The builder's
modules are loaded from its worktree by path and refused unless each hashes (LF) to the plan's
digest. Nothing here changes a record, an outcome or a digest the builder's code computes:

  - pool.derive runs unmodified. Two observers ride on it. A line counter wraps the decompressed
    line iterator (lines, decompressed bytes), and population.Census is replaced, in this process
    only, by a subclass whose add() calls the original and then counts what the sizing needs
    (event types, opened pull requests, agent-matched openings, the age rule at other ceilings,
    identity near misses). Census output is kept only in fields_census and in the hour cache,
    which no digest reads (pool.derive leaves fields_census out of derivation_sha256).
  - accuse runs record by record through accuse.accuse_one, written to the ledger with
    accuse._append and summarized with accuse.summarize, exactly as accuse.run does; the loop is
    written out only so that each record prints one progress line.

Written outside sizing_derived/: nothing but ../sizing_week_2026_09_28.json, by `report`. The
derived pool, outcomes and inputs stay in sizing_derived/ and are not committed. No output names
a repository, an account or a product beside an accusation or a rate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
ROOT = os.path.dirname(HERE)
PLAN_PATH = os.path.join(HERE, "sizing_plan_2026_09_28.json")
DERIVED = os.path.join(ROOT, "sizing_derived")
POOL_DIR = os.path.join(DERIVED, "pool")
SAMPLE_DIR = os.path.join(DERIVED, "sample")
ACCUSE_DIR = os.path.join(DERIVED, "accuse")
STYXX_PATH = os.path.join(DERIVED, "styxx_7_49_0")
CEILINGS_PATH = os.path.join(DERIVED, "ceilings.json")
PARAMS_PATH = os.path.join(DERIVED, "params_provisional.json")
REPORT_PATH = os.path.join(ROOT, "sizing_week_2026_09_28.json")
Z95 = 1.959964
NO_AGE = 2 ** 62
AGENT_WORDS = ("copilot", "devin", "jules", "codex", "cursor", "claude")
CANDIDATE_PREFIXES = ("copilot/", "devin/", "jules/")


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"{utc_now()} {msg}", flush=True)


def read_json(path: str):
    with open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def write_json(path: str, obj) -> None:
    data = (json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)
    with open(path, "rb") as fh:
        if fh.read() != data:
            raise OSError(f"write did not land: {path}")


# ---------------------------------------------------------------- the builder's modules, pinned

PLAN = read_json(PLAN_PATH)
BUILDER_DIR = os.path.join(PLAN["builder"]["repository_path"], "papers", "sealed-ahead")


def _lf_sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def _load(name: str):
    key = "sealed_ahead_" + name
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(BUILDER_DIR, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def _check_builder() -> None:
    want = PLAN["builder"]["modules_lf_sha256"]
    for fname, digest in sorted(want.items()):
        got = _lf_sha256(os.path.join(BUILDER_DIR, fname))
        if got != digest:
            raise SystemExit(f"REFUSED: {fname} hashes to {got}, the sizing plan names {digest}")
    got = _lf_sha256(os.path.join(BUILDER_DIR, "identity_table.json"))
    if got != PLAN["builder"]["identity_table_lf_sha256"]:
        raise SystemExit(f"REFUSED: identity_table.json hashes to {got}")


_check_builder()
common = _load("common")          # loaded before pool.py so its _sibling finds the same module objects
population = _load("population")
pool = _load("pool")
accuse = _load("accuse")


# ---------------------------------------------------------------- observers riding on pool.derive

class LineCounter:
    """Counts the decompressed lines and bytes pool.derive's scan reads; passes them through."""

    def __init__(self, lines):
        self.it, self.lines, self.bytes = iter(lines), 0, 0

    def __iter__(self):
        return self

    def __next__(self):
        raw = next(self.it)
        self.lines += 1
        self.bytes += len(raw)
        return raw


class CountingArchive:
    """pool.HttpArchive with a LineCounter on every attempt; the counts go into the hour's census."""

    def __init__(self, inner):
        self.inner = inner

    def fetch(self, hour, consume):
        def wrapped(lines):
            c = LineCounter(lines)
            out = consume(c)
            out["census"]["sizing_stream"] = {"lines": c.lines, "decompressed_bytes": c.bytes}
            return out
        return self.inner.fetch(hour, wrapped)


class SizingObserver:
    """What the sizing needs from each event, as counts. Keeps no name except app ([bot]) logins
    near an agent word, which are identity evidence (E4), never paired with a count."""

    def __init__(self, rules, alt_ctx: dict):
        self.rules, self.alt_ctx = rules, alt_ctx
        self.types, self.agent_opened = Counter(), Counter()
        self.pr_opened = 0
        self.alt_admitted = {k: Counter() for k in alt_ctx}
        self.alt_excluded = {k: Counter() for k in alt_ctx}
        self.prefix_case_variant, self.candidate_prefix = Counter(), Counter()
        self.login_near_miss = Counter()
        self.near_miss_bot_logins = set()

    def add(self, e) -> None:
        t = e.get("type") if isinstance(e, dict) else None
        self.types[str(t)] += 1
        if t != "PullRequestEvent":
            return
        p = e.get("payload") or {}
        if not isinstance(p, dict) or p.get("action") != "opened":
            return
        self.pr_opened += 1
        login = (e.get("actor") or {}).get("login")
        head_ref = ((p.get("pull_request") or {}).get("head") or {}).get("ref")
        matched = population.match_identity(login, head_ref, self.rules)
        if matched:
            fams = {r["family"] for r in matched}
            self.agent_opened[matched[0]["family"] if len(fams) == 1 else "mixed"] += 1
            for name, ctx in self.alt_ctx.items():
                kind, out = population.classify(e, ctx, "")
                if kind == "admit":
                    self.alt_admitted[name][out["family"]] += 1
                elif kind == "exclude":
                    self.alt_excluded[name][out] += 1
            return
        if isinstance(head_ref, str):
            low = head_ref.lower()
            for r in self.rules:
                if r["match"] == "head_ref_prefix" and low.startswith(r["value"].lower()):
                    self.prefix_case_variant[r["rule"]] += 1
            for pre in CANDIDATE_PREFIXES:
                if low.startswith(pre):
                    self.candidate_prefix[pre] += 1
        if isinstance(login, str):
            low = login.lower()
            for w in AGENT_WORDS:
                if w in low:
                    bot = low.endswith("[bot]")
                    self.login_near_miss[f"{w}:{'bot' if bot else 'not_bot'}"] += 1
                    if bot:
                        self.near_miss_bot_logins.add(login)

    def result(self) -> dict:
        return {"event_types": dict(sorted(self.types.items())), "pr_opened": self.pr_opened,
                "agent_opened_by_family": dict(sorted(self.agent_opened.items())),
                "alt_admitted": {k: dict(sorted(v.items())) for k, v in self.alt_admitted.items()},
                "alt_excluded": {k: dict(sorted(v.items())) for k, v in self.alt_excluded.items()},
                "prefix_case_variant": dict(sorted(self.prefix_case_variant.items())),
                "candidate_prefix_unadmitted": dict(sorted(self.candidate_prefix.items())),
                "login_near_miss": dict(sorted(self.login_near_miss.items())),
                "near_miss_bot_logins": sorted(self.near_miss_bot_logins)}


def install_census(rules, alt_ctx) -> None:
    base = population.Census

    class SizingCensus(base):
        def __init__(self):
            super().__init__()
            self.sizing = SizingObserver(rules, alt_ctx)

        def add(self, e) -> None:
            super().add(e)
            self.sizing.add(e)

        def result(self) -> dict:
            out = super().result()
            out["sizing"] = self.sizing.result()
            return out

    SizingCensus.PARTS = base.PARTS
    population.Census = SizingCensus


# ---------------------------------------------------------------- ceilings and params

def cmd_ceilings(_a) -> int:
    t0 = common.parse_utc(PLAN["window"]["t0"])
    out = read_json(CEILINGS_PATH) if os.path.exists(CEILINGS_PATH) else {"hours": {}}
    archive = pool.HttpArchive(log=log)
    for yd in PLAN["age_rule_sensitivity"]["young_days"]:
        hour = common.ceiling_hour_for(t0, yd)
        if hour != PLAN["age_rule_sensitivity"]["calibration_hours"][str(yd)]:
            raise SystemExit(f"REFUSED: young_days {yd} names {hour}, the plan says otherwise")
        if hour in out["hours"]:
            continue

        def consume(lines):
            obs, bad = population.Ceilings(), population.BadLines()
            for e in population.iter_events(lines, bad):
                obs.add(e)
            return {**obs.result(), **bad.result()}
        t = time.time()
        got = archive.fetch(hour, consume)
        if got["status"] != "ok":
            log(f"ceilings {hour}: {got['status']} {got.get('error')}")
            write_json(CEILINGS_PATH, out)
            return 1
        out["hours"][hour] = {"young_days": yd, "compressed_sha256": got["compressed_sha256"],
                              "compressed_bytes": got["compressed_bytes"], "attempts": got["attempts"],
                              "fetched_at": utc_now(), "seconds": round(time.time() - t, 1), **got["result"]}
        log(f"ceilings {hour}: {got['compressed_bytes']} bytes, {got['result']['events']} events, "
            f"account ceiling {got['result']['account_id_ceiling']}, repo ceiling {got['result']['repo_id_ceiling']}")
        write_json(CEILINGS_PATH, out)
    pp = {k: v for k, v in PLAN["provisional_params"].items() if k != "note"}
    prim = out["hours"][pp["ceiling_hour"]]
    pp["account_id_ceiling"], pp["repo_id_ceiling"] = prim["account_id_ceiling"], prim["repo_id_ceiling"]
    probs = common.validate_params(pp)
    if probs:
        raise SystemExit("REFUSED: " + "; ".join(probs))
    write_json(PARAMS_PATH, pp)
    log(f"params written: {PARAMS_PATH} (params_sha256 {common.params_sha256(pp)})")
    return 0


def alt_contexts(params: dict, table: dict) -> dict:
    ceil = read_json(CEILINGS_PATH)["hours"]
    out = {}
    for yd in PLAN["age_rule_sensitivity"]["young_days"]:
        h = PLAN["age_rule_sensitivity"]["calibration_hours"][str(yd)]
        p = dict(params, account_id_ceiling=ceil[h]["account_id_ceiling"], repo_id_ceiling=ceil[h]["repo_id_ceiling"])
        out[f"young_days_{yd}"] = population.Context(p, table)
    out["no_age_rule"] = population.Context(dict(params, account_id_ceiling=NO_AGE, repo_id_ceiling=NO_AGE), table)
    return out


# ---------------------------------------------------------------- derive

def cmd_derive(_a) -> int:
    params = common.load_params(PARAMS_PATH)
    table = population.load_identity_table()
    install_census(population.rules_of(table), alt_contexts(params, table))
    archive = CountingArchive(pool.HttpArchive(log=log))
    log("derive: start")
    derived = pool.derive(params, table, archive, cache_dir=POOL_DIR, log=log)
    meta = pool.write_pool(POOL_DIR, derived)
    log("derive: " + json.dumps({k: meta[k] for k in ("status", "records", "pool_sha256", "derivation_sha256",
                                                      "gaps", "open_hours")}))
    return 0 if meta["status"] == "COMPLETE" else 1


# ---------------------------------------------------------------- sample

def sample_key(seed: str, pool_key: str) -> str:
    return common.canon_sha256({"pool_key": pool_key, "seed": seed})


def cmd_sample(_a) -> int:
    meta = read_json(os.path.join(POOL_DIR, "pool_meta.json"))
    records = pool.read_pool(os.path.join(POOL_DIR, "pool.jsonl"))
    if pool.records_sha256(records) != meta["pool_sha256"]:
        raise SystemExit("REFUSED: pool.jsonl does not hash to pool_meta.json's pool_sha256")
    if meta["status"] != "COMPLETE":
        raise SystemExit("REFUSED: the pool is INCOMPLETE; sample a complete pool only")
    seed, n = PLAN["sample"]["seed"], PLAN["sample"]["n_max"]
    keyed = sorted(records, key=lambda r: sample_key(seed, pool.pool_key(r)))
    chosen = sorted(keyed[:n], key=lambda r: int(r["event_id"]))
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    data = b"".join(common.canon(r) + b"\n" for r in chosen)
    with open(os.path.join(SAMPLE_DIR, "sample_pool.jsonl"), "wb") as fh:
        fh.write(data)
    sm = {"schema": "styxx.sealed-ahead/sizing-sample/v0", "seed": seed, "n_max": n, "pool_records": len(records),
          "pool_sha256": meta["pool_sha256"], "sample_records": len(chosen),
          "sample_sha256": common.sha256_hex(data),
          "sample_keys_sha256": common.canon_sha256(sorted(pool.pool_key(r) for r in chosen)),
          "largest_key_taken": sample_key(seed, pool.pool_key(keyed[len(chosen) - 1])) if chosen else None,
          "families_for_sizing": dict(sorted(Counter(r["family"] for r in chosen).items()))}
    write_json(os.path.join(SAMPLE_DIR, "sample_meta.json"), sm)
    log("sample: " + json.dumps(sm))
    return 0


# ---------------------------------------------------------------- accuse

def cmd_accuse(a) -> int:
    params = common.load_params(PARAMS_PATH)
    sample = pool.read_pool(os.path.join(SAMPLE_DIR, "sample_pool.jsonl"))
    sm = read_json(os.path.join(SAMPLE_DIR, "sample_meta.json"))
    if common.sha256_hex(b"".join(common.canon(r) + b"\n" for r in sample)) != sm["sample_sha256"]:
        raise SystemExit("REFUSED: the sample does not hash to sample_meta.json")
    gate = accuse.load_gate(params["gate"]["sha256"], STYXX_PATH)
    if gate.identity["styxx_version"] != PLAN["gate"]["release"]:
        raise SystemExit(f"REFUSED: styxx {gate.identity['styxx_version']} loaded, the plan names {PLAN['gate']['release']}")
    os.makedirs(ACCUSE_DIR, exist_ok=True)
    limiter = accuse.RateLimiter(params["fetch"]["rate_limit_per_hour"],
                                 state_path=os.path.join(ACCUSE_DIR, "rate_stamps.txt"))
    gh = accuse.GitHub(limiter, log=log)
    ledger = os.path.join(ACCUSE_DIR, "outcomes.jsonl")
    inputs_dir = os.path.join(ACCUSE_DIR, "inputs")
    order = sorted(sample, key=lambda r: int(r["event_id"]))
    for pass_no in range(1, a.passes + 1):
        done = accuse.read_outcomes(ledger)
        todo = [r for r in order if done.get(pool.pool_key(r), {}).get("outcome") not in accuse.FINAL]
        if not todo:
            break
        log(f"accuse: pass {pass_no}, {len(todo)} records without a final outcome")
        for i, rec in enumerate(todo, 1):
            o, inputs = accuse.accuse_one(rec, gh, gate, params["fetch"])
            accuse._append(ledger, o)
            done[o["pool_key"]] = o
            if inputs is not None and o.get("accusations"):
                os.makedirs(inputs_dir, exist_ok=True)
                with open(os.path.join(inputs_dir, o["pool_key"].replace("#", "_") + ".json"), "wb") as fh:
                    fh.write(common.canon(inputs) + b"\n")
            final = sum(1 for k in (pool.pool_key(r) for r in order) if done.get(k, {}).get("outcome") in accuse.FINAL)
            gated = sum(1 for v in done.values() if v.get("outcome") == "GATED")
            acc = sum(len(v.get("accusations") or []) for v in done.values() if v.get("outcome") == "GATED")
            log(f"PROGRESS pass {pass_no} {i}/{len(todo)} outcome={o['outcome']} | final {final}/{len(order)} "
                f"gated {gated} accusations {acc} | requests {gh.requests} rate-wait {limiter.waited:.0f}s")
    done = accuse.read_outcomes(ledger)
    s = accuse.summarize(sample, done, gate.identity, accuse.loaded_styxx_closure())
    accuse._write_json(os.path.join(ACCUSE_DIR, "accusations.json"), s)
    accuse._write_json(os.path.join(ACCUSE_DIR, "outcome_digests.json"), accuse.outcome_digests(sample, done, s))
    log("accuse: DONE " + json.dumps({k: s[k] for k in ("status", "pool_records", "outcomes", "open", "by_outcome",
                                                        "accusations", "accusations_by_kind")}))
    return 0 if s["status"] == "COMPLETE" else 1


# ---------------------------------------------------------------- report

def wilson(k: int, n: int, z: float = Z95):
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo = 0.0 if k == 0 else max(0.0, centre - half)       # exact ends: float rounding leaves 1e-19 at k = 0
    hi = 1.0 if k == n else min(1.0, centre + half)
    return [lo, hi]


def _r(x, nd=3):
    return None if x is None else round(x, nd)


def window_days(rate_per_record, weekly_records):
    """Days of window for 60 accusations at a rate per admitted record; None means unbounded."""
    if not rate_per_record or not weekly_records:
        return None
    return 60.0 / (rate_per_record * weekly_records) * 7.0


def cmd_report(_a) -> int:
    params = common.load_params(PARAMS_PATH)
    ceil = read_json(CEILINGS_PATH)
    meta = read_json(os.path.join(POOL_DIR, "pool_meta.json"))
    records = pool.read_pool(os.path.join(POOL_DIR, "pool.jsonl"))
    if pool.records_sha256(records) != meta["pool_sha256"]:
        raise SystemExit("REFUSED: pool.jsonl does not hash to pool_sha256")
    hours = [pool._load_hour(POOL_DIR, h["hour"]) if h["status"] in pool.FINAL else h for h in meta["hours"]]
    fetch_log = []
    flp = os.path.join(POOL_DIR, "fetch_log.jsonl")
    if os.path.exists(flp):
        with open(flp, "rb") as fh:
            fetch_log = [json.loads(l) for l in fh.read().split(b"\n") if l.strip()]
    last_fetch = {}
    for f in fetch_log:
        last_fetch[f["hour"]] = f

    # per hour
    per_hour, tot = [], Counter()
    agent_opened, alt_adm, alt_exc = Counter(), {}, {}
    case_var, cand, near, bot_logins = Counter(), Counter(), Counter(), set()
    types_total = Counter()
    for h in hours:
        row = {"hour": h["hour"], "status": h["status"]}
        if h["status"] == "ok":
            sz = (h.get("census") or {}).get("sizing", {})
            st = (h.get("census") or {}).get("sizing_stream", {})
            pr_open = sz.get("pr_opened")
            row.update({"compressed_bytes": h["compressed_bytes"], "compressed_sha256": h["compressed_sha256"],
                        "decompressed_bytes": st.get("decompressed_bytes"), "lines": st.get("lines"),
                        "events": h["events"], "bad_json_lines": h["bad_json_lines"],
                        "pull_request_events": sz.get("event_types", {}).get("PullRequestEvent", 0),
                        "push_events": sz.get("event_types", {}).get("PushEvent", 0),
                        "pr_opened": pr_open,
                        "agent_opened": sum(sz.get("agent_opened_by_family", {}).values()),
                        "admitted": h["admitted"], "excluded": h["excluded"],
                        "attempts": (last_fetch.get(h["hour"]) or {}).get("attempts"),
                        "seconds": (last_fetch.get(h["hour"]) or {}).get("seconds")})
            for k in ("compressed_bytes", "decompressed_bytes", "lines", "events", "bad_json_lines",
                      "pull_request_events", "push_events", "pr_opened", "agent_opened", "admitted"):
                tot[k] += row[k] or 0
            agent_opened.update(sz.get("agent_opened_by_family", {}))
            for k, v in sz.get("alt_admitted", {}).items():
                alt_adm.setdefault(k, Counter()).update(v)
            for k, v in sz.get("alt_excluded", {}).items():
                alt_exc.setdefault(k, Counter()).update(v)
            case_var.update(sz.get("prefix_case_variant", {}))
            cand.update(sz.get("candidate_prefix_unadmitted", {}))
            near.update(sz.get("login_near_miss", {}))
            bot_logins.update(sz.get("near_miss_bot_logins", []))
            types_total.update(sz.get("event_types", {}))
        else:
            row.update({"attempts": (last_fetch.get(h["hour"]) or {}).get("attempts"),
                        "error": (last_fetch.get(h["hour"]) or {}).get("error")})
        per_hour.append(row)
    ok_hours = [r for r in per_hour if r["status"] == "ok"]
    adm_per_hour = [r["admitted"] for r in ok_hours]
    excluded_total = Counter()
    for r in ok_hours:
        excluded_total.update(r["excluded"])

    # the pool, aggregated (no names)
    fam = Counter(r["family"] for r in records)
    by_repo = Counter(r["repo_id"] for r in records)
    by_actor = Counter(r["actor_id"] for r in records)
    actors_by_fam = {f: len({r["actor_id"] for r in records if r["family"] == f}) for f in fam}
    repos_by_fam = {f: len({r["repo_id"] for r in records if r["family"] == f}) for f in fam}
    login_rule_ids = {}
    table = population.load_identity_table()
    for rule in population.rules_of(table):
        if rule["match"] == "actor_login":
            ids = sorted({r["actor_id"] for r in records if rule["rule"] in r["rules"]})
            login_rule_ids[rule["rule"]] = {"distinct_actor_ids": len(ids), "actor_ids": ids[:5]}
    rule_hits = Counter(x for r in records for x in r["rules"])
    caps = {}
    for k in (1, 2, 5, 10, 20, 50):
        caps[str(k)] = {"max_prs_per_repo": sum(min(c, k) for c in by_repo.values()),
                        "max_prs_per_actor": sum(min(c, k) for c in by_actor.values())}
    same_repo = sum(1 for r in records if r.get("head_repo_id") == r["repo_id"])
    n_ok = len(ok_hours)
    week_scale = 168 / n_ok if n_ok else None

    # the sample through the gate
    sm = read_json(os.path.join(SAMPLE_DIR, "sample_meta.json")) if os.path.exists(
        os.path.join(SAMPLE_DIR, "sample_meta.json")) else None
    acc_path = os.path.join(ACCUSE_DIR, "accusations.json")
    sample_block = None
    if sm and os.path.exists(acc_path):
        s = read_json(acc_path)
        sample = pool.read_pool(os.path.join(SAMPLE_DIR, "sample_pool.jsonl"))
        done = accuse.read_outcomes(os.path.join(ACCUSE_DIR, "outcomes.jsonl"))
        outs = [done[pool.pool_key(r)] for r in sample if pool.pool_key(r) in done]
        final = [o for o in outs if o["outcome"] in accuse.FINAL]
        n = len(final)
        gated = [o for o in final if o["outcome"] == "GATED"]
        n_g = len(gated)
        kinds_claimed = Counter()
        acc_kind, rec_kind, any_rec = Counter(), Counter(), 0
        per_rec_acc = []
        for o in gated:
            ks = Counter()
            for c in o.get("claims", []):
                kinds_claimed[c["kind"]] += 1
                if c["verdict"] == "CONTRADICTED":
                    ks[c["kind"]] += 1
            acc_kind.update(ks)
            for k in ks:
                rec_kind[k] += 1
            if ks:
                any_rec += 1
                per_rec_acc.append(sum(ks.values()))
        total_acc = sum(acc_kind.values())
        by_day = {}                     # outcome counts by the day the pull request was opened (no names)
        for o in final:
            by_day.setdefault(o["opened_at"][:10], Counter())[o["outcome"]] += 1
        gaps_s = []                     # BATTERY_vs_build D3: seconds from the head check to the diff GET's end
        for o in gated:
            ats = [f.get("at") for f in o.get("fetch", [])]
            if len(ats) >= 2 and all(ats[-2:]):
                t_a, t_b = (common.parse_utc(x) for x in ats[-2:])
                gaps_s.append((t_b - t_a).total_seconds())
        requests =sum(f.get("attempts") or 0 for o in final for f in o.get("fetch", []))
        req_per_rec = requests / n if n else None
        W = len(records) * (week_scale or 1)

        def kind_row(acc, recs):
            wi = wilson(recs, n)
            wg = wilson(recs, n_g)
            rate = acc / n if n else None
            return {"accusations": acc, "records_with_one_or_more": recs,
                    "per_1000_admitted": _r(1000 * rate) if rate is not None else None,
                    "records_per_1000_admitted_wilson95": [_r(1000 * wi[0]), _r(1000 * wi[1])] if wi else None,
                    "per_1000_gated": _r(1000 * acc / n_g) if n_g else None,
                    "records_per_1000_gated_wilson95": [_r(1000 * wg[0]), _r(1000 * wg[1])] if wg else None}
        per_kind = {k: kind_row(acc_kind[k], rec_kind[k]) for k in sorted(set(kinds_claimed) | set(acc_kind))}
        any_row = kind_row(total_acc, any_rec)
        wi = wilson(any_rec, n)
        rate_pt = total_acc / n if n else None
        lo_rec, hi_rec = (wi or [None, None])
        mean_per_accused = (total_acc / any_rec) if any_rec else None
        proj = {
            "weekly_admitted_records": round(W),
            "accusations_per_7_days_point": _r(rate_pt * W, 1) if rate_pt is not None else None,
            "accused_records_per_7_days_wilson95": [_r(lo_rec * W, 1), _r(hi_rec * W, 1)] if wi else None,
            "window_days_for_60_at_lower_end": _r(window_days(lo_rec, W), 1) if wi else None,
            "window_days_for_60_at_point": _r(window_days(rate_pt, W), 1) if rate_pt else None,
            "window_days_for_60_at_upper_end": _r(window_days(hi_rec, W), 1) if wi else None,
            "window_days_for_60_at_aidev_0p0013": _r(window_days(0.0013, W), 1),
            "note": "The lower and upper ends use the Wilson interval on records with one or more accusations, "
                    "which counts an accused record once; the point estimate counts every accusation. "
                    "None means unbounded (the sample cannot exclude a zero rate).",
        }
        budget = None
        if req_per_rec:
            recs_per_day = 50 * 24 / req_per_rec
            budget = {"requests_measured": requests, "requests_per_record": _r(req_per_rec),
                      "records_gated_per_day_at_50_per_hour": _r(recs_per_day, 1),
                      "share_of_week_fetchable_in_7_days": _r(min(1.0, recs_per_day * 7 / W)) if W else None}
            for label, rate in (("lower_end", lo_rec), ("point", rate_pt), ("upper_end", hi_rec), ("aidev_0p0013", 0.0013)):
                if not rate:
                    budget[f"days_for_60_at_{label}"] = None
                    continue
                need = 60.0 / rate                       # admitted records to fetch
                days_pop = need / (W / 7.0)
                days_budget = need / recs_per_day
                budget[f"days_for_60_at_{label}"] = {"records_needed": round(need), "days_population_bound": _r(days_pop, 1),
                                                     "days_fetch_bound": _r(days_budget, 1),
                                                     "days": _r(max(days_pop, days_budget), 1)}
        sample_block = {
            "sample_meta": {k: v for k, v in sm.items() if k != "families_for_sizing"},   # not needed for sizing
            "status": s["status"], "sample_records": len(sample), "final_outcomes": n,
            "unfetched": len(sample) - n,
            "by_outcome": s["by_outcome"],
            "outcome_shares": {k: _r(v / n) for k, v in sorted(Counter(o["outcome"] for o in final).items())} if n else {},
            "gated": n_g,
            "body_source": dict(Counter((o.get("inputs") or {}).get("body_source") or "not_read" for o in final)),
            "claims_by_kind_and_verdict": s["claims_by_kind_and_verdict"],
            "withheld_by_overlay": s["withheld_by_overlay"],
            "accusations": total_acc, "accused_records": any_rec,
            "accusations_per_accused_record": [min(per_rec_acc), max(per_rec_acc)] if per_rec_acc else None,
            "per_kind": per_kind, "any_kind": any_row,
            "mean_accusations_per_accused_record": _r(mean_per_accused),
            "projection": proj, "fetch_budget": budget,
            "gate": s.get("gate"), "styxx_closure_sha256": s.get("styxx_closure_sha256"),
            "outcomes_sha256": s["outcomes_sha256"], "accusations_sha256": s["accusations_sha256"],
            "fetched_between": [min((f["at"] for o in final for f in o.get("fetch", [])), default=None),
                                max((f["at"] for o in final for f in o.get("fetch", [])), default=None)],
            "outcomes_by_opened_day": {d: dict(sorted(c.items())) for d, c in sorted(by_day.items())},
            "head_check_to_diff_seconds": {"gated": len(gaps_s), "median": statistics.median(gaps_s) if gaps_s else None,
                                           "max": max(gaps_s) if gaps_s else None,
                                           "over_60": sum(1 for g in gaps_s if g > 60),
                                           "over_600": sum(1 for g in gaps_s if g > 600)},
        }

    rep = {
        "schema": "styxx.sealed-ahead/sizing-week/v0",
        "written_at": utc_now(),
        "plan": {"path": "sizing/sizing_plan_2026_09_28.json", "lf_sha256": _lf_sha256(PLAN_PATH)},
        "script": {"path": "sizing/sizing.py", "lf_sha256": _lf_sha256(os.path.abspath(__file__))},
        "builder_commit": PLAN["builder"]["commit"],
        "params_sha256": common.params_sha256(params),
        "ceilings": ceil["hours"],
        "window": meta["window"],
        "archive": {
            "hours_ok": n_ok, "gaps_404": meta["gaps"], "open_hours": meta["open_hours"],
            "compressed_bytes": tot["compressed_bytes"], "decompressed_bytes": tot["decompressed_bytes"],
            "lines": tot["lines"], "events": tot["events"], "bad_json_lines": tot["bad_json_lines"],
            "event_types": dict(sorted(types_total.items())),
            "pull_request_events": tot["pull_request_events"], "pr_opened": tot["pr_opened"],
            "pr_actions": meta["fields_census"]["pr_actions"], "pr_body": meta["fields_census"]["pr_body"],
            "hours_without_push_events": sum(1 for r in ok_hours if not r["push_events"]),
        },
        "population": {
            "agent_opened_before_exclusions": tot["agent_opened"],
            "agent_opened_by_family_before_exclusions": dict(sorted(agent_opened.items())),
            "excluded": dict(sorted(excluded_total.items())),
            "admitted_before_dedup": meta["admitted_before_dedup"], "duplicates": meta["duplicates"],
            "pool_records": meta["records"], "families_for_sizing": meta["families_for_sizing"],
            "rule_hits": dict(sorted(rule_hits.items())),
            "pool_sha256": meta["pool_sha256"], "derivation_sha256": meta["derivation_sha256"],
            "admitted_per_hour": {"min": min(adm_per_hour) if adm_per_hour else None,
                                  "median": statistics.median(adm_per_hour) if adm_per_hour else None,
                                  "max": max(adm_per_hour) if adm_per_hour else None},
            "scaled_to_168_hours": round(meta["records"] * week_scale) if week_scale else None,
            "distinct_actors": len(by_actor), "distinct_repos": len(by_repo),
            "distinct_actors_by_family": dict(sorted(actors_by_fam.items())),
            "distinct_repos_by_family": dict(sorted(repos_by_fam.items())),
            "records_per_repo": {"median": statistics.median(by_repo.values()) if by_repo else None,
                                 "max": max(by_repo.values()) if by_repo else None,
                                 "repos_with_more_than_10": sum(1 for c in by_repo.values() if c > 10)},
            "same_repository_share": _r(same_repo / len(records)) if records else None,
            "records_kept_under_caps": caps,
            "age_rule_sensitivity_before_dedup": {k: {"admitted": sum(v.values()), "by_family": dict(sorted(v.items())),
                                                      "excluded": dict(sorted(alt_exc.get(k, Counter()).items()))}
                                                  for k, v in sorted(alt_adm.items())},
        },
        "identity_evidence": {
            "login_rules_observed_ids": login_rule_ids,
            "prefix_case_variants_not_admitted": dict(sorted(case_var.items())),
            "candidate_prefixes_not_admitted": dict(sorted(cand.items())),
            "unmatched_logins_containing_an_agent_word": dict(sorted(near.items())),
            # Only the count: many such app logins embed a person's or an organization's name, so the
            # list is not written. The ones shaped exactly '<agent word>[bot]' are named, without counts,
            # as candidates for the identity table's E4 review.
            "unmatched_app_logins_containing_an_agent_word_distinct": len(bot_logins),
            "unmatched_app_logins_shaped_word_bot": sorted(l for l in bot_logins
                                                           if l.lower() in {w + "[bot]" for w in AGENT_WORDS}),
        },
        "per_hour": per_hour,
        "sample": sample_block,
    }
    write_json(REPORT_PATH, rep)
    log(f"report written: {REPORT_PATH}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ceilings", "derive", "sample", "report"):
        sub.add_parser(name)
    ac = sub.add_parser("accuse")
    ac.add_argument("--passes", type=int, default=3)
    a = ap.parse_args(argv)
    return {"ceilings": cmd_ceilings, "derive": cmd_derive, "sample": cmd_sample, "accuse": cmd_accuse,
            "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
