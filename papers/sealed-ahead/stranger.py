# -*- coding: utf-8 -*-
"""stranger — re-derive the pool, the accusations and the draw from public URLs; refuse on any mismatch.

    python stranger.py pool   --params P --published DIR [--expect-params-sha256 HEX]
    python stranger.py accuse --params P --published DIR --out MINE [--allow-compare] [--styxx-path PATH]
    python stranger.py draw   --params P --published DIR --out MINE --header HEX --tip-height N \\
                              [--coinbase-hex HEX --merkle-branch H1,H2,...] [--anchor-time ISO]
    python stranger.py bundle --params P --pool-dir D --accuse-dir D --draw FILE --out DIR   # the lab's side

A stranger needs the plan's parameters (whose sha256 the plan posted) and the lab's published
digests: pool_meta.json, outcome_digests.json and draw.json, assembled by `bundle` (which writes
files locally and posts nothing). Every stage prints the digests it derived beside the published
ones, writes its report, and exits 0 when they agree, 1 when it could not reproduce something
because GitHub changed after the lab read it, and 2 when the published record is contradicted.
Exit 1 and exit 2 are both refusals: the stranger does not accept the draw.

pool. The pool is re-derived from GH Archive with the published gap list replayed (a gap hour is
not read into the pool); then every published gap is fetched once more, and an hour that answers
now is reported with its admitted count and the pool digest it would have given. Per hour, the
admitted records' digest is compared, so a mismatch names its hour; an hour whose compressed bytes
differ but whose admitted records agree is reported and is not a mismatch (GH Archive has rebuilt
files before). An hour the lab read and the stranger cannot is a mismatch.

accuse. Every pool record is fetched again and compared with its published digest row:

  MATCH          the same outcome from the same inputs, or for a gated record the same title, body
                 and diff digests and the same claims digest
  DRIFT          the pull request changed after the lab read it (its updated_at is later than the
                 lab's fetch), so the description differs; GitHub's state moved, not the lab's record
  UNREPRODUCIBLE the stranger could not read what the lab read: the head moved after the lab's
                 fetch and the compare route was not allowed, or the pull request or its diff is
                 unavailable now
  INCONSISTENT   the published row is contradicted: equal inputs and a different claims digest
                 (the gate is not the plan's function), a diff at the event's commits that differs,
                 a description that differs though the pull request has not changed since the
                 lab's fetch, or a record the lab set aside (HEAD_MOVED, PR_GONE, EMPTY_BODY, ...)
                 that an unchanged pull request shows was readable: the pruning attack's trace

  Only when every row is MATCH can the accusation set be confirmed, because the published rows do
  not say which records were accused; the stranger's own set is then digested and must equal the
  published accusations_sha256.

draw. With the stranger's confirmed set and the block (header, tip, optional BIP34 proof), the
draw is recomputed by draw.checked_draw and must name the published ids in the published order.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_SCHEMA = "styxx.sealed-ahead/stranger/v0"
MANIFEST_SCHEMA = "styxx.sealed-ahead/bundle/v0"
SET_ASIDE = ("HEAD_MOVED", "PR_GONE", "PR_UNAVAILABLE", "EMPTY_BODY", "DIFF_GONE", "DIFF_UNAVAILABLE")
UNREADABLE_NOW = ("PR_GONE", "PR_UNAVAILABLE", "DIFF_GONE", "DIFF_UNAVAILABLE", "FETCH_FAILED")


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
pool = _sibling("pool")
accuse = _sibling("accuse")
draw = _sibling("draw")


def _exit_code(verdict: str) -> int:
    return {"AGREES": 0, "CANNOT_CONFIRM": 1}.get(verdict, 2)


# ---------------------------------------------------------------- pool

def check_pool(params: dict, table: dict, published: dict, archive, now=None, log=print) -> dict:
    gaps = list(published.get("gaps", []))
    mine = pool.derive(params, table, archive, now=now, skip_hours=set(gaps), log=log)
    meta = mine["meta"]
    pub_hours = {h["hour"]: h for h in published.get("hours", [])}
    hours_differ, bytes_differ = [], []
    for h in meta["hours"]:
        p = pub_hours.get(h["hour"])
        if p is None or p.get("status") != h["status"] or p.get("admitted_sha256") != h.get("admitted_sha256"):
            hours_differ.append({"hour": h["hour"], "mine": h["status"], "published": (p or {}).get("status"),
                                 "admitted_mine": h.get("admitted"), "admitted_published": (p or {}).get("admitted")})
        elif p.get("compressed_sha256") != h.get("compressed_sha256"):
            bytes_differ.append(h["hour"])
    now_available = []
    ctx = population.Context(params, table)
    for g in gaps:
        got = archive.fetch(g, lambda lines, g=g: population.scan_lines(lines, ctx, g))
        if got["status"] == "ok":
            entry = pool.hour_entry(g, "ok", got, got["result"])
            others = [e for e in mine["entries"] if e["hour"] != g] + [entry]
            others.sort(key=lambda e: common.parse_hour_name(e["hour"]))
            _, alt = pool.assemble(others, params)
            now_available.append({"hour": g, "admitted": entry["admitted"],
                                  "admitted_sha256": entry["admitted_sha256"],
                                  "pool_sha256_with_it": alt["pool_sha256"]})
    checks = {
        "params_sha256": meta["params_sha256"] == published.get("params_sha256"),
        "identity_rules_sha256": meta["identity_rules_sha256"] == published.get("identity_rules_sha256"),
        "complete": meta["status"] == "COMPLETE" and published.get("status") == "COMPLETE",
        "pool_sha256": meta["pool_sha256"] == published.get("pool_sha256"),
        "hours": not hours_differ,
    }
    verdict = "AGREES" if all(checks.values()) else "MISMATCH"
    return {"stage": "pool", "verdict": verdict, "checks": checks,
            "pool_sha256": meta["pool_sha256"], "published_pool_sha256": published.get("pool_sha256"),
            "records": meta["records"], "hours_differ": hours_differ, "compressed_bytes_differ": bytes_differ,
            "published_gaps": gaps, "gaps_now_available": now_available, "_records": mine["records"]}


# ---------------------------------------------------------------- accuse

def classify_row(lab: dict, mine: dict) -> tuple:
    """(category, reason) for one pool key: MATCH, DRIFT, UNREPRODUCIBLE or INCONSISTENT."""
    lo, mo = lab["outcome"], mine["outcome"]
    same_inputs = all(lab[k] == mine[k] for k in ("title_sha256", "body_sha256", "diff_sha256"))
    changed_since = (mine.get("pr_updated_at") is not None and lab.get("body_fetched_at") is not None
                     and common.parse_utc(mine["pr_updated_at"]) > common.parse_utc(lab["body_fetched_at"]))
    unchanged_since = (mine.get("pr_updated_at") is not None and lab.get("body_fetched_at") is not None
                       and not changed_since)
    if lo == "GATED":
        if mo == "GATED":
            if same_inputs:
                if lab["claims_sha256"] == mine["claims_sha256"]:
                    return "MATCH", ""
                return "INCONSISTENT", "the same title, body and diff gave other claims: not the plan's gate"
            if lab["diff_sha256"] != mine["diff_sha256"]:
                return "INCONSISTENT", "the diff at the event's commits differs from the one the lab gated"
            if changed_since:
                return "DRIFT", "the description changed after the lab read it"
            if unchanged_since:
                return "INCONSISTENT", "the description differs though the pull request is unchanged since the lab read it"
            return "UNREPRODUCIBLE", "the description differs and no update time orders it"
        if mo == "HEAD_MOVED":
            return "UNREPRODUCIBLE", "the head moved after the lab read it; re-run with --allow-compare"
        if mo == "EMPTY_BODY":
            return ("DRIFT", "the description was emptied after the lab read it") if changed_since else (
                "INCONSISTENT", "the description is empty and unchanged since the lab gated it")
        return "UNREPRODUCIBLE", f"now {mo}"
    if lo == mo:
        if lo == "GATE_ERROR" and not same_inputs:
            return "UNREPRODUCIBLE", "the inputs differ"
        return "MATCH", ""
    if mo in UNREADABLE_NOW:
        return "UNREPRODUCIBLE", f"the lab read {lo}; now {mo}"
    if lo == "FETCH_FAILED":
        return "UNREPRODUCIBLE", "the lab's run is not complete"
    desc_differs = lab["title_sha256"] != mine["title_sha256"] or lab["body_sha256"] != mine["body_sha256"]
    if desc_differs and changed_since:
        return "DRIFT", "the description changed after the lab read it"
    if lo == "HEAD_MOVED" and mo == "GATED":
        return "INCONSISTENT", "the lab set the record aside as HEAD_MOVED; its head is the event's"
    if lo in SET_ASIDE and mo == "GATED":
        if unchanged_since or lo in ("PR_GONE", "PR_UNAVAILABLE", "DIFF_GONE", "DIFF_UNAVAILABLE"):
            return "INCONSISTENT", f"the lab set the record aside as {lo}; it reads now"
        return "UNREPRODUCIBLE", f"the lab read {lo}; now GATED, and no update time orders the two"
    if desc_differs and not unchanged_since:
        return "UNREPRODUCIBLE", f"the lab read {lo}; now {mo}, and no update time orders the two"
    return "INCONSISTENT", f"the lab read {lo}; now {mo}, though the pull request is unchanged since"


def check_accuse(params: dict, records: list, published: dict, gate, gh, out_dir: str, allow_compare: bool,
                 log=print) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    lab_rows = {r["pool_key"]: r for r in published["rows"]}
    ledger = os.path.join(out_dir, "audit.jsonl")
    done = accuse.read_outcomes(ledger)
    order = sorted(records, key=lambda r: int(r["event_id"]))
    keys = [f"{r['repo_id']}#{r['number']}" for r in order]
    missing_rows = [k for k in keys if k not in lab_rows]
    extra_rows = sorted(set(lab_rows) - set(keys))
    fetch = dict(params["fetch"])
    for rec, k in zip(order, keys):
        if k not in lab_rows or done.get(k, {}).get("outcome") in accuse.FINAL:
            continue
        lab = lab_rows[k]
        if lab["outcome"] == "GATED" and allow_compare:
            o, _ = accuse.accuse_one(rec, gh, gate, dict(fetch, diff_source="compare"), require_unmoved_head=False)
        else:
            o, _ = accuse.accuse_one(rec, gh, gate, fetch, require_unmoved_head=True)
        with open(ledger, "ab") as fh:
            fh.write(common.canon(o) + b"\n")
        done[k] = o
    cats, listed = Counter(), []
    for k in keys:
        if k not in lab_rows or k not in done:
            continue
        cat, why = classify_row(lab_rows[k], accuse.digest_row(done[k]))
        cats[cat] += 1
        if cat != "MATCH":
            listed.append({"pool_key": k, "category": cat, "why": why, "lab": lab_rows[k]["outcome"],
                           "mine": done[k]["outcome"]})
    audited = sum(cats.values())
    mine_summary = accuse.summarize(records, done, gate.identity) if audited == len(keys) else None
    checks = {"every_record_has_a_published_row": not missing_rows and not extra_rows,
              "published_run_complete": published.get("status") == "COMPLETE",
              "gate_sha256": (published.get("gate") or {}).get("sha256") == params["gate"]["sha256"],
              "every_row_matches": audited == len(keys) and cats["MATCH"] == len(keys)}
    if checks["every_row_matches"]:
        checks["accusations_sha256"] = mine_summary["accusations_sha256"] == published["accusations_sha256"]
    if all(checks.values()):
        verdict = "AGREES"
    elif cats["INCONSISTENT"] or missing_rows or extra_rows or not checks["gate_sha256"] or (
            checks.get("accusations_sha256") is False):
        verdict = "MISMATCH"
    else:
        verdict = "CANNOT_CONFIRM"
    return {"stage": "accuse", "verdict": verdict, "checks": checks, "categories": dict(sorted(cats.items())),
            "listed": listed, "missing_rows": missing_rows[:50], "extra_rows": extra_rows[:50],
            "accusations_sha256": mine_summary["accusations_sha256"] if mine_summary else None,
            "published_accusations_sha256": published.get("accusations_sha256"),
            "_summary": mine_summary}


# ---------------------------------------------------------------- draw

def check_draw(params: dict, my_summary: dict, published_draw: dict, header_hex: str, tip_height: int,
               coinbase_hex=None, branch=None, anchor_time=None) -> dict:
    try:
        mine = draw.checked_draw(params, my_summary, header_hex, params["draw"]["bitcoin_height"], tip_height,
                                 None, coinbase_hex, branch, anchor_time)
    except draw.Refused as e:
        return {"stage": "draw", "verdict": "MISMATCH", "refused": e.code, "detail": e.detail}
    keys = ("status", "pool_sha256", "pool_size", "beacon", "n", "drawn", "drawn_sha256", "bitcoin_height")
    differ = [k for k in keys if mine.get(k) != published_draw.get(k)]
    return {"stage": "draw", "verdict": "AGREES" if not differ else "MISMATCH", "differ": differ,
            "drawn_sha256": mine.get("drawn_sha256"), "published_drawn_sha256": published_draw.get("drawn_sha256"),
            "beacon": mine.get("beacon"), "order": mine.get("order")}


# ---------------------------------------------------------------- the lab's bundle (written locally, posted nowhere)

def bundle(params_path: str, pool_dir: str, accuse_dir: str, draw_path, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    files = {"plan_params.json": params_path, "pool_meta.json": os.path.join(pool_dir, "pool_meta.json"),
             "outcome_digests.json": os.path.join(accuse_dir, "outcome_digests.json")}
    if draw_path:
        files["draw.json"] = draw_path
    manifest = {"schema": MANIFEST_SCHEMA, "files": {}}
    for name, src in files.items():
        shutil.copyfile(src, os.path.join(out_dir, name))
        with open(os.path.join(out_dir, name), "rb") as fh:
            manifest["files"][name] = common.sha256_hex(fh.read())
    with open(params_path, "rb") as fh:
        manifest["params_sha256"] = common.params_sha256(json.loads(fh.read().decode("utf-8")))
    with open(os.path.join(out_dir, "manifest.json"), "wb") as fh:
        fh.write((json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    return manifest


# ---------------------------------------------------------------- command line

def _read(path):
    with open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def _report(out_dir, section: dict) -> int:
    public = {k: v for k, v in section.items() if not k.startswith("_")}
    public["schema"] = REPORT_SCHEMA
    public["at"] = common.fmt_utc(datetime.now(tz=common.UTC))
    text = json.dumps(public, indent=1, ensure_ascii=False, sort_keys=True) + "\n"
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"stranger_{section['stage']}.json"), "wb") as fh:
            fh.write(text.encode("utf-8"))
    print(text)
    verdict = section["verdict"]
    print(("AGREES" if verdict == "AGREES" else f"REFUSED ({verdict})") + f": {section['stage']}")
    return _exit_code(verdict)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pool", "accuse", "draw"):
        p = sub.add_parser(name)
        p.add_argument("--params", required=True)
        p.add_argument("--published", required=True)
        p.add_argument("--out", default=None)
        p.add_argument("--expect-params-sha256", default=None)
        p.add_argument("--table", default=population.TABLE_PATH)
        if name in ("accuse", "draw"):
            p.add_argument("--styxx-path", default=None)
            p.add_argument("--allow-compare", action="store_true")
        if name == "draw":
            p.add_argument("--header", required=True)
            p.add_argument("--tip-height", type=int, required=True)
            p.add_argument("--coinbase-hex", default=None)
            p.add_argument("--merkle-branch", default="")
            p.add_argument("--anchor-time", default=None)
    b = sub.add_parser("bundle")
    b.add_argument("--params", required=True)
    b.add_argument("--pool-dir", required=True)
    b.add_argument("--accuse-dir", required=True)
    b.add_argument("--draw", default=None)
    b.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "bundle":
        print(json.dumps(bundle(a.params, a.pool_dir, a.accuse_dir, a.draw, a.out), indent=1))
        return 0
    params = common.load_params(a.params)
    if a.expect_params_sha256 and common.params_sha256(params) != a.expect_params_sha256:
        print(f"REFUSED: the parameters hash to {common.params_sha256(params)}, not {a.expect_params_sha256}")
        return 2
    table = population.load_identity_table(a.table)
    published_meta = _read(os.path.join(a.published, "pool_meta.json"))
    section = check_pool(params, table, published_meta, pool.HttpArchive())
    if a.cmd == "pool" or section["verdict"] != "AGREES":
        return _report(a.out, section)
    _report(a.out, section)
    gate = accuse.load_gate(params["gate"]["sha256"], a.styxx_path)
    lim = accuse.RateLimiter(params["fetch"]["rate_limit_per_hour"],
                             state_path=os.path.join(a.out or ".", "rate_stamps.txt"))
    sec = check_accuse(params, section["_records"], _read(os.path.join(a.published, "outcome_digests.json")),
                       gate, accuse.GitHub(lim), a.out or ".", a.allow_compare)
    if a.cmd == "accuse" or sec["verdict"] != "AGREES":
        return _report(a.out, sec)
    _report(a.out, sec)
    branch = [x for x in a.merkle_branch.split(",") if x]
    d = check_draw(params, sec["_summary"], _read(os.path.join(a.published, "draw.json")), a.header,
                   a.tip_height, a.coinbase_hex, branch, a.anchor_time)
    return _report(a.out, d)


if __name__ == "__main__":
    sys.exit(main())
