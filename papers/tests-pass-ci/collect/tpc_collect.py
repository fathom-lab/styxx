# -*- coding: utf-8 -*-
"""tests-pass-ci collector (PREREG_tests_pass_ci_2026_11_02, Part II P2-P4, P5, P6, P17, P18, P21).

    py -3.12 tpc_collect.py run                      catch-up: frames of finished window days, then the
                                                 draw once the beacon is stored. No body, no CI.
    py -3.12 tpc_collect.py run --fetch-ci           the same, then every drawn row due for reading
                                                 (72 hours after it opened). Window rows are refused
                                                 by SEAL_GUARD until the SEALED file names the
                                                 PREREG's signing commit.
    py -3.12 tpc_collect.py run --dry-run [--counts-only]
                                                 count window pull requests by the seal's metadata
                                                 (number, repository, stars, author login,
                                                 created_at). Writes counts only, never rows.
    py -3.12 tpc_collect.py status                   counts per day and agent, and K1's count so far
    py -3.12 tpc_collect.py verify                   re-hash every file the manifest names
    py -3.12 tpc_collect.py digest                   the code digests the lock table's collector row holds
    py -3.12 tpc_collect.py beacon --height 975000 --rule "..." --go
                                                 read the block from two explorers (OD-8: only on
                                                 the operator's word) and store draw/beacon.json

Development (pull requests opened before 2026-11-02 only, in a separate store):

    py -3.12 tpc_collect.py run --dev --days 2026-10-29 --dev-beacon LABEL --fetch-ci --max-reads 120

The run is idempotent: whatever is stored is never fetched again, whatever is missing is fetched.
A run that hits a limit it should not wait out stops cleanly; the next run continues. One run at a
time (run.lock). Every API call is in readlogs/; every file in manifest.jsonl; every run in runs.jsonl.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tpc_beacon  # noqa: E402
from tpc_const import (AGENT_ORDER, BEACON_HEIGHT, CAP, DATA_ROOT, DEV_ROOT, FRAME_FIELDS, READ_LAG, UTC,  # noqa: E402
                       WINDOW_DAYS, WINDOW_START, day_start, iso, parse_utc)
from tpc_draw import membership  # noqa: E402
from tpc_frame import FRAME_KEYS, dedupe, dry_run, enumerate_day, frame_row, kept  # noqa: E402
from tpc_gh import BudgetStop, GitHub, GitHubError  # noqa: E402
from tpc_read import BATCH, DiskGate, fetch_pr_nodes, read_row  # noqa: E402
from tpc_reports import evidence_reader  # noqa: E402
from tpc_seal import SealGuard, SealRefused  # noqa: E402
from tpc_seal import code_digests as _code_digests  # noqa: E402
from tpc_store import Store, dumps  # noqa: E402

# The lock's mtime is a heartbeat: every API call (and every wait, logged before it sleeps)
# touches it. A lock untouched for LOCK_STALE belongs to a run that died; waits are capped below it.
LOCK_STALE = dt.timedelta(hours=2)
MAX_WAIT_CAP = 5400.0
# styxx for P18's report reader: the repository's own package once the collector lives in
# papers/tests-pass-ci/collect/; until then the scouts' `git archive 310df453` copy.
_REPO_ROOT = HERE.parent.parent.parent
STYXX_FOR_EVIDENCE = (_REPO_ROOT if (_REPO_ROOT / "styxx" / "evidence.py").exists()
                      else Path("C:/Users/heyzo/clawd/wt/_direction_a_2026_11_02/_scout/styxx_310df453"))


def NOW() -> dt.datetime:
    """The clock, in one place (tests replace it)."""
    return dt.datetime.now(UTC)


def code_digests() -> dict:
    """The collector's code digests (tpc_seal.code_digests): the combined one fills the lock table's
    collector row at the signing commit, and the seal guard opens only for that code."""
    return _code_digests(HERE)


class Lock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.held = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            age = time.time() - self.path.stat().st_mtime
            if age < LOCK_STALE.total_seconds():
                raise SystemExit(f"another run holds {self.path} (age {age:.0f}s)")
            self.path.unlink()
        fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {iso(NOW())}\n".encode("utf-8"))
        os.close(fd)
        self.held = True
        self._last_touch = 0.0
        return self

    def touch(self) -> None:
        now = time.time()
        if self.held and now - self._last_touch > 30:
            try:
                os.utime(self.path, None)
            except OSError:
                pass
            self._last_touch = now

    def __exit__(self, *exc):
        if self.held and self.path.exists():
            self.path.unlink()
        return False


class Ctx:
    def __init__(self, args) -> None:
        self.args = args
        self.dev = bool(getattr(args, "dev", False))
        root = Path(args.root) if args.root else (DEV_ROOT if self.dev else DATA_ROOT)
        if self.dev and root.resolve() == DATA_ROOT.resolve():
            raise SystemExit("--dev refuses the window store; use a separate --root")
        if not self.dev and root.resolve() == DEV_ROOT.resolve():
            raise SystemExit("the development store is not the window store; pass --dev")
        self.store = Store(root)
        self.now = NOW()
        self.stage = "run"
        self.code = code_digests()
        self.guard = SealGuard()
        self._gh = None
        self.counts: dict = {}
        self.lock = None

    def log(self, row: dict) -> None:
        day = NOW().strftime("%Y%m%d")
        self.store.append_jsonl(f"readlogs/readlog_{self.stage}_{day}.jsonl", row)
        if self.lock is not None:
            self.lock.touch()

    @property
    def gh(self) -> GitHub:
        if self._gh is None:
            a = self.args
            self._gh = GitHub(self.log, max_wait=min(getattr(a, "max_wait", 3700.0), MAX_WAIT_CAP),
                              max_calls=getattr(a, "max_calls", None))
        return self._gh

    def say(self, msg: str) -> None:
        print(msg, flush=True)


# ---- the frame (P2) -----------------------------------------------------------------------------
def frame_days(ctx: Ctx) -> list:
    if ctx.dev:
        days = [dt.date.fromisoformat(d) for d in ctx.args.days.split(",") if d]
        for d in days:
            if day_start(d) + dt.timedelta(days=1) > WINDOW_START:
                raise SystemExit(f"--dev takes days before 2026-11-02 only: {d}")
        return days
    return [d for d in WINDOW_DAYS if ctx.now >= day_start(d) + dt.timedelta(days=1)]


def stage_frame(ctx: Ctx) -> None:
    ctx.stage = "frame"
    st = ctx.store
    for d in frame_days(ctx):
        rel = f"frames/frame_{d.isoformat()}.jsonl"
        if st.exists(rel):
            continue
        started = NOW()
        due_from = day_start(d) + dt.timedelta(days=1)
        late = started > due_from + dt.timedelta(hours=12)
        ctx.say(f"frame {d}: enumerating{' (late)' if late else ''}")
        by_agent, queries = enumerate_day(ctx.gh, d, fields=FRAME_FIELDS, workers=ctx.args.workers,
                                          slice_limit=ctx.args.slice_limit if ctx.dev else None,
                                          progress=ctx.say if ctx.args.verbose else None)
        rows = [frame_row(n, a) for a in AGENT_ORDER for n in by_agent.get(a, [])]
        for r in rows:
            assert tuple(r.keys()) == FRAME_KEYS
        w = st.write(rel, b"".join(dumps(r) for r in rows))
        finished = NOW()
        per = {}
        for a in AGENT_ORDER:
            qa = [q for q in queries if q["agent"] == a]
            per[a] = {"nodes": len(by_agent.get(a, [])), "queries": len(qa),
                      "splits": sum(1 for q in qa if q["split"]),
                      "issueCount_enumerated": sum(q["issueCount"] for q in qa if not q["split"]),
                      "count_mismatch": sum(1 for q in qa if not q["split"] and q["issueCount"] != q["nodes"]),
                      "repeated_nodes": sum(q["nodes"] - q.get("distinct", q["nodes"]) for q in qa if not q["split"]),
                      "truncated": sum(1 for q in qa if q.get("truncated"))}
        st.append_jsonl("frames/frame_digests.jsonl", {
            "day": d.isoformat(), "file": w["path"], "sha256": w["sha256"], "bytes": w["bytes"], "rows": len(rows),
            "enumeration_started": iso(started), "enumeration_finished": iso(finished), "late": late,
            "dev": ctx.dev, "slice_limit": ctx.args.slice_limit if ctx.dev else None,
            "code": ctx.code["combined"], "per_agent": per, "queries": queries}, manifest=True)
        ctx.say(f"frame {d}: {len(rows)} nodes, {len(queries)} queries, sha256 {w['sha256'][:16]}")


# ---- the draw (P3) ------------------------------------------------------------------------------
def load_beacon(ctx: Ctx):
    rel = "draw/beacon.json"
    if not ctx.store.exists(rel):
        if ctx.dev and ctx.args.dev_beacon:
            ctx.store.write("draw/beacon.json", tpc_beacon.dumps(tpc_beacon.dev_beacon(ctx.args.dev_beacon)))
        else:
            return None
    b = ctx.store.read_json(rel)
    why = tpc_beacon.validate(b, official=not ctx.dev)
    if not why and not ctx.dev:  # the window draws only under the block section 3.1 names, read from both explorers
        why = "; ".join(tpc_beacon.named_problems(b)) or None
    if why:
        raise SystemExit(f"beacon refused: {why}")
    return b


def stage_draw(ctx: Ctx) -> None:
    ctx.stage = "draw"
    st = ctx.store
    beacon = load_beacon(ctx)
    if beacon is None:
        ctx.say("draw: no beacon stored yet (OD-8); frames wait")
        return
    bh = beacon["beacon_hex"]
    for fp in sorted(st.path("frames").glob("frame_20[0-9][0-9]-[0-9][0-9]-[0-9][0-9].jsonl")):
        day = fp.stem.replace("frame_", "")
        rel = f"draw/draw_{day}.jsonl"
        if st.exists(rel):
            continue
        frame = st.read_jsonl(f"frames/{fp.name}")
        keep = [r for r in frame if kept(r)]
        uniq, doubles = dedupe(keep, key=lambda r: r["pr_id"])
        drawn = []
        per = {a: {"frame_nodes": 0, "kept_nodes": 0, "kept_distinct": 0, "drawn": 0, "in_M": 0} for a in AGENT_ORDER}
        for r in frame:
            per[r["agent"]]["frame_nodes"] += 1
        for r in keep:
            per[r["agent"]]["kept_nodes"] += 1
        for r in uniq:
            per[r["agent"]]["kept_distinct"] += 1
            if r["pr_id"] is None:
                continue
            m = membership(bh, int(r["pr_id"]), r["agent"])
            if not m["in_D"]:
                continue
            flags = []
            if r["repo"] is None:
                flags.append("NULL_REPO")
            if r["author"] is None:
                flags.append("NULL_AUTHOR")
            drawn.append({"pr_id": r["pr_id"], "repo": r["repo"], "number": r["number"], "agent": r["agent"],
                          "created_at": r["created_at"], "stars": r["stars"], "author": r["author"],
                          "author_type": r["author_type"], "day": day, "u": m["u"], "v": m["v"],
                          "in_D": True, "in_M": m["in_M"], "flags": flags,
                          "matched": doubles.get(r["pr_id"], [r["agent"]])})
            per[r["agent"]]["drawn"] += 1
            per[r["agent"]]["in_M"] += int(m["in_M"])
        drawn.sort(key=lambda x: x["u"])
        w = st.write(rel, b"".join(dumps(x) for x in drawn))
        st.append_jsonl("draw/draw_digests.jsonl", {
            "day": day, "file": w["path"], "sha256": w["sha256"], "drawn": len(drawn),
            "frame_sha256": hashlib.sha256(fp.read_bytes()).hexdigest(), "beacon_hex": bh,
            "beacon_kind": beacon.get("kind"), "drawn_at": iso(NOW()),
            "double_matches": len(doubles), "per_agent": per, "code": ctx.code["combined"]}, manifest=True)
        ctx.say(f"draw {day}: {len(drawn)} drawn, {sum(1 for x in drawn if x['in_M'])} in M")


# ---- reading (P4) -------------------------------------------------------------------------------
def drawn_rows(st: Store) -> list:
    rows = []
    for fp in sorted(st.path("draw").glob("draw_20[0-9][0-9]-[0-9][0-9]-[0-9][0-9].jsonl")) if st.path("draw").exists() else []:
        rows += st.read_jsonl(f"draw/{fp.name}")
    return rows


def stage_read(ctx: Ctx) -> None:
    ctx.stage = "read"
    st = ctx.store
    beacon = load_beacon(ctx)
    if beacon is None:
        return
    beacon_read = parse_utc(beacon["read_at"])
    rows = [r for r in drawn_rows(st) if not st.exists(f"a2/rows/{r['pr_id']}.json")]
    due = [r for r in rows if ctx.now >= parse_utc(r["created_at"]) + READ_LAG]
    sealed = [r for r in due if SealGuard.in_window(r["created_at"])]
    if sealed and not ctx.guard.open:
        st.append_jsonl("a2/seal_refusals.jsonl", {"at": iso(ctx.now), "code": "SEAL_GUARD", "rows": len(sealed),
                                                   "why": ctx.guard.state.get("why")})
        ctx.say(f"SEAL_GUARD: {len(sealed)} due window rows not read: {ctx.guard.state.get('why')}")
        due = [r for r in due if not SealGuard.in_window(r["created_at"])]
    due.sort(key=lambda r: (parse_utc(r["created_at"]), r["u"]))
    if ctx.args.max_reads is not None:
        due = due[: ctx.args.max_reads]
    if not due:
        ctx.say("read: nothing due")
        return
    disk = DiskGate()
    evidence = evidence_reader(STYXX_FOR_EVIDENCE if STYXX_FOR_EVIDENCE.exists() else None)
    ctx.counts["read"] = {"due": len(due), "done": 0, "errors": 0}
    try:
        _read_batches(ctx, st, due, beacon_read, disk, evidence)
    finally:
        if st.exists("a2/rows_index.jsonl"):
            st.seal_log("a2/rows_index.jsonl")


def _read_batches(ctx: Ctx, st: Store, due: list, beacon_read, disk, evidence) -> None:
    counts = ctx.counts["read"]
    for i in range(0, len(due), BATCH):
        batch = due[i:i + BATCH]
        now = NOW()
        attempt = now.strftime("%Y%m%dT%H%M%SZ")
        try:
            nodes = fetch_pr_nodes(ctx.gh, st, ctx.guard, batch, attempt)
        except GitHubError as e:  # the whole batch is retried by the next run
            counts["errors"] += len(batch)
            st.append_jsonl("a2/read_errors.jsonl", {"at": iso(NOW()), "batch": [r["pr_id"] for r in batch],
                                                     "error": f"{type(e).__name__}: {e}"[:500]})
            ctx.say(f"batch error: {e}")
            continue
        for r in batch:
            node, err = nodes[r["pr_id"]]
            backlog = parse_utc(r["created_at"]) < beacon_read
            try:
                rec = read_row(ctx.gh, st, ctx.guard, r, node, err, NOW(), attempt, disk,
                               backlog=backlog, evidence=evidence)
            except (BudgetStop, SealRefused):
                raise
            except Exception as e:  # noqa: BLE001
                counts["errors"] += 1
                st.append_jsonl("a2/read_errors.jsonl", {"at": iso(NOW()), "pr_id": r["pr_id"],
                                                         "error": f"{type(e).__name__}: {e}"[:500],
                                                         "trace": traceback.format_exc()[-2000:]})
                ctx.say(f"read error {r['repo']}#{r['number']}: {type(e).__name__}: {e}")
                continue
            w = st.write_json(f"a2/rows/{r['pr_id']}.json", rec)
            st.append_jsonl("a2/rows_index.jsonl", {"pr_id": r["pr_id"], "day": r["day"], "agent": r["agent"],
                                                    "read_at": rec["read_at"], "sha256": w["sha256"],
                                                    "eligible": rec["eligible"]})
            counts["done"] += 1
            ci = rec.get("ci") or {}
            ctx.say(f"read {counts['done']}/{len(due)} {r['agent']} excl={rec['exclusions']} screen={rec.get('screen')} "
                    f"ci={ci.get('outcome')} logs={sum(1 for x in rec.get('logs', []) if x.get('fetched'))} "
                    f"calls={sum(rec['requests'].values())} {rec['secs']}s")


# ---- status ---------------------------------------------------------------------------------------
def capped(rows: list, cap: int = CAP) -> list:
    """Section 9.2: at most `cap` rows per repository (case-folded), those with the lowest v.

    v = u / p_M(agent) is uniform on [0, 1) inside M for every agent, so the cap in M favours no
    agent; inside one agent's set A_g, ordering by v is ordering by u."""
    by_repo: dict = {}
    for r in sorted(rows, key=lambda x: (x["v"], x["u"])):
        k = (r.get("repo") or "").lower()
        by_repo.setdefault(k, [])
        if len(by_repo[k]) < cap:
            by_repo[k].append(r)
    return [r for v in by_repo.values() for r in v]


def status(ctx: Ctx) -> dict:
    st = ctx.store
    out: dict = {"root": str(st.root),
                 "frames": [{k: d[k] for k in ("day", "rows", "sha256", "late", "dev")}
                            for d in st.read_jsonl("frames/frame_digests.jsonl")]}
    out["draws"] = [{k: d[k] for k in ("day", "drawn", "sha256", "beacon_kind")} for d in st.read_jsonl("draw/draw_digests.jsonl")]
    drawn = drawn_rows(st)
    recs = []
    for r in drawn:
        p = st.path(f"a2/rows/{r['pr_id']}.json")
        if p.exists():
            recs.append(json.loads(p.read_bytes().decode("utf-8")))
    from collections import Counter
    out["drawn"] = {"D": len(drawn), "M": sum(1 for r in drawn if r["in_M"]),
                    "per_agent_D": dict(Counter(r["agent"] for r in drawn)),
                    "per_agent_M": dict(Counter(r["agent"] for r in drawn if r["in_M"]))}
    out["read"] = len(recs)
    out["exclusions"] = dict(Counter(x for r in recs for x in r["exclusions"]))
    elig = [r for r in recs if r["eligible"]]
    out["eligible"] = len(elig)
    out["screen"] = dict(Counter(r.get("screen") for r in elig))
    out["ci_outcome"] = dict(Counter((r.get("ci") or {}).get("outcome") for r in elig))
    claim_ci = [r for r in elig if r.get("screen") == "claimer" and (r.get("ci") or {}).get("has_ci_test_result")]
    out["claimers_with_ci"] = {"D": len(claim_ci), "M": sum(1 for r in claim_ci if r["in_M"]),
                               "M_capped": len(capped([r for r in claim_ci if r["in_M"]])),
                               "per_agent_D": dict(Counter(r["agent"] for r in claim_ci))}
    out["candidates_failed"] = sum(1 for r in claim_ci if r["ci"]["outcome"] == "tests failed")
    out["implied_only"] = sum(1 for r in elig if r.get("screen") == "implied_only")
    out["logs"] = {"rows_with_logs": sum(1 for r in recs if any(x.get("fetched") for x in r.get("logs", []))),
                   "logs_fetched": sum(1 for r in recs for x in r.get("logs", []) if x.get("fetched")),
                   "gz_bytes": sum(x.get("gz_bytes", 0) for r in recs for x in r.get("logs", [])),
                   "truncated": sum(1 for r in recs for x in r.get("logs", []) if x.get("truncated_at_log_max")),
                   "not_served": sum(1 for r in recs for x in r.get("logs", []) if x.get("fetched") is False)}
    out["flags"] = dict(Counter(f for r in recs for f in r.get("flags", [])))
    out["rule_r_fires"] = sum(1 for r in recs if (r.get("rule_r") or {}).get("fires"))
    out["report_artifacts"] = {"rows_with_counted": sum(1 for r in recs if r.get("report_artifact_counted")),
                               "artifacts_downloaded": sum(1 for r in recs for a in r.get("artifacts", []) if a.get("path"))}
    out["requests"] = {"rows": len(recs), "calls": sum(sum(r.get("requests", {}).values()) for r in recs)}
    out["read_errors"] = len(st.read_jsonl("a2/read_errors.jsonl"))
    return out


# ---- CLI ------------------------------------------------------------------------------------------
def cmd_run(args) -> int:
    ctx = Ctx(args)
    st = ctx.store
    with Lock(st.root / "run.lock") as lock:
        ctx.lock = lock
        started = NOW()
        run = {"started": iso(started), "mode": "dry-run" if args.dry_run else ("fetch-ci" if args.fetch_ci else "catch-up"),
               "dev": ctx.dev, "argv": sys.argv[1:], "code": ctx.code, "seal": ctx.guard.state}
        stop = None
        try:
            if args.dry_run:
                ctx.stage = "dryrun"
                counts = dry_run(ctx.gh, ctx.now, counts_only=args.counts_only, workers=args.workers,
                                 progress=ctx.say if args.verbose else None)
                counts["code"] = ctx.code["combined"]
                counts["requests"] = ctx.gh.c.calls
                w = st.write_json(f"dryrun/dryrun_{started.strftime('%Y%m%dT%H%M%SZ')}.json", counts)
                run["dryrun_file"] = w["path"]
                run["dryrun_sha256"] = w["sha256"]
                ctx.say(json.dumps({d: {"search_total": v["search_total"], "stars_100_distinct": v.get("stars_100_distinct")}
                                    for d, v in counts["days"].items()}, indent=1))
            else:
                stages = args.stages.split(",")
                if "frame" in stages:
                    stage_frame(ctx)
                if "draw" in stages:
                    stage_draw(ctx)
                if "read" in stages and args.fetch_ci:
                    stage_read(ctx)
        except BudgetStop as e:
            stop = f"BudgetStop: {e}"
            ctx.say(stop)
        except SealRefused as e:
            stop = str(e)
            ctx.say(stop)
        finally:
            run["finished"] = iso(NOW())
            run["stop"] = stop
            run["counts"] = ctx.counts
            if ctx._gh is not None:
                run["calls"] = ctx._gh.c.calls
                run["bytes"] = ctx._gh.c.bytes
                run["waited_secs"] = round(ctx._gh.c.waited_secs, 1)
            st.append_jsonl("runs.jsonl", run, manifest=True)
            for p in sorted(st.path("readlogs").glob("*.jsonl")) if st.path("readlogs").exists() else []:
                st.seal_log(f"readlogs/{p.name}")
    return 0 if stop is None or stop.startswith("BudgetStop") else 2


def cmd_status(args) -> int:
    ctx = Ctx(args)
    print(json.dumps(status(ctx), indent=1))
    return 0


def cmd_verify(args) -> int:
    ctx = Ctx(args)
    res = ctx.store.verify_manifest()
    print(json.dumps(res, indent=1))
    return 0 if not res["differ"] and not res["missing"] else 1


def cmd_digest(args) -> int:
    """The combined digest the lock table's collector row holds at the signing commit (section 14)."""
    print(json.dumps(code_digests(), indent=1))
    return 0


def cmd_beacon(args) -> int:
    if not args.go:
        print("the beacon is read only on the operator's word (OD-8): pass --go when it has been given")
        return 2
    ctx = Ctx(args)
    if ctx.store.exists("draw/beacon.json"):
        print("a beacon is already stored; it is never replaced")
        return 2
    if args.height != BEACON_HEIGHT and not (ctx.dev or args.pr_opened_at):
        print(f"section 3.1 names block {BEACON_HEIGHT}; another height is read only under the late rule, with "
              "--pr-opened-at (the time the pull request carrying the PREREG opened)")
        return 2
    b = tpc_beacon.read_beacon(args.height, args.rule)
    if args.pr_opened_at:
        b["pr_opened_at"] = args.pr_opened_at
    probs = [] if ctx.dev else tpc_beacon.named_problems(b)
    if probs:
        print("beacon refused, not stored: " + "; ".join(probs))
        return 2
    ctx.store.write("draw/beacon.json", tpc_beacon.dumps(b))
    print(json.dumps({k: b[k] for k in ("height", "beacon_hex", "header_time", "rule")}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", help="store root (default: the window store, or the dev store with --dev)")
    common.add_argument("--dev", action="store_true", help="development on pull requests opened before 2026-11-02")
    r = sub.add_parser("run", parents=[common])
    r.add_argument("--fetch-ci", action="store_true", help="read bodies, CI, logs and artifacts of due rows")
    r.add_argument("--dry-run", action="store_true", help="count window pull requests by metadata only")
    r.add_argument("--counts-only", action="store_true", help="with --dry-run: issueCount per signature only")
    r.add_argument("--stages", default="frame,draw,read")
    r.add_argument("--max-reads", type=int, default=None)
    r.add_argument("--max-calls", type=int, default=None)
    r.add_argument("--max-wait", type=float, default=3700.0)
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--days", default="", help="--dev only: comma-separated days before 2026-11-02")
    r.add_argument("--dev-beacon", default=None, help="--dev only: label for a development beacon")
    r.add_argument("--slice-limit", type=int, default=None, help="--dev only: ranges per agent")
    r.add_argument("--verbose", action="store_true")
    sub.add_parser("status", parents=[common])
    sub.add_parser("verify", parents=[common])
    sub.add_parser("digest", help="print the code digests the lock table's collector row holds")
    b = sub.add_parser("beacon", parents=[common])
    b.add_argument("--height", type=int, required=True)
    b.add_argument("--rule", required=True)
    b.add_argument("--go", action="store_true")
    b.add_argument("--pr-opened-at", default=None,
                   help="the late rule of section 3.1: when the pull request carrying the PREREG opened")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        if a.dry_run and a.fetch_ci:
            ap.error("--dry-run and --fetch-ci do not go together")
        if not a.dev and (a.days or a.dev_beacon or a.slice_limit is not None):
            ap.error("--days, --dev-beacon and --slice-limit are for --dev runs only")
        if a.dev and a.dry_run:
            ap.error("the dry run counts the window; it has no --dev form")
        return cmd_run(a)
    return {"status": cmd_status, "verify": cmd_verify, "beacon": cmd_beacon, "digest": cmd_digest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
