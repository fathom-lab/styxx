# -*- coding: utf-8 -*-
"""styxx.stranger — the checks a person who does not trust this lab runs, as one command.

    python -m styxx.stranger --repo . --expect-head <ferry-log head you were given>
    python -m styxx.stranger --repo . --only ferry_log,draw,reading --json stranger_report.json
    python -m styxx.stranger --repo . --with-tests --network        # the slow and the on-chain steps too

`papers/checksum/STRANGER.md` lists seven things a stranger checks and what each one proves. This
module runs them and prints one table — PASS / FAIL / SKIP per step, with the detail a dispute
needs — and writes a report (`styxx.stranger/report/v1`) naming the commit it ran on. It composes
the verifiers that already exist and adds no verdict of its own:

  checkout   the commit and whether the tree is dirty (a claim that names no commit is not a claim)
  tests      the suite (`--with-tests`; slow; SKIP by default)
  ferry_log  `styxx.charon.verify_log` on papers/charon/charon.log.jsonl against `--expect-head`
             (without it: internal consistency only, and the report says so)
  sworn      every `*.sworn-receipt.json` in the tree re-checked by `python -m styxx.sworn check`
             (VERIFIED = digest matches, the verdict reproduces, same verifier build)
  seals      `styxx.clock.verify` on papers/charon/anchors.jsonl (`--network`; SKIP when the file
             does not exist — no anchor exists yet — or the network is off)
  draw       every certs file under papers/checksum that carries a draw record: the beacon and the
             committed pool must produce exactly the canaries the certs name
  reading    every certs file whose `prereg` names a frozen PREREG the scorer knows, read by
             papers/checksum/score.py; a committed scorecard beside it must equal what the scorer
             reads today
  recipe     SKIP with the command (needs a model stack; see STRANGER.md §7)

Exit code 0 when no step FAILED. SKIP is not a pass and the table says why.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "styxx.stranger/report/v1"
STEPS = ("checkout", "tests", "ferry_log", "sworn", "seals", "draw", "reading", "recipe")


def _git(repo, *args):
    try:
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def step_checkout(repo: Path) -> dict:
    head = _git(repo, "rev-parse", "HEAD")
    dirty = _git(repo, "status", "--porcelain")
    if head is None:
        return {"status": "FAIL", "detail": "not a git checkout; a claim that names no commit is not a claim"}
    return {"status": "PASS" if not dirty else "PASS", "commit": head, "dirty": bool(dirty),
            "detail": f"commit {head[:12]}" + (" — the tree is DIRTY; what you check is not what is committed" if dirty else "")}


def step_tests(repo: Path, with_tests: bool) -> dict:
    if not with_tests:
        return {"status": "SKIP", "detail": "pass --with-tests to run `python -m pytest tests -q` (minutes)"}
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"], cwd=str(repo),
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    return {"status": "PASS" if r.returncode == 0 else "FAIL", "detail": tail, "seconds": round(time.time() - t0)}


def step_ferry_log(repo: Path, expect_head: str | None) -> dict:
    from . import charon
    log = repo / "papers" / "charon" / "charon.log.jsonl"
    if not log.exists():
        return {"status": "FAIL", "detail": "papers/charon/charon.log.jsonl is missing"}
    rep = charon.verify_log(log, repo, expect_head)
    by = rep.get("by_status") or {}
    bad = {k: v for k, v in by.items() if v and k not in ("SAME_LINE", "MOVED_VERIFIER")}
    ok = not rep.get("chain_problems") and rep.get("chain_broken_at_line") is None and not bad and rep.get("head_matches") is not False
    detail = f"{len(rep.get('lines') or [])} lines, head {str(rep.get('head'))[:16]}…, " + \
             ", ".join(f"{k} {v}" for k, v in by.items() if v)
    if expect_head is None:
        detail += " — no external head given: internal consistency only"
    elif rep.get("head_matches"):
        detail += " — head matches the one you were given"
    else:
        detail += " — HEAD MISMATCH: a truncated or rebuilt log"
    return {"status": "PASS" if ok else "FAIL", "detail": detail, "head": rep.get("head"), "head_expected": expect_head,
            "head_matches": rep.get("head_matches"), "by_status": by, "chain_broken_at_line": rep.get("chain_broken_at_line"),
            "external_head_checked": expect_head is not None}


def check_receipt(repo: Path, rc: str) -> dict:
    """One receipt re-checked by `python -m styxx.sworn check <receipt> <target> --repo .`.

    The target is the `.sworn.json` sidecar beside the receipt when it exists: the sidecar carries the
    commit and the manifest binding, so the receipt re-derives exactly; handed the `.md` instead, a
    receipt whose spans cite a harness manifest (rungs r2-r4) reads UNRESOLVED on those spans and
    `check` prints FAILED — which is what a stranger running `check` by hand on the document would see,
    and why STRANGER.md says to pass the sidecar. Without a sidecar the receipt's own `document.name`
    is the target; the sworn-action samples name temporary files that are not in the tree, and those
    are reported as not checkable here, never as failures."""
    rel_rc = os.path.relpath(rc, repo).replace("\\", "/")
    try:
        name = (json.load(open(rc, encoding="utf-8")).get("document") or {}).get("name") or ""
    except Exception as e:  # noqa: BLE001
        return {"receipt": rel_rc, "status": "FAIL", "detail": f"unreadable receipt: {e}"}
    stem = rc[: -len(".sworn-receipt.json")]
    sidecar = stem + ".sworn.json"
    if os.path.exists(sidecar):
        target = sidecar
    elif name and os.path.exists(os.path.join(os.path.dirname(rc), name)):
        target = os.path.join(os.path.dirname(rc), name)
    else:
        return {"receipt": rel_rc, "status": "SKIP", "detail": f"its target {name!r} is not in the tree and it has no sidecar (a sample issued against a temporary file)"}
    rel_target = os.path.relpath(target, repo).replace("\\", "/")
    r = subprocess.run([sys.executable, "-m", "styxx.sworn", "check", rel_rc, rel_target, "--repo", "."], cwd=str(repo),
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    line = (r.stdout.strip().splitlines() or [r.stderr.strip()[-200:]])[-1]
    ok = r.returncode == 0 and line.startswith("VERIFIED") and "verdict-reproduces=True" in line
    return {"receipt": rel_rc, "target": rel_target, "status": "PASS" if ok else "FAIL", "detail": line}


def step_sworn(repo: Path, limit: int | None = None) -> dict:
    receipts = sorted(glob.glob(str(repo / "papers" / "**" / "*.sworn-receipt.json"), recursive=True))
    if limit:
        receipts = receipts[:limit]
    rows = [check_receipt(repo, rc) for rc in receipts]
    fails = [r["receipt"] for r in rows if r["status"] == "FAIL"]
    skipped = sum(1 for r in rows if r["status"] == "SKIP")
    checked = len(receipts) - skipped
    older_build = sum(1 for r in rows if r["status"] == "PASS" and "same-build=False" in r.get("detail", ""))
    detail = f"{checked - len(fails)} of {checked} checkable receipts VERIFIED, {skipped} not checkable here"
    if older_build:
        detail += f" ({older_build} issued under an earlier verifier build and re-derived exactly under this one)"
    if fails:
        detail += f"; failing: {', '.join(fails)}"
    return {"status": "PASS" if checked and not fails else ("FAIL" if fails else "SKIP"), "detail": detail, "receipts": rows}


def step_seals(repo: Path, network: bool) -> dict:
    anchors = repo / "papers" / "charon" / "anchors.jsonl"
    if not anchors.exists():
        return {"status": "SKIP", "detail": "papers/charon/anchors.jsonl does not exist: no anchor exists yet (SEALS_2026_09_13.md lists what will be sealed)"}
    if not network:
        return {"status": "SKIP", "detail": "pass --network to verify the anchors against the chain (`python -m styxx.clock verify`)"}
    from . import clock
    rs = clock.verify(str(anchors))
    ok = bool(rs) and all(r.get("status") == "ANCHORED" for r in rs)
    return {"status": "PASS" if ok else "FAIL", "detail": "; ".join(f"#{r.get('n')} {r.get('kind')} {r.get('status')}" + (f" beacon={r['beacon'][:12]}…" if r.get("beacon") else "") for r in rs),
            "lines": rs}


def _certs_files(repo: Path):
    files = sorted(glob.glob(str(repo / "papers" / "checksum" / "*_certs*.json")))
    return [f for f in files if "_smoke" not in os.path.basename(f)]


def step_draw(repo: Path) -> dict:
    from . import beacon as _beacon
    from . import checksum as ck
    rows, fails, n_drawn = [], [], 0
    for f in _certs_files(repo):
        rel = os.path.relpath(f, repo).replace("\\", "/")
        try:
            certs = json.load(open(f, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            rows.append({"certs": rel, "status": "FAIL", "detail": f"unreadable: {e}"}); fails.append(rel); continue
        can = certs.get("canaries") or {}
        draw = can.get("draw") if isinstance(can, dict) else None
        if not draw:
            rows.append({"certs": rel, "status": "PASS", "detail": "no draw record: the hand-written set" + (f" ({str(can.get('canary_sha256'))[:12]}…)" if isinstance(can, dict) and can.get("canary_sha256") else "")})
            continue
        n_drawn += 1
        try:
            items = _beacon.select(draw["beacon"], int(draw["n"]))
            same_pool = draw.get("pool_sha256") == _beacon.pool_sha256()
            same_hash = ck.canary_sha256(items) == draw.get("canary_sha256") == can.get("canary_sha256")
            ok = same_pool and same_hash
            detail = f"beacon {draw['beacon'][:12]}… → {len(items)} items, canary hash {'re-derives' if same_hash else 'DOES NOT re-derive'}, pool {'matches' if same_pool else 'DOES NOT match'} this checkout"
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"the draw could not be re-derived: {e}"
        rows.append({"certs": rel, "status": "PASS" if ok else "FAIL", "detail": detail})
        if not ok:
            fails.append(rel)
    return {"status": "PASS" if rows and not fails else ("FAIL" if fails else "SKIP"),
            "detail": f"{n_drawn} beacon-drawn certs files re-derived, {len(rows) - n_drawn} hand-set files" + (f"; failing: {', '.join(fails)}" if fails else ""),
            "files": rows}


def _load_scorer(repo: Path):
    path = repo / "papers" / "checksum" / "score.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("styxx_stranger_score", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def step_reading(repo: Path) -> dict:
    scorer = _load_scorer(repo)
    if scorer is None:
        return {"status": "SKIP", "detail": "papers/checksum/score.py is not in this checkout"}
    by_file = {v["prereg"]: k for k, v in scorer.BANDS.items()}
    cards = {}
    for sc in glob.glob(str(repo / "papers" / "checksum" / "*_scorecard*.json")):
        try:
            c = json.load(open(sc, encoding="utf-8"))
            cards[c.get("certs_file")] = (os.path.relpath(sc, repo).replace("\\", "/"), c)
        except Exception:  # noqa: BLE001
            pass
    rows, fails = [], []
    for f in _certs_files(repo):
        rel = os.path.relpath(f, repo).replace("\\", "/")
        try:
            certs = json.load(open(f, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        key = by_file.get(certs.get("prereg"))
        if key is None:
            rows.append({"certs": rel, "status": "SKIP", "detail": f"prereg {certs.get('prereg')!r} is not one the scorer knows"})
            continue
        try:
            card = scorer.score(certs, key, expect_blob=None)
        except Exception as e:  # noqa: BLE001
            rows.append({"certs": rel, "status": "FAIL", "detail": f"the scorer raised: {e}"}); fails.append(rel); continue
        row = {"certs": rel, "prereg": key, "status": "PASS", "reading": card["run_reading"], "counts_as_result": card["counts_as_result"]}
        if rel in cards:
            sc_rel, committed = cards[rel]
            same = committed.get("hypotheses") == card["hypotheses"] and committed.get("gates") == card["gates"]
            row["committed_scorecard"] = sc_rel
            row["committed_scorecard_matches"] = same
            if not same:
                row["status"] = "FAIL"; fails.append(rel)
                row["detail"] = "the committed scorecard is not what the scorer reads today"
        rows.append(row)
    return {"status": "PASS" if rows and not fails else ("FAIL" if fails else "SKIP"),
            "detail": f"{len([r for r in rows if r['status'] != 'SKIP'])} certs files read against their PREREG; "
                      f"{sum(1 for r in rows if r.get('counts_as_result'))} count as a result; "
                      f"{sum(1 for r in rows if 'committed_scorecard' in r)} committed scorecards compared" + (f"; failing: {', '.join(fails)}" if fails else ""),
            "files": rows}


def step_recipe() -> dict:
    return {"status": "SKIP", "detail": "run `python papers/checksum/run_smollm_quant.py` (CPU, minutes) and compare with `python -m styxx.portability` — needs a model stack; STRANGER.md §7"}


def run(repo, expect_head: str | None = None, with_tests: bool = False, network: bool = False, only=None,
        sworn_limit: int | None = None) -> dict:
    repo = Path(repo).resolve()
    only = set(only) if only else set(STEPS)
    steps = {}
    t0 = time.time()
    for name in STEPS:
        if name not in only:
            steps[name] = {"status": "SKIP", "detail": "not selected (--only)"}
            continue
        t1 = time.time()
        if name == "checkout":
            s = step_checkout(repo)
        elif name == "tests":
            s = step_tests(repo, with_tests)
        elif name == "ferry_log":
            s = step_ferry_log(repo, expect_head)
        elif name == "sworn":
            s = step_sworn(repo, sworn_limit)
        elif name == "seals":
            s = step_seals(repo, network)
        elif name == "draw":
            s = step_draw(repo)
        elif name == "reading":
            s = step_reading(repo)
        else:
            s = step_recipe()
        s["seconds"] = round(time.time() - t1, 1)
        steps[name] = s
    failed = [n for n, s in steps.items() if s["status"] == "FAIL"]
    return {"schema": SCHEMA, "repo": str(repo), "commit": steps["checkout"].get("commit"), "dirty": steps["checkout"].get("dirty"),
            "expect_head": expect_head, "steps": steps, "failed": failed, "skipped": [n for n, s in steps.items() if s["status"] == "SKIP"],
            "verdict": "NOTHING FAILED" if not failed else "FAILED: " + ", ".join(failed), "seconds": round(time.time() - t0, 1)}


def render(rep: dict) -> str:
    out = [f"styxx.stranger — {rep['repo']}", f"  commit {rep.get('commit')}{'  (DIRTY)' if rep.get('dirty') else ''}", ""]
    for name, s in rep["steps"].items():
        out.append(f"  {s['status']:5s} {name:10s} {s.get('detail', '')}")
        for r in (s.get("receipts") or []):
            if r["status"] != "PASS":
                out.append(f"          {r['status']}  {r['receipt']}: {r['detail']}")
        for r in (s.get("files") or []):
            if name == "reading" and r["status"] != "SKIP":
                out.append(f"          {r['status']}  {r['certs']}: {r.get('reading', r.get('detail', ''))}")
            elif r["status"] == "FAIL":
                out.append(f"          FAIL  {r['certs']}: {r['detail']}")
    out += ["", f"  {rep['verdict']}  ({rep['seconds']}s)"]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m styxx.stranger")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--expect-head", default=None, help="the ferry-log head you were given outside the log")
    ap.add_argument("--with-tests", action="store_true")
    ap.add_argument("--network", action="store_true", help="verify the anchors against the chain")
    ap.add_argument("--only", default=None, help="comma-separated steps: " + ",".join(STEPS))
    ap.add_argument("--json", default=None, help="write the report here")
    a = ap.parse_args(argv)
    only = [s.strip() for s in a.only.split(",")] if a.only else None
    if only and any(s not in STEPS for s in only):
        ap.error(f"--only must name steps from {', '.join(STEPS)}")
    rep = run(a.repo, a.expect_head, a.with_tests, a.network, only)
    print(render(rep))
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(rep, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
        print(f"  report -> {a.json}")
    return 0 if not rep["failed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
