# -*- coding: utf-8 -*-
"""styxx.stranger — the checks a person who does not trust this lab runs, as one command.

    python -m styxx.stranger --repo . --expect-head <the full 64-hex ferry-log head you were given>
    python -m styxx.stranger --repo . --only ferry_log,draw,reading --json stranger_report.json   # + checkout, always
    python -m styxx.stranger --repo . --with-tests --network        # the slow and the on-chain steps too

`papers/checksum/STRANGER.md` lists seven things a stranger checks and what each one proves. This
module runs them and prints one table — PASS / FAIL / SKIP per step, with the detail a dispute
needs — and writes a report (`styxx.stranger/report/v2`) naming the commit it ran on. It composes
the verifiers that already exist and adds no verdict of its own:

  checkout   the commit, and whether tracked files differ from it: a modified tree FAILS unless
             --allow-dirty is given, because every step below reads the working tree and every
             claim is about the commit (a claim that names no commit is not a claim). It runs on
             every invocation, whether or not --only names it: before 2026-09-14 (verification of
             the night repairs, STR-2) `--only ferry_log,draw,reading` skipped it, so a dirty tree
             read NOTHING FAILED and the report named no commit
  tests      the suite (`--with-tests`; slow; SKIP by default)
  ferry_log  `styxx.charon.verify_log` on papers/charon/charon.log.jsonl against `--expect-head`,
             which must be the full 64-hex head (without one: internal consistency only, and the
             report says so)
  sworn      every `*.sworn-receipt.json` in the tree re-checked by `python -m styxx.sworn check`
             against its `.sworn.json` sidecar (VERIFIED = the receipt's digest matches and its
             verdict reproduces; `same-build` is reported, not required), AND the `.md` on disk
             compared byte for byte with the document the sidecar renders — `check` on a sidecar
             never opens the `.md`, so without this an edited document beside an untouched sidecar
             would read PASS. The document verdicts (SWORN-HELD, SWORN-FAILED, UNSWORN) are tallied:
             a receipt that re-derives a FAILED document is a PASS of the receipt, not of the document.
             A receipt whose sidecar has no `.md` beside it, and a check line that names no
             `document=` verdict, PASS with an info row: no document was compared.
  seals      `styxx.clock.verify` on papers/charon/anchors.jsonl (`--network`; SKIP when the file
             does not exist — no anchor exists yet — or the network is off)
  draw       every certs file under papers/checksum that carries a draw record: the beacon and the
             committed pool must produce exactly the canaries the certs name, and every fingerprint
             in the fingerprints file beside it must pass `checksum.check_draw_record`; a drawn
             certs file with no fingerprints file beside it (or an empty one) FAILS, because the
             draw cannot be checked against the ids
  reading    every certs file whose `prereg` names a frozen PREREG the scorer knows, read by
             papers/checksum/score.py. Beacon-drawn certs are read with the beacon the seals step
             prints on the ANCHORED `sealed-prereg` line for the scorer's beacon-draw digest; when
             that step did not run or prints no such line, no beacon-draw certs file counts as a
             result and the step says so. Every papers/checksum/*_scorecard*.json is accounted for:
             an unreadable card FAILS; two current-schema cards naming one certs file FAIL; a
             current-schema card naming no certs file this step reads FAILS; a card of another
             schema is listed as not compared; the one card naming a certs file must name these
             certs bytes and equal what the scorer reads today. An unreadable certs file FAILS.
  recipe     SKIP with the command (needs a model stack; see STRANGER.md §7)

Exit code 0 when no step FAILED. SKIP is not a pass and the table says why.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import posixpath
import re
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "styxx.stranger/report/v2"
STEPS = ("checkout", "tests", "ferry_log", "sworn", "seals", "draw", "reading", "recipe")
_HEAD = re.compile(r"[0-9a-f]{64}")


def _git(repo, *args):
    try:
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def step_checkout(repo: Path, allow_dirty: bool = False) -> dict:
    head = _git(repo, "rev-parse", "HEAD")
    if head is None:
        return {"status": "FAIL", "detail": "not a git checkout; a claim that names no commit is not a claim"}
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=no")
    if dirty and not allow_dirty:
        return {"status": "FAIL", "commit": head, "dirty": True,
                "detail": f"commit {head[:12]} — tracked files differ from the commit, so every step below would check bytes the "
                          "commit does not carry; commit or stash them, or pass --allow-dirty to check the working tree knowingly"}
    return {"status": "PASS", "commit": head, "dirty": bool(dirty),
            "detail": f"commit {head[:12]}" + (" — DIRTY, checked knowingly (--allow-dirty): what is checked is not what is committed" if dirty else "")}


def step_tests(repo: Path, with_tests: bool) -> dict:
    if not with_tests:
        return {"status": "SKIP", "detail": "pass --with-tests to run `python -m pytest tests -q` (minutes)"}
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"], cwd=str(repo),
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    return {"status": "PASS" if r.returncode == 0 else "FAIL", "detail": tail, "seconds": round(time.time() - t0)}


def step_ferry_log(repo: Path, expect_head: str | None) -> dict:
    if expect_head is not None and not _HEAD.fullmatch(expect_head.strip().lower()):
        return {"status": "FAIL", "head_expected": expect_head, "external_head_checked": False,
                "detail": f"--expect-head must be the full 64-hex head; {expect_head!r} is not (`charon status` prints a prefix — "
                          "use the head the REPORT, the pull request or `charon verify` prints)"}
    from . import charon
    log = repo / "papers" / "charon" / "charon.log.jsonl"
    if not log.exists():
        return {"status": "FAIL", "detail": "papers/charon/charon.log.jsonl is missing"}
    rep = charon.verify_log(log, repo, expect_head.strip().lower() if expect_head else None)
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
            "external_head_checked": expect_head is not None,
            "covers": "each line's sidecar re-derived at the line's commit; the working-tree .md is the sworn step's to compare"}


def check_receipt(repo: Path, rc: str) -> dict:
    """One receipt re-checked by `python -m styxx.sworn check <receipt> <target> --repo .`, and the document on
    disk compared with the document the receipt swore to.

    The target is the `.sworn.json` sidecar beside the receipt when it exists: the sidecar carries the
    commit and the manifest binding, so the receipt re-derives exactly; handed the `.md` instead, a
    receipt whose spans cite a harness manifest reads UNRESOLVED on those spans and `check` prints
    FAILED. But `check` on a sidecar renders the document from the sidecar and never opens the `.md`,
    so the `.md` beside it is compared here, byte for byte, with `sworn.render(sidecar)`: a document
    edited after its receipt FAILS. Without a sidecar the receipt's own `document.name` is the target;
    the sworn-action samples name temporary files that are not in the tree, and those are reported as
    not checkable here, never as failures."""
    rel_rc = os.path.relpath(rc, repo).replace("\\", "/")
    try:
        name = (json.load(open(rc, encoding="utf-8")).get("document") or {}).get("name") or ""
    except Exception as e:  # noqa: BLE001
        return {"receipt": rel_rc, "status": "FAIL", "detail": f"unreadable receipt: {e}"}
    stem = rc[: -len(".sworn-receipt.json")]
    sidecar, md = stem + ".sworn.json", stem + ".md"
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
    m = re.search(r"document=(\S+)", line)
    row = {"receipt": rel_rc, "target": rel_target, "document_verdict": m.group(1) if m else None, "detail": line}
    ok = r.returncode == 0 and line.startswith("VERIFIED") and "verdict-reproduces=True" in line
    if ok and target == sidecar and not os.path.exists(md):
        # the receipt re-derives from its sidecar, but no document on disk was compared: a PASS about the
        # receipt, never about a document — recorded, and rendered as an info row
        row["document_matches_sidecar"] = None
        row["document_note"] = (f"no {os.path.relpath(md, repo).replace(chr(92), '/')} beside the sidecar: "
                                "the receipt re-derives, and no document was compared")
    elif ok and target == sidecar:
        from . import sworn
        try:
            rendered = sworn.render(sworn.load_sidecar(json.load(open(sidecar, encoding="utf-8"))))
            on_disk = open(md, "rb").read()
            row["document_matches_sidecar"] = rendered == on_disk
        except Exception as e:  # noqa: BLE001
            row["document_matches_sidecar"] = False
            row["detail"] += f" — the sidecar could not be rendered: {e}"
        if not row["document_matches_sidecar"]:
            ok = False
            row["detail"] += (f" — BUT {os.path.relpath(md, repo).replace(chr(92), '/')} on disk is not the document this receipt swore to "
                              "(edited after the receipt, or never the sworn bytes)")
    row["status"] = "PASS" if ok else "FAIL"
    return row


def step_sworn(repo: Path, limit: int | None = None) -> dict:
    receipts = sorted(glob.glob(str(repo / "papers" / "**" / "*.sworn-receipt.json"), recursive=True))
    if limit:
        receipts = receipts[:limit]
    rows = [check_receipt(repo, rc) for rc in receipts]
    fails = [r["receipt"] for r in rows if r["status"] == "FAIL"]
    skipped = sum(1 for r in rows if r["status"] == "SKIP")
    checked = len(receipts) - skipped
    older_build = sum(1 for r in rows if r["status"] == "PASS" and "same-build=False" in r.get("detail", ""))
    no_document = sum(1 for r in rows if r["status"] == "PASS" and r.get("document_note"))
    tally: dict = {}
    for r in rows:
        if r["status"] == "PASS":
            tally[r.get("document_verdict") or "?"] = tally.get(r.get("document_verdict") or "?", 0) + 1
    detail = f"{checked - len(fails)} of {checked} checkable receipts VERIFIED, {skipped} not checkable here"
    if tally:
        detail += "; the documents they re-derive read " + ", ".join(f"{k} {v}" for k, v in sorted(tally.items()))
    if older_build:
        detail += f" ({older_build} issued under an earlier verifier build and re-derived exactly under this one)"
    if no_document:
        detail += f"; {no_document} with no .md beside the sidecar (no document compared)"
    if fails:
        detail += f"; failing: {', '.join(fails)}"
    return {"status": "PASS" if checked and not fails else ("FAIL" if fails else "SKIP"), "detail": detail,
            "document_verdicts": tally, "receipts": rows}


def step_seals(repo: Path, network: bool) -> dict:
    anchors = repo / "papers" / "charon" / "anchors.jsonl"
    if not anchors.exists():
        return {"status": "SKIP", "detail": "papers/charon/anchors.jsonl does not exist: no anchor exists yet (SEALS_2026_09_13.md lists what will be sealed)"}
    if not network:
        return {"status": "SKIP", "detail": "pass --network to verify the anchors against the chain (`python -m styxx.clock verify`)"}
    from . import clock
    rs = clock.verify(str(anchors))
    ok = bool(rs) and all(r.get("status") == "ANCHORED" for r in rs)
    return {"status": "PASS" if ok else "FAIL",
            "detail": "; ".join(f"#{r.get('n')} {r.get('kind')} {r.get('status')}" + (f" beacon={r['beacon'][:12]}…" if r.get("beacon") and r.get("status") == "ANCHORED" else "") for r in rs),
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
        can = certs.get("canaries") if isinstance(certs.get("canaries"), dict) else {}
        draw = can.get("draw")
        if not draw:
            rows.append({"certs": rel, "status": "PASS", "detail": "no draw record: the hand-written set" + (f" ({str(can.get('canary_sha256'))[:12]}…)" if can.get("canary_sha256") else "")})
            continue
        n_drawn += 1
        problems = []
        try:
            items = _beacon.select(str(draw["beacon"]), int(draw["n"]))
            if draw.get("pool_sha256") != _beacon.pool_sha256():
                problems.append("the pool DOES NOT match this checkout")
            if not (ck.canary_sha256(items) == draw.get("canary_sha256") == can.get("canary_sha256")):
                problems.append("the canary hash DOES NOT re-derive")
            if can.get("n") is not None and can.get("n") != int(draw["n"]):
                problems.append(f"canaries.n={can.get('n')!r} is not the draw's n={draw['n']!r}")
            base = os.path.basename(f)
            fp_file = os.path.join(os.path.dirname(f), base.replace("_certs", "_fingerprints", 1))   # the file name only: a directory named *_certs* must not move it
            n_fp = 0
            has_fp_file = fp_file != f and os.path.exists(fp_file)
            if has_fp_file:
                for arm, fp in (json.load(open(fp_file, encoding="utf-8")) or {}).items():
                    try:
                        ck.check_draw_record(fp.get("draw"), fp.get("canary_sha256"), fp.get("ids") or [])
                        if fp.get("draw") != draw:
                            problems.append(f"fingerprint {arm} carries a different draw record than the certs")
                    except ValueError as e:
                        problems.append(f"fingerprint {arm}: {e}")
                    n_fp += 1
            # both runners always write the fingerprints beside the certs; without them the draw is checked
            # against the certs' own hash only, never against the ids the arms graded (STR-3)
            if not has_fp_file:
                problems.append(f"no {os.path.basename(fp_file)} beside it: the draw cannot be checked against the ids")
            elif not n_fp:
                problems.append(f"{os.path.basename(fp_file)} carries no fingerprint: the draw cannot be checked against the ids")
            detail = f"beacon {str(draw['beacon'])[:12]}… → {len(items)} items" + (f", {n_fp} fingerprints re-checked" if n_fp else ", no fingerprints checked")
        except Exception as e:  # noqa: BLE001
            problems.append(f"the draw could not be re-derived: {e}")
            detail = "the draw could not be re-derived"
        ok = not problems
        rows.append({"certs": rel, "status": "PASS" if ok else "FAIL", "detail": detail + ("" if ok else " — " + "; ".join(problems))})
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


def _load_json(path):
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict):
        raise ValueError(f"not a JSON object ({type(doc).__name__})")
    return doc


def _norm_certs_file(repo: Path, cf) -> str | None:
    """A scorecard's `certs_file` as a forward-slash path relative to the repo (None when it names nothing):
    `score.py --out` writes it that way, and a card spelled with backslashes or `./` names the same file."""
    if not isinstance(cf, str) or not cf.strip():
        return None
    s = cf.strip().replace("\\", "/")
    if os.path.isabs(s):
        try:
            s = os.path.relpath(s, repo).replace("\\", "/")
        except ValueError:          # another drive: it is not in this tree
            return s
    return posixpath.normpath(s)


def _sealed_beacon(seals: dict | None, digest: str | None):
    """(beacon, why-not): the beacon the seals step printed on the ANCHORED `sealed-prereg` line whose digest is
    the beacon-draw PREREG's sealed digest, or None and the reason there is none."""
    if not digest:
        return None, "the scorer names no sealed digest for the beacon-draw PREREG"
    if not isinstance(seals, dict) or not isinstance(seals.get("lines"), list):
        return None, "the seals step did not run" + (f" ({seals['detail']})" if isinstance(seals, dict) and seals.get("detail") else "")
    beacons = sorted({str(r["beacon"]).lower() for r in seals["lines"]
                      if isinstance(r, dict) and r.get("kind") == "sealed-prereg" and r.get("status") == "ANCHORED"
                      and str(r.get("digest") or "").lower() == digest.lower() and r.get("beacon")})
    if not beacons:
        return None, f"the anchors carry no ANCHORED sealed-prereg line for the beacon-draw PREREG's digest {digest[:12]}…"
    if len(beacons) > 1:
        return None, f"{len(beacons)} ANCHORED sealed-prereg lines for the beacon-draw PREREG's digest print different beacons"
    return beacons[0], None


_CARD_KEYS = ("hypotheses", "gates", "counts_as_result", "run_reading")


def step_reading(repo: Path, seals: dict | None = None) -> dict:
    """Every certs file read by the scorer, and every committed scorecard accounted for.

    `seals` is the seals step's result, when it ran: a beacon-drawn certs file is read with the beacon its
    ANCHORED sealed-prereg line prints (K5's beacon clause); without one, no beacon-draw certs file counts as a
    result here, whatever the scorer's version (the verification of 2026-09-14, S1). Scorecards (STR-1): an
    unreadable one FAILS; two current-schema cards naming one certs file FAIL (one would shadow the other); a
    current-schema card naming no certs file this step reads FAILS; a card of another schema is an info row,
    never silently dropped. An unreadable certs file FAILS (STR-5)."""
    scorer = _load_scorer(repo)
    if scorer is None:
        return {"status": "SKIP", "detail": "papers/checksum/score.py is not in this checkout"}
    by_file = {v["prereg"]: k for k, v in scorer.BANDS.items()}
    beacon, no_beacon_why = _sealed_beacon(seals, (scorer.BANDS.get("beacon_draw") or {}).get("sealed_blob"))
    card_rows, fails = [], []
    by_certs: dict = {}
    for sc in sorted(glob.glob(str(repo / "papers" / "checksum" / "*_scorecard*.json"))):
        sc_rel = os.path.relpath(sc, repo).replace("\\", "/")
        try:
            c = _load_json(sc)
        except Exception as e:  # noqa: BLE001
            card_rows.append({"scorecard": sc_rel, "status": "FAIL", "detail": f"unreadable scorecard: {e}"}); fails.append(sc_rel); continue
        if c.get("schema") != scorer.SCHEMA:
            card_rows.append({"scorecard": sc_rel, "status": "INFO", "schema": c.get("schema"),
                              "detail": f"schema {c.get('schema')!r} is not the scorer's current {scorer.SCHEMA}: not compared "
                                        "(an older scorer's card is history, not a claim about today's reading)"})
            continue
        by_certs.setdefault(_norm_certs_file(repo, c.get("certs_file")), []).append((sc_rel, c))
    rows = []
    for f in _certs_files(repo):
        rel = os.path.relpath(f, repo).replace("\\", "/")
        named = by_certs.pop(rel, [])
        try:
            certs = _load_json(f)
        except Exception as e:  # noqa: BLE001
            rows.append({"certs": rel, "status": "FAIL", "detail": f"unreadable certs file: {e}"}); fails.append(rel); continue
        key = by_file.get(certs.get("prereg"))
        if key is None:
            row = {"certs": rel, "status": "SKIP", "detail": f"prereg {certs.get('prereg')!r} is not one the scorer knows"}
            if named:
                row.update(status="FAIL", committed_scorecards=[n for n, _ in named],
                           detail=row["detail"] + f"; yet {', '.join(n for n, _ in named)} (the current schema) names these certs, "
                                                  "and the scorer cannot re-read them")
                fails.append(rel)
            rows.append(row)
            continue
        kw = {"expect_beacon": beacon} if (key == "beacon_draw" and beacon) else {}   # by keyword: the old and the new scorer take it
        try:
            card = scorer.score(certs, key, **kw)
        except Exception as e:  # noqa: BLE001
            rows.append({"certs": rel, "status": "FAIL", "detail": f"the scorer raised: {e}"}); fails.append(rel); continue
        row = {"certs": rel, "prereg": key, "status": "PASS", "reading": card["run_reading"], "counts_as_result": card["counts_as_result"]}
        if key == "beacon_draw":
            row["expect_beacon"] = beacon
            if not beacon and card["counts_as_result"]:
                # a scorer from before S1 counts a beacon-draw card read without the seal's beacon; this step never does
                row["counts_as_result"] = False
                row["scorer_counts_as_result"] = True
                row["detail"] = (card["run_reading"] + " — NOT counted here: K5's beacon clause was not checked, because no seal's "
                                 f"beacon was given ({no_beacon_why})")
        if len(named) > 1:
            row["status"] = "FAIL"; fails.append(rel)
            row["committed_scorecards"] = [n for n, _ in named]
            row["detail"] = (f"{len(named)} current-schema scorecards name these certs ({', '.join(n for n, _ in named)}): "
                             "one would shadow the other, so none is compared")
        elif named:
            sc_rel, committed = named[0]
            with open(f, "rb") as fh:
                same_bytes = committed.get("certs_sha256") == hashlib.sha256(fh.read()).hexdigest()
            same = same_bytes and all(committed.get(k) == card.get(k) for k in _CARD_KEYS)
            if same_bytes and not same and kw:
                # a card written without the seal's beacon (an instrument check scored before any seal existed) is a
                # true record of that reading only while it claims no result
                try:
                    bare = scorer.score(certs, key)
                except Exception:  # noqa: BLE001
                    bare = None
                if bare is not None and committed.get("counts_as_result") is False and all(committed.get(k) == bare.get(k) for k in _CARD_KEYS):
                    same = True
                    row["committed_scorecard_note"] = ("the committed scorecard was read without the seal's beacon and claims no "
                                                       "result; it matches that reading")
            row["committed_scorecard"] = sc_rel
            row["committed_scorecard_matches"] = same
            if not same:
                row["status"] = "FAIL"; fails.append(rel)
                row["detail"] = ("the committed scorecard was written for other certs bytes" if not same_bytes
                                 else "the committed scorecard is not what the scorer reads today")
        rows.append(row)
    for cf, named in sorted(by_certs.items(), key=lambda kv: str(kv[0])):
        for sc_rel, _ in named:
            if cf is None:
                why = "a current-schema scorecard with no certs_file: it names no certs file, so nothing can be compared with it"
            elif (repo / cf).is_file():
                why = f"names {cf}, which is not a certs file this step reads (papers/checksum/*_certs*.json, smoke files excluded)"
            else:
                why = f"names {cf}, which is not in the tree: a scorecard for certs no one can re-read"
            card_rows.append({"scorecard": sc_rel, "status": "FAIL", "certs_file": cf, "detail": why}); fails.append(sc_rel)
    n_info = sum(1 for r in card_rows if r["status"] == "INFO")
    detail = (f"{len([r for r in rows if r['status'] != 'SKIP'])} certs files read against their PREREG; "
              f"{sum(1 for r in rows if r.get('counts_as_result'))} count as a result; "
              f"{sum(1 for r in rows if 'committed_scorecard' in r)} committed scorecards compared"
              + (f", {n_info} of another schema listed and not compared" if n_info else ""))
    if beacon:
        detail += f"; beacon-draw certs read with the seal's beacon {beacon[:12]}…"
    else:
        detail += f"; no beacon-draw card can count as a result without the seal's beacon (--network): {no_beacon_why}"
    if fails:
        detail += f"; failing: {', '.join(fails)}"
    return {"status": "PASS" if rows and not fails else ("FAIL" if fails else "SKIP"), "detail": detail,
            "expect_beacon": beacon, "files": rows, "scorecards": card_rows}


def step_recipe() -> dict:
    return {"status": "SKIP", "detail": "run `python papers/checksum/run_smollm_quant.py` (CPU, minutes) and compare with `python -m styxx.portability` — needs a model stack; STRANGER.md §7"}


def run(repo, expect_head: str | None = None, with_tests: bool = False, network: bool = False, only=None,
        sworn_limit: int | None = None, allow_dirty: bool = False) -> dict:
    repo = Path(repo).resolve()
    only = set(only) if only else set(STEPS)
    steps = {}
    t0 = time.time()
    for name in STEPS:
        if name not in only and name != "checkout":   # checkout runs on every invocation: a dirty tree FAILS whatever --only says
            steps[name] = {"status": "SKIP", "detail": "not selected (--only)"}
            continue
        t1 = time.time()
        if name == "checkout":
            s = step_checkout(repo, allow_dirty)
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
            s = step_reading(repo, steps.get("seals"))    # the seal's beacon, when the seals step ran and prints one
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
                continue
            if r.get("document_verdict") is None:
                out.append(f"          info  {r['receipt']}: the receipt re-derives; its check line carries no document= field, "
                           "so which verdict the document reads is not known")
            elif r["document_verdict"] != "SWORN-HELD":
                out.append(f"          info  {r['receipt']}: the receipt re-derives; the document it swears to reads {r['document_verdict']}")
            if r.get("document_note"):
                out.append(f"          info  {r['receipt']}: {r['document_note']}")
        for r in (s.get("files") or []):
            if name == "reading" and r["status"] != "SKIP":
                out.append(f"          {r['status']}  {r['certs']}: {r.get('detail') or r.get('reading', '')}")
                if r.get("committed_scorecard_note"):
                    out.append(f"          info  {r['committed_scorecard']}: {r['committed_scorecard_note']}")
            elif r["status"] == "FAIL":
                out.append(f"          FAIL  {r['certs']}: {r['detail']}")
        for r in (s.get("scorecards") or []):
            out.append(f"          {'info' if r['status'] == 'INFO' else r['status']}  {r['scorecard']}: {r['detail']}")
    out += ["", f"  {rep['verdict']}  ({rep['seconds']}s)"]
    return "\n".join(out)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(prog="python -m styxx.stranger")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--expect-head", default=None, help="the full 64-hex ferry-log head you were given outside the log")
    ap.add_argument("--with-tests", action="store_true")
    ap.add_argument("--network", action="store_true", help="verify the anchors against the chain")
    ap.add_argument("--allow-dirty", action="store_true", help="check a working tree whose tracked files differ from the commit")
    ap.add_argument("--only", default=None, help="comma-separated steps: " + ",".join(STEPS) + " (checkout runs whether named or not)")
    ap.add_argument("--json", default=None, help="write the report here")
    a = ap.parse_args(argv)
    only = [s.strip() for s in a.only.split(",")] if a.only else None
    if only and any(s not in STEPS for s in only):
        ap.error(f"--only must name steps from {', '.join(STEPS)}")
    rep = run(a.repo, a.expect_head, a.with_tests, a.network, only, allow_dirty=a.allow_dirty)
    print(render(rep))
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(rep, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
        print(f"  report -> {a.json}")
    return 0 if not rep["failed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
