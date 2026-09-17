#!/usr/bin/env python3
"""score_mutations.py — the published mutation run for papers/checksum/score.py.

    python papers/checksum/score_mutations.py [--jobs 4] [--only ID[,ID...]]

Why it exists. Commit 4e08cf28 said "a mutation run applied 28 changes to score.py one at a time ... and a
test failed for all 28", and published a count. The verification of 2026-09-14 (S2) wrote 18 new
mutations and 15 of them survived: removed n, pool and canary-hash checks, K2 never firing, K4's lower
edge going strict. A count cannot be re-run. This file publishes the list instead.

MUTATIONS holds (id, description, old_text, new_text). Each old_text must occur exactly once in the
current score.py. For each mutation the script copies papers/checksum/ and tests/test_checksum_score.py
into a fresh temporary directory laid out like the repository, replaces old_text with new_text IN THE
COPY (never in the real file), and runs pytest on the copied tests with PYTHONPATH set to this
checkout's root (so `styxx` is this checkout's). A mutation is killed only when pytest exits 1 (a test
failed); a collection error or a timeout is recorded and is not a kill. An unmutated copy runs first and
must pass, or nothing is written. The result, papers/checksum/score_mutations_result.json, records for
every id its description, whether old_text was found exactly once, and whether a test killed it, with
the LF-normalised sha256 of the score.py and the tests it ran against.

Groups, by id prefix:
  v2-01..v2-28  the 28 of commit 4e08cf28, reconstructed from its message's categories as substrings of
                today's file: 18 comparator flips, the seven band changes the F4 finder listed, and the
                removal of the sealed-digest, draw-n and arm-set checks (v2x- are four more of that kind).
  S2-           every mutation the S2 verification named (its 18, plus deploy_quant's K1 and K4 edges).
  S1-, S3-..S7- at least one for each rule added on 2026-09-14 in answer to S1 and S3-S7.

Inside the run the environment carries STYXX_SCORE_MUTANT=1; the one test that reads this file's
published result skips there (the copy it would check is mutated on purpose), and nothing else does.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SCORE = os.path.join(HERE, "score.py")
TESTS = os.path.join(ROOT, "tests", "test_checksum_score.py")
RESULT = os.path.join(HERE, "score_mutations_result.json")
SCHEMA = "styxx.checksum/score-mutations/v1"
MUTANT_ENV = "STYXX_SCORE_MUTANT"

MUTATIONS = [
    # ---- v2: the 28 of commit 4e08cf28 — every comparator flip
    ("v2-01", "beacon H1 floor clause: <= becomes <", '"≤ 0.001", floor, floor <= b["h1_floor_max"]', '"≤ 0.001", floor, floor < b["h1_floor_max"]'),
    ("v2-02", "deploy_quant H1 floor > 0 becomes >= 0", 'floor > b["h1_floor_min_exclusive"]', 'floor >= b["h1_floor_min_exclusive"]'),
    ("v2-03", "deploy_quant H1 floor <= 1e-3 becomes <", 'and floor <= b["h1_floor_max"])', 'and floor < b["h1_floor_max"])'),
    ("v2-04", "H2 band upper edge strict", 'lo <= m4 <= hi', 'lo <= m4 < hi'),
    ("v2-05", "H2 geometry r >= becomes >", 'r4 >= b["h2_r_min"]', 'r4 > b["h2_r_min"]'),
    ("v2-06", "H2 top-1 loss <= becomes <", 'l4 <= b["h2_top1_loss_max"]', 'l4 < b["h2_top1_loss_max"]'),
    ("v2-07", "H3 'below Q4's' < becomes <=", 'm8 < m4 if', 'm8 <= m4 if'),
    ("v2-08", "H3 geometry r >= becomes >", 'r8 >= b["h3_r_min"]', 'r8 > b["h3_r_min"]'),
    ("v2-09", "H3 top-1 loss <= becomes <", 'l8 <= b["h3_top1_loss_max"]', 'l8 < b["h3_top1_loss_max"]'),
    ("v2-10", "H4 mean > becomes >=", 'mr > b["h4_mean_min"]', 'mr >= b["h4_mean_min"]'),
    ("v2-11", "H4 geometry r < becomes <=", 'rr < b["h4_r_max"]', 'rr <= b["h4_r_max"]'),
    ("v2-12", "H4 top-1 hits <= becomes <", 'hr <= b["h4_hits_max"]', 'hr < b["h4_hits_max"]'),
    ("v2-13", "K3 mean > becomes >=", 'm4 > b["k3_mean"]', 'm4 >= b["k3_mean"]'),
    ("v2-14", "K4 upper edge strict", 'lo <= mr <= hi', 'lo <= mr < hi'),
    ("v2-15", "H5 ratio lower edge strict", 'r0 <= ratio <= r1', 'r0 < ratio <= r1'),
    ("v2-16", "H5 ratio upper edge strict", 'r0 <= ratio <= r1', 'r0 <= ratio < r1'),
    ("v2-17", "H6 move <= becomes <", 'mv <= mx if', 'mv < mx if'),
    ("v2-18", "K1 floor > becomes >=", 'k1_fired = floor is not None and floor > b["k1_floor"]', 'k1_fired = floor is not None and floor >= b["k1_floor"]'),
    # ---- v2: the seven band changes the F4 finder listed
    ("v2-19", "band: beacon_draw h1_floor_max 1e-3 -> 1e-2", '"h1_floor_min_exclusive": None, "h1_floor_max": 1e-3,', '"h1_floor_min_exclusive": None, "h1_floor_max": 1e-2,'),
    ("v2-20", "band: deploy_quant h1_floor_max 1e-3 -> 5e-2", '"h1_floor_min_exclusive": 0.0, "h1_floor_max": 1e-3,', '"h1_floor_min_exclusive": 0.0, "h1_floor_max": 5e-2,'),
    ("v2-21", "band: beacon_draw h5_ratio_band -> (0.1, 10)", '"h5_ratio_band": (0.5, 2.0),', '"h5_ratio_band": (0.1, 10.0),'),
    ("v2-22", "band: beacon_draw h6 Q8 0.10 -> 0.50", '"Q8": 0.10,', '"Q8": 0.50,'),
    ("v2-23", "band: beacon_draw h2_band hi 0.60 -> 0.604", '"h2_band": (0.05, 0.60)', '"h2_band": (0.05, 0.604)'),
    ("v2-24", "band: beacon_draw h4_mean_min 5 -> 5.9", '"h4_mean_min": 5.0, "h4_r_max": 0.6,', '"h4_mean_min": 5.9, "h4_r_max": 0.6,'),
    ("v2-25", "band: deploy_quant h2_r_min 0.95 -> 0.954", '"h2_r_min": 0.95,', '"h2_r_min": 0.954,'),
    # ---- v2: removing the digest, n and set checks
    ("v2-26", "sealed-digest check removed", 'if prov.get("prereg_blob_sha256") != b["sealed_blob"]:', 'if False:'),
    ("v2-27", "draw n = 48 check removed", 'if draw.get("n") != N_CANARIES:', 'if False:'),
    ("v2-28", "every graded cert grades the PREREG's set: check removed", 'if expected_hash is not None and c.get("canary_sha256") != expected_hash:', 'if False:'),
    # ---- v2x: four more of the same kind
    ("v2x-01", "--expect-blob agreement check removed", 'if expect_blob is not None and expect_blob.lower() != b["sealed_blob"]:', 'if False:'),
    ("v2x-02", "every graded cert grades 48 items: check removed", 'if c.get("n_items") != N_CANARIES:', 'if False:'),
    ("v2x-03", "the held-out cert is no longer graded", 'graded.append(("h1_held_out", ho))', 'pass'),
    ("v2x-04", "K1 read from the runner's own k1.fired", 'k1_fired = floor is not None and floor > b["k1_floor"]', 'k1_fired = bool(k1_rec.get("fired"))'),
    # ---- S2: every mutation the verification named
    ("S2-01", "K4 lower edge strict", 'k4 = mr is not None and lo <= mr <= hi', 'k4 = mr is not None and lo < mr <= hi'),
    ("S2-02", "K2 never fires", '"fired": dq4.get("verdict") == "SAME" if dq4 else None', '"fired": False if dq4 else None'),
    ("S2-03", "deploy_quant CORRECTION rule 2 branch removed", 'if c1["holds"] and d2.get("verdict") == "INCONCLUSIVE":', 'if False:'),
    ("S2-04", "deploy_quant draw-record refusal removed", 'if prereg == "deploy_quant" and _obj(certs.get("canaries")).get("draw"):', 'if False:'),
    ("S2-05", "pool hash check removed", 'if draw.get("pool_sha256") != _beacon.pool_sha256():', 'if False:'),
    ("S2-06", "sanity.n_canaries check removed", 'if san.get("n_canaries") != N_CANARIES:', 'if False:'),
    ("S2-07", "canaries.n check removed", 'if can.get("n") != N_CANARIES:', 'if False:'),
    ("S2-07b", "an absent canaries.n is tolerated again", 'if can.get("n") != N_CANARIES:', 'if can.get("n") is not None and can.get("n") != N_CANARIES:'),
    ("S2-08", "canaries.canary_sha256 vs the expected set removed", 'elif expected_hash is not None and can.get("canary_sha256") != expected_hash:', 'elif False:'),
    ("S2-09", "H5 model check removed", ' or hand_set.get("model") != certs.get("model") or', ' or'),
    ("S2-10", "H6 AGREE clause always holds", 'portability.get("verdicts") == "AGREE")]', 'True)]'),
    ("S2-11", "k1 record threshold comparison removed", ' or k1_rec.get("threshold_nats") != b["k1_floor"])', ')'),
    ("S2-12", "beacon rule 4 note without floor > 0", 'elif floor is not None and floor > 0 and ho.get("verdict") == "DRIFT":', 'elif ho.get("verdict") == "DRIFT":'),
    ("S2-13", "H5 same-device None guard removed", 'hp.get("cuda_device") is not None and hp.get("cuda_device")', 'hp.get("cuda_device")'),
    ("S2-14", "draw hash vs canaries hash check removed", 'if draw.get("canary_sha256") != can.get("canary_sha256"):', 'if False:'),
    ("S2-15", "K1-fired beacon card drops H5 and H6", ' + (("H5", "H6") if prereg == "beacon_draw" else ())', ''),
    ("S2-16", "graded certs' draw record comparison removed", 'if (c.get("draw") or None) != (draw or None):', 'if False:'),
    ("S2-17", "H2 band lower edge strict", 'lo <= m4 <= hi', 'lo < m4 <= hi'),
    ("S2-18", "K3 top-1 loss > becomes >=", 'l4 > b["k3_top1_loss"]', 'l4 >= b["k3_top1_loss"]'),
    ("S2-19", "deploy_quant K1 edge: > becomes >= under deploy_quant only", 'k1_fired = floor is not None and floor > b["k1_floor"]',
     'k1_fired = floor is not None and (floor >= b["k1_floor"] if prereg == "deploy_quant" else floor > b["k1_floor"])'),
    ("S2-20", "deploy_quant K4 lower edge strict under deploy_quant only", 'k4 = mr is not None and lo <= mr <= hi',
     'k4 = mr is not None and ((lo < mr <= hi) if prereg == "deploy_quant" else (lo <= mr <= hi))'),
    ("S2-21", "deploy_quant K4 upper edge strict under deploy_quant only", 'k4 = mr is not None and lo <= mr <= hi',
     'k4 = mr is not None and ((lo <= mr < hi) if prereg == "deploy_quant" else (lo <= mr <= hi))'),
    # ---- S1: --expect-beacon is required for a beacon_draw result
    ("S1-01", "a card without --expect-beacon counts as a result", 'card["counts_as_result"] = valid_experiment and not beacon_unchecked', 'card["counts_as_result"] = valid_experiment'),
    ("S1-02", "the reading no longer begins by saying the beacon clause was not evaluated",
     '("NOT A RESULT: K5\'s beacon clause not evaluated (no --expect-beacon); " if beacon_unchecked else "")', '""'),
    ("S1-03", "K5 without a beacon reads not fired, the certs are the experiment",
     'card["gates"]["K5"] = {"fired": None, "detail": "not evaluable: " + BEACON_UNCHECKED + "; every other K5 clause holds"}',
     'card["gates"]["K5"] = {"fired": False, "detail": "the certs are the experiment"}'),
    ("S1-04", "a fired K5 no longer says the beacon clause was not evaluated", 'k5 + ([BEACON_UNCHECKED] if beacon_unchecked else [])', 'k5'),
    ("S1-05", "the beacon comparison removed", 'if expect_beacon and str(draw.get("beacon", "")).lower() != expect_beacon.lower():', 'if False:'),
    ("S1-06", "a missing beacon is recorded as checked", 'beacon_unchecked = True', 'beacon_unchecked = False'),
    # ---- S3: when K1 did not fire the arms, the held-out cert and the canary hash must exist
    ("S3-01", "an absent arm cert is not a problem", 'out.append(f"the {arm} cert is absent: K1 did not fire, and the runner writes every arm when it does not")', 'pass'),
    ("S3-02", "an absent held-out cert is not a problem", 'out.append("h1_held_out.cert is absent: K1 did not fire, and the runner writes the held-out reading when it does not")', 'pass'),
    ("S3-03", "an absent canaries.canary_sha256 is not a problem", 'out.append("canaries.canary_sha256 is absent: nothing in the certs names the set they graded")', 'pass'),
    ("S3-04", "deploy_quant never requires the certs", 'problems += _set_problems(certs, HAND_CANARY_SHA256, None, require_certs=not k1_fired)',
     'problems += _set_problems(certs, HAND_CANARY_SHA256, None, require_certs=False)'),
    ("S3-05", "beacon_draw requires the certs even when K1 fired", 'draw if isinstance(draw, dict) else None, require_certs=not k1_fired)',
     'draw if isinstance(draw, dict) else None, require_certs=True)'),
    # ---- S4: is_the_experiment is never trusted alone
    ("S4-01", "a non-empty tag is trusted", 'if certs.get("tag") not in (None, ""):', 'if False:'),
    ("S4-02", "smoke true is trusted", 'if certs.get("smoke"):', 'if False:'),
    ("S4-03", "a missing git_head is trusted", 'if not isinstance(prov.get("git_head"), str) or not prov.get("git_head").strip():', 'if False:'),
    ("S4-03b", "an empty git_head is trusted again", 'if not isinstance(prov.get("git_head"), str) or not prov.get("git_head").strip():', 'if prov.get("git_head") is None:'),
    ("S4-04", "git_dirty_tracked true is trusted", 'if prov.get("git_dirty_tracked"):', 'if False:'),
    ("S4-05", "prereg_blob_is_sealed false is trusted", 'if "prereg_blob_is_sealed" in prov and prov.get("prereg_blob_is_sealed") is not True:', 'if False:'),
    # ---- S5: a valid hand set whose K1 fired leaves H5 PENDING
    ("S5-01", "a K1-fired hand set is compared clause by clause", 'elif hand_card["gates"]["K1"]["fired"]:', 'elif False:'),
    ("S5-02", "a reading that does not exist fails the clause instead of leaving it not evaluable",
     'return (x == want and y == want) if (x is not None and y is not None) else None', 'return x == want and y == want'),
    # ---- S6: H6 is bound to these certs
    ("S6-01", "the portability record is never checked for binding", 'unbound = _portability_unbound(certs, portability) if portability else None', 'unbound = None'),
    ("S6-02", "the portability digest is not re-derived", 'if hashlib.sha256(blob).hexdigest() != portability.get("digest"):', 'if False:'),
    ("S6-03", "the record's input_cert_digests are not compared with these certs", 'if not here:', 'if False:'),
    ("S6-04", "an arm without a digest is skipped", 'return f"these certs carry no {arm} cert digest"', 'continue'),
    ("S6-05", "a record without arms is not refused", 'if not isinstance(arms, dict) or not arms:', 'if False:'),
    # ---- S6, repair round: the binding rests on re-derived bytes and a second input, not on digest fields
    ("S6-06", "these certs' arm digests are trusted as written, not re-derived from their bodies",
     'if _canonical_sha256({k: v for k, v in cert.items() if k not in ("digest", "created")}) != digest:', 'if False:'),
    ("S6-07", "a record that lists these certs as two machines (a self-comparison) is accepted", 'if len(here) > 1:', 'if False:'),
    ("S6-08", "the record's machine count is not checked against its inputs",
     'or _num(n) is None or n < 2 or len(inputs) != n:', 'or _num(n) is None:'),
    ("S6-09", "a one-machine record is accepted", 'or n < 2 or', 'or n < 1 or'),
    ("S6-10", "the record's verdict column for these certs is not compared with their verdicts",
     'if not isinstance(verdicts, list) or len(verdicts) != n or verdicts[i] != dist.get("verdict"):', 'if not isinstance(verdicts, list):'),
    ("S6-11", "the record's mean column for these certs is not compared with their means",
     'if not isinstance(values, list) or len(values) != n or values[i] != own:', 'if not isinstance(values, list):'),
    ("S6-12", "max_abs_diff is read as written, not checked against the spread of its values", 'if leaf.get("max_abs_diff") != spread:', 'if False:'),
    ("S6-13", "the record's verdicts field is read as written, not checked against its per-arm verdicts",
     'if (portability.get("verdicts") == "AGREE") != agree:', 'if False:'),
    ("S6-14", "a record whose inputs are not all lists is read", 'or not all(isinstance(x, list) for x in inputs)', ''),
    # ---- S7: malformed certs are problems, not exceptions; a floor is a mean absolute distance
    ("S7-01", "a top-level block that is not an object is not a problem", 'problems.append(f"{key} is not an object")', 'pass'),
    ("S7-02", "an arm cert that is not an object is not a problem", 'out.append(f"the {arm} cert is not an object")', 'pass'),
    ("S7-03", "a negative floor is accepted", 'elif floor is not None and floor < 0:', 'elif False:'),
    ("S7-04", "a non-finite floor is accepted", 'if floor is not None and not math.isfinite(floor):', 'if False:'),
    ("S7-05", "non-dict blocks are used as they are (the scorer raises)", 'return x if isinstance(x, dict) else {}', 'return x or {}'),
]


def lf_sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:        # universal newlines: the substrings never span a line ending
        return fh.read()


def _stage(text: str | None) -> str:
    """A fresh copy of papers/checksum/ and the scorer's tests, laid out like the repository; score.py replaced by `text`."""
    dest = tempfile.mkdtemp(prefix="score_mutation_")
    shutil.copytree(HERE, os.path.join(dest, "papers", "checksum"), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    os.makedirs(os.path.join(dest, "tests"))
    shutil.copyfile(TESTS, os.path.join(dest, "tests", "test_checksum_score.py"))
    with open(os.path.join(dest, "pytest.ini"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("[pytest]\n")                        # pins rootdir to the copy: nothing of the checkout's config is read
    if text is not None:
        with open(os.path.join(dest, "papers", "checksum", "score.py"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    return dest


def _pytest(dest: str, timeout: int) -> tuple[int | None, str]:
    env = dict(os.environ, PYTHONPATH=ROOT, PYTHONDONTWRITEBYTECODE="1", GIT_OPTIONAL_LOCKS="0", PYTHONIOENCODING="utf-8")
    env[MUTANT_ENV] = "1"
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests/test_checksum_score.py"],
                           cwd=dest, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout}s"
    lines = [ln for ln in (r.stdout or "").splitlines() if ln.strip()]
    return r.returncode, (lines[-1] if lines else (r.stderr or "")[-300:]).strip()


def _run_one(mutation, original: str, timeout: int) -> dict:
    mid, desc, old, new = mutation
    found_once = original.count(old) == 1 and old != new
    row = {"id": mid, "description": desc, "old_text_found_once": found_once, "killed": False, "pytest_returncode": None, "pytest_tail": None}
    if not found_once:
        row["pytest_tail"] = f"old_text occurs {original.count(old)} times in score.py; not applied"
        return row
    dest = _stage(original.replace(old, new, 1))
    try:
        rc, tail = _pytest(dest, timeout)
    finally:
        shutil.rmtree(dest, ignore_errors=True)
    row.update(killed=(rc == 1), pytest_returncode=rc, pytest_tail=tail)
    return row


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--only", default=None, help="comma-separated ids; prints, never writes the published result")
    a = ap.parse_args(argv)
    ids = [m[0] for m in MUTATIONS]
    if len(set(ids)) != len(ids):
        raise SystemExit("mutation ids are not unique")
    chosen = MUTATIONS if not a.only else [m for m in MUTATIONS if m[0] in set(a.only.split(","))]
    original = _read(SCORE)
    t0 = time.time()
    dest = _stage(None)
    try:
        rc, tail = _pytest(dest, a.timeout)
    finally:
        shutil.rmtree(dest, ignore_errors=True)
    baseline = {"passed": rc == 0, "pytest_returncode": rc, "pytest_tail": tail}
    print(f"baseline (unmutated copy): rc={rc}  {tail}", flush=True)
    if rc != 0:
        print("the unmutated copy does not pass; nothing is written", file=sys.stderr)
        return 2
    rows = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        futures = {pool.submit(_run_one, m, original, a.timeout): m[0] for m in chosen}
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            rows[row["id"]] = row
            verdict = "KILLED  " if row["killed"] else ("NOT FOUND" if not row["old_text_found_once"] else "SURVIVED")
            print(f"{verdict} {row['id']:7s} {row['description']}  | {row['pytest_tail']}", flush=True)
    ordered = [rows[m[0]] for m in chosen]
    survivors = [r["id"] for r in ordered if not (r["killed"] and r["old_text_found_once"])]
    print(f"{len(ordered) - len(survivors)}/{len(ordered)} killed in {time.time() - t0:.0f}s" + (f"; not killed: {', '.join(survivors)}" if survivors else ""))
    if a.only:
        return 0 if not survivors else 1
    result = {
        "schema": SCHEMA,
        "score_py": "papers/checksum/score.py", "score_py_sha256_lf": lf_sha256(SCORE),
        "tests": "tests/test_checksum_score.py", "tests_sha256_lf": lf_sha256(TESTS),
        "command": "python papers/checksum/score_mutations.py",
        "pytest": "python -m pytest -q -x -p no:cacheprovider tests/test_checksum_score.py, in a copy, PYTHONPATH=<checkout root>, "
                  f"{MUTANT_ENV}=1; killed means exit code 1",
        "python": sys.version.split()[0],
        "baseline": baseline,
        "n_mutations": len(ordered), "n_killed": sum(1 for r in ordered if r["killed"]),
        "mutations": [{k: r[k] for k in ("id", "description", "old_text_found_once", "killed", "pytest_returncode", "pytest_tail")}
                      for r in ordered],
    }
    with open(RESULT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    print("wrote", os.path.relpath(RESULT, ROOT).replace("\\", "/"))
    return 0 if not survivors else 1


if __name__ == "__main__":
    raise SystemExit(main())
