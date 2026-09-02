# -*- coding: utf-8 -*-
"""range_sanity_report_ab.py -- frozen by PREREG_range_sanity_report_2026_09_02.

The A/B for V14_RANGE_SANITY_REPORT: every certifiable internal document and every rebuilt external
repository, certified with the flag OFF and ON at the same commit, per token; the external
accusations the flag removes are joined to the 2026-08-27 blind panel so the gate can ask whether
what was removed was false. Writes range_sanity_report_ab_result.json. Nothing committed is touched.

  python range_sanity_report_ab.py [--smoke]
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
import importlib                                       # noqa: E402
C = importlib.import_module("styxx.certify")          # the SUBMODULE, never the shadowing function
from styxx.certify import certify_doc                  # noqa: E402
from styxx.corpus_audit import _resolve_receipts       # noqa: E402
from styxx.protocol import Experiment                  # noqa: E402

SMOKE = "--smoke" in sys.argv
PREREG = HERE / "PREREG_range_sanity_report_2026_09_02.md"
CACHE = Path(os.environ.get("OATH_EXT_CACHE", Path(os.environ.get("TEMP", "/tmp")) / "oath_ext_corpus_cache"))


def certify_both(doc: Path, receipts):
    C.V14_RANGE_SANITY_REPORT = False
    off = certify_doc(doc, receipts)
    C.V14_RANGE_SANITY_REPORT = True
    on = certify_doc(doc, receipts)
    C.V14_RANGE_SANITY_REPORT = False
    return off, on


def klass(v):
    return str(v or "").split(",")[0]


def status_map(cert):
    return {(e["line"], e.get("col"), str(e["token"])): e["status"] for e in cert["ledger"]}


def internal():
    docs = held_to_failed = failed_to_held = tokens = moved = 0
    moves = collections.Counter()
    flipped = []
    for cp in sorted(ROOT.glob("papers/**/*.certificate.json")):
        if "anc" in cp.parts:
            continue
        doc = cp.with_name(cp.name.replace(".certificate.json", ".md"))
        if not doc.exists():
            continue
        try:
            rec = json.loads(cp.read_text(encoding="utf-8"))
            receipts, missing, _ = _resolve_receipts(cp, rec, ROOT / "papers")
            if not receipts or missing:
                continue
            off, on = certify_both(doc, receipts)
        except Exception:                               # noqa: BLE001
            continue
        docs += 1
        a, b = status_map(off), status_map(on)
        for k, s in a.items():
            tokens += 1
            if b.get(k) != s:
                moved += 1
                moves[f"{s}->{b.get(k)}"] += 1
        ko, kn = klass(off["verdict"]), klass(on["verdict"])
        if ko == "OATH-HELD" and kn == "OATH-FAILED":
            held_to_failed += 1
            flipped.append({"doc": doc.relative_to(ROOT).as_posix(), "was": ko, "now": kn})
        if ko == "OATH-FAILED" and kn == "OATH-HELD":
            failed_to_held += 1
            flipped.append({"doc": doc.relative_to(ROOT).as_posix(), "was": ko, "now": kn})
        if SMOKE and docs >= 12:
            break
    return {"documents": docs, "tokens": tokens, "tokens_moved": moved, "moves": dict(moves),
            "held_to_failed": held_to_failed, "failed_to_held": failed_to_held, "flipped": flipped}


def external():
    manifest = json.loads((HERE / "oath_external_corpus.json").read_text(encoding="utf-8"))
    adj = json.loads((HERE / "oath_adjudication_result.json").read_text(encoding="utf-8"))
    panel = {(x["repo"], x["line"], str(x["token"])): x["verdict"] for x in adj["per_arm_detail"]["UNGROUNDED"]}
    removed, removed_genuine, remaining_false, remaining, repos, unresolved = 0, 0, 0, 0, 0, 0
    moves = collections.Counter()
    for rec in manifest["per_repo"]:
        if rec.get("status") != "CERTIFIED":
            continue
        repo, sha = rec["repo"], rec["sha"]
        blobs = {}
        ok = True
        for f in rec["files"]:
            key = hashlib.sha256(f"{repo}@{sha}/{f['path']}".encode()).hexdigest()[:32]
            b = CACHE / key
            if not b.exists() or hashlib.sha256(b.read_bytes()).hexdigest() != f["sha256"]:
                ok = False
                break
            blobs[f["path"]] = (f["role"], b.read_bytes())
        if not ok:
            unresolved += 1
            continue
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "README.md"
            rps = []
            for path, (role, data) in blobs.items():
                if role == "document":
                    d.write_bytes(data)
                else:
                    rp = Path(td) / path.replace("/", "__")
                    rp.write_bytes(data)
                    rps.append(rp)
            try:
                off, on = certify_both(d, rps)
            except Exception:                           # noqa: BLE001
                unresolved += 1
                continue
        repos += 1
        a, b = status_map(off), status_map(on)
        for k, s in a.items():
            if s != "UNGROUNDED":
                continue
            pv = panel.get((repo, k[0], k[2]))
            if b.get(k) != "UNGROUNDED":
                removed += 1
                moves[f"UNGROUNDED->{b.get(k)}"] += 1
                if pv == "CLAIM":
                    removed_genuine += 1
            else:
                remaining += 1
                if pv == "NOT_A_CLAIM":
                    remaining_false += 1
        if SMOKE and repos >= 8:
            break
    return {"repos": repos, "unresolved_repos": unresolved, "accusations_removed": removed,
            "removed_moves": dict(moves), "removed_genuine_by_panel": removed_genuine,
            "accusations_remaining": remaining, "remaining_false_by_panel": remaining_false}


def main() -> int:
    i, e = internal(), external()
    fa_before = 95 / 366                                  # oath_adjudication_result.json, 2026-08-27
    fa_after = (e["remaining_false_by_panel"] / e["accusations_remaining"]) if e["accusations_remaining"] else None
    metrics = {
        "internal_held_to_failed": i["held_to_failed"],
        "external_unresolved_repos": e["unresolved_repos"],
        "removed_genuine_share": round(e["removed_genuine_by_panel"] / e["accusations_removed"], 4) if e["accusations_removed"] else 0.0,
        "false_accusation_rate_gain": round(fa_before - fa_after, 4) if fa_after is not None else 0.0,
    }
    v = Experiment(PREREG, repo_root=ROOT).score(metrics, smoke=SMOKE)
    res = {"prereg": PREREG.name, "smoke": SMOKE, "internal": i, "external": e,
           "false_accusation_rate_before": round(fa_before, 4),
           "false_accusation_rate_after": round(fa_after, 4) if fa_after is not None else None,
           "metrics": metrics, "verdict": v.verdict, "gates": v.gates,
           "verifier_sha256": hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest()}
    out = HERE / ("range_sanity_report_ab_smoke.json" if SMOKE else "range_sanity_report_ab_result.json")
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("internal:", {k: i[k] for k in ("documents", "tokens", "tokens_moved", "moves", "held_to_failed", "failed_to_held")})
    print("external:", e)
    print("metrics:", metrics)
    print(f"\n===== VERDICT: {res['verdict']} =====\nwrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
