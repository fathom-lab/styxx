"""Score the SWALLOW-13 preregistration against the receipt. Every prediction and gate is mechanical.

    python papers/harness/swallow13_score.py     # reads swallow13_receipt.json.gz; writes swallow13_scored.json
"""
from __future__ import annotations

import gzip
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREREG = HERE / "PREREG_swallow13_the_frontier_2026_09_22.md"
PREREG_SHA256_FROZEN = "125836d2b841ce31fafd525461cacabbb3a1669eac355914baabdb16cb60ea65"
INSTRUMENT_SHA256_FROZEN = "7b3c2b129750292cf058458cc975f4d2257ea6e642e932e982d5513a4ccbd9d1"
PRODUCT_SHA256_FROZEN = {
    "styxx/ciaudit/engine.py": "5c219887e389ed671d082c18be220af4479d48d0bd152fb348997ba4f5368b24",
    "styxx/ciaudit/actions.py": "ecbcd4f94a050e5b983f91464e681f61f2504a5447c45c68f91e419459c8d42a",
    "styxx/ciaudit/repair.py": "4282dbfa4d5caf22bac274e0964603dc451db164df8c730ba94afee58189241a",
    "styxx/ciaudit/repair_structural.py": "af46c492656e4d1837c97e0ecb3f40088e782c04c306ac94f70dc4bde7846570",
    "styxx/ciaudit/repair_frontier.py": "0fb104f09b10f767d962073a4e6905ff075984d868280a12747e5595a418ae13",
}
POPULATION_SHA256_FROZEN = "6f65d8f755ce00bf7770cffffd9eba507510e7a6a71f28170d42a6b2103ce46a"
POPULATION_SIZE = 549
RECEIPT = HERE / "swallow13_receipt.json.gz"
OUT = HERE / "swallow13_scored.json"
TESTS = ("tests/test_ciaudit_frontier.py", "tests/test_harness_frontier.py", "tests/test_ciaudit.py", "tests/test_harness_live_click.py")


def main() -> int:
    if PREREG_SHA256_FROZEN and hashlib.sha256(PREREG.read_bytes()).hexdigest() != PREREG_SHA256_FROZEN:
        print("INVALID__PREREG_MOVED")
        return 2
    raw = RECEIPT.read_bytes()
    r = json.loads(gzip.decompress(raw).decode("utf-8"))
    s = r["summary"]
    gates, P = {}, {}
    t = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *[str(ROOT / x) for x in TESTS]],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT), timeout=3600)
    gates["G-S13-1"] = {"pass": t.returncode == 0, "detail": (t.stdout.strip().splitlines() or [""])[-1][:120]}
    gates["G-S13-2"] = {"pass": r["instrument_sha256"] == INSTRUMENT_SHA256_FROZEN and r["product_sha256"] == PRODUCT_SHA256_FROZEN
                        and r["population_sha256"] == POPULATION_SHA256_FROZEN,
                        "detail": f"instrument {r['instrument_sha256'][:16]}…, product files at the frozen hashes: "
                                  f"{sum(r['product_sha256'].get(k) == v for k, v in PRODUCT_SHA256_FROZEN.items())} of {len(PRODUCT_SHA256_FROZEN)}, "
                                  f"population {r['population_sha256'][:16]}…"}
    ok = [x for x in r["repos"] if "targets" in x and not x.get("error")]
    gates["G-S13-3"] = {"pass": len(r["repos"]) == POPULATION_SIZE and len(ok) >= 0.9 * POPULATION_SIZE,
                        "detail": f"{len(ok)} of {len(r['repos'])} fetched at the tip and audited without error or timeout "
                                  f"(fetch failed {s['fetch_failed']}, errors {s['errors']}, capped {s['capped']})"}
    gates["G-S13-4"] = {"pass": s["stage3_population"] >= 15, "detail": f"stage-3 population {s['stage3_population']}"}
    gates["G-S13-5"] = {"pass": s["stage3_again_differs"] == 0 and s["stage3_again_equal"] == s["stage3_population"],
                        "detail": f"stage 3 twice: {s['stage3_again_equal']} equal, {s['stage3_again_differs']} differ, of {s['stage3_population']}"}

    pop3, v3 = s["stage3_population"], s["stage3_verified"]
    share = v3 / pop3 if pop3 else 0.0
    P["P1"] = {"hit": pop3 > 0 and share >= 0.25, "predicted": "stage 3 verifies a repair for at least 25% of the stage-3 population",
               "observed": {"verified": v3, "of": pop3, "share": round(share, 3)}}
    d = s["distinct_scripts"]
    dshare = d["stage3_verified"] / d["stage3_population"] if d["stage3_population"] else 0.0
    P["P2"] = {"hit": d["stage3_population"] > 0 and dshare >= 0.20, "predicted": "counted once per distinct script, at least 20%",
               "observed": {"verified": d["stage3_verified"], "of": d["stage3_population"], "share": round(dshare, 3)}}
    P["P3"] = {"hit": s["stage3_repos"] >= 3, "predicted": "stage-3 repairs land in at least 3 repositories", "observed": {"repositories": s["stage3_repos"]}}
    fam = list(s["stage3_by_family"].items())
    P["P4"] = {"hit": bool(fam) and fam[0][0] == "hoist" and (len(fam) == 1 or fam[0][1] > fam[1][1]),
               "predicted": "the hoist family is the largest stage-3 family, strictly", "observed": s["stage3_by_family"]}
    allv = s["all_stages_verified"]
    ashare = allv / s["read"] if s["read"] else 0.0
    P["P5"] = {"hit": s["read"] > 0 and ashare >= 0.85, "predicted": "stages 1-3 verify at least 85% of the read targets",
               "observed": {"stage1": s["stage1_verified"], "stage2": s["stage2_verified"], "stage3": v3, "of": s["read"], "share": round(ashare, 3)}}
    P["P6"] = {"hit": s["stage3_candidates_loud_but_change_the_healthy_run"] >= 1,
               "predicted": "at least one stage-3 candidate is loud and rejected because it changes the healthy run",
               "observed": {"candidates": s["stage3_candidates_loud_but_change_the_healthy_run"]}}
    rshare = s["residue_read"] / s["residue"] if s["residue"] else 0.0
    P["P7"] = {"hit": s["residue"] > 0 and rshare >= 0.20, "predicted": "(a coin) routed or declared matches at least 20% of the residue",
               "observed": {"routed": s["residue_routed"], "declared": s["residue_declared"], "either": s["residue_read"], "of": s["residue"], "share": round(rshare, 3)}}
    lines = s["stage3_lines_changed"]
    med = statistics.median(lines) if lines else None
    P["P8"] = {"hit": med is not None and med <= 6, "predicted": "the median stage-3 repair changes at most 6 lines", "observed": {"median": med, "lines": lines}}

    valid = all(gates[g]["pass"] for g in ("G-S13-1", "G-S13-2", "G-S13-3", "G-S13-4", "G-S13-5"))
    gates["G-S13-6"] = {"pass": len(P) == 8, "detail": "P1–P8 scored"}
    hits = sum(1 for x in P.values() if x["hit"])
    payload = {"valid": valid, "gates": gates, "predictions": P, "hits": hits, "of": len(P), "receipt_sha256": hashlib.sha256(raw).hexdigest(),
               "prereg_sha256_at_freeze": PREREG_SHA256_FROZEN, "summary": s}
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(("VALID" if valid else "INVALID") + f"  {hits}/{len(P)} HIT")
    for k, x in P.items():
        print(f"  {k}: {'HIT' if x['hit'] else 'MISS'}  {json.dumps(x['observed'])[:300]}")
    for g, dd in gates.items():
        print(f"  {g}: {'pass' if dd['pass'] else 'FAIL'} - {dd['detail'][:300]}")
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
