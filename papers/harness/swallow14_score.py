"""Score the SWALLOW-14 preregistration against the receipt. Every prediction and gate is mechanical.

    python papers/harness/swallow14_score.py     # reads swallow14_receipt.json.gz; writes swallow14_scored.json
"""
from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREREG = HERE / "PREREG_swallow14_the_empty_list_2026_09_22.md"
PREREG_SHA256_FROZEN = "a34a06765aff3d25693a2e252dac38fb7a61e7a5d1c565681ab876571d93eb8d"
INSTRUMENT_SHA256_FROZEN = "9b40257a3b718f02bd8e729bb29e54df706bf7ec0d84809669d826e7490e6b5c"
PRODUCT_SHA256_FROZEN = {
    "styxx/ciaudit/engine.py": "5c219887e389ed671d082c18be220af4479d48d0bd152fb348997ba4f5368b24",
    "styxx/ciaudit/actions.py": "ecbcd4f94a050e5b983f91464e681f61f2504a5447c45c68f91e419459c8d42a",
    "styxx/ciaudit/repair.py": "4282dbfa4d5caf22bac274e0964603dc451db164df8c730ba94afee58189241a",
    "styxx/ciaudit/repair_structural.py": "af46c492656e4d1837c97e0ecb3f40088e782c04c306ac94f70dc4bde7846570",
    "styxx/ciaudit/repair_frontier.py": "c10b9975f18a73a02e0250f92207294d0730bdbf2daeec526b0c01d07d58a8c1",
}
POPULATION_SHA256_FROZEN = "21c0e00fdaad3296f641668a48c30d3d33600cdacaf2faed691b36e071e5d58b"
POPULATION_SIZE = 5945
RECEIPT = HERE / "swallow14_receipt.json.gz"
OUT = HERE / "swallow14_scored.json"
TESTS = ("tests/test_ciaudit_frontier.py", "tests/test_harness_empty_list.py", "tests/test_harness_frontier.py", "tests/test_ciaudit.py",
         "tests/test_harness_live_click.py")


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
    gates["G-S14-1"] = {"pass": t.returncode == 0, "detail": (t.stdout.strip().splitlines() or [""])[-1][:120]}
    gates["G-S14-2"] = {"pass": r["instrument_sha256"] == INSTRUMENT_SHA256_FROZEN and r["product_sha256"] == PRODUCT_SHA256_FROZEN
                        and r["population_sha256"] == POPULATION_SHA256_FROZEN,
                        "detail": f"instrument {r['instrument_sha256'][:16]}…, product files at the frozen hashes: "
                                  f"{sum(r['product_sha256'].get(k) == v for k, v in PRODUCT_SHA256_FROZEN.items())} of {len(PRODUCT_SHA256_FROZEN)}, "
                                  f"population {r['population_sha256'][:16]}…"}
    ok = [x for x in r["repos"] if "targets" in x and not x.get("error")]
    gates["G-S14-3"] = {"pass": len(r["repos"]) == POPULATION_SIZE and len(ok) >= 0.9 * POPULATION_SIZE,
                        "detail": f"{len(ok)} of {len(r['repos'])} fetched at the tip and audited without error or timeout "
                                  f"(fetch failed {s['fetch_failed']}, errors {s['errors']}, capped {s['capped']})"}
    gates["G-S14-4"] = {"pass": s["s13_residue"] >= 30, "detail": f"S13 residue {s['s13_residue']}"}
    gates["G-S14-5"] = {"pass": s["stage3_again_differs"] == 0 and s["stage3_again_equal"] == s["stage3_population"],
                        "detail": f"stage 3 twice: {s['stage3_again_equal']} equal, {s['stage3_again_differs']} differ, of {s['stage3_population']}"}

    def share(a, b):
        return a / b if b else 0.0

    P["P1"] = {"hit": s["s13_residue"] > 0 and share(s["new_reach"], s["s13_residue"]) >= 0.20,
               "predicted": "the new edits verify a repair for at least 20% of the S13 residue",
               "observed": {"verified": s["new_reach"], "of": s["s13_residue"], "share": round(share(s["new_reach"], s["s13_residue"]), 3),
                            "by_edit": s["new_reach_by_edit"]}}
    P["P2"] = {"hit": s["wait_class"] >= 10 and share(s["wait_class_verified"], s["wait_class"]) >= 0.50,
               "predicted": "wait-list verifies at least 50% of the wait class (fewer than 10 is a miss)",
               "observed": {"verified": s["wait_class_verified"], "of": s["wait_class"], "share": round(share(s["wait_class_verified"], s["wait_class"]), 3)}}
    P["P3"] = {"hit": s["global_not_local"] == 0, "predicted": "every check the global hoist verifies, the local hoist verifies too",
               "observed": {"global": s["hoist_global_verified"], "local": s["hoist_local_verified"], "global_not_local": s["global_not_local"]}}
    P["P4"] = {"hit": s["local_not_global"] >= 1, "predicted": "(a coin) the local hoist verifies at least one check the global hoist does not",
               "observed": {"local_not_global": s["local_not_global"]}}
    P["P5"] = {"hit": s["wait_repos"] >= 3, "predicted": "wait-list verifies a repair in at least 3 repositories",
               "observed": {"repositories": s["wait_repos"], "checks": s["wait_verified"]}}
    P["P6"] = {"hit": s["stage3_population"] > 0 and share(s["s13_verified"], s["stage3_population"]) >= 0.25,
               "predicted": "SWALLOW-13's stage verifies at least 25% of the stage-3 population",
               "observed": {"verified": s["s13_verified"], "of": s["stage3_population"], "share": round(share(s["s13_verified"], s["stage3_population"]), 3)}}
    P["P7"] = {"hit": s["read"] > 0 and share(s["all_stages_verified"], s["read"]) >= 0.80,
               "predicted": "stages 1, 2 and 3 as the product now runs them verify at least 80% of the read targets",
               "observed": {"stage1": s["stage1_verified"], "stage2": s["stage2_verified"], "stage3": s["all_stages_verified"] - s["stage1_verified"] - s["stage2_verified"],
                            "of": s["read"], "share": round(share(s["all_stages_verified"], s["read"]), 3)}}
    med = s["wait_chosen_lines_median"]
    P["P8"] = {"hit": med is not None and med <= 3, "predicted": "the median wait-list repair the product chooses changes at most 3 lines",
               "observed": {"median": med, "lines": s["wait_chosen_lines"]}}

    valid = all(gates[g]["pass"] for g in ("G-S14-1", "G-S14-2", "G-S14-3", "G-S14-4", "G-S14-5"))
    gates["G-S14-6"] = {"pass": len(P) == 8, "detail": "P1–P8 scored"}
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
