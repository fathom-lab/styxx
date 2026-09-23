"""Score the SWALLOW-15 preregistration against the receipt and the battery. Every prediction and
gate is mechanical.

    python papers/harness/swallow15_score.py     # reads swallow15_receipt.json.gz and swallow15_battery.json
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
PREREG = HERE / "PREREG_swallow15_the_escape_that_isnt_2026_09_23.md"
PREREG_SHA256_FROZEN = "9c429903768701c88c3b8d026147b8ea9f4006a48ff992c690a9f4c03186348e"
INSTRUMENT_SHA256_FROZEN = "7014bc02c46f5344249ed1636d4778c217df6998aa93164f96974817d2e97036"
PRODUCT_SHA256_FROZEN = {
    "styxx/ciaudit/engine.py": "5c219887e389ed671d082c18be220af4479d48d0bd152fb348997ba4f5368b24",
    "styxx/ciaudit/actions.py": "ecbcd4f94a050e5b983f91464e681f61f2504a5447c45c68f91e419459c8d42a",
    "styxx/ciaudit/repair.py": "4282dbfa4d5caf22bac274e0964603dc451db164df8c730ba94afee58189241a",
    "styxx/ciaudit/repair_structural.py": "af46c492656e4d1837c97e0ecb3f40088e782c04c306ac94f70dc4bde7846570",
    "styxx/ciaudit/repair_frontier.py": "bd6eb5af4fe169e9207a0ad67d6d7977cfea306d399ef1ce57f3bde9237f7c09",
    "styxx/ciaudit/confine.py": "46a1b40bf99db0313db24362ec970dc1cede67287fca32806544146718404da6",
    "styxx/ciaudit/__init__.py": "5c021afe1f09ac4c51d0534b4af26db505bd36e30145776f32efa9a80cdc89c7",
    "styxx/ciaudit/differential.py": "dc79fc97e3084df90c31bc64139bd6c44fbd5ff277f0f3fbf35776b319f888b0",
}
POPULATION_SHA256_FROZEN = "21c0e00fdaad3296f641668a48c30d3d33600cdacaf2faed691b36e071e5d58b"
RECEIPT_COMPARED_SHA256_FROZEN = "9eba6f23d66ea62f8918b8f2c0fd067eb5cc3635976dd1c4fb8b3bb578ae7fb7"
BATTERY_SHA256_FROZEN = "9db078c19d3a21e4466e4f64e0367616ec58a36a2af516f886e4a08e170e925c"
POPULATION_SIZE = 5945
RECEIPT = HERE / "swallow15_receipt.json.gz"
BATTERY = HERE / "swallow15_battery.json"
OUT = HERE / "swallow15_scored.json"
TESTS = ("tests/test_ciaudit_confine.py", "tests/test_harness_confined.py", "tests/test_ciaudit.py", "tests/test_ciaudit_action.py",
         "tests/test_ciaudit_frontier.py", "tests/test_harness_empty_list.py")
WIPERS = ("actions/setup-node", "djylb/nps")


def _moved_is_background(rec: dict, moved_targets: list) -> bool:
    """Every target where the chosen repair moved has a candidate that backgrounds a command."""
    idx = {(t["workflow"], t["job"], t["index"]): t for t in rec.get("targets", [])}
    for key in moved_targets:
        t = idx.get(tuple(key))
        if not t:
            return False
        if not any(c["repair"].endswith("background-liveness") for c in t.get("candidates", [])):
            return False
    return True


def main() -> int:
    if hashlib.sha256(PREREG.read_bytes()).hexdigest() != PREREG_SHA256_FROZEN:
        print("INVALID__PREREG_MOVED")
        return 2
    raw = RECEIPT.read_bytes()
    r = json.loads(gzip.decompress(raw).decode("utf-8"))
    s = r["summary"]
    bat_raw = BATTERY.read_bytes()
    bat = json.loads(bat_raw.decode("utf-8"))
    gates, P = {}, {}

    t = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *[str(ROOT / x) for x in TESTS]],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT), timeout=3600)
    gates["G-S15-1"] = {"pass": t.returncode == 0, "detail": (t.stdout.strip().splitlines() or [""])[-1][:120]}

    frozen_ok = (r["instrument_sha256"] == INSTRUMENT_SHA256_FROZEN and r["product_sha256"] == PRODUCT_SHA256_FROZEN
                 and r["population_sha256"] == POPULATION_SHA256_FROZEN and r["compared_receipt_sha256"] == RECEIPT_COMPARED_SHA256_FROZEN
                 and hashlib.sha256(bat_raw).hexdigest() == BATTERY_SHA256_FROZEN)
    gates["G-S15-2"] = {"pass": frozen_ok,
                        "detail": f"instrument {r['instrument_sha256'][:16]}…, product {sum(r['product_sha256'].get(k) == v for k, v in PRODUCT_SHA256_FROZEN.items())} of {len(PRODUCT_SHA256_FROZEN)}, "
                                  f"population {r['population_sha256'][:12]}…, compared receipt {r['compared_receipt_sha256'][:12]}…, battery {hashlib.sha256(bat_raw).hexdigest()[:12]}…"}
    ok = [x for x in r["repos"] if x.get("confined") is not None and not x.get("error")]
    gates["G-S15-3"] = {"pass": len(r["repos"]) == POPULATION_SIZE and len(ok) >= 0.9 * POPULATION_SIZE,
                        "detail": f"{len(ok)} of {len(r['repos'])} audited confined without error (fetch failed {s['fetch_failed']}, errors {s['errors']})"}
    gates["G-S15-4"] = {"pass": s["canaries_breached"] == 0, "detail": f"canary breaches: {s['canaries_breached']}"}
    gates["G-S15-5"] = {"pass": s["differ_from_receipt"] == 0 and s["compared_to_receipt"] > 0,
                        "detail": f"{s['match_receipt']} of {s['compared_to_receipt']} match the receipt's core; {s['differ_from_receipt']} differ"}
    gates["G-S15-6"] = {"pass": bat["confined_neutralised"] == bat["items"] and not bat["any_reached_host"],
                        "detail": f"{bat['confined_neutralised']} of {bat['items']} neutralised confined; reached host: {bat['any_reached_host']}"}

    def share(a, b):
        return a / b if b else 0.0

    P["P1"] = {"hit": s["canaries_breached"] == 0, "predicted": "no canary breach across the population",
               "observed": {"canaries_breached": s["canaries_breached"], "breaches": r.get("canaries_breached", [])[:5]}}
    wipers = {x["repo"]: x for x in r["repos"] if x["repo"] in WIPERS}
    wipers_ok = all(w in wipers and not wipers[w].get("error") and wipers[w].get("confined") for w in WIPERS)
    P["P2"] = {"hit": wipers_ok and s["canaries_breached"] == 0, "predicted": "the two /-wipers audit confined on the bare machine, no error, the machine intact",
               "observed": {w: {"confined": wipers.get(w, {}).get("confined"), "error": wipers.get(w, {}).get("error"), "targets": wipers.get(w, {}).get("targets")} for w in WIPERS}}
    P["P3"] = {"hit": s["differ_from_receipt"] == 0 and s["compared_to_receipt"] > 0, "predicted": "the confined core matches the receipt for every repository compared",
               "observed": {"compared": s["compared_to_receipt"], "match": s["match_receipt"], "differ": s["differ_from_receipt"], "first_differs": s.get("differs", [])[:3]}}
    moved = s.get("verified_repair_moved", [])
    all_bg = all(_moved_is_background(next((x for x in r["repos"] if x["repo"] == m["repo"]), {}), m["targets"]) for m in moved)
    P["P4"] = {"hit": s["compared_to_receipt"] > 0 and share(s["compared_to_receipt"] - s["verified_repair_moved_repos"], s["compared_to_receipt"]) >= 0.99 and all_bg,
               "predicted": "verified_repair matches the receipt on ≥99% compared, every move a backgrounded command",
               "observed": {"moved_repos": s["verified_repair_moved_repos"], "of": s["compared_to_receipt"], "all_moves_background": all_bg, "moved": moved[:5]}}
    P["P5"] = {"hit": bat["confined_neutralised"] == bat["items"] and not bat["any_reached_host"] and bat["control_fired"] >= 8,
               "predicted": "every confined item neutralised, none reaches the host, ≥8 reach it unconfined",
               "observed": {"items": bat["items"], "confined_neutralised": bat["confined_neutralised"], "reached_host": bat["any_reached_host"], "control_fired": bat["control_fired"]}}
    per = bat.get("perimeter", {})
    P["P6"] = {"hit": per.get("read outside") == "yes" and per.get("chmod outside") == "yes" and per.get("udp send") == "yes",
               "predicted": "confined, a read, a chmod and a UDP send all still succeed", "observed": per}
    med = s.get("seconds_median_per_repo")
    P["P7"] = {"hit": med is not None and med <= 10, "predicted": "median seconds per repository ≤ 10", "observed": {"median": med}}

    valid = all(gates[g]["pass"] for g in ("G-S15-1", "G-S15-2", "G-S15-3", "G-S15-4", "G-S15-5", "G-S15-6"))
    gates["G-S15-7"] = {"pass": len(P) == 7, "detail": "P1–P7 scored"}
    hits = sum(1 for x in P.values() if x["hit"])
    payload = {"valid": valid, "gates": gates, "predictions": P, "hits": hits, "of": len(P), "receipt_sha256": hashlib.sha256(raw).hexdigest(),
               "battery_sha256": hashlib.sha256(bat_raw).hexdigest(), "prereg_sha256_at_freeze": PREREG_SHA256_FROZEN, "summary": s}
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(("VALID" if valid else "INVALID") + f"  {hits}/{len(P)} HIT")
    for k, x in P.items():
        print(f"  {k}: {'HIT' if x['hit'] else 'MISS'}  {json.dumps(x['observed'])[:300]}")
    for g, dd in gates.items():
        print(f"  {g}: {'pass' if dd['pass'] else 'FAIL'} - {dd['detail'][:300]}")
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
