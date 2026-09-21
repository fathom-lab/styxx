"""Score the SWALLOW-8 preregistration against the authorship receipt. Every prediction and gate is mechanical.

    python papers/harness/swallow8_score.py     # reads swallow8_receipt.json(.gz); writes swallow8_scored.json
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREREG = HERE / "PREREG_swallow8_who_writes_the_hidden_check_2026_09_21.md"
PREREG_SHA256_FROZEN = "316db0509151e4697c8f5c475812e212754e0d775f35017e38ec096bb535ecb3"
INSTRUMENT_SHA256_FROZEN = "c3fb6e422a5d068df9f57ac3d0aac641b4c140337f09a28a90e8688a677d06ab"
SOURCE_SHA256_FROZEN = "c6b12d0987592927fe19a7f997ddabb1efc28d6c1e1be3070aba894a2de7224b"
DIFFERENTIAL_SHA256_FROZEN = "91e4a4a755b80027ed2e3e8c96d72be299ec44e2afce5ca35b402eecdba11c9e"
RECEIPT = HERE / "swallow8_receipt.json"
RECEIPT_GZ = HERE / "swallow8_receipt.json.gz"
SOURCE = HERE / "swallow7_receipt.json.gz"
OUT = HERE / "swallow8_scored.json"


def load_receipt() -> tuple[dict, str]:
    raw = RECEIPT.read_bytes() if RECEIPT.exists() else gzip.decompress(RECEIPT_GZ.read_bytes())
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def _pct(a: int, b: int) -> float | None:
    return round(100.0 * a / b, 2) if b else None


def _year(t: int) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y") if t else "?"


def main() -> int:
    if hashlib.sha256(PREREG.read_bytes()).hexdigest() != PREREG_SHA256_FROZEN:
        print("INVALID__PREREG_MOVED")
        return 2
    r, receipt_sha = load_receipt()
    gates, P = {}, {}

    t = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests" / "test_harness_authorship.py")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT), timeout=1800)
    gates["G-S8-1"] = {"pass": t.returncode == 0, "detail": (t.stdout.strip().splitlines() or [""])[-1][:120]}
    s = r["summary"]
    gates["G-S8-2"] = {"pass": s["unclassified"] == 0 and s["repos"] >= 90, "detail": f"{s['commits']} commits, {s['unclassified']} unclassified, {s['repos']} repositories; missing clones: {len(r.get('missing_clones', []))}"}
    src_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    gates["G-S8-3"] = {"pass": r.get("instrument_sha256") == INSTRUMENT_SHA256_FROZEN and r.get("source_receipt_sha256") == SOURCE_SHA256_FROZEN
                       and src_sha == SOURCE_SHA256_FROZEN and r.get("differential_sha256") == DIFFERENTIAL_SHA256_FROZEN,
                       "detail": f"authorship {str(r.get('instrument_sha256'))[:16]}…, source {str(r.get('source_receipt_sha256'))[:16]}… (file {src_sha[:16]}…), differential {str(r.get('differential_sha256'))[:16]}…"}

    rows = [c for x in r["repos"] for c in x["commits"] if not c["root"]]
    by = {cls: [c for c in rows if c["class"] == cls] for cls in ("agent", "automation", "human")}
    checks = {cls: [x for c in by[cls] if c["fires"] for x in c.get("checks", [])] for cls in by}
    total_checks = sum(len(v) for v in checks.values())
    fire_rate = {cls: (sum(1 for c in by[cls] if c["fires"]) / len(by[cls]) if by[cls] else None) for cls in by}

    P["P1"] = {"hit": total_checks > 0 and len(checks["agent"]) / total_checks >= 0.20, "predicted": "≥ 20% of newly hidden checks brought by agent-class commits",
               "observed": {"agent": len(checks["agent"]), "human": len(checks["human"]), "automation": len(checks["automation"]), "total": total_checks, "pct_agent": _pct(len(checks["agent"]), total_checks)}}
    ratio = (fire_rate["agent"] / fire_rate["human"]) if fire_rate["agent"] and fire_rate["human"] else None
    P["P2"] = {"hit": ratio is not None and ratio >= 2.0, "predicted": "agent-class commits fire ≥ 2× as often as human-class commits",
               "observed": {"agent_commits": len(by["agent"]), "agent_firing": sum(1 for c in by["agent"] if c["fires"]), "agent_rate_pct": _pct(sum(1 for c in by["agent"] if c["fires"]), len(by["agent"])),
                            "human_commits": len(by["human"]), "human_firing": sum(1 for c in by["human"] if c["fires"]), "human_rate_pct": _pct(sum(1 for c in by["human"] if c["fires"]), len(by["human"])),
                            "ratio": round(ratio, 2) if ratio else None}}
    P["P3"] = {"hit": bool(by["automation"]) and fire_rate["automation"] <= 0.001, "predicted": "≤ 0.1% of automation-class commits fire",
               "observed": {"automation_commits": len(by["automation"]), "firing": sum(1 for c in by["automation"] if c["fires"]), "rate_pct": _pct(sum(1 for c in by["automation"] if c["fires"]), len(by["automation"]))}}
    rep = {cls: (sum(1 for x in checks[cls] if x["repair"]) / len(checks[cls]) if checks[cls] else None) for cls in by}
    P["P4"] = {"hit": rep["agent"] is not None and rep["human"] is not None and rep["agent"] >= rep["human"], "predicted": "verified-repair share of agent-brought checks ≥ human-brought",
               "observed": {"agent": f"{sum(1 for x in checks['agent'] if x['repair'])} of {len(checks['agent'])} ({_pct(sum(1 for x in checks['agent'] if x['repair']), len(checks['agent']))}%)",
                            "human": f"{sum(1 for x in checks['human'] if x['repair'])} of {len(checks['human'])} ({_pct(sum(1 for x in checks['human'] if x['repair']), len(checks['human']))}%)"}}
    per = {cls: (len(checks[cls]) / sum(1 for c in by[cls] if c["fires"]) if any(c["fires"] for c in by[cls]) else None) for cls in by}
    P["P5"] = {"hit": per["agent"] is not None and per["human"] is not None and per["agent"] >= 1.5 * per["human"], "predicted": "mean newly hidden checks per firing commit: agent ≥ 1.5× human",
               "observed": {"agent": round(per["agent"], 2) if per["agent"] else None, "human": round(per["human"], 2) if per["human"] else None,
                            "ratio": round(per["agent"] / per["human"], 2) if per["agent"] and per["human"] else None}}
    yr = s["by_year"]
    share = {y: (v["agent"] / (v["agent"] + v["automation"] + v["human"]) if (v["agent"] + v["automation"] + v["human"]) else 0) for y, v in yr.items()}
    s26, s24 = share.get("2026", 0), share.get("2024", 0)
    P["P6"] = {"hit": s26 >= 0.05 and (s24 == 0 or s26 >= 3 * s24), "predicted": "agent share of workflow-touching commits ≥ 5% in 2026 and ≥ 3× the 2024 share",
               "observed": {"share_by_year_pct": {y: round(100 * v, 2) for y, v in share.items()}, "commits_by_year": yr}}
    born = sum(1 for x in checks["agent"] if x["kind"] == "born hidden")
    P["P7"] = {"hit": bool(checks["agent"]) and born / len(checks["agent"]) >= 0.90, "predicted": "≥ 90% of agent-brought newly hidden checks are born hidden",
               "observed": {"born_hidden": born, "of": len(checks["agent"]), "pct": _pct(born, len(checks["agent"])),
                            "by_kind": {k: sum(1 for x in checks["agent"] if x["kind"] == k) for k in ("born hidden", "acquired", "became a check, hidden")},
                            "human_by_kind": {k: sum(1 for x in checks["human"] if x["kind"] == k) for k in ("born hidden", "acquired", "became a check, hidden")}}}

    # exploratory, not scored: the same rates within the years agents exist (2025-2026), and by signal
    recent = [c for c in rows if _year(c["time"]) in ("2025", "2026")]
    rb = {cls: [c for c in recent if c["class"] == cls] for cls in by}
    exploratory = {"fire_rate_2025_2026_pct": {cls: _pct(sum(1 for c in rb[cls] if c["fires"]), len(rb[cls])) for cls in rb},
                   "commits_2025_2026": {cls: len(rb[cls]) for cls in rb},
                   "agent_firings_by_signal": {}, "agent_commits_by_signal": s["by_class"]["agent"]["signals"],
                   "agent_firing_repos": sorted({x["repo"] for x in r["repos"] for c in x["commits"] if not c["root"] and c["class"] == "agent" and c["fires"]}),
                   "continue_on_error_pct": {cls: _pct(sum(1 for x in checks[cls] if x.get("continue_on_error")), len(checks[cls])) for cls in by}}
    for c in by["agent"]:
        if c["fires"]:
            exploratory["agent_firings_by_signal"][c["signal"]] = exploratory["agent_firings_by_signal"].get(c["signal"], 0) + 1

    valid = all(gates[g]["pass"] for g in ("G-S8-1", "G-S8-2", "G-S8-3"))
    hits = sum(1 for p in P.values() if p["hit"])
    payload = {"valid": valid, "gates": gates, "predictions": P, "hits": hits, "of": len(P), "exploratory_not_scored": exploratory,
               "summary": s, "seconds": r.get("seconds"), "instrument_sha256": r.get("instrument_sha256"), "receipt_sha256": receipt_sha,
               "receipt_file": RECEIPT.name if RECEIPT.exists() else RECEIPT_GZ.name, "prereg_sha256_at_freeze": PREREG_SHA256_FROZEN}
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(("VALID" if valid else "INVALID") + f"  {hits}/{len(P)} HIT")
    for k, p in P.items():
        print(f"  {k}: {'HIT' if p['hit'] else 'MISS'}  {json.dumps(p.get('observed'))[:230]}")
    for g, d in gates.items():
        print(f"  {g}: {'pass' if d['pass'] else 'FAIL'} - {d['detail'][:200]}")
    print("  exploratory (not scored):", json.dumps(exploratory)[:400])
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
