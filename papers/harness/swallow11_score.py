"""Score the SWALLOW-11 preregistration against the one-click receipt. Every prediction and gate is mechanical.

    python papers/harness/swallow11_score.py     # reads swallow11_receipt.json(.gz); writes swallow11_scored.json
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
PREREG = HERE / "PREREG_swallow11_one_click_from_loud_2026_09_22.md"
PREREG_SHA256_FROZEN = "1017fac935d1945880177a07d67f7656b79d069c1504a78619d858e354194ec4"
INSTRUMENT_SHA256_FROZEN = "047a11235ce35f02bb3c1e2e467857bf2c1e54d5cbd0b0fc9eede8dc6a6ff807"
ACTION_SHA256_FROZEN = "a9615d9bb909b4435fd301f4aa6496320af81828177766bf7c4718ee3c25a76f"
DIFFERENTIAL_LIVING_SHA256_FROZEN = "95f6ccf1f981a21882af152296d4ae6cd1f27c0d3a12fe66a3d22ea46b321428"
SOURCES_FROZEN = {"swallow7": "c6b12d09", "swallow9": "b78c4730", "swallow10": "cc6b5dc1"}
PAIRS = 140
RECEIPT = HERE / "swallow11_receipt.json"
RECEIPT_GZ = HERE / "swallow11_receipt.json.gz"
OUT = HERE / "swallow11_scored.json"
AGENTS = ("OpenAI_Codex", "Copilot", "Devin", "Cursor", "Claude_Code", "Google_Jules")


def load_receipt() -> tuple[dict, str]:
    raw = RECEIPT.read_bytes() if RECEIPT.exists() else gzip.decompress(RECEIPT_GZ.read_bytes())
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def _pct(a: int, b: int) -> float | None:
    return round(100.0 * a / b, 2) if b else None


def _count(items) -> dict:
    out: dict = {}
    for x in items:
        out[str(x)] = out.get(str(x), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def main() -> int:
    if hashlib.sha256(PREREG.read_bytes()).hexdigest() != PREREG_SHA256_FROZEN:
        print("INVALID__PREREG_MOVED")
        return 2
    r, receipt_sha = load_receipt()
    gates, P = {}, {}
    t = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests" / "test_harness_one_click.py"),
                        str(ROOT / "tests" / "test_ciaudit_action.py")], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT), timeout=1800)
    gates["G-S11-1"] = {"pass": t.returncode == 0, "detail": (t.stdout.strip().splitlines() or [""])[-1][:120]}

    pairs = r["pairs"]
    ok = [p for p in pairs if "error" not in p]
    rep = [p for p in ok if p.get("reproduced")]
    gates["G-S11-2"] = {"pass": len(pairs) == PAIRS and len(ok) >= 0.95 * PAIRS and len(rep) >= 0.95 * len(ok),
                        "detail": f"{len(ok)} of {len(pairs)} pairs re-read ({_pct(len(ok), len(pairs))}%); {len(rep)} of {len(ok)} give the receipt's set ({_pct(len(rep), len(ok))}%); "
                                  f"errors: {_count((p.get('error') or '')[:60] for p in pairs if 'error' in p)}"}
    src = r.get("sources_sha256", {})
    gates["G-S11-3"] = {"pass": r.get("instrument_sha256") == INSTRUMENT_SHA256_FROZEN and r.get("action_sha256") == ACTION_SHA256_FROZEN
                        and r.get("differential_living_sha256") == DIFFERENTIAL_LIVING_SHA256_FROZEN
                        and all(str(src.get(k, "")).startswith(v) for k, v in SOURCES_FROZEN.items()),
                        "detail": f"instrument {str(r.get('instrument_sha256'))[:16]}…, action {str(r.get('action_sha256'))[:16]}…, "
                                  f"living differential {str(r.get('differential_living_sha256'))[:16]}…, sources { {k: str(v)[:8] for k, v in src.items()} }"}
    checks = [dict(c, group=p["group"], source=p["source"], repo=p["repo"]) for p in ok for c in p["checks"]]
    rebuilt = [c for c in checks if c["rebuilt"]]
    gates["G-S11-4"] = {"pass": bool(checks) and all(c["reproduces"] == c["rebuilt"] for c in checks),
                        "detail": f"{sum(1 for c in rebuilt if c['reproduces'])} of {len(rebuilt)} rebuilt repairs reproduced by their suggestion; "
                                  f"{sum(1 for c in checks if c['reproduces'] and not c['rebuilt'])} reproduced without a rebuild"}

    def share(sel) -> tuple[int, int, float | None]:
        xs = [c for c in checks if sel(c)]
        k = sum(1 for c in xs if c["one_click"])
        return k, len(xs), (k / len(xs) if xs else None)

    located = [c for c in checks if c["located"]]
    P["P1"] = {"hit": bool(checks) and len(located) / len(checks) >= 0.98, "predicted": "≥ 98% of re-read checks located",
               "observed": {"located": len(located), "of": len(checks), "pct": _pct(len(located), len(checks)), "target": _count(c.get("target_what") for c in located)}}
    vis = sum(1 for c in located if c["visible"])
    P["P2"] = {"hit": bool(located) and vis / len(located) >= 0.80, "predicted": "≥ 80% of located annotations inside the change's diff",
               "observed": {"visible": vis, "of": len(located), "pct": _pct(vis, len(located)),
                            "not_visible_by_target": _count(c.get("target_what") for c in located if not c["visible"])}}
    k, n, sh = share(lambda c: True)
    P["P3"] = {"hit": sh is not None and sh >= 0.60, "predicted": "≥ 60% of re-read checks one click away",
               "observed": {"one_click": k, "of": n, "pct": _pct(k, n), "why_not": _count(c.get("why_not") for c in checks if not c["one_click"])}}
    kr = sum(1 for c in rebuilt if c["one_click"])
    P["P4"] = {"hit": bool(rebuilt) and kr / len(rebuilt) >= 0.80, "predicted": "≥ 80% of checks with a rebuilt repair one click away",
               "observed": {"one_click": kr, "of": len(rebuilt), "pct": _pct(kr, len(rebuilt)),
                            "outside_by_repair": _count(c.get("repair") for c in rebuilt if not c["one_click"])}}
    oc = [c for c in checks if c["one_click"]]
    small = sum(1 for c in oc if c.get("size", 99) <= 3)
    P["P5"] = {"hit": bool(oc) and small / len(oc) >= 0.75, "predicted": "≥ 75% of one-click suggestions replace ≤ 3 HEAD lines",
               "observed": {"small": small, "of": len(oc), "pct": _pct(small, len(oc)), "sizes": _count(c.get("size") for c in oc),
                            "by_repair": _count(c.get("repair") for c in oc)}}
    kb, nb, sb = share(lambda c: c["kind"] == "born hidden")
    ka, na, sa = share(lambda c: c["kind"] == "acquired")
    P["P6"] = {"hit": sb is not None and sa is not None and sb > sa, "predicted": "one-click share of born-hidden checks > acquired",
               "observed": {"born_hidden": f"{kb} of {nb} ({_pct(kb, nb)}%)", "acquired": f"{ka} of {na} ({_pct(ka, na)}%)",
                            "became_a_check": "{} of {}".format(*share(lambda c: c["kind"] == "became a check, hidden")[:2])}}
    kp, np_, sp = share(lambda c: c["source"] in ("swallow9", "swallow10"))
    km, nm, sm = share(lambda c: c["source"] == "swallow7")
    P["P7"] = {"hit": sp is not None and sm is not None and sp >= sm, "predicted": "one-click share in pull requests ≥ in mainline commits",
               "observed": {"pull_requests": f"{kp} of {np_} ({_pct(kp, np_)}%)", "mainline": f"{km} of {nm} ({_pct(km, nm)}%)"}}
    kh, nh, shh = share(lambda c: c["group"] == "human")
    kg, ng, sg = share(lambda c: c["group"] in AGENTS)
    P["P8"] = {"hit": shh is not None and sg is not None and shh >= sg, "predicted": "one-click share of people's checks ≥ the agents'",
               "observed": {"people": f"{kh} of {nh} ({_pct(kh, nh)}%)", "agents": f"{kg} of {ng} ({_pct(kg, ng)}%)",
                            "agent_signed": "{} of {}".format(*share(lambda c: c["group"] == "agent-signed")[:2])}}

    exploratory = {"by_source": {s: "{} of {}".format(*share(lambda c, s=s: c["source"] == s)[:2]) for s in ("swallow7", "swallow9", "swallow10")},
                   "by_repair": {rp: "{} of {}".format(*share(lambda c, rp=rp: c.get("repair") == rp)[:2]) for rp in sorted({str(c.get("repair")) for c in checks})},
                   "by_verdict": {v: "{} of {}".format(*share(lambda c, v=v: c["verdict"] == v)[:2]) for v in sorted({c["verdict"] for c in checks})},
                   "repos_with_a_one_click_fix": len({c["repo"] for c in oc}), "repos": len({c["repo"] for c in checks}),
                   "not_reproduced_pairs": [(p["repo"], p.get("number") or p["head"][:8], p.get("expected"), p.get("got")) for p in ok if not p.get("reproduced")][:12],
                   "added_lines": _count(c.get("added") for c in oc), "seconds": r.get("seconds")}
    valid = all(gates[g]["pass"] for g in ("G-S11-1", "G-S11-2", "G-S11-3", "G-S11-4"))
    hits = sum(1 for p in P.values() if p["hit"])
    payload = {"valid": valid, "gates": gates, "predictions": P, "hits": hits, "of": len(P), "exploratory_not_scored": exploratory,
               "summary": r.get("summary"), "instrument_sha256": r.get("instrument_sha256"), "receipt_sha256": receipt_sha,
               "receipt_file": RECEIPT.name if RECEIPT.exists() else RECEIPT_GZ.name, "prereg_sha256_at_freeze": PREREG_SHA256_FROZEN}
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(("VALID" if valid else "INVALID") + f"  {hits}/{len(P)} HIT")
    for k, p in P.items():
        print(f"  {k}: {'HIT' if p['hit'] else 'MISS'}  {json.dumps(p.get('observed'))[:300]}")
    for g, d in gates.items():
        print(f"  {g}: {'pass' if d['pass'] else 'FAIL'} - {d['detail'][:300]}")
    print("  exploratory (not scored):", json.dumps(exploratory)[:900])
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
