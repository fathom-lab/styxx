"""Score the SWALLOW-9 preregistration against the agent-pull-request receipt. Every prediction and gate is mechanical.

    python papers/harness/swallow9_score.py     # reads swallow9_receipt.json(.gz); writes swallow9_scored.json
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
PREREG = HERE / "PREREG_swallow9_the_agents_pull_request_at_the_gate_2026_09_21.md"
PREREG_SHA256_FROZEN = "6b81763692cc2de0bc461aae168934998c58e28ef4ce683f19cbd56e7e937bd5"
INSTRUMENT_SHA256_FROZEN = "ec9ef750826516a8ff425f184598d38b1249759b06bfe172518166e514b2b950"   # amended after run 1 (d591dc86…): the base rule; see RESULT deviation 1
INSTRUMENT_SHA256_RUN1 = "d591dc86712f43bde92aa48f7aa243b81bdff67e51cf11358d768f5f0474d233"
POPULATION_SHA256_FROZEN = "952509e5b731f352b8587ea46804c2cee2b86fc5effc72387f16216c768e459a"
DIFFERENTIAL_SHA256_FROZEN = "91e4a4a755b80027ed2e3e8c96d72be299ec44e2afce5ca35b402eecdba11c9e"
HISTORY_SHA256_FROZEN = "93efb4a9"
FAULTS_SHA256_FROZEN = "d26a407c"
POPULATION_SIZE, REPOS = 2286, 635
RECEIPT = HERE / "swallow9_receipt.json"
RECEIPT_GZ = HERE / "swallow9_receipt.json.gz"
POPULATION = HERE / "swallow9_population.json"
POPULATION_GZ = HERE / "swallow9_population.json.gz"
OUT = HERE / "swallow9_scored.json"


def load_receipt() -> tuple[dict, str]:
    raw = RECEIPT.read_bytes() if RECEIPT.exists() else gzip.decompress(RECEIPT_GZ.read_bytes())
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def _pct(a: int, b: int) -> float | None:
    return round(100.0 * a / b, 2) if b else None


def main() -> int:
    if hashlib.sha256(PREREG.read_bytes()).hexdigest() != PREREG_SHA256_FROZEN:
        print("INVALID__PREREG_MOVED")
        return 2
    r, receipt_sha = load_receipt()
    gates, P = {}, {}

    t = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests" / "test_harness_agent_prs.py")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT), timeout=1800)
    gates["G-S9-1"] = {"pass": t.returncode == 0, "detail": (t.stdout.strip().splitlines() or [""])[-1][:120]}
    s = r["summary"]
    gates["G-S9-2"] = {"pass": s["audited"] >= 0.80 * POPULATION_SIZE and s["clone_failures"] <= 0.10 * REPOS,
                       "detail": f"{s['audited']} of {POPULATION_SIZE} audited ({_pct(s['audited'], POPULATION_SIZE)}%); {s['clone_failures']} of {REPOS} repositories failed to clone "
                                 f"({s['prs_lost_to_clone_failures']} PRs); missing head {s['missing_head']}, no base {s['no_base']}, "
                                 f"suspect {s['suspect']}, head moved {s['head_moved']}, commit list capped {s.get('commit_list_capped', 0)}, capped {s['capped']}"}
    pop_sha = hashlib.sha256(POPULATION.read_bytes() if POPULATION.exists() else gzip.decompress(POPULATION_GZ.read_bytes())).hexdigest()
    gates["G-S9-3"] = {"pass": r.get("instrument_sha256") == INSTRUMENT_SHA256_FROZEN and r.get("population_file_sha256") == POPULATION_SHA256_FROZEN
                       and pop_sha == POPULATION_SHA256_FROZEN and r.get("differential_sha256") == DIFFERENTIAL_SHA256_FROZEN
                       and str(r.get("history_sha256", "")).startswith(HISTORY_SHA256_FROZEN) and str(r.get("faults_sha256", "")).startswith(FAULTS_SHA256_FROZEN),
                       "detail": f"instrument {str(r.get('instrument_sha256'))[:16]}…, population {str(r.get('population_file_sha256'))[:16]}… (file {pop_sha[:16]}…), "
                                 f"differential {str(r.get('differential_sha256'))[:16]}…, history {str(r.get('history_sha256'))[:8]}…, faults {str(r.get('faults_sha256'))[:8]}…"}

    prs = [dict(p, repo=x["repo"]) for x in r["repos"] for p in x["prs"]]
    aud = [p for p in prs if p.get("audited")]
    fires = [p for p in aud if p.get("fires")]
    checks = [c for p in fires for d in p.get("detail", []) for c in d["new_hidden"]]
    rate = len(fires) / len(aud) if aud else None
    P["P1"] = {"hit": rate is not None and 0.005 <= rate <= 0.02, "predicted": "fire rate among audited pull requests in [0.5%, 2.0%]",
               "observed": {"firing": len(fires), "audited": len(aud), "rate_pct": _pct(len(fires), len(aud)), "new_hidden_checks": len(checks)}}
    closed = [p for p in aud if p.get("state") == "closed"]
    cf = [p for p in closed if p.get("fires")]
    cn = [p for p in closed if not p.get("fires")]
    mf = sum(1 for p in cf if p.get("merged")) / len(cf) if cf else None
    mn = sum(1 for p in cn if p.get("merged")) / len(cn) if cn else None
    P["P2"] = {"hit": mf is not None and mn is not None and mf >= 0.8 * mn, "predicted": "merge rate of firing ≥ 0.8× merge rate of non-firing, among pull requests closed at collection",
               "observed": {"firing_merged": f"{sum(1 for p in cf if p.get('merged'))} of {len(cf)} ({_pct(sum(1 for p in cf if p.get('merged')), len(cf))}%)",
                            "non_firing_merged": f"{sum(1 for p in cn if p.get('merged'))} of {len(cn)} ({_pct(sum(1 for p in cn if p.get('merged')), len(cn))}%)",
                            "ratio": round(mf / mn, 3) if mf is not None and mn else None}}
    by_agent = {}
    for p in aud:
        a = by_agent.setdefault(p["agent"], {"audited": 0, "firing": 0, "checks": 0})
        a["audited"] += 1
        if p.get("fires"):
            a["firing"] += 1
            a["checks"] += p["new_hidden"]
    for a in by_agent.values():
        a["rate_pct"] = _pct(a["firing"], a["audited"])
    big = {k: v for k, v in by_agent.items() if v["audited"] >= 100}
    top = max(big, key=lambda k: big[k]["firing"] / big[k]["audited"]) if big else None
    ties = [k for k in big if top and big[k]["firing"] / big[k]["audited"] == big[top]["firing"] / big[top]["audited"]]
    P["P3"] = {"hit": top == "Copilot" and ties == ["Copilot"], "predicted": "Copilot has the highest fire rate among agents with ≥ 100 audited pull requests",
               "observed": {"highest": top, "tied": ties if len(ties) > 1 else None, "by_agent": dict(sorted(by_agent.items()))}}
    rep = sum(1 for c in checks if c.get("fix", {}).get("verified_repair"))
    P["P4"] = {"hit": bool(checks) and rep / len(checks) >= 0.60, "predicted": "≥ 60% of newly hidden checks have a verified repair",
               "observed": {"repaired": rep, "of": len(checks), "pct": _pct(rep, len(checks)),
                            "by_repair": _count(c.get("fix", {}).get("verified_repair") or "none" for c in checks)}}
    coe = sum(1 for c in checks if c.get("continue_on_error"))
    P["P5"] = {"hit": bool(checks) and coe / len(checks) >= 0.50, "predicted": "≥ 50% of newly hidden checks carry continue-on-error: true",
               "observed": {"continue_on_error": coe, "of": len(checks), "pct": _pct(coe, len(checks))}}
    ack = lambda p: bool((p.get("ack") or {}).get("title") or (p.get("ack") or {}).get("body"))  # noqa: E731
    nonfire_rate = {k: (sum(1 for p in aud if p["agent"] == k and not p.get("fires") and ack(p)) / max(1, sum(1 for p in aud if p["agent"] == k and not p.get("fires"))))
                    for k in by_agent}
    expected = sum(nonfire_rate[p["agent"]] for p in fires)
    observed = sum(1 for p in fires if ack(p))
    P["P6"] = {"hit": expected > 0 and observed / expected >= 1.5, "predicted": "acknowledging firing pull requests ≥ 1.5× the count expected from each agent's non-firing rate",
               "observed": {"acknowledging_firing": observed, "of": len(fires), "expected": round(expected, 2), "ratio": round(observed / expected, 2) if expected else None,
                            "non_firing_rate_pct_by_agent": {k: round(100 * v, 1) for k, v in sorted(nonfire_rate.items())},
                            "words": _count((p.get("ack") or {}).get("title") or (p.get("ack") or {}).get("body") for p in fires if ack(p))}}
    born = sum(1 for c in checks if c.get("kind") == "born hidden")
    P["P7"] = {"hit": bool(checks) and born / len(checks) >= 0.70, "predicted": "≥ 70% of newly hidden checks are born hidden",
               "observed": {"born_hidden": born, "of": len(checks), "pct": _pct(born, len(checks)), "by_kind": _count(c.get("kind") for c in checks)}}
    tips = [t for p in fires if p.get("merged") for d in p.get("detail", []) for t in (d.get("at_tip") or [])]
    present = [t for t in tips if t["now"] != "file gone"]
    still = sum(1 for t in present if t["now"] == "still hidden")
    P["P8"] = {"hit": bool(present) and still / len(present) >= 0.70, "predicted": "≥ 70% of the checks merged firing pull requests brought, whose file still exists at the tip, are still hidden",
               "observed": {"still_hidden": still, "of": len(present), "pct": _pct(still, len(present)), "file_gone": len(tips) - len(present), "by_now": _count(t["now"] for t in tips)}}

    # exploratory, not scored
    rc = _count(p["repo"] for p in aud)
    small = [p for p in aud if rc[p["repo"]] < 100]
    sf = [p for p in small if p.get("fires")]
    months = {}
    for p in aud:
        m = months.setdefault((p.get("created_at") or "")[:7], {"audited": 0, "firing": 0})
        m["audited"] += 1
        m["firing"] += int(bool(p.get("fires")))
    secs = sorted(p.get("seconds", 0) for p in aud)
    own = [p for p in aud if p.get("changed_in_git", 0) > 0]
    of_ = [p for p in own if p.get("fires")]
    exploratory = {"without_repos_of_100_or_more_prs": {"audited": len(small), "firing": len(sf), "rate_pct": _pct(len(sf), len(small)),
                                                         "by_agent": _rates(small)},
                   "own_workflow_change_only": {"audited": len(own), "firing": len(of_), "rate_pct": _pct(len(of_), len(own)), "by_agent": _rates(own),
                                                "merged_firing_closed": f"{sum(1 for p in of_ if p.get('merged') and p.get('state') == 'closed')} of {sum(1 for p in of_ if p.get('state') == 'closed')}",
                                                "merged_non_firing_closed": f"{sum(1 for p in own if not p.get('fires') and p.get('merged') and p.get('state') == 'closed')} of {sum(1 for p in own if not p.get('fires') and p.get('state') == 'closed')}"},
                   "no_workflow_change_of_its_own": {"audited": len(aud) - len(own), "by_agent": _count(p["agent"] for p in aud if p.get("changed_in_git", 0) == 0)},
                   "ambiguous_base": sum(1 for p in aud if p.get("ambiguous")), "head_in_tip_merged_firing": sum(1 for p in fires if p.get("merged") and p.get("head_in_tip")),
                   "by_month": dict(sorted(months.items())), "by_base_method": _count(p.get("base_method") for p in aud),
                   "firing_by_base_method": _count(p.get("base_method") for p in fires), "open_at_collection": {"audited": sum(1 for p in aud if p.get("state") == "open"),
                                                                                                             "firing": sum(1 for p in aud if p.get("state") == "open" and p.get("fires"))},
                   "firing_repos": len({p["repo"] for p in fires}), "checks_per_firing_pr": round(len(checks) / len(fires), 2) if fires else None,
                   "batch_firings_3_or_more": sum(1 for p in fires if p["new_hidden"] >= 3), "hidden_after_unread_prs": sum(1 for p in aud if p.get("hidden_after_unread")),
                   "removed_hidden_prs": sum(1 for p in aud if p.get("removed_hidden")), "suspect_firing": sum(1 for p in prs if p.get("suspect") and p.get("fires")),
                   "mechanisms_of_acquired": _count(m for c in checks if c.get("kind") != "born hidden" for m in (c.get("mechanism") or [])),
                   "median_seconds_per_audited_pr": secs[len(secs) // 2] if secs else None, "repos_deepened": s.get("repos_deepened"),
                   "verdicts": _count(c.get("verdict") for c in checks)}

    valid = all(gates[g]["pass"] for g in ("G-S9-1", "G-S9-2", "G-S9-3"))
    hits = sum(1 for p in P.values() if p["hit"])
    payload = {"valid": valid, "gates": gates, "predictions": P, "hits": hits, "of": len(P), "exploratory_not_scored": exploratory,
               "summary": s, "seconds": r.get("seconds"), "instrument_sha256": r.get("instrument_sha256"), "receipt_sha256": receipt_sha,
               "receipt_file": RECEIPT.name if RECEIPT.exists() else RECEIPT_GZ.name, "prereg_sha256_at_freeze": PREREG_SHA256_FROZEN}
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(("VALID" if valid else "INVALID") + f"  {hits}/{len(P)} HIT")
    for k, p in P.items():
        print(f"  {k}: {'HIT' if p['hit'] else 'MISS'}  {json.dumps(p.get('observed'))[:260]}")
    for g, d in gates.items():
        print(f"  {g}: {'pass' if d['pass'] else 'FAIL'} - {d['detail'][:220]}")
    print("  exploratory (not scored):", json.dumps(exploratory)[:600])
    return 0 if valid else 1


def _count(items) -> dict:
    out: dict = {}
    for x in items:
        out[str(x)] = out.get(str(x), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _rates(prs: list[dict]) -> dict:
    out: dict = {}
    for p in prs:
        a = out.setdefault(p["agent"], {"audited": 0, "firing": 0})
        a["audited"] += 1
        a["firing"] += int(bool(p.get("fires")))
    for a in out.values():
        a["rate_pct"] = _pct(a["firing"], a["audited"])
    return dict(sorted(out.items()))


if __name__ == "__main__":
    sys.exit(main())
