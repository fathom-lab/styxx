"""Score the SWALLOW-10 preregistration against the baseline receipt. Every prediction and gate is mechanical.

    python papers/harness/swallow10_score.py     # reads swallow10_receipt.json(.gz); writes swallow10_scored.json
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
PREREG = HERE / "PREREG_swallow10_the_baseline_2026_09_22.md"
PREREG_SHA256_FROZEN = "aed0c1b4b9159036ccdfc76bb4a9e544e3fa9f1b4b92c422c6787920a5586cac"
INSTRUMENT_SHA256_FROZEN = "7d764d5db800da3986a9b5ea4f5d620addd3a794d0bc037fa6b9c6eeebe3afe7"
SAMPLE_SHA256_FROZEN = "854447b532015b9a5cb997aa0913f72c72d576bdcfb4307b0e1cef7f194de6ce"
DIFFERENTIAL_SHA256_FROZEN = "91e4a4a755b80027"
AGENT_PRS_SHA256_FROZEN = "ec9ef750"
AUTHORSHIP_SHA256_FROZEN = "c3fb6e42"
REPOS = 609
RECEIPT = HERE / "swallow10_receipt.json"
RECEIPT_GZ = HERE / "swallow10_receipt.json.gz"
SAMPLE = HERE / "swallow10_sample.json"
SAMPLE_GZ = HERE / "swallow10_sample.json.gz"
OUT = HERE / "swallow10_scored.json"
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

    t = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests" / "test_harness_human_prs.py")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT), timeout=1800)
    gates["G-S10-1"] = {"pass": t.returncode == 0, "detail": (t.stdout.strip().splitlines() or [""])[-1][:120]}
    s = r["summary"]
    prs = [dict(p, repo=x["repo"]) for x in r["repos"] for p in x["prs"]]
    aud = [p for p in prs if p.get("audited")]
    touch = [p for p in aud if p.get("touching")]
    human = [p for p in touch if p["group"] == "human"]
    agents = [p for p in touch if p["group"] in AGENTS]
    auto = [p for p in touch if p["group"] == "automation"]
    repos_audited = len({p["repo"] for p in aud})
    gates["G-S10-2"] = {"pass": repos_audited >= 500 and len(human) >= 600 and s["clone_failures"] <= 0.10 * REPOS,
                        "detail": f"{repos_audited} repositories audited; {len(human)} touching human pull requests; {s['clone_failures']} of {REPOS} repositories failed to clone; "
                                  f"{s['audited']} audited of {s['prs']} ({s['human_sampled']} human sampled of {s['non_agent_in_span']} in span); missing head {s['missing_head']}, no base {s['no_base']}, capped {s['capped']}"}
    sample_sha = hashlib.sha256(SAMPLE.read_bytes() if SAMPLE.exists() else gzip.decompress(SAMPLE_GZ.read_bytes())).hexdigest()
    gates["G-S10-3"] = {"pass": r.get("instrument_sha256") == INSTRUMENT_SHA256_FROZEN and r.get("sample_file_sha256") == SAMPLE_SHA256_FROZEN and sample_sha == SAMPLE_SHA256_FROZEN
                        and str(r.get("differential_sha256", "")).startswith(DIFFERENTIAL_SHA256_FROZEN) and str(r.get("agent_prs_sha256", "")).startswith(AGENT_PRS_SHA256_FROZEN)
                        and str(r.get("authorship_sha256", "")).startswith(AUTHORSHIP_SHA256_FROZEN),
                        "detail": f"instrument {str(r.get('instrument_sha256'))[:16]}…, sample {str(r.get('sample_file_sha256'))[:16]}… (file {sample_sha[:16]}…), "
                                  f"differential {str(r.get('differential_sha256'))[:8]}…, agent_prs {str(r.get('agent_prs_sha256'))[:8]}…, authorship {str(r.get('authorship_sha256'))[:8]}…"}
    ba, fa = s["base_agreement9"], s["fires_agreement9"]
    gates["G-S10-4"] = {"pass": ba["compared"] > 0 and ba["agree"] / ba["compared"] >= 0.90 and fa["compared"] > 0 and fa["agree"] / fa["compared"] >= 0.95,
                        "detail": f"base agrees on {ba['agree']} of {ba['compared']} ({_pct(ba['agree'], ba['compared'])}%); the gate agrees on {fa['agree']} of {fa['compared']} ({_pct(fa['agree'], fa['compared'])}%)"}
    mh = s["merged_heuristic_on_dataset_truth"]
    gates["G-S10-5"] = {"pass": mh["merged"] > 0 and mh["recalled"] / mh["merged"] >= 0.85 and (mh["not_merged"] == 0 or mh["false_positive"] / mh["not_merged"] <= 0.05),
                        "detail": f"recalled {mh['recalled']} of {mh['merged']} dataset-merged ({_pct(mh['recalled'], mh['merged'])}%); flagged {mh['false_positive']} of {mh['not_merged']} not merged "
                                  f"({_pct(mh['false_positive'], mh['not_merged'])}%); by signal {mh.get('recalled_by')}"}

    def rate(group: list[dict]) -> float | None:
        return sum(1 for p in group if p["fires"]) / len(group) if group else None

    def checks(group: list[dict]) -> list[dict]:
        return [c for p in group if p["fires"] for d in p.get("detail", []) for c in d["new_hidden"]]

    hr, ar = rate(human), rate(agents)
    hc, ac = checks(human), checks(agents)
    P["P1"] = {"hit": hr is not None and 0.004 <= hr <= 0.016, "predicted": "human fire rate in [0.4%, 1.6%]",
               "observed": {"firing": sum(1 for p in human if p["fires"]), "touching": len(human), "rate_pct": _pct(sum(1 for p in human if p["fires"]), len(human)), "checks": len(hc)}}
    P["P2"] = {"hit": hr is not None and ar is not None and hr > 0 and ar / hr >= 1.2, "predicted": "agents' fire rate ≥ 1.2× the human rate (this pipeline)",
               "observed": {"agents_firing": sum(1 for p in agents if p["fires"]), "agents_touching": len(agents), "agents_rate_pct": _pct(sum(1 for p in agents if p["fires"]), len(agents)),
                            "human_rate_pct": _pct(sum(1 for p in human if p["fires"]), len(human)), "ratio": round(ar / hr, 2) if hr and ar is not None else None,
                            "by_agent": {a: {"touching": sum(1 for p in agents if p["group"] == a), "firing": sum(1 for p in agents if p["group"] == a and p["fires"])} for a in AGENTS if any(p["group"] == a for p in agents)}}}
    coe = sum(1 for c in hc if c.get("continue_on_error"))
    P["P3"] = {"hit": bool(hc) and coe / len(hc) >= 0.40, "predicted": "≥ 40% of human-brought checks carry continue-on-error",
               "observed": {"continue_on_error": coe, "of": len(hc), "pct": _pct(coe, len(hc)), "agents": f"{sum(1 for c in ac if c.get('continue_on_error'))} of {len(ac)}"}}
    rep = sum(1 for c in hc if (c.get("fix") or {}).get("verified_repair"))
    P["P4"] = {"hit": bool(hc) and rep / len(hc) >= 0.60, "predicted": "≥ 60% of human-brought checks have a verified repair",
               "observed": {"repaired": rep, "of": len(hc), "pct": _pct(rep, len(hc)), "by_repair": _count((c.get("fix") or {}).get("verified_repair") or "none" for c in hc),
                            "agents": f"{sum(1 for c in ac if (c.get('fix') or {}).get('verified_repair'))} of {len(ac)}"}}
    hf = [p for p in human if p["fires"]]
    hn = [p for p in human if not p["fires"]]
    mf = sum(1 for p in hf if p.get("merged_heuristic")) / len(hf) if hf else None
    mn = sum(1 for p in hn if p.get("merged_heuristic")) / len(hn) if hn else None
    P["P5"] = {"hit": mf is not None and mn and mf / mn <= 0.9, "predicted": "firing human pull requests merged ≤ 0.9× as often as non-firing (git signal)",
               "observed": {"firing_merged": f"{sum(1 for p in hf if p.get('merged_heuristic'))} of {len(hf)}", "non_firing_merged": f"{sum(1 for p in hn if p.get('merged_heuristic'))} of {len(hn)}",
                            "ratio": round(mf / mn, 3) if mf is not None and mn else None,
                            "agents_firing_merged": f"{sum(1 for p in agents if p['fires'] and p.get('merged_heuristic'))} of {sum(1 for p in agents if p['fires'])}",
                            "agents_non_firing_merged": f"{sum(1 for p in agents if not p['fires'] and p.get('merged_heuristic'))} of {sum(1 for p in agents if not p['fires'])}"}}
    born = sum(1 for c in hc if c.get("kind") == "born hidden")
    P["P6"] = {"hit": bool(hc) and born / len(hc) >= 0.50, "predicted": "≥ 50% of human-brought checks born hidden",
               "observed": {"born_hidden": born, "of": len(hc), "pct": _pct(born, len(hc)), "by_kind": _count(c.get("kind") for c in hc), "agents_by_kind": _count(c.get("kind") for c in ac)}}
    P["P7"] = {"hit": bool(auto) and rate(auto) <= 0.001, "predicted": "≤ 0.1% of touching automation pull requests fire",
               "observed": {"firing": sum(1 for p in auto if p["fires"]), "touching": len(auto), "rate_pct": _pct(sum(1 for p in auto if p["fires"]), len(auto))}}
    extra = [p for p in aud if p["group"] in AGENTS and not p.get("in_population9")]
    P["P8"] = {"hit": bool(extra) and sum(1 for p in extra if p.get("touching")) / len(extra) <= 0.05, "predicted": "≤ 5% of the agents' unlisted pull requests touch a workflow by git's diff",
               "observed": {"touching": sum(1 for p in extra if p.get("touching")), "of": len(extra), "pct": _pct(sum(1 for p in extra if p.get("touching")), len(extra)),
                            "firing": sum(1 for p in extra if p.get("fires"))}}

    # exploratory, not scored -- including the merge signal against the dataset restricted to pull requests the dataset saw closed
    # (a pull request open at collection can have been merged since; the frozen gate did not say so)
    closed_truth = [p for p in aud if "dataset_merged" in p and p.get("dataset_state") == "closed"]
    merge_signal_closed = {"merged": sum(1 for p in closed_truth if p["dataset_merged"]), "recalled": sum(1 for p in closed_truth if p["dataset_merged"] and p.get("merged_heuristic")),
                           "not_merged": sum(1 for p in closed_truth if not p["dataset_merged"]), "flagged": sum(1 for p in closed_truth if not p["dataset_merged"] and p.get("merged_heuristic")),
                           "open_at_collection_flagged": sum(1 for p in aud if "dataset_merged" in p and p.get("dataset_state") == "open" and p.get("merged_heuristic"))}
    merge_signal_closed["recall_pct"] = _pct(merge_signal_closed["recalled"], merge_signal_closed["merged"])
    merge_signal_closed["flagged_pct"] = _pct(merge_signal_closed["flagged"], merge_signal_closed["not_merged"])
    signed = [p for p in touch if p["group"] == "agent-signed"]
    top_repo = _count(p["repo"] for p in hf)
    without = [p for p in human if not top_repo or p["repo"] != next(iter(top_repo))]
    def shape(c: dict) -> str:
        rh = c.get("run_head", "")
        if c.get("continue_on_error"):
            return "continue-on-error"
        if "|| true" in rh:
            return "or-true"
        if "|| echo" in rh:
            return "default"
        if rh.startswith("set +e"):
            return "set +e"
        return "other (" + str(c.get("verdict")) + ")"
    hrepo = _count(p["repo"] for p in human)
    exploratory = {"merge_signal_on_pull_requests_closed_at_collection_NON_GATING": merge_signal_closed,
                   "agent_signed": {"touching": len(signed), "firing": sum(1 for p in signed if p["fires"]), "rate_pct": _pct(sum(1 for p in signed if p["fires"]), len(signed)),
                                    "checks": len(checks(signed))},
                   "human_firing_by_repo": top_repo,
                   "human_without_top_repo": {"touching": len(without), "firing": sum(1 for p in without if p["fires"]), "rate_pct": _pct(sum(1 for p in without if p["fires"]), len(without)),
                                              "ratio_agents_over_this": round(ar / (sum(1 for p in without if p["fires"]) / len(without)), 2) if without and any(p["fires"] for p in without) and ar else None},
                   "shapes": {"human": _count(shape(c) for c in hc), "agents": _count(shape(c) for c in ac)},
                   "mechanisms_of_acquired": {"human": _count(m for c in hc if c.get("kind") != "born hidden" for m in (c.get("mechanism") or [])),
                                              "agents": _count(m for c in ac if c.get("kind") != "born hidden" for m in (c.get("mechanism") or []))},
                   "human_repos_touching": len(hrepo), "human_firing_repos": len({p["repo"] for p in hf}), "checks_per_firing": {"human": round(len(hc) / len(hf), 2) if hf else None,
                                                                                                                                "agents": round(len(ac) / sum(1 for p in agents if p["fires"]), 2) if any(p["fires"] for p in agents) else None},
                   "repo_weighted_rates_pct": {"human": round(100 * sum((sum(1 for p in human if p["repo"] == rp and p["fires"]) / n) for rp, n in hrepo.items()) / len(hrepo), 2) if hrepo else None},
                   "touching_share_pct": {"human": _pct(len(human), sum(1 for p in aud if p["group"] == "human")), "agents": _pct(len(agents), sum(1 for p in aud if p["group"] in AGENTS)),
                                          "automation": _pct(len(auto), sum(1 for p in aud if p["group"] == "automation"))},
                   "by_base_method": _count(p.get("base_method") for p in touch), "repos_deepened": s.get("repos_deepened"),
                   "median_seconds_per_touching_pr": (lambda v: v[len(v) // 2] if v else None)(sorted(p.get("seconds", 0) for p in touch)),
                   "removed_hidden": {"human": sum(1 for p in human if p.get("removed_hidden")), "agents": sum(1 for p in agents if p.get("removed_hidden"))}}

    valid = all(gates[g]["pass"] for g in ("G-S10-1", "G-S10-2", "G-S10-3", "G-S10-4", "G-S10-5"))
    hits = sum(1 for p in P.values() if p["hit"])
    payload = {"valid": valid, "gates": gates, "predictions": P, "hits": hits, "of": len(P), "exploratory_not_scored": exploratory,
               "summary": s, "seconds": r.get("seconds"), "instrument_sha256": r.get("instrument_sha256"), "receipt_sha256": receipt_sha,
               "receipt_file": RECEIPT.name if RECEIPT.exists() else RECEIPT_GZ.name, "prereg_sha256_at_freeze": PREREG_SHA256_FROZEN}
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(("VALID" if valid else "INVALID") + f"  {hits}/{len(P)} HIT")
    for k, p in P.items():
        print(f"  {k}: {'HIT' if p['hit'] else 'MISS'}  {json.dumps(p.get('observed'))[:300]}")
    for g, d in gates.items():
        print(f"  {g}: {'pass' if d['pass'] else 'FAIL'} - {d['detail'][:260]}")
    print("  exploratory (not scored):", json.dumps(exploratory)[:700])
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
