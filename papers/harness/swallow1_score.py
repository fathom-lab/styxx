"""Score the SWALLOW-1 preregistration against the census receipt. Every prediction is mechanical.

    python papers/harness/swallow1_score.py     # reads swallow1_receipt.json (+ the self-census), writes swallow1_scored.json
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREREG = HERE / "PREREG_swallow1_ci_steps_that_cannot_fail_2026_09_21.md"
PREREG_SHA256_FROZEN = "7342f633a8d738c596e83714cec90c160929944d04d3f1b0132f3c69c2d8d3a3"
POPULATION = HERE / "swallow1_population.json"
RECEIPT = HERE / "swallow1_receipt.json"
RECEIPT_GZ = HERE / "swallow1_receipt.json.gz"      # the repository carries the receipt gzipped (19 MB of JSON)
SELF = HERE / "swallow1_self_census.json"
OUT = HERE / "swallow1_scored.json"
VERIFY = ("test", "lint", "typecheck")


def cannot_fail(s: dict) -> bool:
    return s["verdict"] == "SWALLOWS" or (s["continue_on_error"] and s["verdict"] in ("PROPAGATES", "SWALLOWS"))


def main() -> int:
    if hashlib.sha256(PREREG.read_bytes()).hexdigest() != PREREG_SHA256_FROZEN:
        print("INVALID__PREREG_MOVED")
        return 2
    raw = RECEIPT.read_bytes() if RECEIPT.exists() else gzip.decompress(RECEIPT_GZ.read_bytes())
    receipt_sha256 = hashlib.sha256(raw).hexdigest()
    r = json.loads(raw.decode("utf-8"))
    repos = r["repos"]
    gates, P = {}, {}
    pop_sha = hashlib.sha256(POPULATION.read_bytes()).hexdigest()
    gates["G-S1-1"] = {"pass": r.get("population_sha256") == pop_sha and r.get("population_size") == 100 and len(json.loads(POPULATION.read_text())) == 100,
                       "detail": f"population sha256 {pop_sha[:16]}…, {r.get('population_size')} entries"}
    failed = [x for x in repos if x.get("clone_error")]
    gates["G-S1-2"] = {"pass": len(repos) - len(failed) >= 90, "detail": f"{len(repos) - len(failed)} cloned; failures: {[(x['repo'], x['clone_error']) for x in failed]}"}
    if SELF.exists():
        s = json.loads(SELF.read_text(encoding="utf-8"))
        by = {}
        for st in s["steps"]:
            by[st["verdict"]] = by.get(st["verdict"], 0) + 1
        gates["G-S1-3"] = {"pass": by.get("PROPAGATES") == 33 and by.get("TOOLLESS") == 5 and by.get("SWALLOWS", 0) == 0,
                           "detail": f"self-census: {by}"}
    else:
        gates["G-S1-3"] = {"pass": False, "detail": "no self-census file"}
    gates["G-S1-4"] = {"pass": True, "detail": "verdicts and categories are the instrument's; nothing reclassified"}

    steps = [s for x in repos for s in x.get("steps", [])]
    hand = [s for s in steps if not s.get("generated")]
    executed_repos = [x for x in repos if any(s["verdict"] in ("PROPAGATES", "SWALLOWS") for s in x.get("steps", []))]
    # P1
    n_cf = sum(1 for x in executed_repos if any(cannot_fail(s) for s in x["steps"]))
    P["P1"] = {"hit": n_cf >= len(executed_repos) / 2, "predicted": "≥ 50% of repos with an executed step have a step that cannot fail",
               "observed": {"repos_with_executed_step": len(executed_repos), "repos_with_cannot_fail": n_cf, "rate": round(n_cf / len(executed_repos), 3) if executed_repos else None}}
    # P2
    n_vcf = sum(1 for x in executed_repos if any(cannot_fail(s) and s["category"] in VERIFY for s in x["steps"]))
    P["P2"] = {"hit": n_vcf >= len(executed_repos) / 10, "predicted": "≥ 10% have a verification step that cannot fail",
               "observed": {"repos_with_verification_cannot_fail": n_vcf, "rate": round(n_vcf / len(executed_repos), 3) if executed_repos else None,
                            "repos": sorted(x["repo"] for x in executed_repos if any(cannot_fail(s) and s["category"] in VERIFY for s in x["steps"]))}}
    # P3 — median per-repo shell swallow rate, hand-written workflows only
    rates = []
    for x in repos:
        ex = [s for s in x.get("steps", []) if not s.get("generated") and s["verdict"] in ("PROPAGATES", "SWALLOWS")]
        if ex:
            rates.append(sum(1 for s in ex if s["verdict"] == "SWALLOWS") / len(ex))
    rates.sort()
    med = rates[len(rates) // 2] if rates else None
    P["P3"] = {"hit": med is not None and med < 0.05, "predicted": "median per-repo shell swallow rate < 5% (hand-written)", "observed": {"median": med, "repos": len(rates), "p75": rates[int(len(rates) * 0.75)] if rates else None, "max": rates[-1] if rates else None}}
    # P4
    git_repos = sorted({x["repo"] for x in repos for s in x.get("steps", []) if s.get("git_query_or_true")})
    P["P4"] = {"hit": len(git_repos) >= 3, "predicted": "≥ 3 repos with a git query whose failure becomes an empty answer", "observed": {"repos": git_repos, "count": len(git_repos)}}
    # P5
    gen_repos = sorted({x["repo"] for x in repos if any(s.get("generated") for s in x.get("steps", []))})
    gen_steps = [s for s in steps if s.get("generated")]
    gen_coe = sum(1 for s in gen_steps if s["continue_on_error"])
    P["P5"] = {"hit": len(gen_repos) >= 10 and gen_steps and gen_coe / len(gen_steps) >= 0.30,
               "predicted": "≥ 10 repos with *.lock.yml; ≥ 30% of their steps continue-on-error",
               "observed": {"repos": len(gen_repos), "generated_steps": len(gen_steps), "continue_on_error": gen_coe, "rate": round(gen_coe / len(gen_steps), 3) if gen_steps else None}}
    # P6
    bash_steps = [s for s in steps if s["verdict"] != "NOT_BASH"]
    toolless = sum(1 for s in bash_steps if s["verdict"] == "TOOLLESS")
    bad = sum(1 for s in bash_steps if s["verdict"] in ("SYNTAX", "TIMEOUT"))
    P["P6"] = {"hit": bash_steps and toolless / len(bash_steps) <= 0.25 and bad / len(bash_steps) <= 0.02,
               "predicted": "TOOLLESS ≤ 25% of bash steps; SYNTAX+TIMEOUT ≤ 2%",
               "observed": {"bash_steps": len(bash_steps), "toolless_rate": round(toolless / len(bash_steps), 3) if bash_steps else None, "syntax_timeout_rate": round(bad / len(bash_steps), 4) if bash_steps else None}}
    # P7
    ot = [s for s in hand if s["has_or_true"] and s["verdict"] in ("PROPAGATES", "SWALLOWS")]
    sw = sum(1 for s in ot if s["verdict"] == "SWALLOWS")
    P["P7"] = {"hit": bool(ot) and sw / len(ot) < 0.60, "predicted": "< 60% of hand-written executed steps containing '|| true' actually swallow",
               "observed": {"steps_with_or_true": len(ot), "swallow": sw, "rate": round(sw / len(ot), 3) if ot else None}}

    valid = all(gates[g]["pass"] for g in ("G-S1-1", "G-S1-2", "G-S1-3"))
    hits = sum(1 for p in P.values() if p["hit"])
    payload = {"valid": valid, "gates": gates, "predictions": P, "hits": hits, "of": len(P),
               "summary": r.get("summary"), "instrument_sha256": r.get("instrument_sha256"), "receipt_sha256": receipt_sha256,
               "receipt_file": RECEIPT.name if RECEIPT.exists() else RECEIPT_GZ.name, "prereg_sha256_at_freeze": PREREG_SHA256_FROZEN}
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(("VALID" if valid else "INVALID") + f"  {hits}/{len(P)} HIT")
    for k, p in P.items():
        print(f"  {k}: {'HIT' if p['hit'] else 'MISS'}  {json.dumps(p['observed'])[:160]}")
    for g, d in gates.items():
        print(f"  {g}: {'pass' if d['pass'] else 'FAIL'} - {d['detail'][:200]}")
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
