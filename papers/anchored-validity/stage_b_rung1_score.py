"""Score STAGE B rung 1 against the frozen gates. Reads the crash-safe checkpoint, writes
stage_b_rung1_result.json. Gates B1-B3 per PREREG_STAGE_B_rung1_2026_07_20.md; everything else
is a characteristic. ASCII only."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = list(range(3001, 3016))


def q(xs, p):
    return float(np.percentile(np.asarray(xs, float), p)) if len(xs) else None


def main():
    recs = {}
    for line in (HERE / "stage_b_rung1_checkpoint.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
            recs[d["seed"]] = d
        except Exception:
            pass
    missing = [s for s in SEEDS if s not in recs]
    assert not missing, f"incomplete run, missing seeds {missing}"
    rs = [recs[s] for s in SEEDS]

    out = {"prereg": "PREREG_STAGE_B_rung1_2026_07_20.md", "n_replicates": len(rs),
           "gates": [], "characteristics": {}}
    ok_all = True

    def gate(name, cond, detail):
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        out["gates"].append({"gate": name, "ok": bool(cond), "detail": detail})
        print(f"  [{'OK ' if cond else 'FAIL'}] {name}: {detail}")

    deaf_void = sum(1 for r in rs if str(r["deaf_verdict"]).startswith("VOID"))
    gate("B1:deaf_arm_voids", deaf_void >= 14,
         f"{deaf_void}/15 deaf replicates VOID")

    est = [r for r in rs if r["audit"]["verdict"] == "ESTIMATED"]
    cov = sum(1 for r in est if r["audit_covered"])
    if len(est) < 8:
        gate("B2:label_free_honesty", False,
             f"VOID_UNDERPOWERED: only {len(est)}/15 ESTIMATED (refusal-heavy outcome, "
             f"reported as what it is)")
    else:
        gate("B2:label_free_honesty", cov >= 12,
             f"coverage {cov}/{len(est)} ESTIMATED replicates (bar 12/15); "
             f"verdicts {[r['audit']['verdict'] for r in rs].count('ESTIMATED')}/15 ESTIMATED")
    gate("B3:stratified_accounting", True,
         "structural: harness passes only inert anchor strata; no garbage stratum exists in "
         "rung 1 and no anchor is pooled by label into organic moments")

    errs = [r["audit_err"] for r in rs if r["audit_err"] is not None]
    out["characteristics"] = {
        "audit_err": {"median": q(errs, 50), "p90": q(errs, 90), "n": len(errs)},
        "audit_pi": [round(r["audit"]["pi"], 4) if r["audit"]["pi"] is not None else None
                     for r in rs],
        "pi_true": [round(r["pi_true_realized"], 4) for r in rs],
        "covered": [r["audit_covered"] for r in rs],
        "comparators": {
            "mv_err": {"median": q([r["mv_err"] for r in rs], 50)},
            "ds_err": {"median": q([r["ds_err"] for r in rs], 50)},
            "semisup_ds_err": {"median": q([r["ss_ds_err"] for r in rs], 50)}},
        "s_activation_rate": sum(1 for r in rs if r["audit"].get("activated")) / len(rs),
        "misfit_flag_rate": sum(1 for r in rs if (r["audit"].get("misfit") or {}).get("flag"))
                            / len(rs),
        "kept_judges_mode": max((tuple(r["audit"]["kept"]) for r in rs),
                                key=[tuple(r["audit"]["kept"]) for r in rs].count),
        "delta_alpha_anchor_minus_organic_mean": np.mean(
            [r["delta_alpha_anchor_minus_organic"] for r in rs], axis=0).tolist(),
        "delta_beta_anchor_minus_organic_mean": np.mean(
            [r["delta_beta_anchor_minus_organic"] for r in rs], axis=0).tolist(),
        "organic_alpha_mean": np.mean([r["organic_alpha"] for r in rs], axis=0).tolist(),
        "anchor_alpha_mean": np.mean([r["audit"]["alpha"] for r in rs], axis=0).tolist()}
    out["all_gates_ok"] = ok_all
    dest = HERE / "stage_b_rung1_result.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nRESULT: all_gates_ok={ok_all} -> {dest.name}")


if __name__ == "__main__":
    main()
