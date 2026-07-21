"""Score hardening arc part 1 against the frozen gates (PREREG_hardening_part1_2026_07_20.md).
Reads the arm-keyed checkpoint, writes stage_b_hardening_result.json. ASCII only."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = {"repair_ladder_attr": range(4001, 4016),
        "numeric_blatant": range(5001, 5016),
        "temporal_blatant": range(6001, 6016)}


def q(xs, p):
    return float(np.percentile(np.asarray(xs, float), p)) if len(xs) else None


def main():
    recs = {}
    for line in (HERE / "stage_b_hardening_checkpoint.jsonl").read_text(
            encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
            recs[(d["arm"], d["seed"])] = d
        except Exception:
            pass
    out = {"prereg": "PREREG_hardening_part1_2026_07_20.md", "gates": [], "arms": {}}
    ok_all = True

    def gate(name, cond, detail):
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        out["gates"].append({"gate": name, "ok": bool(cond), "detail": detail})
        print(f"  [{'OK ' if cond else 'FAIL'}] {name}: {detail}")

    for arm, seeds in ARMS.items():
        rs = [recs[(arm, s)] for s in seeds if (arm, s) in recs]
        assert len(rs) == 15, f"{arm}: {len(rs)}/15 replicates present"
        est = [r for r in rs if r["audit"]["verdict"] == "ESTIMATED"]
        cov = sum(1 for r in est if r["audit_covered"])
        deaf = sum(1 for r in rs if str(r["deaf_verdict"]).startswith("VOID"))
        errs = [r["audit_err"] for r in rs if r["audit_err"] is not None]
        flag = sum(1 for r in rs if (r["audit"].get("misfit") or {}).get("flag"))
        da = np.mean([r["delta_alpha_anchor_minus_organic"] for r in rs], axis=0)
        kept_counts = np.mean([r["audit"]["kept"] for r in rs], axis=0)
        out["arms"][arm] = {
            "n_estimated": len(est), "covered": cov, "deaf_void": deaf,
            "verdicts": [r["audit"]["verdict"] for r in rs],
            "audit_err": {"median": q(errs, 50), "p90": q(errs, 90), "n": len(errs)},
            "misfit_flag_rate": flag / len(rs),
            "mean_delta_alpha": [round(float(x), 4) for x in da],
            "max_abs_mean_delta_alpha": float(np.max(np.abs(da))),
            "kept_rate_per_judge": [round(float(x), 3) for x in kept_counts],
            "comparators_median_err": {
                "mv": q([r["mv_err"] for r in rs], 50),
                "ds": q([r["ds_err"] for r in rs], 50),
                "ss_ds": q([r["ss_ds_err"] for r in rs], 50)},
            "s_activation_rate": sum(1 for r in rs if r["audit"].get("activated")) / len(rs)}
        gate(f"H1:deaf_void_{arm}", deaf >= 14, f"{deaf}/15 deaf VOID")

    rep = out["arms"]["repair_ladder_attr"]
    if rep["n_estimated"] < 8:
        gate("H2:repair_coverage", False,
             f"VOID_UNDERPOWERED: only {rep['n_estimated']}/15 ESTIMATED "
             f"(refusal resolution per the pre-run note -- reported as what it is)")
    else:
        gate("H2:repair_coverage", rep["covered"] >= 12,
             f"coverage {rep['covered']}/{rep['n_estimated']} (bar 12/15)")
    gate("H2:repair_alpha_gap_closes", rep["max_abs_mean_delta_alpha"] <= 0.10,
         f"max |mean delta_alpha| {rep['max_abs_mean_delta_alpha']:.4f} (bar 0.10; "
         f"rung-1 blatant measured up to 0.63)")
    for arm in ("numeric_blatant", "temporal_blatant"):
        a = out["arms"][arm]
        gate(f"H3:kill_transfers_{arm}", a["covered"] <= 3,
             f"coverage {a['covered']}/{a['n_estimated']} ESTIMATED (prediction <= 3/15)")

    out["all_gates_ok"] = ok_all
    dest = HERE / "stage_b_hardening_result.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nRESULT: all_gates_ok={ok_all} -> {dest.name}")


if __name__ == "__main__":
    main()
