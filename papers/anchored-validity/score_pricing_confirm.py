"""Score the part-2c PRICING CONFIRMATION on the FRESH seeds only (12013-12024), against the
frozen PC gates (PREREG_part2c_pricing_confirm_2026_07_21.md). Does NOT touch the motivating
part-2c result. Reads the shared verdict cache; pure analysis. ASCII only."""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
from stage_b_corpus import build_corpus
from styxx.anchors import audit_panel

TAGS = ["0.5B", "1.5B", "3B", "7B4b"]
FRESH = list(range(12013, 12025))
N_ORG, K, PI = 200, 80, 0.35


def corpus(seed, style):
    org, anc, truth = build_corpus(seed, n_organic=N_ORG, k_anchor=K, pi=PI,
                                   family="attr", anchor_style=style)
    return (org, [a for a in anc if a["role"] == "neg_anchor"],
            [a for a in anc if a["role"] == "pos_anchor"], truth)


def main():
    cache = {}
    for line in (HERE / "p2c_crossmodel_cache.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line); cache[(d["model"], d["seed"])] = d
        except Exception:
            pass
    rows = []
    for seed in FRESH:
        if not all((t, seed) in cache for t in TAGS):
            continue
        org, negB, posB, truth = corpus(seed, "blatant")
        _, negL, posL, _ = corpus(seed, "ladder")
        y = np.array([truth[it["id"]] for it in org]); pit = float(y.mean())
        col = lambda key: np.array([cache[(t, seed)][key] for t in TAGS]).T
        aB = audit_panel(col("org"), col("negB"), col("posB"), n_boot=300, null_sims=200, seed=seed)
        aL = audit_panel(col("org"), col("negL"), col("posL"), n_boot=300, null_sims=200, seed=seed)
        deaf = audit_panel(col("org_d"), col("negB_d"), col("posB_d"), n_boot=100, null_sims=0,
                           seed=seed)
        rows.append({"seed": seed, "pi_true": pit,
                     "blatant_verdict": aB["verdict"], "blatant_pi": aB.get("pi"),
                     "blatant_cov": (bool(aB["ci"][0] <= pit <= aB["ci"][1])
                                     if aB.get("verdict") == "ESTIMATED" else None),
                     "blatant_err": (abs(aB["pi"] - pit) if aB.get("pi") is not None else None),
                     "ladder_verdict": aL["verdict"], "ladder_pi": aL.get("pi"),
                     "ladder_kept": aL.get("kept"),
                     "ladder_cov": (bool(aL["ci"][0] <= pit <= aL["ci"][1])
                                    if aL.get("verdict") == "ESTIMATED" else None),
                     "ladder_err": (abs(aL["pi"] - pit) if aL.get("pi") is not None else None),
                     "deaf_verdict": deaf["verdict"]})
    R = len(rows)
    out = {"prereg": "PREREG_part2c_pricing_confirm_2026_07_21.md", "seeds": FRESH,
           "n_replicates": R, "gates": [], "rows": rows}
    ok = True

    def gate(name, cond, detail):
        nonlocal ok
        ok = ok and bool(cond)
        out["gates"].append({"gate": name, "ok": bool(cond), "detail": detail})
        print(f"  [{'OK ' if cond else 'FAIL'}] {name}: {detail}")

    bl_est = [r for r in rows if r["blatant_verdict"] == "ESTIMATED"]
    bl_cov = sum(1 for r in bl_est if r["blatant_cov"])
    gate("PC1:kill_replicates", bl_cov <= 3,
         f"blatant coverage {bl_cov}/{len(bl_est)} ESTIMATED (<= 3)")

    ld_est = [r for r in rows if r["ladder_verdict"] == "ESTIMATED"]
    ld_cov = sum(1 for r in ld_est if r["ladder_cov"])
    ld_err = float(np.median([r["ladder_err"] for r in ld_est])) if ld_est else None
    bl_err = float(np.median([r["blatant_err"] for r in bl_est])) if bl_est else None
    margin = (bl_err - ld_err) if (ld_err is not None and bl_err is not None) else None
    out["ladder_median_err"] = ld_err
    out["blatant_median_err"] = bl_err
    out["err_margin"] = margin
    enough = len(ld_est) >= 8
    gate("PC2:pricing_recovers",
         enough and ld_cov >= 10 and margin is not None and margin >= 0.08,
         f"ladder cov {ld_cov}/{len(ld_est)} (>=10, >=8 est), ladder err {ld_err} vs blatant "
         f"{bl_err}, margin {None if margin is None else round(margin, 3)} (>= 0.08)")

    deafv = sum(1 for r in rows if str(r["deaf_verdict"]).startswith("VOID"))
    gate("PC3:deaf_void", deafv >= 11, f"{deafv}/{R} deaf VOID")

    out["all_gates_ok"] = ok
    (HERE / "stage_b_crossmodel_pricing_result.json").write_text(json.dumps(out, indent=1),
                                                                 encoding="utf-8")
    print(f"\nRESULT: all_gates_ok={ok} -> stage_b_crossmodel_pricing_result.json")


if __name__ == "__main__":
    main()
