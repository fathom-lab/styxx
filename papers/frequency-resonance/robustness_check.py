# -*- coding: utf-8 -*-
"""
robustness_check.py — no new training. Re-derives the frequency-sweep finding from the stored
per-K accuracies to test whether the verdict survives (a) the 0.80 capacity threshold and
(b) per-seed variation. Hardens or qualifies RESULT_frequency_sweep_2026_06_04.md.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
r = json.load(open(HERE / "frequency_sweep_result.json"))
fracs = r["config"]["theta_fracs"]
kgrid = r["config"]["kgrid"]
keys = [f"{f:.4f}" for f in fracs]


def kcap_from_acc(acc, thr):
    cap = 0
    for K in kgrid:
        if acc[str(K)] >= thr:
            cap = K
    return cap


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d else 0.0


out = {"threshold_sensitivity": {}, "per_seed": {}}

print("=== (a) THRESHOLD SENSITIVITY — theta* and shape vs the capacity threshold ===")
print(f"{'thr':>5} {'theta*/pi':>9} {'peak':>5} {'base(0)':>8} {'nyq(.97)':>9} {'rho(theta,kcap)':>16} {'interior?':>9}")
for thr in [0.70, 0.75, 0.80, 0.85, 0.90]:
    kcaps = [kcap_from_acc(r["sweep"][k]["mean_acc"], thr) for k in keys]
    istar = int(np.argmax(kcaps))
    rho = spearman(fracs, kcaps)
    interior = 0 < istar < len(fracs) - 1
    out["threshold_sensitivity"][f"{thr:.2f}"] = {
        "theta_star_frac": fracs[istar], "peak": kcaps[istar],
        "baseline": kcaps[0], "nyquist": kcaps[-1], "spearman": round(rho, 3),
        "interior_peak": interior, "nyquist_below_baseline": kcaps[-1] < kcaps[0]}
    print(f"{thr:>5.2f} {fracs[istar]:>9.3f} {kcaps[istar]:>5.1f} {kcaps[0]:>8.1f} "
          f"{kcaps[-1]:>9.1f} {rho:>16.3f} {str(interior):>9}")

print("\n=== (b) PER-SEED CONSISTENCY (threshold 0.80) ===")
print(f"{'seed':>5} {'theta*/pi':>9} {'peak':>5} {'base(0)':>8} {'nyq(.97)':>9} {'interior?':>9} {'nyq<base?':>9}")
seed_stars = []
for s in r["sweep"][keys[0]]["per_seed"].keys():
    kcaps = [kcap_from_acc(r["sweep"][k]["per_seed"][s]["acc"], 0.80) for k in keys]
    istar = int(np.argmax(kcaps))
    interior = 0 < istar < len(fracs) - 1
    seed_stars.append(fracs[istar])
    out["per_seed"][s] = {"theta_star_frac": fracs[istar], "peak": kcaps[istar],
                          "baseline": kcaps[0], "nyquist": kcaps[-1],
                          "interior_peak": interior, "nyquist_below_baseline": kcaps[-1] < kcaps[0]}
    print(f"{s:>5} {fracs[istar]:>9.3f} {kcaps[istar]:>5.1f} {kcaps[0]:>8.1f} {kcaps[-1]:>9.1f} "
          f"{str(interior):>9} {str(kcaps[-1] < kcaps[0]):>9}")

# summary
thr_stars = [v["theta_star_frac"] for v in out["threshold_sensitivity"].values()]
thr_interior = all(v["interior_peak"] for v in out["threshold_sensitivity"].values())
thr_nyq = all(v["nyquist_below_baseline"] for v in out["threshold_sensitivity"].values())
seed_interior = all(v["interior_peak"] for v in out["per_seed"].values())
out["summary"] = {
    "theta_star_across_thresholds": thr_stars,
    "interior_peak_all_thresholds": thr_interior,
    "nyquist_below_baseline_all_thresholds": thr_nyq,
    "interior_peak_all_seeds": seed_interior,
    "theta_star_seed_spread_frac": [min(seed_stars), max(seed_stars)],
}
print("\n=== SUMMARY ===")
print("theta* across thresholds 0.70-0.90:", thr_stars, "-> interior at all:", thr_interior)
print("Nyquist below no-rhythm baseline at all thresholds:", thr_nyq)
print("interior peak in every seed:", seed_interior, "| per-seed theta* spread:",
      [min(seed_stars), max(seed_stars)])
(HERE / "robustness_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("wrote robustness_result.json")
