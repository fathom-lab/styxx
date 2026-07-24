# -*- coding: utf-8 -*-
"""Two-panel figure for RESULT_consistency_horizon. (A) the probabilistic consistency horizon:
oscillation solves at every distance; decay's solve rate decays with the premise->claim gap. (B) the
mechanism: decay's solve rate tracks the surviving signal mag_max^gap."""
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).parent
r = json.load(open(HERE / "consistency_horizon_v2_result.json"))
gaps = np.array(r["config"]["gaps"], float)
res = r["result"]
free = np.array(res["free_solve_by_gap"], float)
psolve = np.array(res["clamped_solve_rate_by_gap"], float)
surv = np.array(res["surviving_signal_by_gap"], float)
Hstar = res["half_horizon_gap"]
rho = res["mechanism_spearman_psolve_vs_surviving"]

AUB, LIL, INK, MUT, WARN = "#2E1A2F", "#B07FD1", "#EDE6F0", "#6E5A70", "#E8A0C0"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": AUB, "axes.facecolor": AUB})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 4.5), gridspec_kw={"wspace": 0.28})

# --- Panel A: the horizon ---
axA.plot(gaps, free, "-o", color=LIL, lw=2.2, ms=6, label="FREE  (oscillation)")
axA.plot(gaps, psolve, "-o", color=WARN, lw=2.2, ms=6, label="CLAMPED  (decay, θ≡0)")
axA.axhline(0.5, ls=(0, (4, 3)), lw=1, color=INK, alpha=0.4)
axA.axvline(Hstar, ls=(0, (2, 2)), lw=1.3, color=MUT)
axA.text(Hstar * 1.06, 0.10, f"half-horizon\n≈ gap {Hstar:.0f}", color=INK, fontsize=8.6, ha="left")
axA.set_xscale("log", base=2)
axA.set_xticks(gaps); axA.set_xticklabels([f"{int(g)}" for g in gaps], color=INK, fontsize=8.5)
axA.set_ylim(-0.05, 1.12); axA.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); axA.tick_params(colors=INK)
axA.set_xlabel("premise → claim gap (positions)", color=INK, fontsize=10)
axA.set_ylabel("solve rate  P(accuracy ≥ 0.90)", color=INK, fontsize=10)
axA.set_title("A · the consistency horizon", color=INK, fontsize=12, loc="left", pad=8)
axA.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK)
for s in ("top", "right"):
    axA.spines[s].set_visible(False)
for s in ("left", "bottom"):
    axA.spines[s].set_color(MUT)

# --- Panel B: the mechanism ---
axB.scatter(surv, psolve, s=70, color=WARN, edgecolor=INK, linewidth=0.6, zorder=3)
for x, y, g in zip(surv, psolve, gaps):
    axB.annotate(f"{int(g)}", (x, y), textcoords="offset points", xytext=(6, 5),
                 color=MUT, fontsize=7.5)
axB.set_xlabel("surviving premise signal  mag_max$^{gap}$", color=INK, fontsize=10)
axB.set_ylabel("decay solve rate", color=INK, fontsize=10)
axB.set_ylim(-0.05, 1.12); axB.tick_params(colors=INK)
axB.set_title(f"B · mechanism  (Spearman {rho:.2f})", color=INK, fontsize=12, loc="left", pad=8)
axB.text(0.60, 1.03, "more signal a magnitude-limited\nchannel retains → higher odds it solves",
         color=MUT, fontsize=8.3)
for s in ("top", "right"):
    axB.spines[s].set_visible(False)
for s in ("left", "bottom"):
    axB.spines[s].set_color(MUT)

fig.suptitle("Decay's odds of keeping a fact consistent decay with distance — oscillation's don't",
             color=INK, fontsize=13, x=0.012, ha="left", y=1.02)
plt.savefig(HERE / "fig_consistency_horizon.png", dpi=170, facecolor=AUB, bbox_inches="tight")
print("wrote fig_consistency_horizon.png")
