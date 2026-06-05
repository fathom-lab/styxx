# -*- coding: utf-8 -*-
"""make_demarcation_figure.py — the visual thesis of the oscillation-memory arc, from committed JSONs.
Three panels: (A) rhythm is dominated by attention; (B) a band not a ladder; (C) a constant multiplier
not a scarcity mechanism. CPU-only. Noir house style."""
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BG, GREEN, CYAN, DIM = "#000000", "#00FF00", "#00FFFF", "#2a2a2a"
plt.rcParams.update({"font.family": "monospace", "text.color": GREEN, "axes.labelcolor": GREEN,
                     "xtick.color": GREEN, "ytick.color": GREEN, "axes.edgecolor": "#00aa00"})

fs = json.load(open(HERE / "frequency_sweep_result.json"))
nec = json.load(open(HERE / "necessity_attention_result.json"))
mux = json.load(open(HERE / "multiplexing_result.json"))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=BG)
for a in ax:
    a.set_facecolor(BG)
    for s in a.spines.values():
        s.set_color("#00aa00")

# --- Panel A: rhythm is dominated (necessity bars) ---
arms = [("LRU\nclamped", nec["arms"]["lru_clamped"]["mean_kcap"], "#6a6a6a"),
        ("LRU\nfree (osc)", nec["arms"]["lru_free"]["mean_kcap"], CYAN),
        ("Transformer\n(no rhythm)", nec["arms"]["transformer"]["mean_kcap"], GREEN)]
xs = range(len(arms))
ax[0].bar(xs, [v for _, v, _ in arms], color=[c for _, _, c in arms], edgecolor="#00aa00", width=0.62)
for i, (_, v, _) in enumerate(arms):
    ax[0].text(i, v + 0.3, f"{v:.1f}", ha="center", color=GREEN, fontsize=12, fontweight="bold")
ax[0].set_xticks(list(xs)); ax[0].set_xticklabels([n for n, _, _ in arms], fontsize=9)
ax[0].set_ylabel("ordered-memory capacity (kcap)")
ax[0].set_ylim(0, 17)
ax[0].set_title("A · rhythm is DOMINATED\noscillation helps recurrence, attention beats it",
                color=CYAN, fontsize=10.5, pad=10)

# --- Panel B: a band not a ladder (resonance curve) ---
sw = fs["sweep"]
th = sorted(float(k) for k in sw.keys())
kc = [sw[f"{t:.4f}"]["mean_kcap"] for t in th]
ax[1].plot(th, kc, "-o", color=CYAN, mfc=GREEN, mec=GREEN, lw=2, ms=6)
peak = th[int(np.argmax(kc))]
ax[1].axvline(peak, color="#00aa00", ls="--", lw=1)
ax[1].text(peak + 0.02, max(kc), f"  peak θ*≈{peak:.3f}π", color=GREEN, fontsize=9, va="top")
ax[1].axhline(kc[0], color=DIM, ls=":", lw=1)
ax[1].text(0.5, kc[0] - 0.45, "no-rhythm baseline", color="#888888", fontsize=8, ha="center")
ax[1].set_xlabel("oscillation frequency  θ / π  (rad/step)")
ax[1].set_ylabel("capacity (kcap)")
ax[1].set_title("B · a BAND, not a ladder\ncapacity is resonant; Nyquist is the MINIMUM",
                color=CYAN, fontsize=10.5, pad=10)

# --- Panel C: constant multiplier not scarcity (multiplexing ratio vs D) ---
ds = sorted(int(k) for k in mux["by_d"].keys())
ratios = [mux["by_d"][str(d)]["ratio"] for d in ds]
ax[2].plot(ds, ratios, "-o", color=GREEN, mfc=CYAN, mec=CYAN, lw=2, ms=6)
mean_r = float(np.mean(ratios))
ax[2].axhline(mean_r, color="#00aa00", ls="--", lw=1)
ax[2].text(ds[-1], mean_r + 0.06, f"  mean ~{mean_r:.2f}×  (flat: ρ=-0.07)", color=GREEN, fontsize=8.5, ha="right")
ax[2].set_xscale("log", base=2); ax[2].set_xticks(ds); ax[2].set_xticklabels(ds, fontsize=8)
ax[2].set_xlabel("state dimension D  (resource)")
ax[2].set_ylabel("oscillation advantage (free / clamped)")
ax[2].set_ylim(0, 3)
ax[2].set_title("C · a CONSTANT multiplier, not scarcity\nadvantage flat across a 16× resource range",
                color=CYAN, fontsize=10.5, pad=10)

fig.suptitle("what oscillation does for memory — and what it doesn't  ·  styxx / fathom-lab  ·  pre-registered, in-silico",
             color=GREEN, fontsize=11, y=1.02)
fig.tight_layout()
out = HERE / "demarcation_summary.png"
fig.savefig(out, dpi=140, facecolor=BG, bbox_inches="tight")
print("wrote", out.name)
