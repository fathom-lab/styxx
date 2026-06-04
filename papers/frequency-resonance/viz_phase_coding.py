# -*- coding: utf-8 -*-
"""
viz_phase_coding.py — make the mechanism visible (no GPU). K ordered items are phasors; the phase
they carry is theta x (their age). As frequency theta climbs, the items spread around the clock
(separable -> high capacity), then WRAP and collide (aliasing -> collapse). This is the intuition
behind the resonance — illustrative of THEORY sec.2, the qualitative mechanism that survived the
analytic negative. Renders phase_coding_clock.png.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
K = 6                                   # items to hold in order
idx = np.arange(K)                      # item 0 = newest ... K-1 = oldest (relative age)

# four regimes across the band, labelled by what happens to the code
PANELS = [
    (0.05, "too slow", "items bunch near one phase\n-> not separable"),
    (2 / K, "resonant (~2pi/K)", "items tile the clock once\n-> maximally separable"),
    (0.70, "too fast", "phases overshoot,\nstart to fold back"),
    (1.00, "Nyquist", "phases collapse to 2 points\n-> items alias / collide"),
]

BG, FG, GREEN, CYAN, RED = "#0a0a0a", "#e6e6e6", "#00ff66", "#00e0ff", "#ff3b3b"
plt.rcParams.update({"font.family": "monospace", "figure.facecolor": BG, "axes.facecolor": BG,
                     "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG})
cols = plt.cm.turbo(np.linspace(0.1, 0.9, K))

fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), subplot_kw={"aspect": "equal"})
for ax, (frac, name, note) in zip(axes, PANELS):
    theta = frac * math.pi
    ph = (theta * idx) % (2 * math.pi)             # phase each item carries
    # unit circle
    tt = np.linspace(0, 2 * math.pi, 200)
    ax.plot(np.cos(tt), np.sin(tt), color="#333", lw=1)
    # item phasors (slight radius stagger so overlapping ones are visible)
    for j in range(K):
        r = 1.0 - 0.045 * j
        x, y = r * math.cos(ph[j]), r * math.sin(ph[j])
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=cols[j], lw=2))
        ax.plot([x], [y], "o", color=cols[j], ms=7, mec=BG)
    # MIN pairwise angular gap = the real separability proxy (catches clustering AND aliasing)
    gaps = [min(abs(ph[i] - ph[j]) % (2 * math.pi), 2 * math.pi - abs(ph[i] - ph[j]) % (2 * math.pi))
            for i in range(K) for j in range(i + 1, K)]
    min_gap_deg = math.degrees(min(gaps))
    ideal = 360.0 / K
    ax.set_title(f"θ = {frac:.2f}π · {name}", color=FG, fontsize=11)
    ax.text(0, -1.62, note, ha="center", va="top", color="#bbb", fontsize=8.5)
    ax.text(0, 1.42, f"min gap {min_gap_deg:4.0f}°  (ideal {ideal:.0f}°)", ha="center",
            color=(GREEN if min_gap_deg >= 0.6 * ideal else RED), fontsize=9)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.9, 1.6); ax.axis("off")

fig.suptitle("why memory is resonant in frequency:  K=6 items as phasors on the clock",
             color=FG, fontsize=13, y=0.99)
fig.text(0.5, 0.045, "the right frequency spreads items around the clock · too fast wraps them back onto "
         "each other — a seizure, not a ladder", ha="center", color="#999", fontsize=9.5)
fig.text(0.012, 0.010, "styxx · papers/frequency-resonance · illustrative of the phase-coding mechanism (THEORY §2)",
         color="#666", fontsize=7)
fig.tight_layout(rect=(0, 0.10, 1, 0.95))
out = HERE / "phase_coding_clock.png"
fig.savefig(out, dpi=150, facecolor=BG)
def _mingap(frac):
    ph = (frac * math.pi * idx) % (2 * math.pi)
    g = [min(abs(ph[i]-ph[j]) % (2*math.pi), 2*math.pi-abs(ph[i]-ph[j]) % (2*math.pi))
         for i in range(K) for j in range(i+1, K)]
    return round(math.degrees(min(g)), 1)
print("min angular gap (deg) per panel:", [(f"{frac:.2f}pi", _mingap(frac)) for frac, *_ in PANELS])
print("wrote", out.name)
