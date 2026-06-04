# -*- coding: utf-8 -*-
"""make_realdrift_figure.py — shareable figure of the killer result: same model, same steps, only the labels
differ -> the meaning monitor sends harmful and helpful fine-tuning opposite ways."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
steps = [0, 50, 100, 200, 400]
dmg = [0.1926, 0.0987, 0.1406, 0.0253, 0.0077]      # fine-tune on random labels (harmful)
ben = [0.1926, 0.5806, 0.4753, 0.4408, 0.4417]      # fine-tune on real categories (helpful)

plt.rcParams["font.family"] = "monospace"
fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=170)
fig.patch.set_facecolor("#0a0a0a"); ax.set_facecolor("#0a0a0a")

ax.plot(steps, ben, "-o", color="#00ff6a", lw=2.6, ms=8, label="fine-tune on real labels  (helpful)")
ax.plot(steps, dmg, "-o", color="#ff4040", lw=2.6, ms=8, label="fine-tune on random labels  (harmful)")
ax.fill_between([0, 400], 0, 0.077, color="#ff4040", alpha=0.07)

ax.annotate("HEALTHY", (400, ben[-1]), xytext=(300, 0.50), color="#00ff6a", fontsize=12, fontweight="bold")
ax.annotate("BROKEN", (400, dmg[-1]), xytext=(305, 0.07), color="#ff4040", fontsize=12, fontweight="bold")

ax.set_xlabel("fine-tuning steps", color="#cccccc", fontsize=11)
ax.set_ylabel("meaning-alignment to humans", color="#cccccc", fontsize=11)
ax.set_title("same model. same steps. only the labels differ.", color="#00ffff", fontsize=13, pad=16)
fig.suptitle("does fine-tuning keep the model's MEANING?", color="#ffffff", fontsize=16, fontweight="bold", y=0.99)
ax.tick_params(colors="#888888")
for s in ax.spines.values():
    s.set_color("#333333")
ax.grid(True, color="#1c1c1c", lw=0.8)
ax.set_ylim(-0.02, 0.66); ax.set_xlim(-10, 410)
leg = ax.legend(loc="center right", framealpha=0.0, fontsize=10.5)
for t in leg.get_texts():
    t.set_color("#dddddd")
fig.text(0.5, 0.015, "styxx.meaning_integrity  ·  github.com/fathom-lab/styxx  ·  pip install styxx",
         ha="center", color="#00ff6a", fontsize=8.5)
fig.subplots_adjust(top=0.84, bottom=0.13)

out = os.path.join(HERE, "meaning_realdrift.png")
fig.savefig(out, facecolor=fig.get_facecolor())
print("wrote", out)
