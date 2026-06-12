"""Render content_wall.png from content_wall_result.json showing ALL readouts (gate vs descriptive),
so the whitening-readout artifact is visible, not hidden. No models. Run after run_content_wall.py."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.loads((HERE / "content_wall_result.json").read_text())
P = d["mapped_primary"]; chance = d["chance_top1"]; gate3x = d["three_x_chance"]
npts = d["n_anchor_points"]; r2 = P["map_val_r2"]; verdict = d["verdict"]

labels = ["chance", "cycle-6\n(40 anchors)", "random-map\nfloor",
          "GATE: gemma-\nwhitened", "RAW\ncosine", "mapped-\nwhitened", "gemma\nceiling"]
vals = [chance, d["cycle6_baseline_llama_top1"], P["randmap_top1_p95"],
        P["gate_gemma_whitened"]["top1"], P["raw"]["top1"], P["mapped_whitened"]["top1"],
        d["in_model_ceiling_item_top1"]]
colors = ["#888", "#b04a4a", "#b0843a", "#c0392b", "#2e7d32", "#1f7a4d", "#155e35"]

fig, ax = plt.subplots(figsize=(9.4, 5.4))
bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.8)
ax.axhline(chance, color="#888", ls="--", lw=0.8)
ax.axhline(gate3x, color="#c0392b", ls=":", lw=0.9)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
ax.set_ylabel("top-1 retrieval accuracy (Llama-3.2-3B -> gemma, 20-way)", fontsize=10)
ax.set_ylim(0, 1.08)
ax.set_title(f"The content wall is a READOUT artifact (720 anchor points, map R2 {r2})\n"
             f"gate (gemma-whitened) at chance; RAW + mapped-whitened transport content -- "
             f"frozen verdict {verdict}", fontsize=11)
ax.grid(True, axis="y", alpha=0.15)
fig.tight_layout()
fig.savefig(HERE / "content_wall.png", dpi=140)
print("figure -> content_wall.png")
