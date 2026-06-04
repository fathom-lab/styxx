# -*- coding: utf-8 -*-
"""make_figures.py — the rest of the shareable thread: quantization breaking point, distillation QA on real
LLMs, and the cross-lingual meaning core. On-brand (dark, monospace) to match meaning_realdrift.png."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams["font.family"] = "monospace"
BG, GRID, GREEN, CYAN, RED, GRAY, WHITE = "#0a0a0a", "#1c1c1c", "#00ff6a", "#00ffff", "#ff4040", "#888888", "#ffffff"


def base(title, subtitle, w=9.2, h=5.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=170)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    fig.suptitle(title, color=WHITE, fontsize=15.5, fontweight="bold", y=0.99)
    ax.set_title(subtitle, color=CYAN, fontsize=12.5, pad=14)
    ax.tick_params(colors=GRAY)
    for s in ax.spines.values():
        s.set_color("#333333")
    ax.grid(True, color=GRID, lw=0.8)
    fig.text(0.5, 0.015, "styxx.meaning_integrity  ·  github.com/fathom-lab/styxx  ·  pip install styxx",
             ha="center", color=GREEN, fontsize=8.5)
    fig.subplots_adjust(top=0.84, bottom=0.13, left=0.12, right=0.95)
    return fig, ax


def save(fig, name):
    out = os.path.join(HERE, name)
    fig.savefig(out, facecolor=fig.get_facecolor()); print("wrote", out)


# (1) quantization breaking point
fig, ax = base("how much can you compress before MEANING breaks?", "a model vs its quantized self — reference-free")
bits = ["8-bit", "4-bit", "2-bit", "1-bit"]; agr = [1.000, 0.975, 0.670, 0.390]
ax.axhspan(0.90, 1.02, color=GREEN, alpha=0.07); ax.axhspan(-0.02, 0.80, color=RED, alpha=0.07)
cols = [GREEN if a > 0.9 else (CYAN if a > 0.8 else RED) for a in agr]
ax.plot(range(4), agr, "-", color=GRAY, lw=2.2, zorder=1)
ax.scatter(range(4), agr, c=cols, s=130, zorder=2)
for i, a in enumerate(agr):
    ax.annotate(f"{a:.2f}", (i, a), xytext=(0, 12), textcoords="offset points", ha="center", color=cols[i], fontsize=11, fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels(bits); ax.set_ylim(-0.02, 1.08)
ax.set_ylabel("meaning kept vs full precision", color="#cccccc", fontsize=11)
ax.text(3.05, 0.93, "meaning intact", color=GREEN, fontsize=10, ha="right")
ax.text(3.05, 0.04, "meaning broken", color=RED, fontsize=10, ha="right")
save(fig, "meaning_quantization.png")

# (2) distillation QA on real LLMs
fig, ax = base("do two models MEAN the same?", "a distilled model keeps its teacher's meaning — real open LLMs")
fig.subplots_adjust(left=0.27)
pairs = ["BLOOM ↔ GPT-2", "Qwen ↔ GPT-2", "Pythia-410m ↔ GPT-2", "Pythia-160m ↔ Pythia-410m", "DistilGPT-2 ↔ GPT-2"]
vals = [0.027, 0.137, 0.274, 0.541, 0.978]
cols = [RED, GRAY, GRAY, CYAN, GREEN]
ax.barh(range(5), vals, color=cols, height=0.6)
for i, v in enumerate(vals):
    ax.annotate(f"{v:.3f}", (v, i), xytext=(6, 0), textcoords="offset points", va="center", color=cols[i], fontsize=11, fontweight="bold")
ax.set_yticks(range(5)); ax.set_yticklabels(pairs, color="#cccccc", fontsize=10.5)
ax.set_xlim(0, 1.12); ax.set_xlabel("meaning agreement (reference-free)", color="#cccccc", fontsize=11)
ax.text(0.985, 4.32, "distilled child — meaning preserved", color=GREEN, fontsize=9.5, ha="right")
ax.grid(True, axis="x", color=GRID, lw=0.8); ax.grid(False, axis="y")
save(fig, "meaning_distillation.png")

# (3) cross-lingual meaning core
fig, ax = base("does a Chinese LM and an English LM mean the same?", "a shared meaning core, above chance — control-validated")
labels = ["concepts\nMISMATCHED\n(control)", "Chinese-BERT\n↔ English-BERT", "Chinese-ERNIE\n↔ English-MiniLM", "Chinese-GPT2\n↔ English-MiniLM"]
vals = [-0.002, 0.156, 0.344, 0.347]; cols = [RED, CYAN, CYAN, GREEN]
ax.bar(range(4), vals, color=cols, width=0.6)
for i, v in enumerate(vals):
    ax.annotate(f"{v:+.3f}", (i, v), xytext=(0, 8 if v >= 0 else -16), textcoords="offset points", ha="center", color=cols[i], fontsize=11, fontweight="bold")
ax.axhline(0, color="#444444", lw=1)
ax.set_xticks(range(4)); ax.set_xticklabels(labels, color="#cccccc", fontsize=9.5)
ax.set_ylim(-0.06, 0.42); ax.set_ylabel("cross-lingual meaning agreement", color="#cccccc", fontsize=11)
ax.text(0, -0.045, "collapses to zero", color=RED, fontsize=9.5, ha="center")
ax.grid(True, axis="y", color=GRID, lw=0.8); ax.grid(False, axis="x")
save(fig, "meaning_crosslingual.png")
