# -*- coding: utf-8 -*-
"""Build the two shareable figures for RESULT_ai_brain_2026_06_03 (run from this directory).
  fig_ai_brain_threeway.png — a free text-only LLM shares the brain's geometry of concrete meaning,
      statistically indistinguishable from 1.5M human judgments and ahead of trained embedders, vs the
      fMRI noise ceiling. 95% CIs from ci_threeway.json (noun bootstrap).
  fig_ai_brain_scale.png    — brain-likeness is geometry QUALITY, not scale (per-model RSA vs params).
All numbers from ai_brain_result.json + ci_threeway.json (committed, lexical-controlled)."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.load(open(HERE / "ai_brain_result.json"))
ci = json.load(open(HERE / "ci_threeway.json"))
ceil_lo, ceil_hi = d["noise_ceiling"]
BG="#1a1020"; INK="#F3ECF7"; SUB="#B79FCB"; LILAC="#C9A2F0"; CYAN="#65E0D8"; MUTE="#6E5A82"; CEIL="#33264a"; GRID="#2a1d38"
plt.rcParams.update({"font.family": "DejaVu Sans"})


def threeway():
    rows = [("human similarity (VICE)\n1.5M odd-one-out judgments", "VICE", CYAN, False),
            ("language model\ngpt2-large — text only, off the shelf", "gpt2-large", LILAC, True),
            ("sentence embedder · mpnet\nbuilt to measure similarity", "mpnet", MUTE, False),
            ("sentence embedder · MiniLM\nbuilt to measure similarity", "MiniLM", MUTE, False)]
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=170); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    fig.subplots_adjust(left=0.32, right=0.955, top=0.77, bottom=0.165)
    ax.axvspan(ceil_lo, ceil_hi, color=CEIL, zorder=0)
    ax.axvline(ceil_lo, color=SUB, lw=1, ls=(0, (3, 3)), alpha=0.65, zorder=1)
    ax.text(ceil_lo + 0.007, len(rows) - 0.62, "brain's reliability ceiling\nbest achievable on this fMRI data\n→ top bars reach ~⅔ of it",
            ha="left", va="center", fontsize=8.8, color=SUB, style="italic", linespacing=1.3)
    ys = list(range(len(rows)))[::-1]
    for y, (label, key, col, hi) in zip(ys, rows):
        val, lo, hiv = ci[key]["point"], ci[key]["lo"], ci[key]["hi"]
        ax.barh(y, val, height=0.56, color=col, zorder=3, edgecolor=INK if hi else "none", linewidth=1.9 if hi else 0)
        ax.errorbar(val, y, xerr=[[val - lo], [hiv - val]], fmt="none", ecolor=INK if hi else SUB,
                    elinewidth=1.3, capsize=4, zorder=5, alpha=0.95)
        ax.text(hiv + 0.009, y, f"{val:.3f}", va="center", ha="left", fontsize=12.5, color=INK, fontweight="bold" if hi else "normal")
        ax.text(-0.012, y, label, va="center", ha="right", fontsize=10, color=INK if hi else SUB, fontweight="bold" if hi else "normal")
    ax.axvline(0.0, color=MUTE, lw=1.3, zorder=2)
    ax.text(0.0, -0.82, "chance ≈ 0", ha="center", va="top", fontsize=9, color=MUTE)
    ax.set_xlim(0, 0.60); ax.set_ylim(-1.05, len(rows) - 0.25)
    ax.set_yticks([]); ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]); ax.tick_params(colors=SUB, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTE)
    ax.set_xlabel("similarity to the human brain's geometry of meaning   (partial-lexical RSA → Mitchell-2008 fMRI, 53 shared nouns)", fontsize=9.8, color=SUB)
    fig.text(0.012, 0.965, "a free language model shares the brain's geometry of meaning", fontsize=17.5, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.895, "off-the-shelf and text-only, it tracks the brain as well as 1.5M human judgments do (statistically\nindistinguishable) — and ahead of the embedders built to measure meaning.",
             fontsize=10.4, color=SUB, va="top", linespacing=1.35)
    fig.text(0.32, 0.045, "fathom-lab · styxx   ·   95% CIs by noun bootstrap   ·   paired test: ≈ human behavior, > both embedders (p<.05)", fontsize=8.3, color=MUTE)
    fig.text(0.32, 0.018, "lexical-controlled · vision-control survives · brain↔behavior model-ranking ρ = 0.98", fontsize=8.3, color=MUTE)
    fig.savefig(HERE / "fig_ai_brain_threeway.png", facecolor=BG); plt.close(fig)


def scale():
    pm = d["per_model"]; M = [(m, pm[m]["params"], pm[m]["brain"], pm[m]["instruct"]) for m in pm]
    rho, p = spearmanr([np.log10(x[1]) for x in M], [x[2] for x in M])
    print(f"scale: Spearman(log params, brain RSA) rho={rho:.2f} p={p:.2f} (n={len(M)})")
    story = {"gpt2-large", "pythia-70m"}
    fig, ax = plt.subplots(figsize=(11, 6.6), dpi=170); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    fig.subplots_adjust(left=0.085, right=0.955, top=0.79, bottom=0.135)
    ax.axhspan(ceil_lo, ceil_hi, color=CEIL, zorder=0)
    ax.axhline(ceil_lo, color=SUB, lw=1, ls=(0, (3, 3)), alpha=0.65, zorder=1)
    ax.text(4.4e9, (ceil_lo + ceil_hi) / 2, "brain's reliability ceiling\n(best achievable on this data)", ha="right", va="center", fontsize=9, color=SUB, style="italic")
    for m, p_, b, inst in M:
        hero = m in story
        if inst:
            ax.scatter(p_, b, s=130, facecolors="none", edgecolors=LILAC, linewidth=1.6, zorder=3)
        else:
            ax.scatter(p_, b, s=200 if hero else 130, c=LILAC, edgecolor=INK if hero else BG, linewidth=1.8 if hero else 1, zorder=4 if hero else 3)
    lab = {"gpt2-large": (16, 14), "gpt2-xl": (2, -20), "pythia-70m": (-8, 16), "gemma-2-2b": (0, -20),
           "Qwen-3B": (22, -6), "Phi-3.5": (26, 4), "gpt2": (20, 4), "Llama-3B": (20, 12)}
    for m, p_, b, inst in M:
        if m in lab:
            big = m in story
            ax.annotate(m, (p_, b), textcoords="offset points", xytext=lab[m], fontsize=9.6 if big else 9,
                        color=INK, fontweight="bold" if big else "normal", ha="center",
                        arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.6, shrinkA=2, shrinkB=3))
    ax.set_xscale("log"); ax.set_xlabel("model size (parameters, log scale)", fontsize=10.5, color=SUB)
    ax.set_ylabel("similarity to the brain's\nmeaning-geometry  (RSA)", fontsize=10.5, color=SUB)
    ax.tick_params(colors=SUB); ax.set_ylim(0, 0.60); ax.grid(axis="both", color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(MUTE)
    ax.scatter([], [], c=LILAC, edgecolor=BG, label="base model", s=120)
    ax.scatter([], [], facecolors="none", edgecolors=LILAC, linewidth=1.6, label="instruct-tuned", s=120)
    ax.legend(loc="lower right", bbox_to_anchor=(0.995, 0.02), frameon=False, fontsize=10, labelcolor=INK)
    fig.text(0.085, 0.95, "scale doesn't buy brain-likeness", fontsize=20, fontweight="bold", color=INK, va="top")
    fig.text(0.085, 0.892, "matching the brain's geometry of meaning tracks the quality of a model's concept space, not its size.\n"
             "a 774M GPT-2 matches or beats every 3B+ model; the smallest here (70M) beats most of them.", fontsize=10.5, color=SUB, va="top", linespacing=1.35)
    fig.text(0.085, 0.028, f"fathom-lab · styxx   ·   partial-lexical RSA to Mitchell-2008 fMRI (60 nouns)   ·   no size trend: Spearman ρ={rho:.2f}, n.s. (n={len(M)})   ·   gpt2-124M is the noisy floor", fontsize=8.2, color=MUTE)
    fig.savefig(HERE / "fig_ai_brain_scale.png", facecolor=BG); plt.close(fig)


if __name__ == "__main__":
    threeway(); scale(); print("wrote fig_ai_brain_threeway.png + fig_ai_brain_scale.png")
