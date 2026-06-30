# -*- coding: utf-8 -*-
"""Build the two shareable figures for RESULT_ai_brain_2026_06_03 (run from this directory).
  fig_ai_brain_threeway.png  — LLM concept-geometry matches the human brain as well as 1.5M human
                                judgments do, beating trained embedders, vs the fMRI noise ceiling.
  fig_ai_brain_scale.png     — brain-likeness is geometry QUALITY, not scale (per-model RSA vs params).
All numbers from ai_brain_result.json + RESULT_ai_brain_2026_06_03.md (committed, lexical-controlled)."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.load(open(HERE / "ai_brain_result.json"))
ceil_lo, ceil_hi = d["noise_ceiling"]
BG="#1a1020"; INK="#F3ECF7"; SUB="#B79FCB"; LILAC="#C9A2F0"; CYAN="#65E0D8"; MUTE="#6E5A82"; CEIL="#33264a"; INST="#65A8E0"
plt.rcParams.update({"font.family":"DejaVu Sans"})


def threeway():
    # three-way comparison on the 53 nouns shared with THINGS/VICE (RESULT_ai_brain §"the three-way")
    rows = [("Human behavior\n1.5M odd-one-out judgments (VICE)", 0.247, CYAN, False),
            ("Language model\ngpt2-large — text only, off the shelf", 0.222, LILAC, True),
            ("Sentence embedder\nmpnet (trained on similarity)", 0.161, MUTE, False),
            ("Sentence embedder\nMiniLM (trained on similarity)", 0.156, MUTE, False)]
    fig, ax = plt.subplots(figsize=(11,6.2), dpi=170); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    fig.subplots_adjust(left=0.30, right=0.965, top=0.80, bottom=0.13)
    ax.axvspan(ceil_lo, ceil_hi, color=CEIL, zorder=0)
    ax.text(ceil_hi-0.004, len(rows)-0.42, "the brain's own reliability\n(ceiling — the most anything can score)",
            ha="right", va="center", fontsize=9.5, color=SUB, style="italic")
    ys = list(range(len(rows)))[::-1]
    for y,(label,val,col,hi) in zip(ys, rows):
        ax.barh(y, val, height=0.62, color=col, zorder=3, edgecolor=INK if hi else "none", linewidth=1.6 if hi else 0)
        ax.text(val+0.006, y, f"{val:.3f}", va="center", ha="left", fontsize=12, color=INK, fontweight="bold" if hi else "normal")
        ax.text(-0.014, y, label, va="center", ha="right", fontsize=10.3, color=INK if hi else SUB, fontweight="bold" if hi else "normal")
    ax.axvline(0.002, color=MUTE, lw=1, ls=(0,(2,2)), zorder=2)
    ax.text(0.002, -0.72, "chance\n(shuffled)", ha="center", va="top", fontsize=8.5, color=MUTE)
    ax.set_xlim(0,0.60); ax.set_ylim(-0.95, len(rows)-0.30); ax.set_yticks([]); ax.set_xticks([0,0.1,0.2,0.3,0.4,0.5])
    for s in ("top","right","left"): ax.spines[s].set_visible(False)
    ax.set_xlabel("similarity to the human brain's geometry of meaning  (RSA → Mitchell-2008 fMRI, 60 concrete nouns)", fontsize=10.5, color=SUB)
    fig.text(0.012,0.975,"A language model means the same things the brain does", fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(0.012,0.905,"A free, off-the-shelf, text-only LLM matches the human brain's structure of concrete meaning\n"
             "as well as 1.5M human judgments do — and beats embedders purpose-built for it.", fontsize=11, color=SUB, va="top", linespacing=1.35)
    fig.text(0.30,0.025,"fathom-lab · styxx   ·   lexical-controlled, noise-ceiling-relative, vision-confound tested   ·   brain↔behavior model ranking ρ = 0.98", fontsize=8.6, color=MUTE)
    fig.savefig(HERE/"fig_ai_brain_threeway.png", facecolor=BG); plt.close(fig)


def scale():
    pm = d["per_model"]; M = [(m, pm[m]["params"], pm[m]["brain"], pm[m]["instruct"]) for m in pm]
    fig, ax = plt.subplots(figsize=(11,6.6), dpi=170); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    fig.subplots_adjust(left=0.085, right=0.97, top=0.80, bottom=0.13)
    ax.axhspan(ceil_lo, ceil_hi, color=CEIL, zorder=0)
    ax.text(4.1e9,(ceil_lo+ceil_hi)/2,"brain's own noise ceiling\n(max anything can score)", ha="right", va="center", fontsize=9, color=SUB, style="italic")
    for m,p,b,inst in M: ax.scatter(p,b,s=130,c=(INST if inst else LILAC),edgecolor=BG,linewidth=1,zorder=3)
    lab={"gpt2-large":(0,12),"gpt2-xl":(0,-16),"pythia-70m":(-4,10),"gemma-2-2b":(6,-14),"Qwen-3B":(6,8),"Phi-3.5":(6,-14),"gpt2":(6,8),"Llama-3B":(8,-4)}
    for m,p,b,inst in M:
        if m in lab: ax.annotate(m,(p,b),textcoords="offset points",xytext=lab[m],fontsize=9.2,color=INK,ha="center")
    ax.set_xscale("log"); ax.set_xlabel("model size (parameters, log scale)", fontsize=10.5, color=SUB)
    ax.set_ylabel("similarity to the brain's\nmeaning-geometry  (RSA)", fontsize=10.5, color=SUB)
    ax.tick_params(colors=SUB); ax.set_ylim(0,0.60)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(MUTE)
    ax.scatter([],[],c=LILAC,label="base model",s=110); ax.scatter([],[],c=INST,label="instruct-tuned",s=110)
    ax.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=INK)
    fig.text(0.085,0.95,"scale doesn't buy brain-likeness", fontsize=20, fontweight="bold", color=INK, va="top")
    fig.text(0.085,0.895,"matching the human brain's geometry of meaning tracks the quality of a model's concept space — not its size.\n"
             "a 774M GPT-2 tops 3B+ modern models; the smallest model here (70M) beats most of them.", fontsize=10.6, color=SUB, va="top", linespacing=1.35)
    fig.text(0.085,0.028,"fathom-lab · styxx   ·   partial-lexical RSA to Mitchell-2008 fMRI (60 concrete nouns)   ·   noisy-geometry gpt2-124M is the floor", fontsize=8.3, color=MUTE)
    fig.savefig(HERE/"fig_ai_brain_scale.png", facecolor=BG); plt.close(fig)


if __name__ == "__main__":
    threeway(); scale(); print("wrote fig_ai_brain_threeway.png + fig_ai_brain_scale.png")
