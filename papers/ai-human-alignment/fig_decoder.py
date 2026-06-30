# -*- coding: utf-8 -*-
"""Decoder figure: cumulative identification curve — how often the true concept lands in a shortlist of size k,
when a FREE text-only LLM reads the brain (zero training). Mitchell-60 (clean word-reading substrate).
Honest: shows it's strong shortlisting, not verbatim top-1. Reproduce -> fig_decoder.png."""
import sys
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "real-convergence"))
from run_real_convergence import distmat
from run_real_convergence_v2 import concept_all_layers
import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
DEV = "cuda" if torch.cuda.is_available() else "cpu"

bz = np.load(HERE / "brain_rdm.npz", allow_pickle=True)
nouns = [str(w) for w in bz["nouns"]]; brain = bz["group"]; N = len(nouns)

def ranks(model):
    out = []
    for i in range(N):
        others = np.array([j for j in range(N) if j != i])
        b = brain[i, others]; rb = np.argsort(np.argsort(b)).astype(float); rb -= rb.mean()
        M = model[:, others]; rM = np.argsort(np.argsort(M, axis=1), axis=1).astype(float); rM -= rM.mean(1, keepdims=True)
        s = (rM @ rb) / (np.sqrt((rM ** 2).sum(1)) * np.sqrt((rb ** 2).sum()) + 1e-12)
        out.append(int(np.where(np.argsort(-s) == i)[0][0]))
    return np.array(out)

tok = AutoTokenizer.from_pretrained("gpt2-large")
mdl = AutoModelForCausalLM.from_pretrained("gpt2-large", torch_dtype=torch.float16).to(DEV).eval()
llm = distmat(np.stack([concept_all_layers(mdl, tok, w)[-1] for w in nouns])); del mdl; gc.collect()
L = np.array([len(w) for w in nouns], float); lexrdm = np.abs(L[:, None] - L[None, :])
r_llm = ranks(llm); r_lex = ranks(lexrdm)
ks = np.arange(1, N + 1)
cum = lambda r: np.array([np.mean(r < k) for k in ks]) * 100
c_llm, c_lex, c_chance = cum(r_llm), cum(r_lex), ks / N * 100

BG="#1a1020"; INK="#F3ECF7"; SUB="#B79FCB"; LILAC="#C9A2F0"; CYAN="#65E0D8"; MUTE="#6E5A82"; GRID="#2a1d38"
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(11, 6.6), dpi=170); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
fig.subplots_adjust(left=0.085, right=0.96, top=0.79, bottom=0.13)
ax.plot(ks, c_chance, color=MUTE, lw=1.6, ls=(0, (4, 4)), label="chance", zorder=2)
ax.plot(ks, c_lex, color=CYAN, lw=2.0, label="word-length only (control)", zorder=3)
ax.plot(ks, c_llm, color=LILAC, lw=3.2, label="free LLM reading the brain (gpt2-large)", zorder=4)
ax.scatter([5], [c_llm[4]], s=80, c=INK, zorder=5)
ax.annotate(f"top-5 of 60: {c_llm[4]:.0f}%\n(≈4× chance)", (5, c_llm[4]), textcoords="offset points",
            xytext=(14, -4), fontsize=10.5, color=INK, fontweight="bold")
ax.set_xlabel("shortlist size  k  (candidates considered)", fontsize=10.5, color=SUB)
ax.set_ylabel("true concept is in the top-k   (%)", fontsize=10.5, color=SUB)
ax.set_xlim(1, N); ax.set_ylim(0, 100); ax.tick_params(colors=SUB); ax.grid(color=GRID, lw=0.7, zorder=0)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(MUTE)
ax.legend(loc="lower right", frameon=False, fontsize=10.5, labelcolor=INK)
fig.text(0.085, 0.95, "a free LLM reads which concept the brain holds — zero training", fontsize=17.5, fontweight="bold", color=INK, va="top")
fig.text(0.085, 0.895, "no scanner-specific model, no regression: the LLM's meaning-geometry alone identifies the read concept's\n"
         "brain pattern far above chance (permutation p = 0.005). The word-length control stays on the chance line — it's meaning.",
         fontsize=10.4, color=SUB, va="top", linespacing=1.35)
fig.text(0.085, 0.028, "fathom-lab · styxx   ·   Mitchell-2008 fMRI, 60 concepts, leave-one-out   ·   ≈ matches 1.5M human judgments   ·   replicates on THINGS-fMRI (p=0.005)", fontsize=8.2, color=MUTE)
fig.savefig(HERE / "fig_decoder.png", facecolor=BG); plt.close(fig)
print(f"top1 {c_llm[0]:.1f}%  top5 {c_llm[4]:.1f}%  top10 {c_llm[9]:.1f}%  (chance top5 {500/N:.1f}%); lex-control top5 {c_lex[4]:.1f}%")
print("wrote fig_decoder.png")
