# -*- coding: utf-8 -*-
"""
run_scale_law_v3_depth.py — settle the scale question with a UNIFORM, non-selected read-out rule.

v1 (fixed 0.66) gave a null; v2 showed the answer flips with layer rule, but both alternatives are
confounded: best-by-cat_struct is a noisy selector, max-over-layers favours models with more layers
to maximize over (gpt2-xl 48 vs pythia-14m 6). This removes BOTH confounds: read every model at the
SAME RELATIVE DEPTH (no selection, no layer-count bias) and map the scale trend as a function of
depth. The honest question becomes: at a fixed relative depth, does semantic alignment rise with
scale -- and is it clean on the perfectly-controlled Pythia ladder?

EXPLORATORY (the pre-registered fixed-0.66 test FAILED and that stands). Decision rule stated up
front: a depth-robust positive requires pooled Spearman rho >= 0.5 AND Pythia rho > 0 across the
upper-depth band (0.8-1.0), where these models actually carry concept geometry.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"

from run_real_convergence import distmat, is_cached
from run_real_convergence_v2 import concept_all_layers
from run_real_convergence_v3_controls import partial_corr
from run_scale_law import LADDERS, spearman
import run_real_convergence as RC
import run_real_convergence_confirm as RCC

CONCEPTS = list(RC.CONCEPTS) + list(RCC.CONCEPTS)
N = len(CONCEPTS)
IU = np.triu_indices(N, 1)
DEPTHS = [0.5, 0.66, 0.8, 0.9, 1.0]   # 1.0 = final layer; uniform across all models


def main():
    charlen = np.array([len(w) for w in CONCEPTS], dtype=float)
    rtok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    toklen = np.array([len(rtok(" " + w, add_special_tokens=False).input_ids) for w in CONCEPTS], dtype=float)
    zc = (charlen - charlen.mean()) / (charlen.std() + 1e-9)
    zt = (toklen - toklen.mean()) / (toklen.std() + 1e-9)
    Zlex = np.column_stack([np.abs(zc[:, None] - zc[None, :])[IU], np.abs(zt[:, None] - zt[None, :])[IU]])

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEV)
    refIU = distmat(st.encode(CONCEPTS, normalize_embeddings=True))[IU]
    del st
    gc.collect()
    torch.cuda.empty_cache() if DEV == "cuda" else None

    results = []
    for ladder, members in LADDERS.items():
        for name, repo, params in members:
            if not is_cached(repo):
                continue
            tok = AutoTokenizer.from_pretrained(repo)
            mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, trust_remote_code=True).to(DEV).eval()
            L = mdl.config.num_hidden_layers
            allrep = np.stack([concept_all_layers(mdl, tok, w) for w in CONCEPTS])  # (N, L+1, d)
            depth_align = {}
            for d in DEPTHS:
                l = min(allrep.shape[1] - 1, max(1, round(d * L)))
                depth_align[d] = round(partial_corr(distmat(allrep[:, l, :])[IU], refIU, Zlex), 3)
            results.append({"ladder": ladder, "name": name, "params": params,
                            "log10_params": round(float(np.log10(params)), 3), "depth_align": depth_align})
            print(f"{ladder:8s} {name:12s} ({params/1e6:6.0f}M): " +
                  "  ".join(f"d{d}={depth_align[d]:+.3f}" for d in DEPTHS), flush=True)
            del mdl, tok, allrep
            gc.collect()
            torch.cuda.empty_cache() if DEV == "cuda" else None

    # scale trend at each uniform depth
    trend = {}
    for d in DEPTHS:
        per = {}
        for ladder in LADDERS:
            rows = [r for r in results if r["ladder"] == ladder]
            per[ladder] = round(spearman([r["log10_params"] for r in rows], [r["depth_align"][d] for r in rows]), 3) if len(rows) >= 3 else None
        pooled = round(spearman([r["log10_params"] for r in results], [r["depth_align"][d] for r in results]), 3)
        trend[d] = {"pooled": pooled, "per_ladder": per}

    upper = [0.8, 0.9, 1.0]
    pooled_upper = [trend[d]["pooled"] for d in upper]
    pythia_upper = [trend[d]["per_ladder"]["Pythia"] for d in upper]
    positive = (np.mean(pooled_upper) >= 0.5) and all((p or -1) > 0 for p in pythia_upper)
    if positive:
        reading = (f"SCALE EFFECT (depth-robust, EXPLORATORY). At a UNIFORM upper-depth read-out (0.8-1.0) where "
                   f"these models carry concept geometry, semantic alignment rises with scale: pooled rho "
                   f"{[trend[d]['pooled'] for d in upper]} and the perfectly-controlled Pythia ladder is positive "
                   f"throughout ({pythia_upper}). The pre-registered fixed-0.66 null was a MIS-SPECIFIED read-out "
                   f"depth, not a real absence of the effect. Needs a pre-registered last-layer replication to claim.")
    else:
        reading = (f"NO DEPTH-ROBUST SCALE EFFECT. Even at uniform upper depths the trend is not cleanly positive "
                   f"(pooled {pooled_upper}, Pythia {pythia_upper}). The scale hypothesis is not supported in 14M-3B.")

    out = {"depths": DEPTHS, "trend_by_depth": trend, "models": results, "positive_depth_robust": bool(positive), "reading": reading}
    (HERE / "scale_law_v3_depth_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== scale trend (Spearman rho vs log params) by UNIFORM read-out depth ===")
    print(f"  {'depth':>6} | {'pooled':>7} | Pythia  GPT-2  Qwen2.5")
    for d in DEPTHS:
        pl = trend[d]["per_ladder"]
        print(f"  {d:>6} | {trend[d]['pooled']:>+7.3f} | {pl['Pythia']:>+.2f}   {pl['GPT-2']:>+.2f}   {pl['Qwen2.5']:>+.2f}")
    print(f"\n>>> {reading}")
    print("wrote scale_law_v3_depth_result.json")


if __name__ == "__main__":
    main()
