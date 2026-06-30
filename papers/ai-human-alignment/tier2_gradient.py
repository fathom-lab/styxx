# -*- coding: utf-8 -*-
"""TIER 2 gradient — does the vision-controlled LLM<->brain MEANING signal climb the cortical hierarchy?
Per ROI (V1->V2->V3->hV4->LOC->ventral): partial-lexical LLM<->brain RSA, then the same partialling out a
CLIP-image RDM (the meaning residual), + how visual the ROI is (CLIP->brain|lex). Bootstrap CI on the residual.
If the residual climbs V1->high-level, the LLM's meaning alignment lives at the semantic end, not the pixel end.
Writes tier2_gradient_result.json."""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; DATA = HERE / "data"; TF = HERE / "things_fmri"
sys.path.insert(0, str(HERE.parent / "real-convergence"))
from run_real_convergence import distmat
from run_real_convergence_v2 import concept_all_layers
from run_real_convergence_v3_controls import partial_corr
from run_ai_brain_vision import get_clip_image_rdm
import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
DEV = "cuda" if torch.cuda.is_available() else "cpu"

Z = np.load(TF / "things_gradient_patterns.npz", allow_pickle=True)
rtok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
subs = ["sub-01", "sub-02", "sub-03"]; ROIS = ["V1", "V2", "V3", "hV4", "LOC", "ventral"]
concepts = [str(c) for c in Z["V1_concepts"]]

kept, vis_full = get_clip_image_rdm(concepts)
ix = [concepts.index(w) for w in kept]; n = len(kept); IU = np.triu_indices(n, 1)
vis = vis_full
print(f"vision: {n}/{len(concepts)} concepts", flush=True)
mdl = AutoModelForCausalLM.from_pretrained("gpt2-large", torch_dtype=torch.float16).to(DEV).eval()
gtok = AutoTokenizer.from_pretrained("gpt2-large")
llm = distmat(np.stack([concept_all_layers(mdl, gtok, w.replace("_", " "))[-1] for w in kept])); del mdl; gc.collect()
cl = np.array([len(w) for w in kept], float); tl = np.array([len(rtok(" " + w.replace("_", " "), add_special_tokens=False).input_ids) for w in kept], float)
zc = (cl - cl.mean()) / (cl.std() + 1e-9); zt = (tl - tl.mean()) / (tl.std() + 1e-9)
def lexp(iu): return np.column_stack([np.abs(zc[iu[0]] - zc[iu[1]]), np.abs(zt[iu[0]] - zt[iu[1]])])
L = lexp(IU); visv = vis[IU]; llmv = llm[IU]

OUT = {"n": n, "rois": {}}
for roi in ROIS:
    rdms = []
    for s in subs:
        M = Z[f"{roi}_{s}"][ix]; Mz = (M - M.mean(0)) / (M.std(0) + 1e-9); rdms.append(1.0 - np.corrcoef(Mz))
    g = np.mean(rdms, axis=0); gv = g[IU]
    lex = float(partial_corr(llmv, gv, L))
    lexvis = float(partial_corr(llmv, gv, np.column_stack([L, visv])))
    visbrain = float(partial_corr(visv, gv, L))
    rng = np.random.default_rng(0); bv = []
    for _ in range(800):
        idx = rng.integers(0, n, n); ii, jj = np.triu_indices(n, 1); keep = idx[ii] != idx[jj]
        a, b = idx[ii][keep], idx[jj][keep]
        bv.append(float(partial_corr(llm[a, b], g[a, b], np.column_stack([np.abs(zc[a] - zc[b]), np.abs(zt[a] - zt[b]), vis[a, b]]))))
    ci = [round(float(np.percentile(bv, 2.5)), 3), round(float(np.percentile(bv, 97.5)), 3)]
    OUT["rois"][roi] = {"LLM|lex": round(lex, 3), "LLM|lex+vision": round(lexvis, 3), "LLM|lex+vision_CI": ci, "vision->brain|lex": round(visbrain, 3)}
    print(f"  {roi:8s} LLM|lex {lex:+.3f}  ->|+vision {lexvis:+.3f} CI {ci}   (vision->brain|lex {visbrain:+.3f})", flush=True)

(HERE / "tier2_gradient_result.json").write_text(json.dumps(OUT, indent=2), encoding="utf-8")
print("\nwrote tier2_gradient_result.json")
