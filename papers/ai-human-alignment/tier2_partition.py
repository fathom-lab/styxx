# -*- coding: utf-8 -*-
"""TIER 2 deeper — UNIQUE variance. Does the free LLM predict brain meaning-geometry BEYOND word-form, a vision
model, AND 1.5M human judgments? partial_corr(LLM, brain | lex + CLIP-image + VICE) on the 823-concept THINGS-fMRI
set (ventral, LOC; image+VICE-matched), full and within-category, with bootstrap CIs + an R^2 variance partition.
A positive unique partial = the LLM captures brain structure neither appearance nor behaviour alone does.
Writes tier2_partition_result.json."""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; DATA = HERE / "data"; TF = HERE / "things_fmri"
sys.path.insert(0, str(HERE.parent / "real-convergence"))
from run_real_convergence import distmat
from run_real_convergence_v2 import concept_all_layers
from run_real_convergence_v3_controls import partial_corr
from run_ai_brain_vision import get_clip_image_rdm, r2
import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
DEV = "cuda" if torch.cuda.is_available() else "cpu"

Z = np.load(TF / "things720_patterns.npz", allow_pickle=True)
rtok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
df = pd.read_csv(DATA / "things_concepts.tsv", sep="\t")
uid2cat = {str(u).strip().lower(): (str(c).strip() if pd.notna(c) and str(c).strip() else None) for u, c in zip(df["uniqueID"], df["Top-down Category (manual selection)"])}
uid2idx = {str(u).strip().lower(): i for i, u in enumerate(df["uniqueID"])}
vice = np.load(DATA / "final_embedding.npy")
subs = ["sub-01", "sub-02", "sub-03"]
concepts = [str(c) for c in Z["ventral_concepts"]]

# common set: has image AND in VICE
kept, vis_full = get_clip_image_rdm(concepts)
common = [w for w in kept if w in uid2idx]
ci_k = [kept.index(w) for w in common]; vis = vis_full[np.ix_(ci_k, ci_k)]
n = len(common); IU = np.triu_indices(n, 1)
print(f"common (image AND VICE): {n}", flush=True)

mdl = AutoModelForCausalLM.from_pretrained("gpt2-large", torch_dtype=torch.float16).to(DEV).eval()
gtok = AutoTokenizer.from_pretrained("gpt2-large")
llm = distmat(np.stack([concept_all_layers(mdl, gtok, w.replace("_", " "))[-1] for w in common])); del mdl; gc.collect()
vcb = distmat(vice[[uid2idx[w] for w in common]])
cl = np.array([len(w) for w in common], float); tl = np.array([len(rtok(" " + w.replace("_", " "), add_special_tokens=False).input_ids) for w in common], float)
zc = (cl - cl.mean()) / (cl.std() + 1e-9); zt = (tl - tl.mean()) / (tl.std() + 1e-9)
def lexpairs(iu): return np.column_stack([np.abs(zc[iu[0]] - zc[iu[1]]), np.abs(zt[iu[0]] - zt[iu[1]])])
cat = np.array([uid2cat.get(w) for w in common], dtype=object); has = np.array([c is not None for c in cat])
L = lexpairs(IU); visv = vis[IU]; vcbv = vcb[IU]; llmv = llm[IU]
same = (cat[IU[0]] == cat[IU[1]]) & has[IU[0]] & has[IU[1]]

OUT = {"n": n, "n_within_pairs": int(same.sum())}
for roi in ["ventral", "LOC"]:
    rdms = []  # z-score voxels then correlation distance
    for s in subs:
        M = Z[f"{roi}_{s}"][[concepts.index(w) for w in common]]; Mz = (M - M.mean(0)) / (M.std(0) + 1e-9)
        rdms.append(1.0 - np.corrcoef(Mz))
    g = np.mean(rdms, axis=0); gv = g[IU]

    def uniq(mask):
        m = mask if mask is not None else slice(None)
        y = gv[m]; Lm = L[m]; vm = visv[m]; bm = vcbv[m]; lm = llmv[m]
        # LLM partial beyond lex+vision+behaviour
        p = float(partial_corr(lm, y, np.column_stack([Lm, vm, bm])))
        # R^2 unique contributions
        R_all = r2(y, Lm, vm, bm, lm); R_lvb = r2(y, Lm, vm, bm)
        R_novis = r2(y, Lm, bm, lm); R_nobeh = r2(y, Lm, vm, lm)
        return {"LLM|lex+vis+beh": round(p, 3),
                "R2_LLM_unique": round(R_all - R_lvb, 4), "R2_vision_unique": round(R_all - R_novis, 4),
                "R2_behaviour_unique": round(R_all - R_nobeh, 4), "R2_all": round(R_all, 3)}

    def boot(mask, B=1000):
        rng = np.random.default_rng(0); v = []
        for _ in range(B):
            idx = rng.integers(0, n, n); ii, jj = np.triu_indices(n, 1)
            keep = idx[ii] != idx[jj]
            if mask is not None: keep &= (cat[idx[ii]] == cat[idx[jj]]) & has[idx[ii]] & has[idx[jj]]
            a, b = idx[ii][keep], idx[jj][keep]
            Zc = np.column_stack([np.abs(zc[a] - zc[b]), np.abs(zt[a] - zt[b]), vis[a, b], vcb[a, b]])
            v.append(float(partial_corr(llm[a, b], g[a, b], Zc)))
        return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]

    res = {"full": uniq(None), "within_category": uniq(same)}
    res["full"]["LLM|lex+vis+beh_CI"] = boot(None)
    res["within_category"]["LLM|lex+vis+beh_CI"] = boot(same)
    OUT[roi] = res
    print(f"\n### {roi} (n={n}) ###", flush=True)
    for k in ("full", "within_category"):
        d = res[k]
        print(f"  {k:16s} LLM|lex+vis+beh {d['LLM|lex+vis+beh']:+.3f} CI {d['LLM|lex+vis+beh_CI']}  "
              f"R2unique LLM {d['R2_LLM_unique']:+.4f} / vision {d['R2_vision_unique']:+.4f} / behaviour {d['R2_behaviour_unique']:+.4f}", flush=True)
    gc.collect()

(HERE / "tier2_partition_result.json").write_text(json.dumps(OUT, indent=2), encoding="utf-8")
print("\nwrote tier2_partition_result.json")
