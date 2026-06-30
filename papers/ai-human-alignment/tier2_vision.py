# -*- coding: utf-8 -*-
"""TIER 2 vision gate — does within-category structure survive a vision control? ventral/LOC are VISUAL
cortex in a VISUAL paradigm, so within-category RSA could be appearance, not meaning. Partial out a CLIP-image
RDM (clip-ViT-B-32 over each concept's THINGS image) on top of word-form, for the FULL and WITHIN-category RSA.
Free LLM + VICE. Writes tier2_vision_result.json."""
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

Z = np.load(TF / "things720_patterns.npz", allow_pickle=True)
rtok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
df = pd.read_csv(DATA / "things_concepts.tsv", sep="\t")
uid2cat = {str(u).strip().lower(): (str(c).strip() if pd.notna(c) and str(c).strip() else None) for u, c in zip(df["uniqueID"], df["Top-down Category (manual selection)"])}
uid2idx = {str(u).strip().lower(): i for i, u in enumerate(df["uniqueID"])}
vice = np.load(DATA / "final_embedding.npy")
subs = ["sub-01", "sub-02", "sub-03"]
concepts = [str(c) for c in Z["ventral_concepts"]]

# CLIP-image RDM over concepts with a THINGS image (slow: fetches images, cached)
kept, vis = get_clip_image_rdm(concepts)
ix = [concepts.index(w) for w in kept]; n = len(kept); IU = np.triu_indices(n, 1)
print(f"vision: {n}/{len(concepts)} concepts have a CLIP-image vector", flush=True)

# LLM + VICE RDMs over kept
mdl = AutoModelForCausalLM.from_pretrained("gpt2-large", torch_dtype=torch.float16).to(DEV).eval()
gtok = AutoTokenizer.from_pretrained("gpt2-large")
llm = distmat(np.stack([concept_all_layers(mdl, gtok, w.replace("_", " "))[-1] for w in kept])); del mdl; gc.collect()
matched = [w for w in kept if w in uid2idx]; mk = [kept.index(w) for w in matched]
vrdm = distmat(vice[[uid2idx[w] for w in matched]])

cl = np.array([len(w) for w in kept], float); tl = np.array([len(rtok(" " + w.replace("_", " "), add_special_tokens=False).input_ids) for w in kept], float)
zc = (cl - cl.mean()) / (cl.std() + 1e-9); zt = (tl - tl.mean()) / (tl.std() + 1e-9)
def lexpairs(iu): return np.column_stack([np.abs(zc[iu[0]] - zc[iu[1]]), np.abs(zt[iu[0]] - zt[iu[1]])])
cat = np.array([uid2cat.get(w) for w in kept], dtype=object); has = np.array([c is not None for c in cat])
visv = vis[IU]; Lex = lexpairs(IU)
same = (cat[IU[0]] == cat[IU[1]]) & has[IU[0]] & has[IU[1]]

OUT = {"n_vision": n, "n_within_pairs": int(same.sum())}
for roi in ["ventral", "LOC"]:
    rdms = []
    for s in subs:
        M = Z[f"{roi}_{s}"][ix]; Mz = (M - M.mean(0)) / (M.std(0) + 1e-9); rdms.append(1.0 - np.corrcoef(Mz))
    group = np.mean(rdms, axis=0); gv = group[IU]
    def cascade(Mrdm, name, mset_iu=None):
        if mset_iu is None:  # full
            full_lex = float(partial_corr(Mrdm[IU], gv, Lex))
            full_lv = float(partial_corr(Mrdm[IU], gv, np.column_stack([Lex, visv])))
            w_lex = float(partial_corr(Mrdm[IU][same], gv[same], Lex[same]))
            w_lv = float(partial_corr(Mrdm[IU][same], gv[same], np.column_stack([Lex, visv])[same]))
            return {"full|lex": round(full_lex, 3), "full|lex+vision": round(full_lv, 3),
                    "within|lex": round(w_lex, 3), "within|lex+vision": round(w_lv, 3)}
    def boot_within_vis(Mrdm, B=1000):
        rng = np.random.default_rng(0); v = []
        for _ in range(B):
            idx = rng.integers(0, n, n); ii, jj = np.triu_indices(n, 1)
            keep = (idx[ii] != idx[jj]) & (cat[idx[ii]] == cat[idx[jj]]) & has[idx[ii]] & has[idx[jj]]
            a, b = idx[ii][keep], idx[jj][keep]
            Zb = np.column_stack([np.abs(zc[a] - zc[b]), np.abs(zt[a] - zt[b]), vis[a, b]])
            v.append(float(partial_corr(Mrdm[a, b], group[a, b], Zb)))
        return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    res = {"free LLM": cascade(llm, "LLM")}
    res["free LLM"]["within|lex+vision_CI"] = boot_within_vis(llm)
    # VICE on matched subset
    iuM = np.triu_indices(len(mk), 1); gmv = group[np.ix_(mk, mk)][iuM]; visM = vis[np.ix_(mk, mk)][iuM]
    catM = cat[mk]; hasM = has[mk]; sameM = (catM[iuM[0]] == catM[iuM[1]]) & hasM[iuM[0]] & hasM[iuM[1]]
    LexM = lexpairs(iuM); vrdmM = vrdm[iuM]
    res["VICE/human"] = {"full|lex": round(float(partial_corr(vrdmM, gmv, LexM)), 3),
                         "full|lex+vision": round(float(partial_corr(vrdmM, gmv, np.column_stack([LexM, visM]))), 3),
                         "within|lex": round(float(partial_corr(vrdm[iuM][sameM], gmv[sameM], LexM[sameM])), 3),
                         "within|lex+vision": round(float(partial_corr(vrdm[iuM][sameM], gmv[sameM], np.column_stack([LexM, visM])[sameM])), 3)}
    res["vision->neural|lex"] = round(float(partial_corr(visv, gv, Lex)), 3)
    OUT[roi] = res
    print(f"\n### {roi} (vision control, n={n}) ###", flush=True)
    for nm, d in res.items():
        if isinstance(d, dict):
            print(f"  {nm:12s} full {d['full|lex']:+.3f}->{d['full|lex+vision']:+.3f} (|+vis)   within {d['within|lex']:+.3f}->{d['within|lex+vision']:+.3f} (|+vis)", flush=True)

(HERE / "tier2_vision_result.json").write_text(json.dumps(OUT, indent=2), encoding="utf-8")
print("\nwrote tier2_vision_result.json")
