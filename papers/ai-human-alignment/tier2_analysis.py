# -*- coding: utf-8 -*-
"""TIER 2 analysis (core) — the within-category WIN on 823 THINGS-fMRI concepts (powered: tens/category).
Loads things720_patterns.npz. Per ROI (ventral, LOC): neural RDM + across-subject noise ceiling; model RDMs
(free LLM gpt2-large, VICE human, mpnet) over the concepts; then full partial-lexical RSA -> ceiling-relative;
category-controlled (27 THINGS cats); within-category-only (now POWERED); + 823-way zero-shot decoding
(top-k, percentile, permutation p) with a word-length control. Vision control = separate pass (tier2_vision.py).
Writes tier2_analysis_result.json."""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; DATA = HERE / "data"; TF = HERE / "things_fmri"
sys.path.insert(0, str(HERE.parent / "real-convergence"))
from run_real_convergence import distmat
from run_real_convergence_v2 import concept_all_layers
from run_real_convergence_v3_controls import partial_corr
from run_ai_human import spearman
import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
DEV = "cuda" if torch.cuda.is_available() else "cpu"

Z = np.load(TF / "things720_patterns.npz", allow_pickle=True)
rtok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
df = pd.read_csv(DATA / "things_concepts.tsv", sep="\t")
CATCOL = "Top-down Category (manual selection)"
uid2cat = {str(u).strip().lower(): (str(c).strip() if pd.notna(c) and str(c).strip() else None) for u, c in zip(df["uniqueID"], df[CATCOL])}
uid2idx = {str(u).strip().lower(): i for i, u in enumerate(df["uniqueID"])}
vice = np.load(DATA / "final_embedding.npy")
subs = ["sub-01", "sub-02", "sub-03"]


def neural_group(roi):
    rdms = []
    for s in subs:
        M = Z[f"{roi}_{s}"]; Mz = (M - M.mean(0)) / (M.std(0) + 1e-9)
        rdms.append(1.0 - np.corrcoef(Mz))
    return rdms, np.mean(rdms, axis=0)


def lexZ(words, iu):
    cl = np.array([len(w) for w in words], float)
    tl = np.array([len(rtok(" " + w.replace("_", " "), add_special_tokens=False).input_ids) for w in words], float)
    zc = (cl - cl.mean()) / (cl.std() + 1e-9); zt = (tl - tl.mean()) / (tl.std() + 1e-9)
    return np.column_stack([np.abs(zc[iu[0]] - zc[iu[1]]), np.abs(zt[iu[0]] - zt[iu[1]])])


def decode(brain, model):
    N = brain.shape[0]; top1 = top5 = 0; perc = []
    rM = np.argsort(np.argsort(model, axis=1), axis=1).astype(float)
    for i in range(N):
        others = np.r_[0:i, i + 1:N]
        rb = np.argsort(np.argsort(brain[i, others])).astype(float); rb -= rb.mean()
        R = rM[:, others] - rM[:, others].mean(1, keepdims=True)
        s = (R @ rb) / (np.sqrt((R ** 2).sum(1)) * np.sqrt((rb ** 2).sum()) + 1e-12)
        r = int(np.where(np.argsort(-s) == i)[0][0]); top1 += r == 0; top5 += r < 5; perc.append(1 - r / (N - 1))
    return {"top1": round(top1 / N, 4), "top5": round(top5 / N, 3), "mean_pctile": round(float(np.mean(perc)), 3), "chance_top1": round(1 / N, 4)}


def perm_p(brain, model, obs, B=100):
    rng = np.random.default_rng(0); N = brain.shape[0]; ge = 1
    for _ in range(B):
        p = rng.permutation(N)
        if decode(brain, model[np.ix_(p, p)])["mean_pctile"] >= obs: ge += 1
    return round(ge / (B + 1), 4)


OUT = {}
for roi in ["ventral", "LOC"]:
    concepts = [str(c) for c in Z[f"{roi}_concepts"]]; N = len(concepts); IU = np.triu_indices(N, 1)
    rdms, group = neural_group(roi)
    def sp(A, B): return spearman(A[IU], B[IU])
    upper = float(np.mean([sp(R, group) for R in rdms]))
    lower = float(np.mean([sp(R, np.mean([rdms[j] for j in range(3) if j != i], axis=0)) for i, R in enumerate(rdms)]))
    cat = np.array([uid2cat.get(c) for c in concepts], dtype=object)
    has = np.array([c is not None for c in cat])
    Zlex = lexZ(concepts, IU)
    catdiff = (cat[IU[0]] != cat[IU[1]]).astype(float)
    same = (cat[IU[0]] == cat[IU[1]]) & has[IU[0]] & has[IU[1]]
    print(f"\n### {roi}: N={N}, {len(set(cat[has]))} categories, {int(has.sum())} categorized, "
          f"{int(same.sum())} within-cat pairs, ceiling [{lower:.3f},{upper:.3f}] ###", flush=True)
    mdl = AutoModelForCausalLM.from_pretrained("gpt2-large", torch_dtype=torch.float16).to(DEV).eval()
    gtok = AutoTokenizer.from_pretrained("gpt2-large")
    llm = distmat(np.stack([concept_all_layers(mdl, gtok, c.replace("_", " "))[-1] for c in concepts])); del mdl; gc.collect()
    mp = distmat(SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=DEV).encode([c.replace("_", " ") for c in concepts], normalize_embeddings=True))
    matched = [c for c in concepts if c in uid2idx]; mi = [concepts.index(c) for c in matched]
    vrdm_full = np.zeros((N, N));
    vsub = distmat(vice[[uid2idx[c] for c in matched]])
    models = {"free LLM (gpt2-large)": llm, "mpnet (embedder)": mp}
    res = {"n_concepts": N, "n_categories": int(len(set(cat[has]))), "n_within_pairs": int(same.sum()),
           "noise_ceiling": [round(lower, 3), round(upper, 3)], "rsa": {}, "decoding": {}}
    for name, M in models.items():
        full = float(partial_corr(M[IU], group[IU], Zlex))
        catc = float(partial_corr(M[IU], group[IU], np.column_stack([Zlex, catdiff])))
        within = float(partial_corr(M[IU][same], group[IU][same], Zlex[same]))
        res["rsa"][name] = {"full": round(full, 3), "pct_ceiling_lo": round(100 * full / lower, 0),
                            "cat_controlled": round(catc, 3), "within_category": round(within, 3)}
    # VICE on matched subset
    gmsub = group[np.ix_(mi, mi)]; iuM = np.triu_indices(len(mi), 1)
    catM = cat[mi]; hasM = has[mi]; sameM = (catM[iuM[0]] == catM[iuM[1]]) & hasM[iuM[0]] & hasM[iuM[1]]
    ZlexM = lexZ(matched, iuM)
    res["rsa"]["VICE/human"] = {"n": len(matched), "full": round(float(partial_corr(vsub[iuM], gmsub[iuM], ZlexM)), 3),
                                "cat_controlled": round(float(partial_corr(vsub[iuM], gmsub[iuM], np.column_stack([ZlexM, (catM[iuM[0]] != catM[iuM[1]]).astype(float)]))), 3),
                                "within_category": round(float(partial_corr(vsub[iuM][sameM], gmsub[iuM][sameM], ZlexM[sameM])), 3)}
    # decoding (823-way)
    d_llm = decode(group, llm); d_llm["perm_p"] = perm_p(group, llm, d_llm["mean_pctile"])
    res["decoding"]["free LLM"] = d_llm
    res["decoding"]["word-length (control)"] = decode(group, np.abs(np.array([len(c) for c in concepts])[:, None] - np.array([len(c) for c in concepts])[None, :]).astype(float))
    res["decoding"]["VICE/human"] = decode(gmsub, vsub)
    OUT[roi] = res
    for nm, d in res["rsa"].items():
        print(f"  {nm:24s} full {d['full']:+.3f} ({d.get('pct_ceiling_lo','?')}% ceil)  cat-ctrl {d['cat_controlled']:+.3f}  within-cat {d['within_category']:+.3f}", flush=True)
    for nm, d in res["decoding"].items():
        print(f"  decode {nm:22s} top1 {d['top1']:.4f} top5 {d['top5']:.3f} pctile {d['mean_pctile']:.3f} (chance {d['chance_top1']})" + (f" perm_p {d.get('perm_p')}" if 'perm_p' in d else ""), flush=True)
    gc.collect()

(HERE / "tier2_analysis_result.json").write_text(json.dumps(OUT, indent=2), encoding="utf-8")
print("\nwrote tier2_analysis_result.json")
