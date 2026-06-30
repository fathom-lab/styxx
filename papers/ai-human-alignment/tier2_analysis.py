# -*- coding: utf-8 -*-
"""TIER 2 analysis — the within-category WIN on 720 THINGS-fMRI concepts (powered: tens/category).
Loads things720_patterns.npz (built by stream_build_things720.py). Per ROI (ventral, LOC):
  neural RDM (z-voxels, corr-dist) + across-subject noise ceiling; model RDMs (free LLM gpt2-large, VICE human,
  mpnet, CLIP-image vision) over the concepts; then:
  (1) full partial-lexical RSA -> noise-ceiling-relative;
  (2) category-controlled RSA (partial out the 27 THINGS categories) — survives taxonomy?
  (3) within-category-only RSA (now POWERED) — fine meaning structure?
  (4) vision-confound control (partial CLIP-image);
  (5) 720-way zero-shot decoding (top-k, percentile, permutation p).
Writes tier2_analysis_result.json."""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; DATA = HERE / "data"; TF = HERE / "things_fmri"
sys.path.insert(0, str(HERE.parent / "real-convergence"))
from run_real_convergence import distmat
from run_real_convergence_v2 import concept_all_layers
from run_real_convergence_v3_controls import partial_corr
from run_ai_brain_vision import get_clip_image_rdm
from run_ai_human import spearman
import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
DEV = "cuda" if torch.cuda.is_available() else "cpu"

Z = np.load(TF / "things720_patterns.npz", allow_pickle=True)
rtok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
df = pd.read_csv(DATA / "things_concepts.tsv", sep="\t")
CATCOL = "Top-down Category (manual selection)"
uid2cat = {str(u).strip().lower(): (str(c).strip() if pd.notna(c) else None) for u, c in zip(df["uniqueID"], df[CATCOL])}
uid2idx = {str(u).strip().lower(): i for i, u in enumerate(df["uniqueID"])}
vice = np.load(DATA / "final_embedding.npy")
subs = ["sub-01", "sub-02", "sub-03"]


def neural_group(roi):
    mats = [Z[f"{roi}_{s}"] for s in subs]
    rdms = []
    for M in mats:
        Mz = (M - M.mean(0)) / (M.std(0) + 1e-9)
        rdms.append(1.0 - np.corrcoef(Mz))
    return rdms, np.mean(rdms, axis=0)


def lexZ(words, iu):
    cl = np.array([len(w) for w in words], float); tl = np.array([len(rtok(" " + w.replace("_", " "), add_special_tokens=False).input_ids) for w in words], float)
    zc = (cl - cl.mean()) / (cl.std() + 1e-9); zt = (tl - tl.mean()) / (tl.std() + 1e-9)
    return np.column_stack([np.abs(zc[iu[0]] - zc[iu[1]]), np.abs(zt[iu[0]] - zt[iu[1]])])


def decode(brain, model):
    N = brain.shape[0]; top1 = top5 = 0; perc = []
    rM_full = np.argsort(np.argsort(model, axis=1), axis=1).astype(float)
    for i in range(N):
        others = np.r_[0:i, i + 1:N]
        rb = np.argsort(np.argsort(brain[i, others])).astype(float); rb -= rb.mean()
        rM = rM_full[:, others] - rM_full[:, others].mean(1, keepdims=True)
        s = (rM @ rb) / (np.sqrt((rM ** 2).sum(1)) * np.sqrt((rb ** 2).sum()) + 1e-12)
        r = int(np.where(np.argsort(-s) == i)[0][0]); top1 += r == 0; top5 += r < 5; perc.append(1 - r / (N - 1))
    return {"top1": round(top1 / N, 3), "top5": round(top5 / N, 3), "mean_pctile": round(float(np.mean(perc)), 3), "chance_top1": round(1 / N, 4)}


def perm_p(brain, model, obs, B=200):
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
    # category vector
    cat = np.array([uid2cat.get(c) for c in concepts], dtype=object)
    has = np.array([c is not None for c in cat]); ncat = len(set(cat[has]))
    Zlex = lexZ(concepts, IU)
    catdiff = (cat[IU[0]] != cat[IU[1]]).astype(float)
    same = (cat[IU[0]] == cat[IU[1]]) & has[IU[0]] & has[IU[1]]
    # model RDMs
    mdl = AutoModelForCausalLM.from_pretrained("gpt2-large", torch_dtype=torch.float16).to(DEV).eval()
    gtok = AutoTokenizer.from_pretrained("gpt2-large")
    llm = distmat(np.stack([concept_all_layers(mdl, gtok, c.replace("_", " "))[-1] for c in concepts])); del mdl; gc.collect()
    mp = distmat(SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=DEV).encode([c.replace("_", " ") for c in concepts], normalize_embeddings=True))
    keptv, vis = get_clip_image_rdm(concepts)
    models = {"free LLM (gpt2-large)": llm, "mpnet (embedder)": mp}
    res = {"n_concepts": N, "n_categories": ncat, "noise_ceiling": [round(lower, 3), round(upper, 3)],
           "vision_concepts": len(keptv), "rsa": {}, "decoding": {}}
    for name, M in models.items():
        full = float(partial_corr(M[IU], group[IU], Zlex))
        catc = float(partial_corr(M[IU], group[IU], np.column_stack([Zlex, catdiff])))
        w = same; within = float(partial_corr(M[w], group[w], Zlex[w])) if w.sum() > 30 else None
        # vision partial (restrict to concepts with an image)
        ix = [concepts.index(x) for x in keptv]; iuv = np.triu_indices(len(ix), 1)
        Lv = lexZ(keptv, iuv); gv = group[np.ix_(ix, ix)][iuv]; Mv = M[np.ix_(ix, ix)][iuv]
        lexvis = float(partial_corr(Mv, gv, np.column_stack([Lv, vis[iuv]])))
        res["rsa"][name] = {"full": round(full, 3), "pct_ceiling_lo": round(100 * full / lower, 0),
                            "cat_controlled": round(catc, 3), "within_category": (round(within, 3) if within is not None else None),
                            "lex+vision": round(lexvis, 3)}
    # VICE + vision decoding/RSA
    matched = [c for c in concepts if c in uid2idx]; mi = [concepts.index(c) for c in matched]
    if len(matched) > 30:
        vrdm = distmat(vice[[uid2idx[c] for c in matched]]); iuM = np.triu_indices(len(mi), 1)
        res["rsa"]["VICE/human"] = {"full": round(float(partial_corr(vrdm[iuM], group[np.ix_(mi, mi)][iuM], lexZ(matched, iuM))), 3), "n": len(matched)}
    # decoding (720-way) for LLM, vision, VICE
    dec = {"free LLM": decode(group, llm)}
    dec["free LLM"]["perm_p"] = perm_p(group, llm, dec["free LLM"]["mean_pctile"])
    gv_full = group[np.ix_([concepts.index(x) for x in keptv], [concepts.index(x) for x in keptv])]
    dec["CLIP-image (vision)"] = decode(gv_full, vis)
    if len(matched) > 30: dec["VICE/human"] = decode(group[np.ix_(mi, mi)], distmat(vice[[uid2idx[c] for c in matched]]))
    dec["word-length (control)"] = decode(group, np.abs(np.array([len(c) for c in concepts])[:, None] - np.array([len(c) for c in concepts])[None, :]))
    res["decoding"] = dec
    OUT[roi] = res
    print(f"\n### ROI {roi}: N={N}, {ncat} categories, ceiling [{lower:.3f},{upper:.3f}] ###")
    for nm, d in res["rsa"].items():
        print(f"  {nm:24s} full {d['full']:+.3f}" + (f" ({d['pct_ceiling_lo']:.0f}% ceil)  cat-ctrl {d['cat_controlled']:+.3f}  within-cat {d['within_category']}  |lex+vis {d['lex+vision']:+.3f}" if 'cat_controlled' in d else f"  (n={d.get('n')})"))
    for nm, d in dec.items():
        print(f"  decode {nm:22s} top5 {d['top5']:.3f} pctile {d['mean_pctile']:.3f} (chance top1 {d['chance_top1']})" + (f" perm_p {d.get('perm_p')}" if 'perm_p' in d else ""))
    gc.collect()

(HERE / "tier2_analysis_result.json").write_text(json.dumps(OUT, indent=2), encoding="utf-8")
print("\nwrote tier2_analysis_result.json")
