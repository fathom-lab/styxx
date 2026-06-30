# -*- coding: utf-8 -*-
"""Within-category control for the LLM<->brain meaning match (the last standing expert attack).
Tests whether RSA-to-brain reflects FINE meaning or just coarse 12-category blocks (animal/tool/...):
  (1) category-controlled: partial out a same/different-category indicator on top of the lexical controls.
  (2) within-category-only: RSA restricted to same-category pairs (does it recover structure finer than category?).
Run for gpt2-large (text-only LLM) with human behaviour (VICE, 1.5M judgments) + embedders as reference.
Writes within_category_result.json."""
import sys, json
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; DATA = HERE / "data"
sys.path.insert(0, str(HERE.parent / "real-convergence"))
from run_real_convergence import distmat
from run_real_convergence_v2 import concept_all_layers
from run_real_convergence_v3_controls import partial_corr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
DEV = "cuda" if torch.cuda.is_available() else "cpu"

CATEGORY = {
 'bear':'animal','cat':'animal','cow':'animal','dog':'animal','horse':'animal',
 'arm':'body','eye':'body','foot':'body','hand':'body','leg':'body',
 'apartment':'building','barn':'building','church':'building','house':'building','igloo':'building',
 'arch':'buildpart','chimney':'buildpart','closet':'buildpart','door':'buildpart','window':'buildpart',
 'coat':'clothing','dress':'clothing','pants':'clothing','shirt':'clothing','skirt':'clothing',
 'bed':'furniture','chair':'furniture','desk':'furniture','dresser':'furniture','table':'furniture',
 'ant':'insect','bee':'insect','beetle':'insect','butterfly':'insect','fly':'insect',
 'bottle':'kitchen','cup':'kitchen','glass':'kitchen','knife':'kitchen','spoon':'kitchen',
 'bell':'manmade','key':'manmade','refrigerator':'manmade','telephone':'manmade','watch':'manmade',
 'chisel':'tool','hammer':'tool','pliers':'tool','saw':'tool','screwdriver':'tool',
 'carrot':'vegetable','celery':'vegetable','corn':'vegetable','lettuce':'vegetable','tomato':'vegetable',
 'airplane':'vehicle','bicycle':'vehicle','car':'vehicle','train':'vehicle','truck':'vehicle'}

bz = np.load(HERE / "brain_rdm.npz", allow_pickle=True)
nouns = [str(w) for w in bz["nouns"]]; brain = bz["group"]
assert all(w in CATEGORY for w in nouns) and len(set(CATEGORY[w] for w in nouns)) == 12
rtok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# representations on the full 60
emb60 = {}
for tag, repo in [("MiniLM", "sentence-transformers/all-MiniLM-L6-v2"), ("mpnet", "sentence-transformers/all-mpnet-base-v2")]:
    st = SentenceTransformer(repo, device=DEV); emb60[tag] = distmat(st.encode(nouns, normalize_embeddings=True)); del st
tok = AutoTokenizer.from_pretrained("gpt2-large")
mdl = AutoModelForCausalLM.from_pretrained("gpt2-large", torch_dtype=torch.float16).to(DEV).eval()
llm60 = distmat(np.stack([concept_all_layers(mdl, tok, w)[-1] for w in nouns])); del mdl

# VICE behavioural + 53 shared THINGS nouns
vice = np.load(DATA / "final_embedding.npy")
rows = (DATA / "things_concepts.tsv").read_text(encoding="utf-8").splitlines()[1:]
tindex = {r.split("\t")[0].strip().lower(): i for i, r in enumerate(rows)}
Snouns = [w for w in nouns if w in tindex]; Sidx = [nouns.index(w) for w in Snouns]
vice_S = distmat(vice[[tindex[w] for w in Snouns]])


def lex(nset):
    cl = np.array([len(w) for w in nset], float); tl = np.array([len(rtok(" " + w, add_special_tokens=False).input_ids) for w in nset], float)
    return (cl - cl.mean()) / (cl.std() + 1e-9), (tl - tl.mean()) / (tl.std() + 1e-9)


def analyze(M, brainM, nset):
    N = len(nset); ii, jj = np.triu_indices(N, 1)
    cat = np.array([CATEGORY[w] for w in nset])
    zc, zt = lex(nset)
    Zlex = np.column_stack([np.abs(zc[ii] - zc[jj]), np.abs(zt[ii] - zt[jj])])
    catdiff = (cat[ii] != cat[jj]).astype(float)
    full = float(partial_corr(M[ii, jj], brainM[ii, jj], Zlex))
    catctrl = float(partial_corr(M[ii, jj], brainM[ii, jj], np.column_stack([Zlex, catdiff])))
    w = ~catdiff.astype(bool)
    within = float(partial_corr(M[ii, jj][w], brainM[ii, jj][w], Zlex[w]))
    return {"full": round(full, 3), "cat_controlled": round(catctrl, 3),
            "within_category": round(within, 3), "n_within_pairs": int(w.sum())}


def boot(M, brainM, nset, mode, B=2000):
    N = len(nset); cat = np.array([CATEGORY[w] for w in nset]); zc, zt = lex(nset)
    rng = np.random.default_rng(0); vals = []
    for _ in range(B):
        idx = rng.integers(0, N, N); ii, jj = np.triu_indices(N, 1)
        distinct = idx[ii] != idx[jj]
        if mode == "within":
            keep = distinct & (cat[idx[ii]] == cat[idx[jj]])
        else:  # cat-controlled: all distinct pairs, category partialled out
            keep = distinct
        a, b = idx[ii][keep], idx[jj][keep]
        Z = np.column_stack([np.abs(zc[a] - zc[b]), np.abs(zt[a] - zt[b])])
        if mode != "within":
            Z = np.column_stack([Z, (cat[a] != cat[b]).astype(float)])
        vals.append(float(partial_corr(M[a, b], brainM[a, b], Z)))
    return round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)


sub = lambda Mx: Mx[np.ix_(Sidx, Sidx)]
res = {
    "gpt2-large @60 (full set)":  analyze(llm60, brain, nouns),
    "gpt2-large @53 (shared)":    analyze(sub(llm60), sub(brain), Snouns),
    "VICE/human @53":             analyze(vice_S, sub(brain), Snouns),
    "mpnet @53":                  analyze(sub(emb60["mpnet"]), sub(brain), Snouns),
    "MiniLM @53":                 analyze(sub(emb60["MiniLM"]), sub(brain), Snouns),
}
res["gpt2-large @60 (full set)"]["within_CI"] = boot(llm60, brain, nouns, "within")
res["VICE/human @53"]["within_CI"] = boot(vice_S, sub(brain), Snouns, "within")
res["gpt2-large @60 (full set)"]["catctrl_CI"] = boot(llm60, brain, nouns, "catctrl")
res["VICE/human @53"]["catctrl_CI"] = boot(vice_S, sub(brain), Snouns, "catctrl")

# positive control: do categories actually separate in brain + LLM? (across/within mean distance ratio)
def catstruct(M, nset):
    cat = np.array([CATEGORY[w] for w in nset]); same = cat[:, None] == cat[None, :]; off = ~np.eye(len(nset), dtype=bool)
    return round(float(M[~same].mean() / M[same & off].mean()), 3)
res["_catstruct_ratio"] = {"brain": catstruct(brain, nouns), "gpt2-large": catstruct(llm60, nouns)}

(HERE / "within_category_result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print(f"category structure (across/within dist, >1 = categories present): {res['_catstruct_ratio']}\n")
print(f"{'representation':26s} {'full':>7} {'+cat-ctrl':>10} {'within-cat':>11} {'n_pairs':>8}  within-CI")
for k, v in res.items():
    if k.startswith("_"): continue
    ci = v.get("within_CI", ""); ci = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else ""
    print(f"{k:26s} {v['full']:+7.3f} {v['cat_controlled']:+10.3f} {v['within_category']:+11.3f} {v['n_within_pairs']:8d}  {ci}")
print("\nwrote within_category_result.json")
