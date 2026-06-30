# -*- coding: utf-8 -*-
"""TIER 1 — does the LLM<->brain meaning match REPLICATE on an independent fMRI dataset?
THINGS-fMRI (Hebart 2023), inferior-temporal cortex, 100 repeatedly-presented test concepts, 3 subjects,
12 reps each. Build a concept RDM per subject, RSA (partial-lexical) to our LLM / VICE / embedder geometries,
report as a fraction of the across-subject noise ceiling, bootstrap CIs.

CAVEAT (loud): IT is high-level VISUAL cortex and subjects SAW PHOTOGRAPHS — the neural RDM carries visual +
semantic structure. This is a pipeline + replication number, NOT yet a clean meaning claim. The vision-confound
control (partial out an image-model RDM) is the next gate; it needs the test images. See report."""
import sys, re, json
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
TF = HERE / "things_fmri" / "betas_testset" / "betas_csv_testset"
DATA = HERE / "data"
sys.path.insert(0, str(HERE.parent / "real-convergence"))
from run_real_convergence import distmat
from run_real_convergence_v2 import concept_all_layers
from run_real_convergence_v3_controls import partial_corr
from run_ai_human import spearman
import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
DEV = "cuda" if torch.cuda.is_available() else "cpu"

concept_of = lambda s: s.rsplit("_", 1)[0].lower()           # 'candelabra_14s.jpg' -> 'candelabra'
subs = ["sub-01", "sub-02", "sub-03"]

# --- neural RDMs: avg 12 reps per concept, z-score voxels, correlation distance ---
neural, concept_sets = [], []
for s in subs:
    a = np.load(TF / f"{s}_TestResponsesIT.npy").astype(np.float64)
    df = pd.read_csv(TF / f"{s}_StimulusMetadataTestset.csv")
    df["concept"] = df["stimulus"].map(concept_of)
    cs = sorted(df["concept"].unique())
    M = np.stack([a[df["concept"].values == c].mean(0) for c in cs])      # 100 x 4145
    M = (M - M.mean(0)) / (M.std(0) + 1e-9)                               # z-score voxels
    neural.append(1.0 - np.corrcoef(M)); concept_sets.append(cs)
assert concept_sets[0] == concept_sets[1] == concept_sets[2], "subjects differ in concept set"
concepts = concept_sets[0]; N = len(concepts); IU = np.triu_indices(N, 1)
group = np.mean(neural, axis=0)
print(f"{N} test concepts, {len(subs)} subjects; IT voxels per subj: "
      f"{[np.load(TF / f'{s}_TestResponsesIT.npy').shape[1] for s in subs]}", flush=True)

# --- across-subject noise ceiling (Nili/Kriegeskorte) ---
def sp(A, B): return spearman(A[IU], B[IU])
upper = np.mean([sp(R, group) for R in neural])
lower = np.mean([sp(R, np.mean([neural[j] for j in range(len(subs)) if j != i], axis=0)) for i, R in enumerate(neural)])
print(f"noise ceiling (Spearman RSA): lower {lower:.3f}, upper {upper:.3f}", flush=True)

# --- lexical control design ---
rtok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
cl = np.array([len(w) for w in concepts], float)
tl = np.array([len(rtok(" " + w, add_special_tokens=False).input_ids) for w in concepts], float)
zc = (cl - cl.mean()) / (cl.std() + 1e-9); zt = (tl - tl.mean()) / (tl.std() + 1e-9)
def lexZ(ii, jj): return np.column_stack([np.abs(zc[ii] - zc[jj]), np.abs(zt[ii] - zt[jj])])
Zlex = lexZ(*IU)

# --- model RDMs over the 100 concept words ---
emb = {}
for tag, repo in [("MiniLM", "sentence-transformers/all-MiniLM-L6-v2"), ("mpnet", "sentence-transformers/all-mpnet-base-v2")]:
    st = SentenceTransformer(repo, device=DEV); emb[tag] = distmat(st.encode(concepts, normalize_embeddings=True)); del st; gc.collect()
tok = AutoTokenizer.from_pretrained("gpt2-large")
mdl = AutoModelForCausalLM.from_pretrained("gpt2-large", torch_dtype=torch.float16).to(DEV).eval()
llm = distmat(np.stack([concept_all_layers(mdl, tok, w)[-1] for w in concepts])); del mdl; gc.collect()

# VICE behavioural (match concept names to THINGS index)
vice = np.load(DATA / "final_embedding.npy")
rows = (DATA / "things_concepts.tsv").read_text(encoding="utf-8").splitlines()[1:]
tindex = {r.split("\t")[0].strip().lower(): i for i, r in enumerate(rows)}
matched = [w for w in concepts if w in tindex]; midx = [concepts.index(w) for w in matched]
print(f"VICE concept match: {len(matched)}/{N}  (unmatched: {[w for w in concepts if w not in tindex]})", flush=True)

bars = {"gpt2-large (text-only LLM)": llm, "mpnet (embedder)": emb["mpnet"], "MiniLM (embedder)": emb["MiniLM"]}

def rsa_full(M):  # partial-lexical RSA to the group neural RDM, full 100 concepts
    return float(partial_corr(M[IU], group[IU], Zlex))

def boot(M, B=2000):
    rng = np.random.default_rng(0); v = []
    for _ in range(B):
        idx = rng.integers(0, N, N); ii, jj = np.triu_indices(N, 1)
        keep = idx[ii] != idx[jj]; a, b = idx[ii][keep], idx[jj][keep]
        v.append(float(partial_corr(M[a, b], group[a, b], lexZ(a, b))))
    return round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)

res = {"n_concepts": N, "n_subjects": len(subs), "noise_ceiling": [round(lower, 3), round(upper, 3)], "per": {}}
for name, M in bars.items():
    r = rsa_full(M); ci = boot(M)
    persub = [round(float(partial_corr(M[IU], R[IU], Zlex)), 3) for R in neural]
    res["per"][name] = {"rsa": round(r, 3), "ci": ci, "pct_ceiling_lo": round(100 * r / lower, 0),
                        "per_subject": persub}
# VICE on matched subset
gm = group[np.ix_(midx, midx)]; iuM = np.triu_indices(len(midx), 1)
ZlexM = np.column_stack([np.abs(zc[midx][iuM[0]] - zc[midx][iuM[1]]), np.abs(zt[midx][iuM[0]] - zt[midx][iuM[1]])])
vice_S = distmat(vice[[tindex[w] for w in matched]])
res["per"][f"VICE/human ({len(matched)} concepts)"] = {"rsa": round(float(partial_corr(vice_S[iuM], gm[iuM], ZlexM)), 3)}

(HERE / "tier1_things_fmri_result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print(f"\n=== TIER 1: RSA to THINGS-fMRI IT cortex (partial-lexical; noise ceiling [{lower:.3f},{upper:.3f}]) ===")
for name, d in res["per"].items():
    extra = f"  95% CI {d['ci']}  = {d['pct_ceiling_lo']:.0f}% of ceiling-lo  per-subj {d['per_subject']}" if "ci" in d else ""
    print(f"  {name:28s} RSA {d['rsa']:+.3f}{extra}")
print("\nCAVEAT: IT = visual cortex; vision-confound NOT yet controlled (needs the test images). Replication+pipeline, not a clean meaning claim yet.")
print("wrote tier1_things_fmri_result.json")
