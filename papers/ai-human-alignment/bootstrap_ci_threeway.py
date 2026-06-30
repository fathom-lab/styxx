# -*- coding: utf-8 -*-
"""Bootstrap 95% CIs for the three-way bars (RSA-to-brain, partial-lexical, 53 shared nouns), by
resampling NOUNS with replacement (stimulus bootstrap), masking duplicate-item pairs so the artificial
zero-distance diagonal doesn't inflate r. Reuses the exact run_ai_brain pipeline. Writes ci_threeway.json."""
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

bz = np.load(HERE / "brain_rdm.npz", allow_pickle=True)
nouns = [str(w) for w in bz["nouns"]]; brain = bz["group"]
rtok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

vice = np.load(DATA / "final_embedding.npy")
rows = (DATA / "things_concepts.tsv").read_text(encoding="utf-8").splitlines()[1:]
tindex = {r.split("\t")[0].strip().lower(): i for i, r in enumerate(rows)}
Snouns = [w for w in nouns if w in tindex]; Sidx = [nouns.index(w) for w in Snouns]; nS = len(Snouns)
brain_S = brain[np.ix_(Sidx, Sidx)]
vice_S = distmat(vice[[tindex[w] for w in Snouns]])
print(f"{nS} shared nouns", flush=True)

# build model/embedder RDMs on the FULL 60 nouns then sub-index (distmat centers over the item set,
# so this must match run_ai_brain exactly); VICE is built on the 53 shared, as in run_ai_brain.
emb60 = {}
for tag, repo in [("MiniLM", "sentence-transformers/all-MiniLM-L6-v2"), ("mpnet", "sentence-transformers/all-mpnet-base-v2")]:
    st = SentenceTransformer(repo, device=DEV); emb60[tag] = distmat(st.encode(nouns, normalize_embeddings=True)); del st
tok = AutoTokenizer.from_pretrained("gpt2-large")
mdl = AutoModelForCausalLM.from_pretrained("gpt2-large", torch_dtype=torch.float16).to(DEV).eval()
llm60 = distmat(np.stack([concept_all_layers(mdl, tok, w)[-1] for w in nouns])); del mdl
sub = lambda M: M[np.ix_(Sidx, Sidx)]

cl = np.array([len(w) for w in Snouns], float)
tl = np.array([len(rtok(" " + w, add_special_tokens=False).input_ids) for w in Snouns], float)
zc = (cl - cl.mean()) / (cl.std() + 1e-9); zt = (tl - tl.mean()) / (tl.std() + 1e-9)
bars = {"VICE": vice_S, "gpt2-large": sub(llm60), "mpnet": sub(emb60["mpnet"]), "MiniLM": sub(emb60["MiniLM"])}


def rsa(idx):
    ii, jj = np.triu_indices(len(idx), 1)
    keep = idx[ii] != idx[jj]                       # drop duplicate-item pairs
    a, b = idx[ii][keep], idx[jj][keep]
    Z = np.column_stack([np.abs(zc[a] - zc[b]), np.abs(zt[a] - zt[b])])
    bvec = brain_S[a, b]
    return {k: float(partial_corr(M[a, b], bvec, Z)) for k, M in bars.items()}


pt = rsa(np.arange(nS))
rng = np.random.default_rng(0); B = 2000
boot = {k: [] for k in bars}
for _ in range(B):
    r = rsa(rng.integers(0, nS, nS))
    for k in r: boot[k].append(r[k])
out = {k: {"point": round(pt[k], 3),
           "lo": round(float(np.percentile(boot[k], 2.5)), 3),
           "hi": round(float(np.percentile(boot[k], 97.5)), 3)} for k in bars}
# pairwise: does the bootstrap difference exclude 0? (does LLM beat embedders / match VICE?)
def pdiff(a, b):
    d = np.array(boot[a]) - np.array(boot[b])
    return {"mean": round(float(d.mean()), 3), "p_a_le_b": round(float((d <= 0).mean()), 3)}
out["_diff"] = {"VICE-gpt2large": pdiff("VICE", "gpt2-large"),
                "gpt2large-mpnet": pdiff("gpt2-large", "mpnet"),
                "gpt2large-MiniLM": pdiff("gpt2-large", "MiniLM")}
(HERE / "ci_threeway.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
for k in bars: print(f"  {k:11s} {out[k]['point']:+.3f}  95% CI [{out[k]['lo']:+.3f}, {out[k]['hi']:+.3f}]")
print("diffs:", json.dumps(out["_diff"]))
print("wrote ci_threeway.json")
