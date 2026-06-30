# -*- coding: utf-8 -*-
"""ZERO-SHOT CROSS-SUBSTRATE CONCEPT DECODER — "the efficient telepathy".
Can we identify WHICH concept a brain is representing, with NO scanner-specific training and NO regression —
using only a free, text-only LLM's concept geometry as the meaning-bridge?

Method (relational / RSA decoding, leave-one-out): for held-out concept i, take its brain dissimilarity
profile (how its neural pattern relates to every other concept). Rank all candidate concepts by how well
each one's MODEL dissimilarity profile matches i's brain profile (Spearman). Predict argmax. Chance = 1/N.
This is training-free, substrate-bridging, and works for novel held-out concepts.

Datasets: Mitchell-2008 (word-reading; clean meaning substrate) + THINGS-fMRI IT (visual paradigm).
Decoders: free LLM (gpt2-large) vs 1.5M-judgment human behaviour (VICE) vs embedders vs CLIP-image (vision)
vs word-length (lexical control, must be ~chance). Reproduce -> zero_shot_decoder_result.json."""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TF = HERE / "things_fmri" / "betas_testset" / "betas_csv_testset"
sys.path.insert(0, str(HERE.parent / "real-convergence"))
from run_real_convergence import distmat
from run_real_convergence_v2 import concept_all_layers
from run_ai_brain_vision import get_clip_image_rdm
import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def rankrows(A):  # rank along axis 1
    return np.argsort(np.argsort(A, axis=1), axis=1).astype(float)


def decode(brain, model):
    """leave-one-out: identify each concept from its brain profile via the model geometry. Spearman."""
    N = brain.shape[0]; top1 = top5 = 0; perc = []
    for i in range(N):
        others = np.array([j for j in range(N) if j != i])
        b = brain[i, others]; rb = np.argsort(np.argsort(b)).astype(float); rb -= rb.mean()
        M = model[:, others]; rM = rankrows(M); rM -= rM.mean(1, keepdims=True)
        s = (rM @ rb) / (np.sqrt((rM ** 2).sum(1)) * np.sqrt((rb ** 2).sum()) + 1e-12)
        order = np.argsort(-s)
        r = int(np.where(order == i)[0][0])
        top1 += (r == 0); top5 += (r < 5); perc.append(1 - r / (N - 1))
    return {"top1": round(top1 / N, 3), "top5": round(top5 / N, 3),
            "mean_pctile": round(float(np.mean(perc)), 3), "N": N, "chance_top1": round(1 / N, 3)}


def gpt2_rdm(words):
    tok = AutoTokenizer.from_pretrained("gpt2-large")
    mdl = AutoModelForCausalLM.from_pretrained("gpt2-large", torch_dtype=torch.float16).to(DEV).eval()
    R = distmat(np.stack([concept_all_layers(mdl, tok, w)[-1] for w in words])); del mdl; gc.collect()
    return R

def emb_rdm(words, repo):
    st = SentenceTransformer(repo, device=DEV); R = distmat(st.encode(words, normalize_embeddings=True)); del st; gc.collect(); return R

def lex_rdm(words):
    L = np.array([len(w) for w in words], float); return np.abs(L[:, None] - L[None, :])

# THINGS index for VICE
rows = (DATA / "things_concepts.tsv").read_text(encoding="utf-8").splitlines()[1:]
tindex = {r.split("\t")[0].strip().lower(): i for i, r in enumerate(rows)}
vice = np.load(DATA / "final_embedding.npy")

def perm_p(brain, model, obs, B=200):
    """permutation test: shuffle concept identities of the model RDM, redo decoding, p on mean_pctile."""
    rng = np.random.default_rng(0); N = brain.shape[0]; ge = 1
    for _ in range(B):
        p = rng.permutation(N)
        if decode(brain, model[np.ix_(p, p)])["mean_pctile"] >= obs:
            ge += 1
    return round(ge / (B + 1), 4)

def run(tag, words, brain):
    print(f"\n### {tag}: N={len(words)} ###", flush=True)
    res = {}
    llm_R = gpt2_rdm(words)
    res["free LLM (gpt2-large)"] = {**decode(brain, llm_R),
                                    "perm_p": perm_p(brain, llm_R, decode(brain, llm_R)["mean_pctile"])}
    res["embedder (mpnet)"] = decode(brain, emb_rdm(words, "sentence-transformers/all-mpnet-base-v2"))
    res["word-length (control)"] = decode(brain, lex_rdm(words))
    # CLIP-image (vision)
    kept, vis = get_clip_image_rdm(words)
    if len(kept) == len(words):
        res["CLIP-image (vision)"] = decode(brain, vis)
    else:
        ix = [words.index(w) for w in kept]
        res["CLIP-image (vision)"] = {**decode(brain[np.ix_(ix, ix)], vis), "note": f"on {len(kept)} w/ image"}
    # VICE on matched subset
    matched = [w for w in words if w in tindex]; mi = [words.index(w) for w in matched]
    vrdm = distmat(vice[[tindex[w] for w in matched]])
    res["human behaviour (VICE,1.5M)"] = {**decode(brain[np.ix_(mi, mi)], vrdm), "note": f"on {len(matched)} matched"}
    for k, v in res.items():
        print(f"  {k:30s} top1 {v['top1']:.3f}  top5 {v['top5']:.3f}  pctile {v['mean_pctile']:.3f}  (chance top1 {v['chance_top1']})"
              + (f"  [{v['note']}]" if 'note' in v else ""))
    return res

OUT = {}
# Mitchell-60 (clean meaning substrate)
bz = np.load(HERE / "brain_rdm.npz", allow_pickle=True)
OUT["Mitchell-2008 (word-reading)"] = run("MITCHELL-60", [str(w) for w in bz["nouns"]], bz["group"])

# THINGS-fMRI IT (visual paradigm)
concept_of = lambda s: s.rsplit("_", 1)[0].lower(); subs = ["sub-01", "sub-02", "sub-03"]; neural = []; cs0 = None
for s in subs:
    a = np.load(TF / f"{s}_TestResponsesIT.npy").astype(np.float64)
    df = pd.read_csv(TF / f"{s}_StimulusMetadataTestset.csv"); df["concept"] = df["stimulus"].map(concept_of)
    cs = sorted(df["concept"].unique()); cs0 = cs0 or cs; assert cs == cs0
    M = np.stack([a[df["concept"].values == c].mean(0) for c in cs]); M = (M - M.mean(0)) / (M.std(0) + 1e-9)
    neural.append(1.0 - np.corrcoef(M))
OUT["THINGS-fMRI IT (visual)"] = run("THINGS-100", cs0, np.mean(neural, axis=0))

(HERE / "zero_shot_decoder_result.json").write_text(json.dumps(OUT, indent=2), encoding="utf-8")
print("\nwrote zero_shot_decoder_result.json")
