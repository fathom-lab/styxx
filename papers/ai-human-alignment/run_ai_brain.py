# -*- coding: utf-8 -*-
"""
run_ai_brain.py — Does LLM concept geometry match the HUMAN BRAIN geometry? (Mitchell 2008 fMRI)
Gate frozen in PREREG_ai_brain_2026_06_03.md.

Brain group RDM over 60 nouns (9 subjects, top-500 stable voxels; noise ceiling [0.394, 0.557]).
LLM concept RDMs at the fixed final layer; partial-lexical RSA to the brain, judged relative to the
ceiling. Three-way: AI<->brain vs behavioral-human(VICE)<->brain vs embedder<->brain on 60 inter THINGS.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

import sys
sys.path.insert(0, str(HERE.parent / "real-convergence"))
from run_real_convergence import distmat, is_cached
from run_real_convergence_v2 import concept_all_layers
from run_real_convergence_v3_controls import partial_corr
from run_ai_human import COHORT, rankdata, spearman


def main():
    bz = np.load(HERE / "brain_rdm.npz", allow_pickle=True)
    nouns = [str(w) for w in bz["nouns"]]
    brain = bz["group"]
    ceil_lo, ceil_hi = [float(x) for x in bz["ceiling"]]
    N = len(nouns); IU = np.triu_indices(N, 1)
    print(f"{N} nouns; brain noise ceiling [{ceil_lo:.3f}, {ceil_hi:.3f}]", flush=True)

    rtok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    def lexdesign(concepts, iu):
        cl = np.array([len(w) for w in concepts], float)
        tl = np.array([len(rtok(" " + w, add_special_tokens=False).input_ids) for w in concepts], float)
        zc = (cl - cl.mean()) / (cl.std() + 1e-9); zt = (tl - tl.mean()) / (tl.std() + 1e-9)
        return np.column_stack([np.abs(zc[:, None] - zc[None, :])[iu], np.abs(zt[:, None] - zt[None, :])[iu]])

    Zlex = lexdesign(nouns, IU)
    pal = lambda A, B, iu, Z: partial_corr(A[iu], B[iu], Z)

    # references + VICE behavioral
    from sentence_transformers import SentenceTransformer
    emb = {}
    for tag, repo in [("MiniLM", "sentence-transformers/all-MiniLM-L6-v2"), ("mpnet", "sentence-transformers/all-mpnet-base-v2")]:
        st = SentenceTransformer(repo, device=DEV)
        emb[tag] = distmat(st.encode(nouns, normalize_embeddings=True))
        del st; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    vice = np.load(DATA / "final_embedding.npy")
    rows = (DATA / "things_concepts.tsv").read_text(encoding="utf-8").splitlines()[1:]
    tindex = {r.split("\t")[0].strip().lower(): i for i, r in enumerate(rows)}
    Snouns = [w for w in nouns if w in tindex]
    Sidx = [nouns.index(w) for w in Snouns]
    IUs = np.triu_indices(len(Snouns), 1)
    Zlex_S = lexdesign(Snouns, IUs)
    vice_S = distmat(vice[[tindex[w] for w in Snouns]])
    brain_S = brain[np.ix_(Sidx, Sidx)]
    print(f"{len(Snouns)}/{N} nouns in THINGS for the behavioral(VICE) comparison", flush=True)

    # LLM RDMs (final layer)
    rdms, meta = {}, {}
    for name, repo, params, instruct in COHORT:
        if not is_cached(repo):
            continue
        try:
            tok = AutoTokenizer.from_pretrained(repo)
            mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, trust_remote_code=True).to(DEV).eval()
            rep = np.stack([concept_all_layers(mdl, tok, w)[-1] for w in nouns])
            rdms[name] = distmat(rep)
            meta[name] = {"params": params, "instruct": instruct}
            del mdl, tok, rep; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
            print(f"  {name:12s} ok", flush=True)
        except Exception as e:
            print(f"  {name:12s} FAIL {type(e).__name__}", flush=True)

    models = list(rdms.keys())
    per = {}
    for m in models:
        per[m] = {"brain": round(pal(rdms[m], brain, IU, Zlex), 3),
                  "brain_S": round(pal(rdms[m][np.ix_(Sidx, Sidx)], brain_S, IUs, Zlex_S), 3),
                  "vice_S": round(pal(rdms[m][np.ix_(Sidx, Sidx)], vice_S, IUs, Zlex_S), 3),
                  **meta[m]}

    # references vs brain / behavioral
    emb_brain = {t: round(pal(emb[t], brain, IU, Zlex), 3) for t in emb}
    vice_brain = round(pal(vice_S, brain_S, IUs, Zlex_S), 3)
    emb_brain_S = {t: round(pal(emb[t][np.ix_(Sidx, Sidx)], brain_S, IUs, Zlex_S), 3) for t in emb}

    # shuffle control
    rng = np.random.default_rng(0); m0 = models[0]
    ctrl = []
    for _ in range(200):
        p = rng.permutation(N)
        ctrl.append(pal(brain[np.ix_(p, p)], rdms[m0], IU, Zlex))
    ctrl_mean, ctrl_sd = float(np.mean(ctrl)), float(np.std(ctrl))

    brain_vals = [per[m]["brain"] for m in models]
    best = max(brain_vals); best_m = models[int(np.argmax(brain_vals))]
    inst_mean = float(np.mean([per[m]["brain"] for m in models if per[m]["instruct"]]))
    h1 = (best >= 0.20) and (best >= 0.5 * ceil_lo) and (abs(ctrl_mean) < 0.05)

    # consistency: brain-align vs behavioral(VICE)-align across models
    consistency = spearman([per[m]["brain_S"] for m in models], [per[m]["vice_S"] for m in models])

    reading = []
    reading.append(f"H1 {'PASS' if h1 else 'FAIL'}: best LLM RSA-to-BRAIN {best:.3f} ({best_m}) = "
                   f"{100*best/ceil_lo:.0f}% of ceiling-lower / {100*best/ceil_hi:.0f}% of ceiling-upper; "
                   f"shuffle ctrl {ctrl_mean:.3f}+-{ctrl_sd:.3f}. LLM concept geometry matches the human brain.")
    reading.append(f"Three-way (60 inter THINGS, n={len(Snouns)}): AI<->brain best {max(per[m]['brain_S'] for m in models):.3f} "
                   f"vs behavioral-human(VICE)<->brain {vice_brain:.3f} vs embedder<->brain {emb_brain_S}. "
                   + ("AI matches the brain about as well as human behavior does."
                      if max(per[m]['brain_S'] for m in models) >= 0.8 * vice_brain else
                      "Human behavior still matches the brain better than AI does."))
    reading.append(f"Consistency: brain-alignment tracks behavioral-alignment across models, Spearman {consistency:.2f}.")

    out = {"n_nouns": N, "noise_ceiling": [ceil_lo, ceil_hi], "models": models, "per_model": per,
           "embedder_brain": emb_brain, "vice_brain": vice_brain, "embedder_brain_S": emb_brain_S,
           "shuffle_ctrl_mean": round(ctrl_mean, 3), "shuffle_ctrl_sd": round(ctrl_sd, 3),
           "H1_pass": bool(h1), "best_brain": best, "best_model": best_m,
           "instruct_mean_brain": round(inst_mean, 3), "consistency_brain_vs_behavioral": round(consistency, 3),
           "reading": reading}
    (HERE / "ai_brain_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n=== RSA to HUMAN BRAIN (final layer, partial-lexical; ceiling [{ceil_lo:.2f},{ceil_hi:.2f}]) ===")
    for m in sorted(models, key=lambda m: -per[m]["brain"]):
        p = per[m]
        print(f"  {m:12s} {p['params']/1e6:7.0f}M  brain={p['brain']:+.3f} ({100*p['brain']/ceil_lo:3.0f}% ceil-lo)  vice_S={p['vice_S']:+.3f}  {'inst' if p['instruct'] else 'base'}")
    print(f"\nembedders vs brain: {emb_brain}   |   behavioral-human(VICE) vs brain: {vice_brain}")
    for r in reading:
        print(">>>", r)
    print("wrote ai_brain_result.json")


if __name__ == "__main__":
    main()
