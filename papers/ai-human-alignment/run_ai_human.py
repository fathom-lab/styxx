# -*- coding: utf-8 -*-
"""
run_ai_human.py — Is the geometry machines CONVERGE to the HUMAN geometry of meaning?
Gate frozen in PREREG_ai_human_2026_06_03.md.

Human ground truth = VICE/SPoSE human concept embedding (1854 objects x 42 human dims, ~1.5M human
odd-one-out judgments). 122 of our concrete concepts have a human vector. For ~14 LLMs (base+instruct,
70M-3B) we read concept geometry at the fixed FINAL layer (where v3 showed it lives; not chosen using
human data), build RDMs, and test:
  H1  do converging models align with humans above chance (partial-lexical)?            [floor, known]
  H2  does machine-convergence predict human-alignment, beyond model size?              [THE claim]
  H3  is the cross-model CONSENSUS more human-aligned than the median single model?     [crowd=human]
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
import run_real_convergence as RC
import run_real_convergence_confirm as RCC

COHORT = [  # (name, repo, params, converging_instruct)
    ("Qwen-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", 1.54e9, True),
    ("Qwen-3B", "Qwen/Qwen2.5-3B-Instruct", 3.09e9, True),
    ("Llama-1B", "meta-llama/Llama-3.2-1B-Instruct", 1.24e9, True),
    ("Llama-3B", "meta-llama/Llama-3.2-3B-Instruct", 3.21e9, True),
    ("Phi-3.5", "microsoft/Phi-3.5-mini-instruct", 3.8e9, True),
    ("gemma-2-2b", "google/gemma-2-2b-it", 2.6e9, True),
    ("Qwen-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", 0.494e9, False),
    ("pythia-70m", "EleutherAI/pythia-70m", 70e6, False),
    ("pythia-160m", "EleutherAI/pythia-160m", 160e6, False),
    ("pythia-410m", "EleutherAI/pythia-410m", 410e6, False),
    ("gpt2", "gpt2", 124e6, False),
    ("gpt2-large", "gpt2-large", 774e6, False),
    ("gpt2-xl", "gpt2-xl", 1.558e9, False),
]


def rankdata(x):
    return np.argsort(np.argsort(np.asarray(x, float))).astype(float)


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    c = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    rxy, rxz, ryz = c(rx, ry), c(rx, rz), c(ry, rz)
    return float((rxy - rxz * ryz) / (np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)) + 1e-12))


def spearman(x, y):
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def main():
    # --- matched concepts + human RDM ---
    vice = np.load(DATA / "final_embedding.npy")
    rows = (DATA / "things_concepts.tsv").read_text(encoding="utf-8").splitlines()[1:]
    things = [r.split("\t")[0].strip().lower() for r in rows]
    tindex = {w: i for i, w in enumerate(things)}
    mine = list(RC.CONCEPTS) + list(RCC.CONCEPTS)
    CONCEPTS, hrows = [], []
    for w in mine:
        if w.lower() in tindex:                 # EXACT match only (safe)
            CONCEPTS.append(w)
            hrows.append(tindex[w.lower()])
    N = len(CONCEPTS)
    IU = np.triu_indices(N, 1)
    human_RDM = distmat(vice[hrows])
    print(f"matched {N} concepts; human embedding {vice.shape}", flush=True)

    # --- lexical control ---
    charlen = np.array([len(w) for w in CONCEPTS], float)
    rtok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    toklen = np.array([len(rtok(" " + w, add_special_tokens=False).input_ids) for w in CONCEPTS], float)
    zc = (charlen - charlen.mean()) / (charlen.std() + 1e-9)
    zt = (toklen - toklen.mean()) / (toklen.std() + 1e-9)
    Zlex = np.column_stack([np.abs(zc[:, None] - zc[None, :])[IU], np.abs(zt[:, None] - zt[None, :])[IU]])

    def palign(Da, Db):
        return partial_corr(Da[IU], Db[IU], Zlex)

    # --- embedder references ---
    from sentence_transformers import SentenceTransformer
    emb_ref = {}
    for tag, repo in [("MiniLM", "sentence-transformers/all-MiniLM-L6-v2"), ("mpnet", "sentence-transformers/all-mpnet-base-v2")]:
        st = SentenceTransformer(repo, device=DEV)
        emb_ref[tag] = distmat(st.encode(CONCEPTS, normalize_embeddings=True))
        del st; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None

    # --- LLM RDMs (final layer) ---
    rdms, meta = {}, {}
    for name, repo, params, instruct in COHORT:
        if not is_cached(repo):
            print(f"  (skip {name}: not cached)", flush=True); continue
        try:
            tok = AutoTokenizer.from_pretrained(repo)
            mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, trust_remote_code=True).to(DEV).eval()
            rep = np.stack([concept_all_layers(mdl, tok, w)[-1] for w in CONCEPTS])  # FINAL layer
            rdms[name] = distmat(rep)
            meta[name] = {"params": params, "log10_params": float(np.log10(params)), "instruct": instruct}
            del mdl, tok, rep; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
            print(f"  {name:12s} ok", flush=True)
        except Exception as e:
            print(f"  {name:12s} FAIL {type(e).__name__}: {str(e)[:70]}", flush=True)

    models = list(rdms.keys())
    # consensus RDM (mean dissimilarity across the machine crowd)
    consensus = np.mean([rdms[m] for m in models], axis=0)

    # per-model metrics
    per = {}
    for m in models:
        human = palign(rdms[m], human_RDM)
        conv = float(np.mean([palign(rdms[m], rdms[o]) for o in models if o != m]))
        per[m] = {"human_align": round(human, 3), "convergence": round(conv, 3), **meta[m]}

    # H1
    conv_inst = [per[m]["human_align"] for m in models if per[m]["instruct"]]
    h1_mean = float(np.mean(conv_inst)) if conv_inst else float("nan")
    rng = np.random.default_rng(0)
    m0 = models[0]
    ctrl = []
    for _ in range(50):
        p = rng.permutation(N)
        ctrl.append(palign(human_RDM[np.ix_(p, p)], rdms[m0]))   # shuffle human concept labels
    ctrl_mean, ctrl_sd = float(np.mean(ctrl)), float(np.std(ctrl))
    h1 = (h1_mean >= 0.20) and (abs(ctrl_mean) < 0.05)

    # H2
    conv_v = [per[m]["convergence"] for m in models]
    hum_v = [per[m]["human_align"] for m in models]
    lp_v = [per[m]["log10_params"] for m in models]
    rho = spearman(conv_v, hum_v)
    rho_partial = partial_spearman(conv_v, hum_v, lp_v)
    h2 = (rho >= 0.50) and (rho_partial > 0)

    # H3
    cons_human = palign(consensus, human_RDM)
    median_ind = float(np.median(hum_v))
    h3 = cons_human > median_ind

    emb_human = {t: round(palign(emb_ref[t], human_RDM), 3) for t in emb_ref}
    phi = per.get("Phi-3.5")
    phi_rank = (sorted(models, key=lambda m: per[m]["human_align"]).index("Phi-3.5") + 1) if "Phi-3.5" in per else None

    reading = []
    reading.append(f"H1 {'PASS' if h1 else 'FAIL'}: converging models align with HUMAN geometry "
                   f"{h1_mean:.3f} (ctrl {ctrl_mean:.3f}+-{ctrl_sd:.3f}).")
    reading.append(f"H2 {'PASS' if h2 else 'FAIL'}: convergence->humanness Spearman {rho:.2f} "
                   f"(partialling out size {rho_partial:.2f}). "
                   + ("The machine consensus IS the human-aligned geometry, beyond capability."
                      if h2 else "Convergence does not cleanly predict human-alignment beyond size."))
    reading.append(f"H3 {'PASS' if h3 else 'FAIL'}: consensus human-align {cons_human:.3f} vs median single "
                   f"{median_ind:.3f}.")
    if phi_rank:
        reading.append(f"Phi-3.5 (machine-convergence outlier) ranks {phi_rank}/{len(models)} in human-alignment "
                       f"({phi['human_align']}).")

    out = {"n_concepts": N, "models": models, "per_model": per,
           "H1": {"pass": bool(h1), "mean_human_align_instruct": round(h1_mean, 3), "ctrl_mean": round(ctrl_mean, 3), "ctrl_sd": round(ctrl_sd, 3)},
           "H2": {"pass": bool(h2), "spearman_conv_human": round(rho, 3), "partial_out_size": round(rho_partial, 3)},
           "H3": {"pass": bool(h3), "consensus_human_align": round(cons_human, 3), "median_individual": round(median_ind, 3)},
           "embedder_human_align": emb_human, "phi_human_rank": phi_rank, "reading": reading}
    (HERE / "ai_human_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== per-model: human-alignment & machine-convergence (final layer, partial-lexical) ===")
    for m in sorted(models, key=lambda m: -per[m]["human_align"]):
        p = per[m]
        print(f"  {m:12s} {p['params']/1e6:7.0f}M  human={p['human_align']:+.3f}  convergence={p['convergence']:+.3f}  {'inst' if p['instruct'] else 'base'}")
    print(f"\nembedder refs vs human: {emb_human}")
    print(f"consensus vs human: {cons_human:.3f}   (median single {median_ind:.3f})")
    for r in reading:
        print(">>>", r)
    print("wrote ai_human_result.json")


if __name__ == "__main__":
    main()
