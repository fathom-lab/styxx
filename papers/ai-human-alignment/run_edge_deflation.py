# -*- coding: utf-8 -*-
"""
run_edge_deflation.py — find the real edge by attacking the shared-language confound.

Deflationary claim: the AI<->brain "universal" is just shallow human-language co-occurrence statistics
(both AI and brain reflect language exposure). Decisive test: does the DEEP LLM predict the human brain
(Mitchell 2008 fMRI, 60 nouns) BEYOND a pure co-occurrence model (GloVe-50, shallow language stats) and
a VISION model (CLIP-image)? Variance partition + partial RSA, vs the fMRI noise ceiling.

  - if GloVe<->brain ~ LLM<->brain AND unique-LLM-beyond-(GloVe+vision) ~ 0  -> DEFLATION: it's shallow
    language statistics; the deep model / "universal meaning" add nothing. (A real, contrarian finding.)
  - if the LLM survives -> deep models capture brain-relevant structure beyond shallow co-occurrence.

HONEST LIMIT (stated up front): GloVe, the LLM, CLIP, VICE and the brain are ALL human-derived. This can
separate "shallow language" from "deeper conceptual", but it CANNOT test "substrate-independent universal"
(would-emerge-without-humans). The strongest honest claim available is about shared HUMAN-derived structure.
"""
from __future__ import annotations

import gc, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"; BRAIN = HERE / "brain"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
import sys
sys.path.insert(0, str(HERE.parent / "real-convergence"))
from run_real_convergence import distmat, is_cached
from run_real_convergence_v2 import concept_all_layers
from run_real_convergence_v3_controls import partial_corr
from run_ai_human import COHORT


def r2(y, *cols):
    X = np.column_stack([np.ones_like(y)] + list(cols))
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    return float(1 - (res @ res) / ((y - y.mean()) @ (y - y.mean())))


def main():
    bz = np.load(HERE / "brain_rdm.npz", allow_pickle=True)
    nouns = [str(w) for w in bz["nouns"]]
    brain = bz["group"]; ceil_lo = float(bz["ceiling"][0])

    # vision-covered subset defines S (also has VICE); GloVe covers all
    clip = np.load(BRAIN / "clip_image_emb.npz", allow_pickle=True)
    S = [w for w in nouns if w in clip.files]
    si = [nouns.index(w) for w in S]
    N = len(S); IU = np.triu_indices(N, 1)
    print(f"analysis set: {N} nouns (vision-covered); GloVe covers all", flush=True)

    import gensim.downloader as api
    glove = api.load("glove-wiki-gigaword-50")

    rtok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    cl = np.array([len(w) for w in S], float); tl = np.array([len(rtok(" " + w, add_special_tokens=False).input_ids) for w in S], float)
    zc = (cl - cl.mean()) / (cl.std() + 1e-9); zt = (tl - tl.mean()) / (tl.std() + 1e-9)
    L = np.column_stack([np.abs(zc[:, None] - zc[None, :])[IU], np.abs(zt[:, None] - zt[None, :])[IU]])

    brain_v = brain[np.ix_(si, si)][IU]
    glove_v = distmat(np.stack([glove[w] for w in S]))[IU]
    vision_v = distmat(np.stack([clip[w] for w in S]))[IU]
    vice = np.load(DATA / "final_embedding.npy")
    rows = (DATA / "things_concepts.tsv").read_text(encoding="utf-8").splitlines()[1:]
    tindex = {r.split("\t")[0].strip().lower(): i for i, r in enumerate(rows)}
    vice_v = distmat(vice[[tindex[w] for w in S]])[IU]

    # deep LLM consensus (final layer) over S
    rdms = []
    for name, repo, params, instruct in COHORT:
        if not is_cached(repo):
            continue
        try:
            tok = AutoTokenizer.from_pretrained(repo)
            mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, trust_remote_code=True).to(DEV).eval()
            rep = np.stack([concept_all_layers(mdl, tok, w)[-1] for w in S])
            rdms.append(distmat(rep)[IU])
            del mdl, tok, rep; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
            print(f"  {name} ok", flush=True)
        except Exception as e:
            print(f"  {name} FAIL {type(e).__name__}", flush=True)
    llm_v = np.mean(rdms, axis=0)

    pl = lambda v: round(partial_corr(v, brain_v, L), 3)
    partials = {"GloVe(shallow language)->brain": pl(glove_v), "deep LLM->brain": pl(llm_v),
                "vision(CLIP-image)->brain": pl(vision_v), "human behavior(VICE)->brain": pl(vice_v)}

    # how much do GloVe vs LLM explain of EACH OTHER (are they the same structure)?
    glove_vs_llm = round(partial_corr(glove_v, llm_v, L), 3)

    # variance partition of the brain
    R = {
        "lex": r2(brain_v, L),
        "lex+glove": r2(brain_v, glove_v, L),
        "lex+glove+vision": r2(brain_v, glove_v, vision_v, L),
        "lex+glove+vision+llm": r2(brain_v, glove_v, vision_v, llm_v, L),
        "lex+vision+llm": r2(brain_v, vision_v, llm_v, L),
        "lex+glove+llm": r2(brain_v, glove_v, llm_v, L),
        "lex+llm": r2(brain_v, llm_v, L),
        "lex+glove+vision+vice+llm": r2(brain_v, glove_v, vision_v, vice_v, llm_v, L),
        "lex+glove+vision+vice": r2(brain_v, glove_v, vision_v, vice_v, L),
    }
    unique = {
        "GloVe_total_over_lex": R["lex+glove"] - R["lex"],
        "LLM_total_over_lex": R["lex+llm"] - R["lex"],
        "LLM_unique_beyond_glove": R["lex+glove+llm"] - R["lex+glove"],
        "LLM_unique_beyond_glove+vision": R["lex+glove+vision+llm"] - R["lex+glove+vision"],
        "LLM_unique_beyond_glove+vision+behavior": R["lex+glove+vision+vice+llm"] - R["lex+glove+vision+vice"],
        "GloVe_unique_beyond_llm+vision": R["lex+glove+vision+llm"] - R["lex+vision+llm"],
    }

    # ROBUSTNESS: repeat the GloVe-vs-LLM deflation on the CLEAN human BEHAVIORAL target (VICE, 1.5M
    # judgments, high reliability) -- if the brain deflation were just low power, the deep LLM should
    # beat GloVe here, where power is not the bottleneck.
    def target_test(y, predA, predB):
        return {"A_RSA": round(partial_corr(predA, y, L), 3), "B_RSA": round(partial_corr(predB, y, L), 3),
                "A_total_over_lex": round(r2(y, predA, L) - r2(y, L), 4),
                "B_unique_beyond_A": round(r2(y, predA, predB, L) - r2(y, predA, L), 4)}
    vice_robust = target_test(vice_v, glove_v, llm_v)   # A=GloVe, B=deep LLM, target=clean human behavior
    print(f"\n[robustness | target = CLEAN human behavior VICE] GloVe->VICE {vice_robust['A_RSA']} vs "
          f"deep-LLM->VICE {vice_robust['B_RSA']}; LLM unique beyond GloVe = {100*vice_robust['B_unique_beyond_A']:.2f}%", flush=True)

    u = unique["LLM_unique_beyond_glove+vision"]
    brain_deflated = (partials["deep LLM->brain"] <= partials["GloVe(shallow language)->brain"] + 0.03) and (u < 0.01)
    depth_real_in_behavior = vice_robust["B_unique_beyond_A"] >= 0.03
    if brain_deflated and depth_real_in_behavior:
        verdict = (f"DEPTH IS REAL IN BEHAVIOR, INVISIBLE IN THE BRAIN. Deep LLMs DO capture human conceptual structure "
                   f"beyond shallow co-occurrence -- on clean behavioral data they add {100*vice_robust['B_unique_beyond_A']:.1f}% "
                   f"unique over GloVe (LLM {vice_robust['B_RSA']} > GloVe {vice_robust['A_RSA']}). BUT that depth does NOT reach "
                   f"the brain: GloVe predicts the brain as well as the deep LLM ({partials['GloVe(shallow language)->brain']} vs "
                   f"{partials['deep LLM->brain']}), deep-unique ~{100*u:.1f}%. So 'deep LLMs UNIQUELY predict the brain' is NOT "
                   f"supported at fMRI resolution -- a 2014 GloVe matches it equally. The deep advantage is real in concepts, "
                   f"absent (or unresolvable) in the neural signal.")
    elif brain_deflated:
        verdict = (f"FULL DEFLATION: shallow co-occurrence predicts the brain as well as the deep LLM, AND the deep LLM adds "
                   f"little even on clean behavioral data ({100*vice_robust['B_unique_beyond_A']:.1f}%). Mostly shallow language statistics.")
    else:
        verdict = (f"DEPTH SURVIVES on the brain: deep LLM {partials['deep LLM->brain']} > GloVe {partials['GloVe(shallow language)->brain']}, "
                   f"unique +{100*u:.1f}% beyond GloVe+vision.")

    out = {"n_nouns": N, "noise_ceiling_lo": ceil_lo, "partial_lexical_RSA_to_brain": partials,
           "glove_vs_llm_partial": glove_vs_llm, "brain_R2": {k: round(v, 4) for k, v in R.items()},
           "unique_contributions": {k: round(v, 4) for k, v in unique.items()},
           "robustness_clean_behavioral_target": vice_robust, "brain_deflated": bool(brain_deflated),
           "depth_real_in_behavior": bool(depth_real_in_behavior), "verdict": verdict,
           "honest_limit": "All sources are human-derived; this separates shallow-language from deeper-conceptual, but cannot test substrate-independence."}
    (HERE / "edge_deflation_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== partial-lexical RSA to the HUMAN BRAIN ===")
    for k, v in partials.items():
        print(f"  {k:34s} {v:+.3f}   ({100*v/ceil_lo:.0f}% of ceiling-lo)")
    print(f"\nGloVe vs deep-LLM geometry (partial-lex): {glove_vs_llm:+.3f}  (high = same structure)")
    print("\nunique contributions to brain R^2:")
    for k, v in unique.items():
        print(f"  {k:42s} {100*v:+.2f}%")
    print(f"\n>>> {verdict}")
    print(">>> HONEST LIMIT:", out["honest_limit"])
    print("wrote edge_deflation_result.json")


if __name__ == "__main__":
    main()
