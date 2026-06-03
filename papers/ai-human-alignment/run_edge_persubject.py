# -*- coding: utf-8 -*-
"""
run_edge_persubject.py — higher-power test of "does depth beat shallow co-occurrence in the BRAIN?"

The group deflation (GloVe 0.180 == deep-LLM 0.182) averaged 9 subjects, which can wash out a small
consistent effect. This is the paired version: per subject, deep-LLM<->brain vs GloVe<->brain
(partial-lexical), then a paired sign/t test across the 9 subjects. A small-but-CONSISTENT deep
advantage (e.g. 8-9/9 subjects) => depth IS in the brain, the group test was underpowered. A coin-flip
(~5/9) => the deflation is robust: depth genuinely does not reach the (Mitchell-resolution) brain.
"""
from __future__ import annotations

import gc, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
import sys
sys.path.insert(0, str(HERE.parent / "real-convergence"))
from run_real_convergence import distmat, is_cached
from run_real_convergence_v2 import concept_all_layers
from run_real_convergence_v3_controls import partial_corr
from run_ai_human import COHORT
from build_brain_rdm import subject_rdm, BRAIN


def main():
    bz = np.load(HERE / "brain_rdm.npz", allow_pickle=True)
    nouns = [str(w) for w in bz["nouns"]]
    N = len(nouns); IU = np.triu_indices(N, 1)

    import gensim.downloader as api
    glove = api.load("glove-wiki-gigaword-50")
    glove_rdm = distmat(np.stack([glove[w] for w in nouns]))

    rtok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    cl = np.array([len(w) for w in nouns], float); tl = np.array([len(rtok(" " + w, add_special_tokens=False).input_ids) for w in nouns], float)
    zc = (cl - cl.mean()) / (cl.std() + 1e-9); zt = (tl - tl.mean()) / (tl.std() + 1e-9)
    Zlex = np.column_stack([np.abs(zc[:, None] - zc[None, :])[IU], np.abs(zt[:, None] - zt[None, :])[IU]])

    # deep LLM consensus over the 60 nouns (final layer)
    rdms = []
    for name, repo, params, instruct in COHORT:
        if not is_cached(repo):
            continue
        try:
            tok = AutoTokenizer.from_pretrained(repo)
            mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, trust_remote_code=True).to(DEV).eval()
            rep = np.stack([concept_all_layers(mdl, tok, w)[-1] for w in nouns])
            rdms.append(distmat(rep))
            del mdl, tok, rep; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
        except Exception:
            pass
    llm_rdm = np.mean(rdms, axis=0)

    glove_v, llm_v = glove_rdm[IU], llm_rdm[IU]

    rows = []
    for p in sorted(BRAIN.glob("data-science-P*.mat")):
        ns, rdm, _ = subject_rdm(p, 500)
        assert ns == nouns
        bv = rdm[IU]
        g = partial_corr(glove_v, bv, Zlex)
        l = partial_corr(llm_v, bv, Zlex)
        rows.append({"subject": p.stem[-2:], "glove": round(g, 3), "llm": round(l, 3), "diff_llm_minus_glove": round(l - g, 4)})
        print(f"  {p.stem}: GloVe {g:+.3f}  deep-LLM {l:+.3f}  diff {l-g:+.4f}", flush=True)

    diffs = np.array([r["diff_llm_minus_glove"] for r in rows])
    n = len(diffs)
    wins = int((diffs > 0).sum())
    mean_d = float(diffs.mean()); sd = float(diffs.std(ddof=1))
    t = mean_d / (sd / np.sqrt(n) + 1e-12)
    # two-sided sign-test p via binomial
    from math import comb
    k = max(wins, n - wins)
    p_sign = 2 * sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)

    if wins >= n - 1 and mean_d > 0:
        verdict = (f"DEPTH IS IN THE BRAIN (group test was underpowered): the deep LLM beats GloVe in {wins}/{n} subjects "
                   f"(mean diff +{mean_d:.4f}, paired t={t:.2f}, sign-test p={p_sign:.3f}). The deep advantage that is large in "
                   f"behavior IS present in the brain, just small at this resolution -> reading (a) MEASUREMENT.")
    elif wins <= n - wins + 1 and abs(mean_d) < 0.003:
        verdict = (f"DEFLATION ROBUST: deep-LLM vs GloVe in the brain is a coin-flip ({wins}/{n} subjects, mean diff {mean_d:+.4f}, "
                   f"t={t:.2f}, sign p={p_sign:.3f}). Depth genuinely does NOT reach the Mitchell-resolution brain -> the shared "
                   f"AI<->brain structure is shallow co-occurrence; the deep advantage is behavioral, not (here) neural.")
    else:
        verdict = (f"INCONCLUSIVE: {wins}/{n} subjects favor deep-LLM, mean diff {mean_d:+.4f}, t={t:.2f}, sign p={p_sign:.3f}. "
                   f"A small lean but not a clean call; needs higher-SNR neural data.")

    out = {"n_subjects": n, "per_subject": rows, "deep_wins": wins, "mean_diff": round(mean_d, 4),
           "paired_t": round(float(t), 2), "sign_test_p": round(float(p_sign), 4), "verdict": verdict}
    (HERE / "edge_persubject_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\ndeep-LLM beats GloVe in {wins}/{n} subjects; mean diff {mean_d:+.4f}; paired t={t:.2f}; sign-test p={p_sign:.3f}")
    print(f">>> {verdict}")
    print("wrote edge_persubject_result.json")


if __name__ == "__main__":
    main()
