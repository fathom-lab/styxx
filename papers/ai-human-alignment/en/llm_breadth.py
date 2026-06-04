# -*- coding: utf-8 -*-
"""
llm_breadth.py — the shipped monitor on REAL, recognizable open LLMs across families (not toy word vectors).
For each of GPT-2 / Pythia / OPT / GPT-Neo: extract concept embeddings (mean-pooled last hidden layer),
and ask (a) which family means most like a HUMAN (alignment to Binder 65-feature ref), and (b) do the
families mean the same as each other (reference-free meaning_agreement) + where they diverge. Uses shipped
styxx 7.12.0.
"""
import os, sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
from styxx.meaning_integrity import MeaningReference, meaning_alignment, meaning_agreement

import torch
from transformers import AutoTokenizer, AutoModel

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = {                                            # diverse families; only safetensors checkpoints load
    "GPT-2": "gpt2",
    "DistilGPT-2": "distilgpt2",
    "Pythia-160m": "EleutherAI/pythia-160m",
    "Pythia-410m": "EleutherAI/pythia-410m",
    "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
    "BLOOM-560m": "bigscience/bloom-560m",
}


def embed_llm(name, words):
    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name).to(DEV).eval()
    except Exception as e:
        print(f"    (skip {name}: {str(e)[:60]})", flush=True)
        return None
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = []
    with torch.no_grad():
        for i in range(0, len(words), 32):
            b = words[i:i + 32]
            enc = tok(b, return_tensors="pt", padding=True).to(DEV)
            hs = model(**enc).last_hidden_state
            am = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((hs * am).sum(1) / am.sum(1)).cpu().numpy())   # mean-pool content tokens
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(out)


def pca(E, k):
    Ec = E - E.mean(0); U, S, _ = np.linalg.svd(Ec, full_matrices=False)
    k = min(k, E.shape[1]); return U[:, :k] * S[:k]


def main():
    df = pd.read_excel(os.path.join(HERE, "binder_data", "WordSet1_Ratings.xlsx"))
    words_all = [str(w).strip().lower() for w in df["Word"]]
    H_all = df.iloc[:, 5:70].apply(pd.to_numeric, errors="coerce").values
    keep = [i for i, w in enumerate(words_all) if w.isalpha() and not np.isnan(H_all[i]).any()]
    words = [words_all[i] for i in keep]
    ref = MeaningReference(H_all[keep].astype(float), words=words, name="binder65")
    print(f"{len(words)} concepts, {len(MODELS)} real open LLMs across families\n")

    emb = {}
    print("(a) WHICH FAMILY MEANS MOST LIKE A HUMAN? (alignment to Binder 65-feature human ref, PCA-50)")
    print("    (note: isolated-word mean-pooled embeddings from small base LMs are weak; the RANKING is the signal)")
    for name, hf in MODELS.items():
        e = embed_llm(hf, words)
        if e is None:
            continue
        emb[name] = e
        print(f"    {name:14s} {meaning_alignment(pca(e, 50), ref):+.3f}", flush=True)

    print("\n(b) DO THE FAMILIES MEAN THE SAME? (reference-free meaning_agreement, PCA-50)")
    names = list(emb)
    P = {k: pca(v, 50) for k, v in emb.items()}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = meaning_agreement(P[names[i]], P[names[j]], words=words, top=4)
            div = ", ".join(w for w, _ in a["most_divergent_concepts"][:3])
            print(f"    {names[i]:12s} vs {names[j]:12s}: agreement {a['agreement']:+.3f}   diverge: {div}")
    print("\n  -> the shipped monitor + meaning_agreement run on real LLM internals across 4 families.")


if __name__ == "__main__":
    main()
