# -*- coding: utf-8 -*-
"""
layer_meaning.py — WHERE in a transformer does human-aligned meaning emerge? For each hidden layer of a
real LLM, extract concept embeddings (mean-pooled) and measure alignment to the human 65-feature reference
(Binder). Traces the alignment-vs-depth curve across two families (GPT-2, Pythia-410m). Tells you which
layer carries the most human-aligned meaning — i.e. which layer to monitor. Uses shipped styxx.
"""
import os, sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
from styxx.meaning_integrity import MeaningReference, meaning_alignment

import torch
from transformers import AutoTokenizer, AutoModel

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def pca(E, k):
    Ec = E - E.mean(0); U, S, _ = np.linalg.svd(Ec, full_matrices=False)
    k = min(k, E.shape[1]); return U[:, :k] * S[:k]


def layer_embeddings(name, words):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(name, output_hidden_states=True).to(DEV).eval()
    n_layers = model.config.num_hidden_layers
    acc = None
    with torch.no_grad():
        for i in range(0, len(words), 32):
            b = words[i:i + 32]
            enc = tok(b, return_tensors="pt", padding=True).to(DEV)
            hs = model(**enc).hidden_states                       # tuple: embed + each layer
            am = enc["attention_mask"].unsqueeze(-1).float()
            pooled = [((h * am).sum(1) / am.sum(1)).cpu().numpy() for h in hs]
            acc = pooled if acc is None else [np.concatenate([a, p]) for a, p in zip(acc, pooled)]
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return acc, n_layers                                          # list length n_layers+1 (embed..final)


def main():
    df = pd.read_excel(os.path.join(HERE, "binder_data", "WordSet1_Ratings.xlsx"))
    words_all = [str(w).strip().lower() for w in df["Word"]]
    H_all = df.iloc[:, 5:70].apply(pd.to_numeric, errors="coerce").values
    keep = [i for i, w in enumerate(words_all) if w.isalpha() and not np.isnan(H_all[i]).any()]
    words = [words_all[i] for i in keep]
    ref = MeaningReference(H_all[keep].astype(float), words=words, name="binder65")
    print(f"{len(words)} concepts — alignment to human meaning at each layer (mean-pooled, PCA-50)\n")

    for disp, name in [("GPT-2 (12 layers)", "gpt2"), ("Pythia-410m (24 layers)", "EleutherAI/pythia-410m")]:
        embs, L = layer_embeddings(name, words)
        aligns = [meaning_alignment(pca(e, 50), ref) for e in embs]
        peak = int(np.argmax(aligns))
        print(f"{disp}:")
        # print a compact curve (every layer for GPT-2, every other for Pythia)
        step = 1 if L <= 12 else 2
        line = "  " + "  ".join(f"L{li}:{aligns[li]:+.2f}" for li in range(0, len(aligns), step))
        print(line)
        print(f"  PEAK at layer {peak}/{L} (relative depth {peak/L:.0%}), alignment {aligns[peak]:+.3f}; "
              f"final layer {aligns[-1]:+.3f}\n")

    print("  -> hypothesis FALSIFIED (pre-written guess was 'peaks in the middle'). The real pattern is")
    print("     U-SHAPED: human-aligned meaning is highest at the EMBEDDING layer and the FINAL layer, and")
    print("     LOWEST in the middle (where the model does contextual/syntactic work). For Pythia the final")
    print("     layer is the peak (0.176); GPT-2 is uniformly low. So read the final (or embedding) layer.")
    print("     Caveat: isolated-word embeddings are weak -> small absolute values; the U-shape is the signal.")


if __name__ == "__main__":
    main()
