# -*- coding: utf-8 -*-
"""
deepfair_binder.py — give DEEP a fair shot on the rich Binder reference before concluding shallow>deep.
The earlier BERT was mean-pooled INCLUDING [CLS]/[SEP] (dilutes the word). Here: proper contextual word
embedding = mean of the word's CONTENT subword tokens (excl specials), averaged over a few templates.
Compare GloVe-300 (shallow) vs MiniLM vs BERT-proper (deep), dim-matched, bootstrapped.
"""
import os, sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cn"))
from meaning_integrity import MeaningReference, alignment, _rdm


def pca(E, k):
    Ec = E - E.mean(0); U, S, _ = np.linalg.svd(Ec, full_matrices=False)
    k = min(k, E.shape[1]); return U[:, :k] * S[:k]


def main():
    df = pd.read_excel(os.path.join(HERE, "binder_data", "WordSet1_Ratings.xlsx"))
    words_all = [str(w).strip().lower() for w in df["Word"]]
    H_all = df.iloc[:, 5:70].apply(pd.to_numeric, errors="coerce").values

    import gensim.downloader as gd
    import torch
    from transformers import AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    glove = gd.load("glove-wiki-gigaword-300")
    keep = [i for i, w in enumerate(words_all) if w in glove and not np.isnan(H_all[i]).any()]
    words = [words_all[i] for i in keep]
    human = H_all[keep].astype(float)
    gv = np.array([glove[w] for w in words], float)
    mini = np.asarray(SentenceTransformer("all-MiniLM-L6-v2").encode(words, show_progress_bar=False), float)

    # proper contextual deep word embedding (BERT): mean of word's content subword tokens, over templates
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    dev = "cuda" if torch.cuda.is_available() else "cpu"; model = model.to(dev).eval()
    templates = ["{w}", "a {w}", "the {w}", "this is a {w} .", "i saw the {w} ."]
    print("BERT proper (content tokens, templated)...", flush=True)
    vecs = np.zeros((len(words), 768))
    with torch.no_grad():
        for ti, tmpl in enumerate(templates):
            for i in range(0, len(words), 48):
                b = words[i:i + 48]
                texts = [tmpl.format(w=w) for w in b]
                # locate the word span via offsets
                enc = tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True)
                offs = enc.pop("offset_mapping")
                hs = model(**{k: v.to(dev) for k, v in enc.items()}).last_hidden_state.cpu().numpy()
                for j, w in enumerate(b):
                    s = texts[j].index(w); e = s + len(w)
                    idx = [t for t, (a, bb) in enumerate(offs[j].tolist()) if a < e and bb > s and bb > a]
                    if idx:
                        vecs[i + j] += hs[j, idx].mean(0)
    bert = vecs / len(templates)

    ref = MeaningReference(human, words=words, name="binder65")
    n = len(words); rng = np.random.default_rng(0); iu = np.triu_indices(n, 1)
    print(f"\nBinder {n} words x 65 features. Alignment (PCA-50):")
    cand = {"GloVe-300 (shallow)": gv, "MiniLM (deep)": mini, "BERT-proper (deep)": bert}
    P = {k: pca(v, 50) for k, v in cand.items()}
    for k in cand:
        print(f"  {k:22s} {alignment(P[k], ref):+.3f}")
    # bootstrap each deep vs shallow
    G = P["GloVe-300 (shallow)"]
    for dk in ["MiniLM (deep)", "BERT-proper (deep)"]:
        D = P[dk]; diffs = []
        for _ in range(2000):
            ix = rng.integers(0, n, n); hr = _rdm(human[ix])[iu]
            diffs.append(np.corrcoef(_rdm(D[ix])[iu], hr)[0, 1] - np.corrcoef(_rdm(G[ix])[iu], hr)[0, 1])
        diffs = np.array(diffs); ci = (round(float(np.percentile(diffs, 2.5)), 3), round(float(np.percentile(diffs, 97.5)), 3))
        p = float((diffs > 0).mean())
        print(f"  {dk} - GloVe: {alignment(D,ref)-alignment(G,ref):+.3f}, 95% CI {ci}, P(deep>shallow)={p:.3f}")


if __name__ == "__main__":
    main()
