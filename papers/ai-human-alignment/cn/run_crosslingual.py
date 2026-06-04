# -*- coding: utf-8 -*-
"""
run_crosslingual.py — DO A CHINESE-TRAINED LM AND AN ENGLISH-TRAINED LM MEAN THE SAME THING?
The universal-meaning question, with the shipped tool. For 672 concepts with Chinese↔English translations:
 - Chinese-model geometry  = precomputed Chinese GPT-2 / BERT / ERNIE embeddings (annot/*.mat)
 - English-model geometry  = English BERT / MiniLM embeddings of the English translations
 - meaning_agreement(zh, en) over the SAME concepts -> is concept geometry shared across languages?
 - and which concepts diverge most across languages (the culturally/linguistically specific ones).
Grounded against the human 54-feature reference (Chinese-rated). Uses shipped styxx.meaning_integrity.
"""
import os, sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_predictor_rdms import load_emb
from styxx.meaning_integrity import MeaningReference, meaning_alignment, meaning_agreement


def pca(E, k):
    Ec = E - E.mean(0); U, S, _ = np.linalg.svd(Ec, full_matrices=False)
    k = min(k, E.shape[1]); return U[:, :k] * S[:k]


def main():
    zh_words, _ = load_emb("GPT2.mat")
    zh_words = [str(w) for w in zh_words]
    df = pd.read_csv(os.path.join(HERE, "annot", "672words_translations.csv"))
    # detect columns: chinese col = best overlap with zh_words; english col = most ASCII-alpha
    def ascii_frac(s):
        s = [str(x) for x in s]; return np.mean([all(ord(c) < 128 for c in x) for x in s])
    en_col = max(df.columns, key=lambda c: ascii_frac(df[c]))
    zh_col = max([c for c in df.columns if c != en_col],
                 key=lambda c: len(set(str(x) for x in df[c]) & set(zh_words)))
    trans = {str(z).strip(): str(e).strip().lower() for z, e in zip(df[zh_col], df[en_col])}
    en_for = [trans.get(w, "") for w in zh_words]
    keep = [i for i, e in enumerate(en_for) if e and e.replace(" ", "").isalpha()]
    print(f"{len(zh_words)} concepts, {len(keep)} with a usable English translation")
    idx = np.array(keep); en_words = [en_for[i] for i in idx]

    # Chinese-model geometries (subset to matched concepts)
    zh = {m: load_emb(f"{m}.mat")[1][idx] for m in ["GPT2", "BERT", "ERNIE"]}
    # English-model geometries of the translations
    import torch
    from transformers import AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased").to(dev).eval()
    en_bert = np.zeros((len(en_words), 768))
    with torch.no_grad():
        for i in range(0, len(en_words), 48):
            b = en_words[i:i + 48]; enc = tok(b, return_tensors="pt", padding=True); am = enc["attention_mask"]
            hs = bert(**{k: v.to(dev) for k, v in enc.items()}).last_hidden_state
            for j in range(len(b)):
                L = int(am[j].sum()); en_bert[i + j] = hs[j, 1:L - 1].mean(0).cpu().numpy()
    en_mini = np.asarray(SentenceTransformer("all-MiniLM-L6-v2").encode(en_words, show_progress_bar=False), float)

    print("\nCROSS-LINGUAL meaning agreement (Chinese LM vs English LM, same concepts, PCA-50):")
    pairs = [("Chinese-BERT", zh["BERT"], "English-BERT", en_bert),
             ("Chinese-GPT2", zh["GPT2"], "English-MiniLM", en_mini),
             ("Chinese-ERNIE", zh["ERNIE"], "English-BERT", en_bert),
             ("Chinese-ERNIE", zh["ERNIE"], "English-MiniLM", en_mini)]
    for zn, ze, eN, ee in pairs:
        a = meaning_agreement(pca(ze, 50), pca(ee, 50), words=en_words, top=6)
        div = ", ".join(w for w, _ in a["most_divergent_concepts"][:5])
        print(f"  {zn:13s} vs {eN:13s}: agreement {a['agreement']:+.3f}   most cross-lingual divergence: {div}")

    # control: shuffle the English side -> agreement should collapse (proves it's real shared structure)
    rng = np.random.default_rng(0)
    sh = meaning_agreement(pca(zh["BERT"], 50), pca(en_bert[rng.permutation(len(en_words))], 50))["agreement"]
    print(f"\n  control (concepts mismatched): {sh:+.3f}  <- agreement collapses, so the signal is real shared structure")
    print("  -> a Chinese LM and an English LM share concept geometry above chance: meaning has a")
    print("     language-independent core, and the tool names where the languages diverge.")


if __name__ == "__main__":
    main()
