# -*- coding: utf-8 -*-
"""
diag_minilm.py — disentangle the two confounds in the weak English result. Re-embed "deep" with a model
built for word/short-text meaning (all-MiniLM-L6-v2) on the SAME Lancaster reference + words. If MiniLM
beats GloVe, the deep<shallow was a BERT-isolated-word artifact. If MiniLM still <= GloVe, the
sensorimotor reference genuinely favors shallow (consistent with: deep's edge is abstract, not perceptual).
"""
import os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cn"))
from meaning_integrity import MeaningReference, alignment


def pca(E, k):
    Ec = E - E.mean(0); U, S, _ = np.linalg.svd(Ec, full_matrices=False)
    k = min(k, E.shape[1]); return U[:, :k] * S[:k]


def main():
    d = np.load(os.path.join(HERE, "en_data.npz"), allow_pickle=True)
    words = [str(w) for w in d["words"]]; human = d["human"]; glove = d["glove"]; bert = d["bert"]
    ref = MeaningReference(human, words=words, name="lancaster11")

    from sentence_transformers import SentenceTransformer
    print("embedding with all-MiniLM-L6-v2 (built for word/short-text meaning)...", flush=True)
    mini = SentenceTransformer("all-MiniLM-L6-v2").encode(words, show_progress_bar=False)

    print(f"\n{'model':12s} {'native':>7} {'PCA-50 align to Lancaster':>26}")
    for name, E in [("GloVe-300", glove), ("BERT-word", bert), ("MiniLM", np.asarray(mini, float))]:
        print(f"  {name:10s} {E.shape[1]:>5}d   {alignment(pca(E, 50), ref):>+24.3f}")
    gd = alignment(pca(glove, 50), ref); md = alignment(pca(np.asarray(mini, float), 50), ref)
    print(f"\n  MiniLM - GloVe (PCA-50): {md - gd:+.3f}")
    print(f"  -> {'deep(MiniLM)>shallow: BERT-word was the artifact' if md > gd + 0.02 else ('deep~shallow: reference too narrow to discriminate' if abs(md-gd)<=0.02 else 'shallow still wins: sensorimotor genuinely favors co-occurrence (deep edge is ABSTRACT not perceptual)')}")
    np.savez(os.path.join(HERE, "en_data.npz"), words=np.array(words), human=human, glove=glove, bert=bert, mini=np.asarray(mini, float))
    print("  (added MiniLM to en_data.npz)")


if __name__ == "__main__":
    main()
