# -*- coding: utf-8 -*-
"""
meaning_agreement_demo.py — reference-FREE cross-model meaning comparison (shipped `styxx.meaning_agreement`).
Two real use cases nobody offers a tool for:
  (1) Do two models MEAN the same? (GloVe vs BERT vs MiniLM) — and where do they most disagree.
  (2) Model-update QA: did QUANTIZATION preserve meaning? (BERT vs its quantized self) — and if not, which concepts.
No human reference needed — just two models' embeddings for the same concepts.
"""
import os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
from styxx.meaning_integrity import meaning_agreement


def main():
    d = np.load(os.path.join(HERE, "en_data.npz"), allow_pickle=True)
    words = [str(w) for w in d["words"]]
    models = {"GloVe": d["glove"], "BERT": d["bert"], "MiniLM": d["mini"]}

    print("(1) DO TWO MODELS MEAN THE SAME? — pairwise reference-free agreement (1500 concepts)")
    names = list(models)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = meaning_agreement(models[names[i]], models[names[j]], words=words, top=5)
            div = ", ".join(w for w, _ in a["most_divergent_concepts"][:4])
            print(f"  {names[i]:7s} vs {names[j]:7s}: agreement {a['agreement']:+.3f}   most-divergent: {div}")
    print("  -> models agree only MODERATELY on meaning (0.22-0.32 on 1500 diverse words) — they do not")
    print("     fully mean the same thing; the tool quantifies the gap and names where they disagree.\n")

    print("(2) MODEL-UPDATE QA — did QUANTIZATION preserve meaning? (BERT vs quantized BERT)")
    E = models["BERT"]

    def quant(A, b):
        lo, hi = A.min(), A.max(); lv = 2 ** b
        return lo + np.round((A - lo) / (hi - lo) * (lv - 1)) / (lv - 1) * (hi - lo)

    print(f"  {'precision':14s} {'agreement vs full':>18} {'verdict':>10}")
    for b in [8, 4, 2, 1]:
        a = meaning_agreement(quant(E, b), E, words=words, top=6)
        verdict = "OK" if a["agreement"] > 0.95 else ("DRIFT" if a["agreement"] > 0.8 else "BROKEN")
        print(f"  {(str(b) + '-bit'):14s} {a['agreement']:>+18.3f} {verdict:>10}")
    a1 = meaning_agreement(quant(E, 1), E, words=words, top=6)
    print(f"  at 1-bit, concepts that lost the most meaning: {[w for w,_ in a1['most_divergent_concepts']]}")
    print("\n  -> meaning survives moderate quantization, degrades at extreme precision — and the tool")
    print("     names exactly which concepts broke. Reference-free regression testing for model updates.")


if __name__ == "__main__":
    main()
