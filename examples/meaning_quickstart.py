# -*- coding: utf-8 -*-
"""
meaning_quickstart.py — did my model update keep its MEANING? (reference-free, ~40 lines)

    pip install styxx sentence-transformers
    python meaning_quickstart.py

Compares two versions of a model by their CONCEPT GEOMETRY — catches when an update / quantization /
fine-tune quietly broke the model's meaning, and names which concepts broke. No human reference, no labels.
(Any embeddings work; here we use all-MiniLM-L6-v2 so the demo is self-contained.)
"""
import numpy as np
from styxx import meaning_agreement

CONCEPTS = ["dog", "cat", "car", "boat", "apple", "banana", "king", "queen", "doctor", "teacher",
            "ocean", "mountain", "music", "anger", "joy", "money", "justice", "gravity", "summer", "winter",
            "book", "computer", "river", "fire", "child", "soldier", "hospital", "garden", "bridge", "clock",
            "honey", "silver", "wolf", "eagle", "desert", "forest", "poem", "engine", "mirror", "ladder"]


def main():
    from sentence_transformers import SentenceTransformer
    base = np.asarray(SentenceTransformer("all-MiniLM-L6-v2").encode(CONCEPTS), float)  # your model's concept geometry
    rng = np.random.default_rng(0)

    good = base + 0.02 * rng.standard_normal(base.shape)          # a GOOD update (e.g. a safe quantization)
    bad = base.copy()                                            # a BAD update that silently scrambled 6 concepts
    damaged = rng.choice(len(CONCEPTS), 6, replace=False)
    bad[damaged] = base[rng.permutation(len(CONCEPTS))][:6]

    print("did the update keep the model's MEANING?  (reference-free `meaning_agreement`)\n")
    g = meaning_agreement(good, base, words=CONCEPTS)
    b = meaning_agreement(bad, base, words=CONCEPTS)
    print(f"  good update : agreement {g['agreement']:.3f}   -> meaning preserved")
    print(f"  bad update  : agreement {b['agreement']:.3f}   -> meaning REGRESSED")
    print(f"  concepts the bad update broke : {[w for w, _ in b['most_divergent_concepts'][:6]]}")
    print(f"  (ground-truth corrupted set   : {sorted(CONCEPTS[i] for i in damaged)})")
    print("\n  -> point it at your model before/after any update. no labels, no human reference needed.")


if __name__ == "__main__":
    main()
