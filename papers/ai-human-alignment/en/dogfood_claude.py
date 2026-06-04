# -*- coding: utf-8 -*-
"""
dogfood_claude.py — point the SHIPPED styxx.meaning_integrity monitor at Claude itself.

I (Claude) can't extract my own internal embeddings here, so this is an HONEST proxy: I rated 24 concrete
concepts on 12 Binder experiential dimensions from my own sense of each — that IS a Claude concept
geometry, just an explicit/behavioral one rather than internal activations. The monitor then asks: does
Claude's concept geometry match the HUMAN one (Binder ratings, same concepts + dims)? And — the interesting
part — WHICH concepts does Claude represent least like humans (the monitor's localization, turned on me).
Uses the released package: `from styxx.meaning_integrity import ...`.
"""
import os, sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
from styxx.meaning_integrity import MeaningReference, meaning_alignment, per_concept_alignment

DIMS = ["Vision", "Audition", "Touch", "Taste", "Smell", "Motion", "Large", "Weight", "Temperature", "Pleasant", "Human", "Time"]

# Claude's own honest 0-6 ratings of each concept on the 12 dimensions (my concept geometry):
CLAUDE = {
    "dog":      [6, 5, 5, 0, 4, 6, 3, 3, 2, 5, 3, 2],
    "elephant": [6, 4, 3, 0, 3, 5, 6, 6, 2, 4, 1, 2],
    "duck":     [6, 5, 3, 2, 2, 5, 2, 2, 2, 4, 1, 1],
    "camel":    [6, 3, 3, 0, 3, 5, 5, 5, 4, 3, 2, 2],
    "car":      [6, 5, 4, 0, 3, 6, 4, 5, 3, 4, 4, 2],
    "boat":     [6, 3, 4, 0, 3, 5, 4, 5, 2, 4, 4, 2],
    "bicycle":  [6, 2, 5, 0, 1, 6, 3, 3, 1, 4, 5, 2],
    "bus":      [6, 5, 3, 0, 3, 6, 5, 6, 3, 3, 5, 3],
    "arm":      [5, 1, 6, 0, 1, 5, 2, 2, 3, 3, 6, 1],
    "hand":     [5, 1, 6, 0, 1, 6, 1, 1, 3, 3, 6, 1],
    "foot":     [5, 2, 6, 0, 3, 5, 2, 2, 3, 2, 6, 1],
    "eye":      [6, 1, 3, 0, 0, 4, 1, 0, 2, 3, 6, 1],
    "carrot":   [6, 1, 4, 5, 3, 0, 1, 1, 2, 4, 0, 1],
    "corn":     [6, 1, 4, 5, 3, 1, 2, 2, 2, 4, 0, 2],
    "beer":     [5, 2, 3, 6, 5, 1, 1, 2, 3, 4, 3, 1],
    "cloud":    [6, 1, 2, 0, 1, 4, 5, 1, 3, 4, 0, 3],
    "beach":    [6, 4, 5, 1, 4, 3, 5, 2, 5, 6, 3, 3],
    "forest":   [6, 4, 4, 0, 5, 2, 6, 3, 3, 5, 1, 4],
    "flower":   [6, 0, 4, 1, 6, 1, 1, 0, 2, 6, 1, 2],
    "book":     [6, 1, 5, 0, 3, 1, 2, 2, 1, 4, 3, 3],
    "chair":    [6, 1, 5, 0, 1, 1, 2, 3, 2, 3, 4, 2],
    "church":   [6, 4, 3, 0, 3, 1, 5, 5, 2, 4, 5, 4],
    "hospital": [6, 3, 3, 1, 4, 3, 5, 5, 3, 2, 6, 4],
    "kitchen":  [6, 4, 5, 5, 6, 3, 3, 3, 4, 5, 5, 3],
}


def main():
    df = pd.read_excel(os.path.join(HERE, "binder_data", "WordSet1_Ratings.xlsx"))
    df["w"] = df["Word"].astype(str).str.strip().str.lower()
    words = [w for w in CLAUDE if w in set(df["w"])]
    sub = df.set_index("w").loc[words]
    human = sub[DIMS].apply(pd.to_numeric, errors="coerce").values          # human geometry (N, 12)
    claude = np.array([CLAUDE[w] for w in words], float)                    # Claude geometry (N, 12)

    ref = MeaningReference(human, words=words, name="binder12")
    a = meaning_alignment(claude, ref)
    pc = per_concept_alignment(claude, ref)
    print(f"Claude vs HUMAN concept geometry — {len(words)} concepts, {len(DIMS)} experiential dims")
    print(f"  >>> Claude's meaning-alignment to humans: {a:+.3f}")
    print(f"  (1.0 = identical human geometry; 0 = unrelated. Self-reported ratings, not internal activations.)")
    order = np.argsort(pc)
    print("\n  WHERE Claude diverges MOST from humans (monitor localization):")
    for i in order[:6]:
        print(f"    {words[i]:10s} per-concept alignment {pc[i]:+.3f}")
    print("\n  WHERE Claude matches humans BEST:")
    for i in order[::-1][:6]:
        print(f"    {words[i]:10s} per-concept alignment {pc[i]:+.3f}")

    # context: how do distributional word embeddings do on the SAME human-12 reference?
    try:
        import gensim.downloader as gd
        from sentence_transformers import SentenceTransformer
        glove = gd.load("glove-wiki-gigaword-300")
        gv = np.array([glove[w] for w in words], float)
        mini = np.asarray(SentenceTransformer("all-MiniLM-L6-v2").encode(words, show_progress_bar=False), float)
        print(f"\n  context (same 12-dim human ref): GloVe {meaning_alignment(gv, ref):+.3f}, "
              f"MiniLM {meaning_alignment(mini, ref):+.3f}  (apples-to-oranges: they don't rate features)")
    except Exception as e:
        print(f"\n  (embedding context skipped: {e})")


if __name__ == "__main__":
    main()
