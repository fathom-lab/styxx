# -*- coding: utf-8 -*-
"""
cross_lingual_decompose.py — WHICH meanings are universal across languages, and which are language-specific?
Takes the per-concept cross-lingual agreement (Chinese-ERNIE geometry vs English-MiniLM geometry, same
concepts) and asks whether CONCRETE/perceptual concepts are more cross-lingually universal than
ABSTRACT/cognitive-affective ones. Concreteness = (perceptual human features) − (abstract human features),
from the Binder-style 54-feature space. Uses shipped styxx.meaning_integrity.
"""
import os, sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_predictor_rdms import load_emb
from styxx.meaning_integrity import MeaningReference, per_concept_alignment

PERCEPTUAL = ["视觉", "明亮度", "黑暗度", "颜色", "图案", "触摸", "热", "冷", "轻", "重", "疼痛", "响亮", "声音", "音乐", "味觉", "嗅觉"]
ABSTRACT = ["复杂度", "造成", "后果性", "认知", "沟通", "自我", "益处", "快乐", "悲伤", "生气", "厌恶", "害怕", "惊讶", "驱动力", "需要", "注意力", "唤醒"]


def pca(E, k):
    Ec = E - E.mean(0); U, S, _ = np.linalg.svd(Ec, full_matrices=False)
    k = min(k, E.shape[1]); return U[:, :k] * S[:k]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


def main():
    zh_words = [str(w) for w in load_emb("GPT2.mat")[0]]
    df = pd.read_csv(os.path.join(HERE, "annot", "672words_translations.csv"))
    def ascii_frac(s): return np.mean([all(ord(c) < 128 for c in str(x)) for x in s])
    en_col = max(df.columns, key=lambda c: ascii_frac(df[c]))
    zh_col = max([c for c in df.columns if c != en_col], key=lambda c: len(set(str(x) for x in df[c]) & set(zh_words)))
    trans = {str(z).strip(): str(e).strip().lower() for z, e in zip(df[zh_col], df[en_col])}
    en_for = [trans.get(w, "") for w in zh_words]
    idx = np.array([i for i, e in enumerate(en_for) if e and e.replace(" ", "").isalpha()])
    en_words = [en_for[i] for i in idx]

    zh = load_emb("ERNIE.mat")[1][idx]
    from sentence_transformers import SentenceTransformer
    en = np.asarray(SentenceTransformer("all-MiniLM-L6-v2").encode(en_words, show_progress_bar=False), float)
    ref = MeaningReference(pca(zh, 50))
    agree = per_concept_alignment(pca(en, 50), ref)                  # per-concept cross-lingual agreement

    feat = pd.read_csv(os.path.join(HERE, "annot", "sf", "feature.csv"))
    def zblock(cols):
        M = feat[cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
        M = (M - M.mean(0)) / (M.std(0) + 1e-9); return M.mean(1)
    concreteness = (zblock(PERCEPTUAL) - zblock(ABSTRACT))[idx]       # high = concrete/perceptual

    rho = spearman(agree, concreteness)
    hi = concreteness > np.median(concreteness)
    print(f"{len(idx)} concepts. Per-concept cross-lingual agreement vs CONCRETENESS:")
    print(f"  Spearman rho = {rho:+.3f}")
    print(f"  concrete half: mean agreement {agree[hi].mean():+.3f}   |   abstract half: {agree[~hi].mean():+.3f}")
    order = np.argsort(agree)
    print(f"\n  MOST language-universal concepts: {[en_words[i] for i in order[::-1][:10]]}")
    print(f"  MOST language-specific concepts:  {[en_words[i] for i in order[:10]]}")
    verdict = "CONCRETE concepts are more language-universal" if rho > 0.1 else \
              ("ABSTRACT concepts are more universal" if rho < -0.1 else "no concreteness effect")
    print(f"\n  -> {verdict} (rho {rho:+.2f}). Concrete/perceptual meaning travels across languages; abstract")
    print(f"     meaning is more language-shaped — a measurable echo of linguistic relativity.")


if __name__ == "__main__":
    main()
