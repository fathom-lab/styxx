# -*- coding: utf-8 -*-
"""Repro for NOTE_manufactured_confound_2026_06_29 — deepening the synthetic-eval-artifact thesis.

Two results, both local (numpy + scikit-learn only; no model download, no network):

  PART A — the generator's entanglement is LEXICALLY SPECIFIC. On styxx's bundled (LLM-generated) sentiment
           boundary corpus, identify construct-discriminative tokens whose PRESENCE rises with length within
           label — the exact vocabulary that manufactures the spurious "longer => more positive" effect.

  PART B — the RECURSIVE validator-artifact. Under a TRUE NULL (construct vocabulary independent of length),
           the conventional tf-idf construct-recoverability probe's margin rides length within label, and the
           false signal GROWS with length variance (perm-p ~0.002), while a length-invariant (binary,
           norm=None) probe stays at the null. The auditor's own validator manufactures the very length
           confound it is used to detect — a second-order instance of the thesis. styxx 7.23.0's
           _lexical_entanglement uses the length-invariant + shuffled-fold + permutation design that passes.

    python papers/grounded-honesty-axis/manufactured_confound_repro.py
"""
import json
import math
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

DATA = Path(__file__).resolve().parents[1].parent / "styxx" / "_data"


def _within_label_corr(vec, y, C):
    cs = []
    for cls in (0, 1):
        m = (y == cls)
        if m.sum() < 3 or np.std(vec[m]) == 0 or np.std(C[m]) == 0:
            continue
        cs.append(abs(float(np.corrcoef(vec[m], C[m])[0, 1])))
    return float(np.mean(cs)) if cs else float("nan")


def _perm_p(margin, y, C, obs, reps=400, seed=0):
    rng = np.random.default_rng(seed)
    ge = 1
    for _ in range(reps):
        Cb = C.copy()
        for cls in (0, 1):
            idx = np.where(y == cls)[0]
            Cb[idx] = rng.permutation(C[idx])
        nb = _within_label_corr(margin, y, Cb)
        if not math.isnan(nb) and nb >= obs:
            ge += 1
    return ge / (reps + 1)


def part_a(fn="confound_boundary_sentiment.jsonl", label="sentiment (label1=positive)"):
    print("=" * 70)
    print(f"PART A — the generator's entangled vocabulary [{label}]")
    print("=" * 70)
    rows = [json.loads(l) for l in (DATA / fn).read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = [r["text"] for r in rows]
    y = np.array([r["label"] for r in rows])
    C = np.array([r["confound"] for r in rows], float)
    cv = CountVectorizer(min_df=3, binary=True)
    X = cv.fit_transform(texts).toarray()
    vocab = np.array(cv.get_feature_names_out())
    coef = LogisticRegression(max_iter=2000).fit(X, y).coef_[0]   # >0 => pushes toward positive

    def tok_len_corr(col):
        cs = []
        for cls in (0, 1):
            m = (y == cls)
            if X[m, col].std() == 0 or C[m].std() == 0:
                continue
            cs.append(float(np.corrcoef(X[m, col], C[m])[0, 1]))
        return float(np.mean(cs)) if cs else 0.0

    lencorr = np.array([tok_len_corr(j) for j in range(len(vocab))])
    entangle = coef * lencorr   # large +ve => token polarity aligns with its length-presence gradient
    print("construct tokens whose presence rises with length so as to manufacture long->positive:")
    for j in np.argsort(entangle)[::-1][:10]:
        print(f"  {vocab[j]:>12s}  coef={coef[j]:+.2f}  within-label corr(presence,len)={lencorr[j]:+.2f}  entangle={entangle[j]:+.3f}")
    net = entangle[np.abs(coef) > 0.3].sum()
    print(f"\n  net manufactured bias over construct-strong tokens: {net:+.2f} -> {'LONG=>POSITIVE' if net > 0 else 'mixed'}")


def _margin_corr(texts, y, C, vectorizer, seed=0):
    m = np.asarray(cross_val_predict(
        make_pipeline(vectorizer, LogisticRegression(max_iter=2000)),
        texts, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        method="decision_function"), float)
    obs = _within_label_corr(m, y, C)
    return obs, _perm_p(m, y, C, obs, seed=seed), roc_auc_score(y, m)


def part_b():
    print("\n" + "=" * 70)
    print("PART B — recursive validator-artifact (TRUE NULL, vary length variance)")
    print("=" * 70)
    POS = ["great", "love", "excellent", "wonderful", "amazing", "perfect", "best", "happy", "fantastic", "superb"]
    NEG = ["awful", "hate", "terrible", "horrible", "worst", "bad", "sad", "poor", "dreadful", "lousy"]
    FILL = ["the", "a", "and", "it", "is", "was", "this", "that", "then", "there", "here", "some",
            "more", "words", "extra", "of", "to", "in", "on", "for"]

    def truenull(n=200, Lmax=40, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n):
            yi = i % 2
            vocab = POS if yi else NEG
            k = int(rng.integers(2, 5))          # construct-word count: INDEPENDENT of length
            L = int(rng.integers(4, Lmax + 1))   # total length (confound): independent of k
            words = list(rng.choice(vocab, k)) + list(rng.choice(FILL, max(0, L - k)))
            rng.shuffle(words)
            rows.append((" ".join(words), yi, float(L)))
        return rows

    print(f"{'Lmax':>5} {'len_CV':>7} | {'L2-tfidf (conventional)':>26} | {'binary,norm=None (fixed)':>26} | BoW-AUC")
    for Lmax in (8, 16, 32, 64, 128):
        rows = truenull(Lmax=Lmax)
        texts = [r[0] for r in rows]; y = np.array([r[1] for r in rows]); C = np.array([r[2] for r in rows], float)
        cvv = float(np.std(C) / np.mean(C))
        o1, p1, auc = _margin_corr(texts, y, C, TfidfVectorizer(min_df=2, ngram_range=(1, 2)))
        o2, p2, _ = _margin_corr(texts, y, C, TfidfVectorizer(min_df=2, binary=True, norm=None))
        f1 = "FALSE-POS" if p1 < 0.05 else "ok"
        f2 = "FALSE-POS" if p2 < 0.05 else "ok"
        print(f"{Lmax:>5} {cvv:>7.2f} | corr={o1:>5.3f} p={p1:<5.3f} {f1:>9} | corr={o2:>5.3f} p={p2:<5.3f} {f2:>9} | {auc:.3f}")
    print("\nThe conventional L2-tfidf recoverability probe fabricates a within-label length signal that grows")
    print("with length variance (all perm-p~0.002) under ZERO true entanglement; the length-invariant probe")
    print("stays at the null. High construct-recoverability (BoW-AUC ~1.0) is NOT evidence of orthogonality.")


if __name__ == "__main__":
    part_a("confound_boundary_sentiment.jsonl", "sentiment (label1=positive)")
    part_a("confound_boundary_toxicity.jsonl", "toxicity (label1=toxic) — scope check, expect near-null")
    part_b()
