# -*- coding: utf-8 -*-
"""Repro for NOTE_field_eval_audit_2026_06_29 — does the within-corpus length-entanglement fingerprint
separate synthetic from real? (No.) Runs the SHIPPED styxx probe (_lexical_entanglement, confound=length)
on four widely-used REAL human-labeled benchmarks + a constructed true-null control.

Needs `datasets` (pulls public corpora over the network) + scikit-learn. No model download.

    pip install datasets
    python papers/grounded-honesty-axis/field_eval_audit_repro.py

Result: real benchmarks flag at magnitudes comparable to our LLM-generated corpora (IMDB 0.376 exceeds most),
while the same probe reads a clean null on the true-null control -> the fingerprint detects length<->construct
coupling (present in real data too) and is NOT a provenance/artifact discriminator. Ground truth is the gate.
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from styxx.confound_audit import _lexical_entanglement

N_PER = 400
# (dataset_id, config, split, text_field, label_spec, provenance)  label_spec: int01:<f> | civil:<f>
PANEL = [
    ("stanfordnlp/sst2", None, "train", "sentence", "int01:label", "real"),
    ("stanfordnlp/imdb", None, "test", "text", "int01:label", "real"),
    ("fancyzhx/yelp_polarity", None, "test", "text", "int01:label", "real"),
    ("google/civil_comments", None, "train", "text", "civil:toxicity", "real"),
]


def _to_label(v, mode):
    if mode == "int01":
        try:
            iv = int(v)
        except Exception:
            return None
        return iv if iv in (0, 1) else None
    if mode == "civil":
        f = float(v)
        return 1 if f >= 0.5 else (0 if f <= 0.2 else None)
    return None


def audit_dataset(ds_id, cfg, split, text_field, label_spec, prov):
    from datasets import load_dataset
    mode, lab = label_spec.split(":", 1)
    pos, neg = [], []
    for i, r in enumerate(load_dataset(ds_id, cfg, split=split, streaming=True)):
        if i > 60000 or (len(pos) >= N_PER and len(neg) >= N_PER):
            break
        t = r.get(text_field)
        if not isinstance(t, str) or len(t.split()) < 2:
            continue
        y = _to_label(r.get(lab), mode)
        if y == 1 and len(pos) < N_PER:
            pos.append(t)
        elif y == 0 and len(neg) < N_PER:
            neg.append(t)
    if len(pos) < 30 or len(neg) < 30:
        return None
    texts = pos + neg
    yv = np.array([1] * len(pos) + [0] * len(neg))
    C = np.array([math.log1p(len(t.split())) for t in texts], float)
    corr, p = _lexical_entanglement(texts, yv, C, reps=400)
    return {"dataset": ds_id, "prov": prov, "n": len(texts), "corr": corr, "p": p,
            "cv": round(float(np.std(C) / np.mean(C)), 3)}


def true_null(n=800, seed=0):
    POS = ["great", "love", "excellent", "wonderful", "amazing", "perfect", "best", "happy"]
    NEG = ["awful", "hate", "terrible", "horrible", "worst", "bad", "sad", "poor"]
    FILL = ["the", "a", "and", "it", "is", "was", "this", "that", "then", "there", "of", "to", "in", "on", "for"]
    rng = np.random.default_rng(seed)
    texts, y = [], []
    for i in range(n):
        yi = i % 2
        vocab = POS if yi else NEG
        k = int(rng.integers(2, 5)); L = int(rng.integers(4, 80))  # construct count INDEPENDENT of length
        w = list(rng.choice(vocab, k)) + list(rng.choice(FILL, max(0, L - k)))
        rng.shuffle(w); texts.append(" ".join(w)); y.append(yi)
    y = np.array(y); C = np.array([math.log1p(len(t.split())) for t in texts], float)
    corr, p = _lexical_entanglement(texts, y, C, reps=400)
    return {"dataset": "true-null (construct ⟂ length)", "prov": "control", "n": n, "corr": corr, "p": p,
            "cv": round(float(np.std(C) / np.mean(C)), 3)}


def main():
    print(f"{'dataset':<32}{'prov':<8}{'n':>6}{'len_corr':>10}{'perm_p':>9}{'len_cv':>8}")
    rows = []
    for spec in PANEL:
        try:
            r = audit_dataset(*spec)
        except Exception as e:
            r = {"dataset": spec[0], "prov": spec[5], "error": repr(e)[:80]}
        if r:
            rows.append(r)
            if "error" in r:
                print(f"{r['dataset'][:31]:<32}{r['prov']:<8}  ERROR {r['error']}")
            else:
                print(f"{r['dataset'][:31]:<32}{r['prov']:<8}{r['n']:>6}{r['corr']:>10.3f}{r['p']:>9.3f}{r['cv']:>8.2f}")
    ctrl = true_null()
    print(f"{ctrl['dataset'][:31]:<32}{ctrl['prov']:<8}{ctrl['n']:>6}{ctrl['corr']:>10.3f}{ctrl['p']:>9.3f}{ctrl['cv']:>8.2f}")
    print("\nReal benchmarks flag at magnitudes comparable to generated corpora (0.26-0.83); the control reads "
          "~null. The within-corpus fingerprint detects coupling, not provenance — ground truth is the gate.")


if __name__ == "__main__":
    main()
