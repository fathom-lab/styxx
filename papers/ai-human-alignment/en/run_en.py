# -*- coding: utf-8 -*-
"""
run_en.py — does the meaning-integrity monitor GENERALIZE to English + an INDEPENDENT human norm?
Reference: Lancaster 11 sensorimotor dims. Models: GloVe-300 (shallow) vs BERT-base (deep). Reuses the
exact same `meaning_integrity` core as the Chinese test — only the reference + models change.

PRE-REGISTERED PREDICTION (from the Chinese decomposition — deep wins on ABSTRACT, shallow suffices on
PERCEPTUAL): on this SENSORIMOTOR (perceptual-heavy) norm, deep's advantage should be SMALL/absent,
while the monitor's mechanical properties (invariance/sensitivity/localization) hold cleanly.
"""
import os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cn"))
from meaning_integrity import MeaningReference, alignment, per_concept_alignment


def pca(E, k):
    Ec = E - E.mean(0); U, S, _ = np.linalg.svd(Ec, full_matrices=False)
    k = min(k, E.shape[1]); return U[:, :k] * S[:k]


def auc(scores, labels):
    o = np.argsort(scores); r = np.empty(len(scores)); r[o] = np.arange(len(scores))
    pos = labels == 1; n1 = pos.sum(); n0 = (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def main():
    d = np.load(os.path.join(HERE, "en_data.npz"), allow_pickle=True)
    words = [str(w) for w in d["words"]]; human = d["human"]; glove = d["glove"]; bert = d["bert"]
    ref = MeaningReference(human, words=words, name="lancaster11")
    n = len(words); rng = np.random.default_rng(0)
    print(f"English test: {n} words, human {human.shape[1]}-dim (Lancaster sensorimotor), "
          f"GloVe {glove.shape[1]}d, BERT {bert.shape[1]}d")

    print("=" * 70)
    print("DEEP vs SHALLOW alignment to the independent English human norm:")
    for name, E in [("GloVe-300", glove), ("BERT-base", bert)]:
        print(f"  {name:10s} full {alignment(E, ref):+.3f}   PCA-50 {alignment(pca(E, 50), ref):+.3f}")
    dg = alignment(pca(glove, 50), ref); db = alignment(pca(bert, 50), ref)
    pred = "HELD (|diff|<0.05, muted as predicted)" if abs(db - dg) < 0.05 else \
           ("deep>shallow even on perceptual" if db > dg else "shallow>deep")
    print(f"  -> deep - shallow (PCA-50): {db - dg:+.3f}   PRE-REG PREDICTION: {pred}")

    print("=" * 70)
    print("MONITOR MECHANICS on English (these must generalize — they are content-agnostic):")
    E = bert; base = alignment(E, ref); dd = E.shape[1]
    Q, _ = np.linalg.qr(rng.standard_normal((dd, dd)))
    rot = alignment(E @ Q, ref); scl = alignment(E * 5.0, ref)
    inv = max(abs(rot - base), abs(scl - base)) < 1e-6
    print(f"  (2) INVARIANCE: base {base:+.4f}  rotated {rot:+.4f}  scaled {scl:+.4f}  -> {'PASS' if inv else 'FAIL'}")
    sd = E.std()
    sens = [alignment(E + s * sd * rng.standard_normal(E.shape), ref) for s in [0, 1, 2, 4]]

    def shuf(A, f):
        A2 = A.copy(); k = int(f * len(A)); idx = rng.choice(len(A), k, replace=False)
        A2[idx] = A2[rng.permutation(idx)]; return A2
    shv = [alignment(shuf(E, f), ref) for f in [0, 0.5, 1.0]]
    sens_ok = sens[-1] < base - 0.05 and shv[-1] < 0.10
    print(f"  (3) SENSITIVITY: noise {[round(x,3) for x in sens]}  shuffle {[round(x,3) for x in shv]}  -> {'PASS' if sens_ok else 'CHECK'}")
    k = int(0.3 * n); C = rng.choice(n, k, replace=False); lab = np.zeros(n); lab[C] = 1
    Ec = E.copy(); Ec[C] = E[rng.permutation(n)][:k]
    a = auc(-per_concept_alignment(Ec, ref), lab)
    print(f"  (5) LOCALIZATION: corrupt {k}/{n} -> ROC-AUC {a:.3f}  -> {'PASS' if a > 0.8 else 'WEAK'}")

    print("=" * 70)
    gen = inv and sens_ok and a > 0.8
    print(f"VERDICT: monitor mechanics generalize to English + independent norm: {'PASS' if gen else 'PARTIAL'}")
    import json
    json.dump({"n": n, "glove_pca50": round(dg, 4), "bert_pca50": round(db, 4),
               "deep_minus_shallow": round(db - dg, 4), "invariance_pass": bool(inv),
               "shuffle_to_zero": round(shv[-1], 4), "localization_auc": round(a, 4),
               "mechanics_generalize": bool(gen)},
              open(os.path.join(HERE, "en_result.json"), "w"), indent=2)
    print("wrote en_result.json")


if __name__ == "__main__":
    main()
