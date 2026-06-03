# -*- coding: utf-8 -*-
"""
meaning_integrity_demo.py — rigorous validation that the meaning-integrity monitor is REAL:
 (1) RANKS models by human-meaning alignment            [positive control]
 (2) INVARIANT to meaning-preserving transforms          [rotation, scale, translation — provable, ~1e-6]
 (3) SENSITIVE to meaning-destroying corruption          [noise, quantization, concept-shuffle — monotone]
 (4) SEPARATES healthy from degraded                     [usable threshold]
 (5) LOCALIZES which concepts are corrupted              [per-concept diagnostics, ROC-AUC + precision]
On the validated human 54-feature reference, 672 concepts. This is what makes it a MEANING monitor and
not a fingerprint: it ignores how the representation is written, and catches when the meaning is wrong.
"""
import os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_predictor_rdms import load_emb
from meaning_integrity import MeaningReference, alignment, per_concept_alignment, integrity_report

ZOO = {"ERNIE": "ERNIE.mat", "GPT2": "GPT2.mat", "BERT": "BERT.mat", "Electra": "Electra.mat",
       "fastText": "fastText.mat", "GloVe": "GloVe.mat", "ResNet": "ResNet.mat", "ViT": "Vit.mat"}


def auc(scores, labels):                                  # labels: 1 = positive (corrupted)
    o = np.argsort(scores); r = np.empty(len(scores)); r[o] = np.arange(len(scores))
    pos = labels == 1; n1 = pos.sum(); n0 = (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def main():
    P = np.load(os.path.join(HERE, "predictor_rdms.npz"), allow_pickle=True)
    words = [str(w) for w in P["words"]]
    ref = MeaningReference.from_human_features(words=words)
    emb = {t: load_emb(fn)[1] for t, fn in ZOO.items()}
    rng = np.random.default_rng(0)

    print("=" * 70)
    print("(1) RANK models by human-meaning alignment  [positive control]")
    al = {t: alignment(emb[t], ref) for t in emb}
    for t in sorted(al, key=lambda x: -al[x]):
        print(f"    {t:9s} {al[t]:+.4f}")
    ok1 = al["ERNIE"] > al["GloVe"] > al["ResNet"]
    print(f"    -> deep top, GloVe mid, vision floor: {'as expected' if ok1 else 'UNEXPECTED'}")

    print("=" * 70)
    print("(2) INVARIANCE to meaning-preserving transforms (change should be ~0)")
    E = emb["GPT2"]; base = alignment(E, ref); d = E.shape[1]
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))      # random rotation
    rot = alignment(E @ Q, ref); scl = alignment(E * 7.3, ref); trn = alignment(E + 3.1, ref)
    print(f"    base {base:+.6f} | rotated {rot:+.6f} (d {abs(rot-base):.1e}) | "
          f"scaled x7.3 {scl:+.6f} (d {abs(scl-base):.1e}) | translated +3.1 {trn:+.6f} (d {abs(trn-base):.1e})")
    inv_ok = max(abs(rot - base), abs(scl - base), abs(trn - base)) < 1e-6
    print(f"    -> INVARIANT: {'PASS' if inv_ok else 'FAIL'}  (tracks meaning, not the surface basis)")

    print("=" * 70)
    print("(3) SENSITIVITY to meaning-destroying corruption (should drop monotonically)")
    sd = E.std()
    print("    add noise (sigma): " + "  ".join(
        f"{s}:{alignment(E + s*sd*rng.standard_normal(E.shape), ref):+.3f}" for s in [0, 0.5, 1, 2, 4]))

    def quant(A, b):
        lo, hi = A.min(), A.max(); lv = 2 ** b
        return lo + np.round((A - lo) / (hi - lo) * (lv - 1)) / (lv - 1) * (hi - lo)
    print("    quantize (bits):   " + "  ".join(f"{b}:{alignment(quant(E, b), ref):+.3f}" for b in [8, 4, 2, 1]))

    def shuf(A, f):
        A2 = A.copy(); k = int(f * len(A)); idx = rng.choice(len(A), k, replace=False)
        A2[idx] = A2[rng.permutation(idx)]; return A2
    sh = [(f, alignment(shuf(E, f), ref)) for f in [0, 0.2, 0.4, 0.6, 0.8, 1.0]]
    print("    shuffle (frac):    " + "  ".join(f"{f}:{v:+.3f}" for f, v in sh))
    mono = all(sh[i][1] >= sh[i + 1][1] - 0.02 for i in range(len(sh) - 1))
    print(f"    -> SENSITIVE & monotone: {'PASS' if mono and sh[-1][1] < 0.15 else 'CHECK'}")

    print("=" * 70)
    print("(4) SEPARATION: healthy band vs degraded band -> usable threshold")
    healthy = [al[t] for t in ["ERNIE", "GPT2", "BERT", "Electra", "fastText", "GloVe"]]
    degraded = ([alignment(shuf(E, f), ref) for f in [0.6, 0.8, 1.0]] +
                [alignment(E + s * sd * rng.standard_normal(E.shape), ref) for s in [2, 4]])
    margin = min(healthy) - max(degraded); thr = (min(healthy) + max(degraded)) / 2
    print(f"    healthy [{min(healthy):.3f}, {max(healthy):.3f}]   degraded [{min(degraded):.3f}, {max(degraded):.3f}]")
    print(f"    -> gap margin {margin:+.3f}, threshold {thr:.3f}: {'PASS (cleanly separable)' if margin > 0 else 'OVERLAP'}")

    print("=" * 70)
    print("(5) LOCALIZATION: detect WHICH concepts are corrupted (per-concept diagnostics)")
    n = len(E); k = int(0.3 * n); C = rng.choice(n, k, replace=False); lab = np.zeros(n); lab[C] = 1
    Ec = E.copy(); Ec[C] = E[rng.permutation(n)][:k]      # give the corrupted concepts wrong meanings
    pc = per_concept_alignment(Ec, ref); a_auc = auc(-pc, lab)
    prec = lab[np.argsort(pc)[:k]].mean()
    print(f"    corrupted {k}/{n} concepts; per-concept ROC-AUC {a_auc:.3f}, precision@{k} {prec:.3f}")
    print(f"    -> LOCALIZES corruption: {'PASS' if a_auc > 0.8 else 'WEAK'}")

    print("=" * 70)
    rep = integrity_report(shuf(E, 0.5), ref, words=words)
    print(f"SAMPLE REPORT (GPT2 @ 50% shuffled): alignment {rep['alignment']}  status {rep['status']}")
    print(f"    most-divergent concepts: {[w for _, w, _ in rep['worst_concepts'][:6]]}")
    print("=" * 70)
    verdicts = [ok1, inv_ok, mono and sh[-1][1] < 0.15, margin > 0, a_auc > 0.8]
    print(f"MONITOR VALIDATION: {sum(verdicts)}/5 properties hold "
          f"(rank={ok1}, invariant={inv_ok}, sensitive={mono and sh[-1][1]<0.15}, separable={margin>0}, localizes={a_auc>0.8})")


if __name__ == "__main__":
    main()
