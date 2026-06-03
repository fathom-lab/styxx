# -*- coding: utf-8 -*-
"""
meaning_integrity_failures.py — does the monitor catch REALISTIC failure modes, not just random noise?
 (A) COLLAPSE    — loss of discrimination (concepts blurred toward the centroid)
 (B) FORGETTING  — capacity loss (a fraction of representation dimensions zeroed)
 (C) PLAUSIBLE-BUT-WRONG — the safety case. Swap each concept to a human-NEIGHBOR's vector: the top-1
     "output" stays in the right semantic area (looks sensible), while the relational meaning erodes.
     We track output-plausibility AND monitor-alignment together: is there a regime where the surface
     still looks fine but the monitor already sees the meaning breaking?
"""
import os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_predictor_rdms import load_emb
from meaning_integrity import MeaningReference, alignment, dispersion


def nn_model(E):                                          # nearest model-neighbor per concept (cosine, excl self)
    Z = E - E.mean(0); Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    S = Z @ Z.T; np.fill_diagonal(S, -9); return S.argmax(1)


def main():
    P = np.load(os.path.join(HERE, "predictor_rdms.npz"), allow_pickle=True)
    words = [str(w) for w in P["words"]]
    ref = MeaningReference.from_human_features(words=words)
    E = load_emb("GPT2.mat")[1]; base = alignment(E, ref); rng = np.random.default_rng(0)
    Hsim = 1 - ref.R                                      # human similarity (cos), in ~[-?,1]
    n = ref.n

    def plausibility(A):                                  # is each concept's top model-neighbor still human-related?
        nn = nn_model(A); return float(np.mean([Hsim[i, nn[i]] for i in range(n)]))

    print("=" * 72); print(f"baseline GPT2: alignment {base:+.3f}   output-plausibility {plausibility(E):+.3f}")

    print("=" * 72); print("(A) COLLAPSE — blur toward centroid (uniform loss of discrimination)")
    mu = E.mean(0); d0 = dispersion(E)
    for f in [0, 0.3, 0.6, 0.9, 0.99]:
        Ecol = (1 - f) * E + f * mu
        print(f"    f={f:<5} angular-align {alignment(Ecol, ref):+.3f}   dispersion-ratio {dispersion(Ecol) / d0:.3f}")
    print("    -> angular alignment is BLIND to uniform collapse (scale-invariant by design); the")
    print("       dispersion channel catches it (ratio falls ~1-f). The vital sign needs BOTH channels.")

    print("=" * 72); print("(B) FORGETTING — zero a fraction of representation dimensions")
    def forget(A, f):
        A2 = A.copy(); d = A.shape[1]; z = rng.choice(d, int(f * d), replace=False); A2[:, z] = 0; return A2
    print("    frac -> align: " + "  ".join(f"{f}:{alignment(forget(E, f), ref):+.3f}" for f in [0, 0.3, 0.6, 0.9]))

    print("=" * 72)
    print("(C) PLAUSIBLE-BUT-WRONG — swap each concept to its human-neighbor #j (output stays sensible)")
    print(f"    {'corruption':22s} {'output-plausibility':>20s} {'monitor-alignment':>18s}")
    order = np.argsort(ref.R, axis=1)                     # per concept: human neighbors by rank (col 0 = self)
    print(f"    {'intact':22s} {plausibility(E):>+20.3f} {base:>+18.3f}")
    for j in [1, 3, 10, 50]:
        Ec = E[order[:, j]]
        print(f"    {'swap@human-nbr#' + str(j):22s} {plausibility(Ec):>+20.3f} {alignment(Ec, ref):>+18.3f}")
    Er = E[rng.permutation(n)]
    print(f"    {'random-shuffle':22s} {plausibility(Er):>+20.3f} {alignment(Er, ref):>+18.3f}")
    print("    -> read the regime: where output-plausibility is still HIGH but monitor-alignment has")
    print("       already FALLEN, the meaning is breaking under a surface that still looks fine.")


if __name__ == "__main__":
    main()
