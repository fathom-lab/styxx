"""styxx.transfer (prototype) — UNIVERSAL CROSS-MODEL DIRECTION TRANSFER.

Compute a steering / concept / integrity direction ONCE in model A, then install it in model B
through an UNSUPERVISED zero-anchor map (no paired data) learned from the two models' concept
geometries. The map is an orthogonal Procrustes rotation between PCA-reduced concept clouds,
recovered by Sinkhorn-annealed Wasserstein-Procrustes with a GW warm-start (structure-only inits;
NO identity-correspondence init — that would leak the answer).

This is the math core (pure numpy, no GPU). The experiment harness that extracts geometries and
measures injected steering lives in run_thought_transfer.py.

Validation built in: TransferMap.self_test() fits A->A-rotated with zero anchors and checks the
recovered map reproduces a held-out direction (the transfer positive control — if this fails, the
instrument is broken, do not trust any cross-model number).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_disjoint_worlds as R   # entropic_gw, distmat


def pca_basis(E, k):
    """Return (mean[d], Vk[d,k]) — centered PCA basis (top-k principal directions)."""
    mean = E.mean(0)
    Ec = E - mean
    _, _, Vt = np.linalg.svd(Ec, full_matrices=False)
    return mean, Vt[:k].T


def _sinkhorn(logP, iters=20):
    logP = logP - logP.max()
    for _ in range(iters):
        logP = logP - np.logaddexp.reduce(logP, axis=1, keepdims=True)
        logP = logP - np.logaddexp.reduce(logP, axis=0, keepdims=True)
    return np.exp(logP)


def _fit_Q(A, B, rng, restarts=20, iters=80):
    """Unsupervised orthogonal map Q (k x k) with A @ Q ~ B (rows = same concepts, hidden order).
    Returns (Q, matched_score). Structure-only inits incl GW warm-start; no identity init."""
    k = A.shape[1]
    inits = []
    try:
        Tgw, _ = R.entropic_gw(R.distmat(A), R.distmat(B))
        _, cgw = linear_sum_assignment(-Tgw)
        U, _, Vt = np.linalg.svd(A.T @ B[cgw]); inits.append(U @ Vt)
    except Exception:
        pass
    for _ in range(restarts):
        Qr, _ = np.linalg.qr(rng.standard_normal((k, k))); inits.append(Qr)
    best = (np.eye(k), -np.inf)
    n = A.shape[0]
    for Q0 in inits:
        Q = Q0; prev = None
        for it in range(iters):
            S = (A @ Q) @ B.T
            eps = max(0.03, 0.5 * (0.85 ** it))
            P = _sinkhorn(S / eps)
            U, _, Vt = np.linalg.svd(A.T @ (P @ B)); Q = U @ Vt
            _, col = linear_sum_assignment(-((A @ Q) @ B.T))
            if prev is not None and np.array_equal(col, prev):
                break
            prev = col
        S = (A @ Q) @ B.T
        _, col = linear_sum_assignment(-S)
        score = float(np.sum(S[np.arange(n), col]))
        if score > best[1]:
            best = (Q, score)
    return best


class TransferMap:
    """Zero-anchor linear map from model A's residual space to model B's, fit from concept clouds."""

    def __init__(self, meanA, VAk, Q, meanB, VBk, score, recovery=None):
        self.meanA, self.VAk, self.Q = meanA, VAk, Q
        self.meanB, self.VBk = meanB, VBk
        self.score, self.recovery = score, recovery

    @classmethod
    def fit(cls, EA, EB, k=60, seed=0, restarts=20, iters=80):
        """EA[n,dA], EB[n,dB]: SAME concept set, geometries only (no labels used to pair)."""
        k = min(k, EA.shape[1], EB.shape[1], EA.shape[0] - 1)
        meanA, VAk = pca_basis(EA, k)
        meanB, VBk = pca_basis(EB, k)
        A = (EA - meanA) @ VAk; A = A / (np.linalg.norm(A) + 1e-9)
        B = (EB - meanB) @ VBk; B = B / (np.linalg.norm(B) + 1e-9)
        Q, score = _fit_Q(A, B, np.random.default_rng(seed), restarts, iters)
        return cls(meanA, VAk, Q, meanB, VBk, score)

    def transfer_direction(self, vA):
        """Map a DIRECTION in A's full residual space to a unit direction in B's full space."""
        cA = vA @ self.VAk            # project (direction: no mean subtraction)
        cB = cA @ self.Q              # zero-anchor rotation
        vB = cB @ self.VBk.T          # lift to B's full space
        return vB / (np.linalg.norm(vB) + 1e-9)

    def transfer_point(self, xA):
        cA = (xA - self.meanA) @ self.VAk
        return (cA @ self.Q) @ self.VBk.T + self.meanB


def self_test(E, k=60, seed=0, rot_seed=1, n_dirs=20, test_dirs=None, in_subspace=True):
    """TRANSFER POSITIVE CONTROL: B = a known random rotation of A (zero-anchor), fit the map, check
    transferred directions align with the true rotated direction. FAIR version: test directions live
    IN the top-k concept subspace (in_subspace=True) — random full-dim directions are unfair because
    a random vector has only ~sqrt(k/d) of its norm in the k-subspace the map operates on. Pass
    test_dirs (e.g. real concept steering vectors) to test the actual transferred objects."""
    rng = np.random.default_rng(rot_seed)
    d = E.shape[1]
    Qtrue, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Erot = (E - E.mean(0)) @ Qtrue
    tm = TransferMap.fit(E, Erot, k=k, seed=seed)
    if test_dirs is None:
        if in_subspace:
            _, VAk = pca_basis(E, k)                       # top-k principal directions of the data
            coeff = rng.standard_normal((n_dirs, VAk.shape[1]))
            test_dirs = coeff @ VAk.T                      # random combos that LIVE in the subspace
        else:
            test_dirs = rng.standard_normal((n_dirs, d))
    cos = []
    for v in test_dirs:
        v = v / (np.linalg.norm(v) + 1e-9)
        true_vB = (v @ Qtrue); true_vB = true_vB / np.linalg.norm(true_vB)
        got = tm.transfer_direction(v)
        cos.append(float(abs(got @ true_vB)))   # abs: sign-free (PCA sign ambiguity)
    return float(np.mean(cos)), tm


if __name__ == "__main__":
    # quick numpy self-test on synthetic near-isometric data
    rng = np.random.default_rng(0)
    E = rng.standard_normal((160, 256))
    c, _ = self_test(E, k=60)
    print(f"transfer positive control (A -> A-rotated) mean |cos| = {c:.3f}  (expect ~1.0)")
