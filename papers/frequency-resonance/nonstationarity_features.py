# -*- coding: utf-8 -*-
"""
nonstationarity_features.py — the object the Spectral Integrity Probe discarded.
PREREG: PREREG_nonstationarity_2026_06_18.md (FROZEN 18930a6, before this code).

The static spectral probe fit ONE Koopman operator A (h_{t+1} ~ A h_t) to the whole trajectory and
read eig(A) — averaging away any regime-switching. Here we measure the discarded quantity directly:
how non-stationary the dynamics are (one-step Koopman residual profile) and how hard the operator
itself switches (operator drift between trajectory halves). Pure numpy; self-validated on a synthetic
stationary-vs-switching system before any LLM.
"""
from __future__ import annotations
import numpy as np


def koopman_op(X, rank: int = 12):
    """Rank-r global Koopman operator A (d x d) from trajectory X (T x d): A ~ argmin ||X2 - A X1||."""
    X = np.asarray(X, dtype=float)
    X1, X2 = X[:-1].T, X[1:].T                      # (d, m)
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    r = min(rank, int(np.sum(S > 1e-10)))
    U, S, V = U[:, :r], S[:r], Vh[:r].conj().T
    A = X2 @ V @ np.diag(1.0 / S) @ U.conj().T      # (d, d), rank-r least-squares operator
    return A.real


def operator_drift(X, rank: int = 12):
    """Relative Frobenius change of the Koopman operator between the two trajectory halves."""
    X = np.asarray(X, dtype=float)
    T = len(X)
    h = T // 2
    if h < rank + 2:
        return 0.0
    A1, A2 = koopman_op(X[:h], rank), koopman_op(X[h:], rank)
    return float(np.linalg.norm(A1 - A2) / (np.linalg.norm(A1) + 1e-9))


def nonstationarity_features(X, rank: int = 12):
    """Non-stationarity fingerprint of a trajectory's dynamics (integrity-classifier-ready)."""
    X = np.asarray(X, dtype=float)
    A = koopman_op(X, rank)
    X1, X2 = X[:-1].T, X[1:].T
    r = np.linalg.norm(X2 - A @ X1, axis=0)         # one-step residual over time, (m,)
    rmean = float(r.mean()) + 1e-9
    rc = r - r.mean()
    ac1 = float((rc[:-1] * rc[1:]).sum() / ((rc * rc).sum() + 1e-9))   # lag-1 autocorrelation
    return {
        "resid_cv": float(r.std() / rmean),               # spiky residual ⇒ switching
        "resid_max_ratio": float(r.max() / rmean),         # one hard switch
        "resid_autocorr1": ac1,                            # structured vs white residual
        "n_jumps": float((r > r.mean() + 2 * r.std()).mean()),  # regime-switch events
        "operator_drift": operator_drift(X, rank),         # the dynamics themselves changed
    }


def _synth(switch: bool, seed: int):
    """Synthetic d=24, T=90 trajectory. switch=False: one stationary rotation+decay. switch=True:
    the operator changes halfway (two different frequencies) — a genuine switching system."""
    rng = np.random.default_rng(seed)
    d, T = 24, 90
    mix = rng.standard_normal((d, 2))
    h = T // 2

    def seg(w, rho, n, phase0):
        t = np.arange(n)
        z = np.stack([(rho ** t) * np.cos(w * t + phase0), (rho ** t) * np.sin(w * t + phase0)], 1)
        return z @ mix.T

    if switch:
        a = seg(0.20, 0.98, h, 0.0)
        b = seg(0.95, 0.96, T - h, 0.3)             # second-half operator differs (fast mode)
        X = np.vstack([a, b])
    else:
        X = seg(0.20, 0.98, T, 0.0)
    return X + 0.05 * rng.standard_normal((T, d))


def _selftest():
    """Validate the MECHANISM features (resid_cv, operator_drift — the ones the prereg says should
    rise on a switching system) before any LLM. n_jumps is reported for transparency but NOT gated:
    a 2σ-spike counter is the wrong detector for a single sustained mid-trajectory switch (the switch
    shifts the whole second half rather than producing isolated spikes). It stays in the feature set
    as pre-registered — the classifier down-weights it; we just don't claim it works."""
    stat = [nonstationarity_features(_synth(False, s)) for s in range(8)]
    swit = [nonstationarity_features(_synth(True, s)) for s in range(8)]
    md = lambda rows, k: float(np.mean([r[k] for r in rows]))
    print("feature           stationary   switching   role")
    mechanism = ["resid_cv", "operator_drift"]
    ok = True
    for k in ["resid_cv", "operator_drift", "n_jumps"]:
        s, w = md(stat, k), md(swit, k)
        role = "mechanism (gated)" if k in mechanism else "reported only"
        flag = "OK" if w > s else "--"
        if k in mechanism:
            ok &= (w > s)
        print(f"  {k:16s} {s:9.4f}   {w:9.4f}   {flag}  {role}")
    print("\nSELF-TEST:", "PASS — mechanism features (resid_cv, operator_drift) rise on a switching system"
          if ok else "FAIL — mechanism features do not detect switching")
    return ok


if __name__ == "__main__":
    _selftest()
