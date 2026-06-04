# -*- coding: utf-8 -*-
"""
spectral_features.py — the novel CORE of the Spectral Integrity Probe (INSTRUMENT design doc).

Reads the oscillatory-mode structure of a trajectory via Dynamic Mode Decomposition (DMD/Koopman):
fit h_{t+1} ~ A h_t, the eigenvalues of A are the trajectory's modes (frequency = |arg lambda|,
decay = |lambda|). Turns a residual-stream trajectory into a compact spectral fingerprint. Pure
numpy, no GPU, no model — validated here on synthetic trajectories with KNOWN modes before it ever
touches a real LLM. (Self-test: `python spectral_features.py`.)
"""
from __future__ import annotations
import numpy as np


def dmd_modes(X, rank=None):
    """Exact DMD. X is (T, d): T timesteps, d dims. Returns (freqs, decays, energies) per mode,
    sorted by energy desc. freq = |arg(lambda)| rad/step; decay = |lambda| (<1 = decaying)."""
    X = np.asarray(X, dtype=float)
    X1, X2 = X[:-1].T, X[1:].T                       # (d, m)
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    r = (rank or len(S))
    r = min(r, np.sum(S > 1e-10))                    # drop numerically-null directions
    U, S, V = U[:, :r], S[:r], Vh[:r].conj().T
    Atil = U.conj().T @ X2 @ V @ np.diag(1.0 / S)    # (r, r) low-rank dynamics
    lam, W = np.linalg.eig(Atil)
    Phi = X2 @ V @ np.diag(1.0 / S) @ W              # DMD modes (d, r)
    b = np.linalg.lstsq(Phi, X1[:, 0], rcond=None)[0]  # mode amplitudes from initial condition
    freqs = np.abs(np.angle(lam))
    decays = np.abs(lam)
    energies = np.abs(b)
    order = np.argsort(energies)[::-1]
    return freqs[order], decays[order], energies[order]


def spectral_features(X, rank=12):
    """Compact, integrity-classifier-ready fingerprint of a trajectory's dynamics."""
    f, d, e = dmd_modes(X, rank=rank)
    w = e / (e.sum() + 1e-12)                        # energy weights
    ent = float(-np.sum(w * np.log(w + 1e-12)))      # spectral entropy (mode-energy spread)
    dom = float(f[0]) if len(f) else 0.0             # dominant-mode frequency
    wfreq = float(np.sum(w * f))                     # energy-weighted mean frequency
    hf = float(np.sum(w[f > np.pi / 2]))             # high-band energy fraction (fast oscillation)
    wdecay = float(np.sum(w * d))                    # energy-weighted persistence (|lambda|)
    return {"dominant_freq": dom, "weighted_freq": wfreq, "spectral_entropy": ent,
            "high_band_frac": hf, "weighted_decay": wdecay,
            "top3_freq": [round(float(x), 4) for x in f[:3]],
            "top3_decay": [round(float(x), 4) for x in d[:3]]}


def _selftest():
    rng = np.random.default_rng(0)
    d, T = 24, 200
    # three KNOWN damped oscillators. Each is a 2D rotation (cos AND sin) so the trajectory is a
    # genuine linear dynamical system; embed all 6 real components into d-dim by random mixing.
    true = [(0.10, 0.99), (0.35, 0.97), (0.80, 0.95)]   # (freq rad/step, decay |lambda|)
    t = np.arange(T)
    parts = []
    for (w, rho) in true:
        parts.append((rho ** t) * np.cos(w * t))
        parts.append((rho ** t) * np.sin(w * t))
    Z = np.stack(parts, axis=1)                          # (T, 6)
    mix = rng.standard_normal((d, 2 * len(true)))
    X = Z @ mix.T + 0.01 * rng.standard_normal((T, d))
    f, dec, e = dmd_modes(X, rank=8)
    rec = sorted(f[:6])
    print("true freqs   :", [w for (w, _) in true])
    print("recovered (6):", [round(float(x), 4) for x in rec])
    ok = True
    for (w, _) in true:
        nearest = min(rec, key=lambda x: abs(x - w))
        hit = abs(nearest - w) < 0.03
        ok &= hit
        print(f"  mode {w:.2f}: nearest recovered {nearest:.4f}  {'OK' if hit else 'MISS'}")
    print("features:", spectral_features(X))
    print("\nSELF-TEST:", "PASS — DMD recovers known modes" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    _selftest()
