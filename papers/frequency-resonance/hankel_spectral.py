# -*- coding: utf-8 -*-
"""
hankel_spectral.py — time-delay (Hankel/Takens) embedding for the Spectral Integrity Probe's
K1 re-test (PREREG_hankel_k1_2026_06_18.md).

The premise of the fix: single-step exact DMD fits h_{t+1} ~ A h_t directly on the observed
coordinates. On a short (~90-sample), noisy, partially-observed trajectory it is variance-limited
and cannot resolve oscillatory modes whose period is a non-trivial fraction of the window. The
standard remedy (Takens / Hankel-DMD / HAVOK) is to delay-embed first: stack q time-shifted copies
so a linear operator on the embedded state can capture modes the raw coordinates cannot.

This module adds ONLY the embedding. Feature computation is the UNCHANGED `spectral_features`
(same dmd_modes, same 5 features) applied to the embedded trajectory — so the LLM re-test changes
exactly one thing vs v2: the trajectory is delay-embedded before the identical spectral fingerprint.
"""
from __future__ import annotations
import numpy as np

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from spectral_features import dmd_modes, spectral_features  # noqa: E402


def delay_embed(X, q: int):
    """Hankel/Takens delay embedding. X is (T, d). Returns (T-q+1, q*d): row t is the
    concatenation [X[t], X[t+1], ..., X[t+q-1]]. q=1 returns X unchanged (the v2 baseline)."""
    X = np.asarray(X, dtype=float)
    if q <= 1:
        return X
    T = X.shape[0]
    if T <= q:
        return X  # too short to embed; degrade gracefully to raw
    return np.concatenate([X[i:T - q + 1 + i] for i in range(q)], axis=1)


def hankel_spectral_features(X, q: int = 12, rank: int = 12):
    """The K1 re-test extractor: delay-embed, then the IDENTICAL spectral fingerprint."""
    return spectral_features(delay_embed(X, q), rank=rank)


def _selftest():
    """Validate the method before it touches an LLM: on a SHORT noisy trajectory with known modes,
    does delay-embedding recover modes that raw single-step DMD misses?"""
    rng = np.random.default_rng(0)
    d, T = 24, 90                       # the real regime: ~90 tokens, 24 PCA dims
    true = [(0.12, 0.985), (0.55, 0.97), (1.20, 0.95)]   # (freq rad/step, decay) — incl. a fast mode
    t = np.arange(T)
    parts = []
    for (w, rho) in true:
        parts.append((rho ** t) * np.cos(w * t))
        parts.append((rho ** t) * np.sin(w * t))
    Z = np.stack(parts, axis=1)
    mix = rng.standard_normal((d, 2 * len(true)))
    X = Z @ mix.T + 0.05 * rng.standard_normal((T, d))   # 5% noise — short + noisy

    def recovery_error(freqs):
        rec = sorted(float(x) for x in freqs)
        errs = [min(abs(r - w) for r in rec) for (w, _) in true]
        return float(np.mean(errs))

    f_raw, _, _ = dmd_modes(X, rank=8)
    f_hank, _, _ = dmd_modes(delay_embed(X, q=12), rank=8)
    e_raw, e_hank = recovery_error(f_raw), recovery_error(f_hank)
    print("true freqs        :", [w for (w, _) in true])
    print("raw DMD     top-6 :", [round(float(x), 4) for x in sorted(f_raw)[:6]], f"  mean |err|={e_raw:.4f}")
    print("Hankel q=12 top-6 :", [round(float(x), 4) for x in sorted(f_hank)[:6]], f"  mean |err|={e_hank:.4f}")
    better = e_hank < e_raw
    print("\nSELF-TEST:", "PASS — delay-embedding recovers short-trajectory modes better than raw DMD"
          if better else "FAIL — no improvement (method does not help this regime)")
    return better


if __name__ == "__main__":
    _selftest()
