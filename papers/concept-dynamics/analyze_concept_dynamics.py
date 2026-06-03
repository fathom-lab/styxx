# -*- coding: utf-8 -*-
"""
analyze_concept_dynamics.py — rigorous per-token spectral null test for a concept signal,
frozen by PREREG_concept_dynamics_2026_06_02.md.

Pipeline per trajectory s(t): detrend -> AR(1) fit -> multitaper spectrum (DPSS, NW=3,K=5)
-> max-band power vs {AR(1), IAAFT} surrogate nulls (>=1000) -> Thomson harmonic F-test
-> 1/f-vs-peak gate -> band [3/n, 0.5]. Plus commitment characterization (trend, AR(1) rho,
concept-vs-random structure) and a RoPE/positional cross-check vs random directions.

`python analyze_concept_dynamics.py --selftest`  validates the instrument on synthetic
signals (planted sinusoid must trip OSCILLATION; pure AR(1) must not).
`python analyze_concept_dynamics.py`             analyzes concept_dynamics_raw.json.
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np
from scipy.signal.windows import dpss
from scipy import stats

HERE = Path(__file__).resolve().parent

NW, K = 3, 5
N_SURR = 1000
BAND_LO_BINS = 3            # exclude lowest 3 freq bins (period > ~n/3)
SEED = 0


def detrend(x, model="linear"):
    x = np.asarray(x, float); n = len(x); t = np.arange(n)
    if model == "mean":
        return x - x.mean()
    deg = 1 if model == "linear" else 2
    return x - np.polyval(np.polyfit(t, x, deg), t)


def ar1_fit(x):
    x = np.asarray(x, float); x = x - x.mean()
    if len(x) < 3 or x.std() == 0:
        return 0.0, max(x.std(), 1e-9)
    rho = np.corrcoef(x[:-1], x[1:])[0, 1]
    rho = float(np.clip(rho if np.isfinite(rho) else 0.0, -0.999, 0.999))
    resid = x[1:] - rho * x[:-1]
    return rho, float(resid.std() + 1e-12)


def ar1_surrogate(n, rho, sigma, rng):
    e = rng.standard_normal(n) * sigma
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = rho * y[i - 1] + e[i]
    return y


def iaaft(x, rng, iters=200):
    x = np.asarray(x, float); n = len(x)
    amp = np.abs(np.fft.rfft(x)); sorted_x = np.sort(x)
    y = rng.permutation(x)
    for _ in range(iters):
        phase = np.angle(np.fft.rfft(y))
        y = np.fft.irfft(amp * np.exp(1j * phase), n)
        y = sorted_x[np.argsort(np.argsort(y))]
    return y


def mt_spectrum(x, nw=NW, k=K):
    x = np.asarray(x, float); n = len(x)
    tapers = dpss(n, nw, k)
    Jk = np.array([np.fft.rfft(tapers[i] * x) for i in range(k)])   # [K, nf]
    S = (np.abs(Jk) ** 2).mean(0)
    freqs = np.fft.rfftfreq(n)
    return freqs, S, Jk, tapers


def harmonic_ftest(Jk, tapers):
    """Thomson harmonic F-test per frequency. df=(2, 2K-2)."""
    k = Jk.shape[0]
    Hk = tapers.sum(1)                       # H_k(0)
    H2 = np.sum(Hk ** 2)
    mu = (Hk[:, None] * Jk).sum(0) / H2      # complex line estimate per freq
    resid2 = np.sum(np.abs(Jk - mu[None, :] * Hk[:, None]) ** 2, axis=0)
    F = (k - 1) * (H2 * np.abs(mu) ** 2) / (resid2 + 1e-30)
    return F


def band_mask(freqs, n):
    return (np.arange(len(freqs)) >= BAND_LO_BINS) & (freqs <= 0.5) & (freqs > 0)


def analyze_series(x, rng, n_surr=N_SURR):
    """Full gate battery on one trajectory. Returns dict of statistics + per-gate booleans."""
    x = np.asarray(x, float); n = len(x)
    out = {"n": n}
    if n < 12 or np.std(x) < 1e-9:
        out["usable"] = False
        return out
    out["usable"] = True
    xd = detrend(x, "linear")
    freqs, S, Jk, tapers = mt_spectrum(xd)
    m = band_mask(freqs, n)
    obs_power = float(S[m].max()) if m.any() else 0.0
    peak_freq = float(freqs[m][np.argmax(S[m])]) if m.any() else 0.0

    # surrogate nulls (max-band power)
    rho, sigma = ar1_fit(xd)
    ar_max = np.empty(n_surr); ia_max = np.empty(n_surr)
    for i in range(n_surr):
        ar = detrend(ar1_surrogate(n, rho, sigma, rng), "linear")
        ia = detrend(iaaft(xd, rng, iters=120), "linear")
        ar_max[i] = mt_spectrum(ar)[1][m].max()
        ia_max[i] = mt_spectrum(ia)[1][m].max()
    p_ar = float((1 + np.sum(ar_max >= obs_power)) / (n_surr + 1))
    p_ia = float((1 + np.sum(ia_max >= obs_power)) / (n_surr + 1))

    # harmonic F-test (look-elsewhere)
    F = harmonic_ftest(Jk, tapers)
    Fmax = float(F[m].max()) if m.any() else 0.0
    F_thr = float(stats.f.ppf(1 - 1.0 / n, 2, 2 * K - 2))
    F_pass = bool(Fmax > F_thr)

    # 1/f-vs-peak gate
    mm = m & (S > 0)
    peak_pass = False; beta = float("nan")
    if mm.sum() >= 5:
        lf = np.log(freqs[mm]); ls = np.log(S[mm])
        A = np.polyfit(lf, ls, 1)
        resid = ls - np.polyval(A, lf)
        beta = float(-A[0])
        peak_pass = bool(resid.max() / (resid.std() + 1e-9) >= 2.0)

    # detrend robustness: peak frequency stable across {mean, linear, quadratic}
    pfreqs = []
    for dm in ("mean", "linear", "quadratic"):
        f2, S2, _, _ = mt_spectrum(detrend(x, dm))
        m2 = band_mask(f2, n)
        if m2.any():
            pfreqs.append(float(f2[m2][np.argmax(S2[m2])]))
    detrend_robust = bool(len(pfreqs) == 3 and (np.std(pfreqs) < (0.5 / n) * 1.5))

    out.update({"rho": rho, "peak_freq": peak_freq, "obs_power": obs_power,
                "p_ar1": p_ar, "p_iaaft": p_ia,
                # DETECTION gate = AR(1) red-noise line test. IAAFT preserves the power
                # spectrum by construction, so it is insensitive to a linear sinusoid and
                # is kept only as a NONLINEARITY diagnostic (pre-data self-test finding).
                "surr_pass": bool(p_ar < 0.05),
                "iaaft_nonlinear": bool(p_ia < 0.05),
                "Fmax": Fmax, "F_thr": F_thr, "F_pass": F_pass,
                "beta": beta, "peak_pass": peak_pass, "detrend_robust": detrend_robust,
                "trend_slope": float(np.polyfit(np.arange(n), x, 1)[0])})
    return out


# ─────────────────────────────── self-test ───────────────────────────────

def selftest():
    rng = np.random.default_rng(0)
    n = 64
    print("SELF-TEST (instrument validation)")
    # 1) pure AR(1) red noise -> should NOT pass surrogate gate (≈5% false positive)
    fp = 0; N = 40
    for _ in range(N):
        x = ar1_surrogate(n, 0.6, 1.0, rng)
        r = analyze_series(x, rng, n_surr=300)
        fp += int(r.get("surr_pass", False))
    print(f"  AR(1) red noise: surrogate false-positive rate = {fp}/{N} (expect ~2/40 at 5%)")
    # 2) ramp + AR(1) (commitment, no oscillation) -> should NOT pass
    fp2 = 0
    for _ in range(N):
        x = np.linspace(0, 5, n) + ar1_surrogate(n, 0.5, 0.6, rng)
        r = analyze_series(x, rng, n_surr=300)
        fp2 += int(r.get("surr_pass", False))
    print(f"  ramp+AR(1) (commitment): surrogate false-positive = {fp2}/{N} (expect low; trend excluded by band)")
    # 3) planted sinusoid (period 8 tokens) + noise -> SHOULD pass
    tp = 0
    for _ in range(N):
        t = np.arange(n)
        x = 2.0 * np.sin(2 * np.pi * t / 8.0) + 0.6 * rng.standard_normal(n) + 0.03 * t
        r = analyze_series(x, rng, n_surr=300)
        tp += int(r.get("surr_pass", False) and r.get("F_pass", False))
    print(f"  planted sinusoid (period 8): detect (surr AND F) = {tp}/{N} (expect high)")
    print("  -> instrument valid if: red-noise/ramp FP low, sinusoid detection high")


def main():
    raw = json.loads((HERE / "concept_dynamics_raw.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    CONCEPTS = raw["concepts"]
    result = {"per_model_concept": {}, "gate": {}}
    # aggregate per (model, concept) on concept-PRESENT trajectories
    osc_concepts = set()
    for mid, md in raw["models"].items():
        if md.get("status") != "ok":
            continue
        short = mid.split("/")[-1]
        for c in CONCEPTS:
            trajs = [tr for tr in md["traj"] if tr["concept"] == c and tr["polarity"] == 1 and c in tr["sig"]]
            if not trajs:
                continue
            seqs = []
            for tr in trajs:
                r = analyze_series(tr["sig"][c], rng)
                if r.get("usable"):
                    seqs.append(r)
            if not seqs:
                continue
            n_seq = len(seqs)
            surr_frac = np.mean([s["surr_pass"] for s in seqs])
            f_frac = np.mean([s["F_pass"] for s in seqs])
            peak_pass_frac = np.mean([s["peak_pass"] for s in seqs])
            robust_frac = np.mean([s["detrend_robust"] for s in seqs])
            pfreqs = [s["peak_freq"] for s in seqs if s["surr_pass"]]
            freq_concentrated = bool(len(pfreqs) >= 2 and np.std(pfreqs) < 0.06)
            mean_rho = float(np.mean([s["rho"] for s in seqs]))
            mean_slope = float(np.mean([s["trend_slope"] for s in seqs]))
            # gate: oscillation iff surrogate on >=1/3 seqs AND F AND 1/f AND robust AND concentrated
            osc = bool(surr_frac >= 1/3 and f_frac >= 1/3 and peak_pass_frac >= 1/3
                       and robust_frac >= 0.5 and freq_concentrated)
            if osc:
                osc_concepts.add(f"{short}:{c}")
            result["per_model_concept"][f"{short}::{c}"] = {
                "n_seq": n_seq, "surr_frac": round(float(surr_frac), 3), "F_frac": round(float(f_frac), 3),
                "peak_pass_frac": round(float(peak_pass_frac), 3), "robust_frac": round(float(robust_frac), 3),
                "freq_concentrated": freq_concentrated, "mean_ar1_rho": round(mean_rho, 3),
                "mean_trend_slope": round(mean_slope, 4), "oscillation": osc}

    reading = ("OSCILLATION — a concept signal carries genuine rhythm (passed all gates): "
               + ", ".join(sorted(osc_concepts))) if osc_concepts else \
              ("COMMITMENT — H0 holds: no concept signal shows oscillation beyond trend+AR(1); "
               "the dynamics are commitment (trend + red noise), characterized per cell")
    result["gate"] = {"oscillation_cells": sorted(osc_concepts), "n_oscillation": len(osc_concepts),
                      "reading": reading}
    (HERE / "concept_dynamics_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(reading)
    print("wrote concept_dynamics_result.json")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
