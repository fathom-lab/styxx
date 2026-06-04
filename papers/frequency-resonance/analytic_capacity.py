# -*- coding: utf-8 -*-
"""
analytic_capacity.py — NO training, NO GPU. A linear-algebra capacity model for the phase code.

Each item is a phasor CODE VECTOR across the modes: item written `a` steps before readout has
code v(a) = [ r_m^a * exp(i*theta*a) ]_m. Holding K ordered items = those K code vectors being
linearly DISTINGUISHABLE; we score that by the log-determinant of their normalized Gram matrix
(higher = more orthogonal = more separable). This reproduces the resonance and *predicts* the
scaling sweep's verdict (does the optimum theta* move with the retention window W, or only with the
item count K?) from first principles, before the empirical run lands.

Modeling choices (honest): single shared theta across modes (matches the fixed-theta experiment);
r_m from the LRU init distribution U(0.9, 0.999); linear-readout idealization (the trained net has
a nonlinear MLP readout — this is the linear lower bound on what's separable).
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
np.random.seed(0)
M = 256
R = np.random.uniform(0.9, 0.999, M)          # |lambda| per mode, the LRU init distribution


K_POOL = 8          # items presented; capacity = how many are recoverable above noise
SIGMA_FRAC = 0.07   # readout noise as a fraction of the largest signal singular value


def capacity(theta, D, K=K_POOL, sigma_frac=SIGMA_FRAC):
    """Recoverable-item count for K items written consecutively, then held an extra delay D,
    read out via a linear map under noise. UN-normalized code vectors keep the decay/SNR that
    actually limits capacity; theta rotates the rows toward orthogonality. Score = expected number
    of recoverable signal dimensions = sum s_i^2 / (s_i^2 + sigma^2) over singular values s_i."""
    a = (K - 1 + D) - np.arange(K)            # age of each item at readout (item 0 oldest)
    V = (R[None, :] ** a[:, None]) * np.exp(1j * theta * a)[:, None]   # K x M, NOT normalized
    s = np.linalg.svd(V, compute_uv=False)    # signal strengths of the K code directions
    sigma = sigma_frac * s[0]
    return float(np.sum(s ** 2 / (s ** 2 + sigma ** 2)))   # soft count of dims above the noise floor


def theta_star(D, n=121):
    thetas = np.linspace(0.005, 0.995, n) * math.pi
    scores = np.array([capacity(t, D) for t in thetas])
    i = int(np.argmax(scores))
    return thetas[i] / math.pi, thetas, scores


out = {"M": M, "r_range": [0.9, 0.999], "K_pool": K_POOL, "sigma_frac": SIGMA_FRAC}

# --- (1) SANITY CHECK: does the model reproduce the empirical resonance at D=0? ---
f0, thetas, scores = theta_star(0)
sc = scores - scores.min()
rng = scores.max() - scores.min()
ipk = int(np.argmax(scores))
interior = 0 < ipk < len(scores) - 1
# "resonance reproduced" iff interior peak AND both ends clearly below the peak
reproduced = interior and (scores[0] < scores.max() - 0.15 * rng) and (scores[-1] < scores.max() - 0.15 * rng)
print(f"=== (1) SANITY — analytic resonance at D=0 (K_pool={K_POOL}) ===")
print(f"theta*={f0:.3f}pi | cap@0+={scores[0]:.2f} | cap@peak={scores.max():.2f} | cap@Nyquist={scores[-1]:.2f}")
print("  -> RESONANCE REPRODUCED (interior peak, both ends down)" if reproduced
      else "  -> NOT a clean interior resonance; predictions below are UNTRUSTWORTHY")
out["sanity_resonance"] = {"theta_star_frac": round(f0, 4), "cap_zero": round(float(scores[0]), 3),
                           "cap_peak": round(float(scores.max()), 3), "cap_nyquist": round(float(scores[-1]), 3),
                           "interior": bool(interior), "reproduced": bool(reproduced)}

# --- (2) THE FORK: does theta* slide with the DELAY (window) or stay put (item-count)? ---
print("\n=== (2) theta*(D) at fixed item pool — does the optimum slide with the hold? ===")
Ds = [0, 6, 12, 24]
tw, Ws = [], []
for D in Ds:
    f, _, _ = theta_star(D)
    W = K_POOL - 1 + D
    tw.append(f); Ws.append(W)
    print(f"  D={D:2d} (W={W:2d}): theta*={f:.3f}pi   theta*xW={f*W:.2f}pi")
prodW = [f * W for f, W in zip(tw, Ws)]
cvW = float(np.std(prodW) / np.mean(prodW)) if np.mean(prodW) else 9.9
spanW = (max(tw) - min(tw))
out["fork_window"] = {"delays": Ds, "windows": Ws, "theta_star_frac": [round(x, 4) for x in tw],
                      "product_thetaXW_pi": [round(p, 4) for p in prodW],
                      "product_cv": round(cvW, 4), "theta_star_span_frac": round(spanW, 4)}

# --- (3) item-count dependence: vary the pool K at fixed delay D=0 ---
print("\n=== (3) theta*(K) at fixed delay D=0 — does the optimum slide with item COUNT? ===")
Ks = [3, 4, 6, 8, 10]
tk = []
for K in Ks:
    thetas_k = np.linspace(0.005, 0.995, 121) * math.pi
    sc_k = np.array([capacity(t, 0, K=K) for t in thetas_k])
    tk.append(float(thetas_k[int(np.argmax(sc_k))] / math.pi))
    print(f"  K={K:2d}: theta*={tk[-1]:.3f}pi   theta*xK={tk[-1]*K:.2f}pi")
prodK = [f * K for f, K in zip(tk, Ks)]
cvK = float(np.std(prodK) / np.mean(prodK)) if np.mean(prodK) else 9.9
spanK = (max(tk) - min(tk))
out["fork_itemcount"] = {"Ks": Ks, "theta_star_frac": [round(x, 4) for x in tk],
                         "product_thetaXK_pi": [round(p, 4) for p in prodK],
                         "product_cv": round(cvK, 4), "theta_star_span_frac": round(spanK, 4)}

# --- verdict (only trustworthy if the sanity resonance reproduced) ---
print("\n=== ANALYTIC PREDICTION FOR THE RUNNING SCALING SWEEP ===")
if not reproduced:
    pred = ("UNTRUSTWORTHY — the linear model did not reproduce the empirical resonance, so its "
            "scaling prediction carries no weight. Honest negative for the analytic approach as built.")
else:
    window_bound = spanW > 0.10 and cvW < cvK
    itemcount_bound = spanK > 0.10 and cvK < cvW
    if window_bound and not itemcount_bound:
        pred = (f"SCALING predicted — analytic theta* slides with the WINDOW (span {spanW:.3f}pi, "
                f"theta*xW CV {cvW:.2f} < theta*xK CV {cvK:.2f}); expect theta*xW~const empirically.")
    elif itemcount_bound and not window_bound:
        pred = (f"NULL predicted — analytic theta* is ITEM-COUNT bound (theta*xK CV {cvK:.2f} stable, "
                f"delay span only {spanW:.3f}pi); expect theta* ~ flat in delay empirically.")
    else:
        pred = (f"MIXED — both budgets move theta* (window span {spanW:.3f}pi CV {cvW:.2f}; "
                f"itemcount span {spanK:.3f}pi CV {cvK:.2f}).")
out["prediction"] = {"trustworthy": bool(reproduced), "window_span_frac": round(spanW, 4),
                     "window_cv": round(cvW, 4), "itemcount_span_frac": round(spanK, 4),
                     "itemcount_cv": round(cvK, 4), "verdict": pred}
print(pred)

(HERE / "analytic_capacity_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("\nwrote analytic_capacity_result.json")
