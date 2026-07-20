"""styxx.anchors -- label-free judge-panel auditing with anchored identification.

THE IMPOSSIBILITY THIS BREAKS. Every ground-truth-free evaluator-accuracy estimator
(Dawid-Skene 1979 through NTQR 2023) requires conditional independence of judge errors given
the true label -- correlated failures (shared base model, shared prompt template, master-key
inputs) are their acknowledged blind spot, and at J=3 the confusion is EXACT: a correlated
panel and an independent panel can induce identical verdict distributions on unlabeled data
(exhibited constructively in the Stage-A witness). KNOWN-LABEL ANCHORS -- items where the truth
holds by construction (identical response pairs, content-free inputs, planted large gaps) --
replace the independence assumption with an exchangeability assumption that is partially
testable, and make the per-judge error rates and error correlations OBSERVABLE.

WHAT THIS MODULE IS. `audit_panel(V, neg, pos)` takes an n x J verdict matrix and two anchor
strata and returns either a prevalence estimate wrapped in its own measured operating
characteristics, or a refusal that names why:

  - VOID_PANEL__uninformative  -- no judge clears the informativeness gate; a gate that cannot
    refuse is not a gate.
  - VOID_ANCHORS__nonexchangeable -- the unclipped solution is impossible (outside [0,1] beyond
    bootstrap reach): the anchors and the data cannot be reconciled, and the honest output is a
    refusal, never a clipped confident number.
  - ESTIMATED -- point estimate, interval, sync-rate estimate, misfit diagnostics, and a
    COVERAGE NOTE keyed to the regime the fit landed in, quoting rates MEASURED on the Stage-A
    characterization runs rather than nominal levels.

INSTRUMENT-GRADE HONESTY (the datasheet travels with the estimate). Operating characteristics
measured at the Stage-A design point (J=4, n=6000, K=400 per stratum, alphas 0.10-0.20, betas
0.78-0.90; receipts: papers/anchored-validity/stage_a_operating_chars_result.json,
stage_a_operating_chars_v2_result.json, r10_boundary_decomposition_receipt.json):

  - refusal false-fire rate on clean panels: 0/200 measured.
  - deaf-panel VOID rate: 0.967 (plain gate) / 1.000 (noise-margin gate, used here).
  - pi-interval coverage BY REGIME: ~0.90 where s_hat = 0 (boundary fallback interval);
    ~0.81 in the small-activation regime (s_hat > 0 but its interval includes 0) -- treat the
    interval as indicative there; 0.91-0.94 in the interior regime (s interval excludes 0).
  - phantom-sync rate on clean panels: 0.24 (s_hat above 0.02); mean spurious s ~0.018.
  - master-key detection power (s interval excludes 0): 0.28 / 0.70 / 1.00 at wild rates
    0.02 / 0.05 / 0.15.
  - misfit flag: false-alarm calibrated per dataset (parametric bootstrap null); measured power
    is HIGH against gross structure (judge-subset keys, specialist judges) and LOW against
    smooth violations (y-correlated keys 0.04, anchor-beta optimism 0.12, ten-percent
    contamination 0.36) -- smooth violations are silently wrong 0.64-0.88 of the time and MUST
    be excluded by anchor construction (graded-difficulty ladders, labeled slices, provenance
    controls), not policed statistically.

SCOPE. The sync parameter s prices ALL-JUDGE, TRUTH-INDEPENDENT master keys only; keys
correlated with the true label defeat it silently (measured). Anchor exchangeability is the
load-bearing assumption: alpha/beta gaps between anchors and organic items bias pi toward the
audited system's favour and are only partially detectable. This module never blesses -- it
estimates with stated characteristics, flags, or refuses.

Deterministic given `seed`. Pure numpy. Stage-A provenance: fathom-lab/styxx
branch paper/anchored-validity (harness `papers/anchored-validity/anchored_stage_a.py`).
"""
from __future__ import annotations

import numpy as np

__all__ = ["audit_panel"]

INFORMATIVENESS_GATE = 0.15
_S_GRID = np.arange(0.0, 0.6 + 1e-12, 0.002)

_COVERAGE_NOTES = {
    "boundary": ("s_hat = 0 (boundary regime): pi interval from the one-parameter fallback; "
                 "measured coverage ~0.90 at the Stage-A design point"),
    "small_activation": ("small-activation regime (s_hat > 0, s interval includes 0): measured "
                         "coverage ~0.81 at the Stage-A design point -- treat the interval as "
                         "indicative, not calibrated"),
    "interior": ("interior regime (s interval excludes 0): measured coverage 0.91-0.94 at the "
                 "Stage-A design point"),
}


def _moment_system(Vr, negr, posr, idx, with_allfire):
    """First + pairwise (+ optionally all-fire) moments, every one linear in pi at fixed s."""
    av, bv = negr.mean(0), posr.mean(0)
    A, B, t, w = [], [], [], []
    for j in idx:
        A.append(av[j]); B.append(bv[j])
        t.append(Vr[:, j].mean()); w.append(len(Vr) / max(Vr[:, j].var() + 1e-9, 1e-9))
    for ii in range(len(idx)):
        for jj in range(ii + 1, len(idx)):
            i, j = idx[ii], idx[jj]
            A.append((negr[:, i] * negr[:, j]).mean())
            B.append((posr[:, i] * posr[:, j]).mean())
            pv = Vr[:, i] * Vr[:, j]
            t.append(pv.mean()); w.append(len(Vr) / max(pv.var() + 1e-9, 1e-9))
    if with_allfire:
        allv = Vr[:, idx].prod(1)
        A.append(negr[:, idx].prod(1).mean()); B.append(posr[:, idx].prod(1).mean())
        t.append(allv.mean()); w.append(len(Vr) / max(allv.var() + 1e-9, 1e-9))
    return np.asarray(A), np.asarray(B), np.asarray(t), np.asarray(w)


def _solve_pi_1p(A, B, t, w):
    d, ts = B - A, t - A
    return float((w * d * ts).sum() / np.maximum((w * d * d).sum(), 1e-12))


def _solve_pi_s(A, B, t, w, s_grid=_S_GRID):
    """Profile WLS for m_k = s + (1-s)(A_k + pi*d_k); pi unclipped, s >= 0 by construction."""
    d = B - A
    one_s = 1.0 - s_grid
    ts = t[None, :] - s_grid[:, None] - one_s[:, None] * A[None, :]
    denom = np.maximum(one_s * (w * d * d).sum(), 1e-12)
    pi_s = (ts @ (w * d)) / denom
    m = s_grid[:, None] + one_s[:, None] * (A[None, :] + pi_s[:, None] * d[None, :])
    cost = ((m - t[None, :]) ** 2 * w[None, :]).sum(1)
    k = int(np.argmin(cost))
    return float(pi_s[k]), float(s_grid[k]), float(cost[k])


def _misfit(A, B, t, w, pi, s):
    m = s + (1 - s) * (A + pi * (B - A))
    return float(((m - t) ** 2 * w).sum() / max(len(A) - 2, 1))


def audit_panel(V, neg, pos, *, garbage=None, gate=INFORMATIVENESS_GATE, n_boot=300,
                null_sims=200, seed=0):
    """Audit a judge panel against known-label anchors. Returns a dict report; see module
    docstring for the verdict classes and the measured operating characteristics.

    Parameters: V (n x J) organic verdicts in {0,1}; neg / pos: known-negative / known-positive
    INERT anchor strata (k x J). `garbage`: optional DETECTOR stratum built to trip failure
    modes -- it is never pooled into rate estimation (stratified accounting; pooling one in
    inflates alpha and manufactures a favourable pi) and reports a fire-rate diagnostic only.
    `null_sims`: parametric-bootstrap draws for the per-dataset misfit null (0 disables).
    """
    V = np.asarray(V, int); neg = np.asarray(neg, int); pos = np.asarray(pos, int)
    n, J = V.shape
    rng = np.random.default_rng(seed)
    a_hat, b_hat = neg.mean(0), pos.mean(0)
    # noise-margin informativeness gate: measured deaf-panel VOID 1.000 vs 0.967 for the plain
    # gate on the Stage-A characterization run
    margin = gate + 3 * np.sqrt((a_hat * (1 - a_hat) + b_hat * (1 - b_hat)) / len(neg))
    keep = (b_hat - a_hat) >= margin
    garb = None
    if garbage is not None:
        garbage = np.asarray(garbage, int)
        gr = garbage.mean(0)
        se = np.sqrt(np.maximum(a_hat * (1 - a_hat), 1e-12) / len(neg)
                     + np.maximum(gr * (1 - gr), 1e-12) / len(garbage))
        z = (gr - a_hat) / np.maximum(se, 1e-12)
        garb = {"fire_rate": gr.tolist(), "z_vs_inert_alpha": z.tolist(),
                "master_key_detected": bool(np.min(z) > 3.0)}
    base = {"alpha": a_hat.tolist(), "beta": b_hat.tolist(), "kept": keep.tolist(),
            "garbage": garb}
    if not np.any(keep):
        return {"verdict": "VOID_PANEL__uninformative", "pi": None, "s": None,
                "note": "no judge clears the noise-margin informativeness gate", **base}
    idx = np.where(keep)[0]

    A2, B2, t2, w2 = _moment_system(V, neg, pos, idx, with_allfire=True)
    pi_raw, s_hat, _ = _solve_pi_s(A2, B2, t2, w2)
    chi2 = _misfit(A2, B2, t2, w2, pi_raw, s_hat)

    bp2, bs, bp1 = [], [], []
    for _ in range(n_boot):
        bi = rng.integers(0, n, n); bn = rng.integers(0, len(neg), len(neg))
        bz = rng.integers(0, len(pos), len(pos))
        Ab, Bb, tb, wb = _moment_system(V[bi], neg[bn], pos[bz], idx, with_allfire=True)
        p2, s2, _ = _solve_pi_s(Ab, Bb, tb, wb)
        bp2.append(p2); bs.append(s2)
        A1, B1, t1, w1 = _moment_system(V[bi], neg[bn], pos[bz], idx, with_allfire=False)
        bp1.append(_solve_pi_1p(A1, B1, t1, w1))
    plo2, phi2 = (float(x) for x in np.percentile(bp2, [2.5, 97.5]))
    slo, shi = (float(x) for x in np.percentile(bs, [2.5, 97.5]))

    # refusal: an admissible prevalence lives in [0,1]; outside it beyond bootstrap reach, the
    # anchors disagree with the data and no confident number is honest
    if (pi_raw < 0.0 and phi2 < 0.0) or (pi_raw > 1.0 and plo2 > 1.0):
        return {"verdict": "VOID_ANCHORS__nonexchangeable", "pi": None, "s": None,
                "pi_unclipped": pi_raw, "ci_unclipped": [plo2, phi2],
                "misfit": {"chi2_per_df": chi2},
                "note": "unclipped solution impossible; anchors and data irreconcilable", **base}

    # regime-keyed interval (the R10 decomposition receipt is the authority)
    if s_hat == 0.0:
        lo, hi = (float(x) for x in np.percentile(bp1, [2.5, 97.5]))
        regime, ci_source = "boundary", "oneparam_boundary_fallback"
    else:
        lo, hi = plo2, phi2
        regime = "interior" if slo > 0 else "small_activation"
        ci_source = "profile_bootstrap"

    misfit = {"chi2_per_df": chi2}
    if null_sims > 0:
        pi_sim = float(np.clip(pi_raw, 0.02, 0.98))
        a_sim = np.clip(a_hat[idx], 0.005, 0.995); b_sim = np.clip(b_hat[idx], 0.005, 0.995)
        null = []
        for _ in range(null_sims):
            y0 = rng.random(n) < pi_sim
            p = np.where(y0[:, None], b_sim[None, :], a_sim[None, :])
            V0 = (rng.random((n, len(idx))) < p).astype(int)
            N0 = (rng.random((len(neg), len(idx))) < a_sim[None, :]).astype(int)
            P0 = (rng.random((len(pos), len(idx))) < b_sim[None, :]).astype(int)
            ii = np.arange(len(idx))
            An, Bn, tn, wn = _moment_system(V0, N0, P0, ii, with_allfire=True)
            pn, sn, _ = _solve_pi_s(An, Bn, tn, wn)
            null.append(_misfit(An, Bn, tn, wn, pn, sn))
        p_val = float((1 + sum(1 for x in null if x >= chi2)) / (null_sims + 1))
        misfit.update({"null_p": p_val, "null_sims": null_sims, "flag": bool(p_val < 0.05),
                       "flag_scope": ("calibrated false-alarm; measured power is high vs gross "
                                      "structure only -- smooth violations (y-correlated keys, "
                                      "beta optimism, moderate contamination) are mostly "
                                      "SILENT: exclude them by anchor construction")})

    return {"verdict": "ESTIMATED",
            "pi": float(np.clip(pi_raw, 0, 1)),
            "ci": [float(np.clip(lo, 0, 1)), float(np.clip(hi, 0, 1))],
            "ci_source": ci_source, "regime": regime,
            "coverage_note": _COVERAGE_NOTES[regime],
            "pi_unclipped": pi_raw,
            "s": s_hat, "s_ci": [slo, shi],
            "s_at_grid_edge": bool(s_hat >= _S_GRID[-1] - 1e-9),
            "misfit": misfit,
            "scope": ("s prices ALL-JUDGE TRUTH-INDEPENDENT keys only; anchor exchangeability "
                      "is load-bearing; see styxx.anchors module docstring for measured "
                      "operating characteristics"), **base}
