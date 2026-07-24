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
  - ESTIMATED -- point estimate, selection-aware interval, sync-rate estimate, misfit
    diagnostics, and a COVERAGE NOTE keyed to the regime the fit landed in, quoting rates
    MEASURED on the Stage-A characterization runs rather than nominal levels.

THE SELECTIVE ESTIMATOR (R11, fully sealed 9/9). The master-key parameter s is engaged only on
EVIDENCE: the profile solve yields cost(s=0) - cost(s_hat), s activates only when that
improvement exceeds tau, and THE BOOTSTRAP MIMICS THE SELECTION -- every resample re-selects
under the same tau, so the interval prices selection uncertainty. tau is calibrated PER DATASET
from the parametric-bootstrap null when `null_sims > 0` (the null draws yield null
improvements; tau is their 95th percentile); with `null_sims = 0` it falls back to the Stage-A
design-point value 14.239037302137804 (receipt: stage_a_operating_chars_v3_result.json).

INSTRUMENT-GRADE HONESTY (the datasheet travels with the estimate). Operating characteristics
measured at the Stage-A design point (J=4, n=6000, K=400 per stratum, alphas 0.10-0.20, betas
0.78-0.90; receipts in papers/anchored-validity/: stage_a_operating_chars_v3_result.json is the
sealed selective datasheet; _result.json / _v2_result.json are the pre-repair records;
r10_boundary_decomposition_receipt.json is the mechanism):

  - pi-interval coverage: 0.95 on clean validation panels; 0.963 under a rho=0.30 shared
    failure factor; 0.912 / 0.938 at wild master-key rates 0.05 / 0.15. All nine calibration
    gates sealed on the characterization run.
  - refusal false-fire rate on clean panels: 0/200 measured.
  - deaf-panel VOID rate: 1.000 under the noise-margin gate used here (0.967 plain).
  - clean phantom-activation rate: 0.020 (validation; nominal 0.05 by construction of tau);
    clean point error median 0.0074, ninetieth percentile 0.0212.
  - activation power against a real all-judge key: 0.30 / 0.71 / 1.00 at wild rates
    0.02 / 0.05 / 0.15 -- below roughly a five-percent key rate, absence of activation is NOT
    evidence of absence.
  - misfit flag: false-alarm calibrated per dataset; measured power is HIGH against gross
    structure (judge-subset keys, specialist judges) and LOW against smooth violations
    (y-correlated keys 0.06, anchor-beta optimism 0.18, ten-percent contamination 0.36) --
    smooth violations are silently wrong 0.60-0.82 of the time and MUST be excluded by anchor
    construction (graded-difficulty ladders, labeled slices, provenance controls), not policed
    statistically.

SCOPE. The sync parameter s prices ALL-JUDGE, TRUTH-INDEPENDENT master keys only; keys
correlated with the true label defeat it silently (measured). Anchor exchangeability is the
load-bearing assumption: alpha/beta gaps between anchors and organic items bias pi toward the
audited system's favour and are only partially detectable. This module never blesses -- it
estimates with stated characteristics, flags, or refuses.

STAGE-B FIELD RESULTS (real correlated weak-judge panel, four constructed task families,
receipts in papers/anchored-validity/): blatant gold anchors (verbatim pairs, direct
negations) produced label-free coverage 0/15 in EVERY family, with the misfit flag firing
anywhere from 15/15 to 1/15 of wrong replicates depending on how smoothly the violation bends
the moments -- build your anchors from the same generating process as the work items, or they
certify nothing. Ladder anchors answered correctly in BOTH directions: restoring calibrated
coverage 13/13 when exactly one informative judge existed, and refusing (VOID 14/15) when
honest anchors revealed no judge was informative at all. A frontier panel under the same
protocol was priced exactly, at ceiling, on every sheet tested.

Deterministic given `seed`. Pure numpy. Stage-A provenance: fathom-lab/styxx
branch paper/anchored-validity (harness `papers/anchored-validity/anchored_stage_a.py`).
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["audit_panel", "anchor_lr", "blindspot_power", "min_anchors_for_power"]

INFORMATIVENESS_GATE = 0.15
_S_GRID = np.arange(0.0, 0.6 + 1e-12, 0.002)
_DESIGN_POINT_TAU = 14.239037302137804   # R11 receipt; used only when null_sims == 0

_COVERAGE_NOTES = {
    "not_activated": ("s not activated (evidence below tau): selection-aware interval; "
                      "measured coverage 0.95 on clean validation panels at the Stage-A "
                      "design point"),
    "activated": ("s activated: selection-aware interval; measured coverage 0.912-0.963 "
                  "across characterized doses and correlation at the Stage-A design point"),
}


def _moment_system(Vr, negr, posr, idx):
    """First + pairwise + all-fire moments, every one linear in pi at fixed s."""
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
    allv = Vr[:, idx].prod(1)
    A.append(negr[:, idx].prod(1).mean()); B.append(posr[:, idx].prod(1).mean())
    t.append(allv.mean()); w.append(len(Vr) / max(allv.var() + 1e-9, 1e-9))
    return np.asarray(A), np.asarray(B), np.asarray(t), np.asarray(w)


def _profile(A, B, t, w, s_grid=_S_GRID):
    """pi(s) and cost(s) for the whole grid: m_k = s + (1-s)(A_k + pi*d_k), pi closed-form per
    s, UNCLIPPED (the refusal branch must see impossibility); s >= 0 by construction."""
    d = B - A
    one_s = 1.0 - s_grid
    ts = t[None, :] - s_grid[:, None] - one_s[:, None] * A[None, :]
    denom = np.maximum(one_s * (w * d * d).sum(), 1e-12)
    pi_s = (ts @ (w * d)) / denom
    m = s_grid[:, None] + one_s[:, None] * (A[None, :] + pi_s[:, None] * d[None, :])
    cost = ((m - t[None, :]) ** 2 * w[None, :]).sum(1)
    return pi_s, cost


def _select(pi_s, cost, tau):
    """Selective activation: engage s only when cost(s=0) - cost(s_hat) exceeds tau."""
    k = int(np.argmin(cost))
    improvement = float(cost[0] - cost[k])
    if improvement > tau and _S_GRID[k] > 0:
        return float(pi_s[k]), float(_S_GRID[k]), float(cost[k]), 2, improvement, True
    return float(pi_s[0]), 0.0, float(cost[0]), 1, improvement, False


def audit_panel(V, neg, pos, *, garbage=None, gate=INFORMATIVENESS_GATE, n_boot=300,
                null_sims=200, seed=0):
    """Audit a judge panel against known-label anchors. Returns a dict report; see module
    docstring for the verdict classes and the measured operating characteristics.

    Parameters: V (n x J) organic verdicts in {0,1}; neg / pos: known-negative / known-positive
    INERT anchor strata (k x J). `garbage`: optional DETECTOR stratum built to trip failure
    modes -- never pooled into rate estimation (stratified accounting; pooling it in inflates
    alpha and manufactures a favourable pi); it reports a fire-rate diagnostic only.
    `null_sims`: parametric-bootstrap draws powering BOTH the per-dataset activation threshold
    tau and the misfit p-value (0 disables; tau then falls back to the design-point value).
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
    A, B, t, w = _moment_system(V, neg, pos, idx)
    pi_s, cost = _profile(A, B, t, w)

    # per-dataset null: simulate the fitted INDEPENDENT no-sync model; each draw yields a null
    # improvement (calibrates tau) and, once tau is known, a null selected-model misfit
    # (calibrates the misfit p-value)
    null_records = []
    if null_sims > 0:
        pi_sim = float(np.clip(pi_s[0], 0.02, 0.98))
        a_sim = np.clip(a_hat[idx], 0.005, 0.995); b_sim = np.clip(b_hat[idx], 0.005, 0.995)
        ii = np.arange(len(idx))
        for _ in range(null_sims):
            y0 = rng.random(n) < pi_sim
            p = np.where(y0[:, None], b_sim[None, :], a_sim[None, :])
            V0 = (rng.random((n, len(idx))) < p).astype(int)
            N0 = (rng.random((len(neg), len(idx))) < a_sim[None, :]).astype(int)
            P0 = (rng.random((len(pos), len(idx))) < b_sim[None, :]).astype(int)
            An, Bn, tn, wn = _moment_system(V0, N0, P0, ii)
            pn, cn = _profile(An, Bn, tn, wn)
            kn = int(np.argmin(cn))
            null_records.append((float(cn[0] - cn[kn]), float(cn[0]), float(cn[kn]),
                                 bool(_S_GRID[kn] > 0), len(An)))
        tau = float(np.percentile([r[0] for r in null_records], 95))
        tau_source = "per_dataset_null"
    else:
        tau, tau_source = _DESIGN_POINT_TAU, "design_point_default"

    pi_raw, s_hat, c_sel, dof, improvement, activated = _select(pi_s, cost, tau)
    m_count = len(A)
    chi2 = c_sel / max(m_count - dof, 1)

    # the bootstrap MIMICS the selection: every resample re-selects under the same tau
    bp, bs = [], []
    for _ in range(n_boot):
        bi = rng.integers(0, n, n); bn = rng.integers(0, len(neg), len(neg))
        bz = rng.integers(0, len(pos), len(pos))
        Ab, Bb, tb, wb = _moment_system(V[bi], neg[bn], pos[bz], idx)
        ps, cs = _profile(Ab, Bb, tb, wb)
        pb, sb, _, _, _, _ = _select(ps, cs, tau)
        bp.append(pb); bs.append(sb)
    plo, phi = (float(x) for x in np.percentile(bp, [2.5, 97.5]))
    slo, shi = (float(x) for x in np.percentile(bs, [2.5, 97.5]))

    misfit = {"chi2_per_df": chi2, "tau": tau, "tau_source": tau_source}
    if null_records:
        null_misfits = [(ck / max(m - 2, 1)) if (imp > tau and spos) else (c0 / max(m - 1, 1))
                        for imp, c0, ck, spos, m in null_records]
        p_val = float((1 + sum(1 for x in null_misfits if x >= chi2)) / (len(null_misfits) + 1))
        misfit.update({"null_p": p_val, "null_sims": null_sims, "flag": bool(p_val < 0.05),
                       "flag_scope": ("calibrated false-alarm; measured power is high vs gross "
                                      "structure only -- smooth violations (y-correlated keys, "
                                      "beta optimism, moderate contamination) are mostly "
                                      "SILENT: exclude them by anchor construction")})

    # refusal: an admissible prevalence lives in [0,1]; outside it beyond bootstrap reach, the
    # anchors disagree with the data and no confident number is honest
    if (pi_raw < 0.0 and phi < 0.0) or (pi_raw > 1.0 and plo > 1.0):
        return {"verdict": "VOID_ANCHORS__nonexchangeable", "pi": None, "s": None,
                "pi_unclipped": pi_raw, "ci_unclipped": [plo, phi], "misfit": misfit,
                "note": "unclipped solution impossible; anchors and data irreconcilable", **base}

    regime = "activated" if activated else "not_activated"
    return {"verdict": "ESTIMATED",
            "pi": float(np.clip(pi_raw, 0, 1)),
            "ci": [float(np.clip(plo, 0, 1)), float(np.clip(phi, 0, 1))],
            "ci_source": "selective_bootstrap", "regime": regime,
            "coverage_note": _COVERAGE_NOTES[regime],
            "pi_unclipped": pi_raw,
            "s": s_hat, "s_ci": [slo, shi],
            "activated": activated, "improvement": improvement,
            "s_at_grid_edge": bool(s_hat >= _S_GRID[-1] - 1e-9),
            "misfit": misfit,
            "scope": ("s prices ALL-JUDGE TRUTH-INDEPENDENT keys only; anchor exchangeability "
                      "is load-bearing; see styxx.anchors module docstring for measured "
                      "operating characteristics"), **base}


# ---------------------------------------------------------------------------
# THE ANCHOR THRESHOLD -- design-time power for catching a shared blind spot.
#
# audit_panel prices a shared, truth-independent, all-judge failure ONLY when enough anchors are
# present; below that budget, absence of activation is not evidence of absence (measured: activation
# power 0.30/0.71/1.00 at wild key rates 0.02/0.05/0.15 at the Stage-A design point). This block
# answers the design-time question the datasheet raises: HOW MANY known-negative anchors do you need
# before absence-of-detection means something?
#
# The impossibility is a statement about the UNLABELED MARGINAL: a synchronized (all-judge,
# truth-independent) blind spot and a benign independent panel can induce identical vote-count
# distributions on unlabeled data, so every consensus estimator (Dawid-Skene onward) is blind to it.
# The class-conditional marginal is where the worlds diverge: a KNOWN-NEGATIVE on which every judge
# votes "positive" is, under independence, a ~f**J event; under a blind spot it is common. Counting
# unanimous-wrong known-negatives is therefore the probe, and its power is an exact binomial.
#
# Test convention: the STANDARD most-powerful one-sided test -- reject "benign" iff the count of
# unanimous-wrong anchors X >= c, where c is the SMALLEST count with P(X >= c | benign) <= alpha.
# (The 2026-07-23 exploratory receipt anchor_threshold_result.json used a conservative variant --
# reject iff X > c -- which controls type-I identically but understates power; its power table is a
# valid LOWER BOUND. This shipped instrument reports the standard, tight power, so its threshold is
# lower than that exploratory table -- e.g. ~15 known-negatives for 0.90 power, not ~20-30 -- and its
# single-anchor case is nonzero, consistent with the single-anchor likelihood ratio.)
# ---------------------------------------------------------------------------

def _binom_pmf(k, n, p):
    """P(X = k) for X ~ Binom(n, p), via log-gamma (stable for large n)."""
    if p <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p >= 1.0:
        return 1.0 if k == n else 0.0
    if k < 0 or k > n:
        return 0.0
    logp = (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))
    return math.exp(logp)


def _binom_sf(c, n, p):
    """P(X >= c) for X ~ Binom(n, p), summed on the lower tail 0..c-1 (c is small when p is)."""
    if c <= 0:
        return 1.0
    if c > n:
        return 0.0
    lower = 0.0
    for k in range(c):
        lower += _binom_pmf(k, n, p)
        if lower >= 1.0:
            return 0.0
    return max(0.0, 1.0 - lower)


def _critical_count(n, p0, alpha):
    """Smallest c >= 1 with P(X >= c | Binom(n, p0)) <= alpha, i.e. the rejection threshold of the
    most-powerful level-alpha one-sided test. Returns n + 1 if no rejection region clears alpha."""
    surv = 1.0                        # P(X >= 0) = 1
    for c in range(n + 1):
        if surv <= alpha:
            return c
        surv -= _binom_pmf(c, n, p0)  # now surv = P(X >= c + 1)
    return n + 1


def _blindspot_probs(J, fp_rate, trap_rate, p_alt, fp_rate_alt):
    """P(a known-negative is unanimously mis-voted 'positive') under benign vs blind-spot worlds.
    benign: J independent judges each false-positive at fp_rate -> fp_rate**J.
    blind spot: a fraction trap_rate of negatives are traps (all judges wrong); the rest behave
    benignly at fp_rate_alt (defaults to fp_rate). Pass p_alt to set the alt probability directly."""
    if not (0.0 < fp_rate < 1.0):
        raise ValueError("fp_rate must be in (0, 1)")
    if J < 1:
        raise ValueError("J (number of judges) must be >= 1")
    p0 = fp_rate ** J
    if p_alt is not None:
        p1 = float(p_alt)
    else:
        if trap_rate is None:
            raise ValueError("provide trap_rate (fraction of negatives that are shared traps) "
                             "or p_alt (the alternative unanimous-wrong probability) directly")
        if not (0.0 <= trap_rate <= 1.0):
            raise ValueError("trap_rate must be in [0, 1]")
        fa = fp_rate if fp_rate_alt is None else fp_rate_alt
        p1 = trap_rate + (1.0 - trap_rate) * (fa ** J)
    if not (0.0 < p1 <= 1.0):
        raise ValueError("alternative unanimous-wrong probability out of (0, 1]")
    return p0, p1


def anchor_lr(*, J, fp_rate, trap_rate=None, p_alt=None, fp_rate_alt=None):
    """Likelihood ratio of ONE known-negative that every judge mis-votes 'positive', blind-spot vs
    benign: p1 / p0. A single unanimous-wrong known-negative is a smoking gun exactly when this is
    large (e.g. J=3, fp_rate=0.10, trap_rate=0.15 -> ~150x). See module ANCHOR THRESHOLD block."""
    p0, p1 = _blindspot_probs(J, fp_rate, trap_rate, p_alt, fp_rate_alt)
    return float("inf") if p0 <= 0.0 else p1 / p0


def blindspot_power(K, *, J, fp_rate, trap_rate=None, p_alt=None, fp_rate_alt=None, alpha=0.05):
    """Power to DETECT a shared (all-judge, truth-independent) blind spot from K known-negative
    anchors, via the count of unanimous-wrong anchors, using the standard most-powerful one-sided
    binomial test at level `alpha`.

    Parameters (keyword-only): K anchors; J judges; fp_rate = per-judge false-positive rate on a
    known-negative under independence; and the alternative, EITHER as trap_rate (fraction of true
    negatives that are shared traps, with non-traps at fp_rate_alt or fp_rate) OR as p_alt (the
    alternative unanimous-wrong probability) directly.

    Returns a dict: p_null, p_alt, single_anchor_lr, reject_at (critical count c; None if no
    level-alpha rejection region exists at this K), alpha_actual (the achieved type-I rate, <=
    alpha), and power. Closed-form and deterministic."""
    p0, p1 = _blindspot_probs(J, fp_rate, trap_rate, p_alt, fp_rate_alt)
    K = int(K)
    if K < 1:
        raise ValueError("K (number of known-negative anchors) must be >= 1")
    c = _critical_count(K, p0, alpha)
    if c > K:
        reject_at, power, alpha_actual = None, 0.0, 0.0
    else:
        reject_at, power, alpha_actual = c, _binom_sf(c, K, p1), _binom_sf(c, K, p0)
    return {"K": K, "p_null": p0, "p_alt": p1,
            "single_anchor_lr": float("inf") if p0 <= 0.0 else p1 / p0,
            "reject_at": reject_at, "alpha": alpha, "alpha_actual": alpha_actual,
            "power": float(power)}


def min_anchors_for_power(target_power, *, J, fp_rate, trap_rate=None, p_alt=None,
                          fp_rate_alt=None, alpha=0.05, k_max=100000):
    """Smallest number of known-negative anchors K at which the level-alpha test first reaches
    `target_power` against the specified blind spot. (Power is not strictly monotone in K -- each
    time the critical count steps up it can dip -- so this returns the first crossing, the smallest
    sufficient budget.) Returns a dict with K and the full blindspot_power report at that K, or
    raises ValueError if k_max is reached without meeting the target."""
    if not (0.0 < target_power < 1.0):
        raise ValueError("target_power must be in (0, 1)")
    for K in range(1, int(k_max) + 1):
        r = blindspot_power(K, J=J, fp_rate=fp_rate, trap_rate=trap_rate, p_alt=p_alt,
                            fp_rate_alt=fp_rate_alt, alpha=alpha)
        if r["power"] >= target_power:
            return {"K": K, "target_power": target_power, **r}
    raise ValueError(f"target_power {target_power} not reached within k_max={k_max} anchors")
