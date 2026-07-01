"""Synthetic, known-answer tests for the depth-vs-truth statistics (analysis.py).

These tests DO NOT touch a model, network, or real data. Every case is
constructed so the correct statistical verdict is known a priori, then the
PREREG §2 machinery in analysis.py is asserted to recover it:

  H1 (signal, §2):
    - depth a PERFECT predictor of y  -> AUROC ~1.0, 95% CI excludes 0.5.
    - depth PURE NOISE (seeded)        -> 95% CI INCLUDES 0.5 (no false signal).
    - depth a perfect NEGATIVE predictor -> AUROC ~0.0, H1 does NOT fire
      (H1 is the one-sided > 0.5 claim).

  H2 (keystone additivity, §2):
    - depth adds REAL signal over SE  -> dAUC>0, CI excludes 0, LRT p<.01.
    - depth REDUNDANT with SE         -> dAUC~0, CI includes 0, LRT does not fire.
    - depth PURE NOISE over a real SE -> no false additivity (guards the
      in-sample-refit overfitting failure mode).

  H3 (OOD retention, §2):
    - fit on ID, FREEZE coefficients, score OOD: the reported dAUC_ood must
      equal a hand-fit ID model scored on OOD (proves NO refit on OOD), and a
      perturbation of the OOD *targets* must NOT change the fitted coefficients.

  Cross-checks: DeLong reproduces sklearn AUCs and agrees with an independent
  slow structural-component DeLong; Holm matches statsmodels-style ordering.

Run:  cd papers/depth-truth/harness && python -m pytest test_synthetic.py -q
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import analysis as A

# sklearn 1.8 emits a FutureWarning for penalty=None on every fit; the analysis
# does tens of thousands of fits. Silence it here so test output is readable.
# (The deprecation itself is flagged to the human in the audit, not masked away.)
pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


# --------------------------------------------------------------------------- #
# Fixtures / helpers                                                          #
# --------------------------------------------------------------------------- #
def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _logistic_y(rng, lin):
    """Draw binary y ~ Bernoulli(sigmoid(lin))."""
    p = 1.0 / (1.0 + np.exp(-lin))
    return (rng.uniform(size=lin.shape[0]) < p).astype(int)


# --------------------------------------------------------------------------- #
# H1 — signal                                                                 #
# --------------------------------------------------------------------------- #
def test_h1_perfect_predictor_auroc_one_ci_excludes_half():
    """depth strictly monotone in y => AUROC == 1.0, CI wholly above 0.5."""
    rng = _rng(1)
    n = 300
    y = rng.integers(0, 2, n)
    # depth = y plus tiny jitter that never crosses the class gap => perfect sep.
    depth = y.astype(float) + rng.uniform(0.0, 0.001, n)
    res = A.h1(depth, y)
    assert res["auroc"] == pytest.approx(1.0, abs=1e-9)
    assert res["ci_lo"] > 0.5
    assert res["excludes_half"] is True


def test_h1_pure_noise_ci_includes_half_no_false_signal():
    """depth independent of y => CI must straddle 0.5; H1 must NOT fire."""
    rng = _rng(20240607)
    n = 400
    y = rng.integers(0, 2, n)
    depth = rng.normal(0.0, 1.0, n)  # pure noise, seeded
    res = A.h1(depth, y)
    # The whole point: no false positive. CI brackets 0.5.
    assert res["ci_lo"] < 0.5 < res["ci_hi"], (res["ci_lo"], res["ci_hi"])
    assert res["excludes_half"] is False


def test_h1_is_one_sided_negative_predictor_does_not_fire():
    """A perfect NEGATIVE predictor (AUROC ~0) must not satisfy the >0.5 claim."""
    rng = _rng(7)
    n = 300
    y = rng.integers(0, 2, n)
    depth = (1 - y).astype(float) + rng.uniform(0.0, 0.001, n)  # anti-correlated
    res = A.h1(depth, y)
    assert res["auroc"] == pytest.approx(0.0, abs=1e-9)
    assert res["excludes_half"] is False  # CI is below 0.5, not above


def test_h1_deterministic_under_seed():
    """Same inputs => identical CI (pinned bootstrap seed 7)."""
    rng = _rng(3)
    n = 200
    y = rng.integers(0, 2, n)
    depth = 0.7 * y + rng.normal(0, 1, n)
    a = A.h1(depth, y)
    b = A.h1(depth, y)
    assert a == b


# --------------------------------------------------------------------------- #
# H2 — additivity (keystone)                                                  #
# --------------------------------------------------------------------------- #
def test_h2_real_added_signal_fires_all_three_gates():
    """depth carries signal ORTHOGONAL to SE => dAUC>0, CI>0, LRT p<.01."""
    rng = _rng(42)
    n = 800
    SE = rng.normal(0, 1, n)
    depth = rng.normal(0, 1, n)  # independent of SE
    y = _logistic_y(rng, 1.2 * SE + 1.2 * depth)  # both contribute
    res = A.h2_additivity(SE, depth, y)
    assert res["dAUC"] > 0
    assert res["ci_lo"] > 0
    assert res["excludes_zero"] is True
    assert res["lrt_p"] < 0.01
    assert res["lrt_pass"] is True
    # DeLong sensitivity must agree the two scorers differ.
    assert res["delong_p"] < 0.05


def test_h2_redundant_depth_no_false_additivity():
    """depth == SE (perfectly redundant) => dAUC~0, CI includes 0, LRT null."""
    rng = _rng(11)
    n = 800
    SE = rng.normal(0, 1, n)
    depth = SE.copy()  # zero independent information
    y = _logistic_y(rng, 1.5 * SE)
    res = A.h2_additivity(SE, depth, y)
    assert res["dAUC"] == pytest.approx(0.0, abs=1e-6)
    assert res["ci_lo"] <= 0.0  # CI does not exclude 0
    assert res["excludes_zero"] is False
    assert res["lrt_p"] > 0.01
    assert res["lrt_pass"] is False


def test_h2_pure_noise_depth_no_false_additivity():
    """The overfitting guard: depth is PURE NOISE added to a real SE model.

    In-sample refit gives 2 params a tiny optimistic edge, but the paired
    bootstrap CI and the LRT must both refuse to call additivity.
    """
    rng = _rng(105)
    n = 800
    SE = rng.normal(0, 1, n)
    depth = rng.normal(0, 1, n)  # independent of BOTH SE and y
    y = _logistic_y(rng, 1.4 * SE)  # y depends on SE only
    res = A.h2_additivity(SE, depth, y)
    assert res["excludes_zero"] is False, res
    assert res["lrt_pass"] is False, res


def test_h2_paired_bootstrap_uses_same_indices_for_both_models():
    """Refuting an UNPAIRED bootstrap: the resampled dAUC is computed from ONE
    index draw feeding BOTH models. We reproduce the point dAUC by hand on a
    single shared resample and confirm the helper does the paired thing."""
    rng = _rng(5)
    n = 300
    SE = rng.normal(0, 1, n)
    depth = rng.normal(0, 1, n)
    y = _logistic_y(rng, 1.0 * SE + 1.0 * depth)

    # One shared bootstrap draw.
    draw = _rng(99).integers(0, n, size=n)
    # Guard: keep both classes present so AUC is defined.
    if np.unique(y[draw]).size == 2:
        paired = A._paired_delta_auc(SE[draw], depth[draw], A._as_1d_binary(y[draw]))
        # Reconstruct: BOTH models refit on the SAME resampled rows.
        Xc = SE[draw].reshape(-1, 1)
        Xcd = np.column_stack([SE[draw], depth[draw]])
        mc = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(Xc, y[draw])
        mcd = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(Xcd, y[draw])
        exp = roc_auc_score(y[draw], mcd.decision_function(Xcd)) - roc_auc_score(
            y[draw], mc.decision_function(Xc)
        )
        assert paired == pytest.approx(exp, abs=1e-9)


def test_h2_lrt_df_is_one():
    """The LRT for adding ONE predictor (depth) must be chi-square df=1.

    We verify by recomputing the p-value from the deviance drop with df=1 and
    confirming a df=2 read would give a different (wrong) answer.
    """
    from scipy import stats as _st

    # Use a WEAK added effect so the deviance drop is modest and the chi2 tail
    # under df=1 vs df=2 is numerically well-separated (a huge effect drives both
    # tails to ~0 and would hide the df distinction under floating-point).
    rng = _rng(8)
    n = 600
    SE = rng.normal(0, 1, n)
    depth = rng.normal(0, 1, n)
    y = _logistic_y(rng, 1.0 * SE + 0.18 * depth)  # small depth contribution

    p_reported = A._lrt_p(SE, depth, y)

    Xc = SE.reshape(-1, 1)
    Xcd = np.column_stack([SE, depth])
    mc = A._fit_logit(Xc, y)
    mcd = A._fit_logit(Xcd, y)
    lr = A._deviance(mc, Xc, y) - A._deviance(mcd, Xcd, y)
    p_df1 = float(_st.chi2.sf(lr, df=1))
    p_df2 = float(_st.chi2.sf(lr, df=2))

    # analysis.py must use df=1 exactly.
    assert p_reported == pytest.approx(p_df1, rel=1e-9)
    # df matters: for a modest deviance drop the df=2 tail is materially larger,
    # so a wrong df would give a distinctly different p-value.
    assert p_df2 > p_df1
    assert abs(p_reported - p_df2) > 1e-6
    assert not np.isclose(p_reported, p_df2, rtol=1e-3, atol=1e-12)


def test_h2_secondary_holm_correction_applied():
    """LP_mean / LP_norm secondaries are Holm-corrected across the family."""
    rng = _rng(31)
    n = 500
    SE = rng.normal(0, 1, n)
    LP_mean = rng.normal(0, 1, n)
    LP_norm = rng.normal(0, 1, n)
    depth = rng.normal(0, 1, n)
    y = _logistic_y(rng, 1.0 * SE + 1.0 * depth)

    full = A.h2_full(SE, depth, y, LP_mean=LP_mean, LP_norm=LP_norm)
    assert set(full["secondary"].keys()) == {"LP_mean", "LP_norm"}
    holm_map = full["secondary_holm_lrt_p"]
    assert set(holm_map.keys()) == {"LP_mean", "LP_norm"}

    raw = [full["secondary"]["LP_mean"]["lrt_p"], full["secondary"]["LP_norm"]["lrt_p"]]
    expected = A.holm(raw)  # order = [LP_mean, LP_norm]
    assert holm_map["LP_mean"] == pytest.approx(expected[0], nan_ok=True)
    assert holm_map["LP_norm"] == pytest.approx(expected[1], nan_ok=True)
    # Holm-adjusted p is never smaller than raw.
    for name, i in (("LP_mean", 0), ("LP_norm", 1)):
        if np.isfinite(raw[i]):
            assert holm_map[name] >= raw[i] - 1e-12


# --------------------------------------------------------------------------- #
# H3 — OOD retention with FROZEN coefficients                                 #
# --------------------------------------------------------------------------- #
def test_h3_scores_ood_with_frozen_id_coefficients_no_refit():
    """The reported dAUC_ood must equal an INDEPENDENTLY hand-fit ID model
    scored on OOD. If h3 secretly refit on OOD, these would diverge."""
    rng = _rng(2024)
    n_id, n_ood = 600, 400
    SE_id = rng.normal(0, 1, n_id)
    depth_id = rng.normal(0, 1, n_id)
    y_id = _logistic_y(rng, 1.0 * SE_id + 1.3 * depth_id)
    SE_o = rng.normal(0, 1, n_ood)
    depth_o = rng.normal(0, 1, n_ood)
    y_o = _logistic_y(rng, 1.0 * SE_o + 1.3 * depth_o)

    res = A.h3_ood(SE_id, depth_id, y_id, SE_o, depth_o, y_o)

    # Hand-fit on ID ONLY, score OOD.
    mc = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(
        SE_id.reshape(-1, 1), y_id
    )
    mcd = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(
        np.column_stack([SE_id, depth_id]), y_id
    )
    sc = mc.decision_function(SE_o.reshape(-1, 1))
    scd = mcd.decision_function(np.column_stack([SE_o, depth_o]))
    manual = roc_auc_score(y_o, scd) - roc_auc_score(y_o, sc)

    assert res["dAUC_ood"] == pytest.approx(manual, abs=1e-9)
    assert res["dAUC_ood"] > 0  # real added signal retained OOD
    assert res["excludes_zero"] is True


def test_h3_ood_targets_do_not_change_fitted_model():
    """FROZEN proof: permuting the OOD *labels* changes AUC values but the
    fitted coefficients (hence the OOD *scores*) are identical, because the fit
    only ever sees ID. We assert the OOD decision scores are invariant to the
    OOD label vector by comparing two h3 runs with different y_ood but identical
    features -> same underlying scores => identical bootstrap ci geometry shift
    is driven ONLY by labels, never by a refit.
    """
    rng = _rng(77)
    n_id, n_ood = 500, 300
    SE_id = rng.normal(0, 1, n_id)
    depth_id = rng.normal(0, 1, n_id)
    y_id = _logistic_y(rng, 1.0 * SE_id + 1.2 * depth_id)
    SE_o = rng.normal(0, 1, n_ood)
    depth_o = rng.normal(0, 1, n_ood)

    # Reproduce the frozen scores the way h3 does, then confirm they are a pure
    # function of ID fit + OOD features (independent of any OOD label vector).
    mc = A._fit_logit(SE_id.reshape(-1, 1), y_id)
    mcd = A._fit_logit(np.column_stack([SE_id, depth_id]), y_id)
    sc = A._logit_scores(mc, SE_o.reshape(-1, 1))
    scd = A._logit_scores(mcd, np.column_stack([SE_o, depth_o]))

    # Two arbitrary, different OOD label vectors.
    y_o_a = _logistic_y(rng, 1.0 * SE_o + 1.2 * depth_o)
    y_o_b = 1 - y_o_a  # flipped

    res_a = A.h3_ood(SE_id, depth_id, y_id, SE_o, depth_o, y_o_a)
    res_b = A.h3_ood(SE_id, depth_id, y_id, SE_o, depth_o, y_o_b)

    # If coefficients were refit on OOD, res_b would fit the flipped labels and
    # dAUC would not be the exact sign-flip-consistent value derived from the
    # SAME frozen scores. With frozen coefficients:
    #   AUC(flip) = 1 - AUC(orig) for each model, so dAUC_b = -dAUC_a exactly.
    if np.unique(y_o_a).size == 2:
        assert res_b["dAUC_ood"] == pytest.approx(-res_a["dAUC_ood"], abs=1e-9)
    # And the frozen scores reproduce res_a exactly (no refit anywhere).
    manual_a = roc_auc_score(y_o_a, scd) - roc_auc_score(y_o_a, sc)
    assert res_a["dAUC_ood"] == pytest.approx(manual_a, abs=1e-9)


# --------------------------------------------------------------------------- #
# DeLong cross-checks                                                         #
# --------------------------------------------------------------------------- #
def _delong_slow(sa, sb, y):
    """Independent structural-component DeLong (no midrank speedup)."""
    from scipy import stats as _st

    y = np.asarray(y)
    sa = np.asarray(sa, float)
    sb = np.asarray(sb, float)
    pos, neg = y == 1, y == 0

    def comps(s):
        P, N = s[pos], s[neg]
        m, n = len(P), len(N)
        M = (P[:, None] > N[None, :]).astype(float) + 0.5 * (P[:, None] == N[None, :])
        return M.mean(), M.mean(axis=1), M.mean(axis=0), m, n

    a1, v10a, v01a, m, n = comps(sa)
    a2, v10b, v01b, _, _ = comps(sb)
    S10 = np.cov(np.vstack([v10a, v10b]))
    S01 = np.cov(np.vstack([v01a, v01b]))
    S = S10 / m + S01 / n
    L = np.array([[1.0, -1.0]])
    var = (L @ S @ L.T).item()
    z = (a1 - a2) / np.sqrt(var)
    return 2 * _st.norm.sf(abs(z))


def test_delong_reproduces_sklearn_auc():
    rng = _rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    sa = y * 0.8 + rng.normal(0, 1, n)
    sb = y * 0.5 + rng.normal(0, 1, n)
    order = (-y).argsort(kind="mergesort")
    m = int(y.sum())
    preds = np.vstack((sa, sb))[:, order]
    aucs, _ = A._fast_delong(preds, m)
    assert aucs[0] == pytest.approx(roc_auc_score(y, sa), abs=1e-9)
    assert aucs[1] == pytest.approx(roc_auc_score(y, sb), abs=1e-9)


def test_delong_matches_independent_implementation():
    rng = _rng(3)
    for _ in range(5):
        n = 300
        y = rng.integers(0, 2, n)
        sa = y * 0.6 + rng.normal(0, 1, n)
        sb = y * 0.4 + rng.normal(0, 1, n)
        assert A.delong_test(sa, sb, y) == pytest.approx(_delong_slow(sa, sb, y), rel=1e-6)


def test_delong_single_class_returns_nan():
    y = np.ones(10, dtype=int)
    assert np.isnan(A.delong_test(np.arange(10.0), np.arange(10.0)[::-1], y))


def test_delong_identical_scores_p_is_one():
    rng = _rng(9)
    n = 200
    y = rng.integers(0, 2, n)
    s = rng.normal(0, 1, n)
    assert A.delong_test(s, s, y) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# AUROC / guards / Holm                                                       #
# --------------------------------------------------------------------------- #
def test_auroc_single_class_is_nan():
    assert np.isnan(A.auroc(np.arange(5.0), np.ones(5, dtype=int)))


def test_binary_guard_rejects_non_binary():
    with pytest.raises(ValueError):
        A._as_1d_binary([0, 1, 2])
    with pytest.raises(ValueError):
        A._as_1d_binary([1, 2])


def test_holm_matches_hand_computation():
    # Known example: raw p = [0.01, 0.04, 0.03]; m=3.
    # sorted: 0.01(*3=0.03), 0.03(*2=0.06), 0.04(*1=0.04 -> max w/ running 0.06 =0.06)
    raw = [0.01, 0.04, 0.03]
    adj = A.holm(raw)
    assert adj[0] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.06)
    assert adj[1] == pytest.approx(0.06)


def test_holm_nan_excluded_from_family():
    raw = [0.01, np.nan, 0.02]
    adj = A.holm(raw)
    # Only 2 finite members => multipliers 2 and 1, NaN slot stays NaN.
    assert np.isnan(adj[1])
    assert adj[0] == pytest.approx(0.02)  # 0.01 * 2
    assert adj[2] == pytest.approx(0.02)  # 0.02 * 1, monotone >= 0.02


def test_holm_never_below_raw_and_capped_at_one():
    raw = [0.6, 0.7, 0.9]
    adj = A.holm(raw)
    for r, a in zip(raw, adj):
        assert a >= r - 1e-12
        assert a <= 1.0
