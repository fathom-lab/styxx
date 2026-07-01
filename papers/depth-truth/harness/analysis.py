"""Statistics for the keystone depth-vs-truth experiment.

Implements PREREG §2 ("Hypotheses and exact tests") EXACTLY, plus the paired /
DeLong / Holm machinery §2 references and the discipline of §11 ("no free
parameters not fixed above ... deterministic given the pinned seeds").

Every test here is the frozen §2 test; any deviation invalidates the finding.

- H1  (§2, "signal"):     AUROC(depth -> correct) > 0.5; 10,000-resample
                          bootstrap 95% CI excludes 0.5.
- H2  (§2, "keystone"):   dAUC = AUC(logistic SE+depth) - AUC(logistic SE) on
                          the SAME ID items > 0; paired-bootstrap 10k 95% CI
                          excludes 0; likelihood-ratio test (df=1) p < .01.
                          Secondary vs LP_mean / LP_norm, Holm-corrected.
- H3  (§2, "OOD"):        logistic(SE) and logistic(SE+depth) fitted on ID ONLY,
                          coefficients frozen, scored on OOD-1; dAUC>0 with
                          paired-bootstrap CI excluding 0.
- DeLong (§2, "Sensitivity"): reported alongside every paired bootstrap.

Pure functions. No GPU, no network, no model loading. numpy / scipy / sklearn
only, for statistics. Deterministic given seed=7 (the pinned bootstrap seed).

Shared results-row contract (written per item to results/*.jsonl), for reference:
    {id, prompt_hash, answer, correct, LP_mean, LP_norm, SE, depth, excluded_flag}
This module consumes only the numeric columns of the complete-case (§5) subset;
callers are responsible for applying complete-case exclusion before calling in.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Pinned per PREREG §1 ("All seeds pinned ... bootstrap seed 7") and §7.
_BOOTSTRAP_SEED = 7
_N_BOOT = 10_000

# Unpenalized logistic regression => true maximum-likelihood coefficients, so
# the deviance-based likelihood-ratio test (H2) is exact. lbfgs on fixed data is
# deterministic (no randomness), so no seed is consumed by the fit itself.
_LOGIT_KW = dict(penalty=None, solver="lbfgs", max_iter=1000)


# --------------------------------------------------------------------------- #
# Low-level helpers                                                           #
# --------------------------------------------------------------------------- #
def _as_1d_float(a) -> np.ndarray:
    return np.asarray(a, dtype=float).ravel()


def _as_1d_binary(y) -> np.ndarray:
    y = np.asarray(y).ravel()
    yb = y.astype(int)
    if not np.array_equal(np.unique(yb), np.unique(yb).clip(0, 1)) or not set(
        np.unique(yb)
    ).issubset({0, 1}):
        raise ValueError("y must be binary 0/1 (correct flags).")
    return yb


def _both_classes(y: np.ndarray) -> bool:
    """AUROC / logistic fits are undefined without both outcome classes present."""
    u = np.unique(y)
    return u.size == 2


def _fit_logit(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Fit an unpenalized logistic model (MLE) on design matrix X (n, k)."""
    model = LogisticRegression(**_LOGIT_KW)
    model.fit(X, y)
    return model


def _logit_scores(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Positive-class scores (monotone in P(correct)); decision_function is fine
    for AUROC and avoids probability clipping. Ordered by model.classes_."""
    return model.decision_function(X)


def _deviance(model: LogisticRegression, X: np.ndarray, y: np.ndarray) -> float:
    """-2 * log-likelihood of a fitted logistic model on (X, y)."""
    p = model.predict_proba(X)[:, 1]
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1.0 - p))
    return -2.0 * ll


# --------------------------------------------------------------------------- #
# DeLong test (§2 "Sensitivity: DeLong tests reported alongside every           #
# paired bootstrap")                                                           #
#                                                                             #
# Standard DeLong covariance estimator for two correlated AUCs.               #
# Method: DeLong, DeLong & Clarke-Pearson (1988), "Comparing the Areas under  #
# Two or More Correlated ROC Curves", Biometrics 44:837-845, using the fast    #
# midrank (structural-component) computation of Sun & Xu (2014), IEEE SPL      #
# 21(11):1389-1393. The AUC-as-U-statistic identity gives V10 (per-positive)   #
# and V01 (per-negative) structural components; their sample covariance yields #
# Var(AUC_a - AUC_b), and z = (AUC_a - AUC_b)/sqrt(Var) is standard normal.    #
# --------------------------------------------------------------------------- #
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks (average ranks, ties shared) of x. Sun & Xu (2014) Algorithm 1."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1  # 1-based average rank over the tie run
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed: np.ndarray, m: int):
    """Sun & Xu (2014) fast DeLong.

    Args:
        predictions_sorted_transposed: (k, n) array of k classifiers' scores with
            all `m` positive samples first, then the negatives.
        m: number of positive samples.
    Returns:
        (aucs, delongcov): aucs shape (k,), covariance shape (k, k).
    """
    k = predictions_sorted_transposed.shape[0]
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n           # per-positive structural components
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m     # per-negative structural components
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(scores_a, scores_b, y) -> float:
    """DeLong two-sided p-value that AUC_a == AUC_b for two correlated ROC
    curves scored on the SAME labels y. Implements the standard DeLong
    covariance estimator (see module comment above for the citation).

    Returns np.nan if y is single-class (AUC undefined).
    """
    scores_a = _as_1d_float(scores_a)
    scores_b = _as_1d_float(scores_b)
    y = _as_1d_binary(y)
    if not _both_classes(y):
        return float("nan")

    order = (-y).argsort(kind="mergesort")  # positives (y==1) first, stable
    label_1_count = int(y.sum())
    preds = np.vstack((scores_a, scores_b))[:, order]
    aucs, cov = _fast_delong(preds, label_1_count)

    l = np.array([[1.0, -1.0]])
    var = float((l @ cov @ l.T).item())
    diff = float(aucs[0] - aucs[1])
    if var <= 0.0:
        # Zero estimated variance: identical (p=1) or a degenerate tie => no evidence.
        return 1.0 if diff == 0.0 else float("nan")
    z = diff / np.sqrt(var)
    p = 2.0 * stats.norm.sf(abs(z))
    return float(p)


# --------------------------------------------------------------------------- #
# AUROC                                                                        #
# --------------------------------------------------------------------------- #
def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC via sklearn.metrics.roc_auc_score. Returns nan if y single-class."""
    scores = _as_1d_float(scores)
    y = _as_1d_binary(y)
    if not _both_classes(y):
        return float("nan")
    return float(roc_auc_score(y, scores))


# --------------------------------------------------------------------------- #
# H1 — bootstrap AUROC CI                                                      #
# --------------------------------------------------------------------------- #
def bootstrap_auroc_ci(scores, y, n: int = _N_BOOT, seed: int = _BOOTSTRAP_SEED):
    """10,000-resample percentile 95% CI for AUROC(scores -> y) (PREREG §2 H1).

    Case-resampling bootstrap: draw n_items indices with replacement, recompute
    AUROC. Resamples whose draw is single-class are skipped (AUROC undefined);
    the percentile CI is taken over the valid resamples. Deterministic for a
    given seed via numpy.random.default_rng.

    Returns (ci_lo, ci_hi) at the 2.5 / 97.5 percentiles.
    """
    scores = _as_1d_float(scores)
    y = _as_1d_binary(y)
    n_items = len(y)
    rng = np.random.default_rng(seed)
    stats_out = np.empty(n, dtype=float)
    for b in range(n):
        idx = rng.integers(0, n_items, size=n_items)
        yb = y[idx]
        if not _both_classes(yb):
            stats_out[b] = np.nan
            continue
        stats_out[b] = roc_auc_score(yb, scores[idx])
    valid = stats_out[~np.isnan(stats_out)]
    if valid.size == 0:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return (float(lo), float(hi))


def h1(depth, y) -> dict:
    """PREREG §2 H1: AUROC(depth -> correct) > 0.5 with 10k-bootstrap 95% CI
    excluding 0.5.

    Returns {auroc, ci_lo, ci_hi, excludes_half}. `excludes_half` is True only
    when the point AUROC > 0.5 AND the whole 95% CI lies strictly above 0.5
    (the one-sided signal claim of §2; a CI wholly below 0.5 does not satisfy H1).
    """
    depth = _as_1d_float(depth)
    y = _as_1d_binary(y)
    a = auroc(depth, y)
    lo, hi = bootstrap_auroc_ci(depth, y, n=_N_BOOT, seed=_BOOTSTRAP_SEED)
    excludes = bool(
        np.isfinite(a) and np.isfinite(lo) and np.isfinite(hi) and a > 0.5 and lo > 0.5
    )
    return {"auroc": a, "ci_lo": lo, "ci_hi": hi, "excludes_half": excludes}


# --------------------------------------------------------------------------- #
# H2 — additivity of depth over a confounder (SE primary; LP_* secondary)      #
# --------------------------------------------------------------------------- #
def _paired_delta_auc(conf: np.ndarray, depth: np.ndarray, y: np.ndarray) -> float:
    """Point dAUC = AUC(logistic conf+depth) - AUC(logistic conf), refit on the
    given items, scored on those same items (H2 in-sample point estimate)."""
    Xc = conf.reshape(-1, 1)
    Xcd = np.column_stack([conf, depth])
    m_c = _fit_logit(Xc, y)
    m_cd = _fit_logit(Xcd, y)
    auc_c = roc_auc_score(y, _logit_scores(m_c, Xc))
    auc_cd = roc_auc_score(y, _logit_scores(m_cd, Xcd))
    return float(auc_cd - auc_c)


def _lrt_p(conf: np.ndarray, depth: np.ndarray, y: np.ndarray) -> float:
    """Likelihood-ratio test, df=1, for adding `depth` to the conf-only logistic
    model (PREREG §2 H2). Statistic = deviance(reduced) - deviance(full) ~ chi2_1.
    """
    Xc = conf.reshape(-1, 1)
    Xcd = np.column_stack([conf, depth])
    m_c = _fit_logit(Xc, y)
    m_cd = _fit_logit(Xcd, y)
    lr = _deviance(m_c, Xc, y) - _deviance(m_cd, Xcd, y)
    lr = max(lr, 0.0)  # numerical guard; nested MLE => full deviance <= reduced
    return float(stats.chi2.sf(lr, df=1))


def h2_additivity(SE, depth, y, seed: int = _BOOTSTRAP_SEED) -> dict:
    """PREREG §2 H2 (keystone additivity), primary opponent SE.

    On the SAME ID items:
      dAUC   = AUC(logistic: SE + depth) - AUC(logistic: SE)
      CI     = paired-bootstrap 10k percentile 95% CI of dAUC (refit both models
               on each resample); excludes_zero := whole CI > 0.
      lrt_p  = likelihood-ratio test df=1 for adding depth (p < .01 => passes).
      delong_p = DeLong p-value on the two logistic scorers over the same items.

    Returns {dAUC, ci_lo, ci_hi, excludes_zero, lrt_p, lrt_pass, delong_p}.
    This is the SE-specialised call of the general confounder machinery; use
    h2_additivity_named(...) directly to reuse it for LP_mean / LP_norm.
    """
    return h2_additivity_named(SE, depth, y, name="SE", seed=seed)


def h2_additivity_named(conf, depth, y, name: str = "conf",
                        seed: int = _BOOTSTRAP_SEED) -> dict:
    """General §2-H2 additivity test for ANY confounder `conf` (so the same code
    scores SE, LP_mean, LP_norm; secondary confounders are Holm-corrected by the
    caller — see h2_full). Semantics identical to h2_additivity."""
    conf = _as_1d_float(conf)
    depth = _as_1d_float(depth)
    y = _as_1d_binary(y)
    n_items = len(y)

    d_auc = _paired_delta_auc(conf, depth, y) if _both_classes(y) else float("nan")
    lrt = _lrt_p(conf, depth, y) if _both_classes(y) else float("nan")

    # Paired bootstrap: resample items, refit BOTH models, recompute dAUC.
    rng = np.random.default_rng(seed)
    boot = np.empty(_N_BOOT, dtype=float)
    for b in range(_N_BOOT):
        idx = rng.integers(0, n_items, size=n_items)
        yb = y[idx]
        if not _both_classes(yb):
            boot[b] = np.nan
            continue
        boot[b] = _paired_delta_auc(conf[idx], depth[idx], yb)
    valid = boot[~np.isnan(boot)]
    if valid.size == 0:
        ci_lo = ci_hi = float("nan")
    else:
        lo, hi = np.percentile(valid, [2.5, 97.5])
        ci_lo, ci_hi = float(lo), float(hi)

    # DeLong on the two fitted logistic scorers (full-data fit), same items.
    if _both_classes(y):
        Xc = conf.reshape(-1, 1)
        Xcd = np.column_stack([conf, depth])
        s_c = _logit_scores(_fit_logit(Xc, y), Xc)
        s_cd = _logit_scores(_fit_logit(Xcd, y), Xcd)
        dl_p = delong_test(s_cd, s_c, y)
    else:
        dl_p = float("nan")

    excludes_zero = bool(
        np.isfinite(d_auc) and np.isfinite(ci_lo) and d_auc > 0 and ci_lo > 0
    )
    lrt_pass = bool(np.isfinite(lrt) and lrt < 0.01)
    return {
        "confounder": name,
        "dAUC": d_auc,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "excludes_zero": excludes_zero,
        "lrt_p": lrt,
        "lrt_pass": lrt_pass,
        "delong_p": dl_p,
    }


def h2_full(SE, depth, y, LP_mean=None, LP_norm=None,
            seed: int = _BOOTSTRAP_SEED) -> dict:
    """Full §2-H2 report: primary SE plus secondary LP_mean / LP_norm with Holm
    correction across the secondary LRT p-values (PREREG §2: "Secondary
    (reported, Holm-corrected): same vs LP_mean and LP_norm").

    Returns {primary: <SE result>, secondary: {LP_mean:..., LP_norm:...},
             secondary_holm_lrt_p: {LP_mean:..., LP_norm:...}}.
    Only present secondaries are included and corrected together.
    """
    out = {"primary": h2_additivity_named(SE, depth, y, name="SE", seed=seed)}
    secondary = {}
    if LP_mean is not None:
        secondary["LP_mean"] = h2_additivity_named(LP_mean, depth, y,
                                                    name="LP_mean", seed=seed)
    if LP_norm is not None:
        secondary["LP_norm"] = h2_additivity_named(LP_norm, depth, y,
                                                    name="LP_norm", seed=seed)
    out["secondary"] = secondary
    if secondary:
        names = list(secondary.keys())
        adj = holm([secondary[k]["lrt_p"] for k in names])
        out["secondary_holm_lrt_p"] = dict(zip(names, adj))
    else:
        out["secondary_holm_lrt_p"] = {}
    return out


# --------------------------------------------------------------------------- #
# H3 — OOD retention (fit on ID, freeze coefficients, score OOD)               #
# --------------------------------------------------------------------------- #
def h3_ood(SE_id, depth_id, y_id, SE_ood, depth_ood, y_ood,
           seed: int = _BOOTSTRAP_SEED) -> dict:
    """PREREG §2 H3: logistic(SE) and logistic(SE+depth) fitted on ID ONLY,
    coefficients FROZEN, scored on OOD-1.

      dAUC_ood = AUC(SE+depth model on OOD) - AUC(SE model on OOD).
      CI       = paired bootstrap on OOD items ONLY, rescoring with the FROZEN
                 ID coefficients (no refit); percentile 95% CI; excludes 0?
      delong_p = DeLong on the two frozen scorers over the OOD items.

    Returns {dAUC_ood, ci_lo, ci_hi, excludes_zero, delong_p}.
    """
    SE_id = _as_1d_float(SE_id)
    depth_id = _as_1d_float(depth_id)
    y_id = _as_1d_binary(y_id)
    SE_ood = _as_1d_float(SE_ood)
    depth_ood = _as_1d_float(depth_ood)
    y_ood = _as_1d_binary(y_ood)

    # Fit on ID, freeze.
    Xc_id = SE_id.reshape(-1, 1)
    Xcd_id = np.column_stack([SE_id, depth_id])
    m_c = _fit_logit(Xc_id, y_id)
    m_cd = _fit_logit(Xcd_id, y_id)

    # Frozen scores on OOD.
    Xc_ood = SE_ood.reshape(-1, 1)
    Xcd_ood = np.column_stack([SE_ood, depth_ood])
    s_c = _logit_scores(m_c, Xc_ood)
    s_cd = _logit_scores(m_cd, Xcd_ood)

    if _both_classes(y_ood):
        d_auc = float(roc_auc_score(y_ood, s_cd) - roc_auc_score(y_ood, s_c))
        dl_p = delong_test(s_cd, s_c, y_ood)
    else:
        d_auc = float("nan")
        dl_p = float("nan")

    # Paired bootstrap over OOD items, frozen coefficients (rescore, no refit).
    n_ood = len(y_ood)
    rng = np.random.default_rng(seed)
    boot = np.empty(_N_BOOT, dtype=float)
    for b in range(_N_BOOT):
        idx = rng.integers(0, n_ood, size=n_ood)
        yb = y_ood[idx]
        if not _both_classes(yb):
            boot[b] = np.nan
            continue
        boot[b] = roc_auc_score(yb, s_cd[idx]) - roc_auc_score(yb, s_c[idx])
    valid = boot[~np.isnan(boot)]
    if valid.size == 0:
        ci_lo = ci_hi = float("nan")
    else:
        lo, hi = np.percentile(valid, [2.5, 97.5])
        ci_lo, ci_hi = float(lo), float(hi)

    excludes_zero = bool(
        np.isfinite(d_auc) and np.isfinite(ci_lo) and d_auc > 0 and ci_lo > 0
    )
    return {
        "dAUC_ood": d_auc,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "excludes_zero": excludes_zero,
        "delong_p": dl_p,
    }


# --------------------------------------------------------------------------- #
# Holm-Bonferroni                                                              #
# --------------------------------------------------------------------------- #
def holm(pvals) -> list:
    """Holm-Bonferroni step-down adjusted p-values (Holm 1979).

    For m sorted p-values p(1)<=...<=p(m): adjusted p(i) = max over j<=i of
    min(1, (m - j + 1) * p(j)) (enforced monotonicity). Returns adjusted
    p-values in the ORIGINAL input order. NaNs are excluded from the family
    (do not consume an m-slot) and returned as NaN.
    """
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(p)
    finite_idx = np.where(finite_mask)[0]
    if finite_idx.size == 0:
        return out.tolist()

    fp = p[finite_idx]
    m = fp.size
    order = np.argsort(fp, kind="mergesort")
    adj_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, j in enumerate(order):
        val = (m - rank) * fp[j]
        running = max(running, val)          # step-down monotonicity
        adj_sorted[j] = min(1.0, running)
    out[finite_idx] = adj_sorted
    return out.tolist()
