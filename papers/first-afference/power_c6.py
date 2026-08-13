"""C6 power basis — computed BEFORE the C6 bar is frozen, on SYNTHETIC data only.

C5's closing sentence: "The successor is a power calculation, not another exam. Any future gate
on this data must state the effective sample size first and derive its bar from that."

E1 then showed the effective sample size is NOT estimable at this series length, and named the
alternative: "use a statistic that does not need an effective-n estimate — a phase-randomised
surrogate." So this script derives the bar the other way: it measures, on synthetic streams with
KNOWN planted coupling, what fraction of subjects the leave-one-out instrument licenses. The
frozen C6 bar is then read off this curve rather than asserted.

`styxx.power` is QUARANTINED (third pre-release quarantine, 2026-08-08) and is deliberately not
imported here. This is a direct Monte-Carlo detection rate under the exact licensing rule C6
will use — not an analytic power formula from a module that cannot be trusted.

NOTHING IN THIS SCRIPT TOUCHES THE REAL SUBJECT TIMESERIES except two published nuisance
parameters already committed in FINDING_c5 (lag-1 autocorrelation range 0.4967-0.8054, n_t=300).
The real effect sizes are not read.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from styxx.coupling import phase_randomize, _confound_matched_perm, _trend_r2  # noqa: E402

SEED = 343
N_T = 300           # matches the committed EAC series length
N_SUB = 7           # matches the cohort
N_COL = 2           # L and R hemisphere means
ALPHA = 0.01        # unchanged from C4/C5
RHO_LO, RHO_HI = 0.4967, 0.8054   # committed in FINDING_c5, per-subject lag-1 range


def ar1(n, d, rho, rng):
    z = np.zeros((n, d))
    z[0] = rng.standard_normal(d)
    for i in range(1, n):
        z[i] = rho * z[i - 1] + np.sqrt(1 - rho**2) * rng.standard_normal(d)
    return z


def loo_stat(x, others_mean):
    """Signed mean matched-column correlation between one subject and the mean of the rest.

    Linear in cross-covariance, no fold, no square — the C3/C4 lesson. Surrogate expectation is
    genuinely zero.
    """
    a = x - x.mean(0)
    b = others_mean - others_mean.mean(0)
    num = (a * b).sum(0)
    den = np.sqrt((a**2).sum(0) * (b**2).sum(0)) + 1e-12
    return float(np.mean(num / den))


def license_loo(x, others_mean, groups, n_perm, rng_s, rng_p, alpha=ALPHA):
    xz = (x - x.mean(0)) / (x.std(0) + 1e-9)
    mz = (others_mean - others_mean.mean(0)) / (others_mean.std(0) + 1e-9)
    if _trend_r2(xz) >= 0.2 and _trend_r2(mz) >= 0.2:
        return {"obs": None, "surrogate_p": None, "matched_p": None,
                "licensed": False, "refused": "shared_trend"}
    obs = loo_stat(x, others_mean)
    ge_s = sum(loo_stat(phase_randomize(x, rng_s), others_mean) >= obs for _ in range(n_perm))
    sp = (ge_s + 1) / (n_perm + 1)
    n = len(x)
    ge_m = sum(loo_stat(x[_confound_matched_perm(n, groups, rng_p)], others_mean) >= obs
               for _ in range(n_perm))
    mp = (ge_m + 1) / (n_perm + 1)
    return {"obs": round(obs, 4), "surrogate_p": round(sp, 4), "matched_p": round(mp, 4),
            "licensed": bool(sp <= alpha and mp <= alpha)}


def planted_cohort(c, rho, rng):
    """N_SUB subjects sharing a common AR(1) stimulus signal at coupling amplitude c.

    x_i = c*s + sqrt(1-c^2)*e_i  ->  pairwise correlation ~ c^2.
    """
    s = ar1(N_T, N_COL, rho, rng)
    return [c * s + np.sqrt(1 - c**2) * ar1(N_T, N_COL, rho, rng) for _ in range(N_SUB)]


def cohort_licensed_fraction(X, n_perm, seed):
    groups = (np.arange(N_T) // 75).astype(int)
    lic = []
    for i in range(len(X)):
        others = np.mean([X[j] for j in range(len(X)) if j != i], axis=0)
        r = license_loo(X[i], others, groups,
                        n_perm,
                        np.random.default_rng(seed + 41000 + i),
                        np.random.default_rng(seed + 31000 + i))
        lic.append(r["licensed"])
    return float(np.mean(lic))


def main():
    smoke = "--smoke" in sys.argv
    n_perm = 100 if smoke else 300
    n_rep = 8 if smoke else 60
    c_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    rho_grid = [RHO_LO, (RHO_LO + RHO_HI) / 2, RHO_HI]
    t0 = time.time()
    out = {"smoke": smoke, "n_perm": n_perm, "n_rep": n_rep, "alpha": ALPHA,
           "n_t": N_T, "n_sub": N_SUB, "n_col": N_COL,
           "note": "synthetic only; no real effect size was read", "curve": {}}
    for rho in rho_grid:
        for c in c_grid:
            fr = []
            for r in range(n_rep):
                rng = np.random.default_rng(SEED + 7717 * r + int(1000 * c) + int(100 * rho))
                X = planted_cohort(c, rho, rng)
                fr.append(cohort_licensed_fraction(X, n_perm, SEED + 999 * r))
            key = f"rho={rho:.4f}|c={c:.2f}"
            out["curve"][key] = {"pairwise_r_implied": round(c**2, 4),
                                 "mean_licensed_fraction": round(float(np.mean(fr)), 4),
                                 "frac_of_cohorts_at_or_above_0.80": round(
                                     float(np.mean([f >= 0.80 for f in fr])), 4),
                                 "frac_of_cohorts_at_or_above_0.60": round(
                                     float(np.mean([f >= 0.60 for f in fr])), 4)}
            print(f">> {key}: mean licensed {out['curve'][key]['mean_licensed_fraction']:.3f} "
                  f"P(>=0.80)={out['curve'][key]['frac_of_cohorts_at_or_above_0.80']:.2f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    # the false-positive leg: c=0 IS the null cohort, already in the grid above
    out["false_positive_at_c0"] = {k: v["mean_licensed_fraction"]
                                   for k, v in out["curve"].items() if k.endswith("c=0.00")}
    (HERE / f"c6_power{'_smoke' if smoke else ''}.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote c6_power{'_smoke' if smoke else ''}.json in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
