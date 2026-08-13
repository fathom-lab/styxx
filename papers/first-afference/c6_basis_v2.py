"""C6 power basis v2 — the two legs `power_c6.py` could not certify.

`power_c6.py` (2026-08-13) produced the detection curve. It is not sufficient to freeze a bar,
for two reasons that are properties of the SCRIPT, not of the data:

1. **The null leg has resolution 1/60.** It reports `frac_of_cohorts_at_or_above_0.80 = 0.0` at
   c=0 from 60 replicates. That bounds the cohort false-positive rate at <1.7%, which cannot
   certify a gate operating at alpha=0.01. A bar whose false-positive rate is only known to
   within the number you are trying to bound is not derived, it is asserted with extra steps.
   Leg A draws 400 null cohorts and reports an exact Clopper-Pearson upper bound per candidate
   bar.

2. **Nobody checked whether both guards bind.** Licensing is `surrogate_p <= a AND matched_p <= a`.
   If one leg is satisfied essentially always, the conjunction is a single test wearing two
   names — the conjunctive form of the meta_audit_v1 error (a disjunction with an always-true
   term is always true). Leg C records, for every subject decision, which leg was binding.

Leg B refines the grid around the knee under the WORST-case autocorrelation, so the minimum
detectable coupling is stated as a number rather than read off a bracket.

Same constraints as v1: synthetic only, no real effect size is read, `styxx.power` stays
quarantined. Base seed is deliberately disjoint from v1's so this is an independent draw and not
a re-slice of the same streams.
"""
from __future__ import annotations
import json, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from styxx.coupling import phase_randomize, _confound_matched_perm, _trend_r2  # noqa: E402

SEED = 90210          # disjoint from v1's 343 by construction
N_T = 300
N_SUB = 7
N_COL = 2
ALPHA = 0.01
RHO_LO, RHO_MID, RHO_HI = 0.4967, 0.6511, 0.8054
N_PERM = 300


def ar1(n, d, rho, rng):
    z = np.zeros((n, d))
    z[0] = rng.standard_normal(d)
    for i in range(1, n):
        z[i] = rho * z[i - 1] + np.sqrt(1 - rho**2) * rng.standard_normal(d)
    return z


def loo_stat(x, others_mean):
    a = x - x.mean(0)
    b = others_mean - others_mean.mean(0)
    num = (a * b).sum(0)
    den = np.sqrt((a**2).sum(0) * (b**2).sum(0)) + 1e-12
    return float(np.mean(num / den))


def license_loo(x, others_mean, groups, rng_s, rng_p, alpha=ALPHA):
    """Identical decision rule to power_c6.py, but it reports which leg bound."""
    xz = (x - x.mean(0)) / (x.std(0) + 1e-9)
    mz = (others_mean - others_mean.mean(0)) / (others_mean.std(0) + 1e-9)
    if _trend_r2(xz) >= 0.2 and _trend_r2(mz) >= 0.2:
        return {"licensed": False, "bind": "trend_refusal", "sp": None, "mp": None}
    obs = loo_stat(x, others_mean)
    ge_s = sum(loo_stat(phase_randomize(x, rng_s), others_mean) >= obs for _ in range(N_PERM))
    sp = (ge_s + 1) / (N_PERM + 1)
    n = len(x)
    ge_m = sum(loo_stat(x[_confound_matched_perm(n, groups, rng_p)], others_mean) >= obs
               for _ in range(N_PERM))
    mp = (ge_m + 1) / (N_PERM + 1)
    s_ok, m_ok = sp <= alpha, mp <= alpha
    if s_ok and m_ok:
        bind = "both_pass"
    elif s_ok and not m_ok:
        bind = "matched_binds"      # surrogate would have licensed; matched vetoed
    elif m_ok and not s_ok:
        bind = "surrogate_binds"    # matched would have licensed; surrogate vetoed
    else:
        bind = "both_veto"
    return {"licensed": bool(s_ok and m_ok), "bind": bind,
            "sp": round(sp, 5), "mp": round(mp, 5)}


def planted_cohort(c, rho, rng):
    s = ar1(N_T, N_COL, rho, rng)
    return [c * s + np.sqrt(1 - c**2) * ar1(N_T, N_COL, rho, rng) for _ in range(N_SUB)]


def block_confound_cohort(c, rho, rng):
    """Shared BLOCK structure, not shared dynamics — the confound the matched permutation exists
    to veto.

    Every subject gets the same per-block mean offset (the blocks are the same 75-sample groups
    the matched permutation shuffles within). There is no shared within-block signal at all, so
    any licensing here is the instrument mistaking common condition structure for coupling.
    Phase randomisation PRESERVES the amplitude spectrum, so a slow block pattern survives it —
    this is precisely the case the surrogate leg is expected to miss and the matched leg to
    catch. If `matched_binds` does not fire here, the second guard is decorative on this design.
    """
    groups = (np.arange(N_T) // 75).astype(int)
    n_blk = int(groups.max()) + 1
    offs = rng.standard_normal((n_blk, N_COL))
    shared = offs[groups]
    return [c * shared + np.sqrt(1 - c**2) * ar1(N_T, N_COL, rho, rng) for _ in range(N_SUB)]


def cohort_pass(X, seed, binds):
    groups = (np.arange(N_T) // 75).astype(int)
    lic = []
    for i in range(len(X)):
        others = np.mean([X[j] for j in range(len(X)) if j != i], axis=0)
        r = license_loo(X[i], others, groups,
                        np.random.default_rng(seed + 41000 + i),
                        np.random.default_rng(seed + 31000 + i))
        binds[r["bind"]] += 1
        lic.append(r["licensed"])
    return int(np.sum(lic))


def clopper_pearson_upper(k, n, conf=0.95):
    """Exact one-sided upper bound on a binomial rate. No scipy dependency: bisection on the
    regularised incomplete beta via numpy-only continued fraction is overkill, so use the
    relationship to the beta quantile through a simple search on the binomial tail."""
    from math import comb
    if k >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        # P(X <= k | p=mid); upper bound is p where this equals 1-conf
        tail = sum(comb(n, i) * mid**i * (1 - mid)**(n - i) for i in range(k + 1))
        if tail > 1 - conf:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def run_cells(cells, n_rep, tag, t0, out, gen=planted_cohort):
    for (rho, c) in cells:
        counts, binds = [], Counter()
        for r in range(n_rep):
            rng = np.random.default_rng(SEED + 7717 * r + int(1000 * c) + int(100000 * rho)
                                        + (55511 if gen is not planted_cohort else 0))
            X = gen(c, rho, rng)
            counts.append(cohort_pass(X, SEED + 999 * r + int(1000 * c), binds))
        counts = np.array(counts)
        key = f"{tag}|rho={rho:.4f}|c={c:.2f}"
        hist = {int(k): int((counts == k).sum()) for k in range(N_SUB + 1)}
        rec = {"n_rep": n_rep, "mean_licensed_fraction": round(float(counts.mean() / N_SUB), 4),
               "count_histogram": hist,
               "bind_profile": dict(binds),
               "bar_table": {}}
        for k in range(1, N_SUB + 1):
            hits = int((counts >= k).sum())
            rec["bar_table"][f">={k}/7"] = {
                "rate": round(hits / n_rep, 4),
                "cp_upper95": round(clopper_pearson_upper(hits, n_rep), 4)}
        # dispersion check: is per-subject licensing independent within a cohort?
        p_hat = counts.mean() / N_SUB
        binom_var = N_SUB * p_hat * (1 - p_hat)
        rec["dispersion_ratio_obs_over_binomial"] = (
            round(float(counts.var(ddof=1) / binom_var), 3) if binom_var > 1e-9 else None)
        out["cells"][key] = rec
        print(f">> {key}: mean {rec['mean_licensed_fraction']:.3f} "
              f"P(>=6/7)={rec['bar_table']['>=6/7']['rate']:.3f} "
              f"disp={rec['dispersion_ratio_obs_over_binomial']} "
              f"binds={dict(binds)} [{time.time()-t0:.0f}s]", flush=True)


def main():
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    out = {"smoke": smoke, "seed": SEED, "n_perm": N_PERM, "alpha": ALPHA,
           "n_t": N_T, "n_sub": N_SUB, "n_col": N_COL,
           "note": "synthetic only; no real effect size was read; independent draw from v1",
           "cells": {}}

    n_null = 40 if smoke else 400
    n_knee = 8 if smoke else 60

    # LEG A — the null, at resolution the bar actually needs, all three autocorrelations.
    run_cells([(RHO_LO, 0.0), (RHO_MID, 0.0), (RHO_HI, 0.0)], n_null, "null", t0, out)

    # LEG B — fine grid around the knee, worst-case autocorrelation only.
    run_cells([(RHO_HI, c) for c in (0.32, 0.36, 0.40, 0.44, 0.48)], n_knee, "knee", t0, out)

    # LEG C — does the second guard ever bind? Plant the confound it exists to veto.
    run_cells([(RHO_HI, c) for c in (0.3, 0.5, 0.7)], n_knee, "blockconf", t0, out,
              gen=block_confound_cohort)

    (HERE / f"c6_basis_v2{'_smoke' if smoke else ''}.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote c6_basis_v2{'_smoke' if smoke else ''}.json in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
