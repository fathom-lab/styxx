"""C6 — the leave-one-out instrument on the real EAC series, against a bar derived from measured power.

Per `PREREG_c6_derived_bar_2026_08_13.md`, frozen and committed (18b8c61) BEFORE this script
existed and before any C6 statistic touched the real data. The bar (k=5 of 7) was read off
`c6_basis_v2.json` by a stated selection rule, not chosen.

Gates, exactly as frozen:
  G1_exceeds_null_ceiling         cohort_licensed_count >= 5
  G2_null_calibration_holds       per-subject licensing on a phase-randomised REAL cohort <= 0.05
  G3_matched_leg_bind_count       RECORDED, NOT GATED (value 0, unfailable by construction)

Outcome mapping:
  G2 false                     -> REFUSED__power_basis_does_not_transfer_to_this_series
  G2 true,  G1 true            -> LICENSED__cohort_coupling_above_derived_null_ceiling
  G2 true,  G1 false           -> NULL__below_derived_ceiling_at_known_power

`styxx.power` stays quarantined and is not imported. `styxx.coupling`'s withdrawal for neural
time series stands; only its three primitives are used, as in C5 and the power basis.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from styxx.coupling import phase_randomize, _confound_matched_perm, _trend_r2  # noqa: E402

SEED = 343
ALPHA = 0.01
BLOCK = 75          # same grouping the matched permutation used in C5 and the power basis
BAR_K = 5           # frozen in the prereg, derived from c6_basis_v2.json


def loo_stat(x, others_mean):
    """Signed mean matched-column correlation: subject vs the mean of the rest.

    Identical to the statistic the power basis was computed on — linear in cross-covariance,
    no fold, no square, so the surrogate expectation is genuinely zero.
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
                "licensed": False, "bind": "trend_refusal", "refused": "shared_trend"}
    obs = loo_stat(x, others_mean)
    ge_s = sum(loo_stat(phase_randomize(x, rng_s), others_mean) >= obs for _ in range(n_perm))
    sp = (ge_s + 1) / (n_perm + 1)
    n = len(x)
    ge_m = sum(loo_stat(x[_confound_matched_perm(n, groups, rng_p)], others_mean) >= obs
               for _ in range(n_perm))
    mp = (ge_m + 1) / (n_perm + 1)
    s_ok, m_ok = sp <= alpha, mp <= alpha
    if s_ok and m_ok:
        bind = "both_pass"
    elif s_ok and not m_ok:
        bind = "matched_binds"     # surrogate would have licensed; matched vetoed
    elif m_ok and not s_ok:
        bind = "surrogate_binds"
    else:
        bind = "both_veto"
    return {"obs": round(obs, 4), "surrogate_p": round(sp, 4), "matched_p": round(mp, 4),
            "licensed": bool(s_ok and m_ok), "bind": bind}


def cohort_pass(X, subs, groups, n_perm, seed):
    """Leave-one-out over the cohort. Returns per-subject records."""
    out = {}
    for i, s in enumerate(subs):
        others = np.mean([X[t] for j, t in enumerate(subs) if j != i], axis=0)
        out[s] = license_loo(X[s], others, groups, n_perm,
                             np.random.default_rng(seed + 41000 + i),
                             np.random.default_rng(seed + 31000 + i))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    data = Path(a.data)
    n_perm = 100 if a.smoke else 500

    subs = sorted({p.stem.split("_")[1] for p in data.glob("eac_sub-*_*.1D")})
    X = {}
    for s in subs:
        cols = [np.loadtxt(data / f"eac_{s}_{h}.1D").ravel() for h in ("L", "R")]
        X[s] = np.stack(cols, axis=1)
    n_t = min(v.shape[0] for v in X.values())
    X = {s: v[:n_t] for s, v in X.items()}
    groups = (np.arange(n_t) // BLOCK).astype(int)
    print(f">> {len(subs)} subjects, n_t={n_t}, n_perm={n_perm}", flush=True)

    # ---- G1: the real cohort -------------------------------------------------------------
    real = cohort_pass(X, subs, groups, n_perm, SEED)
    licensed = [s for s, r in real.items() if r["licensed"]]
    k = len(licensed)
    for s in subs:
        r = real[s]
        print(f">> {s}: obs={r['obs']} sp={r['surrogate_p']} mp={r['matched_p']} "
              f"lic={r['licensed']} bind={r['bind']} [{time.time()-t0:.0f}s]", flush=True)
    print(f">> COHORT LICENSED COUNT = {k}/{len(subs)}  (bar is >= {BAR_K})", flush=True)

    # ---- G2: does the power basis transfer? phase-randomise the REAL cohort --------------
    # Every subject replaced by its own phase-randomised surrogate: same spectrum, same
    # autocorrelation, coupling destroyed. Per-subject licensing here must sit in the regime the
    # synthetic null occupied (0.0057-0.0125), or the bar derived on synthetic data does not
    # apply to this series and the run REFUSES rather than reporting a verdict.
    n_null = 5 if a.smoke else 40
    null_hits, null_tot = 0, 0
    for rep in range(n_null):
        rs = np.random.default_rng(SEED + 77000 + rep)
        Xs = {s: phase_randomize(X[s], rs) for s in subs}
        rr = cohort_pass(Xs, subs, groups, n_perm, SEED + 5000 + rep)
        null_hits += sum(1 for r in rr.values() if r["licensed"])
        null_tot += len(subs)
        if (rep + 1) % 10 == 0 or rep == n_null - 1:
            print(f">> null rep {rep+1}/{n_null}: running rate "
                  f"{null_hits/null_tot:.4f} [{time.time()-t0:.0f}s]", flush=True)
    null_rate = null_hits / max(null_tot, 1)
    print(f">> G2 phase-randomised per-subject licensing rate = {null_rate:.4f} (bar <= 0.05)",
          flush=True)

    # ---- G3: recorded, not gated ----------------------------------------------------------
    n_matched_binds = sum(1 for r in real.values() if r["bind"] == "matched_binds")

    res = {
        "prereg": "PREREG_c6_derived_bar_2026_08_13.md",
        "prereg_commit": "18b8c61",
        "power_basis": ["c6_power.json", "c6_basis_v2.json"],
        "smoke": a.smoke, "n_perm": n_perm, "alpha": ALPHA, "n_t": n_t,
        "n_sub": len(subs), "block": BLOCK, "bar_k": BAR_K,
        "subjects": subs,
        "per_subject": real,
        "cohort_licensed_count": k,
        "licensed_subjects": licensed,
        "per_subject_license_rate_on_phase_randomised_cohort": round(null_rate, 4),
        "null_reps": n_null,
        "matched_leg_binding_count_on_real_cohort": n_matched_binds,
        "bind_profile": {b: sum(1 for r in real.values() if r["bind"] == b)
                         for b in sorted({r["bind"] for r in real.values()})},
    }

    g1 = k >= BAR_K
    g2 = null_rate <= 0.05
    res["gates_evaluated"] = {"G1_exceeds_null_ceiling": bool(g1),
                              "G2_null_calibration_holds": bool(g2),
                              "G3_matched_leg_bind_count": n_matched_binds}
    if not g2:
        res["verdict"] = "REFUSED__power_basis_does_not_transfer_to_this_series"
    elif g1:
        res["verdict"] = "LICENSED__cohort_coupling_above_derived_null_ceiling"
    else:
        res["verdict"] = "NULL__below_derived_ceiling_at_known_power"

    out = HERE / f"c6_result{'_smoke' if a.smoke else ''}.json"
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"\nlicensed {k}/{len(subs)} | null rate {null_rate:.4f} | "
          f"matched_binds {n_matched_binds}")
    print(f"VERDICT: {res['verdict']}")
    print(f"wrote {out.name} in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
