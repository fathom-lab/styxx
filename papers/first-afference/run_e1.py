"""E1 — the effective-sample-size bake-off, per PREREG_e1_effective_n_bakeoff_2026_08_08.md.

Scores candidate estimators against analytic AR(1) truth, disqualifies any that fails silently,
identifies what produced the un-generated committed addendum, and recomputes C5's range with the
winner.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

PREREG = "PREREG_e1_effective_n_bakeoff_2026_08_08.md"
SMOKE = "--smoke" in sys.argv
SEED = 5150
RHOS = (0.5, 0.7, 0.8, 0.9, 0.95)
NREP = 20 if SMOKE else 400
NT = 300


def _acf(a, biased=True):
    """Autocorrelation sequence. biased=True divides every lag by the lag-0 sum (the standard
    Bartlett convention); biased=False uses the overlapping-pair Pearson correlation."""
    a = np.asarray(a, float) - np.mean(a)
    d = float(a @ a)
    n = a.size
    if biased:
        return np.array([1.0] + [float(a[k:] @ a[:-k]) / d for k in range(1, n // 2)])
    out = [1.0]
    for k in range(1, n // 2):
        x, y = a[k:], a[:-k]
        sx, sy = x.std(), y.std()
        out.append(0.0 if sx == 0 or sy == 0 else float(np.mean(x * y) / (sx * sy)))
    return np.array(out)


# ---- candidates. Each takes a series and returns effective n. -------------------------------
def est_truncate_first_nonpositive(a, biased=True):
    """power.effective_n's rule: sum 2*rho_k until the first non-positive lag."""
    r = _acf(a, biased)
    tot = 1.0
    for k in range(1, len(r)):
        if r[k] <= 0:
            break
        tot += 2.0 * r[k]
    return len(a) / tot if tot > 0 else float(len(a))


def est_ar1_closed_form(a):
    """n*(1-rho1)/(1+rho1) — exact if the process really is AR(1)."""
    r = _acf(a)[1]
    return len(a) * (1.0 - r) / (1.0 + r) if r < 1.0 else 1.0


def est_bartlett_sqrt_n(a):
    """Bartlett window at bandwidth sqrt(n), the textbook fixed-bandwidth choice."""
    r = _acf(a)
    n = len(a)
    m = int(np.sqrt(n))
    ks = np.arange(1, min(m, len(r) - 1) + 1)
    tot = 1.0 + 2.0 * float(np.sum((1.0 - ks / (m + 1.0)) * r[ks]))
    return n / tot if tot > 0 else float(n)


def est_initial_positive_sequence(a):
    """Geyer's initial positive sequence: sum ADJACENT PAIRS of autocorrelations, stop when a
    pair sum goes non-positive. Designed for exactly the regime where the single-lag rule
    truncates too early on noise, and it cannot be fooled by an alternating sign pattern."""
    r = _acf(a)
    tot = 1.0
    k = 1
    while k + 1 < len(r):
        pair = r[k] + r[k + 1]
        if pair <= 0:
            break
        tot += 2.0 * pair
        k += 2
    return len(a) / tot if tot > 0 else float(len(a))


CANDIDATES = {
    "truncate_first_nonpositive_biased": lambda a: est_truncate_first_nonpositive(a, True),
    "truncate_first_nonpositive_pearson": lambda a: est_truncate_first_nonpositive(a, False),
    "ar1_closed_form": est_ar1_closed_form,
    "bartlett_sqrt_n": est_bartlett_sqrt_n,
    "initial_positive_sequence": est_initial_positive_sequence,
}


def ar1(n, rho, rng):
    x = np.zeros(n)
    e = rng.standard_normal(n)
    x[0] = e[0] / np.sqrt(1 - rho ** 2)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def silent_probe(rng):
    """Negative lag-1, strong lag-2. The first-nonpositive rule stops at lag 1 and reports NO
    correction; the truth is far below nominal."""
    n = 5000
    x = np.zeros(n)
    e = rng.standard_normal(n)
    for i in range(2, n):
        x[i] = 0.95 * x[i - 2] + e[i]
    return x


def main() -> int:
    rng = np.random.default_rng(SEED)
    rhos = RHOS[:2] if SMOKE else RHOS

    # ---- accuracy on analytic AR(1) ---------------------------------------------------------
    grid, per_cand_err = {}, {k: [] for k in CANDIDATES}
    for rho in rhos:
        truth = NT * (1.0 - rho) / (1.0 + rho)
        cell = {"rho": rho, "analytic_effective_n": round(truth, 4), "estimators": {}}
        series = [ar1(NT, rho, rng) for _ in range(NREP)]
        for name, fn in CANDIDATES.items():
            vals = np.array([fn(s) for s in series], float)
            rel = np.abs(vals - truth) / truth
            per_cand_err[name].extend(rel.tolist())
            cell["estimators"][name] = {
                "median": round(float(np.median(vals)), 4),
                "median_rel_error": round(float(np.median(rel)), 4),
                "p5": round(float(np.percentile(vals, 5)), 4),
                "p95": round(float(np.percentile(vals, 95)), 4)}
        grid[f"rho_{rho}"] = cell
        print(f">> rho={rho} truth={truth:.2f} " + " ".join(
            f"{n.split('_')[0]}={c['median']:.1f}" for n, c in cell["estimators"].items()),
            flush=True)

    pooled = {k: round(float(np.median(v)), 4) for k, v in per_cand_err.items()}

    # ---- silent-failure probe (disqualifying, regardless of accuracy) -----------------------
    probe = silent_probe(np.random.default_rng(SEED + 1))
    probe_res = {}
    for name, fn in CANDIDATES.items():
        eff = float(fn(probe))
        probe_res[name] = {"effective": round(eff, 2), "nominal": len(probe),
                           "flags": bool(eff < 0.5 * len(probe))}
    disqualified = sorted(k for k, v in probe_res.items() if not v["flags"])

    eligible = {k: v for k, v in pooled.items() if k not in disqualified} or pooled
    winner = min(eligible, key=eligible.get)

    # ---- what produced the committed addendum? ----------------------------------------------
    add = json.loads((HERE / "c5_effective_df_addendum.json").read_text(encoding="utf-8"))
    subs = sorted(add["per_subject"])
    series = {s: np.loadtxt(HERE / f"eac_{s}_L.1D") for s in subs}
    ident = {}
    for name, fn in CANDIDATES.items():
        diffs = [abs(float(fn(series[s])) - add["per_subject"][s]["effective_n_bartlett"])
                 for s in subs]
        ident[name] = {"max_abs_diff_vs_addendum": round(float(max(diffs)), 4),
                       "mean_abs_diff_vs_addendum": round(float(np.mean(diffs)), 4)}
    addendum_method = min(ident, key=lambda k: ident[k]["max_abs_diff_vs_addendum"])

    # ---- recompute C5 with the winner --------------------------------------------------------
    fn = CANDIDATES[winner]
    recomputed = {s: round(float(fn(series[s])), 4) for s in subs}
    vals = np.array(list(recomputed.values()))
    med = float(np.median(vals))
    # two-sided t threshold at alpha=0.05 with df = n_eff - 2, expressed as a correlation
    from scipy import stats
    def r_crit(ne):
        df = max(ne - 2.0, 1.0)
        t = stats.t.ppf(0.975, df)
        return float(np.sqrt(t ** 2 / (t ** 2 + df)))

    STRONGEST_PAIR_R = 0.3742      # committed in c5_result.json / FINDING_c5
    new_crit = r_crit(med)
    old_crit = add["significance_threshold_r_at_median_eff"]

    res = {"prereg": PREREG, "smoke": SMOKE, "seed": SEED, "n_replicates": NREP,
           "n_cells_scored": len(grid), "grid": grid,
           "pooled_median_abs_rel_error": pooled,
           "best_median_abs_rel_error": min(pooled.values()),
           "silent_probe": probe_res, "disqualified_by_silent_probe": disqualified,
           "winner": winner, "winner_pooled_median_abs_rel_error": pooled[winner],
           "winner_flags_the_silent_probe": 1.0 if probe_res[winner]["flags"] else 0.0,
           "addendum_method_identified_as": addendum_method,
           "addendum_identification_detail": ident,
           "n_c5_subjects_recomputed": len(recomputed),
           "c5_recomputed_effective_n": recomputed,
           "c5_recomputed_min": round(float(vals.min()), 4),
           "c5_recomputed_max": round(float(vals.max()), 4),
           "c5_recomputed_median": round(med, 4),
           "c5_published_min": add["min_effective_n"], "c5_published_max": add["max_effective_n"],
           "c5_published_median_threshold_r": old_crit,
           "c5_recomputed_threshold_r": round(new_crit, 4),
           "c5_strongest_pair_r": STRONGEST_PAIR_R,
           "c5_strongest_pair_clears_published_threshold": bool(STRONGEST_PAIR_R >= old_crit),
           "c5_strongest_pair_clears_recomputed_threshold": bool(STRONGEST_PAIR_R >= new_crit),
           "c5_conclusion_sensitive_to_estimator": bool(
               (STRONGEST_PAIR_R >= old_crit) != (STRONGEST_PAIR_R >= new_crit)),
           "interpretation_limit": ("AR(1) agreement disqualifies; it does not establish "
                                    "correctness on BOLD, which is not AR(1). The winner is the "
                                    "least disqualified candidate, not a correct one.")}

    try:
        from styxx.protocol import Experiment
        e = Experiment(HERE / PREREG, require_power_basis=True)
        res["metric_check"] = e.check_metrics(res)
        bad = sorted(n for n, d in res["metric_check"].items() if not d["usable"])
        if bad and not SMOKE:
            raise SystemExit(f"unresolvable gate metrics: {bad}")
        v = e.score(res, smoke=SMOKE)
        res["verdict"], res["gates"] = v.verdict, v.gates
        res["prereg_commit"] = v.prereg_commit
    except Exception as exc:
        res["verdict"] = f"UNSCORED__{type(exc).__name__}: {exc}"

    (HERE / f"e1_result{'_smoke' if SMOKE else ''}.json").write_text(
        json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"\npooled median |rel err|: {pooled}")
    print(f"disqualified (silent): {disqualified}")
    print(f"WINNER: {winner} @ {pooled[winner]}")
    print(f"addendum was produced by: {addendum_method} "
          f"(max diff {ident[addendum_method]['max_abs_diff_vs_addendum']})")
    print(f"C5 published {add['min_effective_n']}-{add['max_effective_n']} -> recomputed "
          f"{res['c5_recomputed_min']}-{res['c5_recomputed_max']}")
    print(f"threshold r {old_crit} -> {res['c5_recomputed_threshold_r']} | strongest pair "
          f"{STRONGEST_PAIR_R} | conclusion sensitive: {res['c5_conclusion_sensitive_to_estimator']}")
    print(f"VERDICT: {res['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
