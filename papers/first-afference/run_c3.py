"""C3 — the linear statistic under spectral surrogates, per PREREG_c3_linear_statistic_2026_08_07.md."""
from __future__ import annotations
import argparse, json, sys, time
from itertools import combinations
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from styxx.coupling import phase_randomize, _confound_matched_perm   # noqa: E402

SEED, N_VERT, TR = 343, 500, 1.5


def isc_stat(A, B):
    Ac = A - A.mean(0); Bc = B - B.mean(0)
    num = (Ac * Bc).sum(0)
    den = np.sqrt((Ac**2).sum(0) * (Bc**2).sum(0)) + 1e-12
    return float(np.mean(np.abs(num / den)))


def license_pair(A, B, groups, n_perm, alpha=0.01):
    obs = isc_stat(A, B)
    rs = np.random.default_rng(SEED + 41000)
    ge_s = sum(isc_stat(A, phase_randomize(B, rs)) >= obs for _ in range(n_perm))
    sp = (ge_s + 1) / (n_perm + 1)
    rp = np.random.default_rng(SEED + 31000)
    n = len(A)
    ge_m = sum(isc_stat(A, B[_confound_matched_perm(n, groups, rp)]) >= obs for _ in range(n_perm))
    mp = (ge_m + 1) / (n_perm + 1)
    return {"obs": round(obs, 4), "surrogate_p": round(sp, 4), "matched_p": round(mp, 4),
            "licensed": bool(sp <= alpha and mp <= alpha)}


def ar1(n, d, rho, rng):
    z = np.zeros((n, d)); z[0] = rng.standard_normal(d)
    for i in range(1, n):
        z[i] = rho * z[i-1] + np.sqrt(1 - rho**2) * rng.standard_normal(d)
    return z


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True)
    ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    data, t0 = Path(a.data), time.time()
    n_perm = 100 if a.smoke else 500
    subs = sorted(p.stem.split("_")[0] for p in data.glob("sub-*_L.npy"))
    X = {s: np.load(data / f"{s}_L.npy").astype(float) for s in subs}
    n_t = min(v.shape[0] for v in X.values())
    vert = np.random.default_rng(SEED).choice(X[subs[0]].shape[1], N_VERT, replace=False)
    X = {s: v[:n_t][:, vert] for s, v in X.items()}
    q = (np.arange(n_t) // 75).astype(int)
    pairs = list(combinations(subs, 2))[:3 if a.smoke else None]
    res = {"prereg": "PREREG_c3_linear_statistic_2026_08_07.md", "smoke": a.smoke,
           "real": {}, "reversed": {}, "independent_ar": {}, "shared_trend": {}}
    for x, y in pairs:
        res["real"][f"{x}__{y}"] = license_pair(X[x], X[y], q, n_perm)
        res["reversed"][f"{x}__{y}"] = license_pair(X[x], X[y][::-1], q, n_perm)
        print(f">> {x}~{y}: real lic={res['real'][f'{x}__{y}']['licensed']} "
              f"(obs {res['real'][f'{x}__{y}']['obs']}) rev lic="
              f"{res['reversed'][f'{x}__{y}']['licensed']} [{time.time()-t0:.0f}s]", flush=True)
    g24 = (np.arange(336) % 24).astype(int)
    for s in range(3 if a.smoke else 20):
        A = ar1(336, 6, 0.98, np.random.default_rng(s*1000+1))
        B = ar1(336, 6, 0.98, np.random.default_rng(s*1000+2))
        res["independent_ar"][f"s{s}"] = license_pair(A, B, g24, n_perm)
    for s in range(3 if a.smoke else 10):
        rng = np.random.default_rng(s); t_ = np.arange(336)[:, None]
        A = rng.standard_normal((336, 6)) + t_ * rng.standard_normal((1, 6)) * 0.05
        B = rng.standard_normal((336, 6)) + t_ * rng.standard_normal((1, 6)) * 0.05
        res["shared_trend"][f"s{s}"] = license_pair(A, B, g24, n_perm)
    for k in ("real", "reversed", "independent_ar", "shared_trend"):
        v = [d["licensed"] for d in res[k].values()]
        res[f"frac_{k}_coupled" if k != "real" else "frac_real_coupled"] = round(sum(v)/max(len(v),1), 4)
    res["frac_reversed_coupled"] = res.pop("frac_reversed_coupled")
    try:
        from styxx.protocol import Experiment
        v = Experiment(HERE / "PREREG_c3_linear_statistic_2026_08_07.md").score(res, smoke=a.smoke)
        res["verdict"], res["gates"], res["prereg_commit"] = v.verdict, v.gates, v.prereg_commit
    except Exception as e:
        res["verdict"] = f"UNSCORED__{type(e).__name__}: {e}"
    (HERE / f"c3_result{'_smoke' if a.smoke else ''}.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"\nreal {res['frac_real_coupled']} rev {res['frac_reversed_coupled']} "
          f"ar {res['frac_independent_ar_coupled']} trend {res['frac_shared_trend_coupled']}")
    print(f"VERDICT: {res['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
