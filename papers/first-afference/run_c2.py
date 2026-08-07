"""C2 — the surrogate recalibration exam, per PREREG_c2_surrogate_recalibration_2026_08_07.md."""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from styxx.coupling import couple                 # noqa: E402

TR = 1.5
N_VERT = 500
SEED = 343


def run_pair(A, ts, B, conf, n_perm):
    return couple(A, ts, B, ts, confound=conf, bin_seconds=TR, n_perm=n_perm,
                  min_bins=200, alpha=0.01)


def ar1(n, d, rho, rng):
    z = np.zeros((n, d))
    z[0] = rng.standard_normal(d)
    for i in range(1, n):
        z[i] = rho * z[i - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal(d)
    return z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    data, t0 = Path(a.data), time.time()
    n_perm = 100 if a.smoke else 500

    subs = sorted(p.stem.split("_")[0] for p in data.glob("sub-*_L.npy"))
    X = {s: np.load(data / f"{s}_L.npy").astype(float) for s in subs}
    n_t = min(v.shape[0] for v in X.values())
    vert = np.random.default_rng(SEED).choice(X[subs[0]].shape[1], N_VERT, replace=False)
    X = {s: v[:n_t][:, vert] for s, v in X.items()}
    ts = np.arange(n_t, dtype=float) * TR
    conf = lambda b: (np.asarray(b) // 75).astype(int)   # noqa: E731

    pairs = list(combinations(subs, 2))
    if a.smoke:
        pairs = pairs[:3]
    res = {"prereg": "PREREG_c2_surrogate_recalibration_2026_08_07.md", "smoke": a.smoke,
           "subjects": subs, "n_pairs": len(pairs), "real": {}, "reversed": {},
           "independent_ar": {}, "shared_trend": {}}

    for x, y in pairs:
        r = run_pair(X[x], ts, X[y], conf, n_perm)
        res["real"][f"{x}__{y}"] = {"verdict": r.verdict, "licensed": r.licensed,
                                    "matched_p": r.matched_p,
                                    "surrogate_p": r.dependence.get("surrogate_p")}
        rr = run_pair(X[x], ts, X[y][::-1], conf, n_perm)
        res["reversed"][f"{x}__{y}"] = {"verdict": rr.verdict, "licensed": rr.licensed}
        print(f">> {x}~{y}: real {r.verdict[:44]} | rev {rr.verdict[:34]} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    n_ar = 3 if a.smoke else 20
    hours = np.arange(336, dtype=float) * 3600.0
    hconf = lambda b: (np.asarray(b) % 24)               # noqa: E731
    for s in range(n_ar):
        A = ar1(336, 6, 0.98, np.random.default_rng(s * 1000 + 1))
        B = ar1(336, 6, 0.98, np.random.default_rng(s * 1000 + 2))
        r = couple(A, hours, B, hours, confound=hconf, bin_seconds=3600, n_perm=n_perm,
                   min_bins=200, alpha=0.01)
        res["independent_ar"][f"s{s}"] = {"verdict": r.verdict, "licensed": r.licensed}
        print(f">> ar s{s}: {r.verdict[:60]}", flush=True)

    n_tr = 3 if a.smoke else 10
    for s in range(n_tr):
        rng = np.random.default_rng(s)
        t_col = np.arange(336)[:, None]
        A = rng.standard_normal((336, 6)) + t_col * rng.standard_normal((1, 6)) * 0.05
        B = rng.standard_normal((336, 6)) + t_col * rng.standard_normal((1, 6)) * 0.05
        r = couple(A, hours, B, hours, confound=hconf, bin_seconds=3600, n_perm=n_perm,
                   min_bins=200, alpha=0.01)
        res["shared_trend"][f"s{s}"] = {"verdict": r.verdict, "licensed": r.licensed}
        print(f">> trend s{s}: {r.verdict[:60]}", flush=True)

    def frac(key):
        v = [d["licensed"] for d in res[key].values()]
        return round(sum(v) / max(len(v), 1), 4)
    res["frac_real_coupled"] = frac("real")
    res["frac_reversed_coupled"] = frac("reversed")
    res["frac_independent_ar_coupled"] = frac("independent_ar")
    res["frac_shared_trend_coupled"] = frac("shared_trend")

    try:
        from styxx.protocol import Experiment
        v = Experiment(HERE / "PREREG_c2_surrogate_recalibration_2026_08_07.md").score(res, smoke=a.smoke)
        res["verdict"], res["gates"] = v.verdict, v.gates
        res["prereg_commit"] = v.prereg_commit
    except Exception as e:
        res["verdict"] = f"UNSCORED__{type(e).__name__}: {e}"
    (HERE / f"c2_result{'_smoke' if a.smoke else ''}.json").write_text(
        json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"\nreal {res['frac_real_coupled']} | reversed {res['frac_reversed_coupled']} | "
          f"indep-AR {res['frac_independent_ar_coupled']} | trend {res['frac_shared_trend_coupled']}")
    print(f"VERDICT: {res['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
