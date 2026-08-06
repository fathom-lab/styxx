"""C1 — styxx.coupling on real neural time series, per PREREG_c1_*.md."""
from __future__ import annotations

import argparse
import json
import sys
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    data = Path(a.data)
    subs = sorted(p.stem.split("_")[0] for p in data.glob("sub-*_L.npy"))
    X = {s: np.load(data / f"{s}_L.npy").astype(float) for s in subs}
    n_t = min(v.shape[0] for v in X.values())
    vert = np.random.default_rng(SEED).choice(X[subs[0]].shape[1], N_VERT, replace=False)
    X = {s: v[:n_t][:, vert] for s, v in X.items()}
    ts = np.arange(n_t, dtype=float) * TR
    conf = lambda b: (np.asarray(b) // 75).astype(int)   # noqa: E731  quarter of run

    pairs = list(combinations(subs, 2))
    if a.smoke:
        pairs = pairs[:3]
    res = {"prereg": "PREREG_c1_coupling_on_neural_timeseries_2026_08_06.md", "smoke": a.smoke,
           "subjects": subs, "n_timepoints": int(n_t), "n_vertices": N_VERT, "tr": TR,
           "n_pairs": len(pairs), "real": {}, "reversed": {}}
    for cond in ("real", "reversed"):
        for x, y in pairs:
            B = X[y] if cond == "real" else X[y][::-1]
            r = couple(X[x], ts, B, ts, confound=conf, bin_seconds=TR,
                       n_perm=100 if a.smoke else 500, min_bins=200, alpha=0.01)
            res[cond][f"{x}__{y}"] = {"verdict": r.verdict, "rv": r.rv,
                                      "rv_debiased": r.rv_debiased,
                                      "matched_p": r.matched_p,
                                      "shift_p": r.dependence.get("circular_shift_p"),
                                      "lag1": r.dependence.get("lag1_autocorr_a")}
            print(f">> {cond} {x}~{y}: {r.verdict} (rv {r.rv}, deb {r.rv_debiased})", flush=True)

    def frac(c):
        v = [d["verdict"] for d in res[c].values()]
        return round(sum(x.startswith("COUPLED_BEYOND") for x in v) / max(len(v), 1), 4)
    res["frac_real_coupled"] = frac("real")
    res["frac_reversed_coupled"] = frac("reversed")
    from collections import Counter
    res["real_verdict_counts"] = dict(Counter(d["verdict"] for d in res["real"].values()))
    res["reversed_verdict_counts"] = dict(Counter(d["verdict"] for d in res["reversed"].values()))
    try:
        from styxx.protocol import Experiment
        v = Experiment(HERE / "PREREG_c1_coupling_on_neural_timeseries_2026_08_06.md").score(res, smoke=a.smoke)
        res["verdict"], res["gates"] = v.verdict, v.gates
        res["prereg_commit"] = v.prereg_commit
    except Exception as e:
        res["verdict"] = f"UNSCORED__{type(e).__name__}: {e}"
    (HERE / f"c1_result{'_smoke' if a.smoke else ''}.json").write_text(
        json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"\nreal coupled {res['frac_real_coupled']} | reversed coupled {res['frac_reversed_coupled']}")
    print(f"VERDICT: {res['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
