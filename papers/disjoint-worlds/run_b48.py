"""B48 — the 45-pair legibility matrix, per PREREG_b48_legibility_matrix_ten_2026_08_06.md."""
from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from styxx_transfer import TransferMap          # noqa: E402
from styxx.islands import _gap_p, frame, affinity   # noqa: E402

BANK = ROOT / "papers" / "mind-instrument" / "normeq_reps.npz"
SMOKE = "--smoke" in sys.argv
SEED = 343


def discover(A, B, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(A))
    Bs, truth = B[perm], np.argsort(perm)
    k = min(60, A.shape[1], B.shape[1], len(A) - 1)
    tm = TransferMap.fit(A, Bs, k=k)
    M = np.stack([tm.transfer_point(x) for x in A])
    _, col = linear_sum_assignment(np.linalg.norm(M[:, None, :] - Bs[None, :, :], axis=-1))
    return float((col == truth).mean())


def main() -> int:
    z = np.load(BANK, allow_pickle=True)
    names = list(z.files)[:3] if SMOKE else list(z.files)
    reps = {n: np.asarray(z[n], dtype=float) for n in names}
    n_items = len(next(iter(reps.values())))
    pairs = list(combinations(names, 2))
    res = {"prereg": "PREREG_b48_legibility_matrix_ten_2026_08_06.md", "smoke": SMOKE,
           "members": names, "n_items": n_items, "chance": round(1 / n_items, 4),
           "n_pairs": len(pairs), "pair_legibility": {}, "pair_null": {}, "pair_affinity": {}}
    t0 = time.time()
    for i, (a, b) in enumerate(pairs):
        ab = discover(reps[a], reps[b], SEED + i)
        ba = discover(reps[b], reps[a], SEED + 1000 + i)
        res["pair_legibility"][f"{a}__{b}"] = {"ab": round(ab, 4), "ba": round(ba, 4),
                                               "mean": round((ab + ba) / 2, 4)}
        rng = np.random.default_rng(SEED + 5000 + i)
        Bn = reps[b][rng.permutation(n_items)]
        res["pair_null"][f"{a}__{b}"] = round(discover(reps[a], Bn, SEED + 7000 + i), 4)
        k = min(20, n_items - 1)
        res["pair_affinity"][f"{a}__{b}"] = round(
            affinity(frame(reps[a], k), frame(reps[b], k)), 4)
        print(f">> [{i+1}/{len(pairs)}] {a} <-> {b}: ab={ab:.4f} ba={ba:.4f} "
              f"null={res['pair_null'][f'{a}__{b}']:.4f} [{time.time()-t0:.0f}s]", flush=True)

    means = {m: round(float(np.mean([v["mean"] for k, v in res["pair_legibility"].items()
                                     if m in k.split("__")])), 4) for m in names}
    res["member_mean_legibility"] = means
    res["max_pair_legibility"] = round(max(v["mean"] for v in res["pair_legibility"].values()), 4)
    res["max_null_legibility"] = round(max(res["pair_null"].values()), 4)
    res["bimodality_p_member_legibility"] = _gap_p(np.array(list(means.values())), 1000, SEED)

    if not SMOKE:
        x = np.array([res["pair_affinity"][k] for k in res["pair_legibility"]])
        y = np.array([v["mean"] for v in res["pair_legibility"].values()])
        lin = np.polyfit(x, y, 1)
        r2l = 1 - np.sum((y - np.polyval(lin, x)) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)
        best = (0.0, None)
        for thr in np.linspace(x.min(), x.max(), 40):
            for sharp in (20, 50, 200):
                p = 1 / (1 + np.exp(-sharp * (x - thr)))
                c = np.polyfit(p, y, 1)
                r2 = 1 - np.sum((y - np.polyval(c, p)) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)
                if r2 > best[0]:
                    best = (r2, round(float(thr), 4))
        res["cliff_check"] = {"linear_r2": round(float(r2l), 4),
                              "sigmoid_r2": round(float(best[0]), 4),
                              "sigmoid_minus_linear_r2": round(float(best[0] - r2l), 4),
                              "best_threshold_affinity": best[1],
                              "note": "ungated by prereg; no measured prior for this construction"}
    try:
        from styxx.protocol import Experiment
        v = Experiment(HERE / "PREREG_b48_legibility_matrix_ten_2026_08_06.md").score(res, smoke=SMOKE)
        res["verdict"], res["gates"] = v.verdict, v.gates
        res["prereg_commit"] = v.prereg_commit
    except Exception as e:
        res["verdict"] = f"UNSCORED__{type(e).__name__}: {e}"
    (HERE / f"b48_result{'_smoke' if SMOKE else ''}.json").write_text(
        json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"\nmember mean legibility: {means}")
    print(f"VERDICT: {res['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
