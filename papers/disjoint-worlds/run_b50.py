"""B50 — b48's question re-asked with a null criterion that can be met.

Per PREREG_b50_legibility_islands_2026_08_08.md: the legibility matrix is REUSED from
``b48_result.json`` (it was never in question — b48 failed on its null *bar*, not its nulls),
and the 45 nulls are DRAWN FRESH at a new seed, because b48's were seen and a criterion chosen
after seeing a null distribution is not a preregistration.
"""
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
from styxx_transfer import TransferMap              # noqa: E402
from styxx.islands import _gap_p                    # noqa: E402

BANK = ROOT / "papers" / "mind-instrument" / "normeq_reps.npz"
PRIOR = HERE / "b48_result.json"
PREREG = "PREREG_b50_legibility_islands_2026_08_08.md"
SMOKE = "--smoke" in sys.argv
SEED = 8080                       # deliberately NOT b48's 343: the old nulls were seen


def discover(A, B, seed):
    """Identical construction to run_b48.discover — reused so the nulls stay comparable."""
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
    chance = 1.0 / n_items
    pairs = list(combinations(names, 2))

    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    leg = prior["pair_legibility"]
    missing = [f"{a}__{b}" for a, b in pairs if f"{a}__{b}" not in leg]
    if missing and not SMOKE:
        raise SystemExit(f"reused matrix is missing {len(missing)} pairs: {missing[:3]}")

    res = {"prereg": PREREG, "smoke": SMOKE, "members": names, "n_items": n_items,
           "chance": round(chance, 4), "n_pairs": len(pairs), "null_seed": SEED,
           "legibility_matrix_reused_from": "b48_result.json",
           "legibility_matrix_source_commit": prior.get("prereg_commit"),
           "pair_legibility": {k: leg[k] for a, b in pairs if (k := f"{a}__{b}") in leg},
           "pair_null_fresh": {}, "pair_null_b48": {}}

    t0 = time.time()
    for i, (a, b) in enumerate(pairs):
        key = f"{a}__{b}"
        rng = np.random.default_rng(SEED + 5000 + i)
        Bn = reps[b][rng.permutation(n_items)]
        res["pair_null_fresh"][key] = round(discover(reps[a], Bn, SEED + 7000 + i), 4)
        res["pair_null_b48"][key] = prior["pair_null"].get(key)
        print(f">> [{i+1}/{len(pairs)}] {key}: null={res['pair_null_fresh'][key]:.4f} "
              f"(b48 {res['pair_null_b48'][key]}) [{time.time()-t0:.0f}s]", flush=True)

    nulls = np.array(list(res["pair_null_fresh"].values()), dtype=float)
    five_x = 5.0 * chance
    res["median_null_legibility"] = round(float(np.median(nulls)), 4)
    res["mean_null_legibility"] = round(float(nulls.mean()), 4)
    res["max_null_legibility"] = round(float(nulls.max()), 4)
    res["frac_nulls_above_5x_chance"] = round(float((nulls >= five_x).mean()), 4)
    res["n_nulls_above_5x_chance"] = int((nulls >= five_x).sum())
    res["five_x_chance"] = round(five_x, 4)

    means = {m: round(float(np.mean([v["mean"] for k, v in res["pair_legibility"].items()
                                     if m in k.split("__")])), 4) for m in names}
    res["member_mean_legibility"] = means
    res["max_pair_legibility"] = round(max(v["mean"] for v in res["pair_legibility"].values()), 4)
    res["bimodality_p_member_legibility"] = _gap_p(np.array(list(means.values())), 100_000, SEED)

    b48n = np.array([v for v in res["pair_null_b48"].values() if v is not None], dtype=float)
    if len(b48n):
        res["null_replication"] = {
            "b48_median": round(float(np.median(b48n)), 4),
            "fresh_median": res["median_null_legibility"],
            "b48_max": round(float(b48n.max()), 4), "fresh_max": res["max_null_legibility"],
            "note": "two independent 45-draw null families; b48's max cleared a single-draw bar "
                    "by ordinary luck, which is the defect B50 exists to correct"}

    try:
        from styxx.protocol import Experiment
        e = Experiment(HERE / PREREG, require_power_basis=True)
        chk = e.check_metrics(res)
        res["metric_check"] = chk
        unusable = sorted(n for n, d in chk.items() if not d["usable"])
        if unusable and not SMOKE:
            raise SystemExit(f"gate metrics unresolvable before scoring: {unusable}")
        v = e.score(res, smoke=SMOKE)
        res["verdict"], res["gates"] = v.verdict, v.gates
        res["prereg_commit"] = v.prereg_commit
    except Exception as exc:
        res["verdict"] = f"UNSCORED__{type(exc).__name__}: {exc}"

    (HERE / f"b50_result{'_smoke' if SMOKE else ''}.json").write_text(
        json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"\nnulls: median {res['median_null_legibility']} max {res['max_null_legibility']} "
          f"| {res['n_nulls_above_5x_chance']}/{len(nulls)} at or above 5x chance "
          f"({res['five_x_chance']})")
    print(f"max pair legibility {res['max_pair_legibility']} | "
          f"bimodality p {res['bimodality_p_member_legibility']}")
    print(f"VERDICT: {res['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
