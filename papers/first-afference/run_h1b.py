"""H1b — per-subject readability, per PREREG_h1b_human_readability_2026_08_06.md.

Leave-one-subject-out identification against a group template built from the other seven.
Item space throughout: no voxel alignment, no fitting, nothing tunable toward a verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from styxx.islands import _gap_p                # noqa: E402

SUBJECTS = [f"subj0{i}" for i in range(1, 9)]
SEED = 343


def item_similarity(X: np.ndarray) -> np.ndarray:
    """Item x item correlation of a subject's responses (voxel-count independent)."""
    Xc = X - X.mean(1, keepdims=True)
    Xc /= (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    return Xc @ Xc.T


def readability(S_sim: np.ndarray, others: list, train: np.ndarray, test: np.ndarray) -> float:
    """Fraction of held-out images correctly identified from the group template's profiles."""
    template = np.mean(others, axis=0)
    P_s = S_sim[np.ix_(test, train)]
    P_t = template[np.ix_(test, train)]
    P_s = (P_s - P_s.mean(1, keepdims=True)) / (P_s.std(1, keepdims=True) + 1e-12)
    P_t = (P_t - P_t.mean(1, keepdims=True)) / (P_t.std(1, keepdims=True) + 1e-12)
    C = P_s @ P_t.T / P_s.shape[1]
    _, col = linear_sum_assignment(-C)
    return float((col == np.arange(len(test))).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    data, t0 = Path(a.data), time.time()

    mats, cocos = {}, {}
    for s in SUBJECTS:
        z = np.load(data / f"{s}_shared.npz")
        cocos[s] = z["coco"]
        mats[s] = z["X"].astype(float)
    common = sorted(set.intersection(*(set(c.tolist()) for c in cocos.values())))
    if a.smoke:
        common = common[:80]
    sims = {}
    for s in SUBJECTS:
        pos = {int(c): i for i, c in enumerate(cocos[s])}
        X = mats[s][[pos[c] for c in common]]
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        sims[s] = item_similarity(X)
    n = len(common)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    train, test = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
    print(f"cohort: {len(SUBJECTS)} subjects x {n} shared images "
          f"({len(train)} train / {len(test)} test) [{time.time()-t0:.0f}s]", flush=True)

    read = {}
    for s in SUBJECTS:
        others = [sims[o] for o in SUBJECTS if o != s]
        read[s] = round(readability(sims[s], others, train, test), 4)
        print(f">> {s}: readability {read[s]:.4f}", flush=True)

    chance = 1.0 / len(test)
    vals = np.array([read[s] for s in SUBJECTS])
    res = {"prereg": "PREREG_h1b_human_readability_2026_08_06.md", "smoke": a.smoke,
           "n_subjects": len(SUBJECTS), "n_shared_images": n,
           "n_train": int(len(train)), "n_test": int(len(test)),
           "chance": round(chance, 6), "readability": read,
           "median_readability": round(float(np.median(vals)), 4),
           "min_readability": round(float(vals.min()), 4),
           "max_readability": round(float(vals.max()), 4),
           "median_readability_over_chance": round(float(np.median(vals)) / chance, 4),
           "bimodality_p_readability": _gap_p(vals, 100 if a.smoke else 1000, SEED)}
    try:
        h1a = json.loads((HERE / "h1a_result.json").read_text(encoding="utf-8"))
        al = np.array([h1a["mean_affinity"][s] for s in SUBJECTS])
        res["alignment_vs_readability"] = {
            "alignment": {s: h1a["mean_affinity"][s] for s in SUBJECTS},
            "pearson_r": round(float(np.corrcoef(al, vals)[0, 1]), 4),
            "note": "ungated by prereg; 8 points licenses no fit — recorded for successors"}
    except Exception as e:  # noqa: BLE001
        res["alignment_vs_readability"] = {"error": str(e)}
    try:
        from styxx.protocol import Experiment
        v = Experiment(HERE / "PREREG_h1b_human_readability_2026_08_06.md").score(res, smoke=a.smoke)
        res["verdict"], res["gates"] = v.verdict, v.gates
        res["prereg_commit"] = v.prereg_commit
    except Exception as e:
        res["verdict"] = f"UNSCORED__{type(e).__name__}: {e}"
    (HERE / f"h1b_result{'_smoke' if a.smoke else ''}.json").write_text(
        json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"\nchance {chance:.5f} | median {res['median_readability']} "
          f"({res['median_readability_over_chance']}x) | bimodality p "
          f"{res['bimodality_p_readability']}")
    print(f"VERDICT: {res['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
