"""B34 — label-free nonlinear content transport, per PREREG_b34_labelfree_nonlinear_2026_08_01.

Zero labels anywhere in fitting: the target's fit rows are seeded-shuffled up front, the
initial correspondence comes from entropic GW on per-side geometry alone, and the b31v2 MLP
is refit over 8 assignment-refinement iterations. True correspondence touches only the
held-out scoring and a per-iteration pseudo-pair-accuracy diagnostic (reported, never fed
back). Arms: M-LF (GW + iterations), N0 (random pairs, no iterations — the null), R0 (random
init + iterations — reported). CPU/GPU-light, zero model loads: runs off the b31v2 banked
extractions. `--smoke` = 40 fit / 10 held-out, `_smoke` files, INVALID-only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_disjoint_worlds as R                                    # entropic_gw, distmat
from run_g0clear import CONCEPTS as FULL_CONCEPTS, split_concepts  # noqa: E402
from run_b31v2 import fit_mlp                                      # the committed M1 trainer

SMOKE = "--smoke" in sys.argv
SUFFIX = "_smoke" if SMOKE else ""
SEED = 34
ITERS = 8
TARGETS = ["llama_1b", "gemma_2b", "qwen_1p5b"]


def load_pts(name):
    z = np.load(HERE / name, allow_pickle=True)
    return {c: z["pts"][i] for i, c in enumerate(FULL_CONCEPTS)}


def read_top1(fin, ptsA, fin_ptsB, mapper):
    hits = sum(1 for i, c in enumerate(fin)
               if int(np.argmin(np.linalg.norm(fin_ptsB - mapper(ptsA[c]), axis=1))) == i)
    return hits / len(fin)


def assignment_from_map(XA, XB_shuf, mapper):
    """Full assignment on mapped-A-to-B distances (order-free)."""
    MA = np.stack([mapper(x) for x in XA])
    D = np.linalg.norm(MA[:, None, :] - XB_shuf[None, :, :], axis=-1)
    _, col = linear_sum_assignment(D)
    return col


def run_arm(XA, XB_shuf, true_col, init, iters, seed):
    """One arm: init assignment -> (iterate: fit MLP on pseudo-pairs, re-assign)."""
    col = init
    diag = [float((col == true_col).mean())]
    mapper = None
    for t in range(max(iters, 1)):
        mapper, _ = fit_mlp(XA, XB_shuf[col], seed=seed + t)
        if t + 1 >= iters:
            break
        col = assignment_from_map(XA, XB_shuf, mapper)
        diag.append(float((col == true_col).mean()))
    return mapper, diag


def main():
    rng = np.random.default_rng(SEED)
    if SMOKE:
        concepts = FULL_CONCEPTS[:50]
        tr, fin = concepts[:40], concepts[40:]
    else:
        concepts = FULL_CONCEPTS
        tr_, sel_, fin = split_concepts(seed=0)
        tr = list(tr_) + list(sel_)                  # the b31v2 fit set (392)

    ptsA = load_pts(f"_b31v2_ptsA{'' if not SMOKE else ''}.npz")
    XA = np.array([ptsA[c] for c in tr])

    results = {"prereg": "PREREG_b34_labelfree_nonlinear_2026_08_01.md",
               "seed": SEED, "iters": ITERS, "smoke": SMOKE,
               "n_fit": len(tr), "n_heldout": len(fin),
               "chance": round(1 / len(fin), 4), "targets": {}}

    for tag in TARGETS:
        t0 = time.time()
        ptsB = load_pts(f"_b31v2_pts_{tag}.npz")
        XB = np.array([ptsB[c] for c in tr])
        fin_ptsB = np.array([ptsB[c] for c in fin])

        # seeded shuffle so aligned-order leakage is impossible by construction
        perm = rng.permutation(len(tr))
        XB_shuf = XB[perm]
        true_col = np.argsort(perm)      # column in XB_shuf holding row i's true partner

        # M-LF: GW-seeded assignment + iterations
        Tgw, _ = R.entropic_gw(R.distmat(XA), R.distmat(XB_shuf))
        _, gw_col = linear_sum_assignment(-Tgw)
        mlf, mlf_diag = run_arm(XA, XB_shuf, true_col, gw_col, ITERS, seed=SEED)
        mlf_top1 = read_top1(fin, ptsA, fin_ptsB, mlf)

        # N0: random pairs, no iterations (the null)
        rand_col = rng.permutation(len(tr))
        n0, _ = run_arm(XA, XB_shuf, true_col, rand_col, 1, seed=SEED)
        n0_top1 = read_top1(fin, ptsA, fin_ptsB, n0)

        # R0: random init + full iterations (reported, not gated)
        r0, r0_diag = run_arm(XA, XB_shuf, true_col, rand_col.copy(), ITERS, seed=SEED)
        r0_top1 = read_top1(fin, ptsA, fin_ptsB, r0)

        results["targets"][tag] = {
            "MLF_top1": round(mlf_top1, 4), "N0_top1": round(n0_top1, 4),
            "R0_top1": round(r0_top1, 4),
            "MLF_pseudo_pair_acc_by_iter": [round(x, 4) for x in mlf_diag],
            "R0_pseudo_pair_acc_by_iter": [round(x, 4) for x in r0_diag],
            "x_chance_MLF": round(mlf_top1 * len(fin), 1),
        }
        print(f">> {tag}: M-LF={mlf_top1:.4f} N0={n0_top1:.4f} R0={r0_top1:.4f} "
              f"gw_acc0={mlf_diag[0]:.3f} [{time.time()-t0:.0f}s]", flush=True)

    if SMOKE:
        results["verdict"] = "INVALID__smoke_plumbing_only"
    else:
        t = results["targets"]
        chance = 1 / len(fin)
        g0 = t["llama_1b"]["MLF_top1"] >= 0.29
        g2 = all(v["N0_top1"] <= 2 * chance for v in t.values())
        g1 = t["gemma_2b"]["MLF_top1"] >= 0.143
        results["gates"] = {"G0_machinery": g0, "G2_null": g2, "G1_bar": g1}
        if not g0:
            results["verdict"] = "INVALID__pipeline_broken"
        elif not g2:
            results["verdict"] = "INVALID__pipeline_artifact"
        elif g1:
            results["verdict"] = "TELEPATHY_BAR_CLEARED__pairing_discoverable"
        else:
            results["verdict"] = "PAIRING_NOT_DISCOVERABLE__at_this_class"

    out = HERE / f"b34_result{SUFFIX}.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nVERDICT: {results['verdict']}  -> {out.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
