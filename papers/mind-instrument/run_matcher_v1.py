"""MATCHER v1 — anatomy-blocked two-stage matching vs the v0 baseline (frozen prereg).

PREREG_matcher_v1_2026_06_10.md: block discovery (agglomerative, k=N/12) -> block matching
(signature Hungarian) -> within-block assignment -> v0 global refinement -> label-free consensus
with plain v0. Gate M1: beat 0.1073 at N=192, sign p<0.05, same pairs/subsets as the baseline.

Usage: python papers/mind-instrument/run_matcher_v1.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from styxx import mind  # noqa: E402
from run_gavagai_v0 import translate, FAMILY  # noqa: E402
from run_telepathy_v0 import PROBES, PROBE_CAT  # noqa: E402

REPS = HERE / "telepathy_reps.npz"
BASE = HERE / "gavagai_scale_result.json"
SEED = 0
SIZES = (48, 96, 192)
WORDS = mind.BATTERY + PROBES


def refine(DA: np.ndarray, DB: np.ndarray, pi: np.ndarray, max_it: int = 50) -> np.ndarray:
    """The frozen v0 global refinement from an arbitrary init."""
    for _ in range(max_it):
        DBp = DB[:, pi]
        A = DA - DA.mean(1, keepdims=True)
        B = DBp - DBp.mean(1, keepdims=True)
        num = A @ B.T
        den = np.sqrt((A * A).sum(1))[:, None] * np.sqrt((B * B).sum(1))[None, :] + 1e-12
        _, new_pi = linear_sum_assignment(-(num / den))
        if np.array_equal(new_pi, pi):
            break
        pi = new_pi
    return pi


def objective(DA: np.ndarray, DB: np.ndarray, pi: np.ndarray) -> float:
    """Label-free mapping quality: mean rowwise corr of DA vs DB under pi."""
    DBp = DB[np.ix_(pi, pi)]
    A = DA - DA.mean(1, keepdims=True)
    B = DBp - DBp.mean(1, keepdims=True)
    num = (A * B).sum(1)
    den = np.sqrt((A * A).sum(1) * (B * B).sum(1)) + 1e-12
    return float((num / den).mean())


def cluster_blocks(D: np.ndarray, k: int) -> np.ndarray:
    Z = linkage(squareform(D, checks=False), method="average")
    return fcluster(Z, t=k, criterion="maxclust")


def cluster_signature(D: np.ndarray, members: np.ndarray, pad: int) -> np.ndarray:
    """Permutation-invariant cluster signature: sorted within-distances + sorted centroid row."""
    sub = D[np.ix_(members, members)]
    within = np.sort(sub[np.triu_indices(len(members), 1)]) if len(members) > 1 else np.zeros(1)
    cent = np.sort(D[members].mean(0))
    def z(v):
        return (v - v.mean()) / (v.std() + 1e-9) if len(v) > 1 else v
    def pad_to(v, n):
        return np.pad(v, (0, max(0, n - len(v))), mode="edge")[:n]
    return np.concatenate([pad_to(z(within), pad), pad_to(z(cent), pad)])


def block_seed(DA: np.ndarray, DB: np.ndarray, k: int) -> np.ndarray:
    n = DA.shape[0]
    ca, cb = cluster_blocks(DA, k), cluster_blocks(DB, k)
    ids_a, ids_b = np.unique(ca), np.unique(cb)
    pad = n
    SA = np.stack([cluster_signature(DA, np.where(ca == i)[0], pad) for i in ids_a])
    SB = np.stack([cluster_signature(DB, np.where(cb == j)[0], pad) for j in ids_b])
    cost = ((SA[:, None, :] - SB[None, :, :]) ** 2).sum(-1)
    ra, rb = linear_sum_assignment(cost)
    pi = np.full(n, -1)
    used_b: list[int] = []
    # within matched blocks: v0 sorted-profile cost, rectangular Hungarian
    FA, FB = np.sort(DA, axis=1), np.sort(DB, axis=1)
    for ia, ib in zip(ra, rb):
        ma = np.where(ca == ids_a[ia])[0]
        mb = np.where(cb == ids_b[ib])[0]
        c = ((FA[ma][:, None, :] - FB[mb][None, :, :]) ** 2).sum(-1)
        rr, cc = linear_sum_assignment(c)
        for x, y in zip(rr, cc):
            pi[ma[x]] = mb[y]
        used_b += [mb[y] for y in cc]
    # leftover pool (unmatched A rows x unused B cols)
    rest_a = np.where(pi < 0)[0]
    rest_b = np.array([j for j in range(n) if j not in set(used_b)])
    if len(rest_a):
        c = ((FA[rest_a][:, None, :] - FB[rest_b][None, :, :]) ** 2).sum(-1)
        rr, cc = linear_sum_assignment(c)
        for x, y in zip(rr, cc):
            pi[rest_a[x]] = rest_b[y]
    return pi


def matcher_v1(DA: np.ndarray, DB: np.ndarray, k: int) -> np.ndarray:
    cand1 = refine(DA, DB, block_seed(DA, DB, k))
    cand0 = translate(DA, DB)                      # plain v0 (sorted-signature init + refine)
    return cand1 if objective(DA, DB, cand1) >= objective(DA, DB, cand0) else cand0, \
           (objective(DA, DB, cand1) >= objective(DA, DB, cand0))


def sign_test_p(kk: int, n: int) -> float:
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(kk, n - kk) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> int:
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(WORDS))
    subsets = {n: np.sort(order[:n]) for n in SIZES}
    z = np.load(REPS)
    minds = list(z.keys())
    base = json.loads(BASE.read_text(encoding="utf-8"))
    base_rows = {(r["a"], r["b"]): r for r in base["rows"]}

    # VOID
    R0 = z[minds[0]].astype(float)
    perm = rng.permutation(192)
    D0 = mind.distmat(R0)
    pi, _ = matcher_v1(D0, mind.distmat(R0[perm]), 192 // 12)
    if float((perm[pi] == np.arange(192)).mean()) < 1.0:
        (HERE / "matcher_v1_result.json").write_text(
            json.dumps({"verdict": "VOID-PIPELINE"}, indent=2) + "\n", encoding="utf-8")
        print("VOID-PIPELINE"); return 2

    rows, picks = [], {n: 0 for n in SIZES}
    for a, b in combinations(minds, 2):
        if FAMILY[a] == FAMILY[b]:
            continue
        RA, RB = z[a].astype(float), z[b].astype(float)
        row = {"a": a, "b": b}
        for n in SIZES:
            idx = subsets[n]
            A, B = RA[idx], RB[idx]
            p = rng.permutation(n)
            pi, used_block = matcher_v1(mind.distmat(A), mind.distmat(B[p]), max(2, n // 12))
            rec = p[pi]
            row[f"acc_{n}"] = round(float((rec == np.arange(n)).mean()), 4)
            picks[n] += int(used_block)
        row["base_192"] = base_rows[(a, b)]["acc_192"]
        rows.append(row)
        print(f"[x] {a:13s}<->{b:13s} v1: " +
              " ".join(f"N{n}={row[f'acc_{n}']:.3f}" for n in SIZES) +
              f" | v0 N192={row['base_192']:.3f}", flush=True)

    means = {n: round(float(np.mean([r[f"acc_{n}"] for r in rows])), 4) for n in SIZES}
    d = [r["acc_192"] - r["base_192"] for r in rows]
    n_up = sum(1 for x in d if x > 0)
    n_eff = sum(1 for x in d if x != 0)
    p = sign_test_p(n_up, n_eff)
    m1 = means[192] > 0.1073 and p < 0.05 and n_up > n_eff / 2

    receipt = {"experiment": "MATCHER v1 - anatomy-blocked two-stage + label-free consensus",
               "prereg": "papers/mind-instrument/PREREG_matcher_v1_2026_06_10.md",
               "baseline_receipt": "gavagai_scale_result.json", "seed": SEED,
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "rows": rows, "mean_accuracy_by_N": means,
               "baseline_mean_by_N": base["mean_accuracy_by_N"],
               "pairs_improving_at_192": n_up, "pairs_nonties": n_eff,
               "sign_test_p_two_sided": round(p, 5),
               "consensus_block_pick_rate_by_N": {n: round(picks[n] / len(rows), 3) for n in SIZES},
               "M1_pass": m1,
               "verdict": "MATCHER-BEATEN" if m1 else "DESIGN-INSUFFICIENT"}
    (HERE / "matcher_v1_result.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                 encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("mean_accuracy_by_N", "baseline_mean_by_N",
                                              "pairs_improving_at_192", "sign_test_p_two_sided",
                                              "consensus_block_pick_rate_by_N", "verdict")},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
