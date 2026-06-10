"""GAVAGAI-SCALE — is indeterminacy a shrinking quantity? (frozen prereg)

PREREG_gavagai_scale_2026_06_10.md: nested worlds N=48/96/192 from telepathy_reps.npz, the frozen
matcher, 33 cross-family pairs. S1: paired acc(192) vs acc(96) + two-sided sign test.

Usage: python papers/mind-instrument/run_gavagai_scale.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from styxx import mind  # noqa: E402
from run_gavagai_v0 import translate, FAMILY  # noqa: E402
from run_telepathy_v0 import PROBES, PROBE_CAT  # noqa: E402  (frozen word lists)

REPS = HERE / "telepathy_reps.npz"
SEED = 0
SIZES = (48, 96, 192)
WORDS = mind.BATTERY + PROBES
CATS = mind.BATTERY_CATEGORY + PROBE_CAT


def sign_test_p(k: int, n: int) -> float:
    """Two-sided exact sign test: k successes of n non-ties."""
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> int:
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(WORDS))
    subsets = {n: np.sort(order[:n]) for n in SIZES}
    cats = {n: np.array([CATS[i] for i in subsets[n]]) for n in SIZES}

    z = np.load(REPS)
    minds = list(z.keys())

    # VOID: self-pair at 192
    R0 = z[minds[0]].astype(float)
    perm = rng.permutation(192)
    pi = translate(mind.distmat(R0), mind.distmat(R0[perm]))
    if float((perm[pi] == np.arange(192)).mean()) < 1.0:
        out = {"verdict": "VOID-PIPELINE"}
        (HERE / "gavagai_scale_result.json").write_text(json.dumps(out, indent=2) + "\n",
                                                        encoding="utf-8")
        print("VOID-PIPELINE"); return 2

    rows = []
    for a, b in combinations(minds, 2):
        if FAMILY[a] == FAMILY[b]:
            continue
        RA, RB = z[a].astype(float), z[b].astype(float)
        row = {"a": a, "b": b}
        for n in SIZES:
            idx = subsets[n]
            A, B = RA[idx], RB[idx]
            p = rng.permutation(n)
            pi = translate(mind.distmat(A), mind.distmat(B[p]))
            rec = p[pi]
            row[f"acc_{n}"] = round(float((rec == np.arange(n)).mean()), 4)
            row[f"cat_{n}"] = round(float((cats[n][rec] == cats[n]).mean()), 4)
        rows.append(row)
        print(f"[x] {a:13s}<->{b:13s} " + " ".join(f"N{n}={row[f'acc_{n}']:.3f}" for n in SIZES),
              flush=True)

    means = {n: round(float(np.mean([r[f"acc_{n}"] for r in rows])), 4) for n in SIZES}
    ratios = {n: round(means[n] * n, 2) for n in SIZES}   # accuracy / (1/N) = acc * N
    d = [r["acc_192"] - r["acc_96"] for r in rows]
    n_up = sum(1 for x in d if x > 0)
    n_eff = sum(1 for x in d if x != 0)
    p = sign_test_p(n_up, n_eff)
    mean_up = means[192] > means[96]

    if mean_up and p < 0.05:
        verdict = "INDETERMINACY-SHRINKS"
    elif (not mean_up) and p < 0.05:
        verdict = "STRUCTURE-LOSES"
    else:
        verdict = "SATURATED"

    receipt = {"experiment": "GAVAGAI-SCALE - indeterminacy vs world size",
               "prereg": "papers/mind-instrument/PREREG_gavagai_scale_2026_06_10.md",
               "seed": SEED, "sizes": list(SIZES),
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "rows": rows, "n_xfam_pairs": len(rows),
               "mean_accuracy_by_N": means,
               "chance_by_N": {n: round(1 / n, 4) for n in SIZES},
               "accuracy_over_chance_by_N": ratios,
               "pairs_improving_192_vs_96": n_up, "pairs_nonties": n_eff,
               "sign_test_p_two_sided": round(p, 5),
               "verdict": verdict}
    (HERE / "gavagai_scale_result.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                    encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("mean_accuracy_by_N", "accuracy_over_chance_by_N",
                                              "pairs_improving_192_vs_96", "sign_test_p_two_sided",
                                              "verdict")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
