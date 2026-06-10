"""GAVAGAI v0 — radical translation between artificial minds (frozen prereg).

PREREG_gavagai_v0_2026_06_10.md: recover concept identity across minds from geometry alone —
no labels, no paired data. Hungarian assignment on permutation-invariant sorted-distance
signatures, refined by mapped-RDM correlation. Gates G1/G2/G3, VOID-PIPELINE self-pair control.

Usage:
    python papers/mind-instrument/run_gavagai_v0.py --smoke
    python papers/mind-instrument/run_gavagai_v0.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from styxx import mind  # noqa: E402

NORMEQ = HERE / "normeq_reps.npz"
OLD_ANCH = REPO / "papers" / "real-convergence" / "contextual_reps.npz"
OLD_LIVE = HERE / "atlas_live_reps.npz"
SEED = 0
N_NULL = 100
MAX_IT = 50

FAMILY = {"Qwen2.5-1.5B": "qwen", "Qwen2.5-3B": "qwen", "Qwen2.5-0.5B": "qwen",
          "Llama-3.2-1B": "llama", "Llama-3.2-3B": "llama", "Phi-3.5-mini": "phi",
          "gemma-2-2b": "gemma", "gpt2": "gpt2", "gpt2-large": "gpt2", "pythia-410m": "pythia"}
N = 96
CAT = np.array(mind.BATTERY_CATEGORY)


def translate(DA: np.ndarray, DB: np.ndarray) -> np.ndarray:
    """Unsupervised mapping π: A-index -> B-index from RDMs alone."""
    FA = np.sort(DA, axis=1)
    FB = np.sort(DB, axis=1)
    cost = ((FA[:, None, :] - FB[None, :, :]) ** 2).sum(-1)
    _, pi = linear_sum_assignment(cost)
    for _ in range(MAX_IT):
        # C[i,j] = -corr(DA[i,:], DB[j, pi(:)]) over k != i — vectorized
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


def score_pair(RA: np.ndarray, RB: np.ndarray, rng: np.random.Generator):
    """Hide B's labels by a random permutation; translate; score against truth."""
    perm = rng.permutation(N)                  # B presented in scrambled order
    DA = mind.distmat(RA)
    DB = mind.distmat(RB[perm])
    pi = translate(DA, DB)                      # maps A-i -> scrambled-B index
    recovered = perm[pi]                        # back to true B identity
    acc = float((recovered == np.arange(N)).mean())
    cat_acc = float((CAT[recovered] == CAT).mean())
    return acc, cat_acc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tag = "_SMOKE_INVALID" if args.smoke else ""
    rng = np.random.default_rng(SEED)

    zn = np.load(NORMEQ)
    za, zl = np.load(OLD_ANCH), np.load(OLD_LIVE)
    def old_reps(name):
        return (za[f"fixed__{name}"] if f"fixed__{name}" in za else zl[name]).astype(float)
    minds = list(zn.keys())[:2] if args.smoke else list(zn.keys())

    # VOID-PIPELINE: self-pair with labels hidden must come back perfect
    a0, c0 = score_pair(zn[minds[0]].astype(float), zn[minds[0]].astype(float), rng)
    if a0 < 0.999:
        out = {"verdict": "VOID-PIPELINE", "self_pair_accuracy": a0}
        (HERE / f"gavagai_v0_result{tag}.json").write_text(json.dumps(out, indent=2) + "\n",
                                                           encoding="utf-8")
        print("VOID-PIPELINE", a0); return 2

    pairs = []
    for a, b in combinations(minds, 2):
        acc_n, cat_n = score_pair(zn[a].astype(float), zn[b].astype(float), rng)
        acc_o, cat_o = score_pair(old_reps(a), old_reps(b), rng)
        pairs.append({"a": a, "b": b, "xfam": FAMILY[a] != FAMILY[b],
                      "acc_normeq": round(acc_n, 4), "cat_normeq": round(cat_n, 4),
                      "acc_old": round(acc_o, 4), "cat_old": round(cat_o, 4)})
        print(f"[{'x' if FAMILY[a]!=FAMILY[b] else '=' }] {a:13s}->{b:13s} "
              f"normeq acc={acc_n:.3f} cat={cat_n:.3f} | old acc={acc_o:.3f}", flush=True)

    xf = [p for p in pairs if p["xfam"]]
    mean_acc = float(np.mean([p["acc_normeq"] for p in xf]))
    mean_cat = float(np.mean([p["cat_normeq"] for p in xf]))
    mean_old = float(np.mean([p["acc_old"] for p in xf]))

    receipt = {"experiment": "GAVAGAI v0 - radical translation between artificial minds",
               "prereg": "papers/mind-instrument/PREREG_gavagai_v0_2026_06_10.md",
               "seed": SEED, "chance_concept": round(1 / N, 4), "chance_category": 0.125,
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "self_pair_control": a0, "pairs": pairs,
               "n_pairs": len(pairs), "n_xfam_pairs": len(xf)}

    if args.smoke:
        receipt.update({"smoke": True, "verdict": "SMOKE-OK"})
        (HERE / f"gavagai_v0_result{tag}.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                           encoding="utf-8")
        print("SMOKE-OK"); return 0

    # G1 shuffle-null: identity-decoupled geometry (derangement of B's concept identities applied
    # to one representative cross-family pair population): translator scored against the DERANGED
    # truth — what accuracy looks like when geometry carries no identity signal.
    null_scores = []
    rep_pairs = xf[: min(10, len(xf))]
    for k in range(N_NULL):
        p = rep_pairs[k % len(rep_pairs)]
        RA, RB = zn[p["a"]].astype(float), zn[p["b"]].astype(float)
        fake_truth = rng.permutation(N)         # pretend identities were assigned arbitrarily
        perm = rng.permutation(N)
        pi = translate(mind.distmat(RA), mind.distmat(RB[perm]))
        recovered = perm[pi]
        null_scores.append(float((recovered == fake_truth).mean()))
    null95 = float(np.percentile(null_scores, 95))

    g1 = mean_acc >= 0.1042 and mean_acc > null95
    g3 = mean_cat >= 0.25
    receipt.update({
        "mean_xfam_concept_accuracy_normeq": round(mean_acc, 4),
        "mean_xfam_category_accuracy_normeq": round(mean_cat, 4),
        "mean_xfam_concept_accuracy_old": round(mean_old, 4),
        "G2_normeq_minus_old": round(mean_acc - mean_old, 4),
        "null_95th_percentile": round(null95, 4), "n_null": N_NULL,
        "G1_pass": g1, "G3_pass": g3,
        "verdict": "TRANSLATION-POSSIBLE" if g1 else
                   ("PARTIAL-TRANSLATION (category)" if g3 else "QUINE-UPHELD"),
    })
    (HERE / "gavagai_v0_result.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                 encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in
                      ("mean_xfam_concept_accuracy_normeq", "mean_xfam_category_accuracy_normeq",
                       "mean_xfam_concept_accuracy_old", "null_95th_percentile", "verdict")},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
