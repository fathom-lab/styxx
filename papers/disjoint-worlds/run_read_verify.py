"""Adversarial verification of the gemma read-at-chance anomaly.

gemma-2-2b has the HIGHEST RSA (0.955) yet reads at EXACTLY chance (0.0143 = 1/70). Before that
becomes the headline ("read is recoverability-bound, not isometry-bound"), rule out the artifact:
is gemma's concept cloud DEGENERATE (collapsed -> nearest-neighbor read is random and RSA is
spuriously high), or is the geometry healthy but the cross-map genuinely fails? And is read-at-chance
a single-layer accident? Compares gemma to Qwen (reads 4x, presumed healthy); sweeps gemma layers.

Reuses the cached Llama-3B A extraction (_rung2_ptsA_vecsA.npz) for the transfer map.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "introspection-gate"))

from introspection_gate import load_model
from styxx_transfer import TransferMap
import run_thought_transfer as P
from run_g0clear import CONCEPTS, split_concepts, extract_multi

CACHE_A = HERE / "_rung2_ptsA_vecsA.npz"
SRC = "meta-llama/Llama-3.2-3B-Instruct"


def cloud_health(R):
    """Effective rank (participation ratio of singular values) + pairwise-distance CV."""
    Rc = R - R.mean(0)
    s = np.linalg.svd(Rc, compute_uv=False)
    s2 = s ** 2
    pr = (s2.sum() ** 2) / (np.sum(s2 ** 2) + 1e-12)        # participation ratio ~ effective dim
    d = np.linalg.norm(R[:, None] - R[None], axis=2)[np.triu_indices(len(R), 1)]
    cv = float(d.std() / (d.mean() + 1e-12))                 # low CV => collapsed cloud
    return {"eff_rank": round(float(pr), 1), "dist_mean": round(float(d.mean()), 3),
            "dist_cv": round(cv, 3), "n": len(R)}


@torch.no_grad()
def extract_pts(model, tok, layer):
    pts, _ = P.extract(model, tok, layer)
    return pts


def read_at(ptsA, ptsB, tm, fin):
    fin_ptsB = np.array([ptsB[c] for c in fin])
    hits = sum(1 for i, c in enumerate(fin)
               if int(np.argmin(np.linalg.norm(fin_ptsB - tm.transfer_point(ptsA[c]), axis=1))) == i)
    return hits / len(fin)


def main():
    z = np.load(CACHE_A, allow_pickle=True)
    ptsA = {c: z["pts"][i] for i, c in enumerate(CONCEPTS)}
    RA = np.array([ptsA[c] for c in CONCEPTS])
    s0 = json.loads((HERE / "g0clear_result_llama3b.json").read_text(encoding="utf-8"))
    kstar = s0["locked"]["k"]
    P.CONCEPTS = CONCEPTS
    tr, _, fin = split_concepts(seed=0)
    idx_tr = [CONCEPTS.index(c) for c in tr]
    print(f"A (Llama-3B) cloud health: {cloud_health(RA)}", flush=True)
    chance = 1.0 / len(fin)

    targets = [("gemma-2-2b-it", "google/gemma-2-2b-it", [6, 8, 10, 12, 14]),
               ("Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", [11])]
    report = {"A_cloud": cloud_health(RA), "chance": round(chance, 4)}
    for name, hf, layers in targets:
        tok, model = load_model(hf)
        with torch.no_grad():
            pts_multi, _ = extract_multi(model, tok, layers)   # all candidate layers, one pass
        del model; torch.cuda.empty_cache()
        rows = {}
        for L in layers:
            ptsB = pts_multi[L]
            RB = np.array([ptsB[c] for c in CONCEPTS])
            tm = TransferMap.fit(RA[idx_tr], RB[idx_tr], k=kstar)
            tri = np.triu_indices(len(CONCEPTS), 1)
            rsa = float(np.corrcoef(np.linalg.norm(RA[:, None]-RA[None], axis=2)[tri],
                                    np.linalg.norm(RB[:, None]-RB[None], axis=2)[tri])[0, 1])
            rd = read_at(ptsA, ptsB, tm, fin)
            rows[L] = {"read": round(rd, 4), "x_chance": round(rd / chance, 1), "rsa": round(rsa, 3),
                       "cloud": cloud_health(RB)}
            print(f"  {name} L={L}: read={rd:.4f} ({rd/chance:.0f}x) RSA={rsa:+.3f} cloud={rows[L]['cloud']}", flush=True)
        report[name] = rows
    (HERE / "read_verify_result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nVERIFY_DONE -> read_verify_result.json")


if __name__ == "__main__":
    main()
