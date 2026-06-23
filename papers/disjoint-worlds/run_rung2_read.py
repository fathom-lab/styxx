"""Rung 2 READ sweep — lean + memory-safe for 8 GB.

The committed run_g0_stage1.py stalls cross-family on 8 GB because (a) `extract` runs WITHOUT
torch.no_grad() (11k forwards x full hidden_states build autograd graphs), and (b) it re-extracts
model A (Llama-3B, 6 GB) and co-resides it with each LARGER cross-family target (gemma 5 GB / Phi
7.6 GB) -> OOM/offload thrash.

This driver computes the READ side only (R2-READ + R2-ISOMETRY = the headline: does meaning transfer
across families?) with two fixes: extract A ONCE and cache it (A is always Llama-3B), and load each
target B-only under no_grad. Reuses P.extract / TransferMap / split_concepts / CONCEPTS unchanged —
same measurement, just memory-safe orchestration. WRITE (steer_gain) is the separate stage1b sweep.
"""
from __future__ import annotations

import json
import sys
import time
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
from run_g0clear import CONCEPTS, split_concepts
SRC = "meta-llama/Llama-3.2-3B-Instruct"
CACHE_A = HERE / "_rung2_ptsA_vecsA.npz"
TARGETS = [
    ("Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("gemma-2-2b-it", "google/gemma-2-2b-it"),
    ("Phi-3.5-mini-instruct", "microsoft/Phi-3.5-mini-instruct"),
]


def distmat(R):
    sq = np.sum(R * R, 1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * R @ R.T
    return np.sqrt(np.maximum(d2, 0.0))


def extract_no_grad(model, tok, layer):
    with torch.no_grad():
        return P.extract(model, tok, layer)


def main():
    s0 = json.loads((HERE / "g0clear_result_llama3b.json").read_text(encoding="utf-8"))
    Lstar, kstar = s0["locked"]["layer"], s0["locked"]["k"]
    nlA = AutoConfig.from_pretrained(SRC).num_hidden_layers
    frac = Lstar / nlA
    P.CONCEPTS = CONCEPTS
    tr, sel, fin = split_concepts(seed=0)
    print(f"G0 locked layer_A={Lstar} (frac {frac:.3f}), k={kstar}; {len(CONCEPTS)} concepts, {len(fin)} held-out", flush=True)

    # ---- model A extracted ONCE, cached ----
    if CACHE_A.exists():
        z = np.load(CACHE_A, allow_pickle=True)
        ptsA = {c: z["pts"][i] for i, c in enumerate(CONCEPTS)}
        vecsA = {c: z["vecs"][i] for i, c in enumerate(CONCEPTS)}
        print(f"loaded cached A extraction ({CACHE_A.name})", flush=True)
    else:
        t0 = time.time()
        tokA, mA = load_model(SRC)
        ptsA, vecsA = extract_no_grad(mA, tokA, Lstar)
        del mA
        torch.cuda.empty_cache()
        np.savez(CACHE_A, pts=np.array([ptsA[c] for c in CONCEPTS]),
                 vecs=np.array([vecsA[c] for c in CONCEPTS]))
        print(f"extracted + cached A in {time.time()-t0:.0f}s", flush=True)
    RA = np.array([ptsA[c] for c in CONCEPTS])
    idx_tr = [CONCEPTS.index(c) for c in tr]

    results = {}
    for name, hf in TARGETS:
        out = HERE / f"rung2_read_result_{name.replace('.', '_').replace('-', '_')}.json"
        if out.exists():
            results[name] = json.loads(out.read_text(encoding="utf-8"))
            print(f">> {name}: already banked (RSA={results[name].get('rsa')}, read={results[name].get('read_top1')}) — skip", flush=True)
            continue
        t0 = time.time()
        nlB = AutoConfig.from_pretrained(hf).num_hidden_layers
        LB = round(frac * nlB)
        tokB, mB = load_model(hf)
        ptsB, vecsB = extract_no_grad(mB, tokB, LB)
        del mB
        torch.cuda.empty_cache()
        RB = np.array([ptsB[c] for c in CONCEPTS])
        tri = np.triu_indices(len(CONCEPTS), 1)
        rsa = float(np.corrcoef(distmat(RA)[tri], distmat(RB)[tri])[0, 1])
        tm = TransferMap.fit(RA[idx_tr], RB[idx_tr], k=kstar)
        fin_ptsB = np.array([ptsB[c] for c in fin])
        read_hits = sum(1 for i, c in enumerate(fin)
                        if int(np.argmin(np.linalg.norm(fin_ptsB - tm.transfer_point(ptsA[c]), axis=1))) == i)
        read_top1 = read_hits / len(fin)
        read_chance = 1.0 / len(fin)
        results[name] = {"hf": hf, "layer_B": LB, "rsa": round(rsa, 3),
                         "read_top1": round(read_top1, 4), "read_chance": round(read_chance, 4),
                         "read_x_chance": round(read_top1 / read_chance, 1)}
        out = HERE / f"rung2_read_result_{name.replace('.', '_').replace('-', '_')}.json"
        out.write_text(json.dumps(results[name], indent=2) + "\n", encoding="utf-8")
        print(f">> {name}: RSA={rsa:+.3f} READ top1={read_top1:.3f} ({read_top1/read_chance:.0f}x chance "
              f"{read_chance:.3f}) layer_B={LB}  [{time.time()-t0:.0f}s]", flush=True)

    (HERE / "rung2_read_summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("\nALL_READ_DONE")
    # quick isometry curve incl. same-family anchor from Rung 1b
    print("isometry->read curve (incl. Llama-1B anchor RSA 0.946 read 0.586):")
    for nm, r in results.items():
        print(f"  {nm:24s} RSA={r['rsa']:+.3f} read={r['read_top1']:.3f}")


if __name__ == "__main__":
    main()
