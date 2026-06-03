# -*- coding: utf-8 -*-
"""
run_scale_law_v2.py — robustness of the scale-law NULL to layer choice.

run_scale_law.py (fixed 0.66 layer) returned NOT CONFIRMED: pooled Spearman rho=-0.573, and the
perfectly-controlled Pythia ladder ran the WRONG way (rho=-0.8). Before accepting "scale does not
drive convergence", rule out the obvious threat to validity: a single fixed layer may handicap
larger models if their best semantic layer sits at a different relative depth. This sweeps ALL
layers and re-tests the scale trend under three layer rules:
  (a) fixed 0.66 (the pre-registered measure),
  (b) best-by-cat_struct (model-internal criterion, not the reference -> not circular),
  (c) max-over-layers alignment (optimistic upper bound; uses the reference, so circular for the
      LEVEL but a fair test of whether ANY layer of a bigger model aligns better -> kills the
      "wrong fixed layer" confound for the TREND).
If the scale trend is non-positive (esp. on Pythia) under ALL three rules, the null is robust.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"

from run_real_convergence import distmat, is_cached
from run_real_convergence_v2 import concept_all_layers
from run_real_convergence_v3_controls import partial_corr
from run_scale_law import LADDERS, spearman
import run_real_convergence as RC
import run_real_convergence_confirm as RCC

CONCEPTS = list(RC.CONCEPTS) + list(RCC.CONCEPTS)
CATS = np.concatenate([np.asarray(RC.CAT_OF), np.asarray(RCC.CAT)])
N = len(CONCEPTS)
IU = np.triu_indices(N, 1)
EYE = np.eye(N, dtype=bool)
SAME = CATS[:, None] == CATS[None, :]


def cat_struct(D):
    return float(D[~SAME].mean() / (D[SAME & ~EYE].mean() + 1e-9))


def main():
    charlen = np.array([len(w) for w in CONCEPTS], dtype=float)
    rtok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    toklen = np.array([len(rtok(" " + w, add_special_tokens=False).input_ids) for w in CONCEPTS], dtype=float)
    zc = (charlen - charlen.mean()) / (charlen.std() + 1e-9)
    zt = (toklen - toklen.mean()) / (toklen.std() + 1e-9)
    Zlex = np.column_stack([np.abs(zc[:, None] - zc[None, :])[IU], np.abs(zt[:, None] - zt[None, :])[IU]])

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEV)
    ref = distmat(st.encode(CONCEPTS, normalize_embeddings=True))
    refIU = ref[IU]
    del st
    gc.collect()
    torch.cuda.empty_cache() if DEV == "cuda" else None

    results = []
    for ladder, members in LADDERS.items():
        for name, repo, params in members:
            if not is_cached(repo):
                continue
            tok = AutoTokenizer.from_pretrained(repo)
            mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, trust_remote_code=True).to(DEV).eval()
            L = mdl.config.num_hidden_layers
            allrep = np.stack([concept_all_layers(mdl, tok, w) for w in CONCEPTS])  # (N, L+1, d)
            per_layer_align, per_layer_cs = [], []
            for l in range(allrep.shape[1]):
                Dl = distmat(allrep[:, l, :])
                per_layer_align.append(partial_corr(Dl[IU], refIU, Zlex))
                per_layer_cs.append(cat_struct(Dl))
            fl = max(1, int(0.66 * L))
            bl = int(np.argmax(per_layer_cs))
            ml = int(np.argmax(per_layer_align))
            results.append({"ladder": ladder, "name": name, "params": params,
                            "log10_params": round(float(np.log10(params)), 3),
                            "fixed": round(per_layer_align[fl], 3),
                            "best_catstruct": round(per_layer_align[bl], 3),
                            "max_layer": round(per_layer_align[ml], 3),
                            "best_layer_idx": bl, "max_layer_idx": ml, "n_layers": L})
            print(f"{ladder:8s} {name:12s} ({params/1e6:6.0f}M): fixed={per_layer_align[fl]:+.3f} "
                  f"best_cs={per_layer_align[bl]:+.3f} max={per_layer_align[ml]:+.3f} (maxL {ml}/{L})", flush=True)
            del mdl, tok, allrep
            gc.collect()
            torch.cuda.empty_cache() if DEV == "cuda" else None

    def trends(key):
        per = {}
        for ladder in LADDERS:
            rows = [r for r in results if r["ladder"] == ladder]
            per[ladder] = spearman([r["log10_params"] for r in rows], [r[key] for r in rows]) if len(rows) >= 3 else None
        pooled = spearman([r["log10_params"] for r in results], [r[key] for r in results])
        return {"pooled": round(pooled, 3), "per_ladder": {k: (round(v, 3) if v is not None else None) for k, v in per.items()}}

    T = {rule: trends(rule) for rule in ["fixed", "best_catstruct", "max_layer"]}

    # robust null iff scale trend is non-positive on Pythia under ALL rules AND pooled never >= 0.5
    pythia_nonpos = all((T[r]["per_ladder"].get("Pythia") or 0) <= 0.0 for r in T)
    pooled_never_high = all(T[r]["pooled"] < 0.5 for r in T)
    robust_null = pythia_nonpos and pooled_never_high
    if robust_null:
        reading = (f"NULL IS ROBUST to layer choice. Under all three layer rules the pooled scale trend stays "
                   f"below the +0.50 gate (fixed {T['fixed']['pooled']}, best {T['best_catstruct']['pooled']}, "
                   f"max {T['max_layer']['pooled']}) and the perfectly-controlled Pythia ladder is non-positive "
                   f"throughout (fixed {T['fixed']['per_ladder']['Pythia']}, best {T['best_catstruct']['per_ladder']['Pythia']}, "
                   f"max {T['max_layer']['per_ladder']['Pythia']}). Within 14M-3B, scale does NOT drive semantic "
                   f"convergence; the heterogeneity is family/data-driven. The fixed-layer result was not an artifact.")
    else:
        reading = (f"Layer choice MATTERS: under some rule the scale trend turns positive (pooled fixed "
                   f"{T['fixed']['pooled']} / best {T['best_catstruct']['pooled']} / max {T['max_layer']['pooled']}; "
                   f"Pythia {T['fixed']['per_ladder']['Pythia']}/{T['best_catstruct']['per_ladder']['Pythia']}/"
                   f"{T['max_layer']['per_ladder']['Pythia']}). The fixed-layer null is partly a measurement choice; "
                   f"reported as such.")

    out = {"n_concepts": N, "trends": T, "models": results, "robust_null": robust_null, "reading": reading}
    (HERE / "scale_law_v2_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== scale trend (Spearman rho vs log params) under three layer rules ===")
    for rule in ["fixed", "best_catstruct", "max_layer"]:
        pl = T[rule]["per_ladder"]
        print(f"  {rule:14s}: pooled {T[rule]['pooled']:+.3f}  | Pythia {pl['Pythia']}  GPT-2 {pl['GPT-2']}  Qwen2.5 {pl['Qwen2.5']}")
    print(f"\n>>> {reading}")
    print("wrote scale_law_v2_result.json")


if __name__ == "__main__":
    main()
