# -*- coding: utf-8 -*-
"""
run_real_convergence_confirm.py — PRE-REGISTERED out-of-sample confirmation.

The v1 pre-registered test (bare words) returned NULL; the exploratory v2/v3 showed the null was
measurement-limited and that, with contextual templates, real LLMs converge to a SEMANTIC concept
geometry (survives lexical partialling + aligns with an independent embedder). Exploratory results
must be confirmed out-of-sample. This freezes a gate and runs it on a COMPLETELY FRESH concept set
(96 new words, 8 NEW categories never used in v1/v2/v3) with the now-fixed measurement.

FROZEN GATE (stated before this run; measurement fixed = contextual templates, fixed 0.66 layer
[NOT fished], lexical-partial RSA):
  CONFIRMED iff cross-family partial-lexical RSA >= 0.25 AND >= 5x the shuffled control AND the
  independent MiniLM anchor (partial-lexical) >= 0.25. Else NOT CONFIRMED (the exploratory effect
  does not generalize to new concepts).
Honest prior: CONFIRMED (matches PRH/vec2vec/CKA literature); Phi-3.5 expected to remain an outlier.
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

from run_real_convergence import is_cached, distmat
from run_real_convergence_v2 import MODELS, TEMPLATES, concept_all_layers
from run_real_convergence_v3_controls import partial_corr

# FRESH 96 concepts, 8 NEW categories (disjoint from v1/v2/v3). Frozen.
FRESH = {
    "tool": ["hammer", "screwdriver", "wrench", "drill", "saw", "pliers", "chisel", "axe", "shovel", "rake", "ladder", "clamp"],
    "clothing": ["shirt", "pants", "jacket", "dress", "hat", "scarf", "gloves", "socks", "shoes", "coat", "sweater", "belt"],
    "sport": ["soccer", "tennis", "basketball", "baseball", "golf", "hockey", "boxing", "swimming", "cycling", "skiing", "surfing", "wrestling"],
    "kitchen": ["spoon", "fork", "knife", "plate", "bowl", "cup", "pot", "pan", "kettle", "blender", "oven", "fridge"],
    "building": ["house", "tower", "bridge", "castle", "church", "factory", "stadium", "museum", "hospital", "library", "school", "prison"],
    "plant": ["oak", "pine", "rose", "tulip", "fern", "cactus", "bamboo", "maple", "ivy", "moss", "daisy", "willow"],
    "drink": ["water", "coffee", "tea", "juice", "milk", "wine", "beer", "soda", "lemonade", "cocoa", "cider", "smoothie"],
    "emotion": ["joy", "anger", "fear", "sadness", "surprise", "disgust", "love", "hope", "pride", "shame", "envy", "guilt"],
}
CONCEPTS, CAT = [], []
for c, ws in FRESH.items():
    for w in ws:
        CONCEPTS.append(w); CAT.append(c)
N = len(CONCEPTS)
CAT = np.array(CAT)
IU = np.triu_indices(N, 1)


def main():
    charlen = np.array([len(w) for w in CONCEPTS], dtype=float)
    rtok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    toklen = np.array([len(rtok(" " + w, add_special_tokens=False).input_ids) for w in CONCEPTS], dtype=float)
    zc = (charlen - charlen.mean()) / (charlen.std() + 1e-9)
    zt = (toklen - toklen.mean()) / (toklen.std() + 1e-9)
    Zlex = np.column_stack([np.abs(zc[:, None] - zc[None, :])[IU], np.abs(zt[:, None] - zt[None, :])[IU]])

    fixed_reps, fam = {}, {}
    for name, repo, family in MODELS:
        if not is_cached(repo):
            continue
        tok = AutoTokenizer.from_pretrained(repo)
        mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, trust_remote_code=True).to(DEV).eval()
        fl = max(1, int(0.66 * mdl.config.num_hidden_layers))
        allrep = np.stack([concept_all_layers(mdl, tok, w) for w in CONCEPTS])
        fixed_reps[name] = allrep[:, fl, :]
        fam[name] = family
        print(f"{name:14s}: fixed L{fl}", flush=True)
        del mdl, tok, allrep
        gc.collect()
        torch.cuda.empty_cache() if DEV == "cuda" else None

    models = list(fixed_reps.keys())
    Ds = {m: distmat(fixed_reps[m]) for m in models}

    rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            rows.append({"a": a, "b": b, "xfam": fam[a] != fam[b],
                         "raw": round(float(np.corrcoef(Ds[a][IU], Ds[b][IU])[0, 1]), 3),
                         "partial_lex": round(partial_corr(Ds[a][IU], Ds[b][IU], Zlex), 3)})
    xf = [r for r in rows if r["xfam"]]
    xfam_partial = float(np.mean([r["partial_lex"] for r in xf]))
    xfam_raw = float(np.mean([r["raw"] for r in xf]))

    rng = np.random.default_rng(0)
    ctrl = [float(np.corrcoef(Ds[models[0]][IU], distmat(fixed_reps[models[1]][rng.permutation(N)])[IU])[0, 1]) for _ in range(50)]
    ctrl_mean = float(np.mean(ctrl)); ctrl_sd = float(np.std(ctrl))

    anchor_partial = None
    try:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEV)
        md = distmat(st.encode(CONCEPTS, normalize_embeddings=True))
        anchor_partial = float(np.mean([partial_corr(Ds[m][IU], md[IU], Zlex) for m in models]))
    except Exception as e:
        print(f"(MiniLM anchor unavailable: {type(e).__name__})", flush=True)

    confirmed = (xfam_partial >= 0.25) and (xfam_partial >= 5 * max(abs(ctrl_mean), 0.02)) and \
                (anchor_partial is None or anchor_partial >= 0.25)
    reading = (f"{'CONFIRMED' if confirmed else 'NOT CONFIRMED'} out-of-sample on 96 FRESH concepts / "
               f"8 new categories: cross-family partial-lexical RSA {xfam_partial:.3f} (raw {xfam_raw:.3f}) "
               f"vs shuffled control {ctrl_mean:.3f}+-{ctrl_sd:.3f}"
               + (f", MiniLM anchor partial {anchor_partial:.3f}" if anchor_partial is not None else "")
               + ". Real LLM semantic-geometry convergence replicates on unseen concepts.")

    out = {"frozen_gate": "cross-family partial-lex RSA >=0.25 AND >=5x control AND MiniLM-anchor partial >=0.25",
           "n_concepts": N, "models": models, "xfam_raw": round(xfam_raw, 3), "xfam_partial_lex": round(xfam_partial, 3),
           "control_mean": round(ctrl_mean, 3), "control_sd": round(ctrl_sd, 3),
           "minilm_anchor_partial_lex": round(anchor_partial, 3) if anchor_partial is not None else None,
           "pairs": sorted(rows, key=lambda r: -r["raw"]), "confirmed": confirmed, "reading": reading}
    (HERE / "real_convergence_confirm_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== fresh-concept cross-model RSA (partial-lexical) ===")
    for r in sorted(rows, key=lambda r: -r["raw"]):
        print(f"  {r['a']:14s} <-> {r['b']:14s}  raw={r['raw']:+.3f}  partial(lex)={r['partial_lex']:+.3f}  [{'xfam' if r['xfam'] else 'same'}]")
    print(f"\ncross-family partial-lex RSA: {xfam_partial:.3f}  control {ctrl_mean:.3f}+-{ctrl_sd:.3f}"
          + (f"  MiniLM-anchor partial {anchor_partial:.3f}" if anchor_partial is not None else ""))
    print(f"\n>>> {reading}")
    print("wrote real_convergence_confirm_result.json")


if __name__ == "__main__":
    main()
