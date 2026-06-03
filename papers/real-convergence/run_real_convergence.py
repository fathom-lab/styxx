# -*- coding: utf-8 -*-
"""
run_real_convergence.py — Do REAL, independently-trained LLMs converge to the same concept
geometry over REAL concepts? The non-circular replacement for the synthetic disjoint-worlds
toy (which passed the same latent z to both worlds). Gate frozen in
PREREG_real_convergence_2026_06_03.md.

For each model: mean-pool the hidden state of each concept word at ~0.66 relative depth ->
a 96xd representation -> a 96x96 distance matrix. Cross-model RSA = Pearson correlation of
two models' distance matrices over concept pairs (identical words -> trivial correspondence).
Controls: shuffled-concept (must be ~0), within- vs across-category distance (structure must
be real). Reports same-family vs cross-family separately (cross-family is load-bearing).
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = [
    ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "qwen"),
    ("Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct", "qwen"),
    ("Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct", "llama"),
    ("Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct", "llama"),
    ("Phi-3.5-mini", "microsoft/Phi-3.5-mini-instruct", "phi"),
    ("gemma-2-2b", "google/gemma-2-2b-it", "gemma"),
]

# 96 real concepts, 8 clear semantic categories x 12. Frozen.
CATEGORIES = {
    "animal": ["dog", "cat", "horse", "elephant", "lion", "tiger", "bear", "wolf", "rabbit", "mouse", "eagle", "shark"],
    "fruit": ["apple", "banana", "orange", "grape", "lemon", "peach", "cherry", "mango", "melon", "plum", "strawberry", "pear"],
    "vehicle": ["car", "truck", "bus", "train", "airplane", "bicycle", "boat", "ship", "motorcycle", "helicopter", "submarine", "scooter"],
    "profession": ["doctor", "teacher", "lawyer", "engineer", "nurse", "farmer", "pilot", "chef", "artist", "soldier", "scientist", "judge"],
    "body": ["hand", "foot", "head", "eye", "ear", "nose", "mouth", "arm", "leg", "heart", "brain", "finger"],
    "weather": ["rain", "snow", "wind", "storm", "sunshine", "cloud", "fog", "thunder", "lightning", "frost", "drizzle", "hail"],
    "furniture": ["chair", "table", "bed", "sofa", "desk", "shelf", "lamp", "mirror", "cabinet", "stool", "bench", "drawer"],
    "instrument": ["guitar", "piano", "violin", "drum", "flute", "trumpet", "harp", "cello", "clarinet", "saxophone", "banjo", "organ"],
}
CONCEPTS, CAT_OF = [], []
for c, ws in CATEGORIES.items():
    for w in ws:
        CONCEPTS.append(w)
        CAT_OF.append(c)
N = len(CONCEPTS)
CAT_OF = np.array(CAT_OF)
IU = np.triu_indices(N, 1)


def is_cached(repo: str) -> bool:
    base = Path.home() / ".cache" / "huggingface" / "hub"
    return (base / ("models--" + repo.replace("/", "--"))).exists()


@torch.no_grad()
def word_rep(mdl, tok, word: str, layer: int) -> np.ndarray:
    ids = tok(" " + word, return_tensors="pt", add_special_tokens=False).input_ids
    if ids.shape[1] == 0:
        ids = tok(word, return_tensors="pt").input_ids
    ids = ids.to(mdl.device)
    out = mdl(input_ids=ids, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[layer][0]          # (T, d)
    return h.mean(0).float().cpu().numpy()   # mean-pool over the word's subword tokens


def distmat(R: np.ndarray) -> np.ndarray:
    R = R - R.mean(0)
    R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
    G = R @ R.T
    D = np.sqrt(np.maximum(2.0 - 2.0 * G, 0.0))   # cosine distance geometry
    return D


def cat_structure(D: np.ndarray) -> float:
    """mean across-category distance / mean within-category distance (>1 = structure real)."""
    same = CAT_OF[:, None] == CAT_OF[None, :]
    wm = D[same & ~np.eye(N, dtype=bool)].mean()
    am = D[~same].mean()
    return float(am / (wm + 1e-9))


def main():
    reps, fam = {}, {}
    for name, repo, family in MODELS:
        if not is_cached(repo):
            print(f"  (skip {name}: not cached)", flush=True)
            continue
        tok = AutoTokenizer.from_pretrained(repo)
        mdl = AutoModelForCausalLM.from_pretrained(
            repo, torch_dtype=torch.float16, trust_remote_code=True
        ).to(DEV).eval()
        layer = max(1, int(0.66 * mdl.config.num_hidden_layers))
        R = np.stack([word_rep(mdl, tok, w, layer) for w in CONCEPTS])
        reps[name] = R
        fam[name] = family
        cs = cat_structure(distmat(R))
        print(f"{name:14s}: reps {R.shape} layer {layer}/{mdl.config.num_hidden_layers}  cat_struct(across/within)={cs:.2f}", flush=True)
        del mdl, tok
        gc.collect()
        if DEV == "cuda":
            torch.cuda.empty_cache()

    models = list(reps.keys())
    if len(models) < 2:
        print("FATAL: need >=2 cached models")
        return
    Ds = {m: distmat(reps[m]) for m in models}

    # cross-model RSA
    pairs = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            r = float(np.corrcoef(Ds[a][IU], Ds[b][IU])[0, 1])
            pairs.append({"a": a, "b": b, "rsa": round(r, 3),
                          "cross_family": fam[a] != fam[b]})
    mean_rsa = float(np.mean([p["rsa"] for p in pairs]))
    xfam = [p["rsa"] for p in pairs if p["cross_family"]]
    samefam = [p["rsa"] for p in pairs if not p["cross_family"]]
    mean_xfam = float(np.mean(xfam)) if xfam else float("nan")
    mean_samefam = float(np.mean(samefam)) if samefam else float("nan")

    # shuffled-concept control (break the correspondence)
    rng = np.random.default_rng(0)
    ctrl = []
    a0, b0 = models[0], models[1]
    for _ in range(50):
        perm = rng.permutation(N)
        Dp = distmat(reps[b0][perm])
        ctrl.append(float(np.corrcoef(Ds[a0][IU], Dp[IU])[0, 1]))
    ctrl_mean = float(np.mean(ctrl))
    ctrl_sd = float(np.std(ctrl))

    cat_struct = {m: round(cat_structure(Ds[m]), 3) for m in models}

    # GATE
    convergent = (mean_rsa >= 0.30) and (abs(ctrl_mean) < 0.05) and (mean_rsa >= 5 * max(abs(ctrl_mean), 0.02))
    crossfam_ok = bool(np.isfinite(mean_xfam) and mean_xfam >= 0.30)
    struct_ok = all(v > 1.05 for v in cat_struct.values())
    if convergent and crossfam_ok and struct_ok:
        reading = (f"CONVERGENT (cross-family). Real, independently-trained LLMs share concept "
                   f"geometry on real meaning: mean cross-model RSA {mean_rsa:.3f} "
                   f"(cross-family {mean_xfam:.3f}, same-family {mean_samefam:.3f}) vs shuffled "
                   f"control {ctrl_mean:.3f}+-{ctrl_sd:.3f}. The non-circular version holds.")
    elif convergent and struct_ok:
        reading = (f"CONVERGENT but cross-family weak ({mean_xfam:.3f}). Shared geometry is "
                   f"driven more by same-family similarity than architecture-independent structure.")
    else:
        reading = (f"NOT CONVERGENT by the frozen gate (mean RSA {mean_rsa:.3f}, ctrl "
                   f"{ctrl_mean:.3f}, struct_ok={struct_ok}, xfam={mean_xfam:.3f}).")

    out = {
        "n_concepts": N, "n_models": len(models), "models": models,
        "mean_cross_model_rsa": round(mean_rsa, 3),
        "mean_cross_family_rsa": round(mean_xfam, 3),
        "mean_same_family_rsa": round(mean_samefam, 3),
        "shuffled_control_mean": round(ctrl_mean, 3), "shuffled_control_sd": round(ctrl_sd, 3),
        "category_structure_across_over_within": cat_struct,
        "pairs": sorted(pairs, key=lambda p: -p["rsa"]),
        "gate": {"convergent": convergent, "cross_family_ok": crossfam_ok, "structure_ok": struct_ok},
        "reading": reading,
    }
    (HERE / "real_convergence_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== cross-model RSA (real concepts, real models) ===")
    for p in out["pairs"]:
        tag = "xfam" if p["cross_family"] else "same"
        print(f"  {p['a']:14s} <-> {p['b']:14s}  RSA={p['rsa']:+.3f}  [{tag}]")
    print(f"\nmean RSA            : {mean_rsa:.3f}")
    print(f"  cross-family      : {mean_xfam:.3f}")
    print(f"  same-family       : {mean_samefam:.3f}")
    print(f"shuffled control    : {ctrl_mean:.3f} +- {ctrl_sd:.3f}")
    print(f"category structure  : {cat_struct}")
    print(f"\n>>> {reading}")
    print("wrote real_convergence_result.json")


if __name__ == "__main__":
    main()
