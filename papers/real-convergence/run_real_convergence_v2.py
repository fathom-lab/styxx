# -*- coding: utf-8 -*-
"""
run_real_convergence_v2.py — EXPLORATORY follow-up to the pre-registered null.

The pre-registered run (run_real_convergence.py) returned NOT CONVERGENT (mean RSA 0.075),
BUT its positive control failed: within-model category structure was weak (cat_struct ~1.1-1.2),
i.e. each model barely separated the known semantic categories. A null under a failed positive
control is uninformative (cf. the GW-aligner artifact). This follow-up STRENGTHENS THE
MEASUREMENT and re-tests, with the positive control front and center:

  (1) Contextual templates ending in the concept word (decoder LMs represent bare words poorly),
      last-token hidden state, averaged over templates -> a contextual concept representation.
  (2) Per-model BEST LAYER chosen by within-model category separation (cat_struct) -- a model-
      INTERNAL criterion that never looks at cross-model agreement, so it cannot inflate the
      convergence measure (it only gives each model its best shot at expressing concept geometry).

DISCIPLINE: this is EXPLORATORY / post-hoc, NOT the pre-registered test. The pre-registered null
STANDS as the pre-registered result regardless of what this finds. Reported as: "the pre-reg null
was measurement-limited" iff (a) cat_struct now strong AND (b) RSA now high; otherwise the null
is robust to a much stronger measurement (a more interesting negative).
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

MODELS = [
    ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "qwen"),
    ("Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct", "qwen"),
    ("Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct", "llama"),
    ("Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct", "llama"),
    ("Phi-3.5-mini", "microsoft/Phi-3.5-mini-instruct", "phi"),
    ("gemma-2-2b", "google/gemma-2-2b-it", "gemma"),
]

# carrier templates that END in the concept and do NOT name its category (no leakage). Frozen.
TEMPLATES = [
    "{w}", "a {w}", "the {w}", "I saw a {w}", "there is a {w}",
    "this is a {w}", "they showed me the {w}", "look at that {w}",
]

from run_real_convergence import CONCEPTS, CAT_OF, N, IU, is_cached, distmat, cat_structure  # reuse frozen set


@torch.no_grad()
def concept_all_layers(mdl, tok, word):
    """(L, d): last-token hidden state per layer, averaged over templates."""
    accum = None
    for t in TEMPLATES:
        ids = tok(t.format(w=word), return_tensors="pt").input_ids.to(mdl.device)
        out = mdl(input_ids=ids, output_hidden_states=True, use_cache=False)
        hs = torch.stack([h[0, -1] for h in out.hidden_states], 0).float().cpu().numpy()
        accum = hs if accum is None else accum + hs
    return accum / len(TEMPLATES)


def main():
    best_reps, fixed_reps, fam, info = {}, {}, {}, {}
    for name, repo, family in MODELS:
        if not is_cached(repo):
            print(f"  (skip {name})", flush=True)
            continue
        tok = AutoTokenizer.from_pretrained(repo)
        mdl = AutoModelForCausalLM.from_pretrained(
            repo, torch_dtype=torch.float16, trust_remote_code=True
        ).to(DEV).eval()
        L = mdl.config.num_hidden_layers
        allrep = np.stack([concept_all_layers(mdl, tok, w) for w in CONCEPTS])  # (N, L+1, d)
        # per-layer within-model category separation; pick the best layer (model-internal)
        cs_by_layer = [cat_structure(distmat(allrep[:, l, :])) for l in range(allrep.shape[1])]
        best_l = int(np.argmax(cs_by_layer))
        fixed_l = max(1, int(0.66 * L))
        best_reps[name] = allrep[:, best_l, :]
        fixed_reps[name] = allrep[:, fixed_l, :]
        fam[name] = family
        info[name] = {"best_layer": best_l, "n_layers": L, "best_cat_struct": round(cs_by_layer[best_l], 3),
                      "fixed_layer": fixed_l, "fixed_cat_struct": round(cs_by_layer[fixed_l], 3)}
        print(f"{name:14s}: best layer {best_l}/{L} cat_struct={cs_by_layer[best_l]:.2f} "
              f"(fixed {fixed_l} -> {cs_by_layer[fixed_l]:.2f})", flush=True)
        del mdl, tok, allrep
        gc.collect()
        if DEV == "cuda":
            torch.cuda.empty_cache()

    models = list(best_reps.keys())

    def rsa_summary(reps):
        Ds = {m: distmat(reps[m]) for m in models}
        pairs = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b = models[i], models[j]
                r = float(np.corrcoef(Ds[a][IU], Ds[b][IU])[0, 1])
                pairs.append({"a": a, "b": b, "rsa": round(r, 3), "cross_family": fam[a] != fam[b]})
        mean = float(np.mean([p["rsa"] for p in pairs]))
        xf = [p["rsa"] for p in pairs if p["cross_family"]]
        sf = [p["rsa"] for p in pairs if not p["cross_family"]]
        rng = np.random.default_rng(0)
        ctrl = []
        for _ in range(50):
            perm = rng.permutation(N)
            ctrl.append(float(np.corrcoef(Ds[models[0]][IU], distmat(reps[models[1]][perm])[IU])[0, 1]))
        return {"mean": round(mean, 3), "xfam": round(float(np.mean(xf)), 3),
                "samefam": round(float(np.mean(sf)), 3) if sf else None,
                "ctrl_mean": round(float(np.mean(ctrl)), 3), "ctrl_sd": round(float(np.std(ctrl)), 3),
                "pairs": sorted(pairs, key=lambda p: -p["rsa"])}

    best = rsa_summary(best_reps)
    fixed = rsa_summary(fixed_reps)

    pos_ctrl_ok = all(info[m]["best_cat_struct"] > 1.5 for m in models)
    convergent = (best["mean"] >= 0.30) and (best["xfam"] >= 0.30) and (best["mean"] >= 5 * max(abs(best["ctrl_mean"]), 0.02))
    if pos_ctrl_ok and convergent:
        reading = (f"CONVERGENT under a strengthened, positive-control-validated measurement: "
                   f"cross-model RSA {best['mean']:.2f} (cross-family {best['xfam']:.2f}) vs control "
                   f"{best['ctrl_mean']:.2f}. The pre-registered null was MEASUREMENT-LIMITED (bare "
                   f"words); real LLMs DO share concept geometry. [EXPLORATORY]")
    elif pos_ctrl_ok and not convergent:
        reading = (f"ROBUST NULL: even with strong within-model structure (cat_struct now ok) and "
                   f"contextual best-layer reps, cross-model RSA is only {best['mean']:.2f} "
                   f"(cross-family {best['xfam']:.2f}). Convergence does NOT replicate on this zoo/"
                   f"measure -- a more interesting negative. [EXPLORATORY]")
    else:
        reading = (f"INCONCLUSIVE: positive control still weak (best cat_struct < 1.5 for some "
                   f"model); this measurement still cannot reliably read concept geometry, so the "
                   f"RSA ({best['mean']:.2f}) remains uninformative. [EXPLORATORY]")

    out = {"status": "EXPLORATORY (pre-registered null stands)", "models": models, "per_model": info,
           "best_layer_rsa": best, "fixed_layer_rsa": fixed,
           "positive_control_ok": pos_ctrl_ok, "convergent": convergent, "reading": reading}
    (HERE / "real_convergence_v2_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== best-layer cross-model RSA (contextual templates) ===")
    for p in best["pairs"]:
        print(f"  {p['a']:14s} <-> {p['b']:14s}  RSA={p['rsa']:+.3f}  [{'xfam' if p['cross_family'] else 'same'}]")
    print(f"\nbest-layer mean RSA : {best['mean']:.3f}  (xfam {best['xfam']:.3f}, same {best['samefam']})  ctrl {best['ctrl_mean']:.3f}+-{best['ctrl_sd']:.3f}")
    print(f"fixed-layer mean RSA: {fixed['mean']:.3f}  (xfam {fixed['xfam']:.3f})  ctrl {fixed['ctrl_mean']:.3f}")
    print(f"positive control ok : {pos_ctrl_ok}")
    print(f"\n>>> {reading}")
    print("wrote real_convergence_v2_result.json")


if __name__ == "__main__":
    main()
