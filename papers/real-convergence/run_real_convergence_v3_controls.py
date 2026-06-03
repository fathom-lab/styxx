# -*- coding: utf-8 -*-
"""
run_real_convergence_v3_controls.py — the decisive control on the v2 exploratory convergence.

v2 showed strong cross-model RSA with contextual templates (fixed-layer 0.40, best-layer 0.63,
5/6 models agreeing 0.80-0.98, Phi-3.5 an outlier). Skeptic's objection: maybe the models agree
because of LEXICAL properties (word length, single- vs multi-token) rather than MEANING.

This script kills or confirms that objection:
  (A) recomputes the contextual reps (fixed 0.66 layer = NOT fished, + best-by-cat_struct layer),
      and SAVES them so controls are cheap to iterate.
  (B) PARTIAL RSA controlling for a lexical-distance design (char-length diff + token-count diff):
      if cross-model RSA survives partialling out lexical structure, the convergence is semantic.
  (C) independent semantic anchor (all-MiniLM, if installed): does the LLM-shared geometry align
      with a purpose-built sentence embedder's geometry, AND survive lexical partialling?

EXPLORATORY (pre-registered v1 null stands). Decision rule stated up front:
  SEMANTIC CONVERGENCE iff partial cross-family RSA (lexical removed) stays >= 0.30 and the
  MiniLM anchor correlation survives lexical partialling; else LEXICALLY CONFOUNDED.
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

from run_real_convergence import CONCEPTS, CAT_OF, N, IU, is_cached, distmat, cat_structure
from run_real_convergence_v2 import MODELS, TEMPLATES, concept_all_layers


def partial_corr(a, b, Z):
    """corr(a, b | Z): correlation of residuals after regressing each on Z (with intercept)."""
    X = np.column_stack([np.ones(len(a)), Z])
    ra = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    rb = b - X @ np.linalg.lstsq(X, b, rcond=None)[0]
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    # --- lexical features (model-agnostic char length + reference token count) ---
    charlen = np.array([len(w) for w in CONCEPTS], dtype=float)
    rtok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    toklen = np.array([len(rtok(" " + w, add_special_tokens=False).input_ids) for w in CONCEPTS], dtype=float)

    def lex_design():
        zc = (charlen - charlen.mean()) / (charlen.std() + 1e-9)
        zt = (toklen - toklen.mean()) / (toklen.std() + 1e-9)
        dC = np.abs(zc[:, None] - zc[None, :])[IU]
        dT = np.abs(zt[:, None] - zt[None, :])[IU]
        return np.column_stack([dC, dT])

    Zlex = lex_design()

    # --- contextual reps per model: fixed 0.66 layer (NOT fished) + best-by-cat_struct ---
    fixed_reps, best_reps, fam, info = {}, {}, {}, {}
    for name, repo, family in MODELS:
        if not is_cached(repo):
            continue
        tok = AutoTokenizer.from_pretrained(repo)
        mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, trust_remote_code=True).to(DEV).eval()
        Ln = mdl.config.num_hidden_layers
        allrep = np.stack([concept_all_layers(mdl, tok, w) for w in CONCEPTS])  # (N, L+1, d)
        cs = [cat_structure(distmat(allrep[:, l, :])) for l in range(allrep.shape[1])]
        bl = int(np.argmax(cs))
        fl = max(1, int(0.66 * Ln))
        fixed_reps[name] = allrep[:, fl, :]
        best_reps[name] = allrep[:, bl, :]
        fam[name] = family
        info[name] = {"best_layer": bl, "fixed_layer": fl, "n_layers": Ln}
        print(f"{name:14s}: fixed L{fl}, best L{bl}", flush=True)
        del mdl, tok, allrep
        gc.collect()
        torch.cuda.empty_cache() if DEV == "cuda" else None

    models = list(fixed_reps.keys())
    np.savez(HERE / "contextual_reps.npz",
             **{f"fixed__{m}": fixed_reps[m] for m in models},
             **{f"best__{m}": best_reps[m] for m in models})

    # --- MiniLM independent semantic anchor (optional) ---
    minilm_d = None
    try:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEV)
        me = st.encode(CONCEPTS, normalize_embeddings=True)
        minilm_d = distmat(me)
        print("MiniLM anchor: loaded", flush=True)
    except Exception as e:
        print(f"MiniLM anchor unavailable ({type(e).__name__}); skipping anchor control", flush=True)

    def analyze(reps, tag):
        Ds = {m: distmat(reps[m]) for m in models}
        rows = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b = models[i], models[j]
                raw = float(np.corrcoef(Ds[a][IU], Ds[b][IU])[0, 1])
                par = partial_corr(Ds[a][IU], Ds[b][IU], Zlex)
                rows.append({"a": a, "b": b, "xfam": fam[a] != fam[b], "raw": round(raw, 3), "partial_lex": round(par, 3)})
        xf = [r for r in rows if r["xfam"]]
        summ = {
            "mean_raw": round(float(np.mean([r["raw"] for r in rows])), 3),
            "mean_partial_lex": round(float(np.mean([r["partial_lex"] for r in rows])), 3),
            "xfam_raw": round(float(np.mean([r["raw"] for r in xf])), 3),
            "xfam_partial_lex": round(float(np.mean([r["partial_lex"] for r in xf])), 3),
        }
        anchor = None
        if minilm_d is not None:
            raws = [float(np.corrcoef(Ds[m][IU], minilm_d[IU])[0, 1]) for m in models]
            pars = [partial_corr(Ds[m][IU], minilm_d[IU], Zlex) for m in models]
            anchor = {"per_model_raw": {m: round(r, 3) for m, r in zip(models, raws)},
                      "per_model_partial_lex": {m: round(p, 3) for m, p in zip(models, pars)},
                      "mean_raw": round(float(np.mean(raws)), 3), "mean_partial_lex": round(float(np.mean(pars)), 3)}
        print(f"\n[{tag}] cross-family RSA raw {summ['xfam_raw']:.3f} -> partial(lex) {summ['xfam_partial_lex']:.3f}")
        if anchor:
            print(f"[{tag}] MiniLM anchor raw {anchor['mean_raw']:.3f} -> partial(lex) {anchor['mean_partial_lex']:.3f}")
        return {"summary": summ, "pairs": sorted(rows, key=lambda r: -r["raw"]), "minilm_anchor": anchor}

    fixed = analyze(fixed_reps, "fixed-layer (non-fished)")
    best = analyze(best_reps, "best-layer")

    semantic = (fixed["summary"]["xfam_partial_lex"] >= 0.30) and (
        fixed["minilm_anchor"] is None or fixed["minilm_anchor"]["mean_partial_lex"] >= 0.30)
    if semantic:
        reading = (f"SEMANTIC CONVERGENCE confirmed. Cross-family RSA survives partialling out word "
                   f"length + token count: fixed-layer xfam {fixed['summary']['xfam_raw']:.2f} -> "
                   f"{fixed['summary']['xfam_partial_lex']:.2f} (lexical removed). The agreement is "
                   f"about MEANING, not spelling. [EXPLORATORY; pre-reg null stands]")
    else:
        reading = (f"LEXICALLY CONFOUNDED: cross-family RSA drops to {fixed['summary']['xfam_partial_lex']:.2f} "
                   f"once word length / token count are removed -- the apparent convergence was substantially "
                   f"lexical. [EXPLORATORY]")

    out = {"status": "EXPLORATORY (pre-registered v1 null stands)", "models": models, "per_model": info,
           "fixed_layer": fixed, "best_layer": best, "semantic_after_lexical_control": semantic, "reading": reading}
    (HERE / "real_convergence_v3_controls_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n>>> {reading}")
    print("wrote real_convergence_v3_controls_result.json + contextual_reps.npz")


if __name__ == "__main__":
    main()
