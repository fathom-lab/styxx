# -*- coding: utf-8 -*-
"""
run_ai_human_reliability.py — the decisive control on H2 ("convergence => humanness").

H2 passed numerically (Spearman 0.81) but is confounded: a model with a noisy/degenerate concept
RDM correlates poorly with EVERYTHING (other models AND humans), so convergence and human-alignment
co-vary through a third factor = RDM QUALITY/RELIABILITY, not a special "consensus is the human
structure" link. H3 (consensus more human than median individual) already FAILED, hinting the strong
claim is an artifact. This nails it: estimate each model's split-half RDM reliability (over template
halves) and partial it out of the convergence<->human relationship.

  - if rho(convergence, human | reliability) collapses to ~0  -> H2 was a geometry-QUALITY effect.
  - if it survives                                            -> convergence is human-specific (real).
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

import sys
sys.path.insert(0, str(HERE.parent / "real-convergence"))
from run_real_convergence import distmat, is_cached
from run_real_convergence_v2 import TEMPLATES
from run_real_convergence_v3_controls import partial_corr
import run_real_convergence as RC
import run_real_convergence_confirm as RCC
from run_ai_human import COHORT, rankdata, spearman, partial_spearman


def partial_spearman_multi(x, y, Zcols):
    rx, ry = rankdata(x), rankdata(y)
    Z = np.column_stack([np.ones(len(rx))] + [rankdata(z) for z in Zcols])
    rxr = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ryr = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(rxr, ryr)[0, 1])


@torch.no_grad()
def per_template_final(mdl, tok, CONCEPTS):
    arr = []
    for w in CONCEPTS:
        ts = []
        for t in TEMPLATES:
            ids = tok(t.format(w=w), return_tensors="pt").input_ids.to(mdl.device)
            ts.append(mdl(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[-1][0, -1].float().cpu().numpy())
        arr.append(np.stack(ts))
    return np.stack(arr)  # (N, T, d)


def main():
    vice = np.load(DATA / "final_embedding.npy")
    rows = (DATA / "things_concepts.tsv").read_text(encoding="utf-8").splitlines()[1:]
    tindex = {r.split("\t")[0].strip().lower(): i for i, r in enumerate(rows)}
    mine = list(RC.CONCEPTS) + list(RCC.CONCEPTS)
    CONCEPTS, hrows = [], []
    for w in mine:
        if w.lower() in tindex:
            CONCEPTS.append(w); hrows.append(tindex[w.lower()])
    N = len(CONCEPTS); IU = np.triu_indices(N, 1)
    human_RDM = distmat(vice[hrows])

    charlen = np.array([len(w) for w in CONCEPTS], float)
    rtok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    toklen = np.array([len(rtok(" " + w, add_special_tokens=False).input_ids) for w in CONCEPTS], float)
    zc = (charlen - charlen.mean()) / (charlen.std() + 1e-9); zt = (toklen - toklen.mean()) / (toklen.std() + 1e-9)
    Zlex = np.column_stack([np.abs(zc[:, None] - zc[None, :])[IU], np.abs(zt[:, None] - zt[None, :])[IU]])
    pal = lambda A, B: partial_corr(A[IU], B[IU], Zlex)

    full, halfA, halfB, meta = {}, {}, {}, {}
    for name, repo, params, instruct in COHORT:
        if not is_cached(repo):
            continue
        try:
            tok = AutoTokenizer.from_pretrained(repo)
            mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, trust_remote_code=True).to(DEV).eval()
            A = per_template_final(mdl, tok, CONCEPTS)  # (N,T,d)
            full[name] = distmat(A.mean(1))
            halfA[name] = distmat(A[:, ::2].mean(1)); halfB[name] = distmat(A[:, 1::2].mean(1))
            meta[name] = {"params": params, "log10_params": float(np.log10(params)), "instruct": instruct}
            del mdl, tok, A; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
            print(f"  {name:12s} ok", flush=True)
        except Exception as e:
            print(f"  {name:12s} FAIL {type(e).__name__}", flush=True)

    models = list(full.keys())
    per = {}
    for m in models:
        per[m] = {"human": round(pal(full[m], human_RDM), 3),
                  "convergence": round(float(np.mean([pal(full[m], full[o]) for o in models if o != m])), 3),
                  "reliability": round(pal(halfA[m], halfB[m]), 3), **meta[m]}

    conv = [per[m]["convergence"] for m in models]
    hum = [per[m]["human"] for m in models]
    rel = [per[m]["reliability"] for m in models]
    lp = [per[m]["log10_params"] for m in models]

    rho_raw = spearman(conv, hum)
    rho_rel = partial_spearman(conv, hum, rel)                 # control reliability
    rho_rel_size = partial_spearman_multi(conv, hum, [rel, lp])  # control reliability + size
    rho_hum_rel = spearman(hum, rel)                           # does human track reliability?
    rho_conv_rel = spearman(conv, rel)

    # H3 revisited: consensus vs best single, and a strong-models-only consensus
    consensus = np.mean([full[m] for m in models], axis=0)
    cons_human = pal(consensus, human_RDM)
    best_single = max(hum)
    strong = [m for m in models if per[m]["reliability"] >= 0.5]
    cons_strong = pal(np.mean([full[m] for m in strong], axis=0), human_RDM) if strong else None

    verdict = ("QUALITY ARTIFACT: convergence->human collapses once RDM reliability is removed "
               f"(raw {rho_raw:.2f} -> partial(reliability) {rho_rel:.2f} -> partial(reliability+size) {rho_rel_size:.2f}). "
               "The convergence<->human link is general geometry quality, NOT a human-specific consensus."
               if abs(rho_rel) < 0.3 else
               f"SURVIVES: convergence still predicts human-alignment after removing reliability "
               f"(raw {rho_raw:.2f} -> partial {rho_rel:.2f}); convergence is human-specific.")

    out = {"n_concepts": N, "models": models, "per_model": per,
           "rho_conv_human_raw": round(rho_raw, 3), "rho_partial_reliability": round(rho_rel, 3),
           "rho_partial_reliability_and_size": round(rho_rel_size, 3),
           "rho_human_reliability": round(rho_hum_rel, 3), "rho_conv_reliability": round(rho_conv_rel, 3),
           "consensus_human": round(cons_human, 3), "best_single_human": round(best_single, 3),
           "consensus_strong_only_human": round(cons_strong, 3) if cons_strong is not None else None,
           "verdict": verdict}
    (HERE / "ai_human_reliability_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== per-model (final layer, partial-lexical) ===")
    for m in sorted(models, key=lambda m: -per[m]["human"]):
        p = per[m]
        print(f"  {m:12s} {p['params']/1e6:7.0f}M  human={p['human']:+.3f}  conv={p['convergence']:+.3f}  reliability={p['reliability']:+.3f}")
    print(f"\nconv->human:  raw {rho_raw:+.3f}  | partial(reliability) {rho_rel:+.3f}  | partial(reliability+size) {rho_rel_size:+.3f}")
    print(f"human<->reliability {rho_hum_rel:+.3f}   conv<->reliability {rho_conv_rel:+.3f}")
    print(f"consensus->human {cons_human:.3f}  (best single {best_single:.3f}; strong-only consensus {cons_strong})")
    print(f"\n>>> {verdict}")
    print("wrote ai_human_reliability_result.json")


if __name__ == "__main__":
    main()
