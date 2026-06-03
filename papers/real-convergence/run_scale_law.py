# -*- coding: utf-8 -*-
"""
run_scale_law.py — Does concept-geometry convergence RISE with model scale? (falsifiable PRH test)
Gate frozen in PREREG_scale_law_2026_06_03.md.

Three CONTROLLED scale ladders (Pythia = gold: The Pile + arch + order fixed, only scale varies;
GPT-2; Qwen2.5). For each model: contextual-template last-token reps at fixed 0.66 layer over the
pooled 192 concepts -> distance matrix -> partial-lexical RSA vs an independent semantic reference
(MiniLM primary, mpnet robustness). Then Spearman rho(alignment, log10 params), per-ladder + pooled.
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
import run_real_convergence as RC
import run_real_convergence_confirm as RCC

CONCEPTS = list(RC.CONCEPTS) + list(RCC.CONCEPTS)   # pooled 192
N = len(CONCEPTS)
IU = np.triu_indices(N, 1)

LADDERS = {
    "Pythia": [("pythia-14m", "EleutherAI/pythia-14m", 14e6), ("pythia-70m", "EleutherAI/pythia-70m", 70e6),
               ("pythia-160m", "EleutherAI/pythia-160m", 160e6), ("pythia-410m", "EleutherAI/pythia-410m", 410e6)],
    "GPT-2": [("gpt2", "gpt2", 124e6), ("gpt2-medium", "gpt2-medium", 355e6),
              ("gpt2-large", "gpt2-large", 774e6), ("gpt2-xl", "gpt2-xl", 1558e6)],
    "Qwen2.5": [("Qwen-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", 494e6), ("Qwen-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", 1540e6),
                ("Qwen-3B", "Qwen/Qwen2.5-3B-Instruct", 3090e6)],
}


def spearman(x, y):
    rx = np.argsort(np.argsort(np.asarray(x, float)))
    ry = np.argsort(np.argsort(np.asarray(y, float)))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    # lexical nuisance design (char-length + reference token-count), partialled out of every alignment
    charlen = np.array([len(w) for w in CONCEPTS], dtype=float)
    rtok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    toklen = np.array([len(rtok(" " + w, add_special_tokens=False).input_ids) for w in CONCEPTS], dtype=float)
    zc = (charlen - charlen.mean()) / (charlen.std() + 1e-9)
    zt = (toklen - toklen.mean()) / (toklen.std() + 1e-9)
    Zlex = np.column_stack([np.abs(zc[:, None] - zc[None, :])[IU], np.abs(zt[:, None] - zt[None, :])[IU]])

    # independent semantic references
    from sentence_transformers import SentenceTransformer
    refs = {}
    for tag, repo in [("MiniLM", "sentence-transformers/all-MiniLM-L6-v2"),
                      ("mpnet", "sentence-transformers/all-mpnet-base-v2")]:
        st = SentenceTransformer(repo, device=DEV)
        refs[tag] = distmat(st.encode(CONCEPTS, normalize_embeddings=True))
        del st
        gc.collect()
        torch.cuda.empty_cache() if DEV == "cuda" else None
    print("references loaded:", list(refs), flush=True)

    results = []  # one row per model
    for ladder, members in LADDERS.items():
        for name, repo, params in members:
            if not is_cached(repo):
                print(f"  (skip {name}: not cached)", flush=True)
                continue
            tok = AutoTokenizer.from_pretrained(repo)
            mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, trust_remote_code=True).to(DEV).eval()
            fl = max(1, int(0.66 * mdl.config.num_hidden_layers))
            rep = np.stack([concept_all_layers(mdl, tok, w)[fl] for w in CONCEPTS])  # (N, d) at fixed layer
            D = distmat(rep)
            align = {tag: round(partial_corr(D[IU], refs[tag][IU], Zlex), 3) for tag in refs}
            results.append({"ladder": ladder, "name": name, "params": params,
                            "log10_params": round(float(np.log10(params)), 3), "align": align})
            print(f"{ladder:8s} {name:12s} ({params/1e6:6.0f}M, L{fl}/{mdl.config.num_hidden_layers}): "
                  f"MiniLM={align['MiniLM']:+.3f}  mpnet={align['mpnet']:+.3f}", flush=True)
            del mdl, tok, rep, D
            gc.collect()
            torch.cuda.empty_cache() if DEV == "cuda" else None

    # Spearman alignment-vs-scale, per ladder + pooled
    def rho_for(rows, tag):
        if len(rows) < 3:
            return None
        return round(spearman([r["log10_params"] for r in rows], [r["align"][tag] for r in rows]), 3)

    per_ladder = {}
    for ladder in LADDERS:
        rows = [r for r in results if r["ladder"] == ladder]
        per_ladder[ladder] = {"n": len(rows), "rho_MiniLM": rho_for(rows, "MiniLM"), "rho_mpnet": rho_for(rows, "mpnet"),
                              "smallest_MiniLM": rows[0]["align"]["MiniLM"] if rows else None,
                              "largest_MiniLM": rows[-1]["align"]["MiniLM"] if rows else None}
    pooled_rho = rho_for(results, "MiniLM")
    pooled_rho_mpnet = rho_for(results, "mpnet")
    ladders_positive = sum(1 for L in per_ladder.values() if (L["rho_MiniLM"] or 0) > 0)

    pythia = per_ladder.get("Pythia", {})
    confirmed = (pooled_rho is not None and pooled_rho >= 0.50) and (ladders_positive >= 2)
    pythia_clean = (pythia.get("rho_MiniLM") or 0) > 0
    if confirmed and pythia_clean:
        reading = (f"SCALE EFFECT CONFIRMED (clean). Alignment to an independent semantic reference rises "
                   f"with scale: pooled Spearman rho={pooled_rho:.2f} (mpnet {pooled_rho_mpnet:.2f}), positive in "
                   f"{ladders_positive}/3 ladders INCLUDING the perfectly-controlled Pythia ({pythia.get('rho_MiniLM')}). "
                   f"Convergence is scale-driven -- PRH-as-scaling-limit supported.")
    elif confirmed and not pythia_clean:
        reading = (f"SCALE EFFECT CONFIRMED but CONFOUNDED: pooled rho={pooled_rho:.2f} positive, but the "
                   f"perfectly-controlled Pythia ladder is NOT clean (rho={pythia.get('rho_MiniLM')}). The pooled "
                   f"trend may ride on data/arch differences across ladders, not scale alone.")
    elif pooled_rho is not None and abs(pooled_rho) < 0.2:
        reading = (f"NULL: within 14M-3B, scale does NOT drive convergence (pooled rho={pooled_rho:.2f}). The "
                   f"heterogeneity is data/architecture-driven, not a scale climb -- WEAKENS the PRH-limit story "
                   f"at this range.")
    else:
        reading = (f"PARTIAL/INCONCLUSIVE: pooled rho={pooled_rho}, {ladders_positive}/3 ladders positive, "
                   f"Pythia rho={pythia.get('rho_MiniLM')}. Does not cleanly meet or refute the frozen gate.")

    out = {"n_concepts": N, "n_models": len(results), "pooled_spearman_MiniLM": pooled_rho,
           "pooled_spearman_mpnet": pooled_rho_mpnet, "ladders_positive": ladders_positive,
           "per_ladder": per_ladder, "models": results, "confirmed": confirmed, "reading": reading}
    (HERE / "scale_law_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== alignment (partial-lexical RSA vs MiniLM) by scale ===")
    for r in sorted(results, key=lambda r: r["params"]):
        print(f"  {r['ladder']:8s} {r['name']:12s} {r['params']/1e6:7.0f}M  MiniLM={r['align']['MiniLM']:+.3f}")
    print("\nper-ladder Spearman rho(alignment, log params):")
    for ladder, L in per_ladder.items():
        print(f"  {ladder:8s}: rho_MiniLM={L['rho_MiniLM']}  (smallest {L['smallest_MiniLM']} -> largest {L['largest_MiniLM']})")
    print(f"\npooled Spearman rho: MiniLM={pooled_rho}  mpnet={pooled_rho_mpnet}  ({ladders_positive}/3 ladders positive)")
    print(f"\n>>> {reading}")
    print("wrote scale_law_result.json")


if __name__ == "__main__":
    main()
