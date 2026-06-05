# -*- coding: utf-8 -*-
"""run_real_universality.py — the real-model frontier: are differently-trained REAL models near-isometric
enough for ZERO-ANCHOR alignment? The synthetic warp result proved near-isometry is the make-or-break
condition for unsupervised (zero-pairs) cross-model alignment. This tests it on real brains trained on
DIFFERENT data: GPT-2 (WebText), Pythia (the Pile), Qwen, OPT, BLOOM. For each pair: RSA (shared geometry)
and unsupervised GW recovery (near-isometry / zero-anchor alignability).

HONEST SCOPE: real models all saw English/web text, so shared geometry does NOT isolate data-independence
(that is what the synthetic disjoint-worlds controlled). This answers the PRACTICAL frontier -- does
zero-anchor alignment WORK on real models -- not the metaphysical one. Recovery (near-isometry) is the
load-bearing metric; RSA is context (confounded by shared data).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_disjoint_worlds as R
from transformers import AutoModel, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = "--smoke" in sys.argv

CONCEPTS = ("dog cat horse cow pig sheep lion tiger bear wolf fox deer rabbit mouse elephant "
            "apple banana orange grape lemon peach cherry strawberry potato carrot onion bread cheese "
            "hammer screwdriver wrench saw drill knife fork spoon plate bowl cup bottle "
            "car truck bus train plane boat ship bicycle motorcycle "
            "chair table bed sofa desk shelf lamp mirror clock door window "
            "shirt pants shoe hat coat dress sock glove scarf "
            "tree flower grass leaf river mountain ocean lake forest desert cloud rain snow sun moon star "
            "house school church store bank hospital bridge tower castle "
            "guitar piano drum violin trumpet flute").split()

# safetensors-only (torch<2.6 refuses .bin via torch.load — CVE-2025-32434). Cross-family (different data):
# GPT-2/WebText, Pythia/Pile, Qwen. Same-family scale/version pairs included for the isometry-vs-family contrast.
DECODERS = ["distilgpt2", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B"]
EMBEDDERS = ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2", "BAAI/bge-small-en-v1.5"]
MODELS = (["distilgpt2", "EleutherAI/pythia-160m"] if SMOKE else DECODERS + EMBEDDERS)
EMB_SET = set(EMBEDDERS)
if SMOKE:
    CONCEPTS = CONCEPTS[:30]
N = len(CONCEPTS)
R.N = N
R.GW_INITS = 10
IU = np.triu_indices(N, 1)


# contextual templates ENDING in the concept -> last-token hidden state integrates context (decoder LMs).
# bare words are measurement-limited (known null from prior real-convergence work); templates surface geometry.
TEMPLATES = ["a {}", "the {}", "I saw a {}.", "a photo of a {}.", "this is a {}.", "look at the {}."]


@torch.no_grad()
def embed(model_name):
    is_emb = model_name in EMB_SET                     # embedders: mean-pool (meaning-trained); decoders: last-token
    tok = AutoTokenizer.from_pretrained(model_name)
    m = AutoModel.from_pretrained(model_name, dtype=torch.float32, use_safetensors=True).to(DEV).eval()
    vecs = []
    for c in CONCEPTS:
        reps = []
        for t in TEMPLATES:
            ids = tok(t.format(c), return_tensors="pt").to(DEV)
            h = m(**ids).last_hidden_state[0]          # (T, d)
            reps.append((h.mean(0) if is_emb else h[-1]).float().cpu().numpy())
        vecs.append(np.mean(reps, 0))                  # average across templates
    del m
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return np.array(vecs)


def rsa(EA, EB):
    DA, DB = R.distmat(EA), R.distmat(EB)
    return float(np.corrcoef(DA[IU], DB[IU])[0, 1])


def main():
    print(f"extracting {N} concepts from {len(MODELS)} real models...", flush=True)
    E = {}
    for mn in MODELS:
        E[mn] = embed(mn)
        print(f"  {mn}: {E[mn].shape}", flush=True)

    pairs = []
    rng = np.random.default_rng(0)
    for i in range(len(MODELS)):
        for j in range(i + 1, len(MODELS)):
            a, b = MODELS[i], MODELS[j]
            r = rsa(E[a], E[b])
            perm = rng.permutation(N); true_match = np.argsort(perm)
            assign, _ = R.align(E[a], E[b][perm], rng)
            recov = float(np.mean(assign == true_match))
            kind = "emb-emb" if (a in EMB_SET and b in EMB_SET) else ("dec-dec" if (a not in EMB_SET and b not in EMB_SET) else "mixed")
            pairs.append({"a": a.split("/")[-1], "b": b.split("/")[-1], "kind": kind, "rsa": round(r, 3), "recovery": round(recov, 3)})
            print(f"  [{kind}] {a.split('/')[-1]:16s} <-> {b.split('/')[-1]:18s}  RSA={r:.3f}  zero-anchor recovery={recov:.3f}", flush=True)

    chance = 1.0 / N
    rsas = [p["rsa"] for p in pairs]; recs = [p["recovery"] for p in pairs]
    mean_rsa, max_rec = float(np.mean(rsas)), float(np.max(recs))
    n_align = sum(1 for p in pairs if p["recovery"] >= 0.30)
    def kmax(k): vs = [p["recovery"] for p in pairs if p["kind"] == k]; return round(float(max(vs)), 3) if vs else None
    rec_by_kind = {"emb-emb": kmax("emb-emb"), "dec-dec": kmax("dec-dec"), "mixed": kmax("mixed")}
    print(f"\n  max recovery by kind: {rec_by_kind}  (chance {chance:.3f})", flush=True)
    if mean_rsa >= 0.30 and max_rec >= 0.30:
        reading = (f"NEAR-ISOMETRIC ENOUGH (some pairs) — real differently-trained models share geometry "
                   f"(mean RSA {mean_rsa:.2f}) and {n_align}/{len(pairs)} pairs are zero-anchor alignable "
                   f"(recovery up to {max_rec:.2f} vs chance {chance:.3f}). Zero-anchor cross-model alignment "
                   "WORKS on real brains for the near-isometric pairs -- the practical frontier is reachable.")
    elif mean_rsa >= 0.30:
        reading = (f"SHARED BUT NOT ZERO-ANCHOR ALIGNABLE — real models share geometry (mean RSA {mean_rsa:.2f}) "
                   f"but unsupervised recovery fails (max {max_rec:.2f} ~ chance {chance:.3f}). The shared geometry "
                   "is real but NOT near-isometric enough for zero-pair alignment -- exactly the warp regime: "
                   "transfer needs anchors. The practical limit, on real brains.")
    else:
        reading = (f"WEAK SHARING — real models do not strongly share concept geometry here (mean RSA {mean_rsa:.2f}); "
                   "diagnose extraction (pooling/layer/template) before concluding.")
    out = {"models": MODELS, "n_concepts": N, "pairs": pairs,
           "gate": {"mean_rsa": round(mean_rsa, 3), "max_recovery": round(max_rec, 3),
                    "max_recovery_by_kind": rec_by_kind,
                    "n_alignable_pairs": n_align, "chance": round(chance, 4), "reading": reading}}
    fn = HERE / ("real_universality_smoke.json" if SMOKE else "real_universality_result.json")
    fn.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n===== " + reading)
    print("wrote", fn.name)


if __name__ == "__main__":
    main()
