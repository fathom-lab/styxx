"""Portable conscience v1 — properly-powered transfer test (frozen prereg).

PREREG_portable_conscience_v1_2026_06_10.md: ~100-item balanced true/false test scored by AUROC;
gemma truthfulness direction through a label-free ridge map into Llama-3.2-3B and Qwen2.5-3B; floor
= 200 random directions through the same map. P1 portable iff AUROC>=0.65 and beats floor on both.

Usage: python papers/showcase-viz/run_portable_conscience_v1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

SRC = "google/gemma-2-2b-it"
SRC_LAYER = 12
TARGETS = ["meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
SEED = 0


def build_statements():
    """~240 balanced true/false factual statements, diverse templates, disjoint from v0 facts."""
    S = []
    caps = [("Germany", "Berlin", "Hamburg"), ("Australia", "Canberra", "Sydney"), ("Turkey", "Ankara", "Istanbul"),
            ("Switzerland", "Bern", "Zurich"), ("Portugal", "Lisbon", "Porto"), ("Sweden", "Stockholm", "Gothenburg"),
            ("Poland", "Warsaw", "Krakow"), ("Argentina", "Buenos Aires", "Cordoba"), ("Vietnam", "Hanoi", "Saigon"),
            ("Morocco", "Rabat", "Casablanca"), ("Austria", "Vienna", "Salzburg"), ("Ireland", "Dublin", "Cork"),
            ("Finland", "Helsinki", "Tampere"), ("Chile", "Santiago", "Valparaiso"), ("Hungary", "Budapest", "Debrecen"),
            ("Thailand", "Bangkok", "Phuket"), ("Colombia", "Bogota", "Medellin"), ("Romania", "Bucharest", "Cluj")]
    for c, t, f in caps:
        S += [(f"The capital of {c} is {t}.", 1), (f"The capital of {c} is {f}.", 0)]
    elems = [("hydrogen", "H", "Hy"), ("calcium", "Ca", "Cl"), ("potassium", "K", "Po"), ("magnesium", "Mg", "Mn"),
             ("chlorine", "Cl", "Ch"), ("sulfur", "S", "Su"), ("mercury", "Hg", "Me"), ("platinum", "Pt", "Pl"),
             ("uranium", "U", "Ur"), ("argon", "Ar", "Ag"), ("boron", "B", "Bo"), ("nickel", "Ni", "Nk")]
    for n, t, f in elems:
        S += [(f"The chemical symbol for {n} is {t}.", 1), (f"The chemical symbol for {n} is {f}.", 0)]
    ar = [(3, 4, 7, 8), (5, 6, 11, 12), (8, 8, 16, 15), (9, 3, 12, 13), (7, 5, 12, 11), (6, 9, 15, 14),
          (10, 4, 14, 13), (2, 9, 11, 12), (4, 7, 11, 10), (8, 6, 14, 15), (3, 9, 12, 11), (5, 8, 13, 14),
          (11, 2, 13, 14), (6, 7, 13, 12), (9, 9, 18, 17), (4, 8, 12, 13)]
    for a, b, t, f in ar:
        S += [(f"{a} plus {b} equals {t}.", 1), (f"{a} plus {b} equals {f}.", 0)]
    bio = [("Lions", "mammals", "birds"), ("Tunas", "fish", "mammals"), ("Owls", "birds", "reptiles"),
           ("Crocodiles", "reptiles", "amphibians"), ("Toads", "amphibians", "fish"), ("Pines", "plants", "fungi"),
           ("Ants", "insects", "birds"), ("Dolphins", "mammals", "fish"), ("Penguins", "birds", "mammals"),
           ("Trout", "fish", "insects"), ("Cobras", "reptiles", "mammals"), ("Roses", "plants", "animals")]
    for s, t, f in bio:
        S += [(f"{s} are {t}.", 1), (f"{s} are {f}.", 0)]
    isa = [("Jupiter", "a planet", "a star"), ("Venus", "a planet", "a moon"), ("Saturn", "a planet", "a comet"),
           ("The Nile", "a river", "a desert"), ("The Alps", "mountains", "oceans"), ("Lake Victoria", "a lake", "a sea"),
           ("Tokyo", "a city", "a country"), ("Africa", "a continent", "a country"), ("The Thames", "a river", "a mountain"),
           ("Sicily", "an island", "a continent"), ("The Gobi", "a desert", "a forest"), ("Neptune", "a planet", "a galaxy")]
    for s, t, f in isa:
        S += [(f"{s} is {t}.", 1), (f"{s} is {f}.", 0)]
    sci = [("Sound", "vibrations", "light"), ("Rust", "iron oxide", "gold"), ("Salt", "sodium chloride", "sugar"),
           ("Steam", "water vapor", "smoke"), ("Lightning", "electricity", "sound"), ("Granite", "a rock", "a metal"),
           ("Helium", "a gas", "a liquid"), ("Diamond", "carbon", "silver"), ("Blood", "red", "green"),
           ("Grass", "green", "blue")]
    for s, t, f in sci:
        S += [(f"{s} is made of {t}." if s in ("Rust", "Salt", "Diamond") else f"{s} is {t}.", 1),
              (f"{s} is made of {f}." if s in ("Rust", "Salt", "Diamond") else f"{s} is {f}.", 0)]
    return S


def resid(model, tok, texts, layer):
    import torch
    dev = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for s in texts:
            ids = tok(s, return_tensors="pt").input_ids.to(dev)
            h = model(input_ids=ids, output_hidden_states=True).hidden_states[layer][0, -1, :]
            out.append(h.float().cpu().numpy())
    return np.stack(out)


def auroc(scores, labels):
    s = np.asarray(scores); y = np.asarray(labels)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum() for _ in [0])
    return float(wins / (len(pos) * len(neg)))


def fit_map(X, Y, alpha):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return np.linalg.solve(Xb.T @ Xb + alpha * np.eye(Xb.shape[1]), Xb.T @ Y)


def apply_map(M, X):
    return np.hstack([X, np.ones((X.shape[0], 1))]) @ M


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from styxx.residual_probe.probe import StyxxProbe
    rng = np.random.default_rng(SEED)

    probe = StyxxProbe.from_pretrained(SRC, "truthfulness")
    w = probe.weight.float().cpu().numpy(); b = probe.bias
    stmts = build_statements()
    rng.shuffle(stmts)
    n = len(stmts)
    n_anchor = int(0.585 * n)
    anchors, test = stmts[:n_anchor], stmts[n_anchor:]
    a_txt = [t for t, _ in anchors]
    t_txt = [t for t, _ in test]; t_lab = [l for _, l in test]
    print(f"statements {n} | anchor {len(anchors)} | test {len(test)} (true {sum(t_lab)}/{len(t_lab)})", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("source (gemma) ...", flush=True)
    stok = AutoTokenizer.from_pretrained(SRC)
    smdl = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype=torch.float16).to(dev).eval()
    src_anchor = resid(smdl, stok, a_txt, SRC_LAYER)
    src_test = resid(smdl, stok, t_txt, SRC_LAYER)
    del smdl
    if dev == "cuda":
        torch.cuda.empty_cache()
    ceil = auroc(src_test @ w + b, t_lab)
    print(f"ceiling (gemma self) AUROC {ceil:.3f}", flush=True)

    results = {}
    for TGT in TARGETS:
        print(f"target {TGT} ...", flush=True)
        ttok = AutoTokenizer.from_pretrained(TGT)
        tmdl = AutoModelForCausalLM.from_pretrained(TGT, torch_dtype=torch.float16, trust_remote_code=True).to(dev).eval()
        nL = tmdl.config.num_hidden_layers
        cand = list(range(nL // 3, int(0.8 * nL) + 1, 2))
        ta = {L: resid(tmdl, ttok, a_txt, L) for L in cand}
        tt = {L: resid(tmdl, ttok, t_txt, L) for L in cand}
        del tmdl
        if dev == "cuda":
            torch.cuda.empty_cache()
        # select layer + alpha by anchor-split fit R2 (no test contact)
        perm = rng.permutation(len(anchors)); tr, va = perm[: int(0.8 * len(anchors))], perm[int(0.8 * len(anchors)):]
        best = None
        for L in cand:
            for alpha in (10.0, 100.0, 1000.0):
                M = fit_map(ta[L][tr], src_anchor[tr], alpha)
                pred = apply_map(M, ta[L][va])
                r2 = 1 - ((pred - src_anchor[va]) ** 2).sum() / (((src_anchor[va] - src_anchor[va].mean(0)) ** 2).sum() + 1e-9)
                if best is None or r2 > best[0]:
                    best = (r2, L, alpha)
        r2, L, alpha = best
        M = fit_map(ta[L], src_anchor, alpha)
        trans = auroc(apply_map(M, tt[L]) @ w + b, t_lab)
        wn = np.linalg.norm(w)
        floor = []
        mapped = apply_map(M, tt[L])
        for _ in range(200):
            rw = rng.standard_normal(w.shape); rw = rw / np.linalg.norm(rw) * wn
            floor.append(auroc(mapped @ rw, t_lab))
        floor = np.array(floor)
        f95 = float(np.percentile(floor, 95))
        Mr = rng.standard_normal(M.shape) * np.std(M)
        rand_map = auroc(apply_map(Mr, tt[L]) @ w + b, t_lab)
        pas = (trans >= 0.65) and (trans > f95)
        results[TGT] = {"target_layer": L, "anchor_val_r2": round(r2, 4), "alpha": alpha,
                        "transferred_auroc": round(trans, 4), "floor_p95": round(f95, 4),
                        "floor_median": round(float(np.median(floor)), 4),
                        "random_map_auroc": round(rand_map, 4), "p1_pass": bool(pas)}
        print(f"  {TGT}: layer {L} | transferred AUROC {trans:.3f} | floor95 {f95:.3f} "
              f"(median {np.median(floor):.3f}) | rand-map {rand_map:.3f} | {'PASS' if pas else 'fail'}", flush=True)

    npass = sum(r["p1_pass"] for r in results.values())
    void = ceil < 0.70 or any(r["floor_p95"] >= 0.75 for r in results.values())
    verdict = ("VOID-PIPELINE" if void else
               "CONSCIENCE-PORTABLE-v1" if npass == len(TARGETS) else
               "PARTIAL" if npass == 1 else "STRUCTURE-NOT-DIRECTION")
    out = {"experiment": "portable conscience v1 — powered transfer (AUROC)",
           "prereg": "papers/showcase-viz/PREREG_portable_conscience_v1_2026_06_10.md",
           "source": SRC, "source_layer": SRC_LAYER, "n_test": len(test), "n_true": sum(t_lab),
           "ceiling_gemma_self_auroc": round(ceil, 4), "per_target": results,
           "n_targets_pass": npass, "verdict": verdict}
    (HERE / "portable_conscience_v1_result.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps({k: out[k] for k in ("ceiling_gemma_self_auroc", "n_targets_pass", "verdict")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
