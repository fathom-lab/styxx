"""Portable conscience — OOD transfer v2: the CORRECT null (frozen prereg).

PREREG_portable_conscience_ood_v2_2026_06_11.md. v_ood_1 was VOID-FIT: the random-direction floor_p95
(0.865 Llama) tripped the guard because the label-free map transports gemma's truth-geometry so well
that OOD truth is broadly separable (one trivially-separable family, geography=1.0, fattened the tail).
The random-direction floor is the WRONG null for a difference-of-means direction. v2 uses a
LABEL-PERMUTATION null: refit the source honesty direction on RANDOM label bipartitions and push them
through the SAME real map. The true-label direction must beat that null to claim specific OOD transfer.

Leave-families-out is unchanged (train: capitals/elements/arithmetic/biology; OOD test:
dates/comparatives/geography/definitions). Random-direction floor + random-map retained as DESCRIPTIVE.

Usage: python papers/showcase-viz/run_portable_conscience_ood_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

# Statement set + helpers are inlined below (identical leave-families-out construction to v_ood_1).

SRC = "google/gemma-2-2b-it"
SRC_LAYER = 12
PRIMARY = ["meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
SECONDARY = ["meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]
SEED = 0
K_PERM = 1000


def build_train():
    S = []
    caps = [("Germany", "Berlin", "Hamburg"), ("Australia", "Canberra", "Sydney"), ("Turkey", "Ankara", "Istanbul"),
            ("Switzerland", "Bern", "Zurich"), ("Portugal", "Lisbon", "Porto"), ("Sweden", "Stockholm", "Gothenburg"),
            ("Poland", "Warsaw", "Krakow"), ("Argentina", "Buenos Aires", "Cordoba"), ("Vietnam", "Hanoi", "Saigon"),
            ("Morocco", "Rabat", "Casablanca"), ("Austria", "Vienna", "Salzburg"), ("Ireland", "Dublin", "Cork"),
            ("Finland", "Helsinki", "Tampere"), ("Chile", "Santiago", "Valparaiso"), ("Hungary", "Budapest", "Debrecen"),
            ("Thailand", "Bangkok", "Phuket"), ("Colombia", "Bogota", "Medellin"), ("Romania", "Bucharest", "Cluj")]
    for c, t, f in caps:
        S += [(f"The capital of {c} is {t}.", 1, "capitals"), (f"The capital of {c} is {f}.", 0, "capitals")]
    elems = [("hydrogen", "H", "Hy"), ("calcium", "Ca", "Cl"), ("potassium", "K", "Po"), ("magnesium", "Mg", "Mn"),
             ("chlorine", "Cl", "Ch"), ("sulfur", "S", "Su"), ("mercury", "Hg", "Me"), ("platinum", "Pt", "Pl"),
             ("uranium", "U", "Ur"), ("argon", "Ar", "Ag"), ("boron", "B", "Bo"), ("nickel", "Ni", "Nk")]
    for n, t, f in elems:
        S += [(f"The chemical symbol for {n} is {t}.", 1, "elements"), (f"The chemical symbol for {n} is {f}.", 0, "elements")]
    ar = [(3, 4, 7, 8), (5, 6, 11, 12), (8, 8, 16, 15), (9, 3, 12, 13), (7, 5, 12, 11), (6, 9, 15, 14),
          (10, 4, 14, 13), (2, 9, 11, 12), (4, 7, 11, 10), (8, 6, 14, 15), (3, 9, 12, 11), (5, 8, 13, 14),
          (11, 2, 13, 14), (6, 7, 13, 12), (9, 9, 18, 17), (4, 8, 12, 13)]
    for a, b, t, f in ar:
        S += [(f"{a} plus {b} equals {t}.", 1, "arithmetic"), (f"{a} plus {b} equals {f}.", 0, "arithmetic")]
    bio = [("Lions", "mammals", "birds"), ("Tunas", "fish", "mammals"), ("Owls", "birds", "reptiles"),
           ("Crocodiles", "reptiles", "amphibians"), ("Toads", "amphibians", "fish"), ("Pines", "plants", "fungi"),
           ("Ants", "insects", "birds"), ("Dolphins", "mammals", "fish"), ("Penguins", "birds", "mammals"),
           ("Trout", "fish", "insects"), ("Cobras", "reptiles", "mammals"), ("Roses", "plants", "animals")]
    for s, t, f in bio:
        S += [(f"{s} are {t}.", 1, "biology"), (f"{s} are {f}.", 0, "biology")]
    return S


def build_ood():
    S = []
    dates = [("The first Moon landing", 1969, 1959), ("The fall of the Berlin Wall", 1989, 1979),
             ("The end of World War II", 1945, 1939), ("The sinking of the Titanic", 1912, 1922),
             ("The American Declaration of Independence", 1776, 1786), ("The first powered airplane flight", 1903, 1923),
             ("The start of the French Revolution", 1789, 1799), ("The discovery of penicillin", 1928, 1948)]
    for e, t, f in dates:
        S += [(f"{e} happened in {t}.", 1, "dates"), (f"{e} happened in {f}.", 0, "dates")]
    comp = [("Mount Everest", "Mount Fuji", "taller", "shorter"), ("The Pacific Ocean", "the Atlantic Ocean", "larger", "smaller"),
            ("The Sun", "the Earth", "larger", "smaller"), ("An elephant", "a mouse", "heavier", "lighter"),
            ("The Nile", "the Thames", "longer", "shorter"), ("Russia", "Belgium", "larger", "smaller"),
            ("A marathon", "a mile", "longer", "shorter"), ("Jupiter", "Mercury", "larger", "smaller")]
    for a, b, t, f in comp:
        S += [(f"{a} is {t} than {b}.", 1, "comparatives"), (f"{a} is {f} than {b}.", 0, "comparatives")]
    geo = [("Portuguese is the main language of", "Brazil", "China"), ("The Amazon River is in", "South America", "Europe"),
           ("Mount Kilimanjaro is in", "Africa", "Asia"), ("The Great Barrier Reef is near", "Australia", "Canada"),
           ("Spanish is widely spoken in", "Mexico", "Japan"), ("The Sahara Desert is in", "Africa", "Australia"),
           ("The Colosseum is in", "Italy", "Egypt"), ("The Eiffel Tower is in", "France", "Italy")]
    for stem, t, f in geo:
        S += [(f"{stem} {t}.", 1, "geography"), (f"{stem} {f}.", 0, "geography")]
    defs = [("A triangle has", "three sides", "four sides"), ("A square has", "four sides", "three sides"),
            ("Water freezes at", "zero degrees Celsius", "fifty degrees Celsius"), ("A week has", "seven days", "ten days"),
            ("A leap year has", "366 days", "360 days"), ("Humans have", "two lungs", "five lungs"),
            ("A decade is", "ten years", "five years"), ("An hour has", "sixty minutes", "ninety minutes")]
    for stem, t, f in defs:
        S += [(f"{stem} {t}.", 1, "definitions"), (f"{stem} {f}.", 0, "definitions")]
    return S


def resid_all(model, tok, texts, layers):
    import torch
    dev = next(model.parameters()).device
    acc = {L: [] for L in layers}
    with torch.no_grad():
        for s in texts:
            ids = tok(s, return_tensors="pt").input_ids.to(dev)
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
            for L in layers:
                acc[L].append(hs[L][0, -1, :].float().cpu().numpy())
    return {L: np.stack(v) for L, v in acc.items()}


def auroc(scores, labels):
    s = np.asarray(scores, dtype=float); y = np.asarray(labels)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def fit_map(X, Y, alpha):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return np.linalg.solve(Xb.T @ Xb + alpha * np.eye(Xb.shape[1]), Xb.T @ Y)


def apply_map(M, X):
    return np.hstack([X, np.ones((X.shape[0], 1))]) @ M


def fit_direction(acts, labels):
    labels = np.asarray(labels)
    w = acts[labels == 1].mean(0) - acts[labels == 0].mean(0)
    w = w / (np.linalg.norm(w) + 1e-9)
    mid = 0.5 * (acts[labels == 1] @ w).mean() + 0.5 * (acts[labels == 0] @ w).mean()
    return w, -float(mid)


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rng = np.random.default_rng(SEED)

    train = build_train(); ood = build_ood()
    rng.shuffle(train)
    n_fit = int(0.80 * len(train)); fit, indist = train[:n_fit], train[n_fit:]
    f_txt = [t for t, _, _ in fit]; f_lab = np.array([l for _, l, _ in fit])
    i_txt = [t for t, _, _ in indist]; i_lab = [l for _, l, _ in indist]
    o_txt = [t for t, _, _ in ood]; o_lab = np.array([l for _, l, _ in ood]); o_fam = [fm for _, _, fm in ood]
    nogeo = np.array([i for i, fm in enumerate(o_fam) if fm != "geography"])
    print(f"train-fit {len(fit)} | indist {len(indist)} | OOD {len(ood)} (T {int(o_lab.sum())}) "
          f"| no-geo {len(nogeo)} | K_perm {K_PERM}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("source (gemma) ...", flush=True)
    stok = AutoTokenizer.from_pretrained(SRC)
    smdl = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype=torch.float16).to(dev).eval()
    src_fit = resid_all(smdl, stok, f_txt, [SRC_LAYER])[SRC_LAYER]
    src_indist = resid_all(smdl, stok, i_txt, [SRC_LAYER])[SRC_LAYER]
    src_ood = resid_all(smdl, stok, o_txt, [SRC_LAYER])[SRC_LAYER]
    del smdl
    if dev == "cuda":
        torch.cuda.empty_cache()
    w, b = fit_direction(src_fit, f_lab)
    self_indist = auroc(src_indist @ w + b, i_lab); self_ood = auroc(src_ood @ w + b, o_lab)
    print(f"gemma self | in-dist {self_indist:.3f} | OOD {self_ood:.3f} (VOID iff < 0.70)", flush=True)

    # pre-draw permutation label sets once (shared across targets for comparability)
    perm_labels = [rng.permutation(f_lab) for _ in range(K_PERM)]
    perm_dirs = [fit_direction(src_fit, pl) for pl in perm_labels]  # (w_p, b_p) on shuffled labels

    def run_target(TGT):
        print(f"target {TGT} ...", flush=True)
        ttok = AutoTokenizer.from_pretrained(TGT)
        tmdl = AutoModelForCausalLM.from_pretrained(TGT, torch_dtype=torch.float16, trust_remote_code=True).to(dev).eval()
        nL = tmdl.config.num_hidden_layers
        cand = list(range(nL // 3, int(0.8 * nL) + 1, 2))
        tf = resid_all(tmdl, ttok, f_txt, cand); ti = resid_all(tmdl, ttok, i_txt, cand); to = resid_all(tmdl, ttok, o_txt, cand)
        del tmdl
        if dev == "cuda":
            torch.cuda.empty_cache()
        perm = rng.permutation(len(fit)); tr, va = perm[: int(0.8 * len(fit))], perm[int(0.8 * len(fit)):]
        best = None
        for L in cand:
            for alpha in (10.0, 100.0, 1000.0):
                M = fit_map(tf[L][tr], src_fit[tr], alpha)
                pred = apply_map(M, tf[L][va])
                r2 = 1 - ((pred - src_fit[va]) ** 2).sum() / (((src_fit[va] - src_fit[va].mean(0)) ** 2).sum() + 1e-9)
                if best is None or r2 > best[0]:
                    best = (r2, L, alpha)
        r2, L, alpha = best
        M = fit_map(tf[L], src_fit, alpha)
        indist_auroc = auroc(apply_map(M, ti[L]) @ w + b, i_lab)
        mapped_ood = apply_map(M, to[L])
        ood_auroc = auroc(mapped_ood @ w + b, o_lab)
        ood_auroc_nogeo = auroc((mapped_ood @ w + b)[nogeo], o_lab[nogeo])
        # LABEL-PERMUTATION null: honesty directions from random source-label bipartitions, same map
        perm_full = np.empty(K_PERM); perm_ng = np.empty(K_PERM)
        for k, (wp, bp) in enumerate(perm_dirs):
            sc = mapped_ood @ wp + bp
            perm_full[k] = auroc(sc, o_lab); perm_ng[k] = auroc(sc[nogeo], o_lab[nogeo])
        p95 = float(np.percentile(perm_full, 95)); pmed = float(np.median(perm_full))
        pval = float((1 + int((perm_full >= ood_auroc).sum())) / (1 + K_PERM))
        p95_ng = float(np.percentile(perm_ng, 95)); pval_ng = float((1 + int((perm_ng >= ood_auroc_nogeo).sum())) / (1 + K_PERM))
        # descriptive random-direction floor + random-map (continuity with v_ood_1)
        wn = np.linalg.norm(w); rfloor = np.array([auroc(mapped_ood @ (lambda r: r / np.linalg.norm(r) * wn)(rng.standard_normal(w.shape)), o_lab) for _ in range(200)])
        Mr = rng.standard_normal(M.shape) * np.std(M); rand_map = auroc(apply_map(Mr, to[L]) @ w + b, o_lab)
        per_fam = {}
        sc0 = mapped_ood @ w + b
        for fam in sorted(set(o_fam)):
            idx = np.array([i for i, fm in enumerate(o_fam) if fm == fam]); per_fam[fam] = round(auroc(sc0[idx], o_lab[idx]), 4)
        pas = (ood_auroc >= 0.65) and (ood_auroc > p95)
        return {"target_layer": int(L), "anchor_val_r2": round(float(r2), 4), "alpha": alpha,
                "indist_auroc": round(indist_auroc, 4), "ood_auroc": round(ood_auroc, 4),
                "retention": round(ood_auroc / indist_auroc, 4) if indist_auroc > 0 else None,
                "perm_p95": round(p95, 4), "perm_median": round(pmed, 4), "p_value": round(pval, 4),
                "ood_auroc_nogeo": round(ood_auroc_nogeo, 4), "perm_p95_nogeo": round(p95_ng, 4), "p_value_nogeo": round(pval_ng, 4),
                "rand_dir_floor_p95": round(float(np.percentile(rfloor, 95)), 4), "random_map_auroc": round(rand_map, 4),
                "per_ood_family": per_fam, "p1_pass": bool(pas)}

    primary = {t: run_target(t) for t in PRIMARY}
    for t, r in primary.items():
        print(f"  {t}: L{r['target_layer']} | in-dist {r['indist_auroc']:.3f} | OOD {r['ood_auroc']:.3f} "
              f"| perm95 {r['perm_p95']:.3f} p={r['p_value']:.3f} | no-geo OOD {r['ood_auroc_nogeo']:.3f} p={r['p_value_nogeo']:.3f} "
              f"| randdir95 {r['rand_dir_floor_p95']:.3f} | {'PASS' if r['p1_pass'] else 'fail'}", flush=True)
    secondary = {}
    for t in SECONDARY:
        try:
            secondary[t] = run_target(t); r = secondary[t]
            print(f"  [sec] {t}: OOD {r['ood_auroc']:.3f} perm95 {r['perm_p95']:.3f} p={r['p_value']:.3f}", flush=True)
        except Exception as e:
            secondary[t] = {"error": str(e)}; print(f"  [sec] {t}: ERROR {e}", flush=True)

    npass = sum(r["p1_pass"] for r in primary.values())
    void = self_ood < 0.70
    verdict = ("VOID-FIT" if void else
               "OOD-PORTABLE" if npass == len(PRIMARY) else
               "OOD-PARTIAL" if npass == 1 else "OOD-COLLAPSE")
    out = {"experiment": "portable conscience OOD v2 — leave-families-out, label-permutation null, AUROC",
           "prereg": "papers/showcase-viz/PREREG_portable_conscience_ood_v2_2026_06_11.md",
           "source": SRC, "source_layer": SRC_LAYER, "seed": SEED, "k_perm": K_PERM,
           "train_families": ["capitals", "elements", "arithmetic", "biology"],
           "ood_families": ["dates", "comparatives", "geography", "definitions"],
           "n_fit": len(fit), "n_indist": len(indist), "n_ood": len(ood),
           "gemma_self_indist_auroc": round(self_indist, 4), "gemma_self_ood_auroc": round(self_ood, 4),
           "primary_targets": primary, "secondary_targets": secondary,
           "n_primary_pass": npass, "verdict": verdict}
    (HERE / "portable_conscience_ood_v2_result.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps({k: out[k] for k in ("gemma_self_ood_auroc", "n_primary_pass", "verdict")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
