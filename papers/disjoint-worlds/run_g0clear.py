# -*- coding: utf-8 -*-
"""run_g0clear.py — STAGE 0 of PREREG_thought_transfer_g0clear_2026_06_20.

Decide G0 (the instrument gate) for the cross-model WRITE channel. The parent run
(FINDING_thought_transfer_2026_06_07) hit pc_cos=0.74 < 0.80 with N=110 concepts; the loss is purely
the fraction of a concept steering vector that lies OUTSIDE the top-k concept-PCA subspace. This stage
RAISES that coverage by (a) expanding the concept bank to N~480 and (b) sweeping the read layer and k,
then reads G0 on a CONCEPT SET HELD OUT of hyperparameter selection (the SELECTION/FINAL seal).

THE SEAL: this file loads ONLY model A and computes ONLY pc_cos (the A->A-rotated positive control).
It never loads model B and never computes a cross-model transfer number. Stage 1 is gated on the JSON
this writes (G0_pass == true) and is a separate invocation.

  python run_g0clear.py --src meta-llama/Llama-3.2-3B-Instruct --tag llama3b
  python run_g0clear.py --smoke         # tiny N, 1 layer, fast path check
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "introspection-gate"))
from styxx_transfer import self_test
from introspection_gate import load_model, CONCEPT_TEMPLATES, DEVICE

# ~480 single-word concepts across balanced categories (deduped at load). Superset of the N~110 parent
# bank; the point is subspace COVERAGE of held-out concept steering vectors, so breadth matters.
_BANK = """
dog cat horse cow pig sheep lion tiger bear wolf fox deer rabbit mouse elephant goat donkey camel zebra
giraffe monkey gorilla rat squirrel bat beaver otter seal whale dolphin shark octopus crab lobster shrimp
snail frog toad snake lizard turtle crocodile eagle hawk owl crow sparrow robin pigeon duck goose swan
chicken turkey penguin parrot peacock bee ant spider fly moth butterfly beetle wasp worm
apple banana orange grape lemon lime peach pear cherry plum apricot mango melon watermelon pineapple
strawberry raspberry blueberry coconut fig date olive potato carrot onion garlic tomato cucumber pepper
lettuce cabbage spinach broccoli celery pumpkin corn pea bean mushroom radish beet turnip
bread cheese rice egg milk butter sugar salt flour honey jam soup pasta pizza cake pie cookie chocolate
candy yogurt bacon sausage steak ham fish chicken coffee tea juice wine beer water
hammer wrench saw drill knife fork spoon plate bowl cup bottle jar pan pot kettle axe shovel rake hoe
ladder rope chain nail screw bolt nut pliers scissors needle thread brush broom mop bucket
car truck bus train plane boat ship bicycle helicopter rocket motorcycle scooter van taxi tram subway
canoe kayak yacht submarine tractor ambulance wagon sled
chair table bed sofa desk shelf lamp mirror clock door window roof wall floor ceiling stairs cabinet
drawer couch bench stool cushion pillow blanket curtain carpet rug bookcase wardrobe
shirt pants shoe hat coat dress sock glove scarf belt ring watch jacket sweater skirt boot sandal tie
button zipper pocket collar sleeve hood cap helmet
tree flower grass leaf root branch seed vine bush fern moss cactus mushroom river mountain ocean lake
forest desert cloud rain snow wind storm thunder lightning fog mist hill valley cliff cave island beach
sun moon star planet comet galaxy fire ice stone sand gold silver iron copper steel bronze glass wood
clay coal oil gas crystal diamond rock mud dust ash smoke steam
house school church store bank hospital bridge tower castle factory library museum prison palace barn
stadium hotel restaurant office garage shop market temple lighthouse
guitar piano drum violin trumpet flute harp cello saxophone clarinet banjo accordion organ tambourine
anger fear joy love hope sorrow pride shame guilt envy hatred courage trust doubt grief calm worry
relief disgust surprise boredom curiosity loneliness wonder gratitude jealousy
king queen doctor nurse teacher farmer soldier judge lawyer artist writer singer dancer pilot sailor
chef baker butcher hunter fisher miner builder painter plumber driver guard priest
head hand foot eye ear nose mouth arm leg finger toe knee elbow shoulder neck back chest heart brain
bone skin hair tooth tongue lip cheek chin
red green blue yellow orange purple pink brown black white gray
freedom justice truth power wealth time death birth memory dream idea language number reason wisdom
chaos order peace war
""".split()
_seen = set()
CONCEPTS = [c for c in _BANK if not (c in _seen or _seen.add(c))]

LAYER_FRACS = [0.40, 0.50, 0.60, 0.70]
K_GRID = [60, 90, 120, 150]


@torch.no_grad()
def extract_multi(model, tok, layers):
    """For each candidate layer L: per-concept mean point and steering direction (concept - 'object').
    One pair of forwards per (concept, template); all layers read from the same hidden_states tuple."""
    pts = {L: {} for L in layers}
    vecs = {L: {} for L in layers}
    for i, c in enumerate(CONCEPTS):
        acc_c = {L: [] for L in layers}
        acc_d = {L: [] for L in layers}
        for t in CONCEPT_TEMPLATES:
            hc = model(**tok(t.format(c=c), return_tensors="pt").to(DEVICE),
                       output_hidden_states=True).hidden_states
            ho = model(**tok(t.format(c="object"), return_tensors="pt").to(DEVICE),
                       output_hidden_states=True).hidden_states
            for L in layers:
                lc = hc[L + 1][0, -1].float().cpu().numpy()
                lo = ho[L + 1][0, -1].float().cpu().numpy()
                acc_c[L].append(lc); acc_d[L].append(lc - lo)
        for L in layers:
            pts[L][c] = np.mean(acc_c[L], 0)
            v = np.mean(acc_d[L], 0); vecs[L][c] = v / (np.linalg.norm(v) + 1e-9)
        if (i + 1) % 40 == 0:
            print(f"   extracted {i + 1}/{len(CONCEPTS)}", flush=True)
    return pts, vecs


def split_concepts(seed=0):
    """TRAIN (PCA cloud) / SEL_dirs (hyperparam selection) / FIN_dirs (locked G0 read) — disjoint."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(CONCEPTS))
    n = len(CONCEPTS)
    n_tr = int(0.70 * n); n_sel = int(0.15 * n)
    tr = [CONCEPTS[i] for i in idx[:n_tr]]
    sel = [CONCEPTS[i] for i in idx[n_tr:n_tr + n_sel]]
    fin = [CONCEPTS[i] for i in idx[n_tr + n_sel:]]
    return tr, sel, fin


def pc_cos_at(pts_L, vecs_L, train, test, k):
    """Positive control at one (layer, k): PCA cloud = train POINTS; test = held-out concept DIRS."""
    E = np.array([pts_L[c] for c in train])
    test_dirs = np.array([vecs_L[c] for c in test])
    k = min(k, E.shape[1], E.shape[0] - 1)
    cos, _ = self_test(E, k=k, test_dirs=test_dirs)
    return cos


def run(src, tag, smoke=False):
    global CONCEPTS
    layers_fracs = [0.60] if smoke else LAYER_FRACS
    kgrid = [30, 60] if smoke else K_GRID
    if smoke:
        CONCEPTS = CONCEPTS[:40]
    print(f"N concepts = {len(CONCEPTS)}", flush=True)

    from transformers import AutoConfig
    nl = AutoConfig.from_pretrained(src).num_hidden_layers
    layers = sorted({round(f * nl) for f in layers_fracs})
    print(f"src n_layers={nl}; candidate read layers={layers}; k grid={kgrid}", flush=True)

    tok, model = load_model(src)
    pts, vecs = extract_multi(model, tok, layers)
    del model; torch.cuda.empty_cache()

    tr, sel, fin = split_concepts(seed=0)
    print(f"split: TRAIN={len(tr)} SEL_dirs={len(sel)} FIN_dirs={len(fin)}", flush=True)

    # --- hyperparameter selection on SEL_dirs ONLY ---
    grid = []
    for L in layers:
        for k in kgrid:
            cos = pc_cos_at(pts[L], vecs[L], tr, sel, k)
            grid.append({"layer": L, "k": k, "pc_cos_sel": round(cos, 4)})
            print(f"   L={L:2d} k={k:3d}  pc_cos(SEL)={cos:.4f}", flush=True)
    best = max(grid, key=lambda g: g["pc_cos_sel"])
    L_star, k_star = best["layer"], best["k"]

    # --- locked G0 read on FIN_dirs (never seen in selection) ---
    g0 = pc_cos_at(pts[L_star], vecs[L_star], tr, fin, k_star)
    g0_pass = bool(g0 >= 0.80)
    grid_max = max(g["pc_cos_sel"] for g in grid)

    out = {
        "experiment": "G0-clear (cross-model write-channel instrument upgrade)",
        "prereg": "PREREG_thought_transfer_g0clear_2026_06_20.md",
        "src": src, "n_concepts": len(CONCEPTS), "smoke": smoke,
        "layers_tested": layers, "k_grid": kgrid,
        "selection_grid": grid,
        "locked": {"layer": L_star, "k": k_star, "pc_cos_sel_at_locked": best["pc_cos_sel"]},
        "grid_max_sel": round(grid_max, 4),
        "G0_pc_cos_FINAL": round(g0, 4),
        "G0_gate": 0.80,
        "G0_pass": g0_pass,
        "verdict": ("G0_CLEARED -> Stage 1 interpretable" if g0_pass
                    else "INSTRUMENT-CEILING (landed): concept-PCA subspace cannot host a full steering vector"),
        "parent_baseline": {"N": 110, "pc_cos": 0.74, "k_sweep": "0.705/0.742/0.769 @ k=40/60/85"},
        "seal": "model B not loaded; no cross-model transfer number computed in Stage 0",
    }
    fn = HERE / f"g0clear_result_{'smoke_' if smoke else ''}{tag}.json"
    fn.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps({k2: v for k2, v in out.items() if k2 != "selection_grid"}, indent=2), flush=True)
    print(f"\nwrote {fn.name}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--tag", default="llama3b")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    run(args.src, args.tag, smoke=args.smoke)


if __name__ == "__main__":
    main()
