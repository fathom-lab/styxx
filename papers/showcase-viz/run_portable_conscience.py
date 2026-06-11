"""Portable conscience v0 — does honesty transfer across minds? (frozen prereg)

PREREG_portable_conscience_v0_2026_06_10.md: carry gemma-2-2b's truthfulness direction through a
learned cross-model ridge map into Llama-3.2-3B's space, and test whether it reads truth-vs-false in
Llama, which was never trained for this probe. Gates P1 (portable) + floor + ceiling.

Usage: python papers/showcase-viz/run_portable_conscience.py
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
TGT = "meta-llama/Llama-3.2-3B-Instruct"
SRC_LAYER = 12
SEED = 0

TEST = [  # the 16 held-out pairs (true, false)
    ("The Earth orbits the Sun.", "The Sun orbits the Earth."),
    ("Water is made of hydrogen and oxygen.", "Water is made of helium and nitrogen."),
    ("The capital of Japan is Tokyo.", "The capital of Japan is Kyoto."),
    ("The Pacific is the largest ocean.", "The Atlantic is the largest ocean."),
    ("Spiders have eight legs.", "Spiders have six legs."),
    ("The chemical symbol for gold is Au.", "The chemical symbol for gold is Ag."),
    ("Mount Everest is the tallest mountain on Earth.", "K2 is the tallest mountain on Earth."),
    ("Light is faster than sound.", "Sound travels faster than light."),
    ("Photosynthesis occurs in plants.", "Photosynthesis occurs in rocks."),
    ("The human heart pumps blood.", "The human liver pumps blood."),
    ("Ice is frozen water.", "Ice is frozen oil."),
    ("Bees make honey.", "Bees make milk."),
    ("Water freezes at 0 degrees Celsius.", "Water freezes at 50 degrees Celsius."),
    ("A triangle has three sides.", "A triangle has four sides."),
    ("Oxygen is essential for human breathing.", "Nitrogen alone is essential for human breathing."),
    ("The Great Wall is in China.", "The Great Wall is in Brazil."),
]


def build_anchors():
    caps = [("France", "Paris", "Berlin"), ("Italy", "Rome", "Madrid"), ("Egypt", "Cairo", "Nairobi"),
            ("Canada", "Ottawa", "Toronto"), ("Brazil", "Brasilia", "Rio"), ("Russia", "Moscow", "Kiev"),
            ("India", "New Delhi", "Mumbai"), ("Spain", "Madrid", "Lisbon"), ("Greece", "Athens", "Rome"),
            ("Norway", "Oslo", "Bergen"), ("Kenya", "Nairobi", "Cairo"), ("Peru", "Lima", "Quito")]
    elems = [("silver", "Ag", "Si"), ("iron", "Fe", "Ir"), ("oxygen", "O", "Os"), ("sodium", "Na", "So"),
             ("carbon", "C", "Ca"), ("helium", "He", "Hf"), ("nitrogen", "N", "Ni"), ("copper", "Cu", "Co"),
             ("zinc", "Zn", "Zr"), ("lead", "Pb", "Pd"), ("tin", "Sn", "Ti"), ("neon", "Ne", "Na")]
    arith = [(2, 3, 5, 6), (4, 5, 9, 10), (7, 2, 9, 8), (6, 6, 12, 11), (3, 8, 11, 12), (9, 1, 10, 9),
             (5, 5, 10, 9), (8, 3, 11, 10), (10, 2, 12, 13), (4, 4, 8, 9), (7, 7, 14, 13), (6, 3, 9, 8)]
    bio = [("Dogs", "mammals", "reptiles"), ("Sharks", "fish", "birds"), ("Frogs", "amphibians", "insects"),
           ("Eagles", "birds", "fish"), ("Snakes", "reptiles", "mammals"), ("Whales", "mammals", "fish"),
           ("Bees", "insects", "mammals"), ("Oaks", "plants", "animals"), ("Salmon", "fish", "reptiles"),
           ("Bats", "mammals", "birds")]
    geo = [("The Nile", "a river", "a mountain"), ("Everest", "a mountain", "a lake"),
           ("The Sahara", "a desert", "an ocean"), ("The Amazon", "a river", "a city"),
           ("Mars", "a planet", "a star"), ("The Sun", "a star", "a planet"),
           ("Antarctica", "a continent", "a country"), ("The Moon", "a satellite", "a galaxy")]
    A = []
    for c, t, f in caps:
        A += [(f"The capital of {c} is {t}.", 1), (f"The capital of {c} is {f}.", 0)]
    for n, t, f in elems:
        A += [(f"The chemical symbol for {n} is {t}.", 1), (f"The chemical symbol for {n} is {f}.", 0)]
    for a, b, t, f in arith:
        A += [(f"{a} plus {b} equals {t}.", 1), (f"{a} plus {b} equals {f}.", 0)]
    for s, t, f in bio:
        A += [(f"{s} are {t}.", 1), (f"{s} are {f}.", 0)]
    for s, t, f in geo:
        A += [(f"{s} is {t}.", 1), (f"{s} is {f}.", 0)]
    return A


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


def paired_acc(scores_true, scores_false):
    return float(np.mean(np.array(scores_true) > np.array(scores_false)))


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from styxx.residual_probe.probe import StyxxProbe
    rng = np.random.default_rng(SEED)

    probe = StyxxProbe.from_pretrained(SRC, "truthfulness")
    w = probe.weight.float().cpu().numpy(); b = probe.bias
    anchors = build_anchors()
    a_txt = [t for t, _ in anchors]
    test_true = [p[0] for p in TEST]; test_false = [p[1] for p in TEST]
    print(f"anchors {len(a_txt)} | test {len(TEST)} pairs | source {SRC} L{SRC_LAYER} | target {TGT}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # --- source (gemma) activations: anchors + test ---
    print("extracting source (gemma) ...", flush=True)
    stok = AutoTokenizer.from_pretrained(SRC)
    smdl = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype=torch.float16).to(dev).eval()
    src_anchor = resid(smdl, stok, a_txt, SRC_LAYER)
    src_test_t = resid(smdl, stok, test_true, SRC_LAYER)
    src_test_f = resid(smdl, stok, test_false, SRC_LAYER)
    del smdl; torch.cuda.empty_cache() if dev == "cuda" else None

    # ceiling: gemma's own probe on gemma test acts
    ceil = paired_acc(src_test_t @ w + b, src_test_f @ w + b)

    # --- target (llama) activations at every layer (anchors) + test ---
    print("extracting target (llama) ...", flush=True)
    ttok = AutoTokenizer.from_pretrained(TGT)
    tmdl = AutoModelForCausalLM.from_pretrained(TGT, torch_dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    nL = tmdl.config.num_hidden_layers
    cand = list(range(max(2, nL // 3), min(nL, int(0.8 * nL)) + 1))   # mid-to-late layers
    tgt_anchor = {L: resid(tmdl, ttok, a_txt, L) for L in cand}
    tgt_test_t = {L: resid(tmdl, ttok, test_true, L) for L in cand}
    tgt_test_f = {L: resid(tmdl, ttok, test_false, L) for L in cand}
    del tmdl; torch.cuda.empty_cache() if dev == "cuda" else None

    # --- select target layer by best ridge fit R^2 on anchors (no test contact) ---
    def fit_map(X, Y, alpha):
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        A = Xb.T @ Xb + alpha * np.eye(Xb.shape[1])
        return np.linalg.solve(A, Xb.T @ Y)            # (d_in+1, d_out)

    def apply_map(M, X):
        return np.hstack([X, np.ones((X.shape[0], 1))]) @ M

    # anchor train/val split for layer+alpha selection
    n = src_anchor.shape[0]
    perm = rng.permutation(n); tr, va = perm[: int(0.8 * n)], perm[int(0.8 * n):]
    best = None
    for L in cand:
        for alpha in (10.0, 100.0, 1000.0):
            M = fit_map(tgt_anchor[L][tr], src_anchor[tr], alpha)
            pred = apply_map(M, tgt_anchor[L][va])
            r2 = 1 - ((pred - src_anchor[va]) ** 2).sum() / (((src_anchor[va] - src_anchor[va].mean(0)) ** 2).sum() + 1e-9)
            if best is None or r2 > best[0]:
                best = (r2, L, alpha)
    r2, L, alpha = best
    print(f"selected target layer {L} (anchor val R2 {r2:.3f}, alpha {alpha})", flush=True)

    # refit on all anchors, transfer to test
    M = fit_map(tgt_anchor[L], src_anchor, alpha)
    trans_t = apply_map(M, tgt_test_t[L]) @ w + b
    trans_f = apply_map(M, tgt_test_f[L]) @ w + b
    transferred = paired_acc(trans_t, trans_f)
    gap = float(np.mean(trans_t - trans_f) / (np.std(np.concatenate([trans_t, trans_f])) + 1e-9))

    # floor: random unit directions of matched norm through the SAME map
    wn = np.linalg.norm(w)
    floor = []
    for _ in range(100):
        rw = rng.standard_normal(w.shape); rw = rw / np.linalg.norm(rw) * wn
        floor.append(paired_acc(apply_map(M, tgt_test_t[L]) @ rw, apply_map(M, tgt_test_f[L]) @ rw))
    floor95 = float(np.percentile(floor, 95))

    # P2: random map control (no learned alignment)
    Mr = rng.standard_normal(M.shape) * (np.std(M))
    rand_map = paired_acc(apply_map(Mr, tgt_test_t[L]) @ w + b, apply_map(Mr, tgt_test_f[L]) @ w + b)

    void = ceil < 0.60
    portable = (transferred >= 0.65) and (transferred > floor95) and (transferred > 0.5)
    verdict = "VOID-PIPELINE" if void else ("CONSCIENCE-PORTABLE" if portable else "MODEL-SPECIFIC")

    out = {"experiment": "portable conscience v0 — does honesty transfer across minds?",
           "prereg": "papers/showcase-viz/PREREG_portable_conscience_v0_2026_06_10.md",
           "source": SRC, "source_layer": SRC_LAYER, "target": TGT, "target_layer_selected": L,
           "anchor_n": len(a_txt), "test_pairs": len(TEST), "ridge_alpha": alpha,
           "anchor_val_r2": round(r2, 4),
           "ceiling_gemma_self_paired_acc": round(ceil, 4),
           "transferred_paired_acc": round(transferred, 4),
           "transferred_effect_size": round(gap, 4),
           "random_direction_floor_p95": round(floor95, 4),
           "random_map_control_paired_acc": round(rand_map, 4),
           "chance": 0.5, "verdict": verdict}
    (HERE / "portable_conscience_result.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
