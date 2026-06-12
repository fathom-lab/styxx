"""The telepathy test: can you decode WHICH concept a target model represents, cross-model & label-free?

PREREG_concept_decode_2026_06_12.md (frozen, committed 7fb1600). SEED=0.
Receipt: concept_decode_result.json. Figure: concept_decode.png.

Cross-model concept retrieval: fit a label-free ridge map target->reference on a DISJOINT anchor set of
concepts + ZCA whitening, then for HELD-OUT concepts the map never saw, map the target representation
into the reference frame and retrieve the nearest reference concept centroid. Far above chance = the
cross-model geometry carries CONTENT identity, not just a value coordinate.

Even a positive result is NOT telepathy (needs white-box reps + paired anchors; fixed N-way) -- see the
prereg's honesty rail. This adjudicates the narrower, falsifiable question.

SAFETY SCOPE: neutral concrete concepts, neutral templates, last-token pre-output, no generation.

Usage: python papers/showcase-viz/run_concept_decode.py
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 0
SRC = "google/gemma-2-2b-it"
PRIMARY = "meta-llama/Llama-3.2-3B-Instruct"
SECONDARY = "Qwen/Qwen2.5-3B-Instruct"
N_TEST = 20
WHITEN_EPS = 1e-3

CONCEPTS = [
    "dog", "cat", "horse", "elephant", "dolphin", "eagle", "spider", "snake", "rabbit", "bear",
    "apple", "bread", "cheese", "rice", "coffee", "honey", "lemon", "pepper", "chocolate", "soup",
    "mountain", "river", "ocean", "forest", "desert", "cloud", "thunder", "snow", "volcano", "island",
    "hammer", "mirror", "clock", "lamp", "bottle", "ladder", "anchor", "umbrella", "candle", "pillow",
    "hospital", "library", "kitchen", "harbor", "bridge", "market", "prison", "temple", "garden", "tunnel",
    "piano", "violin", "telescope", "compass", "engine", "magnet", "balloon", "kite", "drum", "helmet",
]
TEMPLATES = ["{w}", "a {w}", "the {w}", "I see a {w}", "think of a {w}", "this is a {w}"]


def reps(model, tok, layer):
    """Per-(concept, template) last-token hidden states at `layer`. Returns dict concept -> (T, d)."""
    import torch
    dev = next(model.parameters()).device
    out = {}
    with torch.no_grad():
        for w in CONCEPTS:
            vs = []
            for tmpl in TEMPLATES:
                s = tmpl.format(w=w)
                ids = tok(s, return_tensors="pt").input_ids.to(dev)
                hs = model(input_ids=ids, output_hidden_states=True).hidden_states
                vs.append(hs[layer][0, -1, :].float().cpu().numpy())
            out[w] = np.stack(vs)
    return out


def centroids(rep_dict, words):
    return np.stack([rep_dict[w].mean(0) for w in words])


def fit_map(X, Y, alpha):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return np.linalg.solve(Xb.T @ Xb + alpha * np.eye(Xb.shape[1]), Xb.T @ Y)


def apply_map(M, X):
    return np.hstack([X, np.ones((X.shape[0], 1))]) @ M


def zca(train, eps):
    mu = train.mean(0); Xc = train - mu
    Sigma = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    s, V = np.linalg.eigh(Sigma); s = np.clip(s, 0, None)
    return mu, V @ np.diag(1.0 / np.sqrt(s + eps)) @ V.T


def l2n(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)


def retrieval(query, db):
    """query (q, d), db (n, d) reference centroids (same row order = identity). top1, top5, mrr."""
    Q = l2n(query); D = l2n(db)
    sims = Q @ D.T                       # (q, n)
    order = np.argsort(-sims, axis=1)    # ranked db indices per query
    truth = np.arange(query.shape[0])    # query i's true concept is db row i
    ranks = np.array([np.where(order[i] == truth[i])[0][0] for i in range(len(truth))])
    top1 = float((ranks == 0).mean())
    top5 = float((ranks < 5).mean())
    mrr = float((1.0 / (ranks + 1)).mean())
    return {"top1": round(top1, 4), "top5": round(top5, 4), "mrr": round(mrr, 4)}


def free_gpu(model):
    import torch
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rng = np.random.default_rng(SEED)

    words = list(CONCEPTS); rng.shuffle(words)
    test_w = sorted(words[:N_TEST]); anchor_w = sorted(words[N_TEST:])
    chance = 1.0 / N_TEST
    print(f"concepts {len(CONCEPTS)} | anchor {len(anchor_w)} | test {len(test_w)} | chance top1 {chance:.4f}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def load_reps(name):
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, trust_remote_code=True).to(dev).eval()
        L = round(0.5 * mdl.config.num_hidden_layers)
        r = reps(mdl, tok, L)
        free_gpu(mdl)
        return r, L

    print("reference gemma ...", flush=True)
    gref, gL = load_reps(SRC)
    g_anchor = centroids(gref, anchor_w)
    g_test = centroids(gref, test_w)
    mu, Wm = zca(g_anchor, WHITEN_EPS)
    g_test_white = (g_test - mu) @ Wm

    # in-model ceiling: gemma per-item (test) -> nearest gemma test centroid (whitened)
    g_items_q = np.vstack([(gref[w] - mu) @ Wm for w in test_w])
    g_items_truth = np.repeat(np.arange(len(test_w)), len(TEMPLATES))
    Qg = l2n(g_items_q); Dg = l2n(g_test_white)
    g_order = np.argsort(-(Qg @ Dg.T), axis=1)
    ceiling_top1 = float((g_order[:, 0] == g_items_truth).mean())
    print(f"gemma layer {gL} | in-model per-item ceiling top1 {ceiling_top1:.4f}", flush=True)

    def run_target(name):
        print(f"target {name} ...", flush=True)
        tref, tL = load_reps(name)
        t_anchor = centroids(tref, anchor_w)
        t_test = centroids(tref, test_w)
        # alpha by held-out R2 on anchor concepts (internal split)
        perm = rng.permutation(len(anchor_w)); k = int(0.8 * len(perm)); tr, va = perm[:k], perm[k:]
        best = None
        for alpha in (1.0, 10.0, 100.0, 1000.0):
            M = fit_map(t_anchor[tr], g_anchor[tr], alpha)
            pred = apply_map(M, t_anchor[va])
            r2 = 1 - ((pred - g_anchor[va]) ** 2).sum() / (((g_anchor[va] - g_anchor[va].mean(0)) ** 2).sum() + 1e-9)
            if best is None or r2 > best[0]:
                best = (r2, alpha)
        r2, alpha = best
        M = fit_map(t_anchor, g_anchor, alpha)
        # cross-model centroid retrieval (TEST concepts the map never saw)
        mapped = (apply_map(M, t_test) - mu) @ Wm
        cen = retrieval(mapped, g_test_white)
        # per-item cross-model retrieval (single "thought")
        item_q = np.vstack([(apply_map(M, tref[w]) - mu) @ Wm for w in test_w])
        item_truth = np.repeat(np.arange(len(test_w)), len(TEMPLATES))
        io = np.argsort(-(l2n(item_q) @ l2n(g_test_white).T), axis=1)
        item_top1 = float((io[:, 0] == item_truth).mean())
        # random-map null (centroid top1 p95)
        sM = np.std(M); null = np.empty(200)
        for i in range(200):
            Mr = rng.standard_normal(M.shape) * sM
            null[i] = retrieval((apply_map(Mr, t_test) - mu) @ Wm, g_test_white)["top1"]
        rand_p95 = float(np.percentile(null, 95))
        return {"layer": int(tL), "map_alpha": alpha, "map_val_r2": round(float(r2), 4),
                "centroid": cen, "item_top1": round(item_top1, 4),
                "randmap_top1_p95": round(rand_p95, 4)}

    primary = run_target(PRIMARY)
    print(f"  [{PRIMARY}] centroid {primary['centroid']} | item_top1 {primary['item_top1']} "
          f"| randmap_p95 {primary['randmap_top1_p95']}", flush=True)
    try:
        secondary = run_target(SECONDARY)
        print(f"  [sec {SECONDARY}] centroid {secondary['centroid']}", flush=True)
    except Exception as e:
        secondary = {"error": str(e)}; print(f"  [sec] ERROR {e}", flush=True)

    acc1 = primary["centroid"]["top1"]; top5 = primary["centroid"]["top5"]
    decodable = (acc1 >= 3 * chance) and (acc1 > primary["randmap_top1_p95"]) and (top5 >= 0.50)
    weak = (acc1 < 3 * chance) or (acc1 <= primary["randmap_top1_p95"])
    verdict = ("CONTENT-DECODABLE" if decodable else
               "CONTENT-WEAK" if weak else "PARTIAL")

    out = {"experiment": "concept-decode — cross-model, label-free 'which concept' retrieval (the telepathy test)",
           "prereg": "papers/showcase-viz/PREREG_concept_decode_2026_06_12.md",
           "reference": SRC, "seed": SEED, "n_concepts": len(CONCEPTS), "n_anchor": len(anchor_w),
           "n_test": N_TEST, "chance_top1": round(chance, 4), "templates": len(TEMPLATES),
           "test_concepts": test_w,
           "gemma_layer": int(gL), "in_model_ceiling_item_top1": round(ceiling_top1, 4),
           "mapped_primary": {"target": PRIMARY, **primary},
           "mapped_secondary": ({"target": SECONDARY, **secondary}),
           "three_x_chance": round(3 * chance, 4), "verdict": verdict,
           "NOTE": "CONTENT-DECODABLE is NOT telepathy: needs white-box reps + paired anchors, fixed N-way; "
                   "zero-paired is closed-negative, cross-vendor universality killed. One categorical gap, not all."}
    (HERE / "concept_decode_result.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    # figure: retrieval bars vs chance + ceiling
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = ["chance", "random-map\n(Llama)", "Llama→gemma\ncentroid", "Llama→gemma\nper-item",
                  "Qwen→gemma\ncentroid", "gemma in-model\nceiling"]
        sec_c = secondary.get("centroid", {}).get("top1", 0.0) if "error" not in secondary else 0.0
        vals = [chance, primary["randmap_top1_p95"], acc1, primary["item_top1"], sec_c, ceiling_top1]
        colors = ["#888", "#b0843a", "#c0392b", "#e07b39", "#7a3fb0", "#2e7d32"]
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.8)
        ax.axhline(chance, color="#888", ls="--", lw=0.8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_ylabel("top-1 retrieval accuracy", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Cross-model 'which concept' decoding (held-out, label-free)\n"
                     f"{N_TEST}-way · chance {chance:.2f} · verdict {verdict}", fontsize=12)
        ax.grid(True, axis="y", alpha=0.15)
        fig.tight_layout()
        fig.savefig(HERE / "concept_decode.png", dpi=140)
        print("figure -> concept_decode.png", flush=True)
    except Exception as e:
        print(f"figure FAILED: {e}", flush=True)

    print("\n" + json.dumps({"verdict": verdict, "chance": round(chance, 4),
                             "llama_centroid_top1": acc1, "llama_top5": top5,
                             "llama_item_top1": primary["item_top1"],
                             "randmap_p95": primary["randmap_top1_p95"],
                             "ceiling": round(ceiling_top1, 4)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
