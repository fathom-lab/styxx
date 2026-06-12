"""B31: heavy-machinery content transport. Was cycle-6's telepathy wall a power problem, or real?

PREREG_content_wall_2026_06_12.md (frozen, committed fa7c4ee). SEED=0. Backlog B31.
Receipt: content_wall_result.json. Figure: content_wall.png.

Cycle 6 (CONTENT-WEAK): cross-model concept retrieval collapsed to chance through a label-free linear
map fit on only 40 anchor concept centroids (R2 ~0.06). Here we change ONLY the anchor count -- scale to
~720 per-instance anchor points (~18x) with the SAME readout, same 20 held-out test concepts -- and ask
whether content identity lifts off chance (power problem) or stays walled despite a well-fit map (real).

Usage: python papers/showcase-viz/run_content_wall.py
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_concept_decode import CONCEPTS, TEMPLATES  # noqa: E402  (the cycle-6 battery + templates)

SEED = 0
SRC = "google/gemma-2-2b-it"
PRIMARY = "meta-llama/Llama-3.2-3B-Instruct"
SECONDARY = "Qwen/Qwen2.5-3B-Instruct"
N_TEST = 20
WHITEN_EPS = 1e-3
CHANCE = 1.0 / N_TEST
GATE_3X = 3 * CHANCE          # 0.15
R2_WELLFIT = 0.40

# ~80 extra concrete nouns, DISJOINT from the 60-concept battery (and the 20 test concepts).
EXTRA_ANCHORS = [
    "table", "chair", "window", "door", "roof", "wall", "floor", "stairs", "fence", "gate",
    "shoe", "hat", "glove", "scarf", "jacket", "shirt", "sock", "belt", "ring", "watch",
    "spoon", "fork", "knife", "plate", "cup", "kettle", "pan", "oven", "fridge", "basket",
    "pencil", "paper", "notebook", "ruler", "eraser", "glue", "stamp", "envelope", "folder", "crayon",
    "table lamp", "flower", "grass", "leaf", "root", "seed", "branch", "bush", "vine", "moss",
    "car", "boat", "plane", "wagon", "sled", "canoe", "scooter", "tractor", "ferry", "raft",
    "sun", "moon", "star", "planet", "comet", "rainbow", "puddle", "fog", "frost", "dew",
    "tiger", "wolf", "fox", "deer", "owl", "frog", "bee", "crab", "whale", "goat",
]


def reps_for(model, tok, words, layer):
    import torch
    dev = next(model.parameters()).device
    out = {}
    with torch.no_grad():
        for w in words:
            vs = []
            for tmpl in TEMPLATES:
                ids = tok(tmpl.format(w=w), return_tensors="pt").input_ids.to(dev)
                hs = model(input_ids=ids, output_hidden_states=True).hidden_states
                vs.append(hs[layer][0, -1, :].float().cpu().numpy())
            out[w] = np.stack(vs)
    return out


def centroids(rep, words):
    return np.stack([rep[w].mean(0) for w in words])


def points(rep, words):
    return np.vstack([rep[w] for w in words])


def fit_map(X, Y, alpha):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return np.linalg.solve(Xb.T @ Xb + alpha * np.eye(Xb.shape[1]), Xb.T @ Y)


def apply_map(M, X):
    return np.hstack([X, np.ones((X.shape[0], 1))]) @ M


def zca(train, eps):
    mu = train.mean(0); Xc = train - mu
    S = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    s, V = np.linalg.eigh(S); s = np.clip(s, 0, None)
    return mu, V @ np.diag(1.0 / np.sqrt(s + eps)) @ V.T


def zca_shrink(train, lam=0.5, eps=1e-8):
    mu = train.mean(0); Xc = train - mu
    S = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    d = S.shape[0]; S = (1 - lam) * S + lam * (np.trace(S) / d) * np.eye(d)
    s, V = np.linalg.eigh(S); s = np.clip(s, eps, None)
    return mu, V @ np.diag(1.0 / np.sqrt(s)) @ V.T


def l2n(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)


def retrieval(query, db):
    sims = l2n(query) @ l2n(db).T
    order = np.argsort(-sims, axis=1)
    truth = np.arange(query.shape[0])
    ranks = np.array([np.where(order[i] == truth[i])[0][0] for i in range(len(truth))])
    return {"top1": round(float((ranks == 0).mean()), 4),
            "top5": round(float((ranks < 5).mean()), 4),
            "mrr": round(float((1.0 / (ranks + 1)).mean()), 4)}


def item_top1(query_items, item_truth, db):
    order = np.argsort(-(l2n(query_items) @ l2n(db).T), axis=1)
    return round(float((order[:, 0] == item_truth).mean()), 4)


def free_gpu(model):
    import torch
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rng = np.random.default_rng(SEED)

    # reconstruct cycle-6's exact split
    words = list(CONCEPTS); rng.shuffle(words)
    test_w = sorted(words[:N_TEST]); cyc6_anchor = sorted(words[N_TEST:])
    anchor_w = sorted(set(cyc6_anchor) | set(EXTRA_ANCHORS))
    assert not (set(anchor_w) & set(test_w)), "anchor/test leakage"
    n_anchor_pts = len(anchor_w) * len(TEMPLATES)
    print(f"test {len(test_w)} | anchor concepts {len(anchor_w)} (cyc6 {len(cyc6_anchor)} + extra {len(EXTRA_ANCHORS)}) "
          f"| anchor POINTS {n_anchor_pts} (cyc6 used 40 centroids) | chance {CHANCE:.3f}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def load(name):
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, trust_remote_code=True).to(dev).eval()
        Lr = round(0.5 * mdl.config.num_hidden_layers)
        ra = reps_for(mdl, tok, anchor_w, Lr)
        rt = reps_for(mdl, tok, test_w, Lr)
        free_gpu(mdl)
        return ra, rt, Lr

    print("reference gemma ...", flush=True)
    g_ra, g_rt, gL = load(SRC)
    g_anchor_pts = points(g_ra, anchor_w)
    g_test = centroids(g_rt, test_w)
    mu, Wm = zca(g_anchor_pts, WHITEN_EPS)       # gemma whitening on the (now large) anchor distribution
    g_test_white = (g_test - mu) @ Wm

    # in-model ceiling
    g_items = np.vstack([(g_rt[w] - mu) @ Wm for w in test_w])
    g_item_truth = np.repeat(np.arange(len(test_w)), len(TEMPLATES))
    ceiling = item_top1(g_items, g_item_truth, g_test_white)
    print(f"gemma L{gL} | in-model ceiling item top1 {ceiling}", flush=True)

    def run_target(name):
        print(f"target {name} ...", flush=True)
        t_ra, t_rt, tL = load(name)
        t_anchor_pts = points(t_ra, anchor_w)
        t_test = centroids(t_rt, test_w)
        # alpha by held-out R2 on anchor POINTS
        perm = rng.permutation(t_anchor_pts.shape[0]); k = int(0.8 * len(perm)); tr, va = perm[:k], perm[k:]
        best = None
        for alpha in (1.0, 10.0, 100.0, 1000.0):
            M = fit_map(t_anchor_pts[tr], g_anchor_pts[tr], alpha)
            pred = apply_map(M, t_anchor_pts[va])
            r2 = 1 - ((pred - g_anchor_pts[va]) ** 2).sum() / (((g_anchor_pts[va] - g_anchor_pts[va].mean(0)) ** 2).sum() + 1e-9)
            if best is None or r2 > best[0]:
                best = (r2, alpha)
        r2, alpha = best
        M = fit_map(t_anchor_pts, g_anchor_pts, alpha)
        mapped_test = apply_map(M, t_test)
        mapped_items = np.vstack([apply_map(M, t_rt[w]) for w in test_w])
        it_truth = np.repeat(np.arange(len(test_w)), len(TEMPLATES))

        # GATE readout (cycle-6-identical: gemma ZCA whiten, cosine retrieval)
        gw = retrieval((mapped_test - mu) @ Wm, g_test_white)
        gw_item = item_top1((mapped_items - mu) @ Wm, it_truth, g_test_white)
        # descriptive: mapped-space whitening (B32 recipe) + raw
        mu_m, W_m = zca_shrink(apply_map(M, t_anchor_pts))
        mw = retrieval((mapped_test - mu_m) @ W_m, (g_test - mu_m) @ W_m)
        raw = retrieval(mapped_test, g_test)
        # random-map null (top1 p95) on the gate readout
        sM = np.std(M); null = np.empty(200)
        for i in range(200):
            Mr = rng.standard_normal(M.shape) * sM
            null[i] = retrieval((apply_map(Mr, t_test) - mu) @ Wm, g_test_white)["top1"]
        return {"layer": int(tL), "map_alpha": alpha, "map_val_r2": round(float(r2), 4),
                "gate_gemma_whitened": gw, "gate_item_top1": gw_item,
                "mapped_whitened": mw, "raw": raw,
                "randmap_top1_p95": round(float(np.percentile(null, 95)), 4)}

    primary = run_target(PRIMARY)
    print(f"  [{PRIMARY}] R2 {primary['map_val_r2']} | gate {primary['gate_gemma_whitened']} "
          f"| item {primary['gate_item_top1']} | randmap_p95 {primary['randmap_top1_p95']}", flush=True)
    try:
        secondary = run_target(SECONDARY)
        print(f"  [sec {SECONDARY}] R2 {secondary['map_val_r2']} | gate {secondary['gate_gemma_whitened']}", flush=True)
    except Exception as e:
        secondary = {"error": str(e)}; print(f"  [sec] ERROR {e}", flush=True)

    acc1 = primary["gate_gemma_whitened"]["top1"]; r2 = primary["map_val_r2"]
    rp95 = primary["randmap_top1_p95"]; top5 = primary["gate_gemma_whitened"]["top5"]
    transports = (acc1 >= GATE_3X) and (acc1 > rp95)
    wall = (acc1 < GATE_3X or acc1 <= rp95) and (r2 >= R2_WELLFIT)
    power = (acc1 < GATE_3X) and (r2 < R2_WELLFIT)
    verdict = ("CONTENT-TRANSPORTS" if transports else
               "CONTENT-WALL" if wall else
               "POWER-BOUND" if power else "PARTIAL")

    out = {"experiment": "B31 content-wall — heavy-machinery cross-model content transport",
           "prereg": "papers/showcase-viz/PREREG_content_wall_2026_06_12.md",
           "reference": SRC, "seed": SEED, "n_test": N_TEST, "chance_top1": round(CHANCE, 4),
           "three_x_chance": round(GATE_3X, 4), "r2_wellfit_threshold": R2_WELLFIT,
           "n_anchor_concepts": len(anchor_w), "n_anchor_points": n_anchor_pts,
           "cycle6_baseline_llama_top1": 0.0, "cycle6_anchor_count": 40,
           "test_concepts": test_w, "gemma_layer": int(gL), "in_model_ceiling_item_top1": ceiling,
           "mapped_primary": {"target": PRIMARY, **primary},
           "mapped_secondary": ({"target": SECONDARY, **secondary}),
           "verdict": verdict,
           "NOTE": "Even CONTENT-TRANSPORTS is not telepathy: needs white-box reps + paired anchors, "
                   "fixed N-way; zero-paired closed-neg, cross-vendor killed. CONTENT-WALL bounds LINEAR transport."}
    (HERE / "content_wall_result.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sec_top1 = secondary.get("gate_gemma_whitened", {}).get("top1", 0.0) if "error" not in secondary else 0.0
        labels = ["chance", "cycle-6\n(40 anchors)", "random-map\n(Llama)",
                  f"Llama (this:\n{n_anchor_pts} anchors)", "Qwen\n(this)", "gemma\nceiling"]
        vals = [CHANCE, 0.0, rp95, acc1, sec_top1, ceiling]
        colors = ["#888", "#b04a4a", "#b0843a", "#c0392b", "#7a3fb0", "#2e7d32"]
        fig, ax = plt.subplots(figsize=(8.6, 5.2))
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.8)
        ax.axhline(CHANCE, color="#888", ls="--", lw=0.8)
        ax.axhline(GATE_3X, color="#c0392b", ls=":", lw=0.9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_ylabel("top-1 retrieval accuracy", fontsize=11); ax.set_ylim(0, 1.05)
        ax.set_title(f"Heavy-machinery content transport ({n_anchor_pts} anchor points, map R2 {r2})\n"
                     f"{N_TEST}-way · chance {CHANCE:.2f} · 3x bar {GATE_3X:.2f} · verdict {verdict}", fontsize=12)
        ax.grid(True, axis="y", alpha=0.15)
        fig.tight_layout(); fig.savefig(HERE / "content_wall.png", dpi=140)
        print("figure -> content_wall.png", flush=True)
    except Exception as e:
        print(f"figure FAILED: {e}", flush=True)

    print("\n" + json.dumps({"verdict": verdict, "map_val_r2": r2, "llama_gate_top1": acc1,
                             "llama_top5": top5, "randmap_p95": rp95, "ceiling": ceiling,
                             "cycle6_was": 0.0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
