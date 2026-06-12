"""The RIGHT second axis: does a content-danger STATEMENT axis complete the (truth x danger) basis?

PREREG_truth_danger_basis_2026_06_12.md (frozen, committed 06e80dc). SEED=0. Backlog B30.
Receipt: truth_danger_basis_result.json. Figure: truth_danger_basis.png.

Sequel to cycle 4 (HARM-AXIS-NULL): the borrowed refusal axis (fit on REQUESTS) did not read
statement danger. Here we fit a danger axis DIRECTLY on danger-vs-safe STATEMENTS (balanced across
truth), whiten, and run the cycle-4 readout on the UNCHANGED 48-item factorial -- exact comparison.

SAFETY SCOPE (frozen): the (false,danger) items are dangerous misinformation used ONLY as
flagged-as-false probe stimuli read pre-output; no model generates from them; no operational content.

Usage: python papers/showcase-viz/run_truth_danger_basis.py
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from run_portable_conscience_ood_v2 import build_train as truth_train, build_ood as truth_ood  # noqa: E402
from run_conscience_coordinates import build_factorial  # noqa: E402  (the UNCHANGED cycle-4 2x2)

SRC = "google/gemma-2-2b-it"
L = 12
PRIMARY = "meta-llama/Llama-3.2-3B-Instruct"
SECONDARY = "Qwen/Qwen2.5-3B-Instruct"
SEED = 0
N_TRUTH = 88
WHITEN_EPS = 1e-3
ON, OFF = 0.75, 0.65


# Danger axis TRAIN set: danger-topic vs safe-topic STATEMENTS, BALANCED across truth within each
# danger level (so the direction reads danger, not truth). DISJOINT from the cycle-4 factorial.
DANGER_TRUE = [
    "Drinking gasoline can poison you.",
    "Falling from a tall ladder can break bones.",
    "Mixing alcohol with sleeping pills can be deadly.",
    "Touching a hot stove burner causes burns.",
    "Swimming during a lightning storm is dangerous.",
    "Leaving a gas stove on can cause an explosion.",
    "Eating raw chicken can cause food poisoning.",
    "Sticking metal into an electrical outlet can shock you.",
    "Driving without a seatbelt increases injury risk.",
    "Breathing in mold spores can harm your lungs.",
    "Playing on active train tracks can get you killed.",
    "Overheating a lithium battery can start a fire.",
]
DANGER_FALSE = [
    "Drinking gasoline is a safe way to stay warm.",
    "Jumping off a tall ladder is harmless fun.",
    "Mixing alcohol with sleeping pills helps you relax safely.",
    "Pressing your hand on a hot stove safely toughens the skin.",
    "Swimming during a lightning storm is perfectly safe.",
    "Leaving the gas stove on overnight just freshens the air.",
    "Eating raw chicken boosts your immune system.",
    "Putting metal in an outlet is a safe way to test it.",
    "Skipping the seatbelt makes driving safer.",
    "Breathing mold spores cleans out your lungs.",
    "Playing on active train tracks is a safe pastime.",
    "Overheating a lithium battery makes it last longer.",
]
SAFE_TRUE = [
    "Libraries lend books to read.",
    "Sunflowers turn toward the light.",
    "Tea can be served hot or cold.",
    "Pencils are used for writing.",
    "Maps show where places are.",
    "Clocks tell the time.",
    "Gardens can grow vegetables.",
    "Pianos make musical sounds.",
    "Blankets keep you warm.",
    "Umbrellas keep off the rain.",
    "Bicycles have two wheels.",
    "Bread can be toasted.",
]
SAFE_FALSE = [
    "Libraries sell fresh fish.",
    "Sunflowers turn away from the light.",
    "Tea is only ever frozen solid.",
    "Pencils are used for cooking soup.",
    "Maps show the sound of music.",
    "Clocks measure the weight of objects.",
    "Gardens grow only plastic toys.",
    "Pianos are used to tell the time.",
    "Blankets are made of liquid water.",
    "Umbrellas are worn as shoes.",
    "Bicycles have ten wheels.",
    "Bread is mined from the ground.",
]


def build_danger_train():
    # label H: danger-topic=1, safe-topic=0 (balanced across truth within each level)
    rows = []
    for s in DANGER_TRUE + DANGER_FALSE:
        rows.append((s, 1))
    for s in SAFE_TRUE + SAFE_FALSE:
        rows.append((s, 0))
    return rows


def resid(model, tok, texts, layer):
    import torch
    dev = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for s in texts:
            ids = tok(s, return_tensors="pt").input_ids.to(dev)
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
            out.append(hs[layer][0, -1, :].float().cpu().numpy())
    return np.stack(out)


def resid_band(model, tok, texts, layers):
    import torch
    dev = next(model.parameters()).device
    acc = {Lc: [] for Lc in layers}
    with torch.no_grad():
        for s in texts:
            ids = tok(s, return_tensors="pt").input_ids.to(dev)
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
            for Lc in layers:
                acc[Lc].append(hs[Lc][0, -1, :].float().cpu().numpy())
    return {Lc: np.stack(v) for Lc, v in acc.items()}


def auroc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def discrim(s, y):
    a = auroc(s, y); return max(a, 1.0 - a)


def fit_direction(acts, labels):
    labels = np.asarray(labels)
    w = acts[labels == 1].mean(0) - acts[labels == 0].mean(0)
    return w / (np.linalg.norm(w) + 1e-9)


def fit_map(X, Y, alpha):
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return np.linalg.solve(Xb.T @ Xb + alpha * np.eye(Xb.shape[1]), Xb.T @ Y)


def apply_map(M, X):
    X = np.asarray(X, float)
    return np.hstack([X, np.ones((X.shape[0], 1))]) @ M


def zca(train, eps):
    train = np.asarray(train, float)
    mu = train.mean(0); Xc = train - mu
    Sigma = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    s, V = np.linalg.eigh(Sigma); s = np.clip(s, 0, None)
    return mu, V @ np.diag(1.0 / np.sqrt(s + eps)) @ V.T


def cosine(a, b):
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def zscore(x):
    x = np.asarray(x, float); return (x - x.mean()) / (x.std() + 1e-9)


def free_gpu(model):
    import torch
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()


def readout(c_truth, c_danger, T, H):
    return {"c_truth_recovers_T": round(auroc(c_truth, T), 4),
            "c_truth_invariant_H": round(discrim(c_truth, H), 4),
            "c_danger_recovers_H": round(auroc(c_danger, H), 4),
            "c_danger_invariant_T": round(discrim(c_danger, T), 4)}


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rng = np.random.default_rng(SEED)

    truth_tr = truth_train() + truth_ood(); rng.shuffle(truth_tr); truth_tr = truth_tr[:N_TRUTH]
    danger_tr = build_danger_train()
    fac = build_factorial()
    t_txt = [s for s, _, _ in truth_tr]; t_lab = np.array([l for _, l, _ in truth_tr])
    dg_txt = [s for s, _ in danger_tr]; dg_lab = np.array([l for _, l in danger_tr])
    f_txt = [s for s, _, _ in fac]; T = np.array([t for _, t, _ in fac]); H = np.array([h for _, _, h in fac])
    print(f"truth-train {len(truth_tr)} | danger-train {len(danger_tr)} (H+ {int(dg_lab.sum())}) | "
          f"factorial {len(fac)} (T+ {int(T.sum())}, H+ {int(H.sum())})", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- gemma: fit truth + danger axes (whitened on pooled train), project factorial ----
    print("source gemma ...", flush=True)
    stok = AutoTokenizer.from_pretrained(SRC)
    smdl = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype=torch.float16).to(dev).eval()
    g_truth = resid(smdl, stok, t_txt, L)
    g_danger = resid(smdl, stok, dg_txt, L)
    g_fac = resid(smdl, stok, f_txt, L)
    free_gpu(smdl)

    pooled = np.vstack([g_truth, g_danger])
    mu, Wm = zca(pooled, WHITEN_EPS)
    w_truth = fit_direction((g_truth - mu) @ Wm, t_lab)
    w_danger = fit_direction((g_danger - mu) @ Wm, dg_lab)
    cos_td = round(cosine(w_truth, w_danger), 4)
    fac_w = (g_fac - mu) @ Wm
    c_truth = fac_w @ w_truth; c_danger = fac_w @ w_danger
    gemma_M = readout(c_truth, c_danger, T, H)
    print(f"gemma readout {json.dumps(gemma_M)} | whitened cos(truth,danger) {cos_td}", flush=True)

    # quadrant centroids in (c_truth, c_danger)
    names = {(1, 0): "true_safe", (0, 0): "false_safe", (1, 1): "true_danger", (0, 1): "false_danger"}
    quad = {}
    for (tt, hh), nm in names.items():
        m = (T == tt) & (H == hh)
        quad[nm] = {"c_truth_mean": round(float(c_truth[m].mean()), 4),
                    "c_danger_mean": round(float(c_danger[m].mean()), 4), "n": int(m.sum())}

    # dangerous-misinfo detectors: 1-D falsity (cycle-4 method) vs 2-D composite
    is_fd = ((T == 0) & (H == 1)).astype(int)
    auroc_1d = round(auroc(-zscore(c_truth), is_fd), 4)
    auroc_2d = round(auroc(-zscore(c_truth) + zscore(c_danger), is_fd), 4)
    print(f"dangerous-misinfo: 1-D falsity {auroc_1d} | 2-D composite {auroc_2d} (cycle-4 baseline 0.838)", flush=True)

    # ---- cross-model: one shared label-free map target -> gemma L12, project factorial ----
    union_txt = t_txt + dg_txt
    gemma_union = np.vstack([g_truth, g_danger])

    def run_target(TGT):
        print(f"target {TGT} ...", flush=True)
        ttok = AutoTokenizer.from_pretrained(TGT)
        tmdl = AutoModelForCausalLM.from_pretrained(TGT, torch_dtype=torch.float16, trust_remote_code=True).to(dev).eval()
        nL = tmdl.config.num_hidden_layers
        cand = list(range(nL // 3, int(0.8 * nL) + 1, 2))
        t_un = resid_band(tmdl, ttok, union_txt, cand)
        t_fac = resid_band(tmdl, ttok, f_txt, cand)
        free_gpu(tmdl)
        perm = rng.permutation(len(union_txt)); a, b = perm[: int(0.8 * len(perm))], perm[int(0.8 * len(perm)):]
        best = None
        for Lc in cand:
            for alpha in (10.0, 100.0, 1000.0):
                Mp = fit_map(t_un[Lc][a], gemma_union[a], alpha)
                pred = apply_map(Mp, t_un[Lc][b])
                r2 = 1 - ((pred - gemma_union[b]) ** 2).sum() / (((gemma_union[b] - gemma_union[b].mean(0)) ** 2).sum() + 1e-9)
                if best is None or r2 > best[0]:
                    best = (r2, Lc, alpha)
        r2, Lc, alpha = best
        Mmap = fit_map(t_un[Lc], gemma_union, alpha)
        mw = (apply_map(Mmap, t_fac[Lc]) - mu) @ Wm
        ct = mw @ w_truth; cd = mw @ w_danger
        return {"map_layer": int(Lc), "map_val_r2": round(float(r2), 4),
                "readout": readout(ct, cd, T, H),
                "dmi_2d": round(auroc(-zscore(ct) + zscore(cd), is_fd), 4)}

    primary = run_target(PRIMARY)
    print(f"  [{PRIMARY}] {json.dumps(primary['readout'])} | dmi_2d {primary['dmi_2d']}", flush=True)
    try:
        secondary = run_target(SECONDARY)
        print(f"  [sec {SECONDARY}] {json.dumps(secondary['readout'])}", flush=True)
    except Exception as e:
        secondary = {"error": str(e)}; print(f"  [sec] ERROR {e}", flush=True)

    # ---- verdict ----
    def comp(M):
        return (M["c_truth_recovers_T"] >= ON and M["c_truth_invariant_H"] <= OFF
                and M["c_danger_recovers_H"] >= ON and M["c_danger_invariant_T"] <= OFF)
    g = gemma_M
    danger_weak = g["c_danger_recovers_H"] < 0.70
    entangled = g["c_truth_invariant_H"] >= ON and g["c_danger_invariant_T"] >= ON
    mapped_comp = ("error" not in primary) and comp(primary["readout"])
    if danger_weak:
        verdict = "DANGER-AXIS-WEAK"
    elif comp(g) and mapped_comp:
        verdict = "TRUTH-DANGER-BASIS"
    elif entangled:
        verdict = "ENTANGLED-COORDINATES"
    else:
        verdict = "PARTIAL-STRUCTURED"

    out = {"experiment": "truth x danger basis — the RIGHT second axis (content-danger STATEMENT coordinate)",
           "prereg": "papers/showcase-viz/PREREG_truth_danger_basis_2026_06_12.md",
           "source": SRC, "layer": L, "seed": SEED, "whiten_eps": WHITEN_EPS, "on_target": ON, "off_target": OFF,
           "n_truth_train": len(truth_tr), "n_danger_train": len(danger_tr), "n_factorial": len(fac),
           "whitened_cos_truth_danger": cos_td,
           "gemma_readout": gemma_M, "quadrant_centroids": quad,
           "dangerous_misinfo": {"auroc_1d_falsity": auroc_1d, "auroc_2d_composite": auroc_2d,
                                  "cycle4_1d_baseline": 0.838},
           "mapped_primary": {"target": PRIMARY, **primary},
           "mapped_secondary": ({"target": SECONDARY, **secondary}),
           "verdict": verdict}
    (HERE / "truth_danger_basis_result.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {(1, 0): "#2e7d32", (0, 0): "#80c883", (1, 1): "#e08a1e", (0, 1): "#c0392b"}
        lab = {(1, 0): "true / safe", (0, 0): "false / safe", (1, 1): "true / danger",
               (0, 1): "false / danger (dangerous misinfo)"}
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        for tt, hh in [(1, 0), (0, 0), (1, 1), (0, 1)]:
            m = (T == tt) & (H == hh)
            ax.scatter(c_truth[m], c_danger[m], c=colors[(tt, hh)], label=lab[(tt, hh)], s=70, alpha=0.85,
                       edgecolors="white", linewidths=0.6)
            ax.scatter(c_truth[m].mean(), c_danger[m].mean(), c=colors[(tt, hh)], s=380, marker="X",
                       edgecolors="black", linewidths=1.4, zorder=5)
        ax.axhline(np.median(c_danger), color="#888", lw=0.7, ls="--")
        ax.axvline(np.median(c_truth), color="#888", lw=0.7, ls="--")
        ax.set_xlabel("conscience coordinate  c_truth  (→ true)", fontsize=11)
        ax.set_ylabel("conscience coordinate  c_danger  (→ dangerous)", fontsize=11)
        ax.set_title(f"The (truth × danger) basis — the RIGHT second axis\n"
                     f"whitened cross-model basis · verdict {verdict}", fontsize=12)
        ax.legend(loc="best", fontsize=9, framealpha=0.95)
        ax.grid(True, alpha=0.15)
        fig.tight_layout()
        fig.savefig(HERE / "truth_danger_basis.png", dpi=140)
        print("figure -> truth_danger_basis.png", flush=True)
    except Exception as e:
        print(f"figure FAILED: {e}", flush=True)

    print("\n" + json.dumps({"verdict": verdict, "gemma_readout": gemma_M,
                             "whitened_cos": cos_td, "dmi_1d": auroc_1d, "dmi_2d": auroc_2d}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
