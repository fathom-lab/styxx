"""B29: does MAPPED-space whitening (with shrinkage) clear cycle-5's 0.0062 cross-model basis miss?

PREREG_mapped_whitening_2026_06_12.md (frozen, committed 891b8fa). SEED=0. Backlog B29.
Receipt: mapped_whitening_result.json.

Cycle 5 whitened only in the SOURCE (gemma) space and read MAPPED Llama states through gemma's
covariance -> missed the primary by 0.0062 (c_truth_invariant_H 0.6562 vs 0.65). Here we recompute the
whitening metric on the MAPPED anchor distribution (shrunk toward scaled identity), read the same
factorial through it, and ask whether the miss clears or is real geometry.

Usage: python papers/showcase-viz/run_mapped_whitening.py
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_truth_danger_basis import (truth_train, truth_ood, build_danger_train, build_factorial)  # noqa: E402

SRC = "google/gemma-2-2b-it"
L = 12
PRIMARY = "meta-llama/Llama-3.2-3B-Instruct"
SECONDARY = "Qwen/Qwen2.5-3B-Instruct"
SEED = 0
N_TRUTH = 88
SOURCE_EPS = 1e-3          # cycle-5 source whitening regularization (reproduction)
LAMBDAS = [0.2, 0.35, 0.5, 0.65, 0.8]
GATE_LAMBDA = 0.5
ON, OFF = 0.75, 0.65


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


def zca_eps(train, eps):                       # cycle-5 source whitening (reproduction)
    train = np.asarray(train, float)
    mu = train.mean(0); Xc = train - mu
    S = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    s, V = np.linalg.eigh(S); s = np.clip(s, 0, None)
    return mu, V @ np.diag(1.0 / np.sqrt(s + eps)) @ V.T


def zca_shrink(train, lam, eps=1e-8):          # shrink covariance toward scaled identity, then whiten
    train = np.asarray(train, float)
    mu = train.mean(0); Xc = train - mu
    S = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    d = S.shape[0]; tgt = np.trace(S) / d
    S = (1.0 - lam) * S + lam * tgt * np.eye(d)
    s, V = np.linalg.eigh(S); s = np.clip(s, eps, None)
    return mu, V @ np.diag(1.0 / np.sqrt(s)) @ V.T


def read_matrix(mu, W, g_truth, t_lab, g_danger, h_lab, fac_points, T, H):
    """Whiten gemma-labeled train + factorial by (mu, W); fit DiM in that space; read the matrix."""
    wt = fit_direction((g_truth - mu) @ W, t_lab)
    wd = fit_direction((g_danger - mu) @ W, h_lab)
    fw = (fac_points - mu) @ W
    ct = fw @ wt; cd = fw @ wd
    return {"c_truth_recovers_T": round(auroc(ct, T), 4),
            "c_truth_invariant_H": round(discrim(ct, H), 4),
            "c_danger_recovers_H": round(auroc(cd, H), 4),
            "c_danger_invariant_T": round(discrim(cd, T), 4)}


def passes(M):
    return (M["c_truth_recovers_T"] >= ON and M["c_truth_invariant_H"] <= OFF
            and M["c_danger_recovers_H"] >= ON and M["c_danger_invariant_T"] <= OFF)


def free_gpu(model):
    import torch
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()


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
    print(f"truth-train {len(truth_tr)} | danger-train {len(danger_tr)} | factorial {len(fac)} | "
          f"lambdas {LAMBDAS} | gate-lambda {GATE_LAMBDA}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print("source gemma ...", flush=True)
    stok = AutoTokenizer.from_pretrained(SRC)
    smdl = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype=torch.float16).to(dev).eval()
    g_truth = resid(smdl, stok, t_txt, L)
    g_danger = resid(smdl, stok, dg_txt, L)
    g_fac = resid(smdl, stok, f_txt, L)
    free_gpu(smdl)
    pooled = np.vstack([g_truth, g_danger])

    # gemma own-factorial readout under SOURCE whitening (reproduces cycle 5; the "gemma passes" check)
    mu_s, W_s = zca_eps(pooled, SOURCE_EPS)
    gemma_M = read_matrix(mu_s, W_s, g_truth, t_lab, g_danger, dg_lab, g_fac, T, H)
    print(f"gemma (source-whitened own factorial): {json.dumps(gemma_M)} | passes {passes(gemma_M)}", flush=True)

    union_txt = t_txt + dg_txt

    def run_target(TGT, descriptive=False):
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
                Mp = fit_map(t_un[Lc][a], pooled[a], alpha)
                pred = apply_map(Mp, t_un[Lc][b])
                r2 = 1 - ((pred - pooled[b]) ** 2).sum() / (((pooled[b] - pooled[b].mean(0)) ** 2).sum() + 1e-9)
                if best is None or r2 > best[0]:
                    best = (r2, Lc, alpha)
        r2, Lc, alpha = best
        Mmap = fit_map(t_un[Lc], pooled, alpha)
        mapped_fac = apply_map(Mmap, t_fac[Lc])
        mapped_anchor = apply_map(Mmap, t_un[Lc])

        # SOURCE-whitened read of mapped factorial (reproduces cycle-5 Llama readout)
        src = read_matrix(mu_s, W_s, g_truth, t_lab, g_danger, dg_lab, mapped_fac, T, H)
        # MAPPED-whitened reads across the shrinkage sweep
        mapped = {}
        for lam in LAMBDAS:
            mu_m, W_m = zca_shrink(mapped_anchor, lam)
            mapped[f"lam_{lam}"] = read_matrix(mu_m, W_m, g_truth, t_lab, g_danger, dg_lab, mapped_fac, T, H)
        out = {"map_layer": int(Lc), "map_val_r2": round(float(r2), 4),
               "source_whitened": src, "mapped_whitened": mapped}
        if not descriptive:
            inv_by_lam = {lam: mapped[f"lam_{lam}"]["c_truth_invariant_H"] for lam in LAMBDAS}
            print(f"  [{TGT}] source c_truth_inv_H {src['c_truth_invariant_H']} | mapped c_truth_inv_H by lam {inv_by_lam}", flush=True)
        return out

    primary = run_target(PRIMARY)
    try:
        secondary = run_target(SECONDARY, descriptive=True)
    except Exception as e:
        secondary = {"error": str(e)}; print(f"  [sec] ERROR {e}", flush=True)

    # ---- verdict ----
    gate_M = primary["mapped_whitened"][f"lam_{GATE_LAMBDA}"]
    inv_vals = [primary["mapped_whitened"][f"lam_{lam}"]["c_truth_invariant_H"] for lam in LAMBDAS]
    stability = sum(1 for v in inv_vals if v <= OFF)
    gemma_ok = passes(gemma_M)
    cleared = gemma_ok and passes(gate_M) and (stability >= 3)
    miss_real = (gate_M["c_truth_invariant_H"] > OFF) or (stability < 3)
    verdict = ("BASIS-CLEARED" if cleared else
               "MISS-REAL" if miss_real else "PARTIAL")

    out = {"experiment": "B29 mapped-space whitening — clear cycle-5's cross-model basis miss?",
           "prereg": "papers/showcase-viz/PREREG_mapped_whitening_2026_06_12.md",
           "source": SRC, "layer": L, "seed": SEED, "lambdas": LAMBDAS, "gate_lambda": GATE_LAMBDA,
           "on_target": ON, "off_target": OFF,
           "gemma_source_whitened": gemma_M, "gemma_passes": bool(gemma_ok),
           "mapped_primary": {"target": PRIMARY, **primary},
           "mapped_secondary": ({"target": SECONDARY, **secondary}),
           "gate_matrix_lambda_0.5": gate_M,
           "c_truth_invariant_H_by_lambda": {str(lam): primary["mapped_whitened"][f"lam_{lam}"]["c_truth_invariant_H"] for lam in LAMBDAS},
           "stability_count_under_ceiling": int(stability),
           "verdict": verdict}
    (HERE / "mapped_whitening_result.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps({"verdict": verdict, "gemma_passes": bool(gemma_ok),
                             "source_llama_inv_H": primary["source_whitened"]["c_truth_invariant_H"],
                             "gate_matrix": gate_M, "inv_H_by_lambda": inv_vals,
                             "stability": int(stability)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
