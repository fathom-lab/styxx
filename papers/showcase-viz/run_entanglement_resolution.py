"""Is the conscience-axis entanglement REAL, ARTIFACT, or WHITENING-removable?

PREREG_entanglement_resolution_2026_06_11.md (frozen, committed 5a510a5). SEED=0.
Receipt: entanglement_resolution_result.json. Backlog B28. Resolves cycle 2's ambiguity
(FINDING_axis_independence, PARTIAL-STRUCTURED) with the correct nulls.

For each truth<->refusal off-diagonal cell, in gemma AND mapped into Llama-3.2-3B:
  obs = discriminability max(AUROC, 1-AUROC);
  permutation null (K=1000 label shuffles) -> the small-n floor;
  random-direction null (1000 isotropic unit dirs) -> the "any direction separates this axis" floor;
  SPECIFIC-REAL = obs > permnull_p95 AND obs > randdir_p95.
Plus a ZCA-whitening refit and a Gram-Schmidt control. Verdict per the frozen gates.

SAFETY SCOPE (frozen): refusal = one-line refuse-worthy intent REQUESTS; valence = benign/truthful;
NO model generates a response (last-token, pre-output); NO operational harmful content anywhere.

Usage: python papers/showcase-viz/run_entanglement_resolution.py
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
from run_portable_values_refusal import TRAIN_PAIRS, OOD_PAIRS, build_pairs  # noqa: E402
from run_axis_independence import VALENCE_PAIRS  # noqa: E402

SRC = "google/gemma-2-2b-it"
COMMON_LAYER = 12
PRIMARY = "meta-llama/Llama-3.2-3B-Instruct"
SECONDARY = "Qwen/Qwen2.5-3B-Instruct"
SEED = 0
N_TRUTH, N_REFUSAL, N_VALENCE = 88, 88, 48
K_PERM, N_RAND = 1000, 1000
WHITEN_EPS = 1e-3
OFF_CEIL, DIAG_FLOOR = 0.65, 0.75


def build_valence():
    S = []
    for pos, neg in VALENCE_PAIRS:
        S += [(pos, 1, "valence"), (neg, 0, "valence")]
    return S


def cap(S, n, rng):
    S = list(S); rng.shuffle(S)
    return S[:n]


def split(S, rng, frac=0.70):
    S = list(S); rng.shuffle(S)
    k = int(frac * len(S))
    return S[:k], S[k:]


def resid(model, tok, texts, L):
    import torch
    dev = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for s in texts:
            ids = tok(s, return_tensors="pt").input_ids.to(dev)
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
            out.append(hs[L][0, -1, :].float().cpu().numpy())
    return np.stack(out)


def resid_band(model, tok, texts, layers):
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


def discrim(scores, labels):
    a = auroc(scores, labels)
    return max(a, 1.0 - a)


def fit_direction(acts, labels):
    labels = np.asarray(labels)
    w = acts[labels == 1].mean(0) - acts[labels == 0].mean(0)
    return w / (np.linalg.norm(w) + 1e-9)


def cosine(a, b):
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def fit_map(X, Y, alpha):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return np.linalg.solve(Xb.T @ Xb + alpha * np.eye(Xb.shape[1]), Xb.T @ Y)


def apply_map(M, X):
    return np.hstack([X, np.ones((X.shape[0], 1))]) @ M


def zca(train, eps):
    mu = train.mean(0)
    Xc = train - mu
    Sigma = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    s, V = np.linalg.eigh(Sigma)
    s = np.clip(s, 0, None)
    W = V @ np.diag(1.0 / np.sqrt(s + eps)) @ V.T
    return mu, W


def cell_stats(test_acts, labels, direction, rng):
    """obs discriminability + permutation null + random-direction null for one off-diagonal cell."""
    labels = np.asarray(labels)
    score = test_acts @ direction
    obs = discrim(score, labels)
    perm = np.array([discrim(score, rng.permutation(labels)) for _ in range(K_PERM)])
    dim = test_acts.shape[1]
    rand = np.empty(N_RAND)
    for i in range(N_RAND):
        r = rng.standard_normal(dim); r /= (np.linalg.norm(r) + 1e-9)
        rand[i] = discrim(test_acts @ r, labels)
    permnull_p95 = float(np.percentile(perm, 95))
    randdir_p95 = float(np.percentile(rand, 95))
    p_perm = float((1 + int((perm >= obs).sum())) / (1 + K_PERM))
    specific_real = bool(obs > permnull_p95 and obs > randdir_p95)
    return {"obs": round(obs, 4), "permnull_p95": round(permnull_p95, 4),
            "randdir_p95": round(randdir_p95, 4), "p_perm": round(p_perm, 4),
            "specific_real": specific_real}


def free_gpu(model):
    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rng = np.random.default_rng(SEED)

    truth = cap(truth_train() + truth_ood(), N_TRUTH, rng)
    refusal = cap(build_pairs(TRAIN_PAIRS + OOD_PAIRS), N_REFUSAL, rng)
    valence = cap(build_valence(), N_VALENCE, rng)
    AX = {"truth": truth, "refusal": refusal, "valence": valence}
    tr, te = {}, {}
    for k, S in AX.items():
        tr[k], te[k] = split(S, rng)
        print(f"{k}: {len(S)} -> train {len(tr[k])} / test {len(te[k])}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    L = COMMON_LAYER

    # ---- gemma ----
    print("source gemma ...", flush=True)
    stok = AutoTokenizer.from_pretrained(SRC)
    smdl = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype=torch.float16).to(dev).eval()
    g_tr = {k: resid(smdl, stok, [t for t, _, _ in tr[k]], L) for k in AX}
    g_te = {k: resid(smdl, stok, [t for t, _, _ in te[k]], L) for k in AX}
    free_gpu(smdl)
    tr_lab = {k: np.array([l for _, l, _ in tr[k]]) for k in AX}
    te_lab = {k: np.array([l for _, l, _ in te[k]]) for k in AX}

    w = {k: fit_direction(g_tr[k], tr_lab[k]) for k in AX}
    A = round(auroc(g_te["truth"] @ w["truth"], te_lab["truth"]), 4)
    D = round(auroc(g_te["refusal"] @ w["refusal"], te_lab["refusal"]), 4)
    print(f"gemma diagonals: truth A {A} | refusal D {D}", flush=True)
    print("gemma off-diagonal nulls ...", flush=True)
    B = cell_stats(g_te["refusal"], te_lab["refusal"], w["truth"], rng)   # w_truth reads refusal
    C = cell_stats(g_te["truth"], te_lab["truth"], w["refusal"], rng)     # w_refusal reads truth
    cos_tr = round(cosine(w["truth"], w["refusal"]), 4)
    print(f"  B (truth->refusal) obs {B['obs']} perm95 {B['permnull_p95']} rand95 {B['randdir_p95']} "
          f"specific_real {B['specific_real']}", flush=True)
    print(f"  C (refusal->truth) obs {C['obs']} perm95 {C['permnull_p95']} rand95 {C['randdir_p95']} "
          f"specific_real {C['specific_real']}", flush=True)

    # ---- whitening ----
    pooled = np.vstack([g_tr[k] for k in AX])
    mu, Wm = zca(pooled, WHITEN_EPS)
    gw_tr = {k: (g_tr[k] - mu) @ Wm for k in AX}
    gw_te = {k: (g_te[k] - mu) @ Wm for k in AX}
    ww = {k: fit_direction(gw_tr[k], tr_lab[k]) for k in AX}
    A_w = round(auroc(gw_te["truth"] @ ww["truth"], te_lab["truth"]), 4)
    D_w = round(auroc(gw_te["refusal"] @ ww["refusal"], te_lab["refusal"]), 4)
    B_w = round(discrim(gw_te["refusal"] @ ww["truth"], te_lab["refusal"]), 4)
    C_w = round(discrim(gw_te["truth"] @ ww["refusal"], te_lab["truth"]), 4)
    cos_tr_w = round(cosine(ww["truth"], ww["refusal"]), 4)
    print(f"whitened: A {A_w} D {D_w} | B {B_w} C {C_w} | cos {cos_tr_w}", flush=True)

    # ---- Gram-Schmidt (descriptive) ----
    def gs(a, b):  # a orthogonalized against b
        p = a - (a @ b) * b
        return p / (np.linalg.norm(p) + 1e-9)
    w_ref_perp = gs(w["refusal"], w["truth"])
    w_truth_perp = gs(w["truth"], w["refusal"])
    gs_res = {"refusal_perp_on_refusal_AUROC": round(auroc(g_te["refusal"] @ w_ref_perp, te_lab["refusal"]), 4),
              "refusal_perp_on_truth_DISCRIM": round(discrim(g_te["truth"] @ w_ref_perp, te_lab["truth"]), 4),
              "truth_perp_on_truth_AUROC": round(auroc(g_te["truth"] @ w_truth_perp, te_lab["truth"]), 4),
              "truth_perp_on_refusal_DISCRIM": round(discrim(g_te["refusal"] @ w_truth_perp, te_lab["refusal"]), 4)}
    print(f"gram-schmidt: {gs_res}", flush=True)

    # ---- mapped into Llama-3B (primary) + Qwen-3B (secondary descriptive) ----
    union_tr_txt = [t for k in AX for t, _, _ in tr[k]]
    gemma_union = np.vstack([g_tr[k] for k in AX])

    def run_target(TGT):
        print(f"target {TGT} ...", flush=True)
        ttok = AutoTokenizer.from_pretrained(TGT)
        tmdl = AutoModelForCausalLM.from_pretrained(TGT, torch_dtype=torch.float16, trust_remote_code=True).to(dev).eval()
        nL = tmdl.config.num_hidden_layers
        cand = list(range(nL // 3, int(0.8 * nL) + 1, 2))
        t_un = resid_band(tmdl, ttok, union_tr_txt, cand)
        t_te = {k: resid_band(tmdl, ttok, [t for t, _, _ in te[k]], cand) for k in AX}
        free_gpu(tmdl)
        perm = rng.permutation(len(union_tr_txt)); a, b = perm[: int(0.8 * len(perm))], perm[int(0.8 * len(perm)):]
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
        m_te = {k: apply_map(Mmap, t_te[k][Lc]) for k in AX}
        Am = round(auroc(m_te["truth"] @ w["truth"], te_lab["truth"]), 4)
        Dm = round(auroc(m_te["refusal"] @ w["refusal"], te_lab["refusal"]), 4)
        Bm = cell_stats(m_te["refusal"], te_lab["refusal"], w["truth"], rng)
        Cm = cell_stats(m_te["truth"], te_lab["truth"], w["refusal"], rng)
        return {"map_layer": int(Lc), "map_alpha": alpha, "map_val_r2": round(float(r2), 4),
                "Am": Am, "Dm": Dm, "B": Bm, "C": Cm}

    primary = run_target(PRIMARY)
    print(f"  [{PRIMARY}] Am {primary['Am']} Dm {primary['Dm']} | B specific_real {primary['B']['specific_real']} "
          f"(obs {primary['B']['obs']} rand95 {primary['B']['randdir_p95']}) | C specific_real {primary['C']['specific_real']}", flush=True)
    try:
        secondary = run_target(SECONDARY)
    except Exception as e:
        secondary = {"error": str(e)}; print(f"  [sec] ERROR {e}", flush=True)

    # ---- verdict ----
    gemma_real = B["specific_real"] and C["specific_real"]
    gemma_artifact = (not B["specific_real"]) and (not C["specific_real"])
    mapped_real = ("error" not in primary) and primary["B"]["specific_real"] and primary["C"]["specific_real"]
    whiten_clean = (B_w <= OFF_CEIL and C_w <= OFF_CEIL and A_w >= DIAG_FLOOR and D_w >= DIAG_FLOOR)

    if gemma_artifact:
        verdict = "ENTANGLEMENT-ARTIFACT"
    elif gemma_real and whiten_clean:
        verdict = "WHITENING-RESOLVES"
    elif gemma_real and mapped_real and not whiten_clean:
        verdict = "ENTANGLEMENT-REAL"
    else:
        verdict = "PARTIAL-STRUCTURED"

    out = {"experiment": "entanglement resolution — REAL vs ARTIFACT vs WHITENING-removable",
           "prereg": "papers/showcase-viz/PREREG_entanglement_resolution_2026_06_11.md",
           "source": SRC, "common_layer": L, "seed": SEED, "k_perm": K_PERM, "n_rand": N_RAND,
           "whiten_eps": WHITEN_EPS, "off_ceiling": OFF_CEIL, "diag_floor": DIAG_FLOOR,
           "n_test": {k: int(len(te[k])) for k in AX},
           "gemma": {"A": A, "D": D, "B_truth_reads_refusal": B, "C_refusal_reads_truth": C,
                     "cos_truth_refusal": cos_tr},
           "whitened": {"A": A_w, "D": D_w, "B": B_w, "C": C_w, "cos_truth_refusal": cos_tr_w,
                        "clean_basis": bool(whiten_clean)},
           "gram_schmidt": gs_res,
           "mapped_primary": {"target": PRIMARY, **primary},
           "mapped_secondary": {"target": SECONDARY, **secondary},
           "decision": {"gemma_specific_real": bool(gemma_real), "gemma_artifact": bool(gemma_artifact),
                        "mapped_specific_real": bool(mapped_real), "whitening_cleans": bool(whiten_clean)},
           "verdict": verdict}
    (HERE / "entanglement_resolution_result.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps({"verdict": verdict, "decision": out["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
