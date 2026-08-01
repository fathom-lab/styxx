"""B36 — the write door, per PREREG_b36_write_door_2026_08_01.md (frozen 5d6e2e6).

Turn the b31v2 paired-MLP machinery on the program's own read!=write law at matched
supervision: 392 TRUE pairs fit both maps (linear Procrustes comparator L1 + the b31v2 MLP
M1), directions transfer through M1 by finite difference at the concept point, and steering
runs the committed protocol (make_hook injection at dst layer 11, lock_dose on native, 3
carriers, MiniLM gain) over the 70 held-out concepts. Gates imported frozen from the
writelayer-decouple prereg. `--smoke` = 5 concepts, `_smoke` files, INVALID-only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "introspection-gate"))

from introspection_gate import load_model, make_hook          # noqa: E402
import run_thought_transfer as P                              # noqa: E402
from run_g0clear import CONCEPTS as FULL_CONCEPTS, split_concepts  # noqa: E402
from run_b31v2 import fit_mlp                                 # noqa: E402

SMOKE = "--smoke" in sys.argv
SUFFIX = "_smoke" if SMOKE else ""
SEED = 36
EPS = 4.0
SRC = "meta-llama/Llama-3.2-3B-Instruct"
DST = "meta-llama/Llama-3.2-1B-Instruct"
L_SRC = 11          # G0-locked src read layer
L_DST = 11          # writelayer-decouple steer-optimal dst layer (of 16)


def extract_cached(hf, layer, tag):
    cp = HERE / f"_b36_{tag}{SUFFIX}.npz"
    if cp.exists():
        z = np.load(cp, allow_pickle=True)
        return ({c: z["pts"][i] for i, c in enumerate(P.CONCEPTS)},
                {c: z["vecs"][i] for i, c in enumerate(P.CONCEPTS)})
    tok, m = load_model(hf)
    with torch.no_grad():
        pts, vecs = P.extract(m, tok, layer)
    del m
    torch.cuda.empty_cache()
    np.savez(cp, pts=np.array([pts[c] for c in P.CONCEPTS]),
             vecs=np.array([vecs[c] for c in P.CONCEPTS]))
    return pts, vecs


def main():
    if SMOKE:
        P.CONCEPTS = FULL_CONCEPTS[:15]
        tr, fin = P.CONCEPTS[:10], P.CONCEPTS[10:]
    else:
        P.CONCEPTS = FULL_CONCEPTS
        tr_, sel_, fin = split_concepts(seed=0)
        tr = list(tr_) + list(sel_)

    print(f"[b36{SUFFIX}] src L{L_SRC} -> dst L{L_DST} | {len(tr)} pairs, {len(fin)} held-out",
          flush=True)
    ptsA, vecsA = extract_cached(SRC, L_SRC, "srcA")
    XA = np.array([ptsA[c] for c in tr])
    ptsB, vecsB = extract_cached(DST, L_DST, "dstB")
    XB = np.array([ptsB[c] for c in tr])

    # ---- maps at matched (maximal) supervision: 392 TRUE pairs ----
    # L1 paired linear: orthogonal Procrustes on centered clouds
    muA, muB = XA.mean(0), XB.mean(0)
    dA, dB = XA.shape[1], XB.shape[1]
    d = min(dA, dB)
    U, _, Vt = np.linalg.svd((XA - muA)[:, :dA].T @ (XB - muB), full_matrices=False)
    W = U @ Vt                                     # dA x dB orthogonal-ish map
    def l1_dir(vA):
        vB = vA @ W
        return vB / (np.linalg.norm(vB) + 1e-9)
    # M1 paired MLP (b31v2 trainer verbatim)
    m1_fn, m1_loss = fit_mlp(XA, XB, seed=SEED)
    def m1_dir(c):
        x = ptsA[c]; v = vecsA[c]
        d1 = m1_fn(x + EPS * v) - m1_fn(x)
        d2 = m1_fn(x + (EPS / 2) * v) - m1_fn(x)
        d1 = d1 / (np.linalg.norm(d1) + 1e-9); d2 = d2 / (np.linalg.norm(d2) + 1e-9)
        return d1, float(d1 @ d2)

    # ---- steering on the dst model ----
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=P.DEVICE)
    cemb = {c: st.encode([c], normalize_embeddings=True)[0] for c in fin}
    tokB, mB = load_model(DST)
    state = {"vec": None, "alpha": 0.0}
    h = mB.model.layers[L_DST].register_forward_hook(make_hook(state))
    rng = np.random.default_rng(SEED)

    dose_pool = list(fin)[:8]
    alpha = P.lock_dose(mB, tokB, state, L_DST, vecsB, dose_pool, st, cemb)
    print(f"locked dose alpha={alpha}", flush=True)

    rows = []
    for i, c in enumerate(fin):
        t0 = time.time()
        native = P.steer_gain(mB, tokB, state, L_DST, vecsB[c], c, st, cemb[c], alpha)
        vm, eps_cos = m1_dir(c)
        m1g = P.steer_gain(mB, tokB, state, L_DST, vm, c, st, cemb[c], alpha)
        l1g = P.steer_gain(mB, tokB, state, L_DST, l1_dir(vecsA[c]), c, st, cemb[c], alpha)
        rv = rng.standard_normal(XB.shape[1]); rv /= np.linalg.norm(rv)
        rg = P.steer_gain(mB, tokB, state, L_DST, rv, c, st, cemb[c], alpha)
        rows.append({"concept": c, "native": round(native, 4), "m1": round(m1g, 4),
                     "l1": round(l1g, 4), "random": round(rg, 4),
                     "eps_stability_cos": round(eps_cos, 4)})
        print(f"  [{i+1}/{len(fin)}] {c}: nat={native:+.3f} m1={m1g:+.3f} "
              f"l1={l1g:+.3f} rnd={rg:+.3f} [{time.time()-t0:.0f}s]", flush=True)
        if (i + 1) % 5 == 0:
            (HERE / f"_b36_rows{SUFFIX}.json").write_text(
                json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    h.remove()

    mean = lambda k: float(np.mean([r[k] for r in rows]))
    nat, m1m, l1m, rnd = mean("native"), mean("m1"), mean("l1"), mean("random")
    sign = float(np.mean([r["m1"] > r["random"] for r in rows]))
    nte = m1m / nat if nat else float("nan")
    results = {"prereg": "PREREG_b36_write_door_2026_08_01.md", "seed": SEED,
               "smoke": SMOKE, "alpha": alpha, "eps": EPS, "n_heldout": len(fin),
               "m1_train_loss": round(m1_loss, 5),
               "native_mean": round(nat, 4), "m1_mean": round(m1m, 4),
               "l1_mean": round(l1m, 4), "random_mean": round(rnd, 4),
               "m1_minus_random": round(m1m - rnd, 4), "m1_sign_vs_random": round(sign, 4),
               "nte": round(nte, 4),
               "eps_stability_mean": round(mean("eps_stability_cos"), 4),
               "rows": rows}
    if SMOKE:
        results["verdict"] = "INVALID__smoke_plumbing_only"
    else:
        pc = nat >= 0.15
        g1 = m1m >= 0.15
        g2m = (m1m - rnd) >= 0.10
        g2s = sign >= 0.70
        g3 = nte >= 0.40
        results["gates"] = {"PC_native": pc, "G1_gain": g1, "G2_magnitude": g2m,
                           "G2_sign": g2s, "G3_nte": g3}
        if not pc:
            results["verdict"] = "INVALID__substrate_not_steerable"
        elif g1 and g2m:
            results["verdict"] = "WRITE_DOOR_OPENS__control_was_capacity_limited"
        elif not g1 and not g2m:
            results["verdict"] = "READ_NEQ_WRITE_SURVIVES_CAPACITY"
        else:
            results["verdict"] = "REPORT_AS_LANDED__mixed_gates"
    out = HERE / f"b36_result{SUFFIX}.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nVERDICT: {results['verdict']}  -> {out.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
