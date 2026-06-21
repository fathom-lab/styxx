# -*- coding: utf-8 -*-
"""run_g0_stage1b.py — RUNG 1b (PREREG_writelayer_decouple_2026_06_21): decouple READ point from WRITE point.

Rung 1 found READ≠WRITE but native steering was itself weak (0.051) at the read-optimal SHALLOW layer (dst 6),
confounding the write-null. This finds the STEER-optimal write layer by native efficacy, instruments the
label-free map THERE, and re-tests the write — settling whether the null was an operating-point artifact or real.

  python run_g0_stage1b.py --src meta-llama/Llama-3.2-3B-Instruct --dst meta-llama/Llama-3.2-1B-Instruct --tag llama3b
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "introspection-gate"))
from styxx_transfer import TransferMap, self_test
from introspection_gate import load_model, make_hook, DEVICE
import run_thought_transfer as P
import run_disjoint_worlds as RDW
from run_g0clear import CONCEPTS, split_concepts, extract_multi

WRITE_FRACS = [0.50, 0.60, 0.70]      # write-layer candidates (skip shallow ~0.39: Rung 1 showed it weak)
N_NATIVE_SEL = 12                      # held-out concepts used to pick the steer-optimal layer


def run(src, dst, tag, g0tag="llama3b"):
    s0 = json.loads((HERE / f"g0clear_result_{g0tag}.json").read_text(encoding="utf-8"))  # G0 is a SOURCE property (DST-independent)
    if not s0.get("G0_pass"):
        print("SEALED: G0 not cleared."); return None
    kstar = s0["locked"]["k"]
    from transformers import AutoConfig
    from sentence_transformers import SentenceTransformer
    nlA = AutoConfig.from_pretrained(src).num_hidden_layers
    nlB = AutoConfig.from_pretrained(dst).num_hidden_layers
    print(f"write-layer sweep fracs={WRITE_FRACS}; k={kstar}", flush=True)

    P.CONCEPTS = CONCEPTS
    tr, _, fin = split_concepts(seed=0)
    native_sel = fin[:N_NATIVE_SEL]
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
    cemb = {c: st.encode([c], normalize_embeddings=True)[0] for c in fin}

    # ---- model A: concept clouds + held-out dirs at all candidate SRC layers, then free ----
    src_layers = sorted({round(f * nlA) for f in WRITE_FRACS})
    tokA, mA = load_model(src)
    ptsA, vecsA = extract_multi(mA, tokA, src_layers)
    del mA; torch.cuda.empty_cache()

    # ---- model B: concept clouds + native dirs at all candidate DST layers ----
    dst_layers = sorted({round(f * nlB) for f in WRITE_FRACS})
    tokB, mB = load_model(dst)
    ptsB, vecsB = extract_multi(mB, tokB, dst_layers)

    # ---- Stage A: native-steering sweep on B -> pick steer-optimal layer ----
    sweep = {}
    for f in WRITE_FRACS:
        Lf = round(f * nlB)
        state = {"vec": None, "alpha": 0.0}
        h = mB.model.layers[Lf].register_forward_hook(make_hook(state))
        alpha = P.lock_dose(mB, tokB, state, Lf, vecsB[Lf], native_sel[:5], st, cemb)
        gains = [P.steer_gain(mB, tokB, state, Lf, vecsB[Lf][c], c, st, cemb[c], alpha) for c in native_sel]
        h.remove()
        sweep[f] = {"layer": Lf, "alpha": alpha, "native_sel_gain": float(np.mean(gains))}
        print(f"   frac {f}: dst layer {Lf}, alpha {alpha}, native(sel) {sweep[f]['native_sel_gain']:+.4f}", flush=True)
    f_star = max(WRITE_FRACS, key=lambda f: sweep[f]["native_sel_gain"])
    L_w, alpha_w = sweep[f_star]["layer"], sweep[f_star]["alpha"]
    src_L = round(f_star * nlA)
    print(f"   STEER-OPTIMAL frac {f_star} -> src layer {src_L}, dst layer {L_w}, alpha {alpha_w}", flush=True)

    # ---- Stage B: instrument the label-free map at f_star ----
    RA_tr = np.array([ptsA[src_L][c] for c in tr])
    RB_tr = np.array([ptsB[L_w][c] for c in tr])
    tm = TransferMap.fit(RA_tr, RB_tr, k=kstar)
    rng = np.random.default_rng(0)
    Qrand, _ = np.linalg.qr(rng.standard_normal((tm.Q.shape[0], tm.Q.shape[0])))
    tm_rand = TransferMap(tm.meanA, tm.VAk, Qrand, tm.meanB, tm.VBk, 0.0)
    pc_cos, _ = self_test(RA_tr, k=kstar, test_dirs=np.array([vecsA[src_L][c] for c in fin]))
    rsa = float(np.corrcoef(RDW.distmat(np.array([ptsA[src_L][c] for c in CONCEPTS]))[np.triu_indices(len(CONCEPTS),1)],
                            RDW.distmat(np.array([ptsB[L_w][c] for c in CONCEPTS]))[np.triu_indices(len(CONCEPTS),1)])[0,1])
    print(f"   pos control pc_cos at src layer {src_L} = {pc_cos:.4f} (need >=0.80); RSA(src_L,L_w)={rsa:.3f}", flush=True)

    # ---- Stage C: full write test at L_w (always run; verdict tree gates on the full numbers) ----
    state = {"vec": None, "alpha": 0.0}
    h = mB.model.layers[L_w].register_forward_hook(make_hook(state))
    rows = []
    for c in fin:
        vT = tm.transfer_direction(vecsA[src_L][c])
        vR = tm_rand.transfer_direction(vecsA[src_L][c])
        wrong = str(rng.choice([x for x in fin if x != c]))
        vW = tm.transfer_direction(vecsA[src_L][wrong])
        g_t = P.steer_gain(mB, tokB, state, L_w, vT, c, st, cemb[c], alpha_w)
        g_n = P.steer_gain(mB, tokB, state, L_w, vecsB[L_w][c], c, st, cemb[c], alpha_w)
        g_r = P.steer_gain(mB, tokB, state, L_w, vR, c, st, cemb[c], alpha_w)
        g_w = P.steer_gain(mB, tokB, state, L_w, vW, c, st, cemb[c], alpha_w)
        rows.append({"concept": c, "transfer": g_t, "native": g_n, "randomQ": g_r, "wrong": g_w})
    h.remove(); del mB; torch.cuda.empty_cache()

    def m(key):
        return float(np.mean([r[key] for r in rows]))
    mt, mn, mr, mw = m("transfer"), m("native"), m("randomQ"), m("wrong")
    nte = mt / mn if mn > 1e-6 else float("nan")
    sign_vs_random = int(sum(1 for r in rows if r["transfer"] > r["randomQ"]))
    native_full = mn

    G1 = mt >= 0.15
    G2 = (mt - mr) >= 0.10 and sign_vs_random >= 0.7 * len(rows)
    G3 = (nte == nte) and nte >= 0.40
    if native_full < 0.15:
        verdict = "VOID-NO-STEER (even native steering < 0.15 at the steer-optimal layer; write line VOID here)"
    elif pc_cos < 0.80:
        verdict = "STRUCTURAL-TENSION (steer-optimal layer's subspace does not host the steering vector; read-good != steer-good)"
    elif G1 and G2 and G3:
        verdict = "WRITE-WORKS-AT-STEER-LAYER (Rung 1 read!=write was an OPERATING-POINT ARTIFACT)"
    else:
        verdict = "CLEAN READ!=WRITE (un-confounded: native steers here, but the label-free transfer does not)"

    out = {"experiment": "Rung 1b — decouple read point from write point",
           "prereg": "PREREG_writelayer_decouple_2026_06_21.md", "src": src, "dst": dst,
           "steer_optimal_frac": f_star, "src_layer": src_L, "dst_write_layer": L_w, "dose_alpha": alpha_w, "k": kstar,
           "native_sweep": {str(f): sweep[f] for f in WRITE_FRACS},
           "pc_cos_poscontrol": round(float(pc_cos), 4), "rsa_at_writelayer": round(rsa, 3),
           "n_heldout": len(fin),
           "mean_transfer_gain": round(mt, 4), "mean_native_gain": round(mn, 4),
           "mean_randomQ_gain": round(mr, 4), "mean_wrong_gain": round(mw, 4),
           "transfer_over_native_NTE": round(nte, 3), "transfer_beats_random": f"{sign_vs_random}/{len(rows)}",
           "gates": {"G1_effect": bool(G1), "G2_vs_random": bool(G2), "G3_ceiling": bool(G3),
                     "native_ge_0.15": bool(native_full >= 0.15), "pc_cos_ge_0.80": bool(pc_cos >= 0.80)},
           "rung1_baseline": {"read_top1": 0.586, "write_transfer": 0.0202, "native_at_shallow": 0.0508, "dst_layer": 6},
           "VERDICT": verdict, "rows": rows,
           "honest_scope": "label-free (NOT zero-paired); read at read-optimal, write at steer-optimal; Llama-3B->1B; rate + controls"}
    (HERE / f"writelayer_decouple_result_{tag}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps({k2: v for k2, v in out.items() if k2 != "rows"}, indent=2), flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--dst", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--tag", default="llama3b")
    ap.add_argument("--g0tag", default="llama3b")
    a = ap.parse_args(argv)
    run(a.src, a.dst, a.tag, a.g0tag)


if __name__ == "__main__":
    main()
