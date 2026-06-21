# -*- coding: utf-8 -*-
"""run_g0_stage1.py — STAGE 1 of PREREG_thought_transfer_g0clear_2026_06_20.

The cross-model WRITE test, run ONLY after Stage 0 clears G0. Installs a concept STEERING direction
from model A into model B through the zero-anchor map at the (layer, k) Stage 0 LOCKED, and measures
whether it steers B — vs native (ceiling), random-Q (null), wrong-concept (specificity). Gate logic
G1..G5 is reused verbatim from run_thought_transfer (the validated parent); only the concept bank
(N=462) and the locked (layer, k) change.

THE SEAL: refuses to run unless g0clear_result_<tag>.json shows G0_pass == true.

  python run_g0_stage1.py --src meta-llama/Llama-3.2-3B-Instruct --dst meta-llama/Llama-3.2-1B-Instruct --tag llama3b
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "introspection-gate"))
from styxx_transfer import TransferMap
from introspection_gate import load_model, make_hook, DEVICE
import run_thought_transfer as P          # reuse extract / steer_gain / lock_dose
import run_disjoint_worlds as RDW          # distmat
from run_g0clear import CONCEPTS, split_concepts


def run(src, dst, tag):
    s0_path = HERE / f"g0clear_result_{tag}.json"
    s0 = json.loads(s0_path.read_text(encoding="utf-8"))
    if not s0.get("G0_pass"):
        print(f"SEALED: G0 not cleared (pc_cos={s0.get('G0_pc_cos_FINAL')} < 0.80). Stage 1 does not run.",
              flush=True)
        print("Verdict stands at INSTRUMENT-CEILING (Stage 0).", flush=True)
        return None
    Lstar, kstar = s0["locked"]["layer"], s0["locked"]["k"]

    from transformers import AutoConfig
    from sentence_transformers import SentenceTransformer
    nlA = AutoConfig.from_pretrained(src).num_hidden_layers
    nlB = AutoConfig.from_pretrained(dst).num_hidden_layers
    frac = Lstar / nlA
    LA, LB = Lstar, round(frac * nlB)
    print(f"G0 cleared (pc_cos={s0['G0_pc_cos_FINAL']}). locked layer_A={LA} (frac {frac:.3f} -> layer_B={LB}), k={kstar}",
          flush=True)

    P.CONCEPTS = CONCEPTS                       # parent helpers iterate this global
    tr, sel, fin = split_concepts(seed=0)       # tr = map cloud; fin = held-out transfer set (the G0 FINAL set)
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
    cemb = {c: st.encode([c], normalize_embeddings=True)[0] for c in fin}

    # ---- model A: geometry + held-out steering directions, then free ----
    tokA, mA = load_model(src)
    ptsA, vecsA = P.extract(mA, tokA, LA)
    del mA; torch.cuda.empty_cache()
    RA = np.array([ptsA[c] for c in CONCEPTS])

    # ---- model B: geometry + native held-out directions ----
    tokB, mB = load_model(dst)
    ptsB, vecsB = P.extract(mB, tokB, LB)
    RB = np.array([ptsB[c] for c in CONCEPTS])
    tri = np.triu_indices(len(CONCEPTS), 1)
    rsa = float(np.corrcoef(RDW.distmat(RA)[tri], RDW.distmat(RB)[tri])[0, 1])

    idx_tr = [CONCEPTS.index(c) for c in tr]
    tm = TransferMap.fit(RA[idx_tr], RB[idx_tr], k=kstar)
    rng = np.random.default_rng(0)
    Qrand, _ = np.linalg.qr(rng.standard_normal((tm.Q.shape[0], tm.Q.shape[0])))
    tm_rand = TransferMap(tm.meanA, tm.VAk, Qrand, tm.meanB, tm.VBk, 0.0)

    state = {"vec": None, "alpha": 0.0}
    h = mB.model.layers[LB].register_forward_hook(make_hook(state))
    alpha = P.lock_dose(mB, tokB, state, LB, vecsB, fin[:5], st, cemb)
    print(f"   LOCKED dst dose alpha={alpha}", flush=True)

    rows = []
    for c in fin:
        vT = tm.transfer_direction(vecsA[c])
        vR = tm_rand.transfer_direction(vecsA[c])
        wrong = str(rng.choice([x for x in fin if x != c]))
        vW = tm.transfer_direction(vecsA[wrong])
        g_t = P.steer_gain(mB, tokB, state, LB, vT, c, st, cemb[c], alpha)
        g_n = P.steer_gain(mB, tokB, state, LB, vecsB[c], c, st, cemb[c], alpha)
        g_r = P.steer_gain(mB, tokB, state, LB, vR, c, st, cemb[c], alpha)
        g_w = P.steer_gain(mB, tokB, state, LB, vW, c, st, cemb[c], alpha)
        rows.append({"concept": c, "transfer": g_t, "native": g_n, "randomQ": g_r, "wrong": g_w})
        print(f"  {c:12} transfer={g_t:+.3f} native={g_n:+.3f} randomQ={g_r:+.3f} wrong={g_w:+.3f}", flush=True)
    h.remove(); del mB; torch.cuda.empty_cache()

    def m(key):
        return float(np.mean([r[key] for r in rows]))
    mt, mn, mr, mw = m("transfer"), m("native"), m("randomQ"), m("wrong")
    frac_nte = mt / mn if mn > 1e-6 else float("nan")
    sign_vs_random = int(sum(1 for r in rows if r["transfer"] > r["randomQ"]))
    G1 = mt >= 0.15
    G2 = (mt - mr) >= 0.10 and sign_vs_random >= 0.7 * len(rows)
    G3 = (frac_nte == frac_nte) and frac_nte >= 0.40
    G4 = (mt - mw) >= 0.10
    G5 = (mt - mr) >= 0.10
    if G1 and G2 and G3 and G4:
        verdict = "TRANSFER WORKS — zero-anchor cross-model control transfer"
    elif G1 and G2:
        verdict = "PARTIAL — real but lossy transfer"
    else:
        verdict = "REPORT_AS_LANDED — reading != writing across minds (G0 valid)"

    out = {"experiment": "cross-model WRITE channel (Stage 1, G0-cleared)",
           "prereg": "PREREG_thought_transfer_g0clear_2026_06_20.md",
           "src": src, "dst": dst, "locked_layer_src": LA, "locked_layer_dst": LB, "locked_k": kstar,
           "G0_pc_cos": s0["G0_pc_cos_FINAL"], "rsa": round(rsa, 3), "dst_dose_alpha": alpha,
           "n_heldout": len(fin),
           "mean_transfer_gain": round(mt, 4), "mean_native_gain": round(mn, 4),
           "mean_randomQ_gain": round(mr, 4), "mean_wrong_concept_gain": round(mw, 4),
           "transfer_over_native_NTE": round(frac_nte, 3),
           "transfer_beats_random": f"{sign_vs_random}/{len(rows)}",
           "gates": {"G1_effect": bool(G1), "G2_vs_random": bool(G2), "G3_ceiling": bool(G3),
                     "G4_specificity": bool(G4), "G5_map_not_anything": bool(G5)},
           "rows": rows, "VERDICT": verdict,
           "honest_scope": ("zero-anchor map; HELD-OUT concept directions; MiniLM steering gain; "
                            "Llama-3B->1B near-isometry; shared training data not controlled")}
    fn = HERE / f"thought_transfer_g0clear_result_{tag}.json"
    fn.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps({k2: v for k2, v in out.items() if k2 != "rows"}, indent=2), flush=True)
    print(f"\nwrote {fn.name}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--dst", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--tag", default="llama3b")
    args = ap.parse_args(argv)
    run(args.src, args.dst, args.tag)


if __name__ == "__main__":
    main()
