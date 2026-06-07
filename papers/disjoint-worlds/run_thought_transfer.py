# -*- coding: utf-8 -*-
"""run_thought_transfer.py — UNIVERSAL CROSS-MODEL CONTROL TRANSFER (the proof).

Compute a concept STEERING direction in model A, install it in model B through the unsupervised
zero-anchor map (styxx_transfer.TransferMap), and measure whether it actually STEERS B toward the
concept — vs B's native direction (ceiling), a random-Q map, and a wrong-concept transfer.

Anti-circularity:
  - the map Q is fit on TRAIN concepts only; we transfer HELD-OUT concept directions (no leakage).
  - REAL-GEOMETRY transfer positive control runs FIRST (A -> A-rotated must reproduce held-out
    directions at high cosine); if it fails the instrument is broken and we stop.
  - random-Q control (does any orthogonal map steer?), wrong-concept control (specificity),
    native-B ceiling, and the steering metric is MiniLM concept-sim gain over a clean baseline.

  python run_thought_transfer.py --src meta-llama/Llama-3.2-3B-Instruct --dst meta-llama/Llama-3.2-1B-Instruct --tag llama3b_to_1b
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
GH = HERE.parent / "introspection-gate"
sys.path.insert(0, str(GH))
from styxx_transfer import TransferMap, self_test, pca_basis
from introspection_gate import (load_model, make_hook, build_prompt, CONCEPT_TEMPLATES,
                                is_coherent, DEVICE)

CONCEPTS = ("dog cat horse cow pig sheep lion tiger bear wolf fox deer rabbit mouse elephant "
            "apple banana orange grape lemon peach cherry potato carrot onion bread cheese rice egg "
            "hammer wrench saw drill knife fork spoon plate bowl cup bottle "
            "car truck bus train plane boat ship bicycle helicopter rocket "
            "chair table bed sofa desk shelf lamp mirror clock door window roof wall "
            "shirt pants shoe hat coat dress sock glove scarf belt ring watch "
            "tree flower grass leaf river mountain ocean lake forest desert cloud rain snow wind "
            "sun moon star fire ice stone sand gold silver iron "
            "house school church store bank hospital bridge tower castle factory library "
            "guitar piano drum violin trumpet flute anger fear joy love hope").split()
_seen = set(); CONCEPTS = [c for c in CONCEPTS if not (c in _seen or _seen.add(c))]
HELDOUT = CONCEPTS[::6]                 # ~every 6th concept transferred (map never trained on these)
TRAIN = [c for c in CONCEPTS if c not in set(HELDOUT)]
STEER_CARRIERS = ["Continue this thought in one sentence: Today I keep thinking about",
                  "Right now my mind keeps drifting to",
                  "I cannot stop picturing"]


@torch.no_grad()
def extract(model, tok, layer):
    """Per-concept point (mean rep at layer L) and paired steering direction (concept vs 'object')."""
    pts, vecs = {}, {}
    for c in CONCEPTS:
        cps, diffs = [], []
        for t in CONCEPT_TEMPLATES:
            lc = model(**tok(t.format(c=c), return_tensors="pt").to(DEVICE),
                       output_hidden_states=True).hidden_states[layer + 1][0, -1].float().cpu().numpy()
            lo = model(**tok(t.format(c="object"), return_tensors="pt").to(DEVICE),
                       output_hidden_states=True).hidden_states[layer + 1][0, -1].float().cpu().numpy()
            cps.append(lc); diffs.append(lc - lo)
        pts[c] = np.mean(cps, 0)
        v = np.mean(diffs, 0); vecs[c] = v / (np.linalg.norm(v) + 1e-9)
    return pts, vecs


@torch.no_grad()
def steer_gain(model, tok, state, layer, vec_np, concept, st, cemb_c, alpha):
    """Inject unit direction at layer L across all positions; MiniLM concept-sim gain over clean."""
    vec = torch.tensor(vec_np / (np.linalg.norm(vec_np) + 1e-9), dtype=model.dtype, device=DEVICE)
    gains = []
    for carrier in STEER_CARRIERS:
        ids = tok(build_prompt(tok, carrier), return_tensors="pt").to(DEVICE)
        plen = ids.input_ids.shape[1]
        state["vec"], state["alpha"] = None, 0.0
        clean = tok.decode(model.generate(**ids, max_new_tokens=28, do_sample=False,
                           pad_token_id=tok.eos_token_id)[0, plen:], skip_special_tokens=True)
        state["vec"], state["alpha"] = vec, float(alpha)
        out = tok.decode(model.generate(**ids, max_new_tokens=28, do_sample=False,
                         pad_token_id=tok.eos_token_id)[0, plen:], skip_special_tokens=True)
        state["vec"], state["alpha"] = None, 0.0
        ce = st.encode([clean], normalize_embeddings=True)[0]
        oe = st.encode([out], normalize_embeddings=True)[0]
        gains.append(float(cemb_c @ oe) - float(cemb_c @ ce))
    return float(np.mean(gains))


def lock_dose(model, tok, state, layer, vecsB, concepts, st, cemb):
    """Lock the dst injection dose: smallest alpha with mean NATIVE gain >= 0.12 (else max-gain)."""
    best = (None, -1.0)
    for a in [8.0, 12.0, 16.0, 20.0]:
        g = float(np.mean([steer_gain(model, tok, state, layer, vecsB[c], c, st, cemb[c], a)
                           for c in concepts]))
        print(f"   dose alpha={a}: mean native gain={g:+.3f}", flush=True)
        if g > best[1]:
            best = (a, g)
        if g >= 0.12:
            return a
    return best[0]


def run(src, dst, tag, k=60):
    from transformers import AutoConfig
    from sentence_transformers import SentenceTransformer
    nlA = AutoConfig.from_pretrained(src).num_hidden_layers
    nlB = AutoConfig.from_pretrained(dst).num_hidden_layers
    LA, LB = round(0.60 * nlA), round(0.60 * nlB)
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
    cemb = {c: st.encode([c], normalize_embeddings=True)[0] for c in HELDOUT}

    # ---- extract A geometry + held-out steering directions, then free A ----
    tokA, mA = load_model(src)
    ptsA, vecsA = extract(mA, tokA, LA)
    del mA; torch.cuda.empty_cache()
    RA = np.array([ptsA[c] for c in CONCEPTS])

    # ---- REAL-GEOMETRY transfer positive control (A -> A-rotated), testing the ACTUAL concept dirs ----
    pc_dirs = np.array([vecsA[c] for c in HELDOUT])
    pc_cos, _ = self_test(np.array([ptsA[c] for c in TRAIN]), k=k, test_dirs=pc_dirs)
    print(f"REAL-geometry transfer positive control (A->A-rot, CONCEPT dirs) mean|cos|={pc_cos:.3f} (need >=0.80)", flush=True)

    # ---- load B, extract geometry + native held-out directions ----
    tokB, mB = load_model(dst)
    ptsB, vecsB = extract(mB, tokB, LB)
    RB = np.array([ptsB[c] for c in CONCEPTS])
    rsa = float(np.corrcoef(
        __import__("run_disjoint_worlds").distmat(RA)[np.triu_indices(len(CONCEPTS), 1)],
        __import__("run_disjoint_worlds").distmat(RB)[np.triu_indices(len(CONCEPTS), 1)])[0, 1])

    idx_train = [CONCEPTS.index(c) for c in TRAIN]
    tm = TransferMap.fit(RA[idx_train], RB[idx_train], k=k)
    rng = np.random.default_rng(0)
    Qrand, _ = np.linalg.qr(rng.standard_normal((tm.Q.shape[0], tm.Q.shape[0])))
    tm_rand = TransferMap(tm.meanA, tm.VAk, Qrand, tm.meanB, tm.VBk, 0.0)

    state = {"vec": None, "alpha": 0.0}
    h = mB.model.layers[LB].register_forward_hook(make_hook(state))
    # lock the dst injection dose so NATIVE steering is real (not floored) before comparing transfer
    alpha = lock_dose(mB, tokB, state, LB, vecsB, HELDOUT[:5], st, cemb)
    print(f"   LOCKED dst dose alpha={alpha}", flush=True)
    rows = []
    for c in HELDOUT:
        vT = tm.transfer_direction(vecsA[c])             # zero-anchor transferred
        vR = tm_rand.transfer_direction(vecsA[c])        # random-Q control
        wrong = rng.choice([x for x in HELDOUT if x != c])
        vW = tm.transfer_direction(vecsA[str(wrong)])    # wrong concept, measured vs c (specificity)
        g_t = steer_gain(mB, tokB, state, LB, vT, c, st, cemb[c], alpha)
        g_n = steer_gain(mB, tokB, state, LB, vecsB[c], c, st, cemb[c], alpha)   # native ceiling
        g_r = steer_gain(mB, tokB, state, LB, vR, c, st, cemb[c], alpha)
        g_w = steer_gain(mB, tokB, state, LB, vW, c, st, cemb[c], alpha)
        rows.append({"concept": c, "transfer": g_t, "native": g_n, "randomQ": g_r, "wrong": g_w})
        print(f"  {c:9} transfer={g_t:+.3f} native={g_n:+.3f} randomQ={g_r:+.3f} wrong={g_w:+.3f}", flush=True)
    h.remove(); del mB; torch.cuda.empty_cache()

    def m(key):
        return float(np.mean([r[key] for r in rows]))
    mt, mn, mr, mw = m("transfer"), m("native"), m("randomQ"), m("wrong")
    frac = mt / mn if mn > 1e-6 else float("nan")
    sign_vs_random = int(sum(1 for r in rows if r["transfer"] > r["randomQ"]))
    sign_vs_wrong = int(sum(1 for r in rows if r["transfer"] > r["wrong"]))
    G0 = pc_cos >= 0.80
    G1 = mt >= 0.15
    G2 = (mt - mr) >= 0.10 and sign_vs_random >= 0.7 * len(rows)
    G3 = (frac == frac) and frac >= 0.40
    G4 = (mt - mw) >= 0.10
    if not G0:
        verdict = "ALIGNER_LIMITED (positive control < 0.80; null not interpretable)"
    elif G1 and G2 and G3 and G4:
        verdict = "TRANSFER WORKS"
    elif G1 and G2:
        verdict = "PARTIAL (real but lossy transfer)"
    else:
        verdict = "REPORT_AS_LANDED — no zero-anchor transfer"
    out = {"experiment": "universal cross-model thought transfer", "src": src, "dst": dst,
           "inject_layer_src": LA, "inject_layer_dst": LB, "rsa": round(rsa, 3), "dst_dose_alpha": alpha,
           "transfer_positive_control_cos": round(pc_cos, 3),
           "n_heldout": len(HELDOUT), "chance_note": "steering gain, clean-subtracted; random-Q is the null",
           "mean_transfer_gain": round(mt, 4), "mean_native_gain": round(mn, 4),
           "mean_randomQ_gain": round(mr, 4), "mean_wrong_concept_gain": round(mw, 4),
           "transfer_over_native_frac": round(frac, 3),
           "transfer_beats_random_count": f"{sign_vs_random}/{len(rows)}",
           "transfer_beats_wrong_count": f"{sign_vs_wrong}/{len(rows)}",
           "gates": {"G0_posctrl": bool(G0), "G1_effect": bool(G1), "G2_vs_random": bool(G2),
                     "G3_ceiling_frac": bool(G3), "G4_specificity": bool(G4)},
           "rows": rows,
           "VERDICT": verdict,
           "honest_scope": ("zero-anchor map from concept clouds; transferred HELD-OUT concept "
                            "directions; steering measured by MiniLM gain in dst; near-isometry "
                            "regime expected to dominate. Shared training data not controlled.")}
    (HERE / f"thought_transfer_result_{tag}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps({k2: v for k2, v in out.items() if k2 != "rows"}, indent=2), flush=True)
    print(f"wrote thought_transfer_result_{tag}.json", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--dst", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--tag", default="llama3b_to_1b")
    args = ap.parse_args(argv)
    run(args.src, args.dst, args.tag)


if __name__ == "__main__":
    main()
