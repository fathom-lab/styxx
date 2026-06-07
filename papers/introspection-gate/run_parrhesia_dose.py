"""Diagnostic (exploratory, NOT the frozen gate): map the apparatus bind for PARRHESIA RUNG 1.

The pinned-dose preconditions failed: at alpha=10 the held-out-direction read saturates (clean==at-pos
==1.00 -> P5 divergence can't separate shadow from saturated-real) AND the half-template injection
steers at 0.108 < 0.15 (P2). Question: is there ANY dose where P2 (steering>=0.15) AND P5 (at-pos -
clean-pos >= 0.10, clean-pos > chance) co-hold? If yes, the cells are climbable at that dose; if no,
the bind is fundamental and RUNG 1 is UNINFORMATIVE on this apparatus.

  python run_parrhesia_dose.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from introspection_gate import load_model, CONCEPTS, CONCEPT_TEMPLATES, DEVICE, steering_gain, make_hook
from introspection_fc import make_hook_skip
from run_parrhesia import cvec, resid_at, fit_probe, acc_of, CARRIERS, PRIMARY

CARR = CARRIERS[:10]
ALPHAS = [1, 2, 4, 6, 10, 16]


def main():
    torch.manual_seed(11)
    tok, model = load_model(PRIMARY)
    nl = model.config.num_hidden_layers
    inj, rd = round(0.60 * nl), round(0.85 * nl)
    TPL_A, TPL_B = CONCEPT_TEMPLATES[:6], CONCEPT_TEMPLATES[6:]
    vA = {c: cvec(model, tok, inj, c, TPL_A) for c in CONCEPTS}
    vB = {c: cvec(model, tok, inj, c, TPL_B) for c in CONCEPTS}
    vF = {c: cvec(model, tok, inj, c, CONCEPT_TEMPLATES) for c in CONCEPTS}  # full-template (council-matched)
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
    cemb = st.encode(CONCEPTS, normalize_embeddings=True)

    rows = []
    for a in ALPHAS:
        sgs = {"vec": None, "alpha": 0.0}
        h = model.model.layers[inj].register_forward_hook(make_hook(sgs))
        gF, _ = steering_gain(model, tok, sgs, vF, a, st, cemb)
        gA, _ = steering_gain(model, tok, sgs, vA, a, st, cemb)
        h.remove()
        state = {"vec": None, "alpha": 0.0, "skip_last": False}
        h = model.model.layers[inj].register_forward_hook(make_hook_skip(state))
        Xtr, Xc, Xp, y = [], [], [], []
        for ci, c in enumerate(CONCEPTS):
            for carr in CARR:
                Xtr.append(resid_at(model, tok, state, rd, carr, vA[c], a, True))
                Xc.append(resid_at(model, tok, state, rd, carr, vB[c], a, True))
                Xp.append(resid_at(model, tok, state, rd, carr, vB[c], a, False))
                y.append(ci)
        h.remove()
        y = np.array(y); sc, clf = fit_probe(np.array(Xtr), y)
        acc_c = acc_of(sc, clf, np.array(Xc), y)
        acc_p = acc_of(sc, clf, np.array(Xp), y)
        div = acc_p - acc_c
        p2 = gF >= 0.15
        p5 = (div >= 0.10) and (acc_c > 0.125)
        rows.append(dict(alpha=a, steer_full=round(gF, 3), steer_halfA=round(gA, 3),
                         acc_clean=round(acc_c, 3), acc_atpos=round(acc_p, 3), divergence=round(div, 3),
                         P2_steer_live=p2, P5_divergence=p5, BOTH=p2 and p5))
        print(f"alpha={a:>2} steerF={gF:+.3f} steerA={gA:+.3f} clean={acc_c:.3f} atpos={acc_p:.3f} "
              f"div={div:+.3f}  P2={'Y' if p2 else 'n'} P5={'Y' if p5 else 'n'} BOTH={'YES' if p2 and p5 else '-'}", flush=True)

    valid = [r for r in rows if r["BOTH"]]
    out = {"experiment": "PARRHESIA apparatus-bind dose sweep (diagnostic, exploratory)",
           "model": PRIMARY, "inject_layer": inj, "read_layer": rd, "rows": rows,
           "valid_doses": [r["alpha"] for r in valid],
           "verdict": ("CLIMBABLE at alpha(s) " + str([r["alpha"] for r in valid]) if valid
                       else "FUNDAMENTAL BIND — no dose co-satisfies steering-live AND non-ceiling-divergence; "
                            "RUNG 1 UNINFORMATIVE on this apparatus (read saturates before/around steering-live dose)")}
    (HERE / "parrhesia_dose_sweep.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\nVERDICT:", out["verdict"])
    del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
