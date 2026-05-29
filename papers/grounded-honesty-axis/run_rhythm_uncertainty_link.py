"""Is the depth suppression-rhythm the UPSTREAM correlate of the single-pass output-uncertainty
signature? PREREG_rhythm_uncertainty_link_2026_05_29.md.

WITHIN the confab group (where alone a flip_layer is defined), correlate depth-rhythm features
of the late overwrite against the already-validated output-uncertainty signals:
  depth-rhythm  : rel_flip = flip_layer/(n_layers-1)  (relative depth correct token last led)
                  sharpness = lr[-1] - lc[-1]          (final wrong-over-correct logit gap)
  output signal : clean_entropy, instability (1-Stability over N=10 @ T=1.0), logit_margin

Predicted (within-confab Spearman): U1 rho(rel_flip, entropy)>0; U2 rho(rel_flip, instability)>0;
U3 rho(sharpness, entropy)<0. B_link (core): >=1 of U1/U2/U3 with |rho|>=0.40, p<0.05, in the
predicted direction. SURVIVED iff B_link. Run once on Qwen.

Usage:
    python papers/grounded-honesty-axis/run_rhythm_uncertainty_link.py --n 6   # pilot
    python papers/grounded-honesty-axis/run_rhythm_uncertainty_link.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_competence_cliff import SPECS, _eval, _expr, parse_int  # noqa: E402
import run_depth_grounding_whitebox as wb  # noqa: E402
from run_detection_locus import (  # noqa: E402
    N_RESAMPLE, TEMPERATURE, single_pass_signals, resample_ints, stability_of)

RECEIPT = HERE / "rhythm_uncertainty_link_result.json"


@torch.no_grad()
def depth_rhythm(model, tok, prompt_text, realized_text, correct_str):
    """Depth-rhythm of the late overwrite at the first divergent answer position.

    Returns dict or None (unalignable / no divergence). Keys:
      corr_lead  : correct token led realized at some layer < final (bool)
      flip_layer : last layer (index < final) where correct still led, or None
      n_layers   : L (incl. embeddings)
      rel_flip   : flip_layer/(L-1) or None
      sharpness  : lr[-1] - lc[-1] (final wrong-over-correct logit gap)
    """
    plen = tok(prompt_text, return_tensors="pt").input_ids.shape[1]
    fids = tok(prompt_text + realized_text, return_tensors="pt").input_ids.to(wb.DEVICE)
    T = fids.shape[1]
    if T <= plen:
        return None
    real_ids = fids[0, plen:T].tolist()
    corr_ids = tok(correct_str, add_special_tokens=False).input_ids
    div = next((i for i in range(min(len(real_ids), len(corr_ids)))
                if real_ids[i] != corr_ids[i]), None)
    if div is None:
        return None
    pos = plen - 1 + div
    out = model(fids, output_hidden_states=True)
    hs = torch.stack(out.hidden_states, dim=0)[:, 0, :, :]
    vec = model.model.norm(hs)[:, pos, :].float()
    W = model.lm_head.weight.float()
    L = vec.shape[0]
    cid, rid = corr_ids[div], real_ids[div]
    lc = vec @ W[cid]
    lr = vec @ W[rid]
    lead = (lc > lr)
    corr_lead = bool(lead[:-1].any().item())
    ahead_idx = torch.nonzero(lead[:-1], as_tuple=False).flatten()
    flip_layer = int(ahead_idx[-1].item()) if ahead_idx.numel() > 0 else None
    rel_flip = (flip_layer / (L - 1)) if flip_layer is not None else None
    sharpness = float((lr[-1] - lc[-1]).item())
    return {"corr_lead": corr_lead, "flip_layer": flip_layer, "n_layers": int(L),
            "rel_flip": rel_flip, "sharpness": sharpness}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=len(SPECS))
    args = ap.parse_args(argv)

    specs = SPECS[: args.n]
    items = [(f, _eval(f), _expr(f), s) for f, _, s in specs]
    key_blob = json.dumps([(e, c) for _, c, e, _ in items], ensure_ascii=False)
    key_hash = hashlib.sha256(key_blob.encode("utf-8")).hexdigest()
    print(f"answer-key SHA-256 (pre-scoring): {key_hash}")
    print(f"model={wb.MODEL_NAME} device={wb.DEVICE} N_resample={N_RESAMPLE} temp={TEMPERATURE}")

    tok = AutoTokenizer.from_pretrained(wb.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        wb.MODEL_NAME, torch_dtype=torch.float16).to(wb.DEVICE).eval()
    print("model loaded\n")

    SYS = "Answer with only the final number, nothing else."
    rows = []
    for form, correct, expr, subset in items:
        p1, a1 = wb.generate(model, tok, SYS, f"What is {expr}?", max_new_tokens=16)
        v1 = parse_int(a1); ok1 = (v1 == correct)
        member = (not ok1) and (v1 is not None)
        row = {"subset": subset, "expr": expr, "correct": correct, "v1": v1,
               "ok1": ok1, "member": bool(member), "usable": False, "has_rhythm": False}
        if member:
            dr = depth_rhythm(model, tok, p1, a1, str(correct))
            sp = single_pass_signals(model, tok, p1, a1)
            if dr is not None and sp is not None:
                ent, margin = sp
                vals = resample_ints(model, tok, SYS, f"What is {expr}?", N_RESAMPLE)
                stab, nd = stability_of(vals)
                row.update({"usable": True, "clean_entropy": ent, "logit_margin": margin,
                            "instability": 1.0 - stab, "n_distinct": nd,
                            "corr_lead": dr["corr_lead"], "flip_layer": dr["flip_layer"],
                            "n_layers": dr["n_layers"], "rel_flip": dr["rel_flip"],
                            "sharpness": dr["sharpness"]})
                row["has_rhythm"] = bool(dr["corr_lead"] and dr["flip_layer"] is not None)
        rows.append(row)
        if row["usable"]:
            fl = row["flip_layer"]
            print(f"[{subset:9}] {expr:>14}={correct:<8} | rhythm={row['has_rhythm']} "
                  f"flip={fl}/{row['n_layers']} relflip="
                  f"{row['rel_flip'] if row['rel_flip'] is not None else float('nan'):.3f} "
                  f"sharp={row['sharpness']:.2f} | ent={row['clean_entropy']:.3f} "
                  f"inst={row['instability']:.2f} margin={row['logit_margin']:.2f}")
        else:
            print(f"[{subset:9}] {expr:>14}={correct:<8} | "
                  f"{'non-member' if not member else 'unalignable'}")

    rhythm = [r for r in rows if r["usable"] and r["has_rhythm"]]
    n_rhythm = len(rhythm)
    n_no_rhythm = sum(1 for r in rows if r["usable"] and not r["has_rhythm"])
    powered = n_rhythm >= 12

    def rho(xk, yk):
        x = np.array([r[xk] for r in rhythm], float)
        y = np.array([r[yk] for r in rhythm], float)
        if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
            return float("nan"), float("nan")
        r, p = stats.spearmanr(x, y)
        return float(r), float(p)

    pairs = {
        "U1_relflip_entropy": ("rel_flip", "clean_entropy", +1),
        "U2_relflip_instability": ("rel_flip", "instability", +1),
        "U3_sharpness_entropy": ("sharpness", "clean_entropy", -1),
        "relflip_margin": ("rel_flip", "logit_margin", 0),
        "sharpness_instability": ("sharpness", "instability", 0),
        "sharpness_margin": ("sharpness", "logit_margin", 0),
    }
    corr_out = {}
    for name, (xk, yk, pred) in pairs.items():
        r, p = rho(xk, yk)
        corr_out[name] = {"x": xk, "y": yk, "predicted_sign": pred,
                          "rho": round(r, 4) if r == r else None,
                          "p": round(p, 4) if p == p else None}

    def held(name):
        c = corr_out[name]; pred = c["predicted_sign"]
        if c["rho"] is None or c["p"] is None:
            return False
        in_dir = (c["rho"] > 0) if pred > 0 else (c["rho"] < 0)
        return powered and in_dir and (abs(c["rho"]) >= 0.40) and (c["p"] < 0.05)

    u1, u2, u3 = held("U1_relflip_entropy"), held("U2_relflip_instability"), held("U3_sharpness_entropy")
    b_link = u1 or u2 or u3
    result = "SURVIVED" if b_link else "REPORT_AS_LANDED"

    def m(rs, k):
        a = np.array([r[k] for r in rs], float)
        return round(float(a.mean()), 4) if len(a) else None

    receipt = {
        "experiment": "rhythm->uncertainty link — does the depth suppression-rhythm predict the single-pass output-uncertainty signature (within confabs)?",
        "prereg": "papers/grounded-honesty-axis/PREREG_rhythm_uncertainty_link_2026_05_29.md",
        "answer_key_sha256_pre_scoring": key_hash,
        "model": wb.MODEL_NAME, "device": wb.DEVICE,
        "n_resample": N_RESAMPLE, "temperature": TEMPERATURE,
        "n_confab_usable": sum(1 for r in rows if r["usable"]),
        "n_rhythm": n_rhythm, "n_no_rhythm": n_no_rhythm, "powered": powered,
        "means_rhythm_group": {
            "rel_flip": m(rhythm, "rel_flip"), "flip_layer": m(rhythm, "flip_layer"),
            "sharpness": m(rhythm, "sharpness"), "clean_entropy": m(rhythm, "clean_entropy"),
            "instability": m(rhythm, "instability"), "logit_margin": m(rhythm, "logit_margin")},
        "correlations_within_confab": corr_out,
        "U1_relflip_entropy_held": bool(u1), "U2_relflip_instability_held": bool(u2),
        "U3_sharpness_entropy_held": bool(u3),
        "B_link": bool(b_link), "RESULT": result,
        "rows": rows,
        "honest_scope": (
            "single open model Qwen2.5-1.5B-Instruct; arithmetic only; one confirmatory run; "
            "feasibility-grade; WITHIN-confab only (flip_layer undefined for correct answers, so "
            "NOT a confab-vs-correct claim and does not re-test detection); rhythm subset = "
            "confabs with corr_lead True AND flip_layer defined; depth features from clean "
            "full-vocab logit-lens at the first divergent answer token; output entropy/margin from "
            "the same clean logit-lens at the first answer token; instability from N=10 resamples "
            "@ T=1.0, exact distinct-integer Stability (no judge); ground truth in-code then hashed "
            "pre-scoring. Correlation not cause: a positive rho shows depth-rhythm and output "
            "signature co-vary across confabs, not that one drives the other. Does NOT touch the "
            "correctness bound: both are confidence readouts; neither corrects the answer."),
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in receipt.items() if k != "rows"}, indent=2))
    print(f"\nn_rhythm={n_rhythm} n_no_rhythm={n_no_rhythm} powered={powered}")
    for nm in ("U1_relflip_entropy", "U2_relflip_instability", "U3_sharpness_entropy"):
        c = corr_out[nm]
        print(f"{nm}: rho={c['rho']} p={c['p']} (pred sign {c['predicted_sign']:+d})")
    print(f"B_link={b_link} -> {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
