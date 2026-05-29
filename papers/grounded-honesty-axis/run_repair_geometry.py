"""Does the one-shot OVERWRITE GEOMETRY predict method-diverse REPAIRABILITY?
PREREG_repair_geometry_2026_05_29.md.

The keystone that joins the arc's two validated halves ON THE SAME MODEL:
  - black-box repair (P1): method-diverse re-derivation recovers most confabulations;
  - white-box mechanism (suppression-rhythm): confabulation is a tight late install.
Question: is repairability written in the mechanism? For each one-shot confab on Qwen,
compute (a) repairability r in {0..5} = how many of the 5 independent reasoning methods
land on truth (Qwen's OWN repair, same model as the geometry), and (b) the one-shot
overwrite geometry (flip_layer, realized_dominance, install_jump). Correlate.

Ground truth computed in-code (arithmetic) and hashed before scoring; correctness is exact
integer match (no judge). Greedy/deterministic -> reproduces the white-box answer key.

Usage:
    python papers/grounded-honesty-axis/run_repair_geometry.py --n 6   # pilot
    python papers/grounded-honesty-axis/run_repair_geometry.py
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
from run_depth_grounding_whitebox import MODEL_NAME, DEVICE, generate  # noqa: E402
from run_path_diverse_grounding import METHODS, parse_answer_line  # noqa: E402

RECEIPT = HERE / "repair_geometry_result.json"


@torch.no_grad()
def overwrite_geometry(model, tok, prompt_text, realized_text, correct_str):
    """At the first divergent answer position, return one-shot overwrite geometry.

    Returns dict or None (unalignable). Keys:
      flip_layer         : last pre-final layer where correct still outranks realized (int)
      realized_dominance : fraction of pre-final layers where realized outranks correct
      install_jump       : max single-layer rise of (realized - correct) lens-logit
      crossed            : correct led at some pre-final layer AND realized wins at final
      n_layers           : L
    """
    plen = tok(prompt_text, return_tensors="pt").input_ids.shape[1]
    fids = tok(prompt_text + realized_text, return_tensors="pt").input_ids.to(DEVICE)
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
    vec = model.model.norm(hs)[:, pos, :].float()          # (L, d)
    W = model.lm_head.weight.float()
    L = vec.shape[0]

    lc = vec @ W[corr_ids[div]]                            # (L,) correct
    lr = vec @ W[real_ids[div]]                            # (L,) realized (wrong)
    pre = slice(0, L - 1)
    corr_ahead = (lc[pre] > lr[pre])                       # correct leads, pre-final
    real_ahead = (lr[pre] > lc[pre])
    ahead_idx = torch.nonzero(corr_ahead, as_tuple=False).flatten()
    flip_layer = int(ahead_idx[-1].item()) if ahead_idx.numel() > 0 else 0
    realized_dominance = float(real_ahead.float().mean().item())
    gap = (lr - lc)                                        # >0 where wrong leads
    install_jump = float(torch.diff(gap).max().item())
    crossed = bool(corr_ahead.any().item()) and bool((lr[-1] >= lc[-1]).item())
    return {"flip_layer": flip_layer, "realized_dominance": realized_dominance,
            "install_jump": install_jump, "crossed": crossed, "n_layers": L}


def repairability(model, tok, expr, correct):
    """Qwen's own 5-method recovery count r in {0..5} (deterministic greedy per method)."""
    r = 0
    per = []
    for m in METHODS:
        _, a = generate(model, tok, m, f"What is {expr}?", max_new_tokens=512)
        v = parse_answer_line(a)
        hit = (v == correct)
        per.append(hit)
        r += int(hit)
    return r, per


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=len(SPECS))
    args = ap.parse_args(argv)

    items = []
    for i, (form, delta, subset) in enumerate(SPECS[: args.n]):
        items.append((form, _eval(form), _expr(form), subset, i))

    key_blob = json.dumps([(e, c) for _, c, e, _, _ in items], ensure_ascii=False)
    key_hash = hashlib.sha256(key_blob.encode("utf-8")).hexdigest()
    print(f"answer-key SHA-256 (pre-scoring): {key_hash}")
    print(f"model={MODEL_NAME} device={DEVICE} items={len(items)} methods={len(METHODS)}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16).to(DEVICE).eval()
    print("model loaded\n")

    rows = []
    for form, correct, expr, subset, idx in items:
        p1, a1 = generate(model, tok, "Answer with only the final number, nothing else.",
                          f"What is {expr}?", max_new_tokens=16)
        v1 = parse_int(a1); ok1 = (v1 == correct)
        geom = r = per = None
        if not ok1 and v1 is not None:
            geom = overwrite_geometry(model, tok, p1, a1, str(correct))
            if geom is not None:
                r, per = repairability(model, tok, expr, correct)
        rows.append({"subset": subset, "expr": expr, "correct": correct,
                     "v1": v1, "ok1": ok1, "geom": geom, "r": r, "per_method": per})
        fl = geom["flip_layer"] if geom else None
        rd = round(geom["realized_dominance"], 2) if geom else None
        print(f"[{idx:2d}|{subset:9}] {expr:>14}={correct:<9} | "
              f"1shot {str(v1):>9} {'OK ' if ok1 else 'BAD'} | "
              f"r={r} flip={fl} realdom={rd}")

    # ---- scoring ----
    use = [row for row in rows if row["geom"] is not None and row["r"] is not None]
    n_use = len(use)
    rv = np.array([row["r"] for row in use], float)
    fl = np.array([row["geom"]["flip_layer"] for row in use], float)
    rd = np.array([row["geom"]["realized_dominance"] for row in use], float)
    ij = np.array([row["geom"]["install_jump"] for row in use], float)

    powered = n_use >= 12

    def spear(x, y):
        if len(x) < 4 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
            return float("nan"), float("nan")
        rho, p = stats.spearmanr(x, y)
        return float(rho), float(p)

    u1_rho, u1_p = spear(rv, fl)            # later install -> more repairable (predict +)
    u2_rho, u2_p = spear(rv, rd)            # entrenched wrong -> less repairable (predict -)
    u1 = powered and (u1_rho == u1_rho) and (u1_rho >= 0.40) and (u1_p < 0.05)
    u2 = powered and (u2_rho == u2_rho) and (u2_rho <= -0.40) and (u2_p < 0.05)

    # U3: best geometry feature separates repairable (r>=3) from stubborn (r<=1).
    hi = [i for i, v in enumerate(rv) if v >= 3]
    lo = [i for i, v in enumerate(rv) if v <= 1]
    u3_powered = (len(hi) >= 6 and len(lo) >= 6)
    feats = {"flip_layer": fl, "realized_dominance": rd, "install_jump": ij}
    aucs = {name: auc([arr[i] for i in hi], [arr[i] for i in lo]) for name, arr in feats.items()}
    best_name, best_auc = (None, float("nan"))
    if u3_powered:
        valid = {k: v for k, v in aucs.items() if v == v}
        if valid:
            best_name = max(valid, key=lambda k: abs(valid[k] - 0.5))
            best_auc = valid[best_name]
    u3 = u3_powered and (best_auc == best_auc) and (best_auc >= 0.70 or best_auc <= 0.30)

    r_hist = {int(k): int((rv == k).sum()) for k in range(6)}

    receipt = {
        "experiment": "does one-shot overwrite geometry predict method-diverse repairability?",
        "prereg": "papers/grounded-honesty-axis/PREREG_repair_geometry_2026_05_29.md",
        "answer_key_sha256_pre_scoring": key_hash,
        "model": MODEL_NAME, "device": DEVICE, "n_items": len(rows),
        "core_signal": "exact integer parse vs in-code arithmetic truth (no judge)",
        "n_usable_confab": n_use, "powered": powered,
        "repairability_histogram_r0_to_r5": r_hist,
        "mean_repairability": round(float(rv.mean()), 4) if n_use else None,
        "U1_later_install_more_repairable": {
            "spearman_rho": round(u1_rho, 4) if u1_rho == u1_rho else None,
            "p": round(u1_p, 5) if u1_p == u1_p else None, "predict_sign": "+"},
        "U2_entrenched_less_repairable": {
            "spearman_rho": round(u2_rho, 4) if u2_rho == u2_rho else None,
            "p": round(u2_p, 5) if u2_p == u2_p else None, "predict_sign": "-"},
        "U3_geometry_predicts_repair": {
            "n_repairable_hi": len(hi), "n_stubborn_lo": len(lo),
            "powered": u3_powered, "aucs": {k: (round(v, 4) if v == v else None) for k, v in aucs.items()},
            "best_feature": best_name, "best_auc": round(best_auc, 4) if best_auc == best_auc else None},
        "rows": rows,
        "U1": bool(u1), "U2": bool(u2), "U3": bool(u3), "U3_powered": u3_powered,
        "RESULT": ("SURVIVED" if (u1 and u3) else "REPORT_AS_LANDED"),
        "honest_scope": (
            "single open model Qwen2.5-1.5B-Instruct, SAE-free full-vocab logit-lens, "
            "feasibility-grade n=36, one confirmatory run; arithmetic ground truth computed "
            "in-code then hashed pre-scoring; exact-integer correctness (no judge). "
            "Repairability is Qwen's OWN 5-method recovery count (same model as the geometry, "
            "removing the cross-model confound in the prior gpt-4o-mini P1 labels); the "
            "repairable arm may be thin on a 1.5B model. Correlational within-model claim; "
            "NOT a causal demonstration that altering geometry changes repairability. A null "
            "refutes only that the two halves are legibly linked in one forward pass; it "
            "leaves both the repair result (P1) and the mechanism (suppression-rhythm) intact."),
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in receipt.items() if k != "rows"}, indent=2))
    print(f"\nn_use={n_use} powered={powered} r_hist={r_hist}")
    print(f"U1={u1}(rho={u1_rho:.3f} p={u1_p:.4f}) U2={u2}(rho={u2_rho:.3f}) "
          f"U3={u3}(best {best_name}={best_auc:.3f}) -> {receipt['RESULT']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
