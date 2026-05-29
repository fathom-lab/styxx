"""Single-layer causal localization sweep: does the install peak sit where the geometry said?
PREREG_single_layer_localization_2026_05_29.md.

For each one-shot confab (clean teacher-forced baseline, alignable divergence), knock out EACH
decoder layer's residual write at the divergence position (gamma=0, single layer) and read the
next-token argmax. Per-layer removal rate r(i). Reference layers: install center c (from this
model's OWN suppression-rhythm median flip) and early control center cc (proportional). Tests
L1 (peak late), L2 (single-layer localization c vs cc), L3 (causal peak == geometric flip).

Runs once per model. Ground truth computed in-code (arithmetic) and hashed before scoring.

Usage:
    python papers/grounded-honesty-axis/run_single_layer_localization.py --model qwen --n 6
    python papers/grounded-honesty-axis/run_single_layer_localization.py --model qwen
    python papers/grounded-honesty-axis/run_single_layer_localization.py --model llama
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
from run_suppression_rhythm import single_digit_token_ids, suppression_profile  # noqa: E402
from run_disinhibition import (divergence, logits_at, entropy_of,  # noqa: E402
                               is_numeric_token)

MODELS = {"qwen": "Qwen/Qwen2.5-1.5B-Instruct",
          "llama": "meta-llama/Llama-3.2-1B-Instruct"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="qwen", choices=list(MODELS))
    ap.add_argument("--n", type=int, default=len(SPECS))
    args = ap.parse_args(argv)
    model_name = MODELS[args.model]
    receipt_path = HERE / f"single_layer_localization_{args.model}_result.json"

    items = []
    for i, (form, delta, subset) in enumerate(SPECS[: args.n]):
        items.append((form, _eval(form), _expr(form), subset, i))
    key_blob = json.dumps([(e, c) for _, c, e, _, _ in items], ensure_ascii=False)
    key_hash = hashlib.sha256(key_blob.encode("utf-8")).hexdigest()
    print(f"answer-key SHA-256 (pre-scoring): {key_hash}")
    print(f"model={model_name} device={wb.DEVICE} items={len(items)}")

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16).to(wb.DEVICE).eval()
    N = len(model.model.layers)
    digit_ids = single_digit_token_ids(tok)
    print(f"model loaded; n_decoder={N}\n")

    # ---- pass 1: one-shot answers + Stage-A flip geometry (to get c) ----
    rows = []
    for form, correct, expr, subset, idx in items:
        p1, a1 = wb.generate(model, tok, "Answer with only the final number, nothing else.",
                             f"What is {expr}?", max_new_tokens=16)
        v1 = parse_int(a1); ok1 = (v1 == correct)
        prof = None
        if (not ok1) and v1 is not None:
            prof = suppression_profile(model, tok, p1, a1, str(correct), digit_ids)
        rows.append({"subset": subset, "expr": expr, "correct": correct, "v1": v1,
                     "ok1": ok1, "p1": p1, "a1": a1, "prof": prof})

    profs = [r["prof"] for r in rows if r["prof"] is not None]
    flips = [p["flip_layer"] for p in profs if p["crossed"] and p["flip_layer"] is not None]
    flip_median = float(np.median(flips)) if flips else float("nan")
    c = int(round(flip_median)) - 1 if flip_median == flip_median else None      # install center
    cc = int(round(8 * N / 28))                                                  # early control
    Lt = int(np.ceil(2 * (N - 1) / 3))                                          # decoder late thresh
    c = None if c is None else max(0, min(N - 1, c))
    cc = max(0, min(N - 1, cc))
    print(f"flip_median(hidden)={flip_median} -> install center c={c}; control cc={cc}; "
          f"late_thresh Lt={Lt}\n")

    # ---- pass 2: per-layer single-layer knockdown sweep at the divergence position ----
    use = []
    for r in rows:
        if r["ok1"] or r["v1"] is None:
            continue
        dv = divergence(tok, r["p1"], r["a1"], str(r["correct"]))
        if dv is None:
            continue
        pos, r_id, c_id = dv
        base = logits_at(model, tok, r["p1"], r["a1"], pos, None, 1.0)
        if int(base.argmax().item()) != r_id:
            continue
        removed = []
        recover = []
        for i in range(N):
            lg = logits_at(model, tok, r["p1"], r["a1"], pos, (i, i), 0.0)
            am = int(lg.argmax().item())
            removed.append(int(am != r_id))
            recover.append(int(am == c_id))
        coh_c = is_numeric_token(tok, int(
            logits_at(model, tok, r["p1"], r["a1"], pos, (c, c), 0.0).argmax().item())) if c is not None else 0
        use.append({"subset": r["subset"], "expr": r["expr"], "pos": pos,
                    "removed_by_layer": removed, "recover_by_layer": recover,
                    "coh_c": int(coh_c)})
        print(f"[{r['subset']:9}] {r['expr']:>14} | peak_layer="
              f"{int(np.argmax(removed))} rm@c={removed[c] if c is not None else None} "
              f"rm@cc={removed[cc]}")

    n_use = len(use)
    powered = n_use >= 12

    # ---- per-layer removal curve ----
    if use:
        R = np.array([u["removed_by_layer"] for u in use], float)   # (n_use, N)
        r_of_layer = R.mean(axis=0)                                  # (N,)
        peak_layer = int(np.argmax(r_of_layer))
        r_c = float(r_of_layer[c]) if c is not None else float("nan")
        r_cc = float(r_of_layer[cc])
        coh_c_rate = float(np.mean([u["coh_c"] for u in use]))
    else:
        r_of_layer = np.array([]); peak_layer = -1
        r_c = r_cc = coh_c_rate = float("nan")

    # L1 peak late
    l1 = (peak_layer >= Lt)
    # L2 single-layer localization c vs cc (discordant sign test)
    if use and c is not None:
        t_only = sum(1 for u in use if u["removed_by_layer"][c] and not u["removed_by_layer"][cc])
        k_only = sum(1 for u in use if u["removed_by_layer"][cc] and not u["removed_by_layer"][c])
        nd = t_only + k_only
        sign_p = (float(stats.binomtest(t_only, nd, 0.5, alternative="greater").pvalue)
                  if nd > 0 else float("nan"))
        l2 = (powered and (r_c - r_cc >= 0.30) and (r_c >= 0.50)
              and (sign_p == sign_p) and (sign_p < 0.05))
    else:
        t_only = k_only = 0; sign_p = float("nan"); l2 = False
    # L3 causal peak == geometric flip (within +-2)
    l3 = (c is not None) and (peak_layer >= 0) and (abs(peak_layer - c) <= 2)

    result = "SURVIVED" if (l2 and l3) else "REPORT_AS_LANDED"

    slim = [{k: v for k, v in u.items()} for u in use]
    receipt = {
        "experiment": "single-layer causal localization sweep — does the install peak sit at the geometric flip layer?",
        "prereg": "papers/grounded-honesty-axis/PREREG_single_layer_localization_2026_05_29.md",
        "answer_key_sha256_pre_scoring": key_hash,
        "model": model_name, "device": wb.DEVICE, "n_decoder_layers": N, "n_items": len(rows),
        "core_signal": "single-position teacher-forced single-layer gamma=0 next-token argmax (full vocab) vs in-code truth (no judge)",
        "n_oneshot_confab": sum(1 for r in rows if not r["ok1"]),
        "geometry": {"median_flip_hidden": flip_median if flip_median == flip_median else None,
                     "install_center_c": c, "early_control_cc": cc, "late_threshold_Lt": Lt},
        "n_usable_confab": n_use, "powered": powered,
        "removal_rate_by_layer": [round(float(x), 4) for x in r_of_layer.tolist()],
        "peak_layer": peak_layer,
        "coherence_rate_at_c": round(coh_c_rate, 4) if coh_c_rate == coh_c_rate else None,
        "L1_peak_late": {"peak_layer": peak_layer, "late_threshold": Lt, "held": bool(l1)},
        "L2_single_layer_localization": {
            "r_at_c": round(r_c, 4) if r_c == r_c else None,
            "r_at_cc": round(r_cc, 4) if r_cc == r_cc else None,
            "delta": round(r_c - r_cc, 4) if (r_c == r_c and r_cc == r_cc) else None,
            "discordant_c_only": t_only, "discordant_cc_only": k_only,
            "sign_test_p": round(sign_p, 5) if sign_p == sign_p else None, "held": bool(l2)},
        "L3_causal_peak_eq_geometric_flip": {
            "peak_layer": peak_layer, "install_center_c": c,
            "abs_diff": (abs(peak_layer - c) if c is not None and peak_layer >= 0 else None),
            "held": bool(l3)},
        "rows": slim,
        "L1": bool(l1), "L2": bool(l2), "L3": bool(l3),
        "RESULT": result,
        "honest_scope": (
            f"{model_name}; SAE-free full-vocab logit-lens; single-position teacher-forced "
            "single-layer gamma=0 knockdown read at the divergence position only (not multi-token "
            "regeneration); one confirmatory run; feasibility-grade n=36; arithmetic ground truth "
            "computed in-code then hashed pre-scoring; exact-integer correctness (no judge); "
            "greedy/deterministic. Install center c is from THIS model's own suppression-rhythm "
            "median flip; control cc and late threshold are fixed by the shared proportional rule "
            "— not tuned to the verdict. Tests causal localization of the install and its agreement "
            "with the descriptive geometry; does NOT touch the correctness bound (single-layer "
            "knockdown yields uncertainty, not truth) and makes no truth-recovery claim."),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in receipt.items() if k != "rows"}, indent=2))
    print(f"\nn_use={n_use} powered={powered} peak={peak_layer} c={c} cc={cc} Lt={Lt}")
    print(f"L1={l1}(peak {peak_layer}>={Lt}) "
          f"L2={l2}(r_c={r_c:.3f} r_cc={r_cc:.3f} d={r_c - r_cc:.3f} p={sign_p}) "
          f"L3={l3}(|{peak_layer}-{c}|<=2) -> {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
