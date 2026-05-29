"""Second-model replication of the corrected confabulation mechanism.
PREREG_second_model_replication_2026_05_29.md.

Stage A (suppression-rhythm geometry): MEASURE the second model's install band from its OWN
flip-layer distribution. Stage B (disinhibition): TEST that band causally (I1/I2/I3). The band
is derived by a pre-committed rule from Stage A — not transferred from Qwen, not hand-tuned.

Default second model: meta-llama/Llama-3.2-1B-Instruct (16 decoder layers). Same n=36 arithmetic
items, SAE-free full-vocab logit-lens, in-code arithmetic truth hashed pre-scoring, exact-integer
correctness (no judge), greedy/deterministic.

Usage:
    python papers/grounded-honesty-axis/run_second_model_replication.py --n 6   # pilot
    python papers/grounded-honesty-axis/run_second_model_replication.py
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
                               is_numeric_token, GAMMAS)

RECEIPT = HERE / "second_model_replication_result.json"
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=len(SPECS))
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = ap.parse_args(argv)

    items = []
    for i, (form, delta, subset) in enumerate(SPECS[: args.n]):
        items.append((form, _eval(form), _expr(form), subset, i))

    key_blob = json.dumps([(e, c) for _, c, e, _, _ in items], ensure_ascii=False)
    key_hash = hashlib.sha256(key_blob.encode("utf-8")).hexdigest()
    print(f"answer-key SHA-256 (pre-scoring): {key_hash}")
    print(f"model={args.model} device={wb.DEVICE} items={len(items)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16).to(wb.DEVICE).eval()
    n_decoder = len(model.model.layers)
    digit_ids = single_digit_token_ids(tok)
    print(f"model loaded; n_decoder={n_decoder}; "
          f"single-digit token ids: { {d: i for d, i in digit_ids.items()} }\n")

    # ---------- pass 1: one-shot answers + Stage-A profiles ----------
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
        fl = prof["flip_layer"] if prof else None
        print(f"[{idx:2d}|{subset:9}] {expr:>14}={correct:<9} | "
              f"1shot {str(v1):>9} {'OK ' if ok1 else 'BAD'} | "
              f"cross={prof['crossed'] if prof else None} flip={fl}")

    # ---------- Stage A scoring (descriptive geometry) ----------
    profs = [r["prof"] for r in rows if r["prof"] is not None]
    L = profs[0]["n_layers"] if profs else (n_decoder + 1)
    late_thresh = int(np.ceil(2 * (L - 1) / 3))

    flips = [p["flip_layer"] for p in profs if p["crossed"] and p["flip_layer"] is not None]
    if flips:
        fa = np.array(flips, float)
        d2_late_frac = float(np.mean(fa >= late_thresh))
        d3_iqr = float(np.percentile(fa, 75) - np.percentile(fa, 25))
        flip_median = float(np.median(fa))
    else:
        d2_late_frac = d3_iqr = flip_median = float("nan")
    d2 = (d2_late_frac == d2_late_frac) and (d2_late_frac >= 0.60)
    d3 = (d3_iqr == d3_iqr) and (d3_iqr <= 5)

    dpos = [p for p in profs if p["digit_pos"] and p["distractors"]]
    per_item = []
    for p in dpos:
        c = 1.0 if p["corr_lead"] else 0.0
        dl = float(np.mean([1.0 if leads else 0.0 for _, leads in p["distractors"]]))
        per_item.append((c, dl))
    if len(per_item) >= 8:
        cvec = np.array([c for c, _ in per_item]); dvec = np.array([d for _, d in per_item])
        d1_corr_rate = float(cvec.mean()); d1_dist_rate = float(dvec.mean())
        d1_delta = d1_corr_rate - d1_dist_rate
        d1_t, d1_p = stats.ttest_rel(cvec, dvec)
        d1_powered = True
    else:
        d1_corr_rate = d1_dist_rate = d1_delta = d1_p = float("nan"); d1_powered = False
    # replication wants D1 NULL (correct not specially privileged)
    d1_truth_specific = (d1_powered and (d1_delta == d1_delta) and (d1_delta >= 0.20)
                         and (d1_p == d1_p) and (d1_p < 0.05) and (d1_delta > 0))
    d1_null = d1_powered and (not d1_truth_specific)

    # ---------- band-derivation rule (pre-committed, depth-proportional) ----------
    # hw and control-center are fractions of decoder depth that REPRODUCE Qwen's published
    # bands at N=28 (hw=2, cc=8 -> control [6,10]); target center from this model's OWN median.
    hw = max(1, int(round(2 * n_decoder / 28)))
    cc = int(round(8 * n_decoder / 28))
    CONTROL_BAND = (max(0, cc - hw), min(n_decoder - 1, cc + hw))
    if flip_median == flip_median:
        c = int(round(flip_median)) - 1
        lo = max(0, c - hw); hi = min(n_decoder - 1, c + hw)
        TARGET_BAND = (lo, hi)
    else:
        TARGET_BAND = None
    band_no_overlap = (TARGET_BAND is not None) and (TARGET_BAND[0] > CONTROL_BAND[1])
    band_is_late = (flip_median == flip_median) and (flip_median >= late_thresh)
    band_valid = band_no_overlap and band_is_late
    print(f"\nStage A: flip_median={flip_median} late_thresh={late_thresh} "
          f"late_frac={d2_late_frac} iqr={d3_iqr} -> TARGET_BAND={TARGET_BAND} "
          f"control={CONTROL_BAND} valid={band_valid}\n")

    # ---------- Stage B: disinhibition at the measured band ----------
    use = []
    if TARGET_BAND is not None:
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
            ent_base = entropy_of(base)
            rec = {"subset": r["subset"], "expr": r["expr"], "pos": pos,
                   "r_id": r_id, "c_id": c_id, "ent_base": ent_base}
            for g, sfx in ((0.0, "g0"), (0.5, "g50")):
                tg = logits_at(model, tok, r["p1"], r["a1"], pos, TARGET_BAND, g)
                cg = logits_at(model, tok, r["p1"], r["a1"], pos, CONTROL_BAND, g)
                ta = int(tg.argmax().item()); ca = int(cg.argmax().item())
                rec.update({
                    f"target_removed_{sfx}": int(ta != r_id),
                    f"ctrl_removed_{sfx}": int(ca != r_id),
                    f"target_recover_{sfx}": int(ta == c_id),
                    f"target_ent_{sfx}": entropy_of(tg),
                    f"target_coherent_{sfx}": int(is_numeric_token(tok, ta)),
                })
            dose = {}
            for g in GAMMAS:
                lg = base if g == 1.0 else logits_at(model, tok, r["p1"], r["a1"], pos,
                                                     TARGET_BAND, g)
                dose[g] = int(lg.argmax().item() == r_id)
            rec["dose_committed"] = dose
            use.append(rec)

    n_use = len(use)
    powered = n_use >= 12
    coh_g0 = float(np.mean([r["target_coherent_g0"] for r in use])) if use else float("nan")
    coh_g50 = float(np.mean([r["target_coherent_g50"] for r in use])) if use else float("nan")
    blunt = (coh_g0 == coh_g0) and (coh_g0 < 0.50)
    sfx = "g50" if blunt else "g0"
    coherence = coh_g50 if blunt else coh_g0

    f_target = float(np.mean([r[f"target_removed_{sfx}"] for r in use])) if use else float("nan")
    f_ctrl = float(np.mean([r[f"ctrl_removed_{sfx}"] for r in use])) if use else float("nan")
    t_only = sum(1 for r in use if r[f"target_removed_{sfx}"] and not r[f"ctrl_removed_{sfx}"])
    c_only = sum(1 for r in use if r[f"ctrl_removed_{sfx}"] and not r[f"target_removed_{sfx}"])
    nd = t_only + c_only
    sign_p = (float(stats.binomtest(t_only, nd, 0.5, alternative="greater").pvalue)
              if nd > 0 else float("nan"))
    i1 = (powered and (f_target - f_ctrl >= 0.30) and (f_target >= 0.50)
          and (sign_p == sign_p) and (sign_p < 0.05))

    removed = [r for r in use if r[f"target_removed_{sfx}"]]
    n_removed = len(removed); i2_powered = n_removed >= 6
    recover_rate = (float(np.mean([r[f"target_recover_{sfx}"] for r in removed]))
                    if removed else float("nan"))
    if removed:
        de = np.array([r[f"target_ent_{sfx}"] - r["ent_base"] for r in removed], float)
        if len(de) >= 2 and np.std(de) > 1e-12:
            ent_t, ent_p = stats.ttest_1samp(de, 0.0); ent_mean = float(de.mean())
        else:
            ent_p = float("nan"); ent_mean = float(de.mean()) if len(de) else float("nan")
    else:
        ent_p = float("nan"); ent_mean = float("nan")
    i2_install = (i2_powered and (recover_rate == recover_rate) and (recover_rate < 0.34)
                  and (ent_mean == ent_mean) and (ent_mean > 0)
                  and (ent_p == ent_p) and (ent_p < 0.05))
    i2_suppression = (i2_powered and (recover_rate == recover_rate) and (recover_rate >= 0.50))

    rate_by_g = {g: (float(np.mean([r["dose_committed"][g] for r in use])) if use else float("nan"))
                 for g in GAMMAS}
    gs = np.array(GAMMAS, float); rs = np.array([rate_by_g[g] for g in GAMMAS], float)
    if np.std(rs) > 1e-12:
        i3_rho, i3_p = stats.spearmanr(gs, rs)
    else:
        i3_rho, i3_p = float("nan"), float("nan")
    i3 = (i3_rho == i3_rho) and (i3_rho >= 0.90)

    stage_a_ok = d2 and d3 and d1_null
    stage_b_ok = i1 and i2_install
    replicated = bool(band_valid and stage_a_ok and stage_b_ok)
    result = "REPLICATION_SURVIVED" if replicated else "REPORT_AS_LANDED"

    # strip bulky text from rows for the receipt
    slim_rows = [{k: v for k, v in r.items() if k not in ("p1", "a1")} for r in rows]
    receipt = {
        "experiment": "second-model replication of the corrected confabulation mechanism (install + disinhibition)",
        "prereg": "papers/grounded-honesty-axis/PREREG_second_model_replication_2026_05_29.md",
        "answer_key_sha256_pre_scoring": key_hash,
        "model": args.model, "device": wb.DEVICE, "n_items": len(rows),
        "n_decoder_layers": n_decoder, "n_layers_hidden": L,
        "anchor_model_qwen": {"median_flip_hidden": 25.0, "late_fraction": 0.88,
                              "flip_iqr": 4.0, "d1_delta": -0.0078,
                              "target_band_decoder": [22, 26]},
        "core_signal": "exact integer parse vs in-code arithmetic truth (no judge)",
        "n_oneshot_confab": sum(1 for r in rows if not r["ok1"]),
        "n_alignable_confab": len(profs),
        "stageA_geometry": {
            "late_threshold_layer_index": late_thresh,
            "D1_truth_specific_flash": {
                "n_digit_pos_confab": len(per_item),
                "correct_digit_lead_rate": round(d1_corr_rate, 4) if d1_corr_rate == d1_corr_rate else None,
                "matched_distractor_lead_rate": round(d1_dist_rate, 4) if d1_dist_rate == d1_dist_rate else None,
                "delta": round(d1_delta, 4) if d1_delta == d1_delta else None,
                "paired_p": round(float(d1_p), 5) if d1_p == d1_p else None,
                "powered": d1_powered, "truth_specific": bool(d1_truth_specific),
                "null_as_in_qwen": bool(d1_null)},
            "D2_late_localized": {
                "n_crossings": len(flips),
                "late_fraction": round(d2_late_frac, 4) if d2_late_frac == d2_late_frac else None,
                "median_flip_layer": flip_median if flip_median == flip_median else None},
            "D3_rhythmic_tight_band": {
                "flip_layer_iqr": round(d3_iqr, 4) if d3_iqr == d3_iqr else None},
            "D1_null": bool(d1_null), "D2": bool(d2), "D3": bool(d3)},
        "band_derivation": {
            "rule": "depth-proportional (reproduces Qwen at N=28): hw=round(2N/28), cc=round(8N/28); target center=round(median flip_hidden)-1; TARGET=[c-hw,c+hw]; CONTROL=[cc-hw,cc+hw]",
            "half_width": hw,
            "target_band_decoder": list(TARGET_BAND) if TARGET_BAND else None,
            "control_band_decoder": list(CONTROL_BAND),
            "no_overlap": bool(band_no_overlap), "is_late": bool(band_is_late),
            "band_valid": bool(band_valid)},
        "stageB_disinhibition": {
            "n_usable_confab": n_use, "powered": powered,
            "scoring_gamma": (0.5 if blunt else 0.0),
            "coherence_rate_target_g0": round(coh_g0, 4) if coh_g0 == coh_g0 else None,
            "coherence_rate_target_g50": round(coh_g50, 4) if coh_g50 == coh_g50 else None,
            "blunt_fallback_triggered": bool(blunt),
            "I1_late_band_causes_commitment": {
                "f_target_removed": round(f_target, 4) if f_target == f_target else None,
                "f_ctrl_removed": round(f_ctrl, 4) if f_ctrl == f_ctrl else None,
                "delta": round(f_target - f_ctrl, 4) if (f_target == f_target and f_ctrl == f_ctrl) else None,
                "discordant_target_only": t_only, "discordant_ctrl_only": c_only,
                "sign_test_p": round(sign_p, 5) if sign_p == sign_p else None},
            "I2_disinhibition_yields_uncertainty": {
                "n_commitment_removed": n_removed, "powered": i2_powered,
                "truth_recovery_rate": round(recover_rate, 4) if recover_rate == recover_rate else None,
                "entropy_rise_mean": round(ent_mean, 4) if ent_mean == ent_mean else None,
                "entropy_rise_p": round(ent_p, 5) if ent_p == ent_p else None,
                "branch": ("installation" if i2_install else
                           ("suppression" if i2_suppression else "ambiguous"))},
            "I3_dose_response": {
                "commitment_rate_by_gamma": {str(g): (round(rate_by_g[g], 4)
                                             if rate_by_g[g] == rate_by_g[g] else None) for g in GAMMAS},
                "spearman_rho": round(float(i3_rho), 4) if i3_rho == i3_rho else None,
                "p": round(float(i3_p), 5) if i3_p == i3_p else None},
            "I1": bool(i1), "I2_install": bool(i2_install),
            "I2_suppression": bool(i2_suppression), "I3": bool(i3)},
        "rows": slim_rows,
        "stage_a_ok": bool(stage_a_ok), "stage_b_ok": bool(stage_b_ok),
        "RESULT": result,
        "honest_scope": (
            "second small open model (Llama-3.2-1B-Instruct, 16 layers) replicating the Qwen "
            "corrected mechanism; SAE-free full-vocab logit-lens; single-position teacher-forced "
            "single confirmatory run per stage; feasibility-grade n=36; arithmetic ground truth "
            "computed in-code then hashed pre-scoring; exact-integer correctness (no judge); "
            "greedy/deterministic. The target band is derived from Llama's OWN Stage-A flip "
            "geometry by the pre-committed rule, NOT transferred from Qwen and NOT hand-tuned; "
            "control is an early matched-size band. A REPLICATION_SURVIVED upgrades the mechanism "
            "from one model to two architectures (bands each found by their own geometry); it does "
            "NOT claim universality and does NOT touch the standing correctness bound (every "
            "internal lever moves confidence; only re-derivation moves correctness)."),
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in receipt.items() if k != "rows"}, indent=2))
    print(f"\nStage A: D1_null={d1_null} D2={d2}(late {d2_late_frac}) D3={d3}(iqr {d3_iqr})")
    print(f"band_valid={band_valid} TARGET={TARGET_BAND} CONTROL={CONTROL_BAND}")
    print(f"Stage B: n_use={n_use} I1={i1}(f_t={f_target} f_c={f_ctrl} p={sign_p}) "
          f"I2={i2_install}(rec={recover_rate} dent={ent_mean}) I3={i3}(rho={i3_rho})")
    print(f"-> {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
