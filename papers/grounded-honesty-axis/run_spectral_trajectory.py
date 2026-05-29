"""Is truth in the SPECTRUM of the answer-formation trajectory?
PREREG_spectral_trajectory_2026_05_29.md.

Reads, per generation, the layer-by-layer logit-lens trajectory of each answer DIGIT
token (logit of the realized digit token under final_norm+unembed at each layer). The
1/f spectral slope beta and a robust short-series "snap index" summarize the trajectory's
SHAPE. Prediction: construction (method-diverse derivation) builds the answer gradually
across layers -> pinker (higher beta, lower snap); retrieval (one-shot confabulation)
snaps onto an attractor -> whiter (beta->0, higher snap). Companion test: on confabs,
does the CORRECT token's logit-lens lead at an intermediate layer before a late hop
overwrites it ("truth-flash-then-death")?

Ground truth is computed in-code (arithmetic) and hashed before scoring; correctness is
exact integer match (no judge). Greedy/deterministic -> reproduces the white-box answer key.

Usage:
    python papers/grounded-honesty-axis/run_spectral_trajectory.py --n 6   # pilot
    python papers/grounded-honesty-axis/run_spectral_trajectory.py
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
from run_depth_grounding_whitebox import (  # noqa: E402
    METHODS, MODEL_NAME, DEVICE, generate, parse_answer_line, auc,
)

RECEIPT = HERE / "spectral_trajectory_result.json"


@torch.no_grad()
def lens_at_answer(model, tok, prompt_text, answer_text):
    """Return (sel (L,A,d) normed residuals at answer positions, tgt ids (A,), W)."""
    plen = tok(prompt_text, return_tensors="pt").input_ids.shape[1]
    fids = tok(prompt_text + answer_text, return_tensors="pt").input_ids.to(DEVICE)
    T = fids.shape[1]
    if T <= plen:
        return None
    out = model(fids, output_hidden_states=True)
    hs = torch.stack(out.hidden_states, dim=0)[:, 0, :, :]   # (L, T, d)
    normed = model.model.norm(hs)                            # (L, T, d)
    ans_pos = torch.arange(plen - 1, T - 1, device=DEVICE)
    sel = normed[:, ans_pos, :].float()                      # (L, A, d)
    tgt = fids[0, plen:T]                                    # (A,)
    return sel, tgt, model.lm_head.weight.float()


def digit_columns(tok, tgt_ids):
    cols = []
    for a, tid in enumerate(tgt_ids):
        if any(ch.isdigit() for ch in tok.decode([int(tid)])):
            cols.append(a)
    return cols


def spectral_beta(traj):
    """1/f^beta slope of a 1-D trajectory via detrended periodogram."""
    x = np.asarray(traj, float)
    n = len(x)
    if n < 8:
        return float("nan")
    t = np.arange(n, dtype=float)
    A = np.column_stack([np.ones(n), t])
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    x = x - A @ coef                                         # remove linear trend
    F = np.fft.rfft(x)
    P = (F.real ** 2 + F.imag ** 2)[1:]                      # drop DC
    freqs = np.arange(1, len(P) + 1, dtype=float)
    mask = P > 0
    if mask.sum() < 4:
        return float("nan")
    B = np.column_stack([np.ones(int(mask.sum())), np.log(freqs[mask])])
    s, *_ = np.linalg.lstsq(B, np.log(P[mask]), rcond=None)
    return float(-s[1])


def snap_index(traj):
    """Fraction of total monotone rise contributed by the single steepest step."""
    d = np.diff(np.asarray(traj, float))
    pos = d[d > 0]
    tot = float(pos.sum())
    if tot <= 1e-9:
        return float("nan")
    return float(pos.max() / tot)


def traj_stats(logits, cols):
    """Per-item beta/snap averaged over digit columns + bootstrap CI width on beta."""
    betas = [b for c in cols if (b := spectral_beta(logits[:, c])) == b]
    snaps = [s for c in cols if (s := snap_index(logits[:, c])) == s]
    if not betas:
        return None
    beta = float(np.mean(betas))
    snap = float(np.mean(snaps)) if snaps else float("nan")
    if len(betas) >= 2:
        rng = np.random.default_rng(0)
        bs = [float(np.mean(rng.choice(betas, len(betas), replace=True))) for _ in range(200)]
        ciw = float(np.percentile(bs, 97.5) - np.percentile(bs, 2.5))
    else:
        ciw = float("nan")
    return {"beta": beta, "snap": snap, "ciw": ciw, "n_cols": len(betas)}


@torch.no_grad()
def flash_crossing(model, tok, prompt_text, realized_text, correct_str):
    """On a confab, does the CORRECT token lead at some layer then lose at the final layer?"""
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
        return None                                          # un-alignable / realized == correct
    pos = plen - 1 + div                                     # position predicting answer token `div`
    out = model(fids, output_hidden_states=True)
    hs = torch.stack(out.hidden_states, dim=0)[:, 0, :, :]
    vec = model.model.norm(hs)[:, pos, :].float()            # (L, d)
    W = model.lm_head.weight.float()
    lc = vec @ W[corr_ids[div]]                              # (L,) correct-token logit
    lr = vec @ W[real_ids[div]]                              # (L,) realized-token logit
    lead = (lc > lr)
    crossed = bool(lead[:-1].any().item()) and bool((lr[-1] >= lc[-1]).item())
    return crossed


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
    print(f"model={MODEL_NAME} device={DEVICE} items={len(items)}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16).to(DEVICE).eval()
    ctrl_id = tok(" the", add_special_tokens=False).input_ids[-1]
    print(f"model loaded; control token id={ctrl_id} ({tok.decode([ctrl_id])!r})\n")

    rows = []
    for form, correct, expr, subset, idx in items:
        p1, a1 = generate(model, tok, "Answer with only the final number, nothing else.",
                          f"What is {expr}?", max_new_tokens=16)
        v1 = parse_int(a1); ok1 = (v1 == correct)
        method = METHODS[idx % len(METHODS)]
        p2, a2 = generate(model, tok, method, f"What is {expr}?", max_new_tokens=320)
        v2 = parse_answer_line(a2); ok2 = (v2 == correct)

        row = {"subset": subset, "expr": expr, "correct": correct,
               "v1": v1, "ok1": ok1, "v2": v2, "ok2": ok2,
               "s1": None, "s2": None, "ctrl1": None, "ctrl2": None, "cross": None}

        for tag, ptxt, atxt in (("1", p1, a1), ("2", p2, a2)):
            r = lens_at_answer(model, tok, ptxt, atxt)
            if r is None:
                continue
            sel, tgt, W = r
            cols = digit_columns(tok, tgt)
            if not cols:
                continue
            real_logits = torch.einsum("lad,ad->la", sel, W[tgt].float()).cpu().numpy()
            ctrl_logits = torch.einsum("lad,d->la", sel, W[ctrl_id]).cpu().numpy()
            row["s" + tag] = traj_stats(real_logits, cols)
            row["ctrl" + tag] = traj_stats(ctrl_logits, cols)

        if not ok1 and v1 is not None:
            row["cross"] = flash_crossing(model, tok, p1, a1, str(correct))

        rows.append(row)
        b1 = row["s1"]["beta"] if row["s1"] else float("nan")
        b2 = row["s2"]["beta"] if row["s2"] else float("nan")
        print(f"[{idx:2d}|{subset:9}] {expr:>14}={correct:<9} | "
              f"1shot {str(v1):>9} {'OK ' if ok1 else 'BAD'} b={b1:5.2f} | "
              f"deriv {str(v2):>9} {'OK ' if ok2 else 'BAD'} b={b2:5.2f} | cross={row['cross']}")

    # ---- scoring ----
    def beta(row, tag):
        s = row["s" + tag]
        return s["beta"] if s else float("nan")

    def snap(row, tag):
        s = row["s" + tag]
        return s["snap"] if s else float("nan")

    def cbeta(row, tag):
        s = row["ctrl" + tag]
        return s["beta"] if s else float("nan")

    confab = [r for r in rows if not r["ok1"]]

    # F1: paired beta, derivation vs one-shot confab (same items). Predicted deriv pinker.
    f1_pairs = [(beta(r, "1"), beta(r, "2")) for r in confab
                if beta(r, "1") == beta(r, "1") and beta(r, "2") == beta(r, "2")]
    if len(f1_pairs) >= 3:
        b1s = [a for a, _ in f1_pairs]; b2s = [b for _, b in f1_pairs]
        diff = np.array(b2s) - np.array(b1s)                 # deriv - confab (predicted >0)
        f1_t, f1_p = stats.ttest_rel(b2s, b1s)
        f1_d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))
        f1_sign = "deriv_pinker" if np.mean(diff) > 0 else "deriv_whiter"
        f1 = (abs(f1_d) >= 0.5) and (f1_p < 0.05) and (np.mean(diff) > 0)
    else:
        f1_d = f1_p = float("nan"); f1 = False; f1_sign = "insufficient"

    # F2: within-mode AUC(beta) correct vs confab, well-powered stratum (>=8 vs >=8).
    def within(tag, okkey):
        cor = [beta(r, tag) for r in rows if r[okkey] and beta(r, tag) == beta(r, tag)]
        wr = [beta(r, tag) for r in rows if not r[okkey] and beta(r, tag) == beta(r, tag)]
        return cor, wr
    d_cor, d_wr = within("2", "ok2")                         # derivation stratum
    o_cor, o_wr = within("1", "ok1")                         # one-shot stratum
    f2_deriv = auc(d_cor, d_wr)
    f2_os = auc(o_cor, o_wr)
    strata = []
    if len(d_cor) >= 8 and len(d_wr) >= 8:
        strata.append(("derivation", f2_deriv))
    if len(o_cor) >= 8 and len(o_wr) >= 8:
        strata.append(("oneshot", f2_os))
    f2 = any(a == a and (a >= 0.70 or a <= 0.30) for _, a in strata)
    f2_powered = bool(strata)

    # F3: truth-flash crossing rate among alignable confab items.
    crosses = [r["cross"] for r in confab if r["cross"] is not None]
    f3_rate = (sum(1 for c in crosses if c) / len(crosses)) if crosses else float("nan")
    f3 = (f3_rate == f3_rate) and (f3_rate >= 0.25)

    # K: (a) control-token shows NO mode difference; (b) snap agrees with beta on F1.
    kc_pairs = [(cbeta(r, "1"), cbeta(r, "2")) for r in confab
                if cbeta(r, "1") == cbeta(r, "1") and cbeta(r, "2") == cbeta(r, "2")]
    if len(kc_pairs) >= 3:
        _, k_ctrl_p = stats.ttest_rel([b for _, b in kc_pairs], [a for a, _ in kc_pairs])
    else:
        k_ctrl_p = float("nan")
    sn_pairs = [(snap(r, "1"), snap(r, "2")) for r in confab
                if snap(r, "1") == snap(r, "1") and snap(r, "2") == snap(r, "2")]
    snap_confab_higher = (len(sn_pairs) >= 3 and
                          np.mean([s1 - s2 for s1, s2 in sn_pairs]) > 0)
    k = ((k_ctrl_p != k_ctrl_p or k_ctrl_p > 0.05) and f1 and snap_confab_higher)

    # precondition: short-series estimator resolution (median per-item beta CI width).
    ciws = [r["s1"]["ciw"] for r in rows if r["s1"] and r["s1"]["ciw"] == r["s1"]["ciw"]]
    ciws += [r["s2"]["ciw"] for r in rows if r["s2"] and r["s2"]["ciw"] == r["s2"]["ciw"]]
    med_ciw = float(np.median(ciws)) if ciws else float("nan")
    precondition = (med_ciw == med_ciw) and (med_ciw <= 1.0)

    receipt = {
        "experiment": "spectral signature of the answer-formation trajectory (1/f beta + truth-flash)",
        "prereg": "papers/grounded-honesty-axis/PREREG_spectral_trajectory_2026_05_29.md",
        "answer_key_sha256_pre_scoring": key_hash,
        "model": MODEL_NAME, "device": DEVICE, "n_items": len(rows),
        "core_signal": "exact integer parse vs in-code arithmetic truth (no judge)",
        "control_token": tok.decode([ctrl_id]),
        "n_oneshot_confab": len(confab),
        "F1_spectral_mode_diff": {
            "n_pairs": len(f1_pairs),
            "cohens_d_paired": round(f1_d, 4) if f1_d == f1_d else None,
            "p": round(float(f1_p), 5) if f1_p == f1_p else None,
            "mean_beta_oneshot_confab": round(float(np.mean([a for a, _ in f1_pairs])), 4) if f1_pairs else None,
            "mean_beta_derivation": round(float(np.mean([b for _, b in f1_pairs])), 4) if f1_pairs else None,
            "sign": f1_sign},
        "F2_within_mode_truth_auc": {
            "derivation_auc": round(f2_deriv, 4) if f2_deriv == f2_deriv else None,
            "derivation_n_correct": len(d_cor), "derivation_n_confab": len(d_wr),
            "oneshot_auc": round(f2_os, 4) if f2_os == f2_os else None,
            "oneshot_n_correct": len(o_cor), "oneshot_n_confab": len(o_wr),
            "powered_strata": [s for s, _ in strata]},
        "F3_truth_flash_crossing": {
            "n_alignable_confab": len(crosses),
            "crossing_rate": round(f3_rate, 4) if f3_rate == f3_rate else None},
        "K_control_and_snap": {
            "control_token_mode_p": round(float(k_ctrl_p), 5) if k_ctrl_p == k_ctrl_p else None,
            "snap_confab_higher": bool(snap_confab_higher),
            "mean_snap_oneshot_confab": round(float(np.mean([s1 for s1, _ in sn_pairs])), 4) if sn_pairs else None,
            "mean_snap_derivation": round(float(np.mean([s2 for _, s2 in sn_pairs])), 4) if sn_pairs else None},
        "precondition_estimator_resolution": {
            "median_beta_ci_width": round(med_ciw, 4) if med_ciw == med_ciw else None,
            "met": bool(precondition)},
        "rows": rows,
        "F1": bool(f1), "F2": bool(f2), "F2_powered": f2_powered,
        "F3": bool(f3), "K": bool(k),
        "RESULT": ("SURVIVED" if (f1 and f2 and f3 and k and precondition)
                   else "REPORT_AS_LANDED"),
        "honest_scope": (
            "single open model Qwen2.5-1.5B-Instruct, SAE-free logit-lens trajectories, "
            "feasibility-grade n=36, one confirmatory run; arithmetic ground truth computed "
            "in-code then hashed pre-scoring; exact-integer correctness (no judge). beta is "
            "estimated from a ~29-point trajectory (crude spectral measurement, reported with "
            "a bootstrap CI and the snap-index corroborator), not a clean 1/f fit. Tests "
            "whether the scale structure of the answer-formation trajectory carries mode "
            "and/or truth information beyond scalar mean depth. A null does not refute the "
            "1/f framing in general (may need an SAE-feature trajectory or a deeper model)."),
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in receipt.items() if k != "rows"}, indent=2))
    print(f"\nprecond={precondition}(ciw={med_ciw:.2f}) F1={f1}({f1_sign}) "
          f"F2={f2}(deriv {f2_deriv:.2f}) F3={f3}(rate {f3_rate:.2f}) K={k} -> {receipt['RESULT']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
