"""Disinhibition: is the late hop (decoder layers ≈22–26) CAUSALLY what installs the
confident wrong answer — and does dampening it yield UNCERTAINTY, not truth?
PREREG_disinhibition_2026_05_29.md.

Causal counterpart of the suppression-rhythm finding. For each one-shot confabulation,
teacher-force prompt+realized_answer and, at the first divergent answer position `pos`,
attenuate a BAND of decoder layers' residual WRITE: h_out -> h_in + g*(h_out - h_in)
(g=0 = full knockdown of that layer's write at pos). Read the next-token distribution at
`pos` (final norm + lm_head, full vocab) under the hook. Compare the measured TARGET band
[22,26] against a matched EARLY control band [6,10].

Ground truth computed in-code (arithmetic) and hashed before scoring; correctness is exact
integer match (no judge). Greedy/deterministic -> reproduces the white-box answer key.

Usage:
    python papers/grounded-honesty-axis/run_disinhibition.py --n 6   # pilot
    python papers/grounded-honesty-axis/run_disinhibition.py
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

RECEIPT = HERE / "disinhibition_result.json"

TARGET_BAND = (22, 26)   # decoder layers; hidden-state idx ≈23–27 (suppression-rhythm band)
CONTROL_BAND = (6, 10)   # matched-size early control
GAMMAS = [1.0, 0.75, 0.5, 0.25, 0.0]


def divergence(tok, prompt_text, realized_text, correct_str):
    """Return (pos, r_id, c_id) at the first divergent answer token, or None."""
    plen = tok(prompt_text, return_tensors="pt").input_ids.shape[1]
    fids = tok(prompt_text + realized_text, return_tensors="pt").input_ids
    T = fids.shape[1]
    if T <= plen:
        return None
    real_ids = fids[0, plen:T].tolist()
    corr_ids = tok(correct_str, add_special_tokens=False).input_ids
    div = next((i for i in range(min(len(real_ids), len(corr_ids)))
                if real_ids[i] != corr_ids[i]), None)
    if div is None:
        return None
    return plen - 1 + div, real_ids[div], corr_ids[div]


def _make_hook(pos, gamma):
    def hook(module, inputs, output):
        h_in = inputs[0]
        h_out = output[0] if isinstance(output, tuple) else output
        new = h_out.clone()
        new[:, pos, :] = h_in[:, pos, :] + gamma * (h_out[:, pos, :] - h_in[:, pos, :])
        if isinstance(output, tuple):
            return (new,) + tuple(output[1:])
        return new
    return hook


@torch.no_grad()
def logits_at(model, tok, prompt_text, realized_text, pos, band, gamma):
    """Next-token logits at `pos` with decoder layers in [band] write-attenuated by gamma."""
    fids = tok(prompt_text + realized_text, return_tensors="pt").input_ids.to(DEVICE)
    handles = []
    if band is not None and gamma != 1.0:
        lo, hi = band
        for i in range(lo, hi + 1):
            handles.append(model.model.layers[i].register_forward_hook(_make_hook(pos, gamma)))
    try:
        out = model(fids)
        logits = out.logits[0, pos, :].float()
    finally:
        for h in handles:
            h.remove()
    return logits


def entropy_of(logits):
    p = torch.softmax(logits, dim=-1)
    return float(-(p * torch.log(p.clamp_min(1e-12))).sum().item())


def is_numeric_token(tok, tid):
    s = tok.decode([tid]).strip()
    return bool(s) and any(ch.isdigit() for ch in s)


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
    print(f"model={MODEL_NAME} device={DEVICE} items={len(items)} "
          f"target={TARGET_BAND} control={CONTROL_BAND}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16).to(DEVICE).eval()
    print("model loaded\n")

    rows = []
    for form, correct, expr, subset, idx in items:
        p1, a1 = generate(model, tok, "Answer with only the final number, nothing else.",
                          f"What is {expr}?", max_new_tokens=16)
        v1 = parse_int(a1); ok1 = (v1 == correct)
        row = {"subset": subset, "expr": expr, "correct": correct, "v1": v1, "ok1": ok1,
               "usable": False}
        if (not ok1) and v1 is not None:
            dv = divergence(tok, p1, a1, str(correct))
            if dv is not None:
                pos, r_id, c_id = dv
                base = logits_at(model, tok, p1, a1, pos, None, 1.0)
                base_arg = int(base.argmax().item())
                if base_arg == r_id:                     # clean baseline only
                    ent_base = entropy_of(base)
                    row.update({"usable": True, "pos": pos, "r_id": r_id, "c_id": c_id,
                                "ent_base": ent_base})
                    # target & control knockdown at the primary g=0 and the pre-named
                    # g=0.5 fallback, for both bands
                    for g, sfx in ((0.0, "g0"), (0.5, "g50")):
                        tg = logits_at(model, tok, p1, a1, pos, TARGET_BAND, g)
                        cg = logits_at(model, tok, p1, a1, pos, CONTROL_BAND, g)
                        ta = int(tg.argmax().item()); ca = int(cg.argmax().item())
                        row.update({
                            f"target_removed_{sfx}": int(ta != r_id),
                            f"ctrl_removed_{sfx}": int(ca != r_id),
                            f"target_recover_{sfx}": int(ta == c_id),
                            f"target_ent_{sfx}": entropy_of(tg),
                            f"target_coherent_{sfx}": int(is_numeric_token(tok, ta)),
                        })
                    # dose-response over target band (commitment indicator at each gamma)
                    dose = {}
                    for g in GAMMAS:
                        lg = base if g == 1.0 else logits_at(model, tok, p1, a1, pos,
                                                             TARGET_BAND, g)
                        dose[g] = int(lg.argmax().item() == r_id)   # 1 = still committed
                    row["dose_committed"] = dose
        rows.append(row)
        tag = ("rm" if row.get("target_removed_g0") else "keep") if row["usable"] else "skip"
        print(f"[{idx:2d}|{subset:9}] {expr:>14}={correct:<9} | "
              f"1shot {str(v1):>9} {'OK ' if ok1 else 'BAD'} | {tag}")

    # ---- scoring ----
    use = [r for r in rows if r["usable"]]
    n_use = len(use)
    powered = n_use >= 12

    # pre-registered fallback: g=0 is primary; if its coherence < 0.50 it is too blunt,
    # re-score I1/I2 at the pre-named g=0.5 (both bands), reporting g=0 alongside.
    coh_g0 = float(np.mean([r["target_coherent_g0"] for r in use])) if use else float("nan")
    coh_g50 = float(np.mean([r["target_coherent_g50"] for r in use])) if use else float("nan")
    blunt = (coh_g0 == coh_g0) and (coh_g0 < 0.50)
    sfx = "g50" if blunt else "g0"
    coherence = coh_g50 if blunt else coh_g0

    f_target = float(np.mean([r[f"target_removed_{sfx}"] for r in use])) if use else float("nan")
    f_ctrl = float(np.mean([r[f"ctrl_removed_{sfx}"] for r in use])) if use else float("nan")
    # discordant-pair sign test (target-only-removed vs ctrl-only-removed)
    t_only = sum(1 for r in use if r[f"target_removed_{sfx}"] and not r[f"ctrl_removed_{sfx}"])
    c_only = sum(1 for r in use if r[f"ctrl_removed_{sfx}"] and not r[f"target_removed_{sfx}"])
    nd = t_only + c_only
    sign_p = (float(stats.binomtest(t_only, nd, 0.5, alternative="greater").pvalue)
              if nd > 0 else float("nan"))
    i1 = (powered and (f_target - f_ctrl >= 0.30) and (f_target >= 0.50)
          and (sign_p == sign_p) and (sign_p < 0.05))

    removed = [r for r in use if r[f"target_removed_{sfx}"]]
    n_removed = len(removed)
    i2_powered = n_removed >= 6
    recover_rate = (float(np.mean([r[f"target_recover_{sfx}"] for r in removed]))
                    if removed else float("nan"))
    if removed:
        de = np.array([r[f"target_ent_{sfx}"] - r["ent_base"] for r in removed], float)
        if len(de) >= 2 and np.std(de) > 1e-12:
            ent_t, ent_p = stats.ttest_1samp(de, 0.0)
            ent_mean = float(de.mean())
        else:
            ent_p = float("nan"); ent_mean = float(de.mean()) if len(de) else float("nan")
    else:
        ent_p = float("nan"); ent_mean = float("nan")
    i2_install = (i2_powered and (recover_rate == recover_rate) and (recover_rate < 0.34)
                  and (ent_mean == ent_mean) and (ent_mean > 0)
                  and (ent_p == ent_p) and (ent_p < 0.05))
    i2_suppression = (i2_powered and (recover_rate == recover_rate) and (recover_rate >= 0.50))

    # I3 dose-response (corroborator)
    rate_by_g = {g: (float(np.mean([r["dose_committed"][g] for r in use])) if use else float("nan"))
                 for g in GAMMAS}
    gs = np.array(GAMMAS, float)
    rs = np.array([rate_by_g[g] for g in GAMMAS], float)
    if np.std(rs) > 1e-12:
        i3_rho, i3_p = stats.spearmanr(gs, rs)
    else:
        i3_rho, i3_p = float("nan"), float("nan")
    i3 = (i3_rho == i3_rho) and (i3_rho >= 0.90)

    result = "SURVIVED" if (i1 and i2_install) else "REPORT_AS_LANDED"

    receipt = {
        "experiment": "disinhibition — is the late band causally the install, and does dampening yield uncertainty not truth?",
        "prereg": "papers/grounded-honesty-axis/PREREG_disinhibition_2026_05_29.md",
        "answer_key_sha256_pre_scoring": key_hash,
        "model": MODEL_NAME, "device": DEVICE, "n_items": len(rows),
        "target_band": list(TARGET_BAND), "control_band": list(CONTROL_BAND),
        "core_signal": "single-position teacher-forced next-token argmax (full vocab) vs in-code truth (no judge)",
        "n_usable_confab": n_use, "powered": powered,
        "I1_late_band_causes_commitment": {
            "f_target_removed": round(f_target, 4) if f_target == f_target else None,
            "f_ctrl_removed": round(f_ctrl, 4) if f_ctrl == f_ctrl else None,
            "delta": round(f_target - f_ctrl, 4) if (f_target == f_target and f_ctrl == f_ctrl) else None,
            "discordant_target_only": t_only, "discordant_ctrl_only": c_only,
            "sign_test_p": round(sign_p, 5) if sign_p == sign_p else None},
        "scoring_gamma": (0.5 if blunt else 0.0),
        "coherence_rate_target_g0": round(coh_g0, 4) if coh_g0 == coh_g0 else None,
        "coherence_rate_target_g50": round(coh_g50, 4) if coh_g50 == coh_g50 else None,
        "coherence_rate_scored": round(coherence, 4) if coherence == coherence else None,
        "blunt_fallback_triggered": bool(blunt),
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
        "rows": rows,
        "I1": bool(i1), "I2_install": bool(i2_install), "I2_suppression": bool(i2_suppression),
        "I3": bool(i3),
        "RESULT": result,
        "honest_scope": (
            "single open model Qwen2.5-1.5B-Instruct; SAE-free full-vocab logit-lens readout; "
            "feasibility-grade n=36; one confirmatory run; arithmetic ground truth computed "
            "in-code then hashed pre-scoring; exact-integer correctness (no judge); "
            "greedy/deterministic. The intervention is a SINGLE-POSITION, teacher-forced "
            "next-token knockdown of a band's residual write (h_in + g*(h_out-h_in) at pos), "
            "tested at the divergence position only — not downstream multi-token regeneration. "
            "Target band [22,26] is fixed from the suppression-rhythm finding, not re-tuned. "
            "Causal LOCALIZATION test within one model. A null refutes only the causal "
            "localizability of the install to this band by this surgical method; it leaves the "
            "descriptive suppression-rhythm finding and the standing arc intact."),
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in receipt.items() if k != "rows"}, indent=2))
    print(f"\nn_use={n_use} powered={powered} coh_g0={coh_g0:.3f} coh_g50={coh_g50:.3f} "
          f"blunt={blunt} scoring_gamma={0.5 if blunt else 0.0}")
    print(f"I1={i1}(f_t={f_target:.3f} f_c={f_ctrl:.3f} p={sign_p}) "
          f"I2={i2_install}(rec={recover_rate} dent={ent_mean} p={ent_p}) "
          f"I3={i3}(rho={i3_rho}) -> {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
