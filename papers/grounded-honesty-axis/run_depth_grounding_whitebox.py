"""White-box unification: does attribution DEPTH explain the belief->truth dial?
PREREG_depth_grounding_whitebox.md.

Measures a SAE-free logit-lens direct-logit-attribution depth (attribution-weighted
mean layer index) on an OPEN model's OWN generations, contrasting one-shot answers
(retrieval mode -> confident confabulation on hard arithmetic) vs method-diverse CoT
derivations (construction mode). Ground truth is computed in-code (arithmetic) and
hashed before scoring; correctness is exact integer match (no judge).

Depth metric (per PREREG): for the generated answer span, at each answer token t and
layer l, logit-lens the realized next token (final_norm + unembed of layer-l residual)
to get logit_l; a_l = relu(logit_l - logit_{l-1}); D = mean_t ( sum_l l*a_l / sum_l a_l ).

Usage:
    python papers/grounded-honesty-axis/run_depth_grounding_whitebox.py --n 6   # pilot
    python papers/grounded-honesty-axis/run_depth_grounding_whitebox.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_competence_cliff import SPECS, _eval, _expr, parse_int  # noqa: E402

RECEIPT = HERE / "depth_grounding_whitebox_result.json"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

METHODS = [
    "Solve it step by step, showing your work. On the final line write 'ANSWER: <number>'.",
    "Decompose the larger factor into hundreds, tens and ones; multiply each part "
    "separately and add the partial results. On the final line write 'ANSWER: <number>'.",
    "Use long multiplication: write each partial product, then sum them. On the final "
    "line write 'ANSWER: <number>'.",
    "First estimate the magnitude, then compute the exact value carefully. On the final "
    "line write 'ANSWER: <number>'.",
    "Compute it digit by digit, tracking each carry, then verify by re-adding. On the "
    "final line write 'ANSWER: <number>'.",
]
_ANSWER = re.compile(r"ANSWER:\s*(-?[\d,]+)", re.IGNORECASE)


def parse_answer_line(text: str):
    m = _ANSWER.search(text or "")
    if m:
        return parse_int(m.group(1))
    ints = re.findall(r"-?\d[\d,]*", (text or "").replace(" ", ""))
    return parse_int(ints[-1]) if ints else None


def _render_chat(tok, system, user):
    """System+user chat prompt; fold the system text into the user turn for templates that
    reject a system role (e.g. Gemma-2). Byte-identical for templates that accept a system role."""
    try:
        return tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        return tok.apply_chat_template(
            [{"role": "user", "content": f"{system}\n\n{user}"}],
            tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tok, system, user, max_new_tokens):
    text = _render_chat(tok, system, user)
    ids = tok(text, return_tensors="pt").to(DEVICE)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    gen_ids = out[0, ids.input_ids.shape[1]:]
    return text, tok.decode(gen_ids, skip_special_tokens=True)


@torch.no_grad()
def depth_of(model, tok, prompt_text, answer_text):
    """Logit-lens attribution-weighted mean layer index over the answer span."""
    plen = tok(prompt_text, return_tensors="pt").input_ids.shape[1]
    fids = tok(prompt_text + answer_text, return_tensors="pt").input_ids.to(DEVICE)
    T = fids.shape[1]
    if T <= plen:
        return float("nan"), 0
    out = model(fids, output_hidden_states=True)
    hs = torch.stack(out.hidden_states, dim=0)[:, 0, :, :]   # (L, T, d)
    L = hs.shape[0]
    normed = model.model.norm(hs)                            # (L, T, d)
    W = model.lm_head.weight                                 # (vocab, d)
    ans_pos = torch.arange(plen - 1, T - 1, device=DEVICE)
    tgt = fids[0, plen:T]                                    # (A,)
    sel = normed[:, ans_pos, :]                              # (L, A, d)
    logits = torch.einsum("lad,ad->la", sel.float(), W[tgt].float())  # (L, A)
    a = (logits[1:] - logits[:-1]).clamp_min(0.0)           # (L-1, A)
    lidx = torch.arange(1, L, device=DEVICE, dtype=a.dtype).unsqueeze(1)
    s = a.sum(0)                                             # (A,)
    d = (lidx * a).sum(0) / s.clamp_min(1e-9)               # (A,)
    valid = s > 1e-9
    if valid.sum() == 0:
        return float("nan"), 0
    return float(d[valid].mean().item()), int(valid.sum().item())


def auc(pos, neg):
    pos = [p for p in pos if p == p]
    neg = [n for n in neg if n == n]
    if not pos or not neg:
        return float("nan")
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


def intercept_test(dx, dy):
    """OLS dy ~ a + b*dx; return (intercept, p_intercept)."""
    dx = np.asarray(dx, float); dy = np.asarray(dy, float)
    X = np.column_stack([np.ones(len(dx)), dx])
    beta, *_ = np.linalg.lstsq(X, dy, rcond=None)
    resid = dy - X @ beta
    dof = max(1, len(dx) - 2)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = float(np.sqrt(cov[0, 0]))
    t = beta[0] / se if se > 0 else 0.0
    p = 2 * (1 - stats.t.cdf(abs(t), dof))
    return float(beta[0]), float(p)


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
    print("model loaded\n")

    rows = []
    for form, correct, expr, subset, idx in items:
        # one-shot (retrieval mode)
        p1, a1 = generate(model, tok,
                          "Answer with only the final number, nothing else.",
                          f"What is {expr}?", max_new_tokens=16)
        v1 = parse_int(a1)
        ok1 = (v1 == correct)
        d1, n1 = depth_of(model, tok, p1, a1)
        # method-diverse derivation (construction mode)
        method = METHODS[idx % len(METHODS)]
        p2, a2 = generate(model, tok, method, f"What is {expr}?", max_new_tokens=320)
        v2 = parse_answer_line(a2)
        ok2 = (v2 == correct)
        d2, n2 = depth_of(model, tok, p2, a2)
        rows.append({"subset": subset, "expr": expr, "correct": correct,
                     "v1": v1, "ok1": ok1, "depth1": d1, "len1": n1,
                     "v2": v2, "ok2": ok2, "depth2": d2, "len2": n2})
        print(f"[{idx:2d}|{subset:9}] {expr:>14} = {correct:<9} | "
              f"1shot {str(v1):>9} {'OK ' if ok1 else 'BAD'} D={d1:5.2f} | "
              f"deriv {str(v2):>9} {'OK ' if ok2 else 'BAD'} D={d2:5.2f}")

    # ---- scoring ----
    confab = [r for r in rows if not r["ok1"]]          # one-shot confabulations
    # W1: paired depth one-shot-confab vs its derivation (same items)
    w1_pairs = [(r["depth1"], r["depth2"]) for r in confab
                if r["depth1"] == r["depth1"] and r["depth2"] == r["depth2"]]
    if len(w1_pairs) >= 3:
        d1s = [a for a, _ in w1_pairs]; d2s = [b for _, b in w1_pairs]
        diff = np.array(d1s) - np.array(d2s)            # confab - deriv (predicted >0)
        w1_t, w1_p = stats.ttest_rel(d1s, d2s)
        w1_d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))
        w1 = (abs(w1_d) >= 0.5) and (w1_p < 0.05)
        w1_sign = "confab_deeper" if np.mean(diff) > 0 else "confab_shallower"
    else:
        w1_d = w1_p = float("nan"); w1 = False; w1_sign = "insufficient_confab"

    # W2: depth separates correct vs confabulated answers (pool both conditions)
    cor_d, wr_d = [], []
    for r in rows:
        for ok, d in ((r["ok1"], r["depth1"]), (r["ok2"], r["depth2"])):
            if d == d:
                (cor_d if ok else wr_d).append(d)
    # predicted: confab deeper -> wrong has higher depth -> AUC(correct>wrong) low.
    w2_auc = auc(cor_d, wr_d)
    w2 = (w2_auc >= 0.70 or w2_auc <= 0.30) if w2_auc == w2_auc else False

    # W3: recovered items (one-shot wrong AND derivation correct) depth shift
    rec = [r for r in rows if (not r["ok1"]) and r["ok2"]
           and r["depth1"] == r["depth1"] and r["depth2"] == r["depth2"]]
    if len(rec) >= 3:
        r1 = [r["depth1"] for r in rec]; r2 = [r["depth2"] for r in rec]
        w3_t, w3_p = stats.ttest_rel(r1, r2)
        w3_shift = float(np.mean(np.array(r1) - np.array(r2)))
        sign_ok = (w3_shift > 0) == (w1_sign == "confab_deeper")
        w3 = (w3_p < 0.05) and sign_ok
    else:
        w3_p = float("nan"); w3_shift = float("nan"); w3 = False

    # K: length control + easy-item null
    dd = [(r["len2"] - r["len1"], r["depth2"] - r["depth1"]) for r in rows
          if r["depth1"] == r["depth1"] and r["depth2"] == r["depth2"]]
    if len(dd) >= 4:
        k_int, k_int_p = intercept_test([x for x, _ in dd], [y for _, y in dd])
    else:
        k_int = k_int_p = float("nan")
    ctrl = [r for r in rows if r["subset"] == "ctrl_3x2"
            and r["depth1"] == r["depth1"] and r["depth2"] == r["depth2"]]
    if len(ctrl) >= 3:
        _, k_ctrl_p = stats.ttest_rel([r["depth1"] for r in ctrl],
                                      [r["depth2"] for r in ctrl])
    else:
        k_ctrl_p = float("nan")
    k = (k_int_p == k_int_p and k_int_p < 0.05) and (k_ctrl_p != k_ctrl_p or k_ctrl_p > 0.05)

    n_confab_hard = sum(1 for r in rows if r["subset"] != "ctrl_3x2" and not r["ok1"])
    precondition = n_confab_hard >= 3

    # WM (secondary, PREREG_depth_within_mode_2026_05_29): does depth separate correct
    # from confabulated WITHIN a fixed mode? Bar pre-stated, greedy-deterministic data.
    deriv_cor = [r["depth2"] for r in rows if r["ok2"] and r["depth2"] == r["depth2"]]
    deriv_wr = [r["depth2"] for r in rows if not r["ok2"] and r["depth2"] == r["depth2"]]
    wm_deriv_auc = auc(deriv_cor, deriv_wr)
    os_cor = [r["depth1"] for r in rows if r["ok1"] and r["depth1"] == r["depth1"]]
    os_wr = [r["depth1"] for r in rows if not r["ok1"] and r["depth1"] == r["depth1"]]
    wm_os_auc = auc(os_cor, os_wr)
    wm_hypothesis = ("H_residual" if (wm_deriv_auc == wm_deriv_auc and
                     (wm_deriv_auc >= 0.70 or wm_deriv_auc <= 0.30)) else "H_mode")

    receipt = {
        "experiment": "white-box depth grounding — does attribution depth explain belief->truth?",
        "prereg": "papers/grounded-honesty-axis/PREREG_depth_grounding_whitebox.md",
        "answer_key_sha256_pre_scoring": key_hash,
        "model": MODEL_NAME, "device": DEVICE, "n_items": len(rows),
        "depth_metric": "SAE-free logit-lens DLA, attribution-weighted mean layer index",
        "core_signal": "exact integer parse vs in-code-computed arithmetic truth (no judge)",
        "precondition_competence_cliff": {
            "n_oneshot_confab_hard": n_confab_hard, "met": precondition},
        "n_oneshot_confab_total": len(confab),
        "W1_confab_depth_signature": {
            "n_pairs": len(w1_pairs), "cohens_d_paired": round(w1_d, 4) if w1_d == w1_d else None,
            "p": round(float(w1_p), 5) if w1_p == w1_p else None, "sign": w1_sign},
        "W2_depth_validity_auc": round(w2_auc, 4) if w2_auc == w2_auc else None,
        "W2_n_correct": len(cor_d), "W2_n_wrong": len(wr_d),
        "W3_recovered_depth_shift": {
            "n_recovered": len(rec),
            "shift_oneshot_minus_deriv": round(w3_shift, 4) if w3_shift == w3_shift else None,
            "p": round(float(w3_p), 5) if w3_p == w3_p else None},
        "K_length_intercept": round(k_int, 4) if k_int == k_int else None,
        "K_length_intercept_p": round(float(k_int_p), 5) if k_int_p == k_int_p else None,
        "K_ctrl_gap_p": round(float(k_ctrl_p), 5) if k_ctrl_p == k_ctrl_p else None,
        "WM_within_mode": {
            "prereg": "papers/grounded-honesty-axis/PREREG_depth_within_mode_2026_05_29.md",
            "derivation_auc_correct_vs_confab": round(wm_deriv_auc, 4) if wm_deriv_auc == wm_deriv_auc else None,
            "derivation_n_correct": len(deriv_cor), "derivation_n_wrong": len(deriv_wr),
            "oneshot_auc_correct_vs_confab": round(wm_os_auc, 4) if wm_os_auc == wm_os_auc else None,
            "oneshot_n_correct": len(os_cor), "oneshot_n_wrong": len(os_wr),
            "hypothesis": wm_hypothesis,
            "interpretation": ("depth carries within-mode truth signal" if wm_hypothesis == "H_residual"
                               else "depth is a MODE indicator, blind to correctness within mode")},
        "rows": rows,
        "W1": bool(w1), "W2": bool(w2), "W3": bool(w3), "K": bool(k),
        "RESULT": ("SURVIVED" if (w1 and w2 and w3 and k and precondition)
                   else ("UNINFORMATIVE_NO_CLIFF" if not precondition else "REPORT_AS_LANDED")),
        "honest_scope": (
            "single open model Qwen2.5-1.5B-Instruct, SAE-free logit-lens depth PROXY for "
            "the published SAE/IG metric, feasibility-grade, one confirmatory run; arithmetic "
            "ground truth computed in-code then hashed pre-scoring; exact-integer correctness "
            "(no judge). Next gates: canonical Gemma Scope SAE depth, second open model."),
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps(receipt, indent=2))
    print(f"\nprecond={precondition} W1={w1}({w1_sign}) W2={w2} W3={w3} K={k} -> {receipt['RESULT']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
