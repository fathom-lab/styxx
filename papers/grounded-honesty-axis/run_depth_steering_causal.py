"""Causal test: can a construction-ward residual-stream push flip a confabulation?
PREREG_depth_steering_causal_2026_05_29.md (Pearl Level 3).

Derives a construction-vs-retrieval direction v_j at each decoder layer j as the
difference of mean answer-token residuals (method-diverse derivation minus one-shot),
across all items. Then, during a one-shot generation, a forward hook adds alpha * rms *
v_j to layer j's output and we test whether confabulating one-shot answers flip CORRECT
— beyond a norm-matched sham direction. Hyperparameters (j*, alpha*) are locked on a
TRAIN split and evaluated once on a held-out TEST split. Ground truth computed in-code,
hashed pre-scoring; exact integer match (no judge).

Usage:
    python papers/grounded-honesty-axis/run_depth_steering_causal.py --n 6   # pilot
    python papers/grounded-honesty-axis/run_depth_steering_causal.py
"""
from __future__ import annotations

import argparse
import contextlib
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
    METHODS, MODEL_NAME, DEVICE, parse_answer_line,
)

RECEIPT = HERE / "depth_steering_causal_result.json"
GRID_LAYERS = [6, 12, 18]
GRID_ALPHAS = [4.0, 8.0, 12.0]


def _decoder_layers(model):
    return model.model.layers


@torch.no_grad()
def _gen(model, tok, system, user, max_new_tokens, hook_layer=None, vec=None, alpha=0.0):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(DEVICE)
    handle = None
    if hook_layer is not None and vec is not None and alpha:
        v = vec.to(DEVICE).float()
        v = v / (v.norm() + 1e-9)

        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            rms = h.float().pow(2).mean(-1, keepdim=True).sqrt()      # (B,T,1)
            h = h + (alpha * rms * v).to(h.dtype)
            if isinstance(out, tuple):
                return (h,) + tuple(out[1:])
            return h
        handle = _decoder_layers(model)[hook_layer].register_forward_hook(hook)
    try:
        gen = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    finally:
        if handle is not None:
            handle.remove()
    return text, tok.decode(gen[0, ids.input_ids.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def _answer_resids(model, tok, prompt_text, answer_text):
    """Mean residual over answer-token positions, per hidden-state layer. (L+1, d)."""
    plen = tok(prompt_text, return_tensors="pt").input_ids.shape[1]
    fids = tok(prompt_text + answer_text, return_tensors="pt").input_ids.to(DEVICE)
    if fids.shape[1] <= plen:
        return None
    out = model(fids, output_hidden_states=True)
    hs = torch.stack(out.hidden_states, dim=0)[:, 0, :, :]          # (L+1, T, d)
    return hs[:, plen:, :].mean(1).float().cpu()                    # (L+1, d)


@torch.no_grad()
def _depth(model, tok, prompt_text, answer_text, hook_layer=None, vec=None, alpha=0.0):
    plen = tok(prompt_text, return_tensors="pt").input_ids.shape[1]
    fids = tok(prompt_text + answer_text, return_tensors="pt").input_ids.to(DEVICE)
    T = fids.shape[1]
    if T <= plen:
        return float("nan")
    handle = None
    if hook_layer is not None and vec is not None and alpha:
        v = vec.to(DEVICE).float(); v = v / (v.norm() + 1e-9)

        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            rms = h.float().pow(2).mean(-1, keepdim=True).sqrt()
            h = h + (alpha * rms * v).to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        handle = _decoder_layers(model)[hook_layer].register_forward_hook(hook)
    try:
        out = model(fids, output_hidden_states=True)
    finally:
        if handle is not None:
            handle.remove()
    hs = torch.stack(out.hidden_states, dim=0)[:, 0, :, :]
    L = hs.shape[0]
    normed = model.model.norm(hs)
    W = model.lm_head.weight
    ans_pos = torch.arange(plen - 1, T - 1, device=DEVICE)
    tgt = fids[0, plen:T]
    sel = normed[:, ans_pos, :]
    logits = torch.einsum("lad,ad->la", sel.float(), W[tgt].float())
    a = (logits[1:] - logits[:-1]).clamp_min(0.0)
    lidx = torch.arange(1, L, device=DEVICE, dtype=a.dtype).unsqueeze(1)
    s = a.sum(0)
    d = (lidx * a).sum(0) / s.clamp_min(1e-9)
    valid = s > 1e-9
    return float(d[valid].mean().item()) if valid.sum() else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=len(SPECS))
    args = ap.parse_args(argv)

    items = [(form, _eval(form), _expr(form), subset, i)
             for i, (form, delta, subset) in enumerate(SPECS[: args.n])]
    key_hash = hashlib.sha256(
        json.dumps([(e, c) for _, c, e, _, _ in items], ensure_ascii=False).encode()
    ).hexdigest()
    print(f"answer-key SHA-256 (pre-scoring): {key_hash}")
    print(f"model={MODEL_NAME} device={DEVICE} items={len(items)}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16).to(DEVICE).eval()
    nlayers = len(_decoder_layers(model))
    print(f"model loaded, {nlayers} decoder layers\n")

    # ---- phase 1: paired generations + answer residuals (defines the axis) ----
    rows = []
    resid_sum = None; n_resid = 0
    for form, correct, expr, subset, idx in items:
        p1, a1 = _gen(model, tok, "Answer with only the final number, nothing else.",
                      f"What is {expr}?", 16)
        v1 = parse_int(a1); ok1 = (v1 == correct)
        method = METHODS[idx % len(METHODS)]
        p2, a2 = _gen(model, tok, method, f"What is {expr}?", 320)
        v2 = parse_answer_line(a2); ok2 = (v2 == correct)
        r1 = _answer_resids(model, tok, p1, a1)
        r2 = _answer_resids(model, tok, p2, a2)
        if r1 is not None and r2 is not None:
            diff = (r2 - r1)                                         # deriv - oneshot
            resid_sum = diff if resid_sum is None else resid_sum + diff
            n_resid += 1
        rows.append({"subset": subset, "expr": expr, "correct": correct,
                     "p1": p1, "a1": a1, "v1": v1, "ok1": ok1, "idx": idx})
        print(f"[{idx:2d}|{subset:9}] {expr:>14}={correct:<8} 1shot={str(v1):>9} "
              f"{'OK' if ok1 else 'BAD'}  deriv={str(v2):>9} {'OK' if ok2 else 'BAD'}")

    directions = (resid_sum / max(1, n_resid))                      # (L+1, d) hidden idx
    # construction-ward = +(deriv - oneshot); hook on layer j uses hidden index j+1
    rng = np.random.default_rng(0)
    sham_vecs = {j: torch.tensor(rng.standard_normal(directions.shape[1]),
                                 dtype=torch.float32) for j in GRID_LAYERS}

    confab = [r for r in rows if not r["ok1"]]
    train = [r for r in confab if r["idx"] % 2 == 0]
    test = [r for r in confab if r["idx"] % 2 == 1]
    print(f"\nconfab one-shot: {len(confab)}  train={len(train)} test={len(test)}")

    def steer_flip(r, j, alpha, vec):
        _, a = _gen(model, tok, "Answer with only the final number, nothing else.",
                    f"What is {r['expr']}?", 16, hook_layer=j, vec=vec, alpha=alpha)
        val = parse_int(a)
        return (val == r["correct"]), (val is not None)

    # ---- phase 2: lock (j*, alpha*) on TRAIN by wrong->correct flips ----
    best = None
    grid = []
    for j in GRID_LAYERS:
        vj = directions[j + 1]
        for alpha in GRID_ALPHAS:
            flips = valids = 0
            for r in train:
                f, ok = steer_flip(r, j, alpha, vj)
                flips += int(f); valids += int(ok)
            fr = flips / max(1, len(train)); vr = valids / max(1, len(train))
            grid.append({"layer": j, "alpha": alpha, "train_flip": fr, "train_valid": vr})
            print(f"  grid j={j:2d} a={alpha:4.1f}  flip={fr:.3f} valid={vr:.3f}")
            cand = (fr, vr, -alpha)
            if vr >= 0.80 and (best is None or cand > best[0]):
                best = (cand, j, alpha)
    if best is None:  # no valid operating point; take max flip ignoring validity
        g = max(grid, key=lambda x: (x["train_flip"], x["train_valid"]))
        best = ((g["train_flip"], g["train_valid"], -g["alpha"]), g["layer"], g["alpha"])
    _, jstar, astar = best
    print(f"\nLOCKED j*={jstar} alpha*={astar}")

    # ---- phase 3: evaluate ONCE on TEST (real vs sham), S3 depth, K break ----
    vstar = directions[jstar + 1]
    real_flip = real_valid = sham_flip = sham_valid = 0
    for r in test:
        f, ok = steer_flip(r, jstar, astar, vstar); real_flip += f; real_valid += ok
        f, ok = steer_flip(r, jstar, astar, sham_vecs[jstar]); sham_flip += f; sham_valid += ok
    nt = max(1, len(test))
    s1_real = real_flip / nt; s1_sham = sham_flip / nt
    s2_valid = real_valid / nt

    # S3 manipulation check: depth of the one-shot answer hook-on vs hook-off (test items)
    d_on, d_off = [], []
    for r in test:
        do = _depth(model, tok, r["p1"], r["a1"])
        dn = _depth(model, tok, r["p1"], r["a1"], hook_layer=jstar, vec=vstar, alpha=astar)
        if do == do and dn == dn:
            d_off.append(do); d_on.append(dn)
    if len(d_on) >= 3:
        s3_t, s3_p = stats.ttest_rel(d_on, d_off)
        s3_shift = float(np.mean(np.array(d_on) - np.array(d_off)))
    else:
        s3_p = float("nan"); s3_shift = float("nan")

    # K: break-rate on initially-correct one-shot items, real vs sham
    correct_os = [r for r in rows if r["ok1"]]
    rb = sb = 0
    for r in correct_os:
        f, _ = steer_flip(r, jstar, astar, vstar); rb += (not f)
        f, _ = steer_flip(r, jstar, astar, sham_vecs[jstar]); sb += (not f)
    nc = max(1, len(correct_os))
    k_real_break = rb / nc; k_sham_break = sb / nc

    precondition = len(test) >= 8
    s1 = (s1_real >= 0.15) and ((s1_real - s1_sham) >= 0.10)
    s2 = s2_valid >= 0.80
    s3 = (s3_p == s3_p and s3_p < 0.05 and s3_shift < 0)
    k = (k_real_break - k_sham_break) <= 0.10

    receipt = {
        "experiment": "causal depth steering — can a construction-ward push flip a confabulation?",
        "prereg": "papers/grounded-honesty-axis/PREREG_depth_steering_causal_2026_05_29.md",
        "answer_key_sha256_pre_scoring": key_hash,
        "model": MODEL_NAME, "device": DEVICE, "n_items": len(rows),
        "n_decoder_layers": nlayers,
        "core_signal": "exact integer parse vs in-code arithmetic truth (no judge)",
        "n_confab_oneshot": len(confab), "n_train": len(train), "n_test": len(test),
        "precondition_enough_test_confab": {"n_test": len(test), "met": precondition},
        "locked_layer": jstar, "locked_alpha": astar, "grid": grid,
        "S1_causal_recovery": {
            "real_flip_rate": round(s1_real, 4), "sham_flip_rate": round(s1_sham, 4),
            "real_minus_sham": round(s1_real - s1_sham, 4)},
        "S2_valid_integer_rate": round(s2_valid, 4),
        "S3_manipulation_check": {
            "depth_shift_on_minus_off": round(s3_shift, 4) if s3_shift == s3_shift else None,
            "p": round(float(s3_p), 5) if s3_p == s3_p else None},
        "K_break_rate": {"real": round(k_real_break, 4), "sham": round(k_sham_break, 4)},
        "S1": bool(s1), "S2": bool(s2), "S3": bool(s3), "K": bool(k),
        "RESULT": ("SURVIVED" if (s1 and s2 and s3 and k and precondition)
                   else ("UNINFORMATIVE_FEW_TEST" if not precondition else "REPORT_AS_LANDED")),
        "honest_scope": (
            "single open model Qwen2.5-1.5B-Instruct, SAE-free logit-lens depth, "
            "difference-of-means residual steering with sham control + train/test-locked "
            "hyperparameters, feasibility-grade, one confirmatory run; arithmetic ground "
            "truth computed in-code then hashed; exact-integer correctness (no judge). "
            "Tests causal sufficiency of a LINEAR residual push along one derived "
            "direction only; a null does not rule out causality via other interventions."),
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in receipt.items() if k != "grid"}, indent=2))
    print(f"\nprecond={precondition} S1={s1}(real {s1_real:.2f} vs sham {s1_sham:.2f}) "
          f"S2={s2} S3={s3}({s3_shift:.2f}) K={k} -> {receipt['RESULT']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
