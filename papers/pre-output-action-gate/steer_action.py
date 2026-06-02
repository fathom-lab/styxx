# -*- coding: utf-8 -*-
"""
steer_action.py — CAUSAL test: can we flip a destructive choice by patching the
mid-layer destructive direction? (Exploratory; random-direction + output-validity
controls are the anti-fooling safeguards.)

The emitted-action study showed the pre-emission residual *predicts* the model's
destructive choice, peaking mid-stream. This asks whether that mid-layer
representation is *causal*: subtract the destructive direction at its best layer
during generation and see if the emitted choice flips destructive -> safe —
WITHOUT breaking the output (a flip into gibberish is not a flip).

Per model:
  1. Re-extract residuals at the best layer L (from emitted_action_result.json)
     for all 40 scenarios; record each baseline choice.
  2. Fit a logistic-regression destructive DIRECTION d at L (unit-normed).
  3. For each scenario the model chose DESTRUCTIVELY, regenerate with a forward
     hook at L: h := h - alpha * d (all positions). alpha swept in units of the
     destructive-coordinate std. Record: flipped-to-safe? still a valid tool?
  4. CONTROL: identical, with a random unit direction (should not flip).

Reports, per (model, alpha): real flip rate, random flip rate, validity rate.
"""
from __future__ import annotations

import gc, json, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from action_scenarios import get_scenarios          # noqa: E402
from run_emitted_action import _prefill, _parse_choice  # noqa: E402

MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"]
ALPHA_K = [2.0, 4.0, 8.0, 16.0]   # multiples of the destructive-coordinate std
SEED = 0


def _layers(model):
    for root in (getattr(model, "model", None), model):
        ls = getattr(root, "layers", None)
        if ls is not None:
            return ls
    raise RuntimeError("no .layers")


def main():
    res = json.load(open(HERE / "emitted_action_result.json", encoding="utf-8"))
    rows = get_scenarios()
    rng = np.random.default_rng(SEED)
    out_all = {}

    for mid in MODELS:
        info = res["per_model"].get(mid, {})
        if info.get("status") != "ok" or not info.get("balanced"):
            continue
        L = info["best_layer"]
        tok = AutoTokenizer.from_pretrained(mid)
        mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to("cuda").eval()
        eos = tok.eos_token_id; eos = eos[0] if isinstance(eos, list) else eos
        pad = tok.pad_token_id if tok.pad_token_id is not None else eos

        # 1+2: residuals at L + baseline choices + fit direction
        X, y, base_choice, prefills, names_list, dtools = [], [], [], [], [], []
        for sid, cat, task, tools, dtool in rows:
            names = {n for n, _ in tools}
            ids = _prefill(tok, task, tools)
            with torch.no_grad():
                o = mdl(input_ids=ids, output_hidden_states=True)
            X.append(o.hidden_states[L][0, -1, :].float().cpu().numpy())
            with torch.no_grad():
                g = mdl.generate(ids, attention_mask=torch.ones_like(ids),
                                 max_new_tokens=40, do_sample=False, pad_token_id=pad)
            ch = _parse_choice(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True), names)
            y.append(1 if ch == dtool else 0)
            base_choice.append(ch); prefills.append(ids); names_list.append(names); dtools.append(dtool)
        X = np.array(X); y = np.array(y)
        sc = StandardScaler().fit(X)
        lr = LogisticRegression(max_iter=3000, C=0.5).fit(sc.transform(X), y)
        # direction in raw residual space (undo standardization scaling on coef)
        d = lr.coef_[0] / sc.scale_
        d = d / (np.linalg.norm(d) + 1e-9)
        proj = X @ d
        std = float(proj.std())
        d_t = torch.tensor(d, device="cuda", dtype=torch.float16)
        rnd = rng.standard_normal(d.shape); rnd = rnd / np.linalg.norm(rnd)
        rnd_t = torch.tensor(rnd, device="cuda", dtype=torch.float16)

        dest_idx = [i for i in range(len(rows)) if y[i] == 1]
        layer_mod = _layers(mdl)[L]
        print(f"\n=== {mid}  layer {L}  destructive baseline n={len(dest_idx)}  coord-std={std:.2f} ===")

        def run_with(vec, alpha):
            flips = valid = 0
            handle = layer_mod.register_forward_hook(
                lambda m, i, o: ((o[0] - alpha * vec,) + tuple(o[1:])) if isinstance(o, tuple)
                else (o - alpha * vec))
            try:
                for i in dest_idx:
                    ids = prefills[i]
                    with torch.no_grad():
                        g = mdl.generate(ids, attention_mask=torch.ones_like(ids),
                                         max_new_tokens=40, do_sample=False, pad_token_id=pad)
                    ch = _parse_choice(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True), names_list[i])
                    if ch in names_list[i]:
                        valid += 1
                        if ch != dtools[i]:
                            flips += 1
            finally:
                handle.remove()
            n = len(dest_idx)
            return flips / n, valid / n

        rec = {"layer": L, "n_destructive": len(dest_idx), "coord_std": round(std, 3), "alphas": {}}
        for k in ALPHA_K:
            alpha = k * std
            rf, rv = run_with(d_t, alpha)
            cf, cv = run_with(rnd_t, alpha)
            rec["alphas"][f"k{k}"] = {"real_flip": round(rf, 3), "real_valid": round(rv, 3),
                                      "rand_flip": round(cf, 3), "rand_valid": round(cv, 3)}
            print(f"  k={k:<4} a={alpha:6.1f} | REAL flip {rf:.2f} valid {rv:.2f} | "
                  f"RAND flip {cf:.2f} valid {cv:.2f}")
        out_all[mid] = rec
        del mdl; gc.collect(); torch.cuda.empty_cache()

    (HERE / "steer_result.json").write_text(json.dumps(out_all, indent=2), encoding="utf-8")
    print("\nwrote steer_result.json")


if __name__ == "__main__":
    main()
