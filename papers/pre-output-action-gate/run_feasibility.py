# -*- coding: utf-8 -*-
"""
run_feasibility.py — does the pre-emission residual encode destructive-vs-safe?

Pre-registered: papers/pre-output-action-gate/PREREG_feasibility_2026_06_02.md
Method + kill-gate frozen there before any data.

For each cached open-weight instruct model: read the END-OF-PREFILL residual
(last prefill token, every layer) for each of the 48 controlled action-intent
prompts — NO token is generated. Then:
  - LOO AUC per layer (leave-one-out): does the signal exist?
  - LODO AUC (leave-one-domain-out) at the LOO-best layer and a fixed ~65%
    depth: is the signal abstract, or just per-domain verb memorization?

Deterministic, inputs hash-pinned. Writes feasibility_result.json.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from action_intent_set import get_action_intents  # noqa: E402

EXPECTED_HASH = "3266ecc75b3e3d402a2fec20fbfcbb283e06215b559f172e4d1add527d40bab7"
GATE_AUC = 0.80          # frozen: feasibility bar on best-layer LOO AUC
GATE_MODELS = 2          # frozen: must clear on >= this many models
LODO_STRONG = 0.70       # frozen: LODO bar that upgrades the reading to "abstract"
C_REG = 0.5              # L2 logistic regression strength (fixed)

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]


def _fit_auc(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=3000, C=C_REG).fit(sc.transform(Xtr), ytr)
    return lr.predict_proba(sc.transform(Xte))[:, 1]


def loo_auc(X, y):
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        preds[i] = _fit_auc(X[tr], y[tr], X[i:i + 1])[0]
    return float(roc_auc_score(y, preds))


def lodo_auc(X, y, domains):
    preds = np.full(len(y), np.nan)
    for d in sorted(set(domains)):
        te = [i for i in range(len(y)) if domains[i] == d]
        tr = [i for i in range(len(y)) if domains[i] != d]
        preds[te] = _fit_auc(X[tr], np.asarray(y)[tr], X[te])
    return float(roc_auc_score(y, preds))


def is_cached(mid):
    hub = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    return (hub / ("models--" + mid.replace("/", "--"))).exists()


def main():
    rows = get_action_intents()
    canon = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    h = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    assert h == EXPECTED_HASH, f"set hash mismatch: {h}"
    prompts = [r[3] for r in rows]
    domains = [r[1] for r in rows]
    y = np.array([r[2] for r in rows])
    print(f"set OK: n={len(rows)} hash={h[:12]}")

    results = {"gate_auc": GATE_AUC, "gate_models": GATE_MODELS,
               "lodo_strong": LODO_STRONG, "hash": h, "per_model": {}}

    for mid in MODELS:
        if not is_cached(mid):
            results["per_model"][mid] = {"status": "skip_uncached"}
            print(f"SKIP {mid}")
            continue
        tok = AutoTokenizer.from_pretrained(mid)
        mdl = (AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16)
               .to("cuda").eval())
        feats = None
        for idx, p in enumerate(prompts):
            ids = tok.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = mdl(input_ids=ids, output_hidden_states=True)
            vec = np.stack([hl[0, -1, :].float().cpu().numpy()
                            for hl in out.hidden_states])      # [n_layers+1, hidden]
            if feats is None:
                feats = np.zeros((len(prompts),) + vec.shape, dtype=np.float32)
            feats[idx] = vec
        n_layers = feats.shape[1]
        per_layer = [loo_auc(feats[:, L, :], y) for L in range(n_layers)]
        best_L = int(np.argmax(per_layer))
        best = per_layer[best_L]
        fixed_L = int(round(0.65 * (n_layers - 1)))
        lodo_best = lodo_auc(feats[:, best_L, :], y, domains)
        lodo_fixed = lodo_auc(feats[:, fixed_L, :], y, domains)
        results["per_model"][mid] = {
            "status": "ok", "n_layers": n_layers,
            "best_layer": best_L, "best_loo_auc": round(best, 4),
            "lodo_auc_best_layer": round(lodo_best, 4),
            "fixed_layer": fixed_L, "lodo_auc_fixed_layer": round(lodo_fixed, 4),
            "loo_auc_per_layer": [round(a, 3) for a in per_layer],
        }
        print(f"{mid}: best LOO AUC={best:.3f} @L{best_L}/{n_layers-1} | "
              f"LODO@best={lodo_best:.3f} | LODO@{fixed_L}={lodo_fixed:.3f}")
        del mdl; gc.collect(); torch.cuda.empty_cache()

    oks = [(m, d) for m, d in results["per_model"].items() if d.get("status") == "ok"]
    n_pass = sum(1 for _, d in oks if d["best_loo_auc"] >= GATE_AUC)
    lodos = [d["lodo_auc_best_layer"] for _, d in oks if d["best_loo_auc"] >= GATE_AUC]
    median_lodo = float(np.median(lodos)) if lodos else None
    survived = (len(oks) >= GATE_MODELS and n_pass >= GATE_MODELS)
    if not survived:
        reading = "NOT SURVIVED"
    elif median_lodo is not None and median_lodo >= LODO_STRONG:
        reading = "SURVIVED (signal exists AND generalizes across action domains)"
    else:
        reading = "SURVIVED-WEAK (signal exists; cross-domain generalization unconfirmed — may be partly lexical)"
    results["gate_eval"] = {
        "n_models": len(oks), "n_pass_ge_gate": n_pass,
        "median_lodo_of_passers": median_lodo,
        "survived": bool(survived), "reading": reading,
    }
    print("\n===== GATE =====")
    print(json.dumps(results["gate_eval"], indent=2))
    (HERE / "feasibility_result.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote feasibility_result.json")


if __name__ == "__main__":
    main()
