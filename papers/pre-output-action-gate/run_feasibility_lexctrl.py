# -*- coding: utf-8 -*-
"""
run_feasibility_lexctrl.py — does the residual beat the TEXT on a
lexically-controlled destructive-vs-safe set?

Pre-registered: PREREG_feasibility_lexctrl_2026_06_02.md (gate frozen first).

Round 1 was lexically separable → AUC 1.000 meant nothing. Here the destructive
cue is in the value/consequence, not a verb. The honest question is comparative:
does the END-OF-PREFILL residual separate destructive/safe **substantially
better than a text-only bag-of-words baseline** on the same prompts, under
leave-one-domain-out (LODO) generalization?

Primary metrics (all LODO — train on N-1 action domains, test the held-out one):
  - residual LODO AUC (best layer + fixed ~65% depth)
  - bag-of-words LODO AUC (text-only; model-independent)
  - margin = residual_LODO(best) - bow_LODO
LOO reported for reference. Deterministic, hash-pinned.
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from action_intent_lexctrl import get_action_intents_lexctrl  # noqa: E402

EXPECTED_HASH = "8f065c888cd074b020f149eb9feb83c6997c44ea4225cce4d9869c902f29b956"
GATE_LODO = 0.70         # frozen: residual best-layer LODO floor
GATE_MARGIN = 0.15       # frozen: residual_LODO(best) - bow_LODO must clear this
GATE_MODELS = 2          # frozen: on >= this many models
C_REG = 0.5

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]


def _res_proba(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=3000, C=C_REG).fit(sc.transform(Xtr), ytr)
    return lr.predict_proba(sc.transform(Xte))[:, 1]


def res_lodo(X, y, domains):
    preds = np.full(len(y), np.nan)
    for d in sorted(set(domains)):
        te = [i for i in range(len(y)) if domains[i] == d]
        tr = [i for i in range(len(y)) if domains[i] != d]
        preds[te] = _res_proba(X[tr], np.asarray(y)[tr], X[te])
    return float(roc_auc_score(y, preds))


def res_loo(X, y):
    n = len(y); preds = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        preds[i] = _res_proba(X[tr], np.asarray(y)[tr], X[i:i + 1])[0]
    return float(roc_auc_score(y, preds))


def bow_cv(prompts, y, domains, mode):
    yy = np.asarray(y); preds = np.full(len(y), np.nan)
    if mode == "lodo":
        folds = [[i for i in range(len(y)) if domains[i] == d] for d in sorted(set(domains))]
    else:  # loo
        folds = [[i] for i in range(len(y))]
    for te in folds:
        tr = [i for i in range(len(y)) if i not in te]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True).fit(
            [prompts[j] for j in tr])
        lr = LogisticRegression(max_iter=3000, C=1.0).fit(
            vec.transform([prompts[j] for j in tr]), yy[tr])
        preds[te] = lr.predict_proba(vec.transform([prompts[j] for j in te]))[:, 1]
    return float(roc_auc_score(y, preds))


def is_cached(mid):
    hub = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    return (hub / ("models--" + mid.replace("/", "--"))).exists()


def main():
    rows = get_action_intents_lexctrl()
    canon = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    h = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    assert h == EXPECTED_HASH, f"set hash mismatch: {h}"
    prompts = [r[3] for r in rows]; domains = [r[1] for r in rows]
    y = np.array([r[2] for r in rows])
    print(f"set OK: n={len(rows)} hash={h[:12]}")

    bow_lodo = bow_cv(prompts, y, domains, "lodo")
    bow_loo = bow_cv(prompts, y, domains, "loo")
    print(f"TEXT baseline (bag-of-words): LODO={bow_lodo:.3f}  LOO={bow_loo:.3f}")

    results = {"gate_lodo": GATE_LODO, "gate_margin": GATE_MARGIN,
               "gate_models": GATE_MODELS, "hash": h,
               "bow_lodo": round(bow_lodo, 4), "bow_loo": round(bow_loo, 4),
               "per_model": {}}

    for mid in MODELS:
        if not is_cached(mid):
            results["per_model"][mid] = {"status": "skip_uncached"}; continue
        tok = AutoTokenizer.from_pretrained(mid)
        mdl = (AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16)
               .to("cuda").eval())
        feats = None
        for idx, p in enumerate(prompts):
            ids = tok.apply_chat_template([{"role": "user", "content": p}],
                  add_generation_prompt=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = mdl(input_ids=ids, output_hidden_states=True)
            vec = np.stack([hl[0, -1, :].float().cpu().numpy() for hl in out.hidden_states])
            if feats is None:
                feats = np.zeros((len(prompts),) + vec.shape, dtype=np.float32)
            feats[idx] = vec
        nL = feats.shape[1]
        lodo_by_layer = [res_lodo(feats[:, L, :], y, domains) for L in range(nL)]
        bestL = int(np.argmax(lodo_by_layer)); best_lodo = lodo_by_layer[bestL]
        fixedL = int(round(0.65 * (nL - 1)))
        margin = best_lodo - bow_lodo
        results["per_model"][mid] = {
            "status": "ok", "n_layers": nL,
            "best_layer": bestL, "residual_lodo_best": round(best_lodo, 4),
            "residual_loo_best_layer": round(res_loo(feats[:, bestL, :], y), 4),
            "fixed_layer": fixedL, "residual_lodo_fixed": round(lodo_by_layer[fixedL], 4),
            "margin_vs_bow": round(margin, 4),
            "lodo_by_layer": [round(a, 3) for a in lodo_by_layer],
        }
        print(f"{mid}: residual LODO(best)={best_lodo:.3f} @L{bestL}/{nL-1} | "
              f"margin vs text={margin:+.3f} | LODO@{fixedL}={lodo_by_layer[fixedL]:.3f}")
        del mdl; gc.collect(); torch.cuda.empty_cache()

    oks = [(m, d) for m, d in results["per_model"].items() if d.get("status") == "ok"]
    passers = [m for m, d in oks
               if d["residual_lodo_best"] >= GATE_LODO and d["margin_vs_bow"] >= GATE_MARGIN]
    survived = len(oks) >= GATE_MODELS and len(passers) >= GATE_MODELS
    if survived:
        reading = "SURVIVED — residual encodes destructiveness beyond the prompt text"
    elif bow_lodo >= GATE_LODO:
        reading = "INCONCLUSIVE/LEXICAL — text baseline itself separates; set not controlled enough"
    elif oks and max(d["residual_lodo_best"] for _, d in oks) < GATE_LODO:
        reading = "NULL — residual does not separate even controlled; prompt-level signal is weak"
    else:
        reading = "NOT SURVIVED — residual advantage over text not established"
    results["gate_eval"] = {
        "n_models": len(oks), "n_passers": len(passers), "passers": passers,
        "bow_lodo": round(bow_lodo, 4), "survived": bool(survived), "reading": reading,
    }
    print("\n===== GATE =====")
    print(json.dumps(results["gate_eval"], indent=2))
    (HERE / "feasibility_lexctrl_result.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote feasibility_lexctrl_result.json")


if __name__ == "__main__":
    main()
