# -*- coding: utf-8 -*-
"""Generate v0.2 (word-boundary) sycophancy weights from the EXACT cached 1200-
response corpus used for v0 — deterministic (seed=0), zero API cost. v0 itself is
left byte-identical; this only produces the successor's numbers. Emits a
paste-ready JSON identical in shape to benchmarks/sycophancy_weights_v0.json."""
from __future__ import annotations
import json, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"papers"/"sycophancy-target-gate"))
from scripts.sycophancy_train_v0 import (
    train_full, feature_ablation, find_critical_k, RESPONSES_CACHE,
)
from styxx.guardrail.sycophancy_signals import extract_sycophancy_features
from target_gate import extract_features_wb

rows = [json.loads(l) for l in open(RESPONSES_CACHE, encoding="utf-8") if l.strip()]
names = list(extract_sycophancy_features("x", "x").keys())

def featurize(extractor):
    X = np.zeros((len(rows), len(names))); y = np.zeros(len(rows), dtype=int)
    for i, r in enumerate(rows):
        f = extractor(r["question"], r["response"])
        X[i] = [f[n] for n in names]; y[i] = int(r["label_sycophantic"])
    return X, y

X, y = featurize(extract_features_wb)
print(f"n={len(rows)}  word-boundary featurization")
fit = train_full(X, y, names, seed=0)
hist = feature_ablation(X, y, names, seed=0)
crit = find_critical_k(hist)
out = {
    **fit,
    "critical_K": crit["critical_K"], "critical_feature": crit["critical_feature"],
    "delta_auc_at_K": crit["delta_auc_at_K"], "auc_at_K": crit["auc_at_K"],
    "training_model": "gpt-4o-mini", "n_samples": int(y.shape[0]),
    "matching": "word-boundary", "supersedes": "v0 (substring)", "seed": 0,
}
dest = ROOT/"benchmarks"/"sycophancy_weights_v0_2.json"
dest.write_text(json.dumps(out, indent=2))
print(f"\nmean_auc={fit['mean_auc']:.4f}  critical_K={crit['critical_K']} "
      f"({crit['critical_feature']})  -> {dest.name}")
print("COEFS:", [round(c,6) for c in fit["coefs"]])
print("INTERCEPT:", round(fit["intercept"],6))
