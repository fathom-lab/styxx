# -*- coding: utf-8 -*-
"""Bug-fix validation: re-run the v0 5-fold CV on the EXACT cached 1200-response
training corpus, comparing substring matching (reproduces published 0.9720)
against word-boundary matching. Zero API cost — pure offline recompute on
benchmarks/data/sycophancy/responses_v0.jsonl. Deterministic (seed=0)."""
from __future__ import annotations
import json, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"papers"/"sycophancy-target-gate"))
from scripts.sycophancy_train_v0 import train_full, RESPONSES_CACHE
from styxx.guardrail.sycophancy_signals import extract_sycophancy_features  # substring
from target_gate import extract_features_wb                                # word-boundary

rows = [json.loads(l) for l in open(RESPONSES_CACHE, encoding="utf-8") if l.strip()]
print(f"loaded {len(rows)} cached responses")
names = list(extract_sycophancy_features("x", "x").keys())

def featurize(extractor):
    X = np.zeros((len(rows), len(names))); y = np.zeros(len(rows), dtype=int)
    for i, r in enumerate(rows):
        f = extractor(r["question"], r["response"])
        X[i] = [f[n] for n in names]; y[i] = int(r["label_sycophantic"])
    return X, y

for label, extractor in (("substring (published)", extract_sycophancy_features),
                         ("word-boundary (fix)", extract_features_wb)):
    print(f"\n=== {label} ===")
    X, y = featurize(extractor)
    fit = train_full(X, y, names, seed=0)
    print(f"  >> mean 5-fold CV AUC = {fit['mean_auc']:.4f} (std {fit['std_auc']:.4f})")
