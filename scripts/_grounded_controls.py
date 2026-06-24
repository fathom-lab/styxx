"""Adversarial controls for the grounded>text length result (is the 0.99 activation probe real or an artifact?).

Three guards, because a linear probe on 3072 dims / n~122 can hit 0.99 on noise:
  1. LABEL-SHUFFLE: shuffle y, re-run CV at the selected layer -> must collapse to ~0.5 (else pipeline bug / leak).
  2. QUESTION-GROUPED CV: never split a question's (calibrated, overconfident) pair across train/test -> kills
     content/pair leakage; the honest AUC.
  3. PCA-50: project activations to 50 dims before the probe -> if the signal is real+low-rank it survives,
     if it was high-dim overfitting it drops.
All on the length-matched (CEM) set AND full, both models. Run: python scripts/_grounded_controls.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from suite_causal_length import cem_match
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

V0 = ROOT / "benchmarks" / "data" / "overconfidence" / "pairs_v0.jsonl"
ACTDIR = ROOT / "benchmarks" / "data" / "overconfidence"
REPOS = ["llama", "qwen"]
def actpath(tag): return ACTDIR / f"_acts_{tag}.npz"
def load(p): return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def cv_strat(X, y, seed=0):
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        s = StandardScaler().fit(X[tr]); c = LogisticRegression(max_iter=2000).fit(s.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], c.predict_proba(s.transform(X[te]))[:, 1]))
    return float(np.mean(aucs))


def cv_grouped(X, y, groups):
    aucs = []
    for tr, te in GroupKFold(5).split(X, y, groups):
        if len(np.unique(y[te])) < 2: continue
        s = StandardScaler().fit(X[tr]); c = LogisticRegression(max_iter=2000).fit(s.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], c.predict_proba(s.transform(X[te]))[:, 1]))
    return float(np.mean(aucs))


def sel_layer(acts, y):
    return int(np.argmax([cv_strat(acts[:, L, :], y) for L in range(acts.shape[1])]))


rows = load(V0)
qids = {q: i for i, q in enumerate(dict.fromkeys(r["question"] for r in rows))}
groups_all = np.array([qids[r["question"]] for r in rows])

out = []
for tag in REPOS:
    if not actpath(tag).exists():
        print(f"[skip] {tag} (no acts yet)"); continue
    d = np.load(actpath(tag)); acts, y, wc = d["acts"], d["y"], d["wc"]
    L = sel_layer(acts, y); A = acts[:, L, :]
    idx = cem_match(wc.astype(float), y, binw=8); Ac, yc, gc = A[idx], y[idx], groups_all[idx]
    rng = np.random.default_rng(0); ysh = rng.permutation(yc)
    pca = PCA(n_components=50, random_state=0)
    res = {
        "model": tag, "layer": L, "n_cem": int(len(idx)),
        "act_cem_strat": cv_strat(Ac, yc),
        "act_cem_SHUFFLE": cv_strat(Ac, ysh),                       # expect ~0.5
        "act_cem_GROUPED": cv_grouped(Ac, yc, gc),                  # anti-leak honest AUC
        "act_cem_PCA50": cv_strat(pca.fit_transform(StandardScaler().fit_transform(Ac)), yc),
        "act_full_GROUPED": cv_grouped(A, y, groups_all),
    }
    out.append(res)
    print(f"\n=== {tag} (layer {L}, n_cem {res['n_cem']}) ===")
    print(f"  act_cem stratified-CV : {res['act_cem_strat']:.3f}")
    print(f"  act_cem SHUFFLE       : {res['act_cem_SHUFFLE']:.3f}   (must be ~0.5)")
    print(f"  act_cem QUESTION-GROUP: {res['act_cem_GROUPED']:.3f}   (anti-leak honest AUC)")
    print(f"  act_cem PCA-50        : {res['act_cem_PCA50']:.3f}   (low-rank robustness)")
    print(f"  act_full QUESTION-GROUP: {res['act_full_GROUPED']:.3f}")

if out:
    (ACTDIR / "_grounded_controls_result.json").write_text(json.dumps(out, indent=2))
    print("\nVERDICT GUARDS: shuffle~0.5 AND grouped>=0.72 AND pca50 holds => the grounded win is REAL, not overfit/leak.")
    print("wrote benchmarks/data/overconfidence/_grounded_controls_result.json")
