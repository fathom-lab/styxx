"""Coarsened-exact length-matching (CEM) robustness check — NO generation, NO generator change.

Addresses the generator-validity objection head-on: subsample each corpus so the honest/dishonest word-count
distributions are identical (within bins), then re-measure the SHIPPED deception-v0 AUC. If the AUC collapses
under length-matching on the instrument's OWN original gpt corpus, length is the cause — with zero generator
confound. Run on original (gpt), Qwen, and Llama corpora.

Run: python scripts/_lenmatch_cem.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from deception_train_v0 import featurize
from length_control_causal import shipped_logit, boot_ci, cv_oof
from sklearn.metrics import roc_auc_score

D = ROOT / "benchmarks" / "data" / "deception"
CORPora = {
    "ORIGINAL gpt-4o-mini": D / "responses_v0.jsonl",
    "Qwen2.5-3B matched":   D / "responses_lenmatched_qwen2.5-3b-instruct.jsonl",
    "Llama-3.2-3B matched": D / "responses_lenmatched_llama-3.2-3b-instruct.jsonl",
}


def load(p): return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def cem_match(rows, binw=8, seed=0):
    """Keep equal #honest/#dishonest within each word-count bin -> identical length dist across classes."""
    rng = np.random.default_rng(seed)
    wc = np.array([len(r["response"].split()) for r in rows])
    y = np.array([r["label_dishonest"] for r in rows])
    b = wc // binw
    keep = []
    for bb in np.unique(b):
        h = np.where((b == bb) & (y == 0))[0]; d = np.where((b == bb) & (y == 1))[0]
        k = min(len(h), len(d))
        if k == 0: continue
        keep += list(rng.choice(h, k, replace=False)) + list(rng.choice(d, k, replace=False))
    return [rows[i] for i in sorted(keep)]


print(f"{'corpus':24s} {'n_full':>6s} {'n_cem':>6s} {'d_len_full':>10s} {'d_len_cem':>9s} {'AUC_full':>8s} {'AUC_cem':>17s}")
for name, p in CORPora.items():
    rows = load(p)
    X, y, names = featurize(rows)
    wc = np.log1p(np.array([len(r["response"].split()) for r in rows], float))
    d_full = (wc[y == 1].mean() - wc[y == 0].mean()) / (wc.std() or 1)
    auc_full = roc_auc_score(y, shipped_logit(X, names))
    cem = cem_match(rows)
    Xc, yc, _ = featurize(cem)
    wcc = np.log1p(np.array([len(r["response"].split()) for r in cem], float))
    d_cem = (wcc[yc == 1].mean() - wcc[yc == 0].mean()) / (wcc.std() or 1)
    sc = shipped_logit(Xc, names); auc_cem = roc_auc_score(yc, sc); ci = boot_ci(yc, sc)
    print(f"{name:24s} {len(rows):6d} {len(cem):6d} {d_full:10.3f} {d_cem:9.3f} {auc_full:8.3f}   {auc_cem:.3f} CI[{ci[0]:.3f},{ci[1]:.3f}]")
print("\nReading: if AUC_cem collapses toward 0.5 when d_len_cem~0 (esp. on the ORIGINAL gpt corpus, where there is\n"
      "NO generator change), the shipped instrument's discrimination was length-carried. Residual AUC_cem>>0.5 = real register signal.")
