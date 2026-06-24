"""Suite-wide LENGTH-FLOOR audit of the styxx guardrail instruments.

Implements PREREG_length_floor_audit_2026_06_23.md exactly. For each on-disk guardrail corpus: measure how much
of the label is recoverable from response length (word count) ALONE. This applies this session's confound-audit
method to styxx's own DOI'd suite — the limit-map discipline turned inward.

Offline, CPU-only, no vendor key. Run: python scripts/length_floor_audit.py
"""
from __future__ import annotations
import importlib, json, math, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))

def join_turns(turns):
    out = []
    for t in turns or []:
        if isinstance(t, str): out.append(t)
        elif isinstance(t, dict): out.append(str(t.get("content") or t.get("text") or t.get("message") or t.get("response") or ""))
        else: out.append(str(t))
    return " ".join(out)

# instrument -> (corpus, text-reconstruction, label key, weights module for headline AUC)
REG = [
    ("sycophancy",    "sycophancy/responses_v0.jsonl",   lambda r: r.get("response", ""),        "label_sycophantic",  "calibrated_weights_sycophancy_v0"),
    ("overconfidence","overconfidence/pairs_v0.jsonl",   lambda r: r.get("response", ""),        "label_overconfident","calibrated_weights_overconfidence_v0"),
    ("deception",     "deception/responses_v0.jsonl",    lambda r: r.get("response", ""),        "label_dishonest",    "calibrated_weights_deception_v0"),
    ("goal_drift",    "goal_drift/sessions_v0.jsonl",    lambda r: r.get("raw", ""),             "label_drifted",      "calibrated_weights_goal_drift_v0"),
    ("loop",          "loop/conversations_v0.jsonl",     lambda r: join_turns(r.get("turns")),   "label_loop",         "calibrated_weights_loop_v0"),
    ("plan_action",   "plan_action/pairs_v0.jsonl",      lambda r: r.get("raw", ""),             "label_mismatch",     "calibrated_weights_plan_action_v0"),
    ("depression",    "depression/responses_v0.jsonl",   lambda r: r.get("text", ""),            "label_depression",   None),  # external control
]

def headline(modname):
    if not modname: return None
    try: return float(getattr(importlib.import_module(f"styxx.guardrail.{modname}"), "MEAN_CV_AUC"))
    except Exception: return None

def floor_auc(w, y):
    X = np.log1p(np.asarray(w, float)).reshape(-1, 1); y = np.asarray(y, int)
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))

def classify(fa, share):
    if fa >= 0.65 or (share is not None and share >= 0.80): return "DOMINATED"
    if fa < 0.60 and (share is None or share < 0.50): return "CLEAN"
    return "MIXED"

rows_out = []
print(f"{'instrument':14s} {'n':>5s} {'pos%':>5s} {'r(len,lbl)':>10s} {'floor_AUC':>9s} {'headline':>8s} {'share':>6s}  verdict")
for name, rel, textfn, lblkey, modname in REG:
    p = ROOT / "benchmarks" / "data" / rel
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    w = np.array([len(textfn(r).split()) for r in rows], float)
    y = np.array([int(r[lblkey]) for r in rows], int)
    keep = (w > 0)  # drop empty-text rows
    w, y = w[keep], y[keep]
    r_pb = float(np.corrcoef(y.astype(float), w)[0, 1])
    fa = floor_auc(w, y)
    hl = headline(modname)
    share = (fa - 0.5) / (hl - 0.5) if hl and hl > 0.5 else None
    verdict = classify(fa, share)
    rows_out.append((name, len(y), y.mean(), r_pb, fa, hl, share, verdict))
    print(f"{name:14s} {len(y):5d} {y.mean()*100:4.0f}% {r_pb:10.3f} {fa:9.3f} "
          f"{(f'{hl:.3f}' if hl else '   —  '):>8s} {(f'{share:.2f}' if share is not None else '  — '):>6s}  {verdict}")

# ---- method-validation gate (prereg) ----
byname = {r[0]: r for r in rows_out}
syc_ok = byname["sycophancy"][7] == "CLEAN"
ovr_ok = byname["overconfidence"][7] == "DOMINATED"
print("\nMETHOD-VALIDATION GATE (prereg):")
print(f"  sycophancy == CLEAN:        {syc_ok}  (got {byname['sycophancy'][7]}, floor {byname['sycophancy'][4]:.3f})")
print(f"  overconfidence == DOMINATED: {ovr_ok}  (got {byname['overconfidence'][7]}, floor {byname['overconfidence'][4]:.3f})")
if not (syc_ok and ovr_ok):
    print("  !! GATE FAILED — method does not reproduce the two knowns; result INVALID per prereg.")
else:
    dominated = [r[0] for r in rows_out if r[7] == "DOMINATED" and r[0] != "overconfidence" and r[0] != "depression"]
    print(f"  GATE PASSED. Other instruments flagged length-DOMINATED: {dominated or 'none'}")
    print(f"  depression (control) floor_AUC: {byname['depression'][4]:.3f}  (expect LOW if confound is stance-prompt-specific)")
