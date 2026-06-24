"""SUITE-WIDE CAUSAL length decomposition of the styxx guardrail instruments.

The length-floor audit (2f003ed) was CORRELATIONAL: "how much of the label is recoverable from word count
alone." This is the CAUSAL complement: for each instrument, measure its OWN full-feature CV-AUC on (a) the
raw corpus vs (b) a coarsened-exact length-MATCHED subsample (honest/positive and negative classes forced to
identical word-count distributions). The drop when length is balanced = the causal contribution of length to
the instrument's discrimination, with NO generator change and NO regeneration.

  causal_length_share = (auc_raw - auc_cem) / (auc_raw - 0.5)

Offline, CPU-only, no key. Run: python scripts/suite_causal_length.py
"""
from __future__ import annotations
import importlib, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from length_control_causal import cv_oof, boot_ci
from sklearn.metrics import roc_auc_score


def join_turns(turns):
    out = []
    for t in turns or []:
        if isinstance(t, str): out.append(t)
        elif isinstance(t, dict): out.append(str(t.get("content") or t.get("text") or t.get("message") or t.get("response") or ""))
        else: out.append(str(t))
    return " ".join(out)


# instrument -> (corpus rel, word-count textfn, label key, train module)
REG = [
    ("sycophancy",     "sycophancy/responses_v0.jsonl",  lambda r: r.get("response", ""),      "label_sycophantic",  "sycophancy_train_v0"),
    ("overconfidence", "overconfidence/pairs_v0.jsonl",  lambda r: r.get("response", ""),      "label_overconfident","overconfidence_train_v0"),
    ("deception",      "deception/responses_v0.jsonl",   lambda r: r.get("response", ""),      "label_dishonest",    "deception_train_v0"),
    ("goal_drift",     "goal_drift/sessions_v0.jsonl",   lambda r: r.get("raw", ""),           "label_drifted",      "goal_drift_train_v0"),
    ("loop",           "loop/conversations_v0.jsonl",    lambda r: join_turns(r.get("turns")), "label_loop",         "loop_train_v0"),
    ("plan_action",    "plan_action/pairs_v0.jsonl",     lambda r: r.get("raw", ""),           "label_mismatch",     "plan_action_train_v0"),
]


def load(p): return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def cem_match(wc, y, binw, seed=0):
    """Indices keeping equal #pos/#neg per word-count bin -> identical length dist across classes."""
    rng = np.random.default_rng(seed); b = (wc // binw).astype(int); keep = []
    for bb in np.unique(b):
        pos = np.where((b == bb) & (y == 1))[0]; neg = np.where((b == bb) & (y == 0))[0]
        k = min(len(pos), len(neg))
        if k: keep += list(rng.choice(pos, k, replace=False)) + list(rng.choice(neg, k, replace=False))
    return np.array(sorted(keep), dtype=int)


def dlen(wc, y):
    l = np.log1p(wc); return float((l[y == 1].mean() - l[y == 0].mean()) / (l.std() or 1))


print(f"{'instrument':14s} {'n':>4s} {'AUC_raw':>7s} {'n_cem':>5s} {'dlen_raw':>8s} {'dlen_cem':>8s} {'AUC_cem':>21s} {'len_share':>9s} verdict")
out = []
for name, rel, textfn, lblkey, modname in REG:
    rows = load(ROOT / "benchmarks" / "data" / rel)
    mod = importlib.import_module(modname)
    X, y, names = mod.featurize(rows)
    wc = np.array([len(textfn(r).split()) for r in rows], float)
    keep = wc > 0
    if not keep.all():
        X, y, wc, rows = X[keep], y[keep], wc[keep], [r for r, k in zip(rows, keep) if k]
    # pick a bin width ~ 1/6 of the IQR so bins are meaningful; floor at 4 words
    binw = max(4, int(np.subtract(*np.percentile(wc, [75, 25])) / 6) or 8)
    _, auc_raw = cv_oof(X, y)
    idx = cem_match(wc, y, binw)
    note = ""
    if len(idx) < 40 or len(np.unique(y[idx])) < 2:
        auc_cem, ci, share, verdict = float("nan"), (float("nan"),) * 2, float("nan"), "THIN(<40 matched)"
    else:
        Xc, yc = X[idx], y[idx]
        oof, auc_cem = cv_oof(Xc, yc); ci = boot_ci(yc, oof)
        share = (auc_raw - auc_cem) / (auc_raw - 0.5) if auc_raw > 0.5 else float("nan")
        if share >= 0.60 or auc_cem <= 0.60: verdict = "LENGTH-CARRIED"
        elif share <= 0.25 and auc_cem >= 0.70: verdict = "CONSTRUCT-ROBUST"
        else: verdict = "MIXED"
    out.append({"instrument": name, "n": int(len(y)), "auc_raw": auc_raw, "n_cem": int(len(idx)),
                "dlen_raw": dlen(wc, y), "dlen_cem": dlen(wc[idx], y[idx]) if len(idx) else float("nan"),
                "auc_cem": auc_cem, "auc_cem_ci": list(ci), "length_share": share, "verdict": verdict, "binw": binw})
    ci_s = f"CI[{ci[0]:.3f},{ci[1]:.3f}]" if ci[0] == ci[0] else "—"
    print(f"{name:14s} {len(y):4d} {auc_raw:7.3f} {len(idx):5d} {dlen(wc,y):8.3f} "
          f"{(dlen(wc[idx],y[idx]) if len(idx) else float('nan')):8.3f} {auc_cem:7.3f} {ci_s:>13s} "
          f"{share:9.2f} {verdict}")

(ROOT / "benchmarks" / "data" / "_suite_causal_length_result.json").write_text(json.dumps(out, indent=2))
print("\nCAUSAL reading (no generator change, no regen — pure length-balancing on each instrument's OWN corpus):")
print("  length_share = fraction of above-chance AUC LOST when the two classes are forced to equal length.")
print("  LENGTH-CARRIED = discrimination is mostly length; CONSTRUCT-ROBUST = survives length-matching.")
print("  (loop length may be construct-INTRINSIC: a balanced-length test understates a real long==loop signal.)")
print("wrote benchmarks/data/_suite_causal_length_result.json")
